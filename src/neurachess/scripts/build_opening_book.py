"""
Build NeuraChess CC0 Opening Book (neurachess.scripts.build_opening_book)
=================================================================
Generates a compact, polyglot-style EPD opening book (`assets/opening_book.json`)
directly from official Lichess CC0 ECO opening data (`lichess-org/chess-openings`,
`a.tsv` through `e.tsv`, licensed under CC0 1.0 Universal).

Every move sequence is parsed with `python-chess`, strictly verified for move
legality from `chess.Board()`, truncated to `board.ply() < max_ply` (default 16),
pruned to fit safely under `max_bytes` (default 195 KB <= 200 KB), and re-validated
after serialization.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import urllib.request
from pathlib import Path

import chess
import chess.pgn

from neurachess._paths import find_repo_root

ROOT_DIR = find_repo_root()
DEFAULT_OUT_PATH = ROOT_DIR / "assets" / "opening_book.json"
LICHESS_OPENINGS_BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
TSV_FILES = ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv")


def _load_tsv_texts(input_dir: Path | None) -> tuple[str, dict[str, str]]:
    if input_dir is not None:
        license_path = input_dir / "COPYING.txt"
        license_text = license_path.read_text(encoding="utf-8") if license_path.exists() else "CC0 1.0 Universal"
        texts = {name: (input_dir / name).read_text(encoding="utf-8") for name in TSV_FILES}
        return license_text, texts

    with urllib.request.urlopen(f"{LICHESS_OPENINGS_BASE}/COPYING.txt", timeout=20) as resp:
        license_text = resp.read().decode("utf-8")
    if "CC0 1.0 Universal" not in license_text:
        raise RuntimeError("Upstream chess-openings license verification failed: expected CC0 1.0 Universal.")

    texts: dict[str, str] = {}
    for name in TSV_FILES:
        with urllib.request.urlopen(f"{LICHESS_OPENINGS_BASE}/{name}", timeout=20) as resp:
            texts[name] = resp.read().decode("utf-8")
    return license_text, texts


def build_opening_book(
    out_path: Path = DEFAULT_OUT_PATH,
    input_dir: Path | None = None,
    max_ply: int = 16,
    top_k: int = 4,
    min_support: int = 2,
    min_share: float = 0.02,
    max_bytes: int = 195 * 1024,
) -> dict:
    _, tsv_texts = _load_tsv_texts(input_dir)

    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    min_ply_seen: dict[str, int] = {}
    parsed_rows = 0

    for fname in TSV_FILES:
        lines = tsv_texts[fname].splitlines()
        for row_idx, line in enumerate(lines[1:], start=2):
            if not line.strip():
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            _, _, pgn_str = parts
            game = chess.pgn.read_game(io.StringIO(pgn_str))
            if game is None:
                continue
            parsed_rows += 1
            board = chess.Board()
            for mv in game.mainline_moves():
                if board.ply() >= max_ply:
                    break
                if not board.is_legal(mv):
                    raise ValueError(f"Illegal move {mv.uci()} in {fname}:{row_idx}: {pgn_str}")
                epd = board.epd()
                counts[epd][mv.uci()] += 1
                if epd not in min_ply_seen or board.ply() < min_ply_seen[epd]:
                    min_ply_seen[epd] = board.ply()
                board.push(mv)

    if parsed_rows == 0 or not counts:
        raise RuntimeError("No opening lines parsed from CC0 Lichess TSV inputs.")

    # Filter candidate moves per EPD: keep mainstream ECO continuations while excluding
    # rare 1-off flank side-lines at high-traffic nodes.
    candidates: list[tuple[int, int, str, list[list]]] = []
    for epd, counter in counts.items():
        total = sum(counter.values())
        if total < min_support:
            continue
        floor = max(min_support, int(total * min_share))
        kept = [
            [uci, int(cnt)]
            for uci, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
            if cnt >= floor
        ]
        if kept:
            candidates.append((total, -min_ply_seen[epd], epd, kept))

    # Deterministic priority: highest position support, shallower ply first
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))

    def _serialize(items: list[tuple[int, int, str, list[list]]]) -> bytes:
        positions = {epd: moves for _, _, epd, moves in sorted(items, key=lambda x: x[2])}
        payload = {
            "source": "https://github.com/lichess-org/chess-openings",
            "license": "CC0-1.0",
            "max_ply": int(max_ply),
            "parsed_lines": int(parsed_rows),
            "num_positions": len(positions),
            "positions": positions,
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    # Binary search largest prefix fitting within max_bytes
    raw_bytes = _serialize(candidates)
    if len(raw_bytes) > max_bytes:
        lo, hi = 1, len(candidates)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if len(_serialize(candidates[:mid])) <= max_bytes:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        candidates = candidates[:best]
        raw_bytes = _serialize(candidates)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw_bytes)

    # Validate serialized payload end-to-end with python-chess
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    if out_path.stat().st_size > 200 * 1024:
        raise RuntimeError(f"Generated opening book exceeds 200 KB: {out_path.stat().st_size} bytes")
    for epd, entries in payload["positions"].items():
        b = chess.Board(f"{epd} 0 1")
        for uci, weight in entries:
            mv = chess.Move.from_uci(uci)
            if not b.is_legal(mv) or weight <= 0:
                raise ValueError(f"Post-write book validation failed: {epd} -> {uci} ({weight})")

    return {
        "out_path": str(out_path),
        "bytes": out_path.stat().st_size,
        "parsed_lines": parsed_rows,
        "num_positions": len(payload["positions"]),
        "startpos_moves": payload["positions"].get(chess.Board().epd(), []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CC0 Lichess ECO opening book JSON")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--input-dir", type=Path, default=None, help="Optional local dir with a.tsv..e.tsv + COPYING.txt")
    parser.add_argument("--max-ply", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-share", type=float, default=0.02)
    parser.add_argument("--max-bytes", type=int, default=195 * 1024)
    args = parser.parse_args()

    stats = build_opening_book(
        out_path=args.out,
        input_dir=args.input_dir,
        max_ply=args.max_ply,
        top_k=args.top_k,
        min_support=args.min_support,
        min_share=args.min_share,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
