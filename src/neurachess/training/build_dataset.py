"""
NeuraChess Bitboard Dataset Builder.

Subcommands:
  - `from-lichess`: Convert a local Lichess evaluation `.jsonl.zst` (or `.jsonl`) file into
    deduplicated, optional eval-disjoint `107B` v1 shards (+ optional `17B` Boltzmann top-K sidecars)
    or `691B` v2 shards with pre-packed legal move bitmasks, plus SHA-256 `dataset_manifest.json`.
  - `from-stockfish`: Generate synthetic self-play/random positions annotated by a local Stockfish UCI binary.
"""

import argparse
import hashlib
import io
import json
import logging
import os
import random
import shutil
import struct
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

import chess
import numpy as np
import zstandard as zstd

from neurachess.core.constants import uci_to_action
from neurachess.core.tensor_movegen import get_legal_action_mask
from neurachess.training.binary_dataset import cp_to_wdl
from neurachess.training.bitboard_dataset import PACK_FORMAT, pack_position, pack_position_v2


def _board_hash_from_107b(chunk: bytes) -> str:
    unp = struct.unpack(PACK_FORMAT, chunk)
    turn, castling, ep_square, _ = unp[12:16]
    return hashlib.md5(f"{unp[:12]}_{turn}_{castling}_{ep_square}".encode("ascii")).hexdigest()


def _extract_pv_move_and_cp(board: chess.Board, pv_entry: dict) -> tuple[str | None, int | None]:
    line = pv_entry.get("line", "")
    if not line:
        return None, None
    first_uci = line.split()[0]
    try:
        mv = board.parse_uci(first_uci)
        norm_uci = mv.uci()
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        logger.debug("Skipping unparseable Lichess PV move %r", first_uci, exc_info=True)
        return None, None

    if "cp" in pv_entry and pv_entry["cp"] is not None:
        cp_stm = int(pv_entry["cp"])
    elif "mate" in pv_entry and pv_entry["mate"] is not None:
        cp_stm = 10000 if int(pv_entry["mate"]) > 0 else -10000
    else:
        return None, None

    # Lichess DB stores evals from White's perspective; convert to side-to-move perspective.
    if board.turn == chess.BLACK:
        cp_stm = -cp_stm
    return norm_uci, max(-10000, min(10000, cp_stm))


def build_from_lichess(args: argparse.Namespace) -> None:
    exclude_hashes: set[str] = set()
    if args.exclude_hashes and args.exclude_hashes.exists():
        exclude_hashes = set(json.loads(args.exclude_hashes.read_text(encoding="utf-8")))
        print(f"[Dataset] Loaded {len(exclude_hashes):,} excluded position hashes.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    records: list[tuple[bytes, bytes]] = []

    raw_f = open(args.input, "rb")
    if str(args.input).endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        text_stream = io.TextIOWrapper(dctx.stream_reader(raw_f), encoding="utf-8")
    else:
        text_stream = io.TextIOWrapper(raw_f, encoding="utf-8")

    scanned = skipped_excluded = skipped_dup = 0
    try:
        for line in text_stream:
            scanned += 1
            if args.skip_lines > 0 and scanned <= args.skip_lines:
                continue
            entry = json.loads(line)
            fen = entry.get("fen")
            evals = entry.get("evals")
            if not fen or not evals:
                continue
            try:
                board = chess.Board(fen)
            except ValueError:
                logger.debug("Skipping invalid FEN %r", fen, exc_info=True)
                continue
            pvs = evals[0].get("pvs") or []
            if not pvs:
                continue

            parsed_pvs: list[tuple[int, int]] = []
            for pv in pvs[:5]:
                mv_uci, cp_val = _extract_pv_move_and_cp(board, pv)
                if mv_uci is None or cp_val is None:
                    continue
                act = uci_to_action(mv_uci, canonical_flip=(board.turn == chess.BLACK))
                parsed_pvs.append((act, cp_val))
            if not parsed_pvs:
                continue

            best_act, best_cp = parsed_pvs[0]
            wdl = tuple(float(x) for x in cp_to_wdl(best_cp, is_mate=(abs(best_cp) >= 10000), mate_dist=(1 if best_cp > 0 else -1)))
            if args.format == "v2":
                mask = get_legal_action_mask(board)
                rec_bytes = pack_position_v2(board, action_index=best_act, eval_cp=best_cp, wdl=wdl, legal_mask=mask)
            else:
                rec_bytes = pack_position(board, action_index=best_act, eval_cp=best_cp, wdl=wdl)

            h = _board_hash_from_107b(rec_bytes[:107])
            if h in exclude_hashes:
                skipped_excluded += 1
                continue
            if h in seen:
                skipped_dup += 1
                continue
            seen.add(h)

            # 17B Boltzmann sidecar: [num_extra(1B), 4x extra_act(int16), 4x extra_cp(int16)]
            extras = parsed_pvs[1:5]
            extra_acts = [a for a, _ in extras] + [0] * (4 - len(extras))
            extra_cps = [c for _, c in extras] + [0] * (4 - len(extras))
            side_bytes = struct.pack("<B4h4h", len(extras), *extra_acts, *extra_cps)
            records.append((rec_bytes, side_bytes))

            if len(records) >= args.max_positions:
                break
    finally:
        text_stream.close()

    rng = random.Random(args.seed)
    rng.shuffle(records)

    shard_meta = []
    for s_idx, offset in enumerate(range(0, len(records), args.shard_size)):
        chunk = records[offset : offset + args.shard_size]
        if len(chunk) < args.shard_size and not args.keep_partial:
            break
        blob = b"".join(r for r, _ in chunk)
        shard_name = f"shard_{s_idx:03d}.bin"
        (args.out_dir / shard_name).write_bytes(blob)
        if args.format == "v1" and args.write_sidecar:
            side_blob = b"".join(s for _, s in chunk)
            (args.out_dir / f"shard_{s_idx:03d}.sidecar.bin").write_bytes(side_blob)
        shard_meta.append(
            {
                "shard_index": s_idx,
                "filename": shard_name,
                "position_count": len(chunk),
                "file_size_bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )

    manifest = {
        "format": args.format,
        "scanned_lines": scanned,
        "skipped_excluded": skipped_excluded,
        "skipped_duplicates": skipped_dup,
        "total_positions_written": sum(s["position_count"] for s in shard_meta),
        "shards": shard_meta,
    }
    (args.out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Dataset] Wrote {len(shard_meta)} shard(s), {manifest['total_positions_written']:,} positions -> {args.out_dir}")


def _resolve_sf(user_path: str | None) -> str:
    if user_path and Path(user_path).exists():
        return user_path
    env_p = os.environ.get("STOCKFISH_PATH")
    if env_p and Path(env_p).exists():
        return env_p
    w = shutil.which("stockfish")
    if w:
        return w
    raise FileNotFoundError("Stockfish binary not found; pass --stockfish-path or set STOCKFISH_PATH.")


def build_from_stockfish(args: argparse.Namespace) -> None:
    sf_path = _resolve_sf(args.stockfish_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sf_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    proc.stdin.write("uci\nsetoption name Threads value 1\nisready\n")
    proc.stdin.flush()
    while "readyok" not in proc.stdout.readline():
        pass

    rng = random.Random(args.seed)
    records: list[bytes] = []
    board = chess.Board()
    try:
        while len(records) < args.num_positions:
            if board.is_game_over() or board.ply() > 120:
                board = chess.Board()
                for _ in range(rng.randint(2, 16)):
                    legals = list(board.legal_moves)
                    if not legals:
                        break
                    board.push(rng.choice(legals))
                if board.is_game_over():
                    continue

            proc.stdin.write(f"position fen {board.fen()}\ngo depth {args.depth}\n")
            proc.stdin.flush()
            best_uci = None
            cp_stm = 0
            while True:
                line = proc.stdout.readline().strip()
                if "score cp" in line:
                    parts = line.split()
                    cp_stm = int(parts[parts.index("cp") + 1])
                elif "score mate" in line:
                    parts = line.split()
                    m = int(parts[parts.index("mate") + 1])
                    cp_stm = 10000 if m > 0 else -10000
                if line.startswith("bestmove"):
                    parts = line.split()
                    best_uci = parts[1] if len(parts) > 1 else None
                    break

            if best_uci and best_uci != "(none)":
                act = uci_to_action(best_uci, canonical_flip=(board.turn == chess.BLACK))
                wdl = tuple(float(x) for x in cp_to_wdl(cp_stm, is_mate=(abs(cp_stm) >= 10000), mate_dist=(1 if cp_stm > 0 else -1)))
                if args.format == "v2":
                    mask = get_legal_action_mask(board)
                    records.append(pack_position_v2(board, action_index=act, eval_cp=cp_stm, wdl=wdl, legal_mask=mask))
                else:
                    records.append(pack_position(board, action_index=act, eval_cp=cp_stm, wdl=wdl))
            legals = list(board.legal_moves)
            if legals:
                board.push(rng.choice(legals))
    finally:
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError):
            logger.debug("Graceful Stockfish shutdown failed; terminating", exc_info=True)
            proc.terminate()

    out_file = args.out_dir / f"shard_000.bin"
    out_file.write_bytes(b"".join(records))
    print(f"[Dataset] Generated {len(records):,} Stockfish-annotated records -> {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build packed NeuraChess bitboard training/eval shards")
    sub = parser.add_subparsers(dest="command", required=True)

    p_li = sub.add_parser("from-lichess", help="Convert Lichess eval .jsonl/.jsonl.zst into bitboard shards")
    p_li.add_argument("--input", type=Path, required=True, help="Local Lichess eval .jsonl or .jsonl.zst path")
    p_li.add_argument("--out-dir", type=Path, required=True)
    p_li.add_argument("--format", choices=["v1", "v2"], default="v1")
    p_li.add_argument("--write-sidecar", action="store_true", help="Write 17B top-K Boltzmann sidecar for v1 shards")
    p_li.add_argument("--max-positions", type=int, default=500_000)
    p_li.add_argument("--shard-size", type=int, default=500_000)
    p_li.add_argument("--skip-lines", type=int, default=0)
    p_li.add_argument("--keep-partial", action="store_true")
    p_li.add_argument("--exclude-hashes", type=Path, default=None)
    p_li.add_argument("--seed", type=int, default=42)

    p_sf = sub.add_parser("from-stockfish", help="Annotate random positions via a local Stockfish UCI binary")
    p_sf.add_argument("--stockfish-path", type=str, default=None)
    p_sf.add_argument("--out-dir", type=Path, required=True)
    p_sf.add_argument("--format", choices=["v1", "v2"], default="v1")
    p_sf.add_argument("--num-positions", type=int, default=10_000)
    p_sf.add_argument("--depth", type=int, default=8)
    p_sf.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "from-lichess":
        build_from_lichess(args)
    else:
        build_from_stockfish(args)


if __name__ == "__main__":
    main()
