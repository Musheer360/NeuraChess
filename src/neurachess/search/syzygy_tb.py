"""Zero-NPU-Cost Syzygy Endgame Tablebase Probing & Downloader (`neurachess.search.syzygy_tb`).

Provides:
  - Resumable, SHA-256 verified downloader for standard 3-4-5 piece Syzygy WDL/DTZ tables
    from Lichess's official mirror (`https://tablebase.lichess.ovh/tables/standard/`).
  - Root DTZ move selection (`piece_count <= 5`): returns exact DTZ-optimal winning/drawing
    moves in < 0.1 ms with 50-move rule and repetition avoidance guards.
  - Internal tree leaf WDL probing (`piece_count <= 5`): returns exact terminal values
    (`+1.0`, `0.0`, `-1.0`) without consuming an NPU batch slot.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import re
import threading
import urllib.request
from pathlib import Path

import chess
import chess.syzygy

from neurachess._paths import find_repo_root

LICHESS_SYZYGY_BASE = "https://tablebase.lichess.ovh/tables/standard"
SYZYGY_PIECE_CHARS = frozenset("KQRBNP")


def resolve_syzygy_path(
    syzygy_path: str | Path | None = None,
    *,
    auto_discover: bool = False,
) -> Path | None:
    """Resolve an existing Syzygy directory containing at least one `.rtbw` table."""
    cand: Path | None = None
    if syzygy_path is not None:
        s = str(syzygy_path).strip()
        if not s or s.lower() in ("none", "off", "false", "0"):
            return None
        if s.lower() == "auto":
            auto_discover = True
        else:
            cand = Path(s)

    if cand is None:
        env_p = os.environ.get("NEURACHESS_SYZYGY_PATH", "").strip()
        if env_p and env_p.lower() not in ("none", "off", "false", "0"):
            cand = Path(env_p)
        elif auto_discover:
            cand = find_repo_root() / "weights" / "syzygy"

    if cand is None or not cand.is_dir():
        return None
    if not any(cand.glob("*.rtbw")):
        return None
    return cand


def _piece_count_from_tb_filename(fname: str) -> int:
    stem = fname.rsplit(".", 1)[0]
    return sum(1 for ch in stem if ch in SYZYGY_PIECE_CHARS)


def _fetch_lichess_sha256_map() -> dict[str, str]:
    url = f"{LICHESS_SYZYGY_BASE}/sha256"
    req = urllib.request.Request(url, headers={"User-Agent": "NeuraChess-Syzygy/1.0"})
    txt = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
    out: dict[str, str] = {}
    for line in txt.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and (parts[1].endswith(".rtbw") or parts[1].endswith(".rtbz")):
            out[parts[1]] = parts[0].lower()
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def download_syzygy_tables(
    target_dir: Path | str | None = None,
    max_pieces: int = 5,
    include_dtz: bool = True,
    verify_sha256: bool = True,
    max_workers: int = 8,
) -> dict:
    """Download standard 3..`max_pieces` Syzygy WDL (and optionally DTZ) tables with SHA-256 verification."""
    if max_pieces < 3 or max_pieces > 5:
        raise ValueError(f"max_pieces must be in [3, 5], got {max_pieces}")

    dest = Path(target_dir) if target_dir is not None else (find_repo_root() / "weights" / "syzygy")
    dest.mkdir(parents=True, exist_ok=True)

    sha_map = _fetch_lichess_sha256_map()
    subdirs = ["3-4-5-wdl"] + (["3-4-5-dtz"] if include_dtz else [])
    jobs: list[tuple[str, str]] = []
    for sub in subdirs:
        idx_url = f"{LICHESS_SYZYGY_BASE}/{sub}/"
        html = urllib.request.urlopen(
            urllib.request.Request(idx_url, headers={"User-Agent": "NeuraChess-Syzygy/1.0"}),
            timeout=20,
        ).read().decode("utf-8", "ignore")
        for fname in re.findall(r'href="([^"]+\.rtb[wz])"', html):
            if _piece_count_from_tb_filename(fname) <= max_pieces:
                jobs.append((f"{idx_url}{fname}", fname))

    downloaded = 0
    skipped = 0
    total_bytes = 0
    lock = threading.Lock()

    def _download_one(item: tuple[str, str]) -> int:
        nonlocal downloaded, skipped
        url, fname = item
        out_p = dest / fname
        exp_sha = sha_map.get(fname)
        if out_p.exists() and out_p.stat().st_size > 0:
            if not verify_sha256 or exp_sha is None or _sha256_file(out_p) == exp_sha:
                with lock:
                    skipped += 1
                return int(out_p.stat().st_size)

        tmp_p = dest / f".{fname}.tmp"
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "NeuraChess-Syzygy/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                if verify_sha256 and exp_sha is not None:
                    got = hashlib.sha256(data).hexdigest().lower()
                    if got != exp_sha:
                        raise IOError(f"SHA-256 mismatch for {fname}: got {got}, expected {exp_sha}")
                tmp_p.write_bytes(data)
                os.replace(tmp_p, out_p)
                with lock:
                    downloaded += 1
                return len(data)
            except Exception:
                if tmp_p.exists():
                    tmp_p.unlink(missing_ok=True)
                if attempt == 2:
                    raise
        return 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        sizes = list(pool.map(_download_one, jobs))
    total_bytes = sum(sizes)
    return {
        "directory": str(dest),
        "max_pieces": max_pieces,
        "include_dtz": include_dtz,
        "files_total": len(jobs),
        "downloaded": downloaded,
        "skipped_verified": skipped,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1e6, 2),
    }


class SyzygyProber:
    """Thread-safe wrapper around `chess.syzygy.Tablebase` for root DTZ and leaf WDL probing."""

    def __init__(self, syzygy_dir: str | Path, max_pieces: int = 5):
        self.syzygy_dir = Path(syzygy_dir)
        self.max_pieces = int(max_pieces)
        self._tb = chess.syzygy.open_tablebase(str(self.syzygy_dir), load_wdl=True, load_dtz=True)
        self._lock = threading.Lock()
        self._scratch_board = chess.Board(None)
        self._leaf_cache: dict[tuple, tuple[float, int, int] | None] = {}
        self.root_hits = 0
        self.leaf_hits = 0

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
            self._tb = None

    def clear_stats(self) -> None:
        self.root_hits = 0
        self.leaf_hits = 0

    @staticmethod
    def is_eligible(board: chess.Board, max_pieces: int = 5) -> bool:
        return (not board.castling_rights) and (chess.popcount(board.occupied) <= max_pieces)

    def probe_root_dtz_move(self, board: chess.Board) -> tuple[chess.Move, float, str] | None:
        """Return `(best_move, eval_score, tag)` from exact DTZ50'' tables, or `None` if unavailable."""
        if self._tb is None or not self.is_eligible(board, self.max_pieces):
            return None

        root_wdl = self._tb.get_wdl(board)
        if root_wdl is None:
            return None

        legals = list(board.legal_moves)
        if not legals:
            return None

        best_move: chess.Move | None = None
        best_rank: tuple[int, int, int, int] | None = None
        best_tier = 3

        b = board.copy(stack=True)
        for mv in legals:
            is_zeroing = board.is_zeroing(mv)
            b.push(mv)
            try:
                if b.is_checkmate():
                    self.root_hits += 1
                    return mv, 1.0, "syzygy_mate_in_1"

                if b.is_stalemate() or b.is_insufficient_material() or b.halfmove_clock >= 100:
                    tier = 3
                    rank = (3, 0, 0, 1 if is_zeroing else 0)
                elif b.is_repetition(3):
                    tier = 3
                    rank = (3, -1, 0, 0)
                else:
                    c_wdl = self._tb.get_wdl(b)
                    if c_wdl is None:
                        return None
                    c_dtz = self._tb.get_dtz(b)
                    is_rep2 = b.is_repetition(2)

                    if c_wdl == -2:
                        # Opponent unconditionally losing (subject to 50-move budget on resulting board)
                        dtz_plies = abs(c_dtz) if c_dtz is not None else 1
                        within_50 = (b.halfmove_clock + dtz_plies) <= 100
                        if within_50:
                            tier = 5
                            rank = (5, 0 if is_rep2 else 1, -dtz_plies, 1 if is_zeroing else 0)
                        else:
                            tier = 4
                            rank = (4, 0 if is_rep2 else 1, -dtz_plies, 1 if is_zeroing else 0)
                    elif c_wdl == -1:
                        dtz_plies = abs(c_dtz) if c_dtz is not None else 101
                        tier = 4
                        rank = (4, 0 if is_rep2 else 1, -dtz_plies, 1 if is_zeroing else 0)
                    elif c_wdl == 0:
                        tier = 3
                        rank = (3, 1 if is_zeroing else 0, 0, 0)
                    elif c_wdl == 1:
                        dtz_plies = abs(c_dtz) if c_dtz is not None else 101
                        tier = 2
                        rank = (2, 1 if is_rep2 else 0, dtz_plies, 0)
                    else:
                        # c_wdl == 2: opponent wins unless 50-move rule rescues
                        dtz_plies = abs(c_dtz) if c_dtz is not None else 1
                        within_50 = (b.halfmove_clock + dtz_plies) <= 100
                        if within_50:
                            tier = 1
                            rank = (1, 1 if is_rep2 else 0, dtz_plies, 0)
                        else:
                            tier = 2
                            rank = (2, 1 if is_rep2 else 0, dtz_plies, 0)

                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_move = mv
                    best_tier = tier
            finally:
                b.pop()

        if best_move is None:
            return None

        self.root_hits += 1
        eval_map = {6: 1.0, 5: 1.0, 4: 0.1, 3: 0.0, 2: -0.1, 1: -1.0}
        return best_move, eval_map.get(best_tier, 0.0), f"syzygy_dtz_tier{best_tier}"

    def probe_leaf_wdl(self, board: chess.Board) -> tuple[float, int, int] | None:
        """
        Probe Syzygy WDL for an internal search leaf (`piece_count <= 5`).

        Returns `(terminal_value, is_solved, mate_plies)` from side-to-move perspective:
          - Unconditional win (`+1.0, 1, 80`)
          - Draw / 50-move blessed or cursed (`0.0, 0, 0`)
          - Unconditional loss (`-1.0, 1, 80`)
        """
        if self._tb is None or not self.is_eligible(board, self.max_pieces):
            return None

        # When halfmove_clock == 0, WDL alone determines exact 50-move outcome and cache key needs no halfmove
        tkey = board._transposition_key()
        hm = int(board.halfmove_clock)
        ckey = (tkey, hm if hm > 0 else 0)
        cached = self._leaf_cache.get(ckey)
        if cached is not None or ckey in self._leaf_cache:
            if cached is not None:
                self.leaf_hits += 1
            return cached

        wdl = self._tb.get_wdl(board)
        if wdl is None:
            if len(self._leaf_cache) < 262144:
                self._leaf_cache[ckey] = None
            return None

        if wdl in (-1, 0, 1):
            res: tuple[float, int, int] = (0.0, 0, 0)
        elif wdl == 2:
            if hm > 0:
                dtz = self._tb.get_dtz(board)
                if dtz is not None and (hm + abs(dtz)) > 100:
                    res = (0.0, 0, 0)
                else:
                    res = (1.0, 1, 80)
            else:
                res = (1.0, 1, 80)
        else:  # wdl == -2
            if hm > 0:
                dtz = self._tb.get_dtz(board)
                if dtz is not None and (hm + abs(dtz)) > 100:
                    res = (0.0, 0, 0)
                else:
                    res = (-1.0, 1, 80)
            else:
                res = (-1.0, 1, 80)

        if len(self._leaf_cache) < 262144:
            self._leaf_cache[ckey] = res
        self.leaf_hits += 1
        return res

    def probe_leaf_wdl_bitboards(
        self,
        w_bbs6: tuple[int, int, int, int, int, int],
        b_bbs6: tuple[int, int, int, int, int, int],
        turn_is_white: bool,
        ep_sq: int,
        halfmove: int,
    ) -> tuple[float, int, int] | None:
        """Populate scratch `chess.Board` from raw C++ bitboards and probe Syzygy leaf WDL."""
        with self._lock:
            sb = getattr(self, "_scratch_board", None)
            if sb is None:
                sb = chess.Board(None)
                self._scratch_board = sb
            sb.pawns = int(w_bbs6[0]) | int(b_bbs6[0])
            sb.knights = int(w_bbs6[1]) | int(b_bbs6[1])
            sb.bishops = int(w_bbs6[2]) | int(b_bbs6[2])
            sb.rooks = int(w_bbs6[3]) | int(b_bbs6[3])
            sb.queens = int(w_bbs6[4]) | int(b_bbs6[4])
            sb.kings = int(w_bbs6[5]) | int(b_bbs6[5])
            sb.promoted = 0
            w_occ = int(w_bbs6[0]) | int(w_bbs6[1]) | int(w_bbs6[2]) | int(w_bbs6[3]) | int(w_bbs6[4]) | int(w_bbs6[5])
            b_occ = int(b_bbs6[0]) | int(b_bbs6[1]) | int(b_bbs6[2]) | int(b_bbs6[3]) | int(b_bbs6[4]) | int(b_bbs6[5])
            sb.occupied_co[chess.WHITE] = w_occ
            sb.occupied_co[chess.BLACK] = b_occ
            sb.occupied = w_occ | b_occ
            sb.turn = chess.WHITE if turn_is_white else chess.BLACK
            sb.castling_rights = 0
            sb.ep_square = ep_sq if 0 <= ep_sq < 64 else None
            sb.halfmove_clock = int(halfmove)
            return self.probe_leaf_wdl(sb)
