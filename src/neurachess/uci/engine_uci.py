"""
NeuraChess Universal Chess Interface (UCI) Protocol Driver.

Connects NeuraChess to UCI chess GUIs and tournament managers (CuteChess, Arena, Banksia).
Supports pondering (`Ponder`, `bestmove X ponder Y`, `go ponder`, `ponderhit`, `stop`) and
`EncoderLayout` selection (`gpu_unpack`, `legacy_flip`).
"""

import json
import logging
import math
import random
import sys
import threading
import time
from pathlib import Path

from neurachess._paths import find_repo_root

logger = logging.getLogger(__name__)
ROOT_DIR = find_repo_root()
DEFAULT_OPENING_BOOK_PATH = ROOT_DIR / "assets" / "opening_book.json"

import chess

from neurachess.core.board_tensor import board_to_tensor
from neurachess.models.npu_compiler import DEFAULT_CACHE_DIR, NPUModel, resolve_default_model_paths
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch
from neurachess.search.ponder_manager import PonderManager


def value_to_lichess_cp(v: float) -> int:
    """Convert side-to-move value v in [-1, 1] to centipawns via 400*log10((1+v)/(1-v))."""
    v_clamped = max(-0.9999, min(0.9999, float(v)))
    return int(round(400.0 * math.log10((1.0 + v_clamped) / (1.0 - v_clamped))))


class UCIEngine:
    def __init__(
        self,
        encoder_layout: str = "gpu_unpack",
        own_book: bool = True,
        book_path: Path | str | None = None,
        book_seed: int | None = None,
        use_ponder: bool = True,
        syzygy_path: str = "auto",
        move_overhead_ms: int = 150,
        threads: int = 1,
        batch_size: int = 16,
        hash_mb: int = 64,
    ):
        self.board = chess.Board()
        self.history: list[chess.Board] = []
        self.encoder_layout = encoder_layout
        self.use_ponder = bool(use_ponder)
        self.own_book = bool(own_book)
        self.syzygy_path = str(syzygy_path)
        self.move_overhead_ms = max(0, min(2000, int(move_overhead_ms)))
        self.threads = max(1, min(16, int(threads)))
        self.batch_size = max(1, min(32, int(batch_size)))
        self.hash_mb = max(1, min(1024, int(hash_mb)))
        self.book_path = Path(book_path) if book_path is not None else DEFAULT_OPENING_BOOK_PATH
        self._book_rng = random.Random(book_seed)
        self._opening_book: dict[str, list] | None = None
        self.search_engine: NPUBatchedPUCTSearch | None = None
        self.ponder_manager: PonderManager | None = None
        self._pending_ponder_telemetry: dict | None = None
        self._stdout_lock = threading.Lock()

    def _emit_lines(self, lines: list[str]) -> None:
        with self._stdout_lock:
            for ln in lines:
                print(ln)
            sys.stdout.flush()

    def _load_opening_book(self) -> dict[str, list]:
        if self._opening_book is not None:
            return self._opening_book
        if not self.book_path.exists():
            self._opening_book = {}
            return self._opening_book
        try:
            payload = json.loads(self.book_path.read_text(encoding="utf-8"))
            positions = payload.get("positions", {}) if isinstance(payload, dict) else {}
            self._opening_book = positions if isinstance(positions, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError):
            logger.debug("Failed loading opening book from %s", self.book_path, exc_info=True)
            self._opening_book = {}
        return self._opening_book

    def lookup_book_move(self, board: chess.Board) -> tuple[str | None, str | None]:
        """Look up a verified-legal CC0 opening book move (and optional reply hint) for ply < 16."""
        if not self.own_book or board.ply() >= 16:
            return None, None
        positions = self._load_opening_book()
        if not positions:
            return None, None
        entries = positions.get(board.epd())
        if not entries:
            return None, None

        legal_ucis = {m.uci() for m in board.legal_moves}
        candidates: list[str] = []
        weights: list[float] = []
        for item in entries:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            uci, weight = str(item[0]), float(item[1])
            if uci in legal_ucis and weight > 0.0:
                candidates.append(uci)
                weights.append(weight)
        if not candidates:
            return None, None

        chosen = self._book_rng.choices(candidates, weights=weights, k=1)[0]
        ponder_uci = None
        if board.ply() + 1 < 16:
            after = board.copy(stack=False)
            after.push_uci(chosen)
            reply_entries = positions.get(after.epd(), [])
            after_legals = {m.uci() for m in after.legal_moves}
            for item in reply_entries:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and str(item[0]) in after_legals:
                    ponder_uci = str(item[0])
                    break
        return chosen, ponder_uci

    def ensure_engine(self):
        if self.search_engine is None:
            xml_path, _ = resolve_default_model_paths()
            if xml_path is not None:
                root_m = NPUModel(
                    batch_size=1,
                    cache_dir=DEFAULT_CACHE_DIR,
                    device="NPU",
                    use_turbo=True,
                    performance_hint="LATENCY",
                    ov_ir_xml_path=xml_path,
                    verbose=False,
                )
                batch_m = NPUModel(
                    batch_size=self.batch_size,
                    cache_dir=DEFAULT_CACHE_DIR,
                    device="NPU",
                    use_turbo=True,
                    performance_hint="LATENCY",
                    ov_ir_xml_path=xml_path,
                    verbose=False,
                )
            else:
                root_m = NPUModel(batch_size=1, verbose=False)
                batch_m = NPUModel(batch_size=self.batch_size, verbose=False)

            self.search_engine = NPUBatchedPUCTSearch(
                root_model=root_m,
                batch_model=batch_m,
                batch_size=self.batch_size,
                c_puct=1.5,
                use_fpu_reduction=True,
                fpu_reduction=0.20,
                root_fpu_reduction=0.10,
                t_root=1.0,
                use_exact_solver=True,
                use_tree_reuse=True,
                use_nn_cache=True,
                nn_cache_max_bytes=self.hash_mb * 1024 * 1024,
                use_fast_tree=True,
                use_ponder=self.use_ponder,
                encoder_layout=self.encoder_layout,
                syzygy_path=self.syzygy_path,
            )
            self.ponder_manager = PonderManager(self.search_engine, use_ponder=self.use_ponder)

    @staticmethod
    def _format_info_line(
        depth: int,
        seldepth: int,
        score: float,
        nodes: int,
        elapsed_ms: int,
        pv: list[str] | None,
        fallback_uci: str,
        mate_moves: int | None = None,
    ) -> str:
        nodes_i = max(0, int(nodes))
        dt_ms = max(1, int(elapsed_ms))
        nps = int(round(nodes_i * 1000.0 / dt_ms))
        d_i = max(1, int(depth)) if fallback_uci != "0000" else 0
        sd_i = max(d_i, int(seldepth)) if fallback_uci != "0000" else 0
        if mate_moves is not None and int(mate_moves) != 0:
            score_tok = f"mate {int(mate_moves)}"
        else:
            score_tok = f"cp {value_to_lichess_cp(score)}"
        pv_str = " ".join(pv) if pv else fallback_uci
        return f"info depth {d_i} seldepth {sd_i} score {score_tok} nodes {nodes_i} nps {nps} time {dt_ms} pv {pv_str}"

    def _on_uci_ponder_finished(self, best_uci: str, ponder_uci: str | None, nodes: int, score: float):
        meta = getattr(self.ponder_manager, "last_uci_meta", {}) if self.ponder_manager else {}
        info_line = self._format_info_line(
            depth=int(meta.get("depth", 1)),
            seldepth=int(meta.get("seldepth", 1)),
            score=score,
            nodes=int(meta.get("ponder_nodes", nodes)),
            elapsed_ms=int(meta.get("elapsed_ms", 1)),
            pv=meta.get("pv"),
            fallback_uci=best_uci,
            mate_moves=meta.get("mate_moves"),
        )
        bm_line = f"bestmove {best_uci} ponder {ponder_uci}" if (ponder_uci and best_uci != "0000") else f"bestmove {best_uci}"
        self._emit_lines([info_line, bm_line])

    @staticmethod
    def _parse_setoption(line: str) -> tuple[str, str]:
        lower = line.lower()
        name_idx = lower.find(" name ")
        if name_idx == -1:
            return "", ""
        rest = line[name_idx + 6 :]
        val_idx = rest.lower().find(" value ")
        if val_idx == -1:
            return rest.strip(), ""
        return rest[:val_idx].strip(), rest[val_idx + 7 :].strip()

    def _apply_setoption(self, line: str) -> None:
        name, val = self._parse_setoption(line)
        key = name.lower()
        if not key:
            return
        if key == "ponder":
            self.use_ponder = val.lower() == "true"
            if self.ponder_manager:
                self.ponder_manager.use_ponder = self.use_ponder
            if self.search_engine:
                self.search_engine.set_ponder_mode(self.use_ponder)
        elif key == "ownbook":
            self.own_book = val.lower() == "true"
        elif key == "syzygypath":
            self.syzygy_path = val if val else "auto"
            if self.search_engine:
                self.search_engine.reconfigure(syzygy_path=self.syzygy_path)
        elif key == "move overhead":
            try:
                self.move_overhead_ms = max(0, min(2000, int(val)))
            except ValueError:
                pass
        elif key == "threads":
            try:
                self.threads = max(1, min(16, int(val)))
            except ValueError:
                pass
        elif key == "batchsize":
            try:
                new_bs = max(1, min(32, int(val)))
            except ValueError:
                return
            if new_bs != self.batch_size:
                self.batch_size = new_bs
                if self.ponder_manager:
                    self.ponder_manager.stop(emit_on_stop=False)
                self.search_engine = None
                self.ponder_manager = None
        elif key == "hash":
            try:
                self.hash_mb = max(1, min(1024, int(val)))
            except ValueError:
                return
            if self.search_engine is not None and hasattr(self.search_engine, "nn_cache"):
                self.search_engine.nn_cache.max_bytes = self.hash_mb * 1024 * 1024
        elif key == "encoderlayout":
            if val in ("gpu_unpack", "legacy_flip"):
                self.encoder_layout = val
                if self.search_engine:
                    self.search_engine.encoder_layout = val

    def run(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    if self.ponder_manager:
                        self.ponder_manager.stop(emit_on_stop=False)
                    break
                line = line.strip()
                if not line:
                    continue

                tokens = line.split()
                cmd = tokens[0]

                if cmd == "uci":
                    self._emit_lines(
                        [
                            "id name NeuraChess 1.1.0 (OpenVINO Intel NPU)",
                            "id author Musheer Alam",
                            "option name Hash type spin default 64 min 1 max 1024",
                            "option name Ponder type check default true",
                            "option name OwnBook type check default true",
                            "option name SyzygyPath type string default auto",
                            "option name Move Overhead type spin default 150 min 0 max 2000",
                            "option name Threads type spin default 1 min 1 max 16",
                            "option name BatchSize type spin default 16 min 1 max 32",
                            "option name EncoderLayout type combo default gpu_unpack var gpu_unpack var legacy_flip",
                            "uciok",
                        ]
                    )

                elif cmd == "setoption":
                    self._apply_setoption(line)

                elif cmd == "isready":
                    self.ensure_engine()
                    self._emit_lines(["readyok"])

                elif cmd == "ucinewgame":
                    if self.ponder_manager:
                        self.ponder_manager.stop(emit_on_stop=False)
                    if self.search_engine:
                        self.search_engine.clear_state()
                    self._pending_ponder_telemetry = None
                    self.board = chess.Board()
                    self.history = []

                elif cmd == "position":
                    self.handle_position(tokens[1:])

                elif cmd == "go":
                    self.handle_go(tokens[1:])

                elif cmd == "ponderhit":
                    if self.ponder_manager:
                        self.ponder_manager.ponderhit()

                elif cmd == "stop":
                    if self.ponder_manager:
                        self.ponder_manager.stop(emit_on_stop=True)

                elif cmd == "quit":
                    if self.ponder_manager:
                        self.ponder_manager.stop(emit_on_stop=False)
                    break

            except Exception as e:
                logger.debug("UCI command handling exception", exc_info=True)
                print(f"info string error: {e}", file=sys.stderr, flush=True)
                with self._stdout_lock:
                    sys.stdout.flush()

    def handle_position(self, args: list[str]):
        if not args:
            return

        if args[0] == "startpos":
            self.board = chess.Board()
            self.history = []
            move_idx = 1
        elif args[0] == "fen":
            fen_parts = []
            move_idx = 1
            while move_idx < len(args) and args[move_idx] != "moves":
                fen_parts.append(args[move_idx])
                move_idx += 1
            fen_str = " ".join(fen_parts)
            self.board = chess.Board(fen_str)
            self.history = []
        else:
            return

        last_move = None
        if move_idx < len(args) and args[move_idx] == "moves":
            for m in args[move_idx + 1:]:
                mv_obj = chess.Move.from_uci(m)
                self.history.append(self.board.copy())
                self.board.push(mv_obj)
                last_move = mv_obj

        if last_move and self.ponder_manager:
            self._pending_ponder_telemetry = self.ponder_manager.on_opponent_move(last_move)
        else:
            if self.ponder_manager:
                self.ponder_manager.stop(emit_on_stop=False)
            self._pending_ponder_telemetry = None

    @staticmethod
    def _parse_go_params(args: list[str]) -> dict[str, int]:
        int_keys = {"movetime", "wtime", "btime", "winc", "binc", "movestogo"}
        params: dict[str, int] = {}
        i = 0
        while i < len(args):
            tok = args[i]
            if tok in int_keys and i + 1 < len(args):
                try:
                    params[tok] = int(args[i + 1])
                except ValueError:
                    logger.debug("Ignoring invalid integer UCI go value for %s: %r", tok, args[i + 1], exc_info=True)
                i += 2
                continue
            i += 1
        return params

    def compute_move_time_ms(self, args: list[str], consume_ponder: bool = False) -> int:
        params = self._parse_go_params(args)
        telemetry = self._pending_ponder_telemetry
        if telemetry is None and self.ponder_manager is not None:
            telemetry = getattr(self.ponder_manager, "last_telemetry", None)
        if consume_ponder:
            self._pending_ponder_telemetry = None
            if self.ponder_manager is not None and isinstance(getattr(self.ponder_manager, "last_telemetry", None), dict):
                self.ponder_manager.last_telemetry = {
                    **self.ponder_manager.last_telemetry,
                    "ponder_hit": False,
                    "visits_retained": 0,
                }

        if "movetime" in params:
            return max(1, params["movetime"])

        side_time_key = "wtime" if self.board.turn == chess.WHITE else "btime"
        side_inc_key = "winc" if self.board.turn == chess.WHITE else "binc"
        if side_time_key in params:
            remaining = max(0, params[side_time_key])
            inc = max(0, params.get(side_inc_key, 0))
            movestogo = params.get("movestogo")
            if movestogo is not None and movestogo <= 0:
                movestogo = None
            safety_ceiling = remaining * 0.25
            time_for_move = min(remaining * 0.04 + inc * 0.8, safety_ceiling)
            retained_visits = int(telemetry.get("visits_retained", 0) or 0) if isinstance(telemetry, dict) else 0
            if telemetry and telemetry.get("ponder_hit") and retained_visits >= 800:
                time_for_move *= 0.75
            elif retained_visits < 100:
                time_for_move = min(time_for_move * 1.30, safety_ceiling)
            overhead = max(0, int(self.move_overhead_ms))
            if remaining <= 25:
                return max(1, min(int(round(time_for_move)), remaining))
            adjusted = time_for_move - overhead
            max_safe = max(25, remaining - overhead)
            return max(25, min(int(round(adjusted)), max_safe))

        return 600

    def handle_go(self, args: list[str]):
        if "ponder" not in args and "searchmoves" not in args and self.own_book:
            book_uci, book_ponder = self.lookup_book_move(self.board)
            if book_uci is not None:
                self.compute_move_time_ms(args, consume_ponder=True)
                bm_line = f"bestmove {book_uci} ponder {book_ponder}" if book_ponder else f"bestmove {book_uci}"
                self._emit_lines(
                    [
                        f"info depth 1 seldepth 1 score cp 0 nodes 0 nps 0 time 1 pv {book_uci}",
                        "info string book move",
                        bm_line,
                    ]
                )
                if self.use_ponder and self.ponder_manager is not None:
                    board_after = self.board.copy()
                    board_after.push_uci(book_uci)
                    if not board_after.is_game_over():
                        expected = chess.Move.from_uci(book_ponder) if book_ponder else None
                        self.ponder_manager.start_internal_ponder(board_after, expected_reply=expected)
                return

        self.ensure_engine()

        movetime_ms = self.compute_move_time_ms(args, consume_ponder=True)

        if "ponder" in args:
            self.ponder_manager.start_external_uci_ponder(
                self.board,
                movetime_ms=movetime_ms,
                uci_callback=self._on_uci_ponder_finished,
            )
            return

        t0 = time.perf_counter()
        tensor = board_to_tensor(self.board, layout=self.encoder_layout)
        _, best_uci, score, nodes, pv, depth = self.search_engine.search(
            self.board, tensor, movetime_ms=movetime_ms
        )
        elapsed_ms = max(1, int(round((time.perf_counter() - t0) * 1000.0)))

        prof = getattr(self.search_engine, "last_profile", {}) or {}
        if best_uci != "0000":
            seldepth = max(1, int(depth or 1), int(prof.get("max_leaf_ply", 0) or 0))
            d_avg = int(round(float(prof.get("avg_leaf_ply", 0) or 0)))
            info_depth = max(1, min(seldepth, d_avg if d_avg > 0 else int(depth or 1)))
        else:
            seldepth = 0
            info_depth = 0

        info_line = self._format_info_line(
            depth=info_depth,
            seldepth=seldepth,
            score=score,
            nodes=nodes,
            elapsed_ms=elapsed_ms,
            pv=pv,
            fallback_uci=best_uci,
            mate_moves=prof.get("mate_moves"),
        )

        ponder_uci = pv[1] if (best_uci != "0000" and pv and len(pv) >= 2) else None
        bm_line = f"bestmove {best_uci} ponder {ponder_uci}" if ponder_uci else f"bestmove {best_uci}"
        self._emit_lines([info_line, bm_line])

        if self.use_ponder and best_uci != "0000" and self.ponder_manager is not None:
            board_after = self.board.copy()
            board_after.push_uci(best_uci)
            if not board_after.is_game_over():
                expected = chess.Move.from_uci(ponder_uci) if ponder_uci else None
                self.ponder_manager.start_internal_ponder(board_after, expected_reply=expected)


def main():
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        print("NeuraChess UCI engine (`python -m neurachess.uci` or `neurachess-uci`). Reads UCI commands from stdin.")
        return
    engine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
