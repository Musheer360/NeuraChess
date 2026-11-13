"""
NeuraChess Universal Chess Interface (UCI) Protocol Driver.

Connects NeuraChess to UCI chess GUIs and tournament managers (CuteChess, Arena, Banksia).
Supports pondering (`Ponder`, `bestmove X ponder Y`, `go ponder`, `ponderhit`, `stop`) and
`EncoderLayout` selection (`gpu_unpack`, `legacy_flip`).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chess

from core.board_tensor import board_to_tensor
from core.constants import value_to_centipawns
from models.npu_compiler import DEFAULT_CACHE_DIR, NPUModel, resolve_default_model_paths
from search.npu_batched_puct import NPUBatchedPUCTSearch
from search.ponder_manager import PonderManager


class UCIEngine:
    def __init__(self, encoder_layout: str = "gpu_unpack"):
        self.board = chess.Board()
        self.history: list[chess.Board] = []
        self.encoder_layout = encoder_layout
        self.use_ponder = False
        self.search_engine: NPUBatchedPUCTSearch | None = None
        self.ponder_manager: PonderManager | None = None
        self._pending_ponder_telemetry: dict | None = None

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
                    batch_size=16,
                    cache_dir=DEFAULT_CACHE_DIR,
                    device="NPU",
                    use_turbo=True,
                    performance_hint="LATENCY",
                    ov_ir_xml_path=xml_path,
                    verbose=False,
                )
            else:
                root_m = NPUModel(batch_size=1, verbose=False)
                batch_m = NPUModel(batch_size=16, verbose=False)

            self.search_engine = NPUBatchedPUCTSearch(
                root_model=root_m,
                batch_model=batch_m,
                batch_size=16,
                c_puct=1.5,
                use_fpu_reduction=True,
                fpu_reduction=0.20,
                root_fpu_reduction=0.10,
                t_root=1.0,
                use_exact_solver=True,
                use_tree_reuse=True,
                use_nn_cache=True,
                use_fast_tree=True,
                encoder_layout=self.encoder_layout,
            )
            self.ponder_manager = PonderManager(self.search_engine, use_ponder=self.use_ponder)

    def _on_uci_ponder_finished(self, best_uci: str, ponder_uci: str | None, nodes: int, score: float):
        cp_score = value_to_centipawns(score)
        print(f"info score cp {cp_score} nodes {nodes}")
        if ponder_uci:
            print(f"bestmove {best_uci} ponder {ponder_uci}")
        else:
            print(f"bestmove {best_uci}")
        sys.stdout.flush()

    def run(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue

                tokens = line.split()
                cmd = tokens[0]

                if cmd == "uci":
                    print("id name NeuraChess 1.0 (OpenVINO NPU)")
                    print("id author Musheer Alam")
                    print("option name Hash type spin default 64 min 1 max 1024")
                    print("option name Ponder type check default false")
                    print("option name EncoderLayout type combo default gpu_unpack var gpu_unpack var legacy_flip")
                    print("uciok")
                    sys.stdout.flush()

                elif cmd == "setoption":
                    if "Ponder" in line:
                        val = tokens[-1].strip().lower()
                        self.use_ponder = (val == "true")
                        if self.ponder_manager:
                            self.ponder_manager.use_ponder = self.use_ponder
                    elif "EncoderLayout" in line:
                        layout_val = tokens[-1].strip()
                        if layout_val in ("gpu_unpack", "legacy_flip"):
                            self.encoder_layout = layout_val
                            if self.search_engine:
                                self.search_engine.encoder_layout = layout_val

                elif cmd == "isready":
                    self.ensure_engine()
                    print("readyok")
                    sys.stdout.flush()

                elif cmd == "ucinewgame":
                    if self.ponder_manager:
                        self.ponder_manager.stop()
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
                        self.ponder_manager.stop()

                elif cmd == "quit":
                    if self.ponder_manager:
                        self.ponder_manager.stop()
                    break

            except Exception as e:
                print(f"info string error: {e}", file=sys.stderr)
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
                self.ponder_manager.stop()
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
                    pass
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
                self.ponder_manager.last_telemetry = {**self.ponder_manager.last_telemetry, "ponder_hit": False}

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
            time_for_move = min(remaining * 0.04 + inc * 0.8, remaining * 0.25)
            if telemetry and telemetry.get("ponder_hit") and telemetry.get("visits_retained", 0) >= 800:
                time_for_move *= 0.75
            if remaining <= 30:
                return max(1, min(int(round(time_for_move)), remaining))
            return max(10, min(int(round(time_for_move)), remaining - 20))

        return 600

    def handle_go(self, args: list[str]):
        self.ensure_engine()

        movetime_ms = self.compute_move_time_ms(args, consume_ponder=True)

        if "ponder" in args:
            self.ponder_manager.start_external_uci_ponder(
                self.board,
                movetime_ms=movetime_ms,
                uci_callback=self._on_uci_ponder_finished,
            )
            return

        tensor = board_to_tensor(self.board, layout=self.encoder_layout)
        _, best_uci, score, nodes, pv, depth = self.search_engine.search(
            self.board, tensor, movetime_ms=movetime_ms
        )

        cp_score = value_to_centipawns(score)
        pv_str = " ".join(pv) if pv else best_uci

        print(f"info depth {depth} score cp {cp_score} nodes {nodes} pv {pv_str}")

        ponder_uci = pv[1] if (pv and len(pv) >= 2) else None
        if ponder_uci:
            print(f"bestmove {best_uci} ponder {ponder_uci}")
        else:
            print(f"bestmove {best_uci}")
        sys.stdout.flush()

        if self.use_ponder and best_uci != "0000":
            board_after = self.board.copy()
            board_after.push_uci(best_uci)
            expected = chess.Move.from_uci(ponder_uci) if ponder_uci else None
            self.ponder_manager.start_internal_ponder(board_after, expected_reply=expected)


def main():
    engine = UCIEngine()
    engine.run()


if __name__ == "__main__":
    main()
