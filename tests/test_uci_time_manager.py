"""
Unit tests for UCI Blitz/Rapid time management and ponder-hit scaling in `uci/engine_uci.py`.
"""

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
import chess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from uci.engine_uci import UCIEngine


class _RecordingSearchEngine:
    def __init__(self):
        self.last_movetime_ms: list[int] = []
        self.cleared = 0

    def clear_state(self):
        self.cleared += 1

    def search(self, board: chess.Board, board_tensor, movetime_ms: int = 600):
        self.last_movetime_ms.append(int(movetime_ms))
        mv = next(iter(board.legal_moves))
        return 0, mv.uci(), 0.1, 10, [mv.uci()], 2


class _RecordingPonderManager:
    def __init__(self, telemetry: dict):
        self.telemetry = telemetry
        self.use_ponder = False
        self.stopped = 0

    def stop(self):
        self.stopped += 1
        return 0.0

    def on_opponent_move(self, actual_move: chess.Move):
        return dict(self.telemetry)

    def start_internal_ponder(self, board_after, expected_reply=None):
        pass


class TestUCITimeManager(unittest.TestCase):
    def test_movetime_precedence_and_fallback(self):
        uci = UCIEngine()
        self.assertEqual(uci.compute_move_time_ms(["movetime", "350"]), 350)
        self.assertEqual(
            uci.compute_move_time_ms(["wtime", "60000", "btime", "60000", "movetime", "420"]),
            420,
        )
        self.assertEqual(uci.compute_move_time_ms([]), 600)

    def test_blitz_rapid_formula_and_cap(self):
        uci = UCIEngine()
        # White to move: min(60000*0.04 + 1000*0.8, 60000*0.25) = 3200
        uci.board = chess.Board()
        self.assertEqual(
            uci.compute_move_time_ms(["wtime", "60000", "btime", "40000", "winc", "1000", "binc", "500"]),
            3200,
        )
        # Black to move with movestogo: min(40000*0.04 + 500*0.8, 40000*0.25) = 2000
        uci.board.push_san("e4")
        self.assertEqual(
            uci.compute_move_time_ms(
                ["wtime", "60000", "btime", "40000", "winc", "1000", "binc", "500", "movestogo", "40"]
            ),
            2000,
        )
        # Increment capped at 25% of remaining: wtime=2000, winc=2000 -> min(1680, 500) = 500
        uci.board = chess.Board()
        self.assertEqual(
            uci.compute_move_time_ms(["wtime", "2000", "btime", "2000", "winc", "2000", "binc", "0"]),
            500,
        )

    def test_ponder_hit_scaling_threshold_and_single_turn_consumption(self):
        uci = UCIEngine()
        uci.search_engine = _RecordingSearchEngine()

        # Case 1: ponder_hit=True with >= 800 retained visits -> 25% reduction (3200 * 0.75 = 2400)
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": True, "visits_retained": 800})
        uci.handle_position(["startpos", "moves", "e2e4"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 2400)

        # Telemetry was consumed on the turn; a bare follow-up go does not re-apply it
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 3200)

        # Case 2: ponder_hit=True with 799 visits (< 800) -> no reduction (3200)
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": True, "visits_retained": 799})
        uci.handle_position(["startpos", "moves", "e2e4"])
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 3200)

        # Case 3: ponder_hit=False even with >= 800 visits -> no reduction (3200)
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": False, "visits_retained": 1500})
        uci.handle_position(["startpos", "moves", "e2e4"])
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 3200)


if __name__ == "__main__":
    unittest.main()
