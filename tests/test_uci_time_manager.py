"""
Unit tests for UCI Blitz/Rapid time management and ponder-hit scaling in `uci/engine_uci.py`.
"""

import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
import chess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from neurachess.scripts.match_runner import _select_engine_move
from neurachess.uci.engine_uci import UCIEngine


class _RecordingSearchEngine:
    def __init__(self):
        self.encoder_layout = "gpu_unpack"
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
        self.last_telemetry = dict(telemetry)
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
        # Steady-state (>=100 retained visits): min(60000*0.04 + 1000*0.8, 60000*0.25) = 3200
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": False, "visits_retained": 250})
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
        # Cold-start (<100 retained visits): 1.30x buffer -> 3200 * 1.30 = 4160
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": False, "visits_retained": 99})
        uci.board = chess.Board()
        self.assertEqual(
            uci.compute_move_time_ms(["wtime", "60000", "btime", "40000", "winc", "1000", "binc", "500"]),
            4160,
        )
        # No ponder manager (cold start = 0 visits), capped at 25% of remaining: wtime=2000, winc=2000 -> 500
        uci.ponder_manager = None
        uci.board = chess.Board()
        self.assertEqual(
            uci.compute_move_time_ms(["wtime", "2000", "btime", "2000", "winc", "2000", "binc", "0"]),
            500,
        )

    def test_ponder_hit_scaling_threshold_and_single_turn_consumption(self):
        uci = UCIEngine(own_book=False)
        uci.search_engine = _RecordingSearchEngine()

        # Case 1: ponder_hit=True with >= 800 retained visits -> 25% reduction (3200 * 0.75 = 2400)
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": True, "visits_retained": 800})
        uci.handle_position(["startpos", "moves", "e2e4"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 2400)

        # Telemetry was consumed (ponder_hit=False, visits_retained=0 -> cold-start 1.30x = 4160)
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 4160)

        # Case 2: ponder_hit=True with 799 visits (< 800, >= 100) -> neutral 1.0x (3200)
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": True, "visits_retained": 799})
        uci.handle_position(["startpos", "moves", "e2e4"])
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 3200)

        # Case 3: ponder_hit=False even with >= 800 visits -> neutral 1.0x (3200)
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": False, "visits_retained": 1500})
        uci.handle_position(["startpos", "moves", "e2e4"])
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 3200)

        # Case 4: ponder_hit=False with < 100 visits -> cold-start 1.30x buffer capped by 25% ceiling
        uci.ponder_manager = _RecordingPonderManager({"ponder_hit": False, "visits_retained": 0})
        uci.handle_position(["startpos", "moves", "e2e4"])
        with redirect_stdout(buf):
            uci.handle_go(["wtime", "60000", "btime", "60000", "winc", "1000", "binc", "1000"])
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 4160)

    def test_uci_own_book_option_and_lookup(self):
        uci = UCIEngine(book_seed=7)
        uci.search_engine = _RecordingSearchEngine()
        self.assertTrue(uci.own_book)
        self.assertTrue(uci.book_path.exists())
        self.assertLessEqual(uci.book_path.stat().st_size, 200 * 1024)

        # Startpos hits CC0 opening book without running PUCT search
        uci.handle_position(["startpos"])
        buf = io.StringIO()
        with redirect_stdout(buf):
            uci.handle_go(["movetime", "300"])
        out = buf.getvalue()
        self.assertIn("bestmove ", out)
        self.assertEqual(len(uci.search_engine.last_movetime_ms), 0)
        best_mv = [ln for ln in out.splitlines() if ln.startswith("bestmove ")][0].split()[1]
        self.assertIn(chess.Move.from_uci(best_mv), uci.board.legal_moves)

        # Turning OwnBook off routes straight to PUCT search
        uci.own_book = False
        buf = io.StringIO()
        with redirect_stdout(buf):
            uci.handle_go(["movetime", "300"])
        self.assertEqual(len(uci.search_engine.last_movetime_ms), 1)
        self.assertEqual(uci.search_engine.last_movetime_ms[-1], 300)

        # UCI loop exposes OwnBook option defaulting to true and updates on setoption
        uci2 = UCIEngine()
        fake_stdin = io.StringIO("uci\nsetoption name OwnBook value false\nquit\n")
        buf2 = io.StringIO()
        orig_stdin = sys.stdin
        try:
            sys.stdin = fake_stdin
            with redirect_stdout(buf2):
                uci2.run()
        finally:
            sys.stdin = orig_stdin
        uci_out = buf2.getvalue()
        self.assertIn("option name OwnBook type check default true", uci_out)
        self.assertFalse(uci2.own_book)

    def test_cc0_opening_book_legality_and_match_runner_toggle(self):
        book_path = os.path.join(ROOT_DIR, "assets", "opening_book.json")
        self.assertTrue(os.path.exists(book_path))
        self.assertLessEqual(os.path.getsize(book_path), 200 * 1024)
        with open(book_path, "r", encoding="utf-8") as bf:
            payload = json.load(bf)
        self.assertEqual(payload.get("license"), "CC0-1.0")
        self.assertIn("lichess-org/chess-openings", str(payload.get("source", "")))
        positions = payload.get("positions", {})
        self.assertGreaterEqual(len(positions), 1000)

        for epd, entries in positions.items():
            b = chess.Board(f"{epd} 0 1")
            for uci, weight in entries:
                self.assertGreater(weight, 0)
                self.assertIn(chess.Move.from_uci(uci), b.legal_moves)

        # match_runner toggle: None runs search, book_driver hits book on startpos
        rec = _RecordingSearchEngine()
        start_b = chess.Board()
        mv_search, nodes_search, _, _ = _select_engine_move(start_b, rec, 250, book_driver=None)
        self.assertEqual(nodes_search, 10)
        self.assertEqual(len(rec.last_movetime_ms), 1)
        self.assertIn(chess.Move.from_uci(mv_search), start_b.legal_moves)

        book_driver = UCIEngine(own_book=True, book_seed=7)
        mv_book, nodes_book, _, _ = _select_engine_move(start_b, rec, 250, book_driver=book_driver)
        self.assertEqual(nodes_book, 0)
        self.assertEqual(len(rec.last_movetime_ms), 1)
        self.assertIn(chess.Move.from_uci(mv_book), start_b.legal_moves)


if __name__ == "__main__":
    unittest.main()
