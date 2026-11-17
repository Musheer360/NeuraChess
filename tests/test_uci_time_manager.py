"""
Unit tests for UCI Blitz/Rapid time management, CC0 opening book, and ponder-hit scaling in `neurachess.uci.engine_uci`.
"""

import io
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock
import chess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from neurachess.scripts import match_runner
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

    def test_h2h_forwards_openings_syzygy_fpu_and_adjudication(self):
        class _WarmableEngine(_RecordingSearchEngine):
            def search(self, board: chess.Board, board_tensor, movetime_ms: int = 600, max_nodes=None):
                return super().search(board, board_tensor, movetime_ms=movetime_ms)

        build_calls: list[dict] = []

        def _fake_build(xml_path, **kwargs):
            build_calls.append({"xml": xml_path, **kwargs})
            return _WarmableEngine()

        with tempfile.TemporaryDirectory(prefix="nc_h2h_cfg_") as td:
            td_p = Path(td)
            op_file = td_p / "openings.json"
            op_file.write_text(json.dumps([["e2e4"], ["d2d4", "d7d5"]]), encoding="utf-8")
            out_file = td_p / "h2h.json"
            args = Namespace(
                xml_a="a.xml",
                xml_b="b.xml",
                pairs=1,
                movetime_ms=250,
                batch_size=16,
                max_plies=4,
                device="CPU",
                dynamic_gear_a=False,
                dynamic_gear_b=False,
                cache_dir_a=None,
                cache_dir_b=None,
                openings_file=op_file,
                start_pair=1,
                adjudicate_material=True,
                fpu_reduction=0.20,
                syzygy_path="weights/syzygy",
                own_book=False,
                output=out_file,
            )
            with mock.patch.object(match_runner, "_build_engine", side_effect=_fake_build):
                with redirect_stdout(io.StringIO()):
                    summary = match_runner.run_h2h_match(args)

            self.assertEqual(len(build_calls), 2)
            for call in build_calls:
                self.assertEqual(call["fpu_reduction"], 0.20)
                self.assertEqual(call["syzygy_path"], "weights/syzygy")
                self.assertEqual(call["movetime_ms"], 250)
            self.assertEqual(summary["total_games"], 2)
            self.assertEqual(summary["games"][0]["pair"], 1)
            self.assertTrue(summary["games"][0]["adjudicated"])
            self.assertTrue(out_file.is_file())


if __name__ == "__main__":
    unittest.main()
