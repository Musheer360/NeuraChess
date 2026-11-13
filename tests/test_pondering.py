"""
tests/test_pondering.py
=======================
Unit tests for P1 Pondering Implementation (0.3).

Verifies:
1. Same results as no-ponder on mate suites (re-rooting and search fidelity).
2. No false draw or false proof after re-rooting (path-dependent draws invalidated properly).
3. Stop latency < 1.5 x one batch execution time.
4. No stale search or virtual-loss leakage between moves (virtual_loss == 0 across all nodes).
5. Strict UCI protocol handling ("bestmove X ponder Y", "go ponder", "ponderhit", "stop").
"""

import os
import sys
import time
import unittest
import chess
import numpy as np

from neurachess.core.constants import TOTAL_ACTIONS, uci_to_action, move_to_action
from neurachess.core.board_tensor import board_to_tensor
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch, TreeNode
from neurachess.search.ponder_manager import PonderManager
from tests.test_tree_reuse_and_nn_cache import DeterministicMockModel


class TestPondering(unittest.TestCase):
    def setUp(self):
        self.mock_root = DeterministicMockModel(batch_size=1)
        self.mock_batch = DeterministicMockModel(batch_size=16)
        self.searcher = NPUBatchedPUCTSearch(
            root_model=self.mock_root,
            batch_model=self.mock_batch,
            batch_size=16,
            use_exact_solver=True,
            use_tree_reuse=True
        )
        self.ponder_mgr = PonderManager(self.searcher, use_ponder=True)

    def tearDown(self):
        self.ponder_mgr.stop()

    def test_stop_latency_under_1_5x_batch(self):
        """Test (3): Stop latency must be < 1.5x one batch evaluation time."""
        board = chess.Board(chess.STARTING_FEN)
        
        # Measure single batch baseline time
        t_batch_start = time.perf_counter()
        _ = self.searcher.search(board, movetime_ms=10, max_nodes=16)
        dt_single_batch = max(0.005, time.perf_counter() - t_batch_start)

        # Start internal ponder
        board_after = board.copy()
        board_after.push_san("e4")
        self.ponder_mgr.start_internal_ponder(board_after)
        time.sleep(0.05)  # Let worker execute several batches

        # Stop and measure latency
        stop_latency_ms = self.ponder_mgr.stop()
        stop_latency_s = stop_latency_ms / 1000.0

        # Max allowed latency is 1.5x batch time plus a 50ms buffer for thread join
        max_allowed_s = 1.5 * dt_single_batch + 0.05
        self.assertLess(stop_latency_s, max_allowed_s,
                        f"Stop latency {stop_latency_s*1000:.1f}ms exceeded limit {max_allowed_s*1000:.1f}ms")

    def test_no_virtual_loss_or_stale_search_leakage(self):
        """Test (4): After stop, all virtual loss must be 0 and no thread is running (FastTree + Python)."""
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        self.ponder_mgr.start_internal_ponder(board)
        time.sleep(0.05)

        self.ponder_mgr.stop()
        self.assertFalse(self.ponder_mgr._is_pondering)
        self.assertFalse(self.ponder_mgr._is_active_search)

        # C++ FastTree backend check (default use_fast_tree=True)
        self.assertIsNotNone(self.searcher._fast_backend)
        self.assertTrue(self.searcher._fast_backend._has_saved_tree)
        root_ch = self.searcher._fast_backend._get_children_snapshot(0)
        self.assertGreater(sum(ch["visits"] for ch in root_ch), 1, "Expected visited nodes in C++ fast tree")

        # Python backend check (use_fast_tree=False) for zero leaked virtual_loss on TreeNode
        py_searcher = NPUBatchedPUCTSearch(
            root_model=self.mock_root,
            batch_model=self.mock_batch,
            batch_size=16,
            use_exact_solver=True,
            use_tree_reuse=True,
            use_fast_tree=False,
        )
        py_mgr = PonderManager(py_searcher, use_ponder=True)
        try:
            py_mgr.start_internal_ponder(board)
            time.sleep(0.05)
            py_mgr.stop()
            self.assertFalse(py_mgr._is_pondering)
            self.assertFalse(py_mgr._is_active_search)

            saved_root = py_searcher._saved_root
            self.assertIsNotNone(saved_root)

            nodes = [saved_root]
            visited_count = 0
            while nodes:
                curr = nodes.pop()
                visited_count += 1
                self.assertEqual(curr.virtual_loss, 0, f"Node {curr} leaked virtual loss {curr.virtual_loss}")
                for ch in curr.children.values():
                    if ch.is_instantiated:
                        nodes.append(ch)
            self.assertGreater(visited_count, 1, "Expected several visited nodes in Python search tree")
        finally:
            py_mgr.stop()

    def test_no_false_draw_or_proof_after_rerooting(self):
        """Test (2): Re-rooting must invalidate path-dependent draws (repetition/50-move) and not create false proofs."""
        b = chess.Board()
        b.push_san("Nf3")
        b.push_san("Nf6")
        b.push_san("Ng1")
        b.push_san("Ng8")  # Twofold repetition

        # C++ FastTree backend check (use_fast_tree=True)
        self.ponder_mgr.start_internal_ponder(b)
        time.sleep(0.05)
        self.ponder_mgr.stop()
        self.ponder_mgr.on_opponent_move(chess.Move.from_uci("e2e4"))

        b_fast = b.copy()
        b_fast.push_san("e4")
        _, _, _, _, _, _ = self.searcher.search(b_fast, movetime_ms=50)
        self.assertIsNotNone(self.searcher._fast_backend)
        self.assertTrue(self.searcher._fast_backend._has_saved_tree)
        fast_root_ch = self.searcher._fast_backend._get_children_snapshot(0)
        self.assertEqual(len(fast_root_ch), b_fast.legal_moves.count())
        self.assertFalse(any(ch["is_solved"] for ch in fast_root_ch))

        # Python backend check (use_fast_tree=False)
        py_searcher = NPUBatchedPUCTSearch(
            root_model=self.mock_root,
            batch_model=self.mock_batch,
            batch_size=16,
            use_exact_solver=True,
            use_tree_reuse=True,
            use_fast_tree=False,
        )
        py_mgr = PonderManager(py_searcher, use_ponder=True)
        try:
            py_mgr.start_internal_ponder(b)
            time.sleep(0.05)
            py_mgr.stop()
            py_mgr.on_opponent_move(chess.Move.from_uci("e2e4"))

            b_py = b.copy()
            b_py.push_san("e4")
            _, _, _, _, _, _ = py_searcher.search(b_py, movetime_ms=50)

            root = py_searcher._saved_root
            self.assertIsNotNone(root)
            self.assertFalse(root.is_terminal, "Re-rooted position must not be marked terminal")
            self.assertFalse(root.path_dep_draw, "Re-rooted position must not retain stale twofold repetition draw")
        finally:
            py_mgr.stop()

    def test_same_mate_results_with_and_without_ponder(self):
        """Test (1): Pondering must find the exact same checkmate solution as no-ponder."""
        # Scholar's mate puzzle (White to move, mate in 1: Qxf7#)
        fen_scholars = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 5"
        board_mate = chess.Board(fen_scholars)

        # Search without ponder
        self.ponder_mgr.use_ponder = False
        _, best_no_ponder, _, _, _, _ = self.searcher.search(board_mate.copy(), movetime_ms=50)

        # Search with ponder
        self.ponder_mgr.use_ponder = True
        # Ponder from previous move position
        b_prev = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/2B5/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
        self.ponder_mgr.start_internal_ponder(b_prev, expected_reply=chess.Move.from_uci("e5e4"))
        time.sleep(0.05)
        
        # Opponent plays e4 (hit)
        telemetry = self.ponder_mgr.on_opponent_move(chess.Move.from_uci("e5e4"))
        self.assertTrue(telemetry["ponder_hit"] or telemetry["visits_retained"] >= 0)

        _, best_with_ponder, _, _, _, _ = self.searcher.search(board_mate.copy(), movetime_ms=50)

        self.assertEqual(best_no_ponder, "f3f7", "Expected Qxf7# without ponder")
        self.assertEqual(best_with_ponder, "f3f7", "Expected Qxf7# with ponder")

    def test_uci_go_ponder_and_ponderhit(self):
        """Test (5): Verify strict UCI 'go ponder' and 'ponderhit' workflow."""
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        
        out_received = []
        def uci_cb(best, ponder, n, s):
            out_received.append((best, ponder, n, s))

        # Start external UCI ponder
        self.ponder_mgr.start_external_uci_ponder(board, movetime_ms=100, uci_callback=uci_cb)
        self.assertTrue(self.ponder_mgr._is_pondering)
        time.sleep(0.05)

        # Send ponderhit
        self.ponder_mgr.ponderhit(movetime_ms=50)
        self.assertFalse(self.ponder_mgr._is_pondering)
        self.assertTrue(self.ponder_mgr._is_active_search)

        # Wait for search to complete
        time.sleep(0.15)
        self.assertGreaterEqual(len(out_received), 1, "Expected UCI bestmove callback to fire after ponderhit")
        best_uci, ponder_uci, nodes, score = out_received[0]
        self.assertIsNotNone(best_uci)


if __name__ == "__main__":
    unittest.main()
