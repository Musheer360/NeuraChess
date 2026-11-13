"""
Unit tests for `use_opening_root_guard` across Python PUCT and FastTree C++ backends.
"""

import os
import sys
import unittest
import chess
import numpy as np

from neurachess.core.constants import TOTAL_ACTIONS, uci_to_action
from neurachess.engine.fast_tree import pack_move
from neurachess.search.candidate_selector import select_opening_guarded_root_act
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch


class GuardTestMockModel:
    """Custom mock returning controlled root policy logits and leaf values based on side to move."""

    def __init__(self, batch_size: int, white_root_logits: dict[str, float], black_leaf_val: float):
        self.batch_size = batch_size
        self.white_root_logits = white_root_logits
        self.black_leaf_val = black_leaf_val

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_in = x[np.newaxis, ...] if x.ndim == 3 else x
        b = x_in.shape[0]
        p = np.full((b, TOTAL_ACTIONS), -20.0, dtype=np.float32)
        v = np.zeros((b, 1), dtype=np.float32)
        for i in range(b):
            # Plane 17 is 1.0 when Black-to-move canonicalized, else 0.0 (White-to-move)
            is_black = bool(np.mean(x_in[i, 17]) > 0.5)
            if not is_black:
                for uci, logit in self.white_root_logits.items():
                    p[i, uci_to_action(uci, canonical_flip=False)] = logit
                v[i, 0] = 0.0
            else:
                v[i, 0] = self.black_leaf_val
        return p, v


class TestOpeningRootGuard(unittest.TestCase):
    def test_helper_prior_threshold_q_override_and_proven_mate(self):
        # 6 candidates: top-4 by prior are acts 1..4; act 5 has prior >= 0.08; act 6 is outside (prior=0.02)
        # Case A: outside act 6 has more visits and +0.10 Q (below +0.15 threshold) -> blocked
        cands_blocked = [
            (1, 0.30, 50, 0.10, False, False),
            (2, 0.25, 20, 0.05, False, False),
            (3, 0.15, 10, 0.00, False, False),
            (4, 0.10, 5, -0.05, False, False),
            (5, 0.09, 60, 0.12, False, False),  # allowed via prior >= 0.08
            (6, 0.02, 99, 0.20, False, False),  # Q=0.20 is only +0.10 > top3_best_q=0.10
        ]
        self.assertEqual(select_opening_guarded_root_act(cands_blocked), 5)

        # Case B: outside act 6 has +0.16 Q (>= +0.15 over top3_best_q=0.10) -> allowed
        cands_override = list(cands_blocked[:-1]) + [(6, 0.02, 99, 0.26, False, False)]
        self.assertEqual(select_opening_guarded_root_act(cands_override), 6)

        # Case C: outside act 6 is proven win even with Q delta < 0.15 -> allowed
        cands_mate = [
            (1, 0.40, 50, 0.90, False, False),
            (2, 0.20, 20, 0.10, False, False),
            (3, 0.15, 10, 0.05, False, False),
            (4, 0.10, 5, 0.00, False, False),
            (6, 0.01, 55, 0.95, True, False),
        ]
        self.assertEqual(select_opening_guarded_root_act(cands_mate), 6)

    def test_snapshot_and_python_fullmove_bypass_and_reconfigure(self):
        root_ch = [
            {"idx": 1, "act": uci_to_action("e2e4", False), "mp": 0, "visits": 80, "q": -0.10, "prior": 0.40, "is_solved": False, "solved_val": 0.0, "mate_plies": 0},
            {"idx": 2, "act": uci_to_action("d2d4", False), "mp": 0, "visits": 20, "q": -0.05, "prior": 0.25, "is_solved": False, "solved_val": 0.0, "mate_plies": 0},
            {"idx": 3, "act": uci_to_action("c2c4", False), "mp": 0, "visits": 10, "q": 0.00, "prior": 0.15, "is_solved": False, "solved_val": 0.0, "mate_plies": 0},
            {"idx": 4, "act": uci_to_action("g1f3", False), "mp": 0, "visits": 5, "q": 0.02, "prior": 0.10, "is_solved": False, "solved_val": 0.0, "mate_plies": 0},
            {"idx": 5, "act": uci_to_action("a2a3", False), "mp": 0, "visits": 120, "q": -0.20, "prior": 0.01, "is_solved": False, "solved_val": 0.0, "mate_plies": 0},
        ]
        for ch in root_ch:
            mv = chess.Move.from_uci(
                {v: k for k, v in {
                    "e2e4": uci_to_action("e2e4", False),
                    "d2d4": uci_to_action("d2d4", False),
                    "c2c4": uci_to_action("c2c4", False),
                    "g1f3": uci_to_action("g1f3", False),
                    "a2a3": uci_to_action("a2a3", False),
                }.items()}[ch["act"]]
            )
            ch["mp"] = pack_move(mv)

        for use_fast_tree in (True, False):
            s = NPUBatchedPUCTSearch(
                root_model=GuardTestMockModel(1, {}, 0.0),
                batch_model=GuardTestMockModel(8, {}, 0.0),
                batch_size=8,
                use_fast_tree=use_fast_tree,
                use_opening_root_guard=False,
            )
            self.assertFalse(s.use_opening_root_guard)
            s.reconfigure(use_opening_root_guard=True)
            self.assertTrue(s.use_opening_root_guard)
            if use_fast_tree:
                self.assertIsNotNone(s._fast_backend)
                self.assertTrue(s._fast_backend.use_opening_root_guard)
                _, uci_guarded, _, _, _ = s._fast_backend._select_best_move_from_root_snapshot(
                    root_ch, apply_opening_guard=True
                )
                self.assertEqual(uci_guarded, "e2e4")
                _, uci_unguarded, _, _, _ = s._fast_backend._select_best_move_from_root_snapshot(
                    root_ch, apply_opening_guard=False
                )
                self.assertEqual(uci_unguarded, "a2a3")

            # Mate-in-1 still found cleanly with guard ON when fullmove <= 12
            m1 = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 5")
            _, best_m1, eval_m1, _, _, _ = s.search(m1, movetime_ms=400, max_nodes=48)
            self.assertEqual(best_m1, "a1a8")
            self.assertGreaterEqual(eval_m1, 0.99)

            # After move 12 (fullmove_number=13), guard is inactive even with flag=True
            s_late = NPUBatchedPUCTSearch(
                root_model=GuardTestMockModel(1, {"e2e4": 4.0, "d2d4": 3.5, "c2c4": 3.0, "g1f3": 2.5}, 0.0),
                batch_model=GuardTestMockModel(8, {}, 0.0),
                batch_size=8,
                use_fast_tree=use_fast_tree,
                use_opening_root_guard=True,
            )
            b13 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 13")
            _, uci_late, _, _, _, _ = s_late.search(b13, movetime_ms=300, max_nodes=32)
            self.assertIn(uci_late, [m.uci() for m in b13.legal_moves])


if __name__ == "__main__":
    unittest.main()
