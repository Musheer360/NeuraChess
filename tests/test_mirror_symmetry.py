"""
Unit Test: Position & Colour-Mirror Symmetry (tests/test_mirror_symmetry.py)
=============================================================================
Verifies that the engine produces identical policy priors and value evaluation
for a position and its colour-mirror after canonical mirror remapping.
"""

import sys
import os
import unittest
import chess
import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.constants import move_to_action
from core.board_tensor import (
    board_to_tensor,
    ENCODER_LAYOUT_LEGACY_FLIP,
    ENCODER_LAYOUT_GPU_UNPACK,
)
from models.pytorch_se_resnet import ChessSEResNet


class TestMirrorSymmetry(unittest.TestCase):
    def test_mirror_tensor_identity_legacy_flip(self):
        """Under legacy_flip layout, board_to_tensor(P) and board_to_tensor(P.mirror()) are bitwise identical."""
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        b_w = chess.Board(fen)
        b_b = b_w.mirror()

        t_w = board_to_tensor(b_w, layout=ENCODER_LAYOUT_LEGACY_FLIP)
        t_b = board_to_tensor(b_b, layout=ENCODER_LAYOUT_LEGACY_FLIP)

        np.testing.assert_array_equal(t_w, t_b, err_msg="Legacy flip tensor representation must be identical under mirror.")
        print("  [PASS] test_mirror_tensor_identity_legacy_flip")

    def test_mirror_move_action_identity(self):
        """Action indices of every legal move in P must match its mirrored move in P.mirror()."""
        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        b_w = chess.Board(fen)
        b_b = b_w.mirror()

        w_legals = list(b_w.legal_moves)
        b_legals = set(b_b.legal_moves)

        self.assertEqual(len(w_legals), len(b_legals))

        for m_w in w_legals:
            # Construct mirrored move for black
            m_b = chess.Move(
                chess.square_mirror(m_w.from_square),
                chess.square_mirror(m_w.to_square),
                promotion=m_w.promotion
            )
            self.assertIn(m_b, b_legals, f"Mirrored move {m_b} not found in black legal moves!")

            act_w = move_to_action(m_w, canonical_flip=False)
            act_b = move_to_action(m_b, canonical_flip=True)
            self.assertEqual(act_w, act_b, f"Action mismatch: {m_w} -> {act_w} vs {m_b} -> {act_b}")

        print("  [PASS] test_mirror_move_action_identity")

    def test_engine_inference_mirror_symmetry_legacy_flip(self):
        """Tests that a model evaluated on P and P.mirror() produces identical legal policy distribution and value."""
        torch.manual_seed(42)
        model = ChessSEResNet(num_blocks=6, channels=128)
        model.eval()

        fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        b_w = chess.Board(fen)
        b_b = b_w.mirror()

        t_w = torch.from_numpy(board_to_tensor(b_w, layout=ENCODER_LAYOUT_LEGACY_FLIP)).unsqueeze(0)
        t_b = torch.from_numpy(board_to_tensor(b_b, layout=ENCODER_LAYOUT_LEGACY_FLIP)).unsqueeze(0)

        with torch.no_grad():
            p_w, v_w = model(t_w)
            p_b, v_b = model(t_b)

        # Values must be bit-identical
        np.testing.assert_allclose(v_w.numpy(), v_b.numpy(), atol=1e-6, err_msg="Value prediction must match under mirror.")

        # Every legal move's probability must match
        legals_w = list(b_w.legal_moves)
        acts_w = [move_to_action(m, canonical_flip=False) for m in legals_w]
        acts_b = [
            move_to_action(
                chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), promotion=m.promotion),
                canonical_flip=True
            )
            for m in legals_w
        ]

        probs_w = torch.softmax(p_w[0, acts_w], dim=-1).numpy()
        probs_b = torch.softmax(p_b[0, acts_b], dim=-1).numpy()

        np.testing.assert_allclose(probs_w, probs_b, atol=1e-6, err_msg="Policy probabilities must match under mirror.")
        print("  [PASS] test_engine_inference_mirror_symmetry_legacy_flip")


if __name__ == "__main__":
    t = TestMirrorSymmetry()
    t.test_mirror_tensor_identity_legacy_flip()
    t.test_mirror_move_action_identity()
    t.test_engine_inference_mirror_symmetry_legacy_flip()
    print("\nALL 3 MIRROR SYMMETRY UNIT TESTS PASSED!")
