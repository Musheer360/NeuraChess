"""Unit tests verifying board encoding, vertical mirror symmetry, and legal mask consistency."""

import sys
import unittest
from pathlib import Path

import chess
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.board_tensor import board_to_tensor
from core.constants import TOTAL_ACTIONS, action_to_uci, uci_to_action
from core.tensor_movegen import action_to_move, get_legal_action_indices, get_legal_action_mask


class TestEncoderConsistency(unittest.TestCase):
    def test_500_random_positions_mirror_and_roundtrip(self) -> None:
        rng = np.random.default_rng(42)
        board = chess.Board()
        positions_tested = 0
        total_moves_tested = 0

        while positions_tested < 500:
            if board.is_game_over():
                board = chess.Board()
                continue

            is_black = board.turn == chess.BLACK
            t_orig = board_to_tensor(board, layout="legacy_flip")
            m_board = board.mirror()
            t_mirror = board_to_tensor(m_board, layout="legacy_flip")

            # Channel 12 in legacy_flip encodes side-to-move (flipped); mirrored board toggles side-to-move.
            np.testing.assert_allclose(t_orig[:12], t_mirror[:12], atol=1e-6)
            self.assertEqual(t_orig.shape, (18, 8, 8))

            mask = get_legal_action_mask(board)
            indices = get_legal_action_indices(board)
            legal_moves = list(board.legal_moves)
            self.assertEqual(int(mask.sum()), len(legal_moves))
            self.assertEqual(len(indices), len(legal_moves))

            for mv in legal_moves:
                act = uci_to_action(mv.uci(), canonical_flip=is_black)
                self.assertEqual(mask[act], 1.0)
                self.assertEqual(action_to_move(board, act), mv)
                piece = board.piece_at(mv.from_square)
                if piece is not None and piece.piece_type == chess.PAWN:
                    self.assertEqual(action_to_uci(act, canonical_flip=is_black), mv.uci())
                else:
                    self.assertEqual(action_to_uci(act, canonical_flip=is_black)[:4], mv.uci()[:4])
                total_moves_tested += 1

            positions_tested += 1
            board.push(legal_moves[rng.integers(len(legal_moves))])

        self.assertGreater(total_moves_tested, 5000)


if __name__ == "__main__":
    unittest.main()
