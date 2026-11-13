"""Unit tests for tensor state transition parity against python-chess."""

import sys
import unittest
from pathlib import Path

import chess
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.board_tensor import board_to_tensor
from core.constants import uci_to_action
from core.tensor_transitions import apply_tensor_transition


class TestTransitions(unittest.TestCase):
    def _assert_transition_matches(self, fen: str, uci: str) -> None:
        board = chess.Board(fen)
        state = board_to_tensor(board, layout="legacy_flip")
        act = uci_to_action(uci, canonical_flip=(board.turn == chess.BLACK))
        pred_next = apply_tensor_transition(state, act)
        board.push_uci(uci)
        expected_next = board_to_tensor(board, layout="legacy_flip")
        np.testing.assert_allclose(
            pred_next[:17],
            expected_next[:17],
            atol=1e-6,
            err_msg=f"Tensor mismatch after {uci} from {fen}",
        )

    def test_standard_moves_and_captures(self) -> None:
        self._assert_transition_matches(chess.STARTING_FEN, "e2e4")
        self._assert_transition_matches("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "c7c5")
        self._assert_transition_matches("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2", "e4d5")

    def test_castling_and_promotions(self) -> None:
        # Kingside castling White & Black
        self._assert_transition_matches("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1")
        self._assert_transition_matches("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1")
        self._assert_transition_matches("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8g8")
        # Promotion (kings safely separated)
        self._assert_transition_matches("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8q")
        self._assert_transition_matches("8/P7/8/8/8/8/8/4K2k w - - 0 1", "a7a8n")


if __name__ == "__main__":
    unittest.main()
