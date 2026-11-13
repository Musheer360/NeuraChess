"""Unit tests for canonical 4,672-action move generation and masking."""

import random
import sys
import unittest
from pathlib import Path

import chess

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.constants import TOTAL_ACTIONS, action_to_uci, uci_to_action
from core.tensor_movegen import action_to_move, get_legal_action_indices, get_legal_action_mask


class TestMoveGen(unittest.TestCase):
    def test_startpos_and_perft_depth2(self) -> None:
        board = chess.Board()
        mask = get_legal_action_mask(board)
        self.assertEqual(mask.shape, (TOTAL_ACTIONS,))
        indices = get_legal_action_indices(board)
        self.assertEqual(len(indices), 20)

        total_depth2 = 0
        for act in indices:
            mv = action_to_move(board, act)
            self.assertIn(mv, board.legal_moves)
            board.push(mv)
            total_depth2 += len(get_legal_action_indices(board))
            board.pop()
        self.assertEqual(total_depth2, 400)

    def test_special_positions_castling_ep_promotions(self) -> None:
        fens = [
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2P5/8/8/8/8/3p4/4K2k w - - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        ]
        for fen in fens:
            board = chess.Board(fen)
            is_black = board.turn == chess.BLACK
            mask = get_legal_action_mask(board)
            indices = get_legal_action_indices(board)
            self.assertEqual(len(indices), len(list(board.legal_moves)))
            self.assertEqual(int(mask.sum()), len(indices))
            for mv in board.legal_moves:
                act = uci_to_action(mv.uci(), canonical_flip=is_black)
                self.assertEqual(mask[act], 1.0)
                self.assertEqual(action_to_uci(act, canonical_flip=is_black), mv.uci())
                self.assertEqual(action_to_move(board, act), mv)

    def test_random_rollouts_match_python_chess(self) -> None:
        rng = random.Random(42)
        for _ in range(15):
            board = chess.Board()
            for _ply in range(35):
                if board.is_game_over():
                    break
                indices = get_legal_action_indices(board)
                legals = list(board.legal_moves)
                self.assertEqual(len(indices), len(legals))
                for idx in indices:
                    self.assertIn(action_to_move(board, idx), legals)
                board.push(rng.choice(legals))


if __name__ == "__main__":
    unittest.main()
