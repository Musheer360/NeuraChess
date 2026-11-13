"""Unit tests for canonical UCI <-> AlphaZero 4,672 action roundtrip conversion."""

import random
import sys
import unittest
from pathlib import Path

import chess

from neurachess.core.constants import TOTAL_ACTIONS, action_to_uci, move_to_action, uci_to_action
from neurachess.core.tensor_movegen import action_to_move


class TestMoveToAction(unittest.TestCase):
    def test_random_games_roundtrip(self) -> None:
        rng = random.Random(1234)
        checked = 0
        for _ in range(10):
            board = chess.Board()
            for _ply in range(60):
                if board.is_game_over():
                    break
                is_black = board.turn == chess.BLACK
                legals = list(board.legal_moves)
                for mv in legals:
                    uci = mv.uci()
                    act = uci_to_action(uci, canonical_flip=is_black)
                    self.assertGreaterEqual(act, 0)
                    self.assertLess(act, TOTAL_ACTIONS)
                    self.assertEqual(move_to_action(mv, canonical_flip=is_black), act)
                    # Disambiguating queen promotion vs non-pawn rank-8 slide requires board context
                    self.assertEqual(action_to_move(board, act), mv)
                    piece = board.piece_at(mv.from_square)
                    if piece is not None and piece.piece_type == chess.PAWN:
                        self.assertEqual(action_to_uci(act, canonical_flip=is_black), uci)
                    else:
                        self.assertEqual(action_to_uci(act, canonical_flip=is_black)[:4], uci[:4])
                    checked += 1
                board.push(rng.choice(legals))
        self.assertGreater(checked, 1000)

    def test_queen_and_underpromotion_actions(self) -> None:
        # White promotion a7a8q/n/b/r
        for promo in ("q", "n", "b", "r"):
            uci = f"a7a8{promo}"
            act = uci_to_action(uci, canonical_flip=False)
            self.assertEqual(action_to_uci(act, canonical_flip=False), uci)

        # Black promotion a2a1q/n/b/r with canonical vertical flip
        for promo in ("q", "n", "b", "r"):
            uci = f"a2a1{promo}"
            act = uci_to_action(uci, canonical_flip=True)
            self.assertEqual(action_to_uci(act, canonical_flip=True), uci)


if __name__ == "__main__":
    unittest.main()
