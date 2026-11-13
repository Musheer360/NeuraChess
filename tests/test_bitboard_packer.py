"""Unit tests for 107-byte v1 and 691-byte v2 bitboard dataset packing/unpacking."""

import random
import struct
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
from core.tensor_movegen import get_legal_action_mask
from training.bitboard_dataset import (
    PACK_FORMAT,
    RECORD_SIZE,
    RECORD_SIZE_V1,
    RECORD_SIZE_V2,
    fast_tensor_from_unpacked,
    pack_position,
    pack_position_v2,
    unpack_position,
    unpack_position_v2,
)


class TestBitboardPacker(unittest.TestCase):
    def _verify_board_roundtrip(self, board: chess.Board, action_idx: int, eval_cp: int, wdl: tuple[float, float, float]) -> None:
        raw_v1 = pack_position(board, action_index=action_idx, eval_cp=eval_cp, wdl=wdl)
        self.assertEqual(len(raw_v1), RECORD_SIZE_V1)
        rec1 = unpack_position(raw_v1)

        rb: chess.Board = rec1["board"]
        self.assertEqual(rb.board_fen(), board.board_fen())
        self.assertEqual(rb.turn, board.turn)
        self.assertEqual(rb.castling_rights, board.castling_rights)
        self.assertEqual(rb.ep_square, board.ep_square)
        self.assertEqual(rb.halfmove_clock, min(255, board.halfmove_clock))
        self.assertEqual(rec1["action_index"], action_idx)
        self.assertEqual(rec1["eval_cp"], eval_cp)
        for p, q in zip(rec1["wdl"], wdl):
            self.assertAlmostEqual(p, q, delta=1.0 / 255.0 + 1e-6)

        # Verify 691-byte v2 roundtrip including 4672-bit legal action mask
        legal_mask = get_legal_action_mask(board)
        raw_v2 = pack_position_v2(board, action_index=action_idx, eval_cp=eval_cp, wdl=wdl, legal_mask=legal_mask)
        self.assertEqual(len(raw_v2), RECORD_SIZE_V2)
        rec2 = unpack_position_v2(raw_v2)
        np.testing.assert_array_equal(rec2["legal_mask"][:4672], legal_mask)

        # Verify fast bitboard tensor builder matches board_to_tensor(layout="legacy_flip")
        unp = struct.unpack(PACK_FORMAT, raw_v1)
        t_fast = fast_tensor_from_unpacked(unp[:12], unp[12], unp[13], unp[14], unp[15])
        t_ref = board_to_tensor(board, layout="legacy_flip")
        self.assertEqual(t_fast.shape, (18, 8, 8))
        np.testing.assert_allclose(t_fast, t_ref, atol=1e-6)

    def test_record_constants(self) -> None:
        self.assertEqual(RECORD_SIZE, 107)
        self.assertEqual(RECORD_SIZE_V1, 107)
        self.assertEqual(RECORD_SIZE_V2, 691)

    def test_curated_and_random_positions_roundtrip(self) -> None:
        curated_fens = [
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "8/2P5/8/8/8/8/3p4/4K2k w - - 12 40",
        ]
        for fen in curated_fens:
            b = chess.Board(fen)
            mv = next(iter(b.legal_moves))
            act = uci_to_action(mv.uci(), canonical_flip=(b.turn == chess.BLACK))
            self._verify_board_roundtrip(b, act, eval_cp=135, wdl=(0.7, 0.2, 0.1))

        rng = random.Random(99)
        tested = 0
        while tested < 250:
            b = chess.Board()
            for _ in range(40):
                if b.is_game_over() or tested >= 250:
                    break
                legals = list(b.legal_moves)
                mv = rng.choice(legals)
                act = uci_to_action(mv.uci(), canonical_flip=(b.turn == chess.BLACK))
                cp = rng.randint(-1500, 1500)
                self._verify_board_roundtrip(b, act, eval_cp=cp, wdl=(0.25, 0.50, 0.25))
                b.push(mv)
                tested += 1


if __name__ == "__main__":
    unittest.main()
