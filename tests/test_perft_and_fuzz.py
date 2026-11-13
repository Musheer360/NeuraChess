"""C++23 bitboard perft verification, 2,000-position movegen/Zobrist fuzzer, and compiler flag tests."""

from __future__ import annotations

import ctypes
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import chess

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.constants import move_to_action
from engine.fast_tree import (
    c_perft,
    ensure_compiled,
    get_lib,
    pack_move,
    unpack_move,
)
import engine.fast_tree as fast_tree_mod


CANONICAL_CPW_PERFT = [
    ("Startpos", chess.STARTING_FEN, 5, 4_865_609),
    (
        "Kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        4,
        4_085_603,
    ),
    ("Position 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674_624),
    (
        "Position 4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        4,
        422_333,
    ),
    ("Position 5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2_103_487),
    (
        "Position 6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        4,
        3_894_594,
    ),
]


def _board_bbs(board: chess.Board):
    return (ctypes.c_uint64 * 8)(
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
    )


class TestCompilerFlags(unittest.TestCase):
    def test_no_finite_math_only_and_march_env(self) -> None:
        captured_cmds: list[list[str]] = []

        def _fake_check_call(cmd, *args, **kwargs):
            captured_cmds.append(list(cmd))
            return 0

        with mock.patch.dict(os.environ, {"NEURACHESS_ARCH": "x86-64-v3"}):
            with mock.patch.object(fast_tree_mod.subprocess, "check_call", side_effect=_fake_check_call):
                fast_tree_mod.ensure_compiled(force=True)

        self.assertEqual(len(captured_cmds), 1)
        cmd = captured_cmds[0]
        self.assertIn("-ffast-math", cmd)
        self.assertIn("-fno-finite-math-only", cmd)
        self.assertIn("-mcpu=x86_64_v3", cmd)

    def test_stamp_invalidation_and_x86_64_v3_build_no_avx512(self) -> None:
        # 1) Switching NEURACHESS_ARCH produces distinct flag-keyed DLL & stamp paths
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEURACHESS_ARCH", None)
            native_dll = ensure_compiled(force=False)
            native_stamp = Path(f"{native_dll}.stamp")
            self.assertTrue(Path(native_dll).exists())
            self.assertTrue(native_stamp.exists())

        with mock.patch.dict(os.environ, {"NEURACHESS_ARCH": "x86-64-v3"}):
            v3_dll = ensure_compiled(force=False)
            v3_stamp = Path(f"{v3_dll}.stamp")
            self.assertNotEqual(os.path.normcase(v3_dll), os.path.normcase(native_dll))
            self.assertTrue(Path(v3_dll).exists())
            stamp_meta = json.loads(v3_stamp.read_text(encoding="utf-8"))
            self.assertEqual(stamp_meta["arch_env"], "x86-64-v3")
            self.assertIn("-mcpu=x86_64_v3", stamp_meta["flags"])
            self.assertIn("-fno-finite-math-only", stamp_meta["flags"])

            # Load and execute the built x86-64-v3 library
            v3_lib = get_lib()
            self.assertEqual(c_perft(chess.STARTING_FEN, 2), 400)
            self.assertEqual(int(v3_lib.ft_test_math_guards_active(float("nan"), float("inf"))), 1 | 8)

            # Mutating the stamp forces a rebuild even when DLL mtime is newer than sources
            saved_stamp = v3_stamp.read_text(encoding="utf-8")
            try:
                v3_stamp.write_text('{"arch_env":"stale","flags":[]}', encoding="utf-8")
                with mock.patch.object(
                    fast_tree_mod.subprocess, "check_call", return_value=0
                ) as rebuild_spy:
                    ensure_compiled(force=False)
                self.assertEqual(rebuild_spy.call_count, 1)
            finally:
                v3_stamp.write_text(saved_stamp, encoding="utf-8")

        # 2) Confirm Zig x86_64_v3 codegen emits zero AVX-512 registers/opmasks/EVEX prefixes
        cpp_src = Path(fast_tree_mod.__file__).resolve().parent / "fast_tree.cpp"
        with tempfile.TemporaryDirectory() as tmpdir:
            asm_out = Path(tmpdir) / "fast_tree_v3.s"
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "ziglang",
                    "c++",
                    "-std=c++23",
                    "-O3",
                    "-ffast-math",
                    "-fno-finite-math-only",
                    "-mcpu=x86_64_v3",
                    "-S",
                    "-o",
                    str(asm_out),
                    str(cpp_src),
                ]
            )
            asm_text = asm_out.read_text(encoding="utf-8", errors="ignore")
        self.assertIsNone(
            re.search(r"\bzmm\d+|\{k[0-7]\}|\bevex\b", asm_text, re.IGNORECASE),
            "Unexpected AVX-512 instruction/register found in x86-64-v3 assembly!",
        )

    def test_runtime_isnan_isinf_guards_active(self) -> None:
        lib = get_lib()
        mask1 = int(lib.ft_test_math_guards_active(float("nan"), float("inf")))
        mask2 = int(lib.ft_test_math_guards_active(float("-inf"), float("nan")))
        self.assertEqual(mask1, 1 | 8)
        self.assertEqual(mask2, 2 | 4)
        self.assertTrue(math.isnan(float("nan")))


def _py_perft(board: chess.Board, depth: int) -> int:
    if depth <= 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    total = 0
    for mv in board.legal_moves:
        board.push(mv)
        total += _py_perft(board, depth - 1)
        board.pop()
    return total


class TestCPWPerftSuite(unittest.TestCase):
    def test_six_canonical_cpw_perft_positions(self) -> None:
        ensure_compiled(force=False)
        for name, fen, depth, expected in CANONICAL_CPW_PERFT:
            with self.subTest(position=name, depth=depth):
                got = c_perft(fen, depth)
                self.assertEqual(got, expected, f"{name} depth {depth}: got {got}, expected {expected}")

    def test_differential_perft_depth2_and_3_vs_python_chess(self) -> None:
        ensure_compiled(force=False)
        rng = random.Random(20261113)
        seed_fens = [fen for _, fen, _, _ in CANONICAL_CPW_PERFT] + [
            "8/8/8/8/k2Pp2Q/8/8/4K3 b - d3 0 1",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            "r3k2r/8/8/8/8/8/1b6/R3K2R w KQkq - 0 1",
            "rnb1kbnr/pppp1ppp/8/8/4q2N/8/PPPP1P1P/RNBQK2R w KQkq - 0 1",
        ]
        samples: list[chess.Board] = []
        while len(samples) < 25:
            b = chess.Board(seed_fens[len(samples) % len(seed_fens)])
            for _ in range(rng.randint(0, 6)):
                if b.is_game_over():
                    break
                b.push(rng.choice(list(b.legal_moves)))
            samples.append(b)

        for idx, b in enumerate(samples):
            depth = 3 if idx < 8 else 2
            with self.subTest(sample=idx, depth=depth, fen=b.fen()):
                self.assertEqual(c_perft(b, depth), _py_perft(b.copy(stack=False), depth))


class TestFuzzerMovegenAndZobrist(unittest.TestCase):
    def test_2000_positions_movegen_and_zobrist_parity(self) -> None:
        lib = get_lib()

        special_fens = [
            chess.STARTING_FEN,
            # En passant horizontal discovered check & valid EP
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "8/8/8/8/k2Pp2Q/8/8/4K3 b - d3 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            # Promotions & underpromotions & captures
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
            # Castling with attacked corner squares (rook can pass through attacked b1/b8)
            "r3k2r/8/8/8/8/8/1b6/R3K2R w KQkq - 0 1",
            # Double check evasions (and 1-ply discovered double check triggers)
            "4r1k1/8/8/4N3/7b/8/8/3QK3 w - - 0 1",
            "4k3/8/8/1B6/8/2n5/8/4R1K1 b - - 0 1",
            "4k3/8/5N2/8/8/8/8/4R1K1 b - - 0 1",
            "4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1",
            # Kiwipete & middlegames
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ]

        out_bbs = (ctypes.c_uint64 * 8)()
        out_cr = ctypes.c_uint64(0)
        out_ep = ctypes.c_int32(-1)
        out_turn = ctypes.c_int32(0)
        inc_hi = ctypes.c_uint64(0)
        inc_lo = ctypes.c_uint64(0)
        scr_hi = ctypes.c_uint64(0)
        scr_lo = ctypes.c_uint64(0)
        inc_moves = (ctypes.c_uint16 * 256)()
        scr_acts = (ctypes.c_uint16 * 256)()
        scr_moves = (ctypes.c_uint16 * 256)()
        scr_bonuses = (ctypes.c_double * 256)()

        rng = random.Random(20261113)
        checked = 0
        seed_idx = 0
        coverage = {
            "promotions": 0,
            "en_passant": 0,
            "castling": 0,
            "double_checks": 0,
            "played_promotions": 0,
            "played_en_passant": 0,
            "played_castling": 0,
        }

        def _verify_position(start_board: chess.Board, move_seq: list[chess.Move], cur_board: chess.Board) -> None:
            if len(cur_board.checkers()) >= 2:
                coverage["double_checks"] += 1
            for lm in cur_board.legal_moves:
                if lm.promotion is not None:
                    coverage["promotions"] += 1
                if cur_board.is_en_passant(lm):
                    coverage["en_passant"] += 1
                if cur_board.is_castling(lm):
                    coverage["castling"] += 1
            s_bbs = _board_bbs(start_board)
            s_ep = -1 if start_board.ep_square is None else int(start_board.ep_square)
            s_turn = 1 if start_board.turn == chess.WHITE else 0
            seq_len = len(move_seq)
            seq_arr = (ctypes.c_uint16 * max(1, seq_len))(*[pack_move(m) for m in move_seq]) if seq_len > 0 else None

            n_inc = int(
                lib.ft_test_push_chain_native(
                    s_bbs,
                    ctypes.c_uint64(start_board.castling_rights),
                    s_ep,
                    s_turn,
                    seq_arr,
                    seq_len,
                    out_bbs,
                    ctypes.byref(out_cr),
                    ctypes.byref(out_ep),
                    ctypes.byref(out_turn),
                    ctypes.byref(inc_hi),
                    ctypes.byref(inc_lo),
                    inc_moves,
                )
            )

            c_bbs = _board_bbs(cur_board)
            c_ep = -1 if cur_board.ep_square is None else int(cur_board.ep_square)
            c_turn = 1 if cur_board.turn == chess.WHITE else 0

            # 1) Incremental push-chain bitboards & state vs python-chess
            for idx in range(8):
                self.assertEqual(int(out_bbs[idx]), int(c_bbs[idx]), f"bb[{idx}] mismatch at {cur_board.fen()}")
            self.assertEqual(int(out_cr.value), int(cur_board.clean_castling_rights()))
            self.assertEqual(int(out_turn.value), c_turn)

            # 2) Incremental CBoard::push Zobrist vs fresh scratch CBoard Zobrist
            lib.ft_compute_cboard_zobrist(
                c_bbs,
                ctypes.c_uint64(cur_board.castling_rights),
                c_ep,
                c_turn,
                ctypes.byref(scr_hi),
                ctypes.byref(scr_lo),
            )
            self.assertEqual(
                (int(inc_hi.value), int(inc_lo.value)),
                (int(scr_hi.value), int(scr_lo.value)),
                f"Zobrist mismatch incremental vs scratch at {cur_board.fen()}",
            )

            # 3) Legal move-set parity (both push-chain and scratch movegen vs python-chess)
            expected_ucis = {m.uci() for m in cur_board.legal_moves}
            inc_ucis = {unpack_move(int(inc_moves[i])).uci() for i in range(n_inc)}
            self.assertEqual(n_inc, len(expected_ucis), f"Move count mismatch at {cur_board.fen()}")
            self.assertEqual(inc_ucis, expected_ucis, f"Push-chain movegen mismatch at {cur_board.fen()}")

            n_scr = int(
                lib.ft_test_generate_moves_native(
                    c_bbs,
                    ctypes.c_uint64(cur_board.castling_rights),
                    c_ep,
                    c_turn,
                    scr_acts,
                    scr_moves,
                    scr_bonuses,
                )
            )
            scr_ucis = {unpack_move(int(scr_moves[i])).uci() for i in range(n_scr)}
            self.assertEqual(n_scr, len(expected_ucis))
            self.assertEqual(scr_ucis, expected_ucis, f"Scratch movegen mismatch at {cur_board.fen()}")

            cflip = cur_board.turn == chess.BLACK
            for i in range(n_scr):
                mv = unpack_move(int(scr_moves[i]))
                self.assertEqual(int(scr_acts[i]), move_to_action(mv, canonical_flip=cflip))

        # Check all curated special positions directly and verify every legal reply
        for fen in special_fens:
            b = chess.Board(fen)
            _verify_position(b.copy(stack=False), [], b)
            checked += 1
            for mv in b.legal_moves:
                if checked >= 2000:
                    break
                b2 = b.copy(stack=False)
                b2.push(mv)
                _verify_position(b.copy(stack=False), [mv], b2)
                checked += 1

        # Fill remaining checks up to 2,000 positions with random rollouts favoring captures/promotions/checks
        while checked < 2000:
            start_fen = special_fens[seed_idx % len(special_fens)]
            seed_idx += 1
            start_b = chess.Board(start_fen)
            cur_b = chess.Board(start_fen)
            seq: list[chess.Move] = []
            max_plies = 48
            for _ in range(max_plies):
                if cur_b.is_game_over() or checked >= 2000:
                    break
                legals = list(cur_b.legal_moves)
                tactical = [m for m in legals if m.promotion is not None or cur_b.is_en_passant(m) or cur_b.is_castling(m) or cur_b.gives_check(m)]
                mv = rng.choice(tactical) if (tactical and rng.random() < 0.45) else rng.choice(legals)
                if mv.promotion is not None:
                    coverage["played_promotions"] += 1
                if cur_b.is_en_passant(mv):
                    coverage["played_en_passant"] += 1
                if cur_b.is_castling(mv):
                    coverage["played_castling"] += 1
                cur_b.push(mv)
                seq.append(mv)
                _verify_position(start_b, seq, cur_b)
                checked += 1

        self.assertGreaterEqual(checked, 2000)
        self.assertGreaterEqual(coverage["promotions"], 20, f"Low promotion coverage: {coverage}")
        self.assertGreaterEqual(coverage["en_passant"], 5, f"Low EP coverage: {coverage}")
        self.assertGreaterEqual(coverage["castling"], 20, f"Low castling coverage: {coverage}")
        self.assertGreaterEqual(coverage["double_checks"], 4, f"Low double-check coverage: {coverage}")

    def test_en_passant_zobrist_convention(self) -> None:
        """
        Documented convention: CBoard hashes `ep_square` into `zobrist_hi`/`zobrist_lo`
        ONLY when at least one legal en-passant capture exists (Polyglot convention),
        NOT unconditionally after every double pawn push.
        """
        lib = get_lib()

        def _scratch_hash(board: chess.Board) -> tuple[int, int]:
            bbs = _board_bbs(board)
            ep = -1 if board.ep_square is None else int(board.ep_square)
            turn = 1 if board.turn == chess.WHITE else 0
            hi, lo = ctypes.c_uint64(0), ctypes.c_uint64(0)
            lib.ft_compute_cboard_zobrist(
                bbs, ctypes.c_uint64(board.castling_rights), ep, turn, ctypes.byref(hi), ctypes.byref(lo)
            )
            return int(hi.value), int(lo.value)

        def _push_one_hash(start_board: chess.Board, mv: chess.Move) -> tuple[int, int]:
            s_bbs = _board_bbs(start_board)
            s_ep = -1 if start_board.ep_square is None else int(start_board.ep_square)
            s_turn = 1 if start_board.turn == chess.WHITE else 0
            seq = (ctypes.c_uint16 * 1)(pack_move(mv))
            out_bbs = (ctypes.c_uint64 * 8)()
            out_cr = ctypes.c_uint64(0)
            out_ep = ctypes.c_int32(-1)
            out_turn = ctypes.c_int32(0)
            hi, lo = ctypes.c_uint64(0), ctypes.c_uint64(0)
            moves = (ctypes.c_uint16 * 256)()
            lib.ft_test_push_chain_native(
                s_bbs,
                ctypes.c_uint64(start_board.castling_rights),
                s_ep,
                s_turn,
                seq,
                1,
                out_bbs,
                ctypes.byref(out_cr),
                ctypes.byref(out_ep),
                ctypes.byref(out_turn),
                ctypes.byref(hi),
                ctypes.byref(lo),
                moves,
            )
            return int(hi.value), int(lo.value)

        # Case 1: Double push with NO adjacent enemy pawn -> EP square NOT hashed
        b1 = chess.Board()
        mv1 = chess.Move.from_uci("e2e4")
        b1_after = b1.copy(stack=False)
        b1_after.push(mv1)
        b1_no_ep = b1_after.copy(stack=False)
        b1_no_ep.ep_square = None
        self.assertFalse(b1_after.has_legal_en_passant())
        self.assertEqual(_push_one_hash(b1, mv1), _scratch_hash(b1_after))
        self.assertEqual(_scratch_hash(b1_after), _scratch_hash(b1_no_ep))

        # Case 2: Double push with pinned adjacent pawn (pseudo-legal EP is ILLEGAL) -> EP NOT hashed
        b2 = chess.Board("8/2p5/8/KP5r/8/8/8/4k3 b - - 0 1")
        mv2 = chess.Move.from_uci("c7c5")
        b2_after = b2.copy(stack=False)
        b2_after.push(mv2)
        b2_no_ep = b2_after.copy(stack=False)
        b2_no_ep.ep_square = None
        self.assertTrue(b2_after.has_pseudo_legal_en_passant())
        self.assertFalse(b2_after.has_legal_en_passant())
        self.assertEqual(_push_one_hash(b2, mv2), _scratch_hash(b2_after))
        self.assertEqual(_scratch_hash(b2_after), _scratch_hash(b2_no_ep))

        # Case 3: Double push WITH a legal EP capture -> EP square IS hashed
        b3 = chess.Board("rnbqkbnr/ppp1pppp/8/8/3p4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        mv3 = chess.Move.from_uci("e2e4")
        b3_after = b3.copy(stack=False)
        b3_after.push(mv3)
        b3_no_ep = b3_after.copy(stack=False)
        b3_no_ep.ep_square = None
        self.assertTrue(b3_after.has_legal_en_passant())
        self.assertEqual(_push_one_hash(b3, mv3), _scratch_hash(b3_after))
        self.assertNotEqual(_scratch_hash(b3_after), _scratch_hash(b3_no_ep))


if __name__ == "__main__":
    unittest.main()
