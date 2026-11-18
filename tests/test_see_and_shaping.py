"""Unit & Tactical Exchange Tests for C++23 Bitboard SEE, FPU Shaping & Dynamic Batch Gear-Shifting."""

from __future__ import annotations

import unittest

import chess
import numpy as np

from neurachess.core.constants import uci_to_action
from neurachess.engine.fast_tree import bench_see_ns, c_see, ensure_compiled
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch


# 56 hand-verified Static Exchange Evaluation test cases (centipawns from mover view)
# Piece values: P=100, N=320, B=330, R=500, Q=900, K=20000
SEE_CASES: list[tuple[str, str, str, int]] = [
    # --- 1..12: Basic captures & exchanges ---
    ("undefended_pxp", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", 100),
    ("undefended_rxq", "4k3/8/3q4/8/8/8/8/3RK3 w - - 0 1", "d1d6", 900),
    ("defended_pxn_by_p", "4k3/2p5/3n4/4P3/8/8/8/4K3 w - - 0 1", "e5d6", 220),
    ("defended_nxp_by_p", "4k3/2p5/3p4/8/4N3/8/8/4K3 w - - 0 1", "e4d6", -220),
    ("defended_qxp_by_n", "4k3/1n6/3p4/8/8/8/3Q4/4K3 w - - 0 1", "d2d6", -800),
    ("defended_rxb_by_p", "4k3/2p5/3b4/8/8/8/3R4/4K3 w - - 0 1", "d2d6", -170),
    ("defended_bxn_by_p", "4k3/2p5/3n4/8/1B6/8/8/4K3 w - - 0 1", "b4d6", -10),
    ("defended_nxb_by_p", "4k3/2p5/3b4/8/4N3/8/8/4K3 w - - 0 1", "e4d6", 10),
    ("defended_nxr_by_p", "4k3/2p5/3r4/8/4N3/8/8/4K3 w - - 0 1", "e4d6", 180),
    ("pxp_with_defenders", "4k3/2p5/3p4/4P3/8/3R4/8/4K3 w - - 0 1", "e5d6", 100),
    ("rxp_standpat_trap", "4k3/2b5/3p4/4P3/8/3R4/8/4K3 w - - 0 1", "d3d6", -70),
    ("black_capture_nxr", "4k3/2n5/8/3R4/2P5/8/8/4K3 b - - 0 1", "c7d5", 180),

    # --- 13..20: Multi-attacker pileups & non-recapture stand-pat ---
    ("pileup_standpat_keep_knight", "4k3/3r4/2b5/3n4/4P3/2N5/3R4/4K3 w - - 0 1", "e4d5", 320),
    ("pileup_rxr_equal_trade", "4k3/3r4/3r4/8/3R4/8/8/4K3 w - - 0 1", "d4d6", 0),
    ("pileup_bxn_winning_net", "4k3/2p5/3n4/3r4/4P3/2B5/3R4/4K3 w - - 0 1", "c3d4", 0),
    ("symmetric_e4e5_d5", "r1bqkbnr/ppp2ppp/2n5/3pp3/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 4", "d4e5", 100),
    ("symmetric_d4e5_dxe", "r1bqkbnr/ppp2ppp/2n5/3pp3/4P3/3P1N2/PPP2PPP/RNBQKB1R b KQkq - 0 3", "d5e4", 0),
    ("qx_minor_defended_twice_backed", "4k3/3q4/3n4/2p5/8/8/3Q4/3RK2R w - - 0 1", "d2d6", 320),
    ("qx_minor_defended_pawn_queen_loses", "4k3/2pq4/3n4/8/8/8/3Q4/4K3 w - - 0 1", "d2d6", -580),
    ("minor_trade_with_rook_def", "4k3/3r4/3b4/8/4N3/3R4/8/4K3 w - - 0 1", "e4d6", 330),

    # --- 21..30: X-ray batteries & self-unblocking X-rays ---
    ("xray_rook_behind_rook_wins", "4k3/r7/p7/8/R7/8/8/R3K3 w - - 0 1", "a4a6", 100),
    ("no_xray_rook_loses", "4k3/r7/p7/8/R7/8/8/4K3 w - - 0 1", "a4a6", -400),
    ("xray_queen_behind_bishop", "4k3/4p1b1/5p2/8/3B4/8/1Q6/4K3 w - - 0 1", "d4f6", -230),
    ("xray_bishop_behind_queen_blocks_king", "6k1/5n2/8/3Q4/8/1B6/8/4K3 w - - 0 1", "d5f7", 320),
    ("no_xray_bishop_king_takes_queen", "6k1/5n2/8/3Q4/8/8/8/4K3 w - - 0 1", "d5f7", -580),
    ("xray_pawn_uncovers_bishop", "4k3/2p5/4p3/3p4/2P5/1B6/8/4K3 w - - 0 1", "c4d5", 100),
    ("xray_uncovers_enemy_queen_diag", "4k3/q7/8/8/3Q4/8/5r2/2K5 w - - 0 1", "d4f2", -400),
    ("xray_king_defends_f2_counter_battery", "4k3/q7/8/8/3Q4/8/5r2/4K3 w - - 0 1", "d4f2", 500),
    ("xray_uncovers_enemy_rook_file", "4k3/8/8/4Q3/8/8/4r3/4K3 b - - 0 1", "e2e5", 900),
    ("xray_uncovers_enemy_battery_loss", "q3k3/8/8/4R3/8/8/4r3/4K3 b - - 0 1", "e2e5", 500),

    # --- 31..40: Pinned defenders & pinned attackers ---
    ("pinned_bishop_cannot_defend_knight", "4k3/4b3/5n2/7N/8/8/8/4R1K1 w - - 0 1", "h5f6", 320),
    ("unpinned_bishop_defends_knight", "4k3/4b3/5n2/7N/8/8/8/6K1 w - - 0 1", "h5f6", 0),
    ("pinned_bishop_captures_along_pin_ray", "4k3/3b4/2n5/8/3N4/1B6/8/4K3 w - - 0 1", "d4c6", 0),
    ("pinned_pawn_cannot_capture_off_file", "4k3/4p3/3n4/8/4N3/8/8/4R1K1 w - - 0 1", "e4d6", 320),
    ("pinned_queen_cannot_recapture_off_ray", "4k3/4q3/2r5/3B4/8/8/8/4R1K1 w - - 0 1", "d5c6", 500),
    ("pinned_rook_along_rank_can_capture", "4k3/8/8/q1rR2K1/8/8/8/8 b - - 0 1", "c5d5", 500),
    ("white_pinned_knight_cant_protect", "4r1k1/8/8/b7/4N3/2P5/8/4K3 b - - 0 1", "a5c3", 100),
    ("king_blocked_by_pinned_bishop", "4k1r1/3p4/8/8/6B1/8/8/3R2K1 w - - 0 1", "d1d7", 100),
    ("king_not_blocked_when_no_bishop", "4k2r/3p4/8/8/8/8/8/3R2K1 w - - 0 1", "d1d7", -400),
    ("king_blocked_by_xray_rook_behind_queen", "8/8/5k2/4p3/8/4Q3/8/4R1K1 w - - 0 1", "e3e5", 100),

    # --- 41..48: Promotions & Underpromotions ---
    ("quiet_promo_queen_undefended", "7k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q", 800),
    ("quiet_promo_knight_undefended", "7k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8n", 220),
    ("quiet_promo_queen_defended_by_rook", "r6k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q", -100),
    ("capture_promo_q_defended_by_rook", "r2r3k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8q", 400),
    ("capture_promo_r_defended_by_rook", "r2r3k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8r", 400),
    ("capture_promo_q_defended_with_rook_back", "r2r3k/4P3/8/8/8/8/8/3RK3 w - - 0 1", "e7d8q", 900),
    ("capture_promo_n_undefended", "3n3k/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7d8n", 540),
    ("black_capture_promo_king_recap", "4k3/8/8/8/8/8/4p3/3RK3 b - - 0 1", "e2d1q", 400),

    # --- 49..56: En Passant, Undefended Promo, Quiet & Castling ---
    ("ep_undefended", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 100),
    ("ep_defended_by_pawn", "4k3/2p5/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 0),
    ("ep_uncovers_black_rook_on_d_file", "4k3/8/8/3pP3/8/8/3r4/4K3 w - d6 0 1", "e5d6", 0),
    ("ep_uncovers_black_rook_stopped_by_queen", "4k3/8/8/3pP3/5Q2/8/3r4/4K3 w - d6 0 1", "e5d6", 100),
    ("ep_uncovers_white_bishop_battery", "4k3/2p5/8/3pP3/8/6B1/8/4K3 w - d6 0 1", "e5d6", 100),
    ("black_capture_promo_undefended", "4k3/8/8/8/8/8/4p3/3R3K b - - 0 1", "e2d1q", 1300),
    ("quiet_hang_queen_to_pawn", "4k3/8/6p1/8/8/8/8/3QK3 w - - 0 1", "d1h5", -900),
    ("castling_kingside_see_zero", "4k3/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", 0),
]


class _FixedDummyModel:
    """Deterministic stub neural net with configurable zero outputs."""

    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    def infer(self, x: np.ndarray):
        b = x.shape[0]
        return np.zeros((b, 4672), dtype=np.float32), np.zeros((b, 1), dtype=np.float32)


class TestBitboardSEE(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_compiled(force=False)

    def test_case_count_at_least_50(self) -> None:
        self.assertGreaterEqual(len(SEE_CASES), 50)

    def test_56_tactical_see_positions(self) -> None:
        failures: list[str] = []
        for name, fen, move_uci, expected_see in SEE_CASES:
            board = chess.Board(fen)
            mv = board.parse_uci(move_uci)
            self.assertIn(mv, board.legal_moves, msg=f"Case {name} move {move_uci} not legal in {fen}")
            actual = c_see(board, mv)
            if actual != expected_see:
                failures.append(f"{name} ({move_uci} on {fen}): expected {expected_see}, got {actual}")
        self.assertEqual(failures, [], msg="SEE mismatches:\n" + "\n".join(failures))

    def test_king_capture_blocked_by_pinned_defender(self) -> None:
        # 1) White Q captures Black Pawn c6; Ba4 is pinned to Ka1 by Ra8 along a-file,
        #    so Ba4 still defends c6 against Black King d7 -> Queen wins +100 cleanly.
        fen_pinned = "r7/3k4/2p5/8/B7/8/2Q5/K7 w - - 0 1"
        self.assertEqual(c_see(fen_pinned, "c2c6"), 100)

        # 2) Without Ba4 (and moving Ra8 off a-file so Ka1 isn't in check), Kd7xc6 is legal -> -800.
        fen_unprotected = "1r6/3k4/2p5/8/8/8/2Q5/K7 w - - 0 1"
        self.assertEqual(c_see(fen_unprotected, "c2c6"), -800)

    def test_see_speed_under_2_microseconds(self) -> None:
        fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        ns_per_call = bench_see_ns(fen, "e5f7", iters=100_000)
        self.assertLess(ns_per_call, 2000.0, msg=f"SEE too slow: {ns_per_call:.1f} ns")


class TestSEEAndGearShiftSearchFlags(unittest.TestCase):
    def test_new_flags_default_off(self) -> None:
        s = NPUBatchedPUCTSearch(root_model=_FixedDummyModel(1), batch_model=_FixedDummyModel(16), batch_size=16)
        self.assertFalse(s.use_see_prior_shaping)
        self.assertFalse(s.use_dynamic_batch_shift)

    def test_see_fpu_bias_redirects_hanging_capture_and_boosts_winning_capture(self) -> None:
        # Queen takes Pawn d5 defended by c6 pawn (SEE = -800 cp) -> -0.15 FPU penalty when prior < 0.15
        # vs Knight takes undefended Pawn e5 (SEE = +100 cp) -> +0.05 FPU boost
        fen = "4k3/8/2p5/3pp3/8/5N2/3Q4/4K3 w - - 0 1"
        b = chess.Board(fen)

        class RootPriorStub:
            def __init__(self, high_act: int | None = None, high_logit: float = 0.0):
                self.batch_size = 1
                self.high_act = high_act
                self.high_logit = high_logit

            def infer(self, x):
                logits = np.zeros((1, 4672), dtype=np.float32)
                if self.high_act is not None:
                    logits[0, self.high_act] = self.high_logit
                return logits, np.zeros((1, 1), dtype=np.float32)

        class LeafValueStub:
            def __init__(self, good_fen_epd_prefix: str):
                self.batch_size = 16
                self.good_prefix = good_fen_epd_prefix

            def infer(self, x):
                logits = np.zeros((x.shape[0], 4672), dtype=np.float32)
                vals = np.zeros((x.shape[0], 1), dtype=np.float32)
                return logits, vals

        act_qxd5 = uci_to_action("d2d5", canonical_flip=False)
        act_nxe5 = uci_to_action("f3e5", canonical_flip=False)

        # With uniform priors (< 0.15) and SEE shaping ON:
        # - On batch_size=1, first selected leaf MUST be f3e5 (only +0.05 FPU boost)
        # - At 16 nodes (batch 1), d2d5 (-0.15 FPU penalty) MUST remain unvisited (0 visits)
        s_first = NPUBatchedPUCTSearch(
            root_model=RootPriorStub(),
            batch_model=_FixedDummyModel(1),
            batch_size=1,
            use_see_prior_shaping=True,
            use_fast_tree=True,
        )
        # 1 root eval + 1 leaf batch (max_nodes=2): first selected leaf must be f3e5 (+0.05 FPU boost)
        _, first_uci, _, _, _, _ = s_first.search(b, movetime_ms=200, max_nodes=2)
        self.assertEqual(first_uci, "f3e5")
        self.assertEqual(s_first.last_root_visits[act_nxe5], 1)

        s_on = NPUBatchedPUCTSearch(
            root_model=RootPriorStub(),
            batch_model=_FixedDummyModel(16),
            batch_size=16,
            use_see_prior_shaping=True,
            use_fast_tree=True,
        )
        s_on.search(b, movetime_ms=200, max_nodes=16)
        self.assertGreater(s_on.last_root_visits[act_nxe5], 0)
        self.assertEqual(s_on.last_root_visits[act_qxd5], 0)

        # When losing capture has raw prior >= 0.15, no FPU penalty is applied and it IS searched
        s_guard = NPUBatchedPUCTSearch(
            root_model=RootPriorStub(high_act=act_qxd5, high_logit=4.0),
            batch_model=_FixedDummyModel(16),
            batch_size=16,
            use_see_prior_shaping=True,
            use_fast_tree=True,
        )
        s_guard.search(b, movetime_ms=200, max_nodes=16)
        self.assertGreater(s_guard.last_root_visits[act_qxd5], 0)
        self.assertNotEqual(act_qxd5, act_nxe5)

    def test_dynamic_batch_gear_recommendation_and_loop(self) -> None:
        models = {4: _FixedDummyModel(4), 8: _FixedDummyModel(8), 16: _FixedDummyModel(16)}
        s = NPUBatchedPUCTSearch(
            root_model=_FixedDummyModel(1),
            batch_model=models[16],
            batch_size=16,
            use_dynamic_batch_shift=True,
            batch_models=models,
            use_fast_tree=True,
        )
        # Rule (b) Narrow Root: In check -> B=4 (reason 2) while root_visits < 32, then graduates to B=16 (>= 32)
        in_check_fen = "4k3/8/8/8/8/8/4r3/4K2R w - - 0 1"
        s.clear_state()
        _, uci_chk, _, nodes_chk, _, _ = s.search(chess.Board(in_check_fen), movetime_ms=200, max_nodes=32)
        self.assertTrue(uci_chk)
        self.assertGreater(nodes_chk, 0)
        self.assertEqual(s.last_profile.get("mode"), "dynamic_gear_shift")
        gears_chk = s.last_profile.get("batch_gear_counts", {})
        reasons_chk = s.last_profile.get("gear_reasons", {})
        self.assertEqual(list(gears_chk.keys()), ["4"])
        self.assertIn("2", reasons_chk)

        s.clear_state()
        s.search(chess.Board(in_check_fen), movetime_ms=300, max_nodes=64)
        self.assertIn("4", s.last_profile.get("batch_gear_counts", {}))
        self.assertIn("16", s.last_profile.get("batch_gear_counts", {}))
        self.assertIn("2", s.last_profile.get("gear_reasons", {}))

        # Rule (b) Narrow Root: <= 5 legal moves (not in check) -> B=4 (reason 2) while < 32 visits
        five_moves_board = chess.Board("7k/8/8/8/4p3/4P3/8/4K3 w - - 0 1")
        self.assertEqual(five_moves_board.legal_moves.count(), 5)
        self.assertFalse(five_moves_board.is_check())
        s.clear_state()
        s.search(five_moves_board, movetime_ms=200, max_nodes=32)
        self.assertEqual(list(s.last_profile.get("batch_gear_counts", {}).keys()), ["4"])
        self.assertIn("2", s.last_profile.get("gear_reasons", {}))

        # Async nireq=2 double-buffered B=16 inside gear_shift_v3
        class _AsyncReqStub:
            def __init__(self, bs: int):
                self.bs = bs
                self.results: dict = {}

            def start_async(self, feeds=None):
                b = self.bs if feeds is None else feeds["board_state"].shape[0]
                self.results = {
                    0: np.zeros((b, 4672), dtype=np.float32),
                    1: np.zeros((b, 1), dtype=np.float32),
                }

            def wait(self):
                return None

        class _AsyncBatchModelStub:
            def __init__(self, bs: int):
                self.batch_size = bs
                self.infer_request = _AsyncReqStub(bs)
                self.infer_request_async = _AsyncReqStub(bs)

            def infer(self, x):
                b = x.shape[0]
                return np.zeros((b, 4672), dtype=np.float32), np.zeros((b, 1), dtype=np.float32)

        async_b16 = _AsyncBatchModelStub(16)
        s_async = NPUBatchedPUCTSearch(
            root_model=_FixedDummyModel(1),
            batch_model=async_b16,
            batch_size=16,
            use_dynamic_batch_shift=True,
            batch_models={4: _FixedDummyModel(4), 8: _FixedDummyModel(8), 16: async_b16},
            use_fast_tree=True,
        )
        _, uci_as, _, nodes_as, _, _ = s_async.search(chess.Board(), movetime_ms=300, max_nodes=64)
        self.assertTrue(uci_as)
        self.assertTrue(s_async.last_profile.get("dynamic_async_b16"))
        self.assertGreaterEqual(s_async.last_profile.get("batch_gear_counts", {}).get("16", 0), 2)
        self.assertGreaterEqual(nodes_as, 48)

        # 6 legal moves (not in check) -> Cold-Start B=4 (<16 visits, reason 1), then B=16 (>=16 visits, reason 0)
        six_moves_board = chess.Board("7k/8/8/8/8/4P3/8/4K3 w - - 0 1")
        self.assertEqual(six_moves_board.legal_moves.count(), 6)
        self.assertFalse(six_moves_board.is_check())
        s.clear_state()
        s.search(six_moves_board, movetime_ms=200, max_nodes=32)
        gears_6 = s.last_profile.get("batch_gear_counts", {})
        reasons_6 = s.last_profile.get("gear_reasons", {})
        self.assertEqual(gears_6.get("4"), 4)
        self.assertIn("16", gears_6)
        self.assertEqual(reasons_6.get("1"), 4)
        self.assertIn("0", reasons_6)

        # Rule (a) Cold-Start Ramp + B=16 workhorse on broad startpos even with sharply skewed prior (no entropy trigger)
        skew_logits = np.full((1, 4672), -20.0, dtype=np.float32)
        skew_logits[0, uci_to_action("e2e4", canonical_flip=False)] = 20.0

        class _SkewDummyModel:
            def __init__(self, batch_size: int):
                self.batch_size = batch_size

            def infer(self, x):
                b = x.shape[0]
                return np.repeat(skew_logits, b, axis=0), np.zeros((b, 1), dtype=np.float32)

        s_skew = NPUBatchedPUCTSearch(
            root_model=_SkewDummyModel(1),
            batch_model=_SkewDummyModel(16),
            batch_size=16,
            use_dynamic_batch_shift=True,
            batch_models={4: _SkewDummyModel(4), 8: _SkewDummyModel(8), 16: _SkewDummyModel(16)},
            use_fast_tree=True,
        )
        _, uci_start, _, _, _, _ = s_skew.search(chess.Board(), movetime_ms=300, max_nodes=48)
        self.assertTrue(uci_start)
        gears_start = s_skew.last_profile.get("batch_gear_counts", {})
        reasons_start = s_skew.last_profile.get("gear_reasons", {})
        self.assertEqual(gears_start.get("4"), 4)
        self.assertGreaterEqual(gears_start.get("16", 0), 2)
        self.assertEqual(reasons_start.get("1"), 4)
        self.assertGreaterEqual(reasons_start.get("0", 0), 2)
        self.assertIn("avg_leaf_ply", s_skew.last_profile)

    def test_dynamic_batch_collision_bounded_flush(self) -> None:
        import ctypes

        b = chess.Board()
        legals = list(b.legal_moves)
        # Choose legals[0] as warm-up branch (visited 16 times during cold start, Q=-0.10)
        # Then legals[1..12] have uniform low priors so wave 5 visits them once root visits >= 16.
        m_warm = legals[0]
        act_warm = uci_to_action(m_warm.uci(), canonical_flip=False)
        next_moves = legals[1:13]
        next_acts = [uci_to_action(m.uci(), canonical_flip=False) for m in next_moves]

        logits_root = np.full((1, 4672), -40.0, dtype=np.float32)
        logits_root[0, act_warm] = 10.0  # p_warm ~ 0.67
        next_logit = float(10.0 + np.log(0.027 / 0.67))
        for act in next_acts:
            logits_root[0, act] = next_logit  # p_next ~ 0.027 each

        logits_leaf = np.zeros((1, 4672), dtype=np.float32)

        class _FlushRootStub:
            batch_size = 1

            def infer(self, x):
                return logits_root.copy(), np.zeros((1, 1), dtype=np.float32)

        class _FlushWarmStub:
            def __init__(self, batch_size: int):
                self.batch_size = batch_size
                self.calls = 0

            def infer(self, x):
                nb = x.shape[0]
                self.calls += 1
                # Call 1 is ply 1 (Black to move): +0.05 => -0.05 for White.
                # Calls 2..15 are ply 2 (White to move): -0.05 => -0.05 for White.
                # Call 16 is ply 2 (White to move): -0.95 => drops Q(m_warm) to -0.106 at N=16 so next_moves win Wave 5.
                if self.calls == 1:
                    v = 0.05
                elif self.calls < 16:
                    v = -0.05
                else:
                    v = -0.95
                return np.zeros((nb, 4672), dtype=np.float32), np.full((nb, 1), v, dtype=np.float32)

        for n_uncached, expected_b, expected_reason in [(2, 4, 3), (6, 8, 4)]:
            warm_stub = _FlushWarmStub(1)
            s = NPUBatchedPUCTSearch(
                root_model=_FlushRootStub(),
                batch_model=_FlushWarmStub(16),
                batch_size=16,
                use_fpu_reduction=False,
                use_nn_cache=True,
                use_dynamic_batch_shift=True,
                batch_models={4: _FlushWarmStub(4), 8: _FlushWarmStub(8), 16: _FlushWarmStub(16)},
                use_fast_tree=True,
            )
            fb = s._fast_backend
            self.assertIsNotNone(fb)

            def _set_root_native(board_obj: chess.Board) -> None:
                cr, ep, tw, hm, rv = fb._extract_cboard_args(board_obj)
                fb.lib.ft_set_root_native(
                    fb._tree_ptr,
                    fb._bbs8_buf,
                    ctypes.c_uint64(cr),
                    ep,
                    tw,
                    hm,
                    rv,
                    fb._rev_hi_buf,
                    fb._rev_lo_buf,
                    ctypes.byref(fb._c_u64_a),
                    ctypes.byref(fb._c_u64_b),
                )

            # Pre-cache 6 children right after the n_uncached uncached leaves (ft_set_root_native preserves C++ LRU cache)
            cached_targets = next_moves[n_uncached : n_uncached + 6]
            for mv in cached_targets:
                cb = b.copy()
                cb.push(mv)
                _set_root_native(cb)
                fb.lib.ft_expand_root_native(
                    fb._tree_ptr,
                    ctypes.c_double(0.0),
                    logits_leaf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    1,
                )

            # Set up root board WITHOUT overwriting the cache, expand root uncached, warm m_warm to 16 visits (root_visits == 16)
            _set_root_native(b)
            fb.lib.ft_expand_root_native(
                fb._tree_ptr,
                ctypes.c_double(0.0),
                logits_root.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                0,
            )
            prof_dummy = {
                "novel_npu_evals": 0,
                "wasted_dup_key_evals": 0,
                "cache_expanded_nodes": 0,
                "collisions": 0,
                "dedup_avoided": 0,
                "nn_cache_hits": 0,
                "nn_cache_misses": 0,
                "terminal_backups": 0,
                "max_depth_cache": 0,
            }
            for _ in range(16):
                # Read-only check: while root_visits < 16, selector stays in Cold-Start B=4 (reason 1)
                rec_b = int(
                    fb.lib.ft_recommend_batch_size(
                        fb._tree_ptr,
                        0,
                        ctypes.byref(fb._c_reason),
                        ctypes.byref(fb._c_entropy),
                        ctypes.byref(fb._c_in_check),
                    )
                )
                self.assertEqual(rec_b, 4)
                self.assertEqual(int(fb._c_reason.value), 1)

                hs1, bt1, nev1, _ = fb.select_and_pack_native(max_depth=14, use_pending=False, batch_size_override=1)
                self.assertEqual(nev1, 1)
                pl, vb = warm_stub.infer(bt1)
                fb.backup_batch_native(hs1, pl, vb, 14, prof_dummy)

            # Now root_visits == 16 >= 16 -> targets B=16, collects n_uncached leaves then hits 6 cached nodes -> flushes!
            hs5, ch_b5, r_code5, nev5, _ = fb.select_dynamic_native(max_depth=14, use_pending=False)
            fb.lib.ft_destroy_inflight_batch(hs5)
            self.assertEqual(nev5, n_uncached)
            self.assertEqual(ch_b5, expected_b)
            self.assertEqual(r_code5, expected_reason)

    def test_research_head_variants_graph_shapes(self) -> None:
        from neurachess.models.se_resnet_backbone import build_openvino_research_head_variant

        specs = [
            ("baseline_scalar", [(4, 4672), (4, 1)]),
            ("hl_gauss_mlh", [(4, 4672), (4, 64), (4, 1)]),
            ("dual_conv73_with_val", [(4, 4672), (4, 4672), (4, 1)]),
            ("dual_conv73_only", [(4, 4672), (4, 4672)]),
        ]
        for variant, expected_shapes in specs:
            m = build_openvino_research_head_variant(variant=variant, batch_size=4, num_blocks=2, channels=32)
            self.assertEqual(len(m.outputs), len(expected_shapes))
            for out_port, exp_shape in zip(m.outputs, expected_shapes):
                self.assertEqual(tuple(out_port.shape), exp_shape)

    def test_hl_gauss_value_head_and_openvino_fold(self) -> None:
        import openvino as ov
        import torch
        import torch.nn.functional as F

        from neurachess.models.se_resnet_backbone import build_openvino_se_resnet
        from neurachess.training.train_hl_gauss_value import (
            HLGaussValueHead,
            compute_hl_gauss_targets,
            hl_gauss_centers_and_edges,
        )

        z_np, edges_np, sigma = hl_gauss_centers_and_edges(64)
        self.assertEqual(z_np.shape, (64,))
        self.assertAlmostEqual(float(z_np[0]), -1.0, places=6)
        self.assertAlmostEqual(float(z_np[-1]), 1.0, places=6)

        cps = torch.tensor([0.0, 200.0, -400.0, 2500.0], dtype=torch.float32)
        probs = compute_hl_gauss_targets(cps, torch.from_numpy(edges_np), sigma)
        self.assertEqual(probs.shape, (4, 64))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5))
        exp_val = probs @ torch.from_numpy(z_np)
        self.assertAlmostEqual(float(exp_val[0]), 0.0, places=4)
        self.assertGreater(float(exp_val[1]), 0.3)
        self.assertLess(float(exp_val[2]), -0.7)

        # Verify OpenVINO HL-Gauss expectation folding matches PyTorch head on [B, 1] value output
        torch.manual_seed(7)
        head = HLGaussValueHead(in_channels=32, val_channels=16, hidden_dim=256, num_bins=64).eval()
        rng = np.random.default_rng(7)
        w = {
            "init_conv_w": (rng.standard_normal((32, 18, 3, 3)) * 0.05).astype(np.float32),
            "val_conv_w": head.val_conv.weight.detach().numpy(),
            "val_conv_b": head.val_conv.bias.detach().numpy(),
            "val_dense1_w": head.val_fc1.weight.detach().numpy(),
            "val_dense1_b": head.val_fc1.bias.detach().numpy(),
            "val_dense2_w": head.val_fc2.weight.detach().numpy(),
            "val_dense2_b": head.val_fc2.bias.detach().numpy(),
            "val_hl_z": z_np,
        }
        ov_m = build_openvino_se_resnet(batch_size=2, num_blocks=0, channels=32, in_channels=18, weights_dict=w)
        comp = ov.Core().compile_model(ov_m, "CPU")
        x = rng.standard_normal((2, 18, 8, 8)).astype(np.float32)
        ov_outs = list(comp({"board_state": x}).values())
        self.assertEqual(tuple(ov_outs[0].shape), (2, 4672))
        self.assertEqual(tuple(ov_outs[1].shape), (2, 1))
        with torch.no_grad():
            feat = F.relu(F.conv2d(torch.from_numpy(x), torch.from_numpy(w["init_conv_w"]), padding=1))
            pt_val = (F.softmax(head(feat), dim=-1) @ torch.from_numpy(z_np)).unsqueeze(-1).numpy()
        self.assertLess(float(np.max(np.abs(ov_outs[1] - pt_val))), 1e-4)

    def test_ponder_locks_static_b16_unless_short_movetime(self) -> None:
        class _AsyncReqStub:
            def __init__(self, bs: int):
                self.bs = bs
                self.results: dict = {}

            def start_async(self, feeds=None):
                b = self.bs if feeds is None else feeds["board_state"].shape[0]
                self.results = {
                    0: np.zeros((b, 4672), dtype=np.float32),
                    1: np.zeros((b, 1), dtype=np.float32),
                }

            def wait(self):
                return None

        class _AsyncModelStub:
            def __init__(self, bs: int):
                self.batch_size = bs
                self.infer_request = _AsyncReqStub(bs)
                self.infer_request_async = _AsyncReqStub(bs)

            def infer(self, x):
                b = x.shape[0]
                return np.zeros((b, 4672), dtype=np.float32), np.zeros((b, 1), dtype=np.float32)

        m16 = _AsyncModelStub(16)
        s = NPUBatchedPUCTSearch(
            root_model=_FixedDummyModel(1),
            batch_model=m16,
            batch_size=16,
            use_dynamic_batch_shift=True,
            use_ponder=True,
            batch_models={4: _FixedDummyModel(4), 8: _FixedDummyModel(8), 16: m16},
            use_fast_tree=True,
        )
        # Pondering ON + movetime > 150ms -> locks static B=16 double_buffer (no B=4/8 gear-shift)
        s.search(chess.Board(), movetime_ms=400, max_nodes=48)
        self.assertEqual(s.last_profile.get("mode"), "double_buffer")
        self.assertEqual(s.last_profile.get("batch_gear_counts"), {})

        # Pondering ON + short movetime <= 150ms -> gear_shift_v3 remains active
        s.clear_state()
        s.search(chess.Board(), movetime_ms=100, max_nodes=48)
        self.assertEqual(s.last_profile.get("mode"), "dynamic_gear_shift")
        self.assertIn("4", s.last_profile.get("batch_gear_counts", {}))

        # Pondering OFF + movetime > 150ms -> gear_shift_v3 active
        s.set_ponder_mode(False)
        s.clear_state()
        s.search(chess.Board(), movetime_ms=400, max_nodes=48)
        self.assertEqual(s.last_profile.get("mode"), "dynamic_gear_shift")
        self.assertIn("4", s.last_profile.get("batch_gear_counts", {}))


if __name__ == "__main__":
    unittest.main()
