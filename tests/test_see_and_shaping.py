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
        # Position in check -> selects B=4 gear
        in_check_fen = "4k3/8/8/8/8/8/4r3/4K2R w - - 0 1"
        s.clear_state()
        _, uci_chk, _, nodes_chk, _, _ = s.search(chess.Board(in_check_fen), movetime_ms=100, max_nodes=24)
        self.assertTrue(uci_chk)
        self.assertGreater(nodes_chk, 0)
        self.assertEqual(s.last_profile.get("mode"), "dynamic_gear_shift")
        gears = s.last_profile.get("batch_gear_counts", {})
        self.assertIn("4", gears)

        # Broad startpos with flat dummy priors -> selects B=16 gear
        s.clear_state()
        _, uci_start, _, _, _, _ = s.search(chess.Board(), movetime_ms=100, max_nodes=32)
        self.assertTrue(uci_start)
        self.assertIn("16", s.last_profile.get("batch_gear_counts", {}))
        self.assertIn("avg_leaf_ply", s.last_profile)


if __name__ == "__main__":
    unittest.main()
