"""Unit tests for Syzygy DTZ/WDL probing hooks and C++23 Lc0 v6 stream converter."""

from __future__ import annotations

import gzip
import io
import json
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from neurachess.core.board_tensor import board_to_tensor
from neurachess.core.constants import move_to_action, uci_to_action
from neurachess.scripts.benchmark_phase4 import (
    BEAST_SPECS,
    build_synthetic_hl64_weights,
    count_folded_params,
)
from neurachess.models.se_resnet_backbone import build_openvino_se_resnet
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch
from neurachess.search.syzygy_tb import SyzygyProber, resolve_syzygy_path
from neurachess.training.bitboard_dataset import RECORD_SIZE_V1, unpack_position

ROOT_DIR = Path(__file__).resolve().parents[1]
CONVERTER_CPP = ROOT_DIR / "src" / "neurachess" / "training" / "lc0_v6_converter.cpp"


def _rev_bits_in_bytes(v: int) -> int:
    out = 0
    for b_idx in range(8):
        b = (v >> (8 * b_idx)) & 0xFF
        rb = int(f"{b:08b}"[::-1], 2)
        out |= rb << (8 * b_idx)
    return out


def _rev_bytes(v: int) -> int:
    return int.from_bytes(v.to_bytes(8, "little")[::-1], "little")


def _encode_mock_lc0_v6(
    hist_boards: list[chess.Board],
    probs_map: dict[int, float],
    *,
    root_q: float = 0.25,
    best_q: float = 0.30,
    root_d: float = 0.40,
    best_d: float = 0.38,
    plies_left: float = 42.0,
    result_q: float = 1.0,
    visits: int = 400,
    castling_override: tuple[int, int, int, int] | None = None,
) -> bytes:
    """Pack a single 8,356-byte Lc0 V6TrainingData record from python-chess history."""
    cur = hist_boards[-1]
    is_black = cur.turn == chess.BLACK
    planes = [0] * 104
    pts = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    for h in range(min(8, len(hist_boards))):
        hb = hist_boards[-1 - h]
        us_col = chess.BLACK if is_black else chess.WHITE
        th_col = not us_col
        base = h * 13
        for idx, pt in enumerate(pts):
            mu = int(hb.pieces_mask(pt, us_col))
            mt = int(hb.pieces_mask(pt, th_col))
            if is_black:
                mu = _rev_bytes(mu)
                mt = _rev_bytes(mt)
            planes[base + idx] = _rev_bits_in_bytes(mu)
            planes[base + 6 + idx] = _rev_bits_in_bytes(mt)

    if castling_override is not None:
        c_us000, c_us00, c_th000, c_th00 = castling_override
    else:
        us_c = chess.BLACK if is_black else chess.WHITE
        th_c = not us_c
        c_us000 = int(cur.has_queenside_castling_rights(us_c))
        c_us00 = int(cur.has_kingside_castling_rights(us_c))
        c_th000 = int(cur.has_queenside_castling_rights(th_c))
        c_th00 = int(cur.has_kingside_castling_rights(th_c))

    probs = [0.0] * 1858
    best_idx = 0
    best_p = -1.0
    for k, v in probs_map.items():
        probs[k] = float(v)
        if v > best_p:
            best_p = float(v)
            best_idx = k

    rec = bytearray(8356)
    struct.pack_into("<II", rec, 0, 6, 1)
    struct.pack_into("<1858f", rec, 8, *probs)
    struct.pack_into("<104Q", rec, 8 + 1858 * 4, *planes)
    struct.pack_into(
        "<8B",
        rec,
        8 + 1858 * 4 + 104 * 8,
        c_us000,
        c_us00,
        c_th000,
        c_th00,
        1 if is_black else 0,
        min(100, cur.halfmove_clock),
        0x20,
        0,
    )
    struct.pack_into(
        "<15f",
        rec,
        8 + 1858 * 4 + 104 * 8 + 8,
        root_q,
        best_q,
        root_d,
        best_d,
        plies_left,
        plies_left,
        plies_left,
        result_q,
        0.2,
        best_q,
        best_d,
        plies_left,
        root_q,
        root_d,
        plies_left,
    )
    struct.pack_into("<IHHff", rec, 8 + 1858 * 4 + 104 * 8 + 8 + 60, visits, best_idx, best_idx, 0.1, 0.05)
    return bytes(rec)


class _MockTablebase:
    def __init__(self, wdl_fn, dtz_fn):
        self._wdl_fn = wdl_fn
        self._dtz_fn = dtz_fn

    def get_wdl(self, board: chess.Board):
        return self._wdl_fn(board)

    def get_dtz(self, board: chess.Board):
        return self._dtz_fn(board)

    def close(self):
        pass


class _CountingDummyModel:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.calls = 0
        self.seen_5_or_less_pieces = 0

    def infer(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        arr = np.expand_dims(x, 0) if x.ndim == 3 else x
        for bi in range(arr.shape[0]):
            # Planes 0..11 hold piece bitboards in gpu_unpack layout
            piece_sum = float(np.sum(arr[bi, :12]))
            if 0.5 < piece_sum <= 5.5:
                self.seen_5_or_less_pieces += 1
        b = arr.shape[0]
        return np.zeros((b, 4672), dtype=np.float32), np.zeros((b, 1), dtype=np.float32)


class TestSyzygyAndLc0Converter(unittest.TestCase):
    def test_syzygy_root_dtz_guards_repetition_and_50move(self) -> None:
        prober = SyzygyProber.__new__(SyzygyProber)
        prober.syzygy_dir = Path(".")
        prober.max_pieces = 5
        prober._leaf_cache = {}
        prober.root_hits = 0
        prober.leaf_hits = 0

        # Losing King at halfmove_clock=99 with one legal non-zeroing shuffle -> must hit 50-move draw
        b_save = chess.Board("8/8/8/8/8/3k4/8/2K1Q3 b - - 99 60")
        prober._tb = _MockTablebase(wdl_fn=lambda _b: 2, dtz_fn=lambda _b: 5)
        res_save = prober.probe_root_dtz_move(b_save)
        self.assertIsNotNone(res_save)
        mv_save, eval_save, _ = res_save
        self.assertEqual(eval_save, 0.0)
        self.assertIn(mv_save, b_save.legal_moves)

        # Mate in 1 immediately overrides non-mate DTZ options
        b_m1 = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        prober._tb = _MockTablebase(wdl_fn=lambda _b: -2, dtz_fn=lambda _b: -4)
        res_m1 = prober.probe_root_dtz_move(b_m1)
        self.assertIsNotNone(res_m1)
        mv_m1, eval_m1, tag_m1 = res_m1
        b_chk = b_m1.copy()
        b_chk.push(mv_m1)
        self.assertTrue(b_chk.is_checkmate())
        self.assertEqual(eval_m1, 1.0)
        self.assertEqual(tag_m1, "syzygy_mate_in_1")

        # Winning side prefers shortest DTZ (-c_dtz) and avoids 2-fold/3-fold repetition
        b_rep = chess.Board("6k1/8/5K2/8/8/8/8/R7 w - - 0 1")
        b_rep.push_uci("a1a7")
        b_rep.push_uci("g8h8")
        # Returning a7a1 repeats previous board, while a7g7 is progressive with equal DTZ
        rep_key = b_rep.copy()
        rep_key.push_uci("a7a1")
        rep_tkey = rep_key._transposition_key()
        prober._tb = _MockTablebase(
            wdl_fn=lambda b: -2,
            dtz_fn=lambda b: -4,
        )
        res_prog = prober.probe_root_dtz_move(b_rep)
        self.assertIsNotNone(res_prog)
        mv_prog, eval_prog, _ = res_prog
        self.assertEqual(eval_prog, 1.0)
        b_test = b_rep.copy()
        b_test.push(mv_prog)
        self.assertNotEqual(b_test._transposition_key(), rep_tkey)

    def test_syzygy_internal_leaf_zero_npu_slots_both_backends(self) -> None:
        # 6-piece root (K+Q+P vs k+r+p): captures drop cleanly into <=5 pieces, while quiet moves stay 6 pieces.
        b6 = chess.Board("4r1k1/5p2/6K1/8/8/8/5P2/4Q3 w - - 0 1")
        import threading

        prober = SyzygyProber.__new__(SyzygyProber)
        prober.syzygy_dir = Path(".")
        prober.max_pieces = 5
        prober._lock = threading.Lock()
        prober._scratch_board = chess.Board(None)
        prober._leaf_cache = {}
        prober.root_hits = 0
        prober.leaf_hits = 0
        # Return +1 (wdl=-2 for child => win for White) on any 5-piece leaf
        prober._tb = _MockTablebase(
            wdl_fn=lambda b: (-2 if b.turn == chess.BLACK else 2),
            dtz_fn=lambda b: (-6 if b.turn == chess.BLACK else 6),
        )

        for use_ft in (True, False):
            prober.clear_stats()
            m1 = _CountingDummyModel(1)
            m16 = _CountingDummyModel(16)
            srch = NPUBatchedPUCTSearch(
                root_model=m1,
                batch_model=m16,
                batch_size=16,
                use_fast_tree=use_ft,
                use_nn_cache=True,
                use_tree_reuse=False,
                syzygy_path=None,
            )
            try:
                srch._syzygy = prober
                if srch._fast_backend is not None:
                    srch._fast_backend.set_syzygy_prober(prober)

                _, best_uci, _, nodes, _, _ = srch.search(b6, movetime_ms=300, max_nodes=64)
                self.assertTrue(best_uci)
                self.assertGreater(nodes, 0)
                self.assertGreaterEqual(prober.leaf_hits, 1, f"use_fast_tree={use_ft} missed Syzygy leaf probe")
                self.assertGreaterEqual(
                    srch.last_profile.get("syzygy_leaf_hits", 0),
                    1,
                    f"use_fast_tree={use_ft} did not report syzygy_leaf_hits",
                )
                self.assertEqual(
                    m16.seen_5_or_less_pieces,
                    0,
                    f"use_fast_tree={use_ft} leaked a <=5 piece leaf into NPU batch inference!",
                )
            finally:
                srch._syzygy = None
                if srch._fast_backend is not None:
                    srch._fast_backend._syzygy = None
                    srch._fast_backend._syzygy_cb_ref = None
                    srch._fast_backend._attach_syzygy_to_tree()

    def test_real_syzygy_if_present_submillisecond_root(self) -> None:
        import time

        syz_dir = resolve_syzygy_path("auto")
        if syz_dir is None:
            self.skipTest("weights/syzygy not downloaded yet")
        prober = SyzygyProber(syz_dir, max_pieces=5)
        try:
            b = chess.Board("8/8/8/8/8/4k3/4P3/4K3 w - - 0 1")
            # Warm once then time root probe
            _ = prober.probe_root_dtz_move(b)
            t0 = time.perf_counter()
            res = prober.probe_root_dtz_move(b)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            self.assertIsNotNone(res)
            mv, val, _ = res
            self.assertIn(mv, b.legal_moves)
            self.assertGreaterEqual(val, 0.0)
            self.assertLess(dt_ms, 2.0)
        finally:
            prober.close()

    def test_lc0_v6_converter_tar_gz_ply_frc_ep_and_18plane_parity(self) -> None:
        exe_name = "lc0_v6_converter.exe" if os.name == "nt" else "lc0_v6_converter"
        with tempfile.TemporaryDirectory(prefix="nc_lc0_conv_test_") as td:
            td_p = Path(td)
            exe_p = td_p / exe_name
            cmd = [
                sys.executable,
                "-m",
                "ziglang",
                "c++",
                "-std=c++23",
                "-O3",
                "-ffast-math",
                "-fno-finite-math-only",
                "-march=native",
                "-o",
                str(exe_p),
                str(CONVERTER_CPP),
            ]
            subprocess.check_call(cmd)

            # Game 1: Real standard chess opening -> ply 0..12 including c1e3 + d1d2 so O-O-O (`e1c1`) is legal
            # at ply 12 alongside double-push `f7f5` (legal `e5f6` en-passant capture).
            import re as _re

            cpp_src = CONVERTER_CPP.read_text(encoding="utf-8")
            tbl_m = _re.search(r"kLc0ToNcAct\[1858\]\s*=\s*\{([^}]+)\}", cpp_src, _re.S)
            self.assertIsNotNone(tbl_m)
            k_lc0_to_nc = [int(x) for x in _re.findall(r"\d+", tbl_m.group(1))]
            self.assertEqual(len(k_lc0_to_nc), 1858)
            nc_to_lc0: dict[int, int] = {}
            for li, ai in enumerate(k_lc0_to_nc):
                nc_to_lc0.setdefault(ai, li)

            b = chess.Board()
            hist = [b.copy()]
            for u in ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "e4e5", "f6d7", "c1e3", "c7c5", "d1d2", "f7f5"]:
                b.push_uci(u)
                hist.append(b.copy())
            # hist has 13 boards (ply 0..12); ply 12 is White to move right after Black played f7f5 -> legal e5xf6 EP + O-O-O!
            self.assertEqual(hist[-1].ep_square, chess.F6)
            self.assertIn(chess.Move.from_uci("e1c1"), hist[-1].legal_moves)
            self.assertIn(chess.Move.from_uci("e5f6"), hist[-1].legal_moves)

            rec_g1 = bytearray()
            ep_act = uci_to_action("e5f6", canonical_flip=False)
            ep_lidx = nc_to_lc0[ep_act]
            for ply_i in range(len(hist)):
                cur = hist[ply_i]
                is_blk = cur.turn == chess.BLACK
                if ply_i == 12:
                    # Lc0 e1a1 (97) -> O-O-O (e1c1 = act 335), plus legal e5xf6 en-passant capture!
                    pmap = {97: 0.60, ep_lidx: 0.40}
                else:
                    first_m = next(m for m in cur.legal_moves if not cur.is_castling(m))
                    pmap = {nc_to_lc0[move_to_action(first_m, canonical_flip=is_blk)]: 1.0}
                rec_g1.extend(_encode_mock_lc0_v6(hist[: ply_i + 1], pmap, plies_left=float(50 - ply_i)))

            # Build real Lc0 index map only for legal moves of hist[9..12] using uci lookup via CBoard?
            # Simpler: put a valid Chess960 game (King on c1, Rook on a1) that must be SKIPPED completely (`skipped_960`)!
            b_960 = chess.Board("r1k1rb2/pppb1qpp/2n1ppn1/3p4/3PP3/P1N1BP2/1PP1N1PP/R1KR1BQ1 w - - 0 1")
            rec_960 = bytearray()
            for ply_i in range(12):
                rec_960.extend(
                    _encode_mock_lc0_v6(
                        [b_960] * (ply_i + 1),
                        {159: 1.0},
                        castling_override=(1, 0, 1, 0),
                    )
                )

            tar_p = td_p / "sample_t80.tar"
            with tarfile.open(tar_p, "w") as tf:
                gz1 = gzip.compress(bytes(rec_g1))
                ti1 = tarfile.TarInfo("t80/game1.gz")
                ti1.size = len(gz1)
                tf.addfile(ti1, io.BytesIO(gz1))

                gz2 = gzip.compress(bytes(rec_960))
                ti2 = tarfile.TarInfo("t80/game960.gz")
                ti2.size = len(gz2)
                tf.addfile(ti2, io.BytesIO(gz2))

            out_pref = td_p / "converted_sample"
            run = subprocess.run(
                [str(exe_p), "--out-prefix", str(out_pref), "--min-ply", "9", "--threads", "2", "--verify-18planes", str(tar_p)],
                check=True,
                capture_output=True,
                text=True,
            )
            stats = json.loads(run.stdout.strip().splitlines()[-1])
            self.assertEqual(stats["chunks_seen"], 2)
            self.assertEqual(stats["skipped_ply"], 9)  # ply 0..8 skipped!
            self.assertEqual(stats["skipped_960"], 12)  # entire FRC game skipped cleanly!
            self.assertEqual(stats["records_written"], 4)  # plies 9, 10, 11, 12 written!
            self.assertGreaterEqual(stats["ep_detected"], 1)

            bin_p = Path(stats["out_bin"])
            sc_p = Path(stats["out_sidecar"])
            self.assertEqual(bin_p.stat().st_size, stats["records_written"] * RECORD_SIZE_V1)
            self.assertEqual(sc_p.stat().st_size, stats["records_written"] * 68)

            # Verify last written record corresponds to hist[12] with exact EP=F6, castling, and e1a1 -> e1c1 action=335!
            bin_bytes = bin_p.read_bytes()
            sc_bytes = sc_p.read_bytes()
            last_rec = unpack_position(bin_bytes[-RECORD_SIZE_V1:])
            self.assertEqual(last_rec["board"].ep_square, chess.F6)
            self.assertEqual(last_rec["board"].board_fen(), hist[12].board_fen())
            self.assertEqual(last_rec["action_index"], uci_to_action("e1c1", canonical_flip=False))
            self.assertTrue(
                np.allclose(
                    board_to_tensor(last_rec["board"], layout="gpu_unpack"),
                    board_to_tensor(hist[12], layout="gpu_unpack"),
                )
            )
            num_topk, flags, vis = struct.unpack_from("<BBH", sc_bytes, len(sc_bytes) - 68)
            acts8 = struct.unpack_from("<8H", sc_bytes, len(sc_bytes) - 64)
            probs8 = struct.unpack_from("<8f", sc_bytes, len(sc_bytes) - 48)
            bq, bd, rq, pl = struct.unpack_from("<4f", sc_bytes, len(sc_bytes) - 16)
            self.assertEqual(num_topk, 2)
            self.assertEqual(acts8[0], uci_to_action("e1c1", canonical_flip=False))
            self.assertEqual(acts8[1], ep_act)
            self.assertAlmostEqual(probs8[0], 0.60, places=4)
            self.assertAlmostEqual(probs8[1], 0.40, places=4)
            self.assertAlmostEqual(sum(probs8[:num_topk]), 1.0, places=5)
            self.assertAlmostEqual(bq, 0.30, places=4)
            self.assertAlmostEqual(bd, 0.38, places=4)
            self.assertAlmostEqual(rq, 1.0, places=4)
            self.assertAlmostEqual(pl, 38.0, places=3)

    def test_beast_synthetic_architectures_build_cleanly(self) -> None:
        import openvino as ov

        x = np.zeros((2, 18, 8, 8), dtype=np.float32)
        core = ov.Core()
        for spec in BEAST_SPECS:
            w = build_synthetic_hl64_weights(spec["num_blocks"], spec["channels"])
            self.assertGreater(count_folded_params(w), 6_000_000)
            m = build_openvino_se_resnet(
                batch_size=2,
                num_blocks=spec["num_blocks"],
                channels=spec["channels"],
                in_channels=18,
                weights_dict=w,
            )
            cm = core.compile_model(m, "CPU")
            res = cm([x])
            self.assertEqual(tuple(res[0].shape), (2, 4672))
            self.assertEqual(tuple(res[1].shape), (2, 1))
