"""Comprehensive tests for Phase 4 Cloud Spot data converters, models, QAT, expansion, and OV/NNCF export."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import openvino as ov
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neurachess.core.constants import uci_to_action
from neurachess.models.pytorch_se_resnet import ChessSEResNet
from neurachess.scripts.evaluate import unpack_raw_to_gpu_inputs
from neurachess.training.bitboard_dataset import RECORD_SIZE_V1, pack_position, unpack_position
from neurachess.training.cloud_phase4_worker import (
    SIDECAR_SIZE,
    LichessContinuousStream,
    _build_mixed_epoch_pool,
    _capture_first_batch_pol_logits,
    _train_on_pool,
    compute_cosine_lr,
    ensure_cpp_converters,
    evaluate_unified_150k,
    load_unified_chunk,
    stream_lc0_t80_parallel,
    unpack_unified_sidecar_batch,
    unpack_v1_tensor_to_inputs_and_meta,
)
from neurachess.training.phase4_model import (
    ChessSEResNetPhase4,
    W8A8FakeQuantConv2d,
    apply_middle_w8a8_qat,
    enforce_frozen_bn_eval,
    expand_10x192_to_12x192_identity,
    export_phase4_to_folded_npz_and_ov,
    load_346m_into_10x192_phase4,
)
from tests.test_syzygy_and_lc0_converter import _encode_mock_lc0_v6

LC0_CPP = ROOT_DIR / "src" / "neurachess" / "training" / "lc0_v6_converter.cpp"


def _encode_pack_for_test(board: chess.Board, uci: str, cp: int) -> bytes:
    act = uci_to_action(uci, board.turn == chess.BLACK)
    p_w = 1.0 / (1.0 + 10.0 ** (-cp / 400.0))
    p_l = 1.0 / (1.0 + 10.0 ** (cp / 400.0))
    p_d = max(0.0, 1.0 - p_w - p_l)
    return pack_position(board, act, cp, (p_w, p_d, p_l))


class TestPhase4CloudPipeline(unittest.TestCase):
    def test_unpack_v1_tensor_matches_evaluate_exact(self) -> None:
        cases = [
            (chess.Board(), "e2e4", 35),
            (chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"), "c7c5", -28),
            (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 14 25"), "e1g1", 120),
            (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 3 25"), "e8c8", -410),
            (chess.Board("6k1/P7/8/8/8/8/8/R3K2R w KQ - 0 40"), "a7a8n", 650),
            (chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"), "e5d6", 85),
        ]
        raw_bytes = b"".join(_encode_pack_for_test(b, u, cp) for b, u, cp in cases)
        ref_inputs = unpack_raw_to_gpu_inputs(raw_bytes, len(cases))

        arr_u8 = torch.from_numpy(np.frombuffer(raw_bytes, dtype=np.uint8).reshape(len(cases), RECORD_SIZE_V1).copy())
        got_inputs, got_acts, got_cps = unpack_v1_tensor_to_inputs_and_meta(arr_u8)

        self.assertEqual(float((ref_inputs - got_inputs).abs().max().item()), 0.0)
        for i, (b, u, cp) in enumerate(cases):
            self.assertEqual(int(got_acts[i].item()), uci_to_action(u, b.turn == chess.BLACK))
            self.assertEqual(int(got_cps[i].item()), cp)

    def test_lichess_and_lc0_converters_parity_and_exclusions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="nc_p4_conv_") as td:
            td_p = Path(td)
            lc0_exe, sf_exe = ensure_cpp_converters(td_p)

            # 1. Test Lichess Eval JSONL converter (White, Black STM sign-flip, promo/castle, excl hashes + excl bin)
            fens = [
                ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", [("e2e4", 32, None), ("d2d4", 28, None)]),
                ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", [("c7c5", -18, None), ("e7e5", -25, None)]),
                ("6k1/P7/8/8/8/8/8/R3K2R w KQ - 0 40", [("a7a8q", None, 2), ("e1g1", 310, None)]),
                ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", [("f1b5", 45, None)]),
                ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", [("a7a6", 12, None)]),
            ]

            excl_hash_f = td_p / "excl_hashes.json"
            core4_f3 = " ".join(fens[3][0].split()[:4])
            h3 = hashlib.sha256(core4_f3.encode("utf-8")).hexdigest()[:16]
            excl_hash_f.write_text(json.dumps([h3]), encoding="utf-8")

            b4 = chess.Board(fens[4][0])
            excl_bin_f = td_p / "excl.bin"
            excl_bin_f.write_bytes(_encode_pack_for_test(b4, "a7a6", -12))

            jsonl_p = td_p / "lichess_sample.jsonl"
            lines = []
            for fen_s, pvs in fens:
                pv_objs = []
                for mv_s, cp_v, mate_v in pvs:
                    d_pv = {"line": f"{mv_s} a2a3"}
                    if cp_v is not None:
                        d_pv["cp"] = cp_v
                    if mate_v is not None:
                        d_pv["mate"] = mate_v
                    pv_objs.append(d_pv)
                lines.append(json.dumps({"fen": fen_s, "evals": [{"knodes": 100, "depth": 30, "pvs": pv_objs}]}))
            jsonl_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

            sf_pref = td_p / "sf_out"
            out_sf = subprocess.check_output(
                [
                    str(sf_exe),
                    "--out-prefix",
                    str(sf_pref),
                    "--exclude-hashes",
                    str(excl_hash_f),
                    "--exclude-bin",
                    str(excl_bin_f),
                    str(jsonl_p),
                ],
                text=True,
            )
            stats_sf = json.loads(out_sf.strip().splitlines()[-1])
            self.assertEqual(stats_sf["records_written"], 3)
            self.assertEqual(stats_sf["skipped_overlap"], 2)

            v1_sf, sc_sf = load_unified_chunk(sf_pref.with_suffix(".bin"), sf_pref.with_suffix(".sidecar68.bin"))
            self.assertEqual(v1_sf.shape, (3, RECORD_SIZE_V1))
            self.assertEqual(sc_sf.shape, (3, SIDECAR_SIZE))

            # Check White startpos + Black sign flip (+18 STM when White eval=-18) + Promo/Mate
            rec0 = unpack_position(v1_sf[0].tobytes())
            rec1 = unpack_position(v1_sf[1].tobytes())
            rec2 = unpack_position(v1_sf[2].tobytes())
            self.assertEqual(rec0["action_index"], uci_to_action("e2e4", False))
            self.assertEqual(rec0["eval_cp"], 32)
            self.assertEqual(rec1["action_index"], uci_to_action("c7c5", True))
            self.assertEqual(rec1["eval_cp"], 18)  # -(-18) for Black STM
            self.assertEqual(rec2["action_index"], uci_to_action("a7a8q", False))
            self.assertEqual(rec2["eval_cp"], 10000)

            acts8, probs8, plies_l, has_mlh = unpack_unified_sidecar_batch(torch.from_numpy(sc_sf.copy()), torch.device("cpu"))
            self.assertTrue((has_mlh == 0.0).all())
            # Boltzmann T=0.5 for [32, 28]
            ex = np.exp(np.array([32.0, 28.0]) / 50.0)
            p_exp = ex / ex.sum()
            self.assertAlmostEqual(float(probs8[0, 0].item()), float(p_exp[0]), places=4)
            self.assertAlmostEqual(float(probs8[0, 1].item()), float(p_exp[1]), places=4)

            # 2. Test Lc0 V6 converter MLH bit3 flag & `--exclude-bin`
            cpp_src = LC0_CPP.read_text(encoding="utf-8")
            tbl_m = re.search(r"kLc0ToNcAct\[1858\]\s*=\s*\{([^}]+)\}", cpp_src, re.S)
            k_lc0_to_nc = [int(x) for x in re.findall(r"\d+", tbl_m.group(1))]
            nc_to_lc0 = {ai: li for li, ai in enumerate(k_lc0_to_nc)}

            b_lc0 = chess.Board()
            b_lc0.push_uci("e2e4")
            b_lc0.push_uci("c7c5")
            act_nf3 = uci_to_action("g1f3", b_lc0.turn == chess.BLACK)
            lc0_mv_idx = nc_to_lc0[act_nf3]

            rec_keep = _encode_mock_lc0_v6([b_lc0], {lc0_mv_idx: 0.9}, plies_left=27.5)
            rec_nomlh = _encode_mock_lc0_v6([b_lc0], {lc0_mv_idx: 0.9}, plies_left=0.0)

            b_excl = chess.Board("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
            act_excl = uci_to_action("g8f6", b_excl.turn == chess.BLACK)
            rec_excl = _encode_mock_lc0_v6([b_excl], {nc_to_lc0[act_excl]: 0.9}, plies_left=15.0)
            lc0_excl_f = td_p / "lc0_excl.bin"
            lc0_excl_f.write_bytes(_encode_pack_for_test(b_excl, "g8f6", 0))

            v6_bin = td_p / "sample.v6.bin"
            v6_bin.write_bytes(rec_excl + rec_keep + rec_nomlh)
            lc0_pref = td_p / "lc0_out"
            out_lc0 = subprocess.check_output(
                [
                    str(lc0_exe),
                    "--out-prefix",
                    str(lc0_pref),
                    "--min-ply",
                    "0",
                    "--exclude-bin",
                    str(lc0_excl_f),
                    str(v6_bin),
                ],
                text=True,
            )
            stats_lc0 = json.loads(out_lc0.strip().splitlines()[-1])
            self.assertEqual(stats_lc0["records_written"], 2)

            v1_l, sc_l = load_unified_chunk(lc0_pref.with_suffix(".bin"), lc0_pref.with_suffix(".lc0_sidecar.bin"))
            acts_l, probs_l, pl_l, mlh_l = unpack_unified_sidecar_batch(torch.from_numpy(sc_l.copy()), torch.device("cpu"))
            self.assertEqual(int(mlh_l[0].item()), 1)
            self.assertAlmostEqual(float(pl_l[0].item()), 27.5, places=3)
            self.assertEqual(int(mlh_l[1].item()), 0)

    def test_expand_10x192_to_12x192_identity_zero_drift(self) -> None:
        torch.manual_seed(7)
        m10 = ChessSEResNet(num_blocks=10, channels=192, value_dim=1, policy_head="conv73", use_bn=True)
        with tempfile.TemporaryDirectory(prefix="nc_p4_exp_") as td:
            pt = Path(td) / "mock_346m.pt"
            torch.save({"ema_09999_state_dict": m10.state_dict()}, pt)
            m12, step0_diff = expand_10x192_to_12x192_identity(pt, seed=42, verify_tolerance=1e-6)
            self.assertLess(step0_diff, 1e-6)

            for new_i in (5, 11):
                self.assertEqual(float(m12.res_blocks[new_i].conv2.weight.abs().max().item()), 0.0)
                self.assertGreater(float(m12.res_blocks[new_i].conv1.weight.abs().max().item()), 0.0)

            x = torch.randn(4, 18, 8, 8)
            m10.eval()
            m12.eval()
            with torch.no_grad():
                p10, _ = m10(x)
                p12, _, _, _ = m12(x)
            self.assertLess(float((p10 - p12).abs().max().item()), 1e-6)

    def test_job_a1_frozen_trunk_and_a2_qat_export(self) -> None:
        torch.manual_seed(11)
        device = torch.device("cpu")
        m_ref = ChessSEResNet(num_blocks=10, channels=192, value_dim=1, policy_head="conv73", use_bn=True)

        with tempfile.TemporaryDirectory(prefix="nc_p4_a12_") as td:
            td_p = Path(td)
            ckpt_pt = td_p / "m346_ref.pt"
            torch.save({"model_state_dict": m_ref.state_dict()}, ckpt_pt)

            # Build 12 mixed training + eval positions
            b_list = [
                (chess.Board(), "e2e4", 30, 50.0, True),
                (chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"), "c7c5", -20, 40.0, True),
                (chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"), "f1b5", 45, 0.0, False),
                (chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 20"), "e1g1", 110, 22.0, True),
            ] * 3
            raw_v1 = np.frombuffer(b"".join(_encode_pack_for_test(b, u, cp) for b, u, cp, _, _ in b_list), dtype=np.uint8).reshape(len(b_list), RECORD_SIZE_V1).copy()
            raw_sc = np.zeros((len(b_list), SIDECAR_SIZE), dtype=np.uint8)
            for idx, (b, u, _cp, pl, has_m) in enumerate(b_list):
                raw_sc[idx, 0] = 1
                raw_sc[idx, 1] = 8 if has_m else 0
                raw_sc[idx, 4:20].view(np.uint16)[0] = uci_to_action(u, b.turn == chess.BLACK)
                raw_sc[idx, 20:52].view(np.float32)[0] = 1.0
                raw_sc[idx, 64:68].view(np.float32)[0] = pl

            eval_p = td_p / "tiny_eval.bin"
            eval_p.write_bytes(raw_v1[:4].tobytes())
            eval_files = {"tiny_eval": eval_p}

            # A1: frozen trunk + Conv73 policy -> guaranteed 0.0000 drift
            m_a1 = ChessSEResNetPhase4(num_blocks=10, channels=192, use_bn=True).to(device)
            load_346m_into_10x192_phase4(m_a1, ckpt_pt)
            ref_pol = _capture_first_batch_pol_logits(m_a1, eval_files, device)

            for p in list(m_a1.init_conv.parameters()) + list(m_a1.init_bn.parameters()) + list(m_a1.res_blocks.parameters()) + list(m_a1.pol_conv.parameters()) + list(m_a1.pol_bn.parameters()):
                p.requires_grad_(False)
            for mod in (m_a1.init_bn, m_a1.res_blocks, m_a1.pol_bn):
                mod.eval()

            opt1 = torch.optim.AdamW(list(m_a1.hl_head.parameters()) + list(m_a1.ml_head.parameters()), lr=1e-3)
            st1 = {"step": 0, "seen": 0, "seed": 1, "t0": 0.0, "last_ckpt": 1e18}
            _train_on_pool(
                m_a1, raw_v1, raw_sc, device, opt1,
                batch_size=4, freeze_trunk=True, pol_weight=0.0, val_weight=1.0, mlh_weight=0.05,
                ema_list=[], step_state=st1, lr_fn=lambda o, s, n: None, max_positions=8,
            )
            ev1 = evaluate_unified_150k(m_a1, eval_files, device, batch_size=4, max_eval=4, ref_baseline_pol=ref_pol)
            self.assertEqual(ev1["tiny_eval"]["max_policy_logit_drift_vs_346m"], 0.0)

            # A2: middle-block QAT (2..7) + frozen remaining BN invariant + folded OV FP16 & Middle-INT8 export
            m_a2 = ChessSEResNetPhase4(num_blocks=10, channels=192, use_bn=True).to(device)
            m_a2.load_state_dict(m_a1.state_dict())
            for p in m_a2.parameters():
                p.requires_grad_(True)
            apply_middle_w8a8_qat(m_a2, qat_blocks=(2, 3, 4, 5, 6, 7))
            self.assertIsInstance(m_a2.res_blocks[2].conv1, W8A8FakeQuantConv2d)
            self.assertNotIsInstance(m_a2.res_blocks[1].conv1, W8A8FakeQuantConv2d)

            init_bn_mean_before = m_a2.init_bn.running_mean.clone()
            b8_bn_mean_before = m_a2.res_blocks[8].bn1.running_mean.clone()
            m_a2.train()
            enforce_frozen_bn_eval(m_a2)
            opt2 = torch.optim.AdamW([p for p in m_a2.parameters() if p.requires_grad], lr=2e-5)
            st2 = {"step": 0, "seen": 0, "seed": 2, "t0": 0.0, "last_ckpt": 1e18}
            _train_on_pool(
                m_a2, raw_v1, raw_sc, device, opt2,
                batch_size=4, freeze_trunk=False, pol_weight=1.0, val_weight=1.0, mlh_weight=0.05,
                ema_list=[], step_state=st2, lr_fn=lambda o, s, n: None, max_positions=8,
            )
            self.assertEqual(float((m_a2.init_bn.running_mean - init_bn_mean_before).abs().max().item()), 0.0)
            self.assertEqual(float((m_a2.res_blocks[8].bn1.running_mean - b8_bn_mean_before).abs().max().item()), 0.0)

            calib_in = unpack_v1_tensor_to_inputs_and_meta(torch.from_numpy(raw_v1[:4].copy()))[0].numpy().astype(np.float32)
            calib_list = [calib_in[i : i + 1] for i in range(4)]
            exp = export_phase4_to_folded_npz_and_ov(
                m_a2, td_p / "ov_out", "test_a2", calib_inputs=calib_list, int8_blocks=(2, 3, 4, 5, 6, 7)
            )
            self.assertNotIn("hybrid_int8_error", exp)
            self.assertTrue(Path(exp["folded_npz"]).is_file())
            self.assertTrue(Path(exp["fp16_xml"]).is_file())
            self.assertTrue(Path(exp["hybrid_int8_xml"]).is_file())

            with np.load(exp["folded_npz"]) as npz_d:
                self.assertIn("val_hl_z", npz_d)
                self.assertIn("mlh_d2_w", npz_d)

            core = ov.Core()
            for xml_k in ("fp16_xml", "hybrid_int8_xml"):
                ov_m = core.read_model(exp[xml_k])
                ov_m.reshape({"board_state": [2, 18, 8, 8]})
                comp = core.compile_model(ov_m, "CPU")
                res = comp({"board_state": calib_in[:2]})
                outs = [res[o] for o in comp.outputs]
                self.assertEqual(outs[0].shape, (2, 4672))
                self.assertEqual(outs[1].shape, (2, 1))
                self.assertLessEqual(float(np.abs(outs[1]).max()), 1.0 + 1e-5)

        self.assertGreater(compute_cosine_lr(10, 100, 10, 1e-3, 1e-5), compute_cosine_lr(90, 100, 10, 1e-3, 1e-5))
        mv1, msc = _build_mixed_epoch_pool(np.zeros((2, 107), dtype=np.uint8), np.zeros((2, 68), dtype=np.uint8), np.ones((3, 107), dtype=np.uint8), np.ones((3, 68), dtype=np.uint8), seed=42)
        self.assertEqual(mv1.shape, (5, 107))
        self.assertEqual(msc.shape, (5, 68))

    def test_lichess_continuous_stream_no_reset(self) -> None:
        import io
        import zstandard as zstd

        with tempfile.TemporaryDirectory(prefix="nc_p4_cont_") as td:
            td_p = Path(td)
            _, sf_exe = ensure_cpp_converters(td_p)

            # Build > 17,000 unique valid FEN JSONL lines (> 2 * kBatchLines=8192 in lichess_eval_converter)
            # so Chunk 1 exits cleanly on its 8192-line flush and Chunk 2 continues without reopening the stream.
            pad_fens = []
            b = chess.Board()
            rng = np.random.default_rng(123)
            seen = set()
            while len(pad_fens) < 17500:
                if b.is_game_over() or b.fullmove_number > 35:
                    b.reset()
                ms = list(b.legal_moves)
                mv = ms[int(rng.integers(0, len(ms)))]
                k = b.fen().rsplit(" ", 2)[0]
                if k not in seen:
                    seen.add(k)
                    pad_fens.append(json.dumps({"fen": b.fen(), "evals": [{"pvs": [{"line": f"{mv.uci()} a2a3", "cp": (len(pad_fens) % 50) - 25}]}]}))
                b.push(mv)
            big_lines = "\n".join(pad_fens) + "\n"
            comp = zstd.ZstdCompressor().compress(big_lines.encode("utf-8"))

            opens = {"n": 0}

            def _single_opener(_url: str):
                opens["n"] += 1
                self.assertEqual(opens["n"], 1, "LichessContinuousStream must not reopen stream across chunks")
                return io.BytesIO(comp)

            st = LichessContinuousStream(sf_exe, None, None, initial_skip=2, threads=1, opener=_single_opener)
            try:
                bp1, sp1, cnt1 = st.read_chunk(td_p / "sf_c1", 5000)
                bp2, sp2, cnt2 = st.read_chunk(td_p / "sf_c2", 5000)
                self.assertEqual(opens["n"], 1)
                self.assertEqual(cnt1, 5000)
                self.assertGreater(cnt2, 0)
                v1_1, _ = load_unified_chunk(bp1, sp1)
                v1_2, _ = load_unified_chunk(bp2, sp2)
                keys1 = {v.tobytes()[:98] for v in v1_1}
                keys2 = {v.tobytes()[:98] for v in v1_2}
                self.assertEqual(len(keys1 & keys2), 0, "Continuous chunks overlapped instead of advancing!")
            finally:
                st.close()


if __name__ == "__main__":
    unittest.main()
