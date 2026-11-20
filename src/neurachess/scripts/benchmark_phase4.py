"""Phase 4 Benchmarks: Syzygy Downloader, Expanded NPU Backbones (B=1/4/8/16), & Cloud FinOps Audit."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import chess
import numpy as np
import openvino as ov
import psutil

from neurachess._paths import find_repo_root
from neurachess.models.npu_compiler import DEFAULT_WEIGHTS_NPZ, NPUModel
from neurachess.models.se_resnet_backbone import build_openvino_se_resnet
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch
from neurachess.search.syzygy_tb import download_syzygy_tables

ROOT_DIR = find_repo_root()
S3_BUCKET = "neurachess-spot-162290422654-us-east-1"

BEAST_SPECS = [
    {"name": "12x192_mid2_9_hl64", "num_blocks": 12, "channels": 192, "int8_range": (2, 9)},
    {"name": "14x192_mid2_11_hl64", "num_blocks": 14, "channels": 192, "int8_range": (2, 11)},
    {"name": "10x224_mid2_7_hl64", "num_blocks": 10, "channels": 224, "int8_range": (2, 7)},
    {"name": "12x224_mid2_9_hl64", "num_blocks": 12, "channels": 224, "int8_range": (2, 9)},
]


def assert_ac_power() -> dict:
    batt = psutil.sensors_battery()
    if batt is not None:
        if not batt.power_plugged:
            raise RuntimeError(f"ABORT: Laptop is on battery ({batt.percent}%). Connect AC power.")
        return {"power_plugged": True, "battery_pct": round(float(batt.percent), 1)}
    return {"power_plugged": True, "battery_pct": 100.0}


def build_synthetic_hl64_weights(num_blocks: int, channels: int, val_ch: int = 16, val_h: int = 256, seed: int = 42) -> dict[str, np.ndarray]:
    """Create deterministic He-initialized SE-ResNet + HL-Gauss-64 weights matching production folded NPZ naming."""
    rng = np.random.default_rng(seed + num_blocks * 1000 + channels)
    se_ch = max(8, channels // 4)
    w: dict[str, np.ndarray] = {}

    def _conv(out_c: int, in_c: int, k: int, res_scale: float = 1.0) -> np.ndarray:
        std = math_std(in_c * k * k) * res_scale
        return (rng.standard_normal((out_c, in_c, k, k)) * std).astype(np.float32)

    def _linear(out_f: int, in_f: int) -> np.ndarray:
        std = math_std(in_f)
        return (rng.standard_normal((out_f, in_f)) * std).astype(np.float32)

    def math_std(fan_in: int) -> float:
        return float(np.sqrt(2.0 / max(1, fan_in)))

    w["init_conv_w"] = _conv(channels, 18, 3)
    w["init_conv_b"] = np.zeros(channels, dtype=np.float32)

    # Slight residual scaling keeps random untrained activations well-conditioned across 10-14 blocks for NNCF
    res_scale = 1.0 / np.sqrt(max(1.0, float(num_blocks)))
    for i in range(num_blocks):
        w[f"res_{i}_conv1_w"] = _conv(channels, channels, 3)
        w[f"res_{i}_conv1_b"] = np.zeros(channels, dtype=np.float32)
        w[f"res_{i}_conv2_w"] = _conv(channels, channels, 3, res_scale=res_scale)
        w[f"res_{i}_conv2_b"] = np.zeros(channels, dtype=np.float32)
        w[f"res_{i}_se_sq_w"] = _conv(se_ch, channels, 1)
        w[f"res_{i}_se_ex_w"] = _conv(channels, se_ch, 1)

    w["pol_73_w"] = _conv(73, channels, 1)
    w["pol_73_b"] = np.zeros(73, dtype=np.float32)
    w["val_conv_w"] = _conv(val_ch, channels, 1)
    w["val_conv_b"] = np.zeros(val_ch, dtype=np.float32)
    w["val_dense1_w"] = _linear(val_h, val_ch * 64)
    w["val_dense1_b"] = np.zeros(val_h, dtype=np.float32)
    w["val_dense2_w"] = _linear(64, val_h)
    w["val_dense2_b"] = np.zeros(64, dtype=np.float32)
    w["val_hl_z"] = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    return w


def count_folded_params(weights_dict: dict[str, np.ndarray]) -> int:
    return int(sum(int(arr.size) for k, arr in weights_dict.items() if k != "val_hl_z"))


def _load_500_calib(private_root: Path) -> list[np.ndarray]:
    from neurachess.scripts.evaluate import unpack_raw_to_gpu_inputs
    from neurachess.training.bitboard_dataset import RECORD_SIZE_V1

    candidates = [
        private_root / "data" / "train_300k.bin",
        private_root / "data" / "val_100k.bin",
        private_root / "data" / "test_far_50k.bin",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is not None:
        with src.open("rb") as f:
            raw = f.read(500 * RECORD_SIZE_V1)
        x = unpack_raw_to_gpu_inputs(raw, len(raw) // RECORD_SIZE_V1).numpy().astype(np.float32)
        return [x[i : i + 1] for i in range(x.shape[0])]

    from neurachess.core.board_tensor import board_to_tensor

    rng = np.random.default_rng(7)
    out = []
    b = chess.Board()
    for _ in range(500):
        out.append(board_to_tensor(b, layout="gpu_unpack")[None, ...])
        leg = list(b.legal_moves)
        if not leg or b.is_game_over():
            b.reset()
        else:
            b.push(leg[int(rng.integers(0, len(leg)))])
    return out


def _benchmark_single_beast_worker(spec: dict, private_root: Path, out_json: Path) -> dict:
    """Run inside a dedicated clean subprocess so RSS/memory/blob telemetry is isolated per architecture."""
    import nncf

    assert_ac_power()
    name = spec["name"]
    num_blocks = int(spec["num_blocks"])
    channels = int(spec["channels"])
    lo_int8, hi_int8 = int(spec["int8_range"][0]), int(spec["int8_range"][1])

    weights_dict = build_synthetic_hl64_weights(num_blocks, channels)
    param_count = count_folded_params(weights_dict)
    base_ov = build_openvino_se_resnet(
        batch_size=-1,
        num_blocks=num_blocks,
        channels=channels,
        in_channels=18,
        weights_dict=weights_dict,
        model_name=f"NeuraChess_{name}",
    )

    calib_500 = _load_500_calib(private_root)
    calib_ds = nncf.Dataset(calib_500, lambda a: {"board_state": a})
    fp16_blocks = [j for j in range(num_blocks) if not (lo_int8 <= j <= hi_int8)]
    ig_patterns = [
        "^init_.*",
        ".*_se_.*",
        "^pol_.*",
        "^val_.*",
        "^(value|wdl)$",
        ".*_out_.*",
    ] + [f"^res_{j}_.*" for j in fp16_blocks]
    ig = nncf.IgnoredScope(patterns=ig_patterns, validate=True)

    work_dir = Path(tempfile.mkdtemp(prefix=f"nc_beast_{name}_"))
    ir_xml = work_dir / f"{name}.xml"
    cache_dir = work_dir / "ov_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        tq0 = time.perf_counter()
        qm = nncf.quantize(
            base_ov,
            calib_ds,
            target_device=nncf.TargetDevice.NPU,
            subset_size=len(calib_500),
            fast_bias_correction=True,
            ignored_scope=ig,
        )
        quant_time_s = round(time.perf_counter() - tq0, 2)
        ov.save_model(qm, str(ir_xml), compress_to_fp16=True)
        bin_size_mb = round(ir_xml.with_suffix(".bin").stat().st_size / 1048576.0, 2)

        del base_ov, qm, calib_500, calib_ds
        gc.collect()

        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        hyb_ov = ov.Core().read_model(str(ir_xml))
        co_models: dict[int, NPUModel] = {}
        exec_devices: dict[str, list[str]] = {}
        for b in (1, 4, 8, 16):
            m = NPUModel(
                batch_size=b,
                cache_dir=str(cache_dir),
                device="NPU",
                use_turbo=True,
                performance_hint="LATENCY",
                ov_model_override=hyb_ov,
                verbose=False,
            )
            co_models[b] = m
            try:
                ed = m.compiled_model.get_property("EXECUTION_DEVICES")
                exec_devices[f"B={b}"] = [ed] if isinstance(ed, str) else list(ed)
            except Exception:
                exec_devices[f"B={b}"] = ["NPU"]

        rss_after = proc.memory_info().rss
        coresident_rss_delta_mb = round(max(0, rss_after - rss_before) / 1048576.0, 2)
        blob_bytes = sum(p.stat().st_size for p in cache_dir.glob("*") if p.is_file())
        coresident_blob_mb = round(blob_bytes / 1048576.0, 2)

        shape_lats: dict[str, dict] = {}
        for b in (1, 4, 8, 16):
            req = co_models[b].infer_request
            req.get_input_tensor().data[...] = 0.0
            for _ in range(10):
                req.infer()
            lats_ms = []
            for _ in range(60):
                t0 = time.perf_counter()
                req.infer()
                lats_ms.append((time.perf_counter() - t0) * 1000.0)
            med = float(np.median(lats_ms))
            shape_lats[f"B={b}"] = {
                "batch_size": b,
                "compile_time_s": round(co_models[b].compile_time_s, 3),
                "turnaround_median_ms": round(med, 3),
                "turnaround_mean_ms": round(float(np.mean(lats_ms)), 3),
                "turnaround_p90_ms": round(float(np.percentile(lats_ms, 90)), 3),
                "raw_pos_per_sec": round(b * 1000.0 / max(1e-6, med), 1),
            }

        # Analytical in-engine projections calibrated from Phase 3 real-game telemetry:
        #   - Phase 3 hybrid gear_shift_v3 ratio: 4092.4 h2h NPS / 2956.6 B=16 raw pos/s = 1.3842x
        #   - v1.0.2 Pure FP16 conservative match ratio: 3314.4 h2h NPS / 2555.2 B=16 raw pos/s = 1.2971x
        # Note: our 12-ply uniform-prior unpondered synthetic probe under-predicts trained match NPS
        # (even real v1.0.2 Pure FP16 measures 3,121.9 NPS in that 12-ply loop vs 3,314.4 in full games).
        b16_raw_nps = shape_lats["B=16"]["raw_pos_per_sec"]
        weighted_batch_raw_nps = (
            0.808 * shape_lats["B=16"]["raw_pos_per_sec"]
            + 0.037 * shape_lats["B=8"]["raw_pos_per_sec"]
            + 0.155 * shape_lats["B=4"]["raw_pos_per_sec"]
        )
        projected_in_engine_nps_with_reuse = round(b16_raw_nps * (4092.4 / 2956.6), 1)
        projected_fp16_calibrated_nps = round(b16_raw_nps * (3314.4 / 2555.2), 1)
        projected_in_engine_nps_cold = round(weighted_batch_raw_nps * (3550.0 / 2860.0), 1)

        def _make_srch():
            return NPUBatchedPUCTSearch(
                root_model=co_models[1],
                batch_model=co_models[16],
                batch_size=16,
                c_puct=1.5,
                use_fpu_reduction=True,
                fpu_reduction=0.20,
                root_fpu_reduction=0.10,
                t_root=1.0,
                use_exact_solver=True,
                use_nn_cache=True,
                use_tree_reuse=True,
                use_fast_tree=True,
                use_dynamic_batch_shift=True,
                batch_models={4: co_models[4], 8: co_models[8], 16: co_models[16]},
                syzygy_path=None,
            )

        warm = _make_srch()
        wb = chess.Board()
        warm.search(wb, movetime_ms=120)
        wb.push_uci("e2e4")
        warm.search(wb, movetime_ms=120)
        del warm

        sw = _make_srch()
        sb = _make_srch()
        board = chess.Board()
        for u in ("e2e4", "c7c5", "g1f3", "d7d6"):
            board.push_uci(u)
        tot_nodes = 0
        tot_added = 0
        tot_wall = 0.0
        plies = 0
        gears_agg: dict[str, int] = {}
        for _ in range(12):
            if board.is_game_over(claim_draw=True):
                break
            active = sw if board.turn == chess.WHITE else sb
            t0 = time.perf_counter()
            _, uci, _, nodes, _, _ = active.search(board, movetime_ms=400)
            tot_wall += time.perf_counter() - t0
            p = active.last_profile or {}
            tot_nodes += int(nodes)
            tot_added += int(p.get("added_root_visits", nodes))
            for gk, gv in (p.get("batch_gear_counts") or {}).items():
                gears_agg[gk] = gears_agg.get(gk, 0) + int(gv)
            board.push_uci(uci)
            plies += 1

        measured_total_nps = round(tot_nodes / max(1e-6, tot_wall), 1)
        measured_added_nps = round(tot_added / max(1e-6, tot_wall), 1)

        row = {
            "name": name,
            "num_blocks": num_blocks,
            "channels": channels,
            "int8_blocks": f"{lo_int8}..{hi_int8}",
            "fp16_blocks": fp16_blocks,
            "param_count": param_count,
            "param_millions": round(param_count / 1e6, 2),
            "bin_size_mb": bin_size_mb,
            "quant_time_s": quant_time_s,
            "coresident_rss_delta_mb": coresident_rss_delta_mb,
            "coresident_blob_mb": coresident_blob_mb,
            "execution_devices": exec_devices,
            "shape_latencies": shape_lats,
            "b1_ms": shape_lats["B=1"]["turnaround_median_ms"],
            "b4_ms": shape_lats["B=4"]["turnaround_median_ms"],
            "b8_ms": shape_lats["B=8"]["turnaround_median_ms"],
            "b16_ms": shape_lats["B=16"]["turnaround_median_ms"],
            "b16_raw_pos_s": b16_raw_nps,
            "projected_gear_shift_v3_nps": projected_in_engine_nps_with_reuse,
            "projected_fp16_calibrated_nps": projected_fp16_calibrated_nps,
            "projected_cold_nps": projected_in_engine_nps_cold,
            "measured_in_engine_gear_shift_v3_nps": measured_total_nps,
            "measured_added_visits_nps": measured_added_nps,
            "measured_gears": gears_agg,
            "meets_3250_nps_gate": bool(projected_fp16_calibrated_nps >= 3250.0 and b16_raw_nps >= 2555.2),
        }
        out_json.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return row
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def run_step2_beast_benchmarks(private_root: Path) -> dict:
    assert_ac_power()
    print("\n" + "=" * 96, flush=True)
    print("PHASE 4 STEP 2: EXPANDED MIDDLE-INT8 HYBRID NPU BACKBONES (B=1/4/8/16, LATENCY, NPU_TURBO=YES)", flush=True)
    print("=" * 96, flush=True)

    results = []
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["PYTHONUNBUFFERED"] = "1"

    with tempfile.TemporaryDirectory(prefix="nc_beast_json_") as td:
        for spec in BEAST_SPECS:
            assert_ac_power()
            out_j = Path(td) / f"{spec['name']}.json"
            cmd = [
                sys.executable,
                "-m",
                "neurachess.scripts.benchmark_phase4",
                "--step",
                "2-worker",
                "--worker-spec",
                json.dumps(spec),
                "--worker-out",
                str(out_j),
                "--private-root",
                str(private_root),
            ]
            subprocess.run(cmd, check=True, env=env)
            row = json.loads(out_j.read_text(encoding="utf-8"))
            results.append(row)
            print(
                f"  [{row['name']:20s}] Params={row['param_millions']:5.2f}M | .bin={row['bin_size_mb']:5.2f}MB | "
                f"NPU Blobs={row['coresident_blob_mb']:5.2f}MB RSS+={row['coresident_rss_delta_mb']:6.1f}MB | "
                f"B4={row['b4_ms']:5.2f}ms B8={row['b8_ms']:5.2f}ms B16={row['b16_ms']:5.2f}ms | "
                f"ProjHyb={row['projected_gear_shift_v3_nps']:6.1f} ProjCal={row['projected_fp16_calibrated_nps']:6.1f} Meas12ply={row['measured_in_engine_gear_shift_v3_nps']:6.1f} | "
                f"Gate(>=3250)={'PASS' if row['meets_3250_nps_gate'] else 'FAIL'}",
                flush=True,
            )

    passing = [r for r in results if r["meets_3250_nps_gate"]]
    winner = max(passing, key=lambda r: (r["param_count"], r["measured_in_engine_gear_shift_v3_nps"])) if passing else None
    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_10x192_fp16_nps": 3314.4,
        "baseline_10x192_fp16_12ply_cold_nps": 3121.9,
        "baseline_10x192_fp16_b16_ms": 6.262,
        "baseline_10x192_hybrid_gear_shift_v3_nps": 4092.4,
        "nps_gate_threshold": 3250.0,
        "architectures": results,
        "selected_largest_passing": winner["name"] if winner else None,
        "selected_summary": winner,
    }
    out_path = ROOT_DIR / "eval_results" / "phase4_beast_archs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[Step 2 Winner] {payload['selected_largest_passing']} -> saved {out_path}", flush=True)
    return payload


def run_step3_finops_audit(private_root: Path, lc0_sample_stats: dict | None = None) -> dict:
    """Inspect AWS Guard spend, quotas, S3 inventory, 0 active jobs, and Job A/B Spot cost projections."""
    import boto3

    guard_dir = private_root / "cloud" / "guard"
    sys.path.insert(0, str(guard_dir))
    import spend as guard_spend

    cfg = guard_spend.load_config()
    acct = guard_spend.account_id()
    rep = guard_spend.evaluate(cfg, acct)

    # Query Service Quotas & SageMaker InProgress counts (zero write/mutate actions)
    quotas = {}
    try:
        sq = boto3.client("service-quotas", region_name="us-east-1")
        for qcode, label in [
            ("L-1DE24427", "ml.g6e.xlarge_spot"),
            ("L-CAEE7DB7", "ml.g5.2xlarge_spot"),
            ("L-1E3225F6", "ml.g6e.xlarge_ondemand"),
            ("L-2D6DEB3C", "ml.g5.2xlarge_ondemand"),
        ]:
            try:
                q = sq.get_service_quota(ServiceCode="sagemaker", QuotaCode=qcode)["Quota"]
                quotas[label] = float(q["Value"])
            except Exception as e:
                quotas[label] = f"err:{type(e).__name__}"
    except Exception as e:
        quotas["error"] = str(e)

    active_sm_jobs = {}
    for reg in cfg["watch_regions"]:
        try:
            sm = boto3.client("sagemaker", region_name=reg)
            in_prog = sm.list_training_jobs(StatusEquals="InProgress").get("TrainingJobSummaries", [])
            stopping = sm.list_training_jobs(StatusEquals="Stopping").get("TrainingJobSummaries", [])
            active_sm_jobs[reg] = len(in_prog) + len(stopping)
        except Exception as e:
            active_sm_jobs[reg] = f"err:{type(e).__name__}"

    beast_path = ROOT_DIR / "eval_results" / "phase4_beast_archs.json"
    beast_doc = json.loads(beast_path.read_text(encoding="utf-8")) if beast_path.exists() else {}
    winner = beast_doc.get("selected_summary") or {}
    win_name = winner.get("name") or "12x192_mid2_9_hl64"
    win_params = float(winner.get("param_millions") or 8.52)

    freeze_at = float(rep.get("freeze_at_usd", 75.0))
    est_spend = float(rep.get("estimate_usd", 0.0))
    remaining_before_freeze = round(freeze_at - est_spend, 3)

    # Cost model grounded in historical SageMaker runs:
    # - neurachess-full-10x192-1794469303: ml.g6e.xlarge (L40S 48GB, Spot), train_s=9641, bill_s=8026 (0.832x),
    #   344M positions -> ~35,681 pos/s end-to-end incl. setup & validation (~$2.61/hr on-demand -> ~$0.783/hr @70% off
    #   or ~$2.17/hr effective at historical 0.832x bill_s/train_s during peak capacity).
    # - neurachess-qat-10x192-1794514883: ml.g5.2xlarge (A10G 24GB, Spot), train_s=3196, bill_s=2880 (0.901x),
    #   20M QAT positions -> 7,081 pos/s (~$1.515/hr on-demand -> ~$0.455/hr @70% off or ~$1.365/hr @0.901x).
    # Scale throughput inversely with compute ratio vs 10x192 (~6.73M params):
    g6e_10x192_pos_s = 35681.0
    g5_10x192_pos_s = 16500.0  # pure BF16 without QAT fake-quant hooks (~2.3x faster than QAT's 7,081 pos/s)
    scale_vs_10x192 = max(1.0, win_params / 6.73)
    job_a_hours_g6e = (40_000_000 / g6e_10x192_pos_s) / 3600.0 * 1.15  # +15% Lc0 top-8 KL + 150K GPU eval
    job_a_hours_g5 = (40_000_000 / g5_10x192_pos_s) / 3600.0 * 1.15
    job_b_pos_s_g6e = g6e_10x192_pos_s / scale_vs_10x192
    job_b_pos_s_g5 = g5_10x192_pos_s / scale_vs_10x192
    job_b_hours_g6e = (300_000_000 / job_b_pos_s_g6e) / 3600.0
    job_b_hours_g5 = (300_000_000 / job_b_pos_s_g5) / 3600.0

    g6e_od_hr, g6e_spot70_hr, g6e_hist_hr = 2.61, 0.783, 2.17
    g5_od_hr, g5_spot70_hr, g5_hist_hr = 1.515, 0.455, 1.365

    job_a_spot_g6e = round(job_a_hours_g6e * g6e_spot70_hr, 2)
    job_a_hist_g6e = round(job_a_hours_g6e * g6e_hist_hr, 2)
    job_a_od_g6e = round(job_a_hours_g6e * g6e_od_hr, 2)
    job_b_spot_g6e = round(job_b_hours_g6e * g6e_spot70_hr, 2)
    job_b_hist_g6e = round(job_b_hours_g6e * g6e_hist_hr, 2)
    job_b_od_g6e = round(job_b_hours_g6e * g6e_od_hr, 2)

    plan = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "aws_guard": {
            "stage": rep.get("stage"),
            "frozen_iam_policy_attached": bool(rep.get("frozen")),
            "estimate_spend_gross_usd": est_spend,
            "worst_case_total_usd": rep.get("worst_case_total_usd"),
            "freeze_at_usd": freeze_at,
            "kill_at_usd": rep.get("kill_at_usd"),
            "remaining_below_freeze_usd": remaining_before_freeze,
            "active_running_jobs_guard": len(rep.get("running_jobs", [])),
            "active_sagemaker_jobs_by_region": active_sm_jobs,
            "quotas": quotas,
        },
        "lc0_converter_sample": lc0_sample_stats,
        "jobs": {
            "job_a_10x192_hl_gauss64_40m_mix": {
                "description": "40M-pos joint BF16 fine-tune of 346M 10x192 + HL-Gauss-64 (20M Stockfish + 20M Lc0 T80 top-8) + 150K unified GPU eval before teardown",
                "primary_instance": "ml.g6e.xlarge (1x L40S 48GB, Spot)",
                "fallback_instance": "ml.g5.2xlarge (1x A10G 24GB, Spot)",
                "est_runtime_g6e_min": round(job_a_hours_g6e * 60.0, 1),
                "est_spot70_cost_g6e_usd": job_a_spot_g6e,
                "est_hist83_cost_g6e_usd": job_a_hist_g6e,
                "max_ondemand_cost_g6e_usd": job_a_od_g6e,
                "est_runtime_g5_min": round(job_a_hours_g5 * 60.0, 1),
                "est_spot70_cost_g5_usd": round(job_a_hours_g5 * g5_spot70_hr, 2),
                "est_hist90_cost_g5_usd": round(job_a_hours_g5 * g5_hist_hr, 2),
                "max_ondemand_cost_g5_usd": round(job_a_hours_g5 * g5_od_hr, 2),
                "guard_max_runtime_s": 3600,
                "guard_worst_case_cap_usd": 2.61,
            },
            "job_b_beast_scale": {
                "architecture": win_name,
                "param_millions": round(win_params, 2),
                "scale_vs_10x192": round(scale_vs_10x192, 2),
                "description": f"Full-scale Spot training of winning Step 2 architecture ({win_name}, {win_params:.2f}M params) + HL-Gauss-64 + MLH on hybrid Lc0 T80 + Stockfish stream (~300M positions)",
                "primary_instance": "ml.g6e.xlarge (1x L40S 48GB, Spot)",
                "fallback_instance": "ml.g5.2xlarge (1x A10G 24GB, Spot)",
                "est_runtime_g6e_hr": round(job_b_hours_g6e, 2),
                "est_spot70_cost_g6e_usd": job_b_spot_g6e,
                "est_hist83_cost_g6e_usd": job_b_hist_g6e,
                "max_ondemand_cost_g6e_usd": job_b_od_g6e,
                "est_runtime_g5_hr": round(job_b_hours_g5, 2),
                "est_spot70_cost_g5_usd": round(job_b_hours_g5 * g5_spot70_hr, 2),
                "est_hist90_cost_g5_usd": round(job_b_hours_g5 * g5_hist_hr, 2),
                "max_ondemand_cost_g5_usd": round(job_b_hours_g5 * g5_od_hr, 2),
                "guard_max_runtime_s": 21600,
                "guard_worst_case_cap_usd": 15.66,
            },
        },
        "total_expected_spot70_usd": round(job_a_spot_g6e + job_b_spot_g6e, 2),
        "total_expected_hist83_usd": round(job_a_hist_g6e + job_b_hist_g6e, 2),
        "total_worst_case_ondemand_cap_usd": round(2.61 + 15.66, 2),
        "projected_spend_after_both_spot70_usd": round(est_spend + job_a_spot_g6e + job_b_spot_g6e, 2),
        "projected_spend_after_both_hist83_usd": round(est_spend + job_a_hist_g6e + job_b_hist_g6e, 2),
        "projected_spend_after_both_worst_usd": round(est_spend + 2.61 + 15.66, 2),
        "note": "Account currently has NeuraChessFreeze attached (frozen=True); owner approval (`python cloud/guard/guardctl.py unfreeze`) is required before Spot submission.",
    }
    out_p = ROOT_DIR / "eval_results" / "phase4_cloud_plan.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["1b-download", "2", "2-worker", "3-finops"], required=True)
    parser.add_argument("--private-root", type=Path, default=ROOT_DIR.parent / "NeuraChess-Private")
    parser.add_argument("--max-pieces", type=int, default=5)
    parser.add_argument("--worker-spec", type=str, default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    args = parser.parse_args()

    if args.step == "1b-download":
        res = download_syzygy_tables(ROOT_DIR / "weights" / "syzygy", max_pieces=args.max_pieces, include_dtz=True, verify_sha256=True, max_workers=8)
        print(json.dumps(res, indent=2))
    elif args.step == "2":
        run_step2_beast_benchmarks(args.private_root)
    elif args.step == "2-worker":
        spec = json.loads(args.worker_spec)
        _benchmark_single_beast_worker(spec, args.private_root, args.worker_out)
    elif args.step == "3-finops":
        res = run_step3_finops_audit(args.private_root)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
