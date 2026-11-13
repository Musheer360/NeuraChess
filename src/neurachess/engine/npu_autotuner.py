"""
Hardware-Agnostic Intel NPU Auto-Tuner (engine/npu_autotuner.py)
================================================================
Automatically profiles any Intel NPU generation (Meteor Lake / Arrow Lake NPU 3720,
Lunar Lake NPU 4000, Panther Lake NPU 5000+) for a given folded SE-ResNet model.

Features:
  - Queries OpenVINO NPU device properties: FULL_DEVICE_NAME, NPU_DRIVER_VERSION,
    DEVICE_ARCHITECTURE, OPTIMAL_NUMBER_OF_INFER_REQUESTS, and NPU_TURBO support.
  - Caches selected profiles by {device_name, driver_version, model_sha256} in
    weights/npu_profile_cache.json (bypassed with --retune / retune=True).
  - Runs a ~3-second micro-benchmark across B in {8, 16, 32, 64} and nireq in {1, 2, 4}
    measuring raw pos/s, single-batch sequential latency (ms/batch), and async turnaround
    latency (ms/batch).
  - Selects the optimal (batch_size, nireq, turbo) tuple maximizing effective search
    throughput subject to >= 40 sequential batch updates per 400 ms move (<= 10.0 ms
    turnaround latency per batch).
"""

import os
import sys
import time
import json
import hashlib
import logging
import argparse
import numpy as np
import openvino as ov

logger = logging.getLogger(__name__)

from neurachess._paths import find_repo_root

ROOT_DIR = find_repo_root()

from neurachess.models.se_resnet_backbone import build_openvino_se_resnet
from neurachess.models.resnet_backbone import build_openvino_resnet

DEFAULT_PROFILE_CACHE_PATH = os.path.join(ROOT_DIR, "weights", "npu_profile_cache.json")
DEFAULT_OV_BLOB_CACHE_DIR = os.path.join(ROOT_DIR, "weights", "cache")


def compute_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def infer_arch_from_weights(weights_dict: dict[str, np.ndarray]) -> tuple[int, int, int, bool]:
    """Infers (num_blocks, channels, input_planes, use_se) directly from folded .npz weights."""
    init_w = weights_dict["init_conv_w"]
    channels = int(init_w.shape[0])
    input_planes = int(init_w.shape[1])
    block_indices = []
    use_se = False
    for k in weights_dict.keys():
        if (k.startswith("res_") or k.startswith("block_")) and k.endswith("_conv1_w"):
            idx = int(k.split("_")[1])
            block_indices.append(idx)
        if "_se_sq_w" in k or "_se_squeeze" in k:
            use_se = True
    num_blocks = (max(block_indices) + 1) if block_indices else 6
    return num_blocks, channels, input_planes, use_se


def classify_npu_generation(device_name: str, device_arch: str) -> str:
    arch = str(device_arch).strip()
    if "3720" in arch:
        return "Meteor Lake / Arrow Lake (NPU 3720)"
    if "4000" in arch or "40" in arch[:2]:
        return "Lunar Lake (NPU 4000)"
    if "5000" in arch or "50" in arch[:2]:
        return "Panther Lake (NPU 5000)"
    return f"Intel NPU ({device_name}, arch={arch})"


def query_npu_capabilities(core: ov.Core | None = None, device: str = "NPU") -> dict:
    """Queries OpenVINO Core for hardware-agnostic NPU properties and turbo support."""
    if core is None:
        core = ov.Core()

    available = list(core.available_devices)
    if device not in available:
        raise RuntimeError(f"Requested device '{device}' not found in available OpenVINO devices: {available}")

    try:
        supported = list(core.get_property(device, "SUPPORTED_PROPERTIES"))
    except (RuntimeError, ValueError, TypeError):
        logger.debug("Querying SUPPORTED_PROPERTIES for %s failed", device, exc_info=True)
        supported = []

    def _safe_prop(prop_name: str, default=None):
        if supported and prop_name not in supported:
            return default
        try:
            return core.get_property(device, prop_name)
        except (RuntimeError, ValueError, TypeError, KeyError):
            logger.debug("Querying OpenVINO property %s failed", prop_name, exc_info=True)
            return default

    device_name = str(_safe_prop("FULL_DEVICE_NAME") or _safe_prop("DEVICE_NAME") or device)
    driver_version = str(_safe_prop("NPU_DRIVER_VERSION") or "unknown")
    device_arch = str(_safe_prop("DEVICE_ARCHITECTURE") or "unknown")
    opt_nireq = int(_safe_prop("OPTIMAL_NUMBER_OF_INFER_REQUESTS") or 1)
    range_async = _safe_prop("RANGE_FOR_ASYNC_INFER_REQUESTS")

    supports_turbo = "NPU_TURBO" in supported
    if supports_turbo:
        try:
            core.set_property(device, {"NPU_TURBO": True})
        except (RuntimeError, ValueError):
            logger.debug("Setting NPU_TURBO=True during capability probe failed", exc_info=True)
            supports_turbo = False

    return {
        "device": device,
        "device_name": device_name,
        "driver_version": driver_version,
        "device_architecture": device_arch,
        "npu_generation": classify_npu_generation(device_name, device_arch),
        "optimal_number_of_infer_requests": opt_nireq,
        "range_for_async_infer_requests": list(range_async) if range_async is not None else None,
        "supports_npu_turbo": supports_turbo,
    }


def _make_cache_key(device_name: str, driver_version: str, model_sha256: str) -> str:
    return f"{device_name}|{driver_version}|{model_sha256}"


def load_cached_profile(
    device_name: str,
    driver_version: str,
    model_sha256: str,
    cache_path: str = DEFAULT_PROFILE_CACHE_PATH,
) -> dict | None:
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        profiles = data.get("profiles", {})
        key = _make_cache_key(device_name, driver_version, model_sha256)
        return profiles.get(key)
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        logger.debug("Failed reading NPU profile cache at %s", cache_path, exc_info=True)
        return None


def save_cached_profile(profile: dict, cache_path: str = DEFAULT_PROFILE_CACHE_PATH):
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    data = {"schema_version": 1, "profiles": {}}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "profiles" in loaded:
                    data = loaded
        except (OSError, json.JSONDecodeError, TypeError):
            logger.debug("Starting fresh NPU profile cache; existing %s unreadable", cache_path, exc_info=True)

    key = _make_cache_key(
        profile["key"]["device_name"],
        profile["key"]["driver_version"],
        profile["key"]["model_sha256"],
    )
    data["profiles"][key] = profile
    data["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _benchmark_single_config(
    compiled_model: ov.CompiledModel,
    batch_size: int,
    input_planes: int,
    nireq: int,
    duration_sec: float,
) -> dict:
    """
    Benchmarks a compiled NPU model for `duration_sec` at `(batch_size, nireq)`.
    Measures:
      - seq_latency_ms: single-batch synchronous latency (`nireq == 1`) or async turnaround
        latency (`t_complete - t_dispatch`) for an individual batch through the `nireq` queue.
      - cadence_ms: average inter-batch completion interval (`1000 / batches_per_sec`).
      - raw_pos_per_sec: total positions evaluated per second (`batches * B / elapsed`).
      - effective_pos_per_sec: search-effective positions/s accounting for PUCT virtual-loss
        batch dilution (`B / (1 + 0.005 * max(0, B - 8))` scaling).
    """
    rng = np.random.default_rng(42 + batch_size * 10 + nireq)
    dummy_input = rng.standard_normal((batch_size, input_planes, 8, 8), dtype=np.float32)

    if nireq <= 1:
        req = compiled_model.create_infer_request()
        # Warmup
        for _ in range(3):
            req.infer({0: dummy_input})

        latencies_ms = []
        t_start = time.perf_counter()
        t_end = t_start + duration_sec
        batches = 0
        while time.perf_counter() < t_end or batches < 4:
            t0 = time.perf_counter()
            req.infer({0: dummy_input})
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            batches += 1
        elapsed = max(1e-6, time.perf_counter() - t_start)
        seq_ms = float(np.median(latencies_ms))
        cadence_ms = (elapsed * 1000.0) / batches
    else:
        reqs = [compiled_model.create_infer_request() for _ in range(nireq)]
        # Warmup
        for r in reqs:
            r.start_async({0: dummy_input})
        for r in reqs:
            r.wait()

        dispatch_times = [0.0] * nireq
        turnaround_ms = []
        t_start = time.perf_counter()
        t_end = t_start + duration_sec

        # Prime async ring buffer
        for slot in range(nireq):
            dispatch_times[slot] = time.perf_counter()
            reqs[slot].start_async({0: dummy_input})

        batches = 0
        slot = 0
        while time.perf_counter() < t_end or batches < (2 * nireq):
            reqs[slot].wait()
            t_done = time.perf_counter()
            turnaround_ms.append((t_done - dispatch_times[slot]) * 1000.0)
            batches += 1
            if t_done < t_end or batches < (2 * nireq):
                dispatch_times[slot] = time.perf_counter()
                reqs[slot].start_async({0: dummy_input})
            slot = (slot + 1) % nireq

        # Drain remaining in-flight requests cleanly
        for r in reqs:
            try:
                r.wait()
            except RuntimeError:
                logger.debug("Async InferRequest wait during drain raised RuntimeError", exc_info=True)

        elapsed = max(1e-6, time.perf_counter() - t_start)
        seq_ms = float(np.median(turnaround_ms))
        cadence_ms = (elapsed * 1000.0) / max(1, batches)

    raw_pos_s = (batches * batch_size) / elapsed
    seq_batches_400ms = 400.0 / max(0.05, seq_ms)
    # Effective search throughput: raw NPU throughput for depth-compliant configs (>= 40 seq batches / 400ms),
    # scaled down proportionally if sequential depth falls below the 40-wave search horizon floor.
    depth_factor = min(1.0, seq_batches_400ms / 40.0)
    effective_pos_s = raw_pos_s * depth_factor

    return {
        "batch_size": batch_size,
        "nireq": nireq,
        "batches_measured": batches,
        "seq_batch_latency_ms": round(seq_ms, 3),
        "pipeline_cadence_ms": round(cadence_ms, 3),
        "seq_batches_per_400ms": round(seq_batches_400ms, 2),
        "meets_depth_floor_40": bool(seq_batches_400ms >= 40.0),
        "raw_pos_per_sec": round(raw_pos_s, 1),
        "effective_pos_per_sec": round(effective_pos_s, 1),
    }


def autotune_npu(
    weights_path: str,
    device: str = "NPU",
    batch_sizes: tuple[int, ...] = (8, 16, 32, 64),
    nireq_values: tuple[int, ...] = (1, 2, 4),
    benchmark_budget_sec: float = 3.0,
    min_seq_batches_per_400ms: float = 40.0,
    movetime_ms: float = 400.0,
    retune: bool = False,
    sweep_turbo: bool = False,
    cache_path: str = DEFAULT_PROFILE_CACHE_PATH,
    ov_cache_dir: str = DEFAULT_OV_BLOB_CACHE_DIR,
    verbose: bool = True,
) -> dict:
    """
    Returns the optimal (batch_size, nireq, turbo) NPU configuration for `weights_path`.
    Loads instantly from `weights/npu_profile_cache.json` unless missing or `retune=True`.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    core = ov.Core()
    caps = query_npu_capabilities(core, device=device)
    model_sha256 = compute_file_sha256(weights_path)

    if not retune:
        cached = load_cached_profile(
            device_name=caps["device_name"],
            driver_version=caps["driver_version"],
            model_sha256=model_sha256,
            cache_path=cache_path,
        )
        if cached is not None:
            if verbose:
                sel = cached["selected"]
                print(
                    f"[NPU AutoTuner Cache Hit] {caps['device_name']} (drv {caps['driver_version']}) | "
                    f"{cached['model']['arch']} -> B={sel['batch_size']}, nireq={sel['nireq']}, "
                    f"turbo={sel['turbo']} ({sel['raw_pos_per_sec']} raw pos/s, {sel['seq_batch_latency_ms']} ms/batch)"
                )
            return cached

    npz = np.load(weights_path)
    weights_dict = {k: npz[k] for k in npz.files}
    num_blocks, channels, input_planes, use_se = infer_arch_from_weights(weights_dict)
    arch_str = f"{num_blocks}x{channels}"

    os.makedirs(ov_cache_dir, exist_ok=True)
    core.set_property(device, {"CACHE_DIR": ov_cache_dir})
    core.set_property(device, {"INFERENCE_PRECISION_HINT": ov.Type.f16})
    core.set_property(device, {"PERFORMANCE_HINT": ov.properties.hint.PerformanceMode.THROUGHPUT})

    if caps["supports_npu_turbo"]:
        turbo_candidates = [True, False] if sweep_turbo else [True]
    else:
        turbo_candidates = [False]

    total_points = max(1, len(turbo_candidates) * len(batch_sizes) * len(nireq_values))
    per_point_sec = max(0.12, benchmark_budget_sec / total_points)

    if verbose:
        print("=" * 78)
        print(f"[NPU AutoTuner] Profiling {arch_str} ({os.path.basename(weights_path)})")
        print(f"  Device: {caps['device_name']} | Gen: {caps['npu_generation']} | Driver: {caps['driver_version']}")
        print(f"  OV Optimal NIREQ Hint: {caps['optimal_number_of_infer_requests']} | Turbo Supported: {caps['supports_npu_turbo']}")
        print(f"  Grid: B={list(batch_sizes)} x nireq={list(nireq_values)} x turbo={turbo_candidates} (~{benchmark_budget_sec:.1f}s compute budget)")
        print("=" * 78)

    grid_results = []
    t_bench_total_0 = time.perf_counter()

    for turbo_val in turbo_candidates:
        if caps["supports_npu_turbo"]:
            try:
                core.set_property(device, {"NPU_TURBO": bool(turbo_val)})
            except (RuntimeError, ValueError):
                logger.debug("Setting NPU_TURBO=%s skipped", turbo_val, exc_info=True)

        for b_sz in batch_sizes:
            if use_se:
                ov_model = build_openvino_se_resnet(
                    batch_size=b_sz,
                    num_blocks=num_blocks,
                    channels=channels,
                    in_channels=input_planes,
                    weights_dict=weights_dict,
                )
            else:
                ov_model = build_openvino_resnet(
                    batch_size=b_sz,
                    num_blocks=num_blocks,
                    channels=channels,
                    weights_dict=weights_dict,
                )

            t_comp0 = time.perf_counter()
            compiled = core.compile_model(ov_model, device)
            comp_ms = (time.perf_counter() - t_comp0) * 1000.0
            try:
                compiled_opt_nireq = int(compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS"))
            except (RuntimeError, ValueError, TypeError):
                logger.debug("Compiled model OPTIMAL_NUMBER_OF_INFER_REQUESTS query unavailable", exc_info=True)
                compiled_opt_nireq = caps["optimal_number_of_infer_requests"]

            for nireq in nireq_values:
                res = _benchmark_single_config(
                    compiled_model=compiled,
                    batch_size=b_sz,
                    input_planes=input_planes,
                    nireq=nireq,
                    duration_sec=per_point_sec,
                )
                res["turbo"] = bool(turbo_val)
                res["compile_ms"] = round(comp_ms, 1)
                res["compiled_optimal_nireq"] = compiled_opt_nireq
                seq_batches = movetime_ms / max(0.05, res["seq_batch_latency_ms"])
                res["seq_batches_per_move"] = round(seq_batches, 2)
                res["meets_depth_constraint"] = bool(seq_batches >= min_seq_batches_per_400ms)
                grid_results.append(res)

                if verbose:
                    flag = "PASS" if res["meets_depth_constraint"] else "STARVED"
                    print(
                        f"  B={b_sz:2d} | nireq={nireq} | turbo={str(turbo_val):5s} -> "
                        f"raw: {res['raw_pos_per_sec']:7.1f} pos/s | eff: {res['effective_pos_per_sec']:7.1f} pos/s | "
                        f"lat: {res['seq_batch_latency_ms']:6.2f} ms/batch ({res['seq_batches_per_move']:5.1f} seq/400ms) [{flag}]"
                    )

    total_elapsed_s = time.perf_counter() - t_bench_total_0

    viable = [r for r in grid_results if r["meets_depth_constraint"]]
    if viable:
        best = max(viable, key=lambda r: (r["effective_pos_per_sec"], r["raw_pos_per_sec"], -r["nireq"]))
        selection_reason = f"Maximized effective_pos_per_sec among {len(viable)} configs with >= {min_seq_batches_per_400ms:.0f} sequential batches per {movetime_ms:.0f}ms"
    else:
        best = max(grid_results, key=lambda r: (r["seq_batches_per_move"], r["effective_pos_per_sec"]))
        selection_reason = f"Fallback: no config reached {min_seq_batches_per_400ms:.0f} sequential batches/{movetime_ms:.0f}ms; chose highest sequential depth"

    profile = {
        "key": {
            "device_name": caps["device_name"],
            "driver_version": caps["driver_version"],
            "model_sha256": model_sha256,
        },
        "hardware": caps,
        "model": {
            "weights_path": os.path.abspath(weights_path),
            "weights_basename": os.path.basename(weights_path),
            "arch": arch_str,
            "num_blocks": num_blocks,
            "channels": channels,
            "input_planes": input_planes,
            "use_se": use_se,
        },
        "constraints": {
            "movetime_ms": movetime_ms,
            "min_seq_batches_per_move": min_seq_batches_per_400ms,
            "max_seq_batch_latency_ms": round(movetime_ms / min_seq_batches_per_400ms, 3),
        },
        "selected": {
            "batch_size": best["batch_size"],
            "nireq": best["nireq"],
            "turbo": best["turbo"],
            "raw_pos_per_sec": best["raw_pos_per_sec"],
            "effective_pos_per_sec": best["effective_pos_per_sec"],
            "seq_batch_latency_ms": best["seq_batch_latency_ms"],
            "pipeline_cadence_ms": best["pipeline_cadence_ms"],
            "seq_batches_per_400ms": best["seq_batches_per_400ms"],
            "selection_reason": selection_reason,
        },
        "grid_results": grid_results,
        "tuning_wall_sec": round(total_elapsed_s, 3),
        "tuned_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    save_cached_profile(profile, cache_path=cache_path)
    if verbose:
        sel = profile["selected"]
        print("-" * 78)
        print(
            f"[NPU AutoTuner Selected] {arch_str}: batch_size={sel['batch_size']}, "
            f"nireq={sel['nireq']}, turbo={sel['turbo']} -> "
            f"{sel['raw_pos_per_sec']} raw pos/s ({sel['seq_batch_latency_ms']} ms/batch, "
            f"{sel['seq_batches_per_400ms']} seq batches/400ms)"
        )
        print(f"[Saved Profile Cache] {cache_path}")
        print("=" * 78)

    return profile


def main():
    parser = argparse.ArgumentParser(description="Hardware-Agnostic Intel NPU Auto-Tuner for NeuraChess")
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        default=[
            os.path.join(ROOT_DIR, "weights", "neurachess-10x192-fp16.npz"),
            os.path.join(ROOT_DIR, "weights", "neurachess-10x128-fp16.npz"),
        ],
        help="One or more folded .npz weight files to profile.",
    )
    parser.add_argument("--device", type=str, default="NPU")
    parser.add_argument("--budget-sec", type=float, default=3.0)
    parser.add_argument("--retune", action="store_true", help="Force re-running the benchmark even if cached.")
    parser.add_argument("--sweep-turbo", action="store_true", help="Benchmark both NPU_TURBO=True and False.")
    parser.add_argument("--cache-path", type=str, default=DEFAULT_PROFILE_CACHE_PATH)
    args = parser.parse_args()

    for wp in args.weights:
        autotune_npu(
            weights_path=wp,
            device=args.device,
            benchmark_budget_sec=args.budget_sec,
            retune=args.retune,
            sweep_turbo=args.sweep_turbo,
            cache_path=args.cache_path,
            verbose=True,
        )


if __name__ == "__main__":
    main()
