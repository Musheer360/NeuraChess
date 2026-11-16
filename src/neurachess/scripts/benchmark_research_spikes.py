"""Local NPU Research Spikes Benchmark Suite (`research/v1.1-spikes`, $0 Cloud Spend).

Executes and logs raw measurements for three local Intel NPU 3720 spikes on `neurachess-10x192-fp16`:
  1. Spike 1: C++23 Bitboard SEE Accuracy (56 cases), Speed (<2 us), and `use_see_prior_shaping`
     on 19 Category (c) Tactical Horizon FENs, 19 Category (a) Value/Search FENs, and 500 Lichess puzzles (50n, 200n).
  2. Spike 2: Dual-Shape Co-Resident NPU Gear-Shifting (`B=4`, `B=8`, `B=16` alongside `B=1`)
     measuring NPU/RSS memory, single-batch latency, sequential ply depth, and solve rates on
     100 Mate-in-3 puzzles & 19 Category (c) Tactical Horizon FENs at 400 ms/move.
  3. Spike 3: Layerwise Quantization Sensitivity Profile (`block_0..block_9`, middle `blocks 2..7` INT8,
     and empirically least-sensitive 6 blocks) via NNCF (`fast_bias_correction=True`), reporting 5,000-pos
     Far-Test Top-1 agreement, Mean Policy KL, Max Value Diff, and in-engine `fast_tree` NPS at B=16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import chess
import numpy as np
import openvino as ov
import psutil

os.environ.setdefault("NNCF_TELEM_OPT_OUT", "1")
os.environ.setdefault("OPENVINO_TELEMETRY_OPTOUT", "1")

from neurachess._paths import find_repo_root
from neurachess.core.constants import uci_to_action
from neurachess.core.tensor_movegen import get_legal_action_mask
from neurachess.engine.fast_tree import bench_see_ns, c_see
from neurachess.models.npu_compiler import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_XML,
    DEFAULT_WEIGHTS_NPZ,
    NPUModel,
)
from neurachess.models.se_resnet_backbone import (
    build_openvino_research_head_variant,
    build_openvino_se_resnet,
)
from neurachess.scripts.evaluate import eval_stm_to_nc_drop, resolve_stockfish_path, unpack_raw_to_gpu_inputs
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch
from neurachess.training.bitboard_dataset import RECORD_SIZE_V1, RECORD_SIZE_V2, unpack_position

ROOT_DIR = find_repo_root()


def assert_ac_power() -> dict:
    batt = psutil.sensors_battery()
    plugged = True if batt is None else bool(batt.power_plugged)
    pct = 100 if batt is None else int(round(batt.percent))
    if not plugged:
        raise RuntimeError(f"ABORT: System on battery power ({pct}%). Connect AC power before NPU benchmarking.")
    return {"power_plugged": plugged, "battery_pct": pct}


class FenStockfishGrader:
    """Thin Stockfish UCI wrapper supporting arbitrary FEN evaluation at fixed depth."""

    def __init__(self, sf_path: str | None = None, depth: int = 14):
        binary = resolve_stockfish_path(sf_path)
        self.depth = int(depth)
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        while "uciok" not in self.proc.stdout.readline():
            pass
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 32")
        self._send("setoption name UCI_LimitStrength value false")
        self._send("isready")
        while "readyok" not in self.proc.stdout.readline():
            pass

    def _send(self, cmd: str) -> None:
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def eval_fen(self, fen: str, move_uci: str | None = None) -> tuple[str | None, int]:
        if move_uci:
            self._send(f"position fen {fen} moves {move_uci}")
        else:
            self._send(f"position fen {fen}")
        self._send(f"go depth {self.depth}")
        best_m = None
        score_cp = 0
        while True:
            line = self.proc.stdout.readline().strip()
            if "score cp" in line:
                parts = line.split()
                if "cp" in parts:
                    score_cp = int(parts[parts.index("cp") + 1])
            elif "score mate" in line:
                parts = line.split()
                if "mate" in parts:
                    mv = int(parts[parts.index("mate") + 1])
                    score_cp = 30000 if mv > 0 else -30000
            if line.startswith("bestmove"):
                parts = line.split()
                best_m = parts[1] if len(parts) > 1 else None
                break
        return best_m, score_cp

    def eval_move_drop(self, fen: str, move_uci: str) -> dict:
        sf_best, ev_before = self.eval_fen(fen, None)
        _, ev_after_stm = self.eval_fen(fen, move_uci)
        b_cp, a_cp, drop_cp = eval_stm_to_nc_drop(ev_before, ev_after_stm)
        return {
            "sf_best_d14": sf_best,
            "eval_before_cp": b_cp,
            "eval_after_cp": a_cp,
            "eval_drop_cp": drop_cp,
        }

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.terminate()
            except Exception:
                pass


def load_loss_anatomy_subsets(private_root: Path) -> tuple[list[dict], list[dict]]:
    path = private_root / "eval_results" / "step4_loss_anatomy.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    cat_a = []
    cat_c = []
    for entry in data["losses"]:
        ba = entry["blunder_analysis"]
        item = {
            "elo": entry["elo"],
            "game_id": entry["game_id"],
            "nc_color": entry["nc_color"],
            "ply": ba["ply"],
            "fen": ba["fen"],
            "orig_nc_move": ba["nc_move"],
            "orig_sf_best_move": ba["sf_best_move"],
            "orig_eval_drop_cp": ba["eval_drop_cp"],
            "orig_category": ba["category"],
            "game_phase": ba.get("game_phase", ""),
        }
        if ba["category"] == "a_search_value_error":
            cat_a.append(item)
        elif ba["category"] == "c_tactical_horizon_hung_material":
            cat_c.append(item)
    return cat_c, cat_a


def load_clean_500_puzzles(private_root: Path) -> list[dict]:
    audit_p = private_root / "eval_results" / "shards_000_039_contamination_audit.json"
    contam = set()
    if audit_p.exists():
        audit = json.loads(audit_p.read_text(encoding="utf-8"))
        contam = {h["puzzle_id"] for h in audit.get("findings", {}).get("puzzle_hits", [])}
    strat_p = private_root / "scoreboard" / "puzzles_2000_stratified.json"
    data = json.loads(strat_p.read_text(encoding="utf-8"))
    out: list[dict] = []
    for bucket, plist in data.items():
        clean_b = [p for p in plist if p["puzzle_id"] not in contam]
        out.extend(clean_b[:100])
    assert len(out) == 500, f"Expected 500 clean stratified puzzles, got {len(out)}"
    return out


def load_mate_in_3_100(private_root: Path) -> list[dict]:
    mate_p = private_root / "eval_results" / "puzzles_mate_suite.json"
    data = json.loads(mate_p.read_text(encoding="utf-8"))
    puzzles = data["mateIn3"][:100]
    assert len(puzzles) == 100
    return puzzles


def _save_report(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =============================================================================
# SPIKE 1: C++23 Bitboard SEE in PUCT
# =============================================================================
def _eval_loss_fens(
    searcher: NPUBatchedPUCTSearch,
    grader: FenStockfishGrader,
    items: list[dict],
    max_nodes: int | None = None,
    movetime_ms: int = 2000,
) -> dict:
    sf_exact_orig = 0
    sf_exact_live = 0
    avoided_orig_blunder = 0
    blunders_ge150 = 0
    blunders_ge100 = 0
    drops: list[int] = []
    cpu_ms_list: list[float] = []
    max_ply_list: list[int] = []
    avg_ply_list: list[float] = []
    batches_list: list[int] = []
    nodes_list: list[int] = []
    gear_counts: dict[str, int] = {}
    gear_reasons: dict[str, int] = {}
    per_fen = []

    for item in items:
        fen = item["fen"]
        board = chess.Board(fen)
        searcher.clear_state()
        t0 = time.perf_counter()
        _, uci, best_eval, nodes, pv, max_d = searcher.search(
            board, movetime_ms=movetime_ms, max_nodes=max_nodes
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        prof = searcher.last_profile or {}

        sf_res = grader.eval_move_drop(fen, uci)
        drop_cp = int(sf_res["eval_drop_cp"])
        drops.append(drop_cp)
        if uci == item["orig_sf_best_move"]:
            sf_exact_orig += 1
        if uci == sf_res["sf_best_d14"]:
            sf_exact_live += 1
        if uci != item["orig_nc_move"]:
            avoided_orig_blunder += 1
        if drop_cp >= 150:
            blunders_ge150 += 1
        if drop_cp >= 100:
            blunders_ge100 += 1

        cpu_ms = float(
            prof.get("select_s", 0.0)
            + prof.get("pack_s", 0.0)
            + prof.get("dispatch_s", 0.0)
            + prof.get("backup_s", 0.0)
            + prof.get("pre_loop_s", 0.0)
        ) * 1000.0
        cpu_ms_list.append(cpu_ms)
        max_ply_list.append(int(prof.get("max_leaf_ply", max_d)))
        avg_ply_list.append(float(prof.get("avg_leaf_ply", 0.0)))
        batches_list.append(int(prof.get("batches", 0)))
        nodes_list.append(int(nodes))

        for k, v in (prof.get("batch_gear_counts") or {}).items():
            gear_counts[k] = gear_counts.get(k, 0) + int(v)
        for k, v in (prof.get("gear_reasons") or {}).items():
            gear_reasons[k] = gear_reasons.get(k, 0) + int(v)

        per_fen.append(
            {
                "elo": item["elo"],
                "ply": item["ply"],
                "fen": fen,
                "played": uci,
                "orig_nc_move": item["orig_nc_move"],
                "orig_sf_best": item["orig_sf_best_move"],
                "live_sf_best_d14": sf_res["sf_best_d14"],
                "eval_drop_cp": drop_cp,
                "nodes": int(nodes),
                "batches": int(prof.get("batches", 0)),
                "max_leaf_ply": int(prof.get("max_leaf_ply", max_d)),
                "avg_leaf_ply": float(prof.get("avg_leaf_ply", 0.0)),
                "wall_ms": round(wall_ms, 2),
                "cpu_ms": round(cpu_ms, 2),
            }
        )

    n = max(1, len(items))
    total_gears = sum(gear_counts.values())
    batch_share_pct = (
        {k: round(100.0 * v / total_gears, 2) for k, v in sorted(gear_counts.items())}
        if total_gears > 0
        else {}
    )
    return {
        "total": len(items),
        "orig_sf_best_matches": sf_exact_orig,
        "live_sf_d14_matches": sf_exact_live,
        "avoided_orig_nc_blunder": avoided_orig_blunder,
        "blunders_ge_150cp": blunders_ge150,
        "blunders_ge_100cp": blunders_ge100,
        "mean_cp_drop": round(sum(drops) / n, 2),
        "median_cp_drop": round(float(statistics.median(drops)), 2),
        "total_nodes": sum(nodes_list),
        "mean_nodes": round(sum(nodes_list) / n, 2),
        "total_batches": total_gears if total_gears > 0 else sum(batches_list),
        "mean_batches": round(sum(batches_list) / n, 2),
        "mean_max_ply": round(sum(max_ply_list) / n, 3),
        "global_max_ply": max(max_ply_list) if max_ply_list else 0,
        "mean_avg_ply": round(sum(avg_ply_list) / n, 3),
        "mean_cpu_ms": round(sum(cpu_ms_list) / n, 3),
        "gear_counts": gear_counts,
        "batch_share_pct": batch_share_pct,
        "gear_reasons": gear_reasons,
        "per_fen": per_fen,
    }


def _eval_500_puzzles(
    searcher: NPUBatchedPUCTSearch,
    puzzles: list[dict],
    max_nodes: int | None = None,
    movetime_ms: int | None = None,
) -> dict:
    solved = 0
    by_bucket: dict[str, dict[str, int | float]] = {}
    cpu_ms_sum = 0.0
    total_nodes = 0
    total_batches = 0
    max_plies: list[int] = []
    avg_plies: list[float] = []
    gear_counts: dict[str, int] = {}
    gear_reasons: dict[str, int] = {}
    search_wall_s = 0.0
    eff_movetime = int(movetime_ms) if movetime_ms is not None else max(1000, int(max_nodes or 200) * 4)
    t0 = time.perf_counter()

    for p in puzzles:
        bname = p["bucket"]
        if bname not in by_bucket:
            by_bucket[bname] = {"solved": 0, "total": 0}
        by_bucket[bname]["total"] += 1

        board = chess.Board(p["fen"])
        sol = p["solution_move"]
        searcher.clear_state()
        ts0 = time.perf_counter()
        _, uci, _, nodes, pv, max_d = searcher.search(board, movetime_ms=eff_movetime, max_nodes=max_nodes)
        search_wall_s += time.perf_counter() - ts0
        prof = searcher.last_profile or {}
        cpu_ms_sum += float(
            prof.get("select_s", 0.0)
            + prof.get("pack_s", 0.0)
            + prof.get("dispatch_s", 0.0)
            + prof.get("backup_s", 0.0)
            + prof.get("pre_loop_s", 0.0)
        ) * 1000.0
        total_nodes += int(nodes)
        total_batches += int(prof.get("batches", 0))
        max_plies.append(int(prof.get("max_leaf_ply", max_d)))
        avg_plies.append(float(prof.get("avg_leaf_ply", 0.0)))
        for k, v in (prof.get("batch_gear_counts") or {}).items():
            gear_counts[k] = gear_counts.get(k, 0) + int(v)
        for k, v in (prof.get("gear_reasons") or {}).items():
            gear_reasons[k] = gear_reasons.get(k, 0) + int(v)

        ok = (uci == sol) or (len(pv) > 0 and pv[0] == sol)
        if ok:
            solved += 1
            by_bucket[bname]["solved"] += 1

    dt = time.perf_counter() - t0
    n = max(1, len(puzzles))
    total_gears = sum(gear_counts.values())
    batch_share_pct = (
        {k: round(100.0 * v / total_gears, 2) for k, v in sorted(gear_counts.items())}
        if total_gears > 0
        else {}
    )
    for bname, bs in by_bucket.items():
        bs["accuracy_pct"] = round(100.0 * bs["solved"] / max(1, bs["total"]), 2)
    return {
        "solved": solved,
        "total": len(puzzles),
        "accuracy_pct": round(100.0 * solved / n, 2),
        "elapsed_s": round(dt, 2),
        "search_wall_s": round(search_wall_s, 2),
        "total_nodes": total_nodes,
        "mean_nodes": round(total_nodes / n, 2),
        "total_batches": total_batches,
        "mean_batches": round(total_batches / n, 2),
        "nps_search": round(total_nodes / max(1e-6, search_wall_s), 1),
        "nps_wall": round(total_nodes / max(1e-6, dt), 1),
        "mean_avg_ply": round(sum(avg_plies) / n, 3),
        "mean_max_ply": round(sum(max_plies) / n, 3),
        "global_max_ply": max(max_plies) if max_plies else 0,
        "mean_cpu_ms_per_puzzle": round(cpu_ms_sum / n, 3),
        "gear_counts": gear_counts,
        "batch_share_pct": batch_share_pct,
        "gear_reasons": gear_reasons,
        "by_bucket": by_bucket,
    }


def run_spike_1(private_root: Path, sf_path: str | None) -> dict:
    print("\n" + "=" * 80, flush=True)
    print("SPIKE 1: C++23 BITBOARD SEE + `use_see_prior_shaping` IN PUCT (NPU 10x192 FP16)", flush=True)
    print("=" * 80, flush=True)
    assert_ac_power()

    from tests.test_see_and_shaping import SEE_CASES

    see_ok = 0
    for _, fen, mv, exp in SEE_CASES:
        if c_see(fen, mv) == exp:
            see_ok += 1
    ns_per_see = bench_see_ns("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "e5f7", 200_000)
    print(f"[Spike 1] SEE unit accuracy: {see_ok}/{len(SEE_CASES)} | Latency: {ns_per_see:.1f} ns/call", flush=True)

    root_m = NPUModel(batch_size=1, cache_dir=DEFAULT_CACHE_DIR, device="NPU", use_turbo=True, performance_hint="LATENCY", ov_ir_xml_path=DEFAULT_MODEL_XML, verbose=False)
    batch_m = NPUModel(batch_size=16, cache_dir=DEFAULT_CACHE_DIR, device="NPU", use_turbo=True, performance_hint="LATENCY", ov_ir_xml_path=DEFAULT_MODEL_XML, verbose=False)

    # Warmup
    warm = NPUBatchedPUCTSearch(root_model=root_m, batch_model=batch_m, batch_size=16, use_fast_tree=True, use_tree_reuse=False)
    warm.search(chess.Board(), movetime_ms=100, max_nodes=32)

    cat_c, cat_a = load_loss_anatomy_subsets(private_root)
    puzzles_500 = load_clean_500_puzzles(private_root)
    grader = FenStockfishGrader(sf_path, depth=14)

    try:
        res: dict = {
            "see_verification": {
                "cases_passed": see_ok,
                "cases_total": len(SEE_CASES),
                "latency_ns_per_call": round(ns_per_see, 2),
                "latency_us_per_call": round(ns_per_see / 1000.0, 4),
            },
            "loss_fens_cat_c_19": {},
            "loss_fens_cat_a_19": {},
            "lichess_500_puzzles": {},
        }

        for arm_name, see_flag in [("baseline_off", False), ("see_shaping_on", True)]:
            searcher = NPUBatchedPUCTSearch(
                root_model=root_m,
                batch_model=batch_m,
                batch_size=16,
                c_puct=1.5,
                use_fpu_reduction=True,
                fpu_reduction=0.20,
                root_fpu_reduction=0.10,
                t_root=1.0,
                use_exact_solver=True,
                use_nn_cache=True,
                use_tree_reuse=False,
                use_fast_tree=True,
                use_see_prior_shaping=see_flag,
            )

            res["loss_fens_cat_c_19"][arm_name] = {}
            res["loss_fens_cat_a_19"][arm_name] = {}
            res["lichess_500_puzzles"][arm_name] = {}

            for budget in (50, 200):
                assert_ac_power()
                rc = _eval_loss_fens(searcher, grader, cat_c, max_nodes=budget, movetime_ms=2500)
                print(
                    f"[Spike 1] Cat(c) 19 FENs | {arm_name:14s} @ {budget:3d}n: "
                    f"SF_best={rc['orig_sf_best_matches']}/19 | AvoidedNC={rc['avoided_orig_nc_blunder']}/19 | "
                    f"Blunders>=150cp={rc['blunders_ge_150cp']}/19 | MeanDrop={rc['mean_cp_drop']:.1f}cp",
                    flush=True,
                )
                rc_summary = {k: v for k, v in rc.items() if k != "per_fen"}
                rc_summary["per_fen"] = rc["per_fen"]
                res["loss_fens_cat_c_19"][arm_name][f"nodes_{budget}"] = rc_summary

                ra = _eval_loss_fens(searcher, grader, cat_a, max_nodes=budget, movetime_ms=2500)
                print(
                    f"[Spike 1] Cat(a) 19 FENs | {arm_name:14s} @ {budget:3d}n: "
                    f"SF_best={ra['orig_sf_best_matches']}/19 | AvoidedNC={ra['avoided_orig_nc_blunder']}/19 | "
                    f"Blunders>=150cp={ra['blunders_ge_150cp']}/19 | MeanDrop={ra['mean_cp_drop']:.1f}cp",
                    flush=True,
                )
                ra_summary = {k: v for k, v in ra.items() if k != "per_fen"}
                ra_summary["per_fen"] = ra["per_fen"]
                res["loss_fens_cat_a_19"][arm_name][f"nodes_{budget}"] = ra_summary

                rp = _eval_500_puzzles(searcher, puzzles_500, max_nodes=budget)
                print(
                    f"[Spike 1] 500 Puzzles    | {arm_name:14s} @ {budget:3d}n: "
                    f"{rp['solved']}/500 ({rp['accuracy_pct']:.2f}%) | CPU={rp['mean_cpu_ms_per_puzzle']:.2f}ms/pz | {rp['elapsed_s']:.1f}s",
                    flush=True,
                )
                res["lichess_500_puzzles"][arm_name][f"nodes_{budget}"] = rp

        return res
    finally:
        grader.close()


# =============================================================================
# SPIKE 2: Dual-Shape Co-Resident NPU Gear-Shifting
# =============================================================================
def _eval_mate_in_3_400ms(searcher: NPUBatchedPUCTSearch, mate_puzzles: list[dict], movetime_ms: int = 400) -> dict:
    solved = 0
    max_plies: list[int] = []
    avg_plies: list[float] = []
    batches_l: list[int] = []
    nodes_l: list[int] = []
    gears: dict[str, int] = {}
    reasons: dict[str, int] = {}
    t0 = time.perf_counter()

    for p in mate_puzzles:
        board = chess.Board(p["fen"])
        board.push(chess.Move.from_uci(p["moves"][0]))
        expected = p["moves"][1]

        searcher.clear_state()
        _, uci, _, nodes, pv, max_d = searcher.search(board, movetime_ms=movetime_ms, max_depth=18)
        prof = searcher.last_profile or {}

        ok = (uci == expected) or (len(pv) > 0 and pv[0] == expected)
        if ok:
            solved += 1

        max_plies.append(int(prof.get("max_leaf_ply", max_d)))
        avg_plies.append(float(prof.get("avg_leaf_ply", 0.0)))
        batches_l.append(int(prof.get("batches", 0)))
        nodes_l.append(int(nodes))
        for k, v in (prof.get("batch_gear_counts") or {}).items():
            gears[k] = gears.get(k, 0) + int(v)
        for k, v in (prof.get("gear_reasons") or {}).items():
            reasons[k] = reasons.get(k, 0) + int(v)

    n = max(1, len(mate_puzzles))
    return {
        "solved": solved,
        "total": len(mate_puzzles),
        "accuracy_pct": round(100.0 * solved / n, 2),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "mean_max_ply": round(sum(max_plies) / n, 3),
        "global_max_ply": max(max_plies) if max_plies else 0,
        "mean_avg_ply": round(sum(avg_plies) / n, 3),
        "mean_batches_per_move": round(sum(batches_l) / n, 2),
        "mean_nodes_per_move": round(sum(nodes_l) / n, 2),
        "gear_counts": gears,
        "gear_reasons": reasons,
    }


def _make_sync_npu_wrapper(npu_model: NPUModel):
    """Wrap an NPUModel without `infer_request_async` to benchmark pure sequential `B=16` control."""

    class _SyncOnlyProxy:
        def __init__(self, inner: NPUModel):
            self.batch_size = inner.batch_size
            self.device = inner.device
            self.infer_request = inner.infer_request
            self.compiled_model = inner.compiled_model

        def infer(self, x: np.ndarray):
            res = self.infer_request.infer({"board_state": np.ascontiguousarray(x, dtype=np.float32)})
            outs = list(res.values())
            return outs[0], outs[1]

    return _SyncOnlyProxy(npu_model)


def run_spike_2(private_root: Path, sf_path: str | None) -> dict:
    print("\n" + "=" * 80, flush=True)
    print("SPIKE 2: DUAL-SHAPE CO-RESIDENT NPU GEAR-SHIFTING (B=4, B=8, B=16 + B=1)", flush=True)
    print("=" * 80, flush=True)
    assert_ac_power()

    proc = psutil.Process(os.getpid())
    core = ov.Core()
    core.set_property(
        "NPU",
        {
            "CACHE_DIR": DEFAULT_CACHE_DIR,
            "PERFORMANCE_HINT": ov.properties.hint.PerformanceMode.LATENCY,
            "NPU_TURBO": True,
            "INFERENCE_PRECISION_HINT": ov.Type.f16,
        },
    )

    rss_before_mb = proc.memory_info().rss / 1048576.0
    try:
        npu_total_mb = int(core.get_property("NPU", "NPU_DEVICE_TOTAL_MEM_SIZE")) / 1048576.0
        npu_alloc_before_mb = int(core.get_property("NPU", "NPU_DEVICE_ALLOC_MEM_SIZE")) / 1048576.0
    except Exception:
        npu_total_mb = None
        npu_alloc_before_mb = None

    base_ov = core.read_model(DEFAULT_MODEL_XML)
    co_models: dict[int, NPUModel] = {}
    shape_profile: dict[str, dict] = {}

    # Compile B=1, 4, 8, 16 simultaneously and measure latencies WITH ALL 4 CO-RESIDENT
    prev_rss = rss_before_mb
    for b in (1, 4, 8, 16):
        m = NPUModel(
            batch_size=b,
            cache_dir=DEFAULT_CACHE_DIR,
            device="NPU",
            use_turbo=True,
            performance_hint="LATENCY",
            ov_model_override=base_ov,
            verbose=False,
        )
        co_models[b] = m
        now_rss = proc.memory_info().rss / 1048576.0
        try:
            npu_alloc_now = int(core.get_property("NPU", "NPU_DEVICE_ALLOC_MEM_SIZE")) / 1048576.0
        except Exception:
            npu_alloc_now = None
        shape_profile[f"B={b}"] = {
            "batch_size": b,
            "compile_time_s": round(m.compile_time_s, 3),
            "delta_rss_mb": round(now_rss - prev_rss, 2),
            "cumulative_rss_mb": round(now_rss - rss_before_mb, 2),
            "npu_device_alloc_mem_mb": round(npu_alloc_now, 2) if npu_alloc_now is not None else None,
        }
        prev_rss = now_rss

    # Benchmark single-batch turnaround latency for each shape while ALL 4 remain loaded
    for b in (1, 4, 8, 16):
        req = co_models[b].infer_request
        in_arr = req.get_input_tensor().data
        in_arr[...] = 0.0
        for _ in range(10):
            req.infer()
        lats_ms = []
        for _ in range(80):
            t0 = time.perf_counter()
            req.infer()
            lats_ms.append((time.perf_counter() - t0) * 1000.0)
        med = float(np.median(lats_ms))
        shape_profile[f"B={b}"].update(
            {
                "turnaround_median_ms": round(med, 3),
                "turnaround_mean_ms": round(float(np.mean(lats_ms)), 3),
                "turnaround_p10_ms": round(float(np.percentile(lats_ms, 10)), 3),
                "turnaround_p90_ms": round(float(np.percentile(lats_ms, 90)), 3),
                "raw_pos_per_sec": round(b * 1000.0 / max(1e-6, med), 1),
            }
        )
        print(
            f"[Spike 2] Co-resident B={b:2d}: med={med:.3f}ms | raw={shape_profile[f'B={b}']['raw_pos_per_sec']:.1f} pos/s | "
            f"delta_RSS={shape_profile[f'B={b}']['delta_rss_mb']:.1f}MB",
            flush=True,
        )

    memory_summary = {
        "rss_before_mb": round(rss_before_mb, 2),
        "rss_after_all4_mb": round(proc.memory_info().rss / 1048576.0, 2),
        "total_delta_rss_mb": round((proc.memory_info().rss / 1048576.0) - rss_before_mb, 2),
        "npu_total_mem_mb": round(npu_total_mb, 2) if npu_total_mb is not None else None,
        "npu_alloc_before_mb": round(npu_alloc_before_mb, 2) if npu_alloc_before_mb is not None else None,
        "npu_alloc_after_all4_mb": shape_profile["B=16"]["npu_device_alloc_mem_mb"],
    }

    mate_100 = load_mate_in_3_100(private_root)
    cat_c, _ = load_loss_anatomy_subsets(private_root)
    grader = FenStockfishGrader(sf_path, depth=14)

    try:
        arms = {
            "baseline_b16_prod_db": NPUBatchedPUCTSearch(
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
                use_tree_reuse=False,
                use_fast_tree=True,
                use_dynamic_batch_shift=False,
            ),
            "baseline_b16_sync": NPUBatchedPUCTSearch(
                root_model=co_models[1],
                batch_model=_make_sync_npu_wrapper(co_models[16]),
                batch_size=16,
                c_puct=1.5,
                use_fpu_reduction=True,
                fpu_reduction=0.20,
                root_fpu_reduction=0.10,
                t_root=1.0,
                use_exact_solver=True,
                use_nn_cache=True,
                use_tree_reuse=False,
                use_fast_tree=True,
                use_dynamic_batch_shift=False,
            ),
            "gear_shift_b4_b8_b16": NPUBatchedPUCTSearch(
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
                use_tree_reuse=False,
                use_fast_tree=True,
                use_dynamic_batch_shift=True,
                batch_models={4: co_models[4], 8: co_models[8], 16: co_models[16]},
            ),
        }

        mate_res = {}
        cat_c_res = {}
        for name, srch in arms.items():
            assert_ac_power()
            srch.search(chess.Board(), movetime_ms=80, max_nodes=32)

            rm = _eval_mate_in_3_400ms(srch, mate_100, movetime_ms=400)
            mate_res[name] = rm
            print(
                f"[Spike 2] Mate-in-3 100 @ 400ms | {name:22s}: {rm['solved']}/100 ({rm['accuracy_pct']:.2f}%) | "
                f"avg_ply={rm['mean_avg_ply']:.2f} max_ply={rm['mean_max_ply']:.2f} (gmax={rm['global_max_ply']}) | "
                f"batches={rm['mean_batches_per_move']:.1f} gears={rm['gear_counts']}",
                flush=True,
            )

            rc = _eval_loss_fens(srch, grader, cat_c, max_nodes=None, movetime_ms=400)
            cat_c_res[name] = rc
            print(
                f"[Spike 2] Cat(c) 19   @ 400ms | {name:22s}: SF_best={rc['orig_sf_best_matches']}/19 | "
                f"Blunders>=150cp={rc['blunders_ge_150cp']}/19 MeanDrop={rc['mean_cp_drop']:.1f}cp | "
                f"avg_ply={rc['mean_avg_ply']:.2f} max_ply={rc['mean_max_ply']:.2f} (gmax={rc['global_max_ply']}) | "
                f"batches={rc['mean_batches']:.1f} gears={rc['gear_counts']}",
                flush=True,
            )

        return {
            "co_resident_memory": memory_summary,
            "co_resident_shapes": shape_profile,
            "mate_in_3_100_400ms": mate_res,
            "tactical_horizon_cat_c_19_400ms": cat_c_res,
        }
    finally:
        grader.close()


# =============================================================================
# SPIKE 3: Layerwise Quantization Sensitivity Profile (NPU 3720)
# =============================================================================
def _load_2000_calib(private_root: Path) -> np.ndarray:
    shard_paths = [
        private_root / "data" / "sample_shard_000.bin",
        private_root / "data" / "sample_shard_040.bin",
    ]
    rng = np.random.default_rng(42)
    chunks = []
    for sp in shard_paths:
        n_total = sp.stat().st_size // RECORD_SIZE_V1
        chosen = np.sort(rng.choice(n_total, size=1000, replace=False))
        buf = bytearray(1000 * RECORD_SIZE_V1)
        with open(sp, "rb") as f:
            for out_i, rec_i in enumerate(chosen):
                f.seek(int(rec_i) * RECORD_SIZE_V1)
                buf[out_i * RECORD_SIZE_V1 : (out_i + 1) * RECORD_SIZE_V1] = f.read(RECORD_SIZE_V1)
        chunks.append(unpack_raw_to_gpu_inputs(bytes(buf), 1000).numpy().astype(np.float32))
    calib = np.concatenate(chunks, axis=0)
    return np.ascontiguousarray(calib[rng.permutation(calib.shape[0])], dtype=np.float32)


def _load_5000_far_test_with_targets(
    private_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bin_path = private_root / "data" / "test_far_50k.bin"
    raw_bytes = bin_path.read_bytes()[: 5000 * RECORD_SIZE_V1]
    x = unpack_raw_to_gpu_inputs(raw_bytes, 5000).numpy().astype(np.float32)
    masks = np.empty((5000, 4672), dtype=np.bool_)
    acts = np.empty((5000,), dtype=np.int64)
    val_targets = np.empty((5000,), dtype=np.float32)
    cps = np.empty((5000,), dtype=np.int32)
    for i in range(5000):
        rec = unpack_position(raw_bytes[i * RECORD_SIZE_V1 : (i + 1) * RECORD_SIZE_V1])
        masks[i] = get_legal_action_mask(rec["board"]) > 0.5
        acts[i] = int(rec["action_index"])
        cp = int(rec["eval_cp"])
        cps[i] = cp
        val_targets[i] = float(np.tanh(cp * 0.00287823))
    return np.ascontiguousarray(x, dtype=np.float32), masks, acts, val_targets, cps


def _load_5000_far_test(private_root: Path) -> tuple[np.ndarray, np.ndarray]:
    x, masks, _, _, _ = _load_5000_far_test_with_targets(private_root)
    return x, masks


def _compute_ground_truth_metrics(
    logits: np.ndarray,
    vals: np.ndarray,
    masks: np.ndarray,
    acts: np.ndarray,
    val_targets: np.ndarray,
    cps: np.ndarray,
) -> dict:
    """Compute ground-truth Stockfish policy accuracy, masked cross-entropy, and value MSE."""
    n = logits.shape[0]
    masked_logits = np.where(masks, logits.astype(np.float64), -10000.0)
    preds = np.argmax(masked_logits, axis=1)
    top1_ok = int(np.sum(preds == acts))

    top3_idx = np.argpartition(masked_logits, -3, axis=1)[:, -3:]
    top3_ok = int(np.sum(np.any(top3_idx == acts[:, None], axis=1)))

    illegal_targets = int(np.sum(~masks[np.arange(n), acts]))
    max_row = np.max(masked_logits, axis=1, keepdims=True)
    log_z = max_row[:, 0] + np.log(np.sum(np.exp(masked_logits - max_row), axis=1))
    target_logits = masked_logits[np.arange(n), acts]
    ce_sum = float(np.sum(log_z - target_logits))

    v_err = vals.astype(np.float64) - val_targets.astype(np.float64)
    val_mse = float(np.mean(v_err * v_err))

    dec_mask = np.abs(cps) >= 50
    dec_total = int(np.sum(dec_mask))
    dec_ok = int(np.sum(((cps[dec_mask] > 0) & (vals[dec_mask] > 0)) | ((cps[dec_mask] < 0) & (vals[dec_mask] < 0))))

    return {
        "gt_masked_top1_pct": round(100.0 * top1_ok / max(1, n), 2),
        "gt_masked_top3_pct": round(100.0 * top3_ok / max(1, n), 2),
        "gt_masked_policy_ce": round(ce_sum / max(1, n), 4),
        "gt_value_mse": round(val_mse, 5),
        "gt_decisive_sign_pct": round(100.0 * dec_ok / max(1, dec_total), 2),
        "gt_decisive_count": dec_total,
        "gt_illegal_targets": illegal_targets,
    }


def _eval_5000_on_npu_model(npu_m: NPUModel, x_5000: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    req = npu_m.infer_request
    zc_in = req.get_input_tensor().data
    logits_out = np.empty((5000, 4672), dtype=np.float32)
    vals_out = np.empty((5000,), dtype=np.float32)

    zc_in[...] = x_5000[:16]
    for _ in range(5):
        req.infer()

    t_inf0 = time.perf_counter()
    for start in range(0, 5000, 16):
        end = min(5000, start + 16)
        k = end - start
        if k == 16:
            zc_in[...] = x_5000[start:end]
        else:
            zc_in.fill(0.0)
            zc_in[:k] = x_5000[start:end]
        req.infer()
        p_b = req.get_output_tensor(0).data[:k]
        v_b = req.get_output_tensor(1).data[:k]
        logits_out[start:end] = p_b
        vals_out[start:end] = (v_b[:, 0] - v_b[:, 2]) if (v_b.ndim == 2 and v_b.shape[-1] == 3) else v_b.reshape(-1)
    dt = time.perf_counter() - t_inf0
    raw_pos_s = round(5000.0 / max(1e-6, dt), 1)
    return logits_out, vals_out, raw_pos_s


def _eval_5000_on_npu(ov_model: ov.Model, x_5000: np.ndarray, cache_dir: str) -> tuple[np.ndarray, np.ndarray, float]:
    npu_m = NPUModel(
        batch_size=16,
        cache_dir=cache_dir,
        device="NPU",
        use_turbo=True,
        performance_hint="LATENCY",
        ov_model_override=ov_model,
        verbose=False,
    )
    return _eval_5000_on_npu_model(npu_m, x_5000)


def _compute_fidelity(fp_l: np.ndarray, fp_v: np.ndarray, vr_l: np.ndarray, vr_v: np.ndarray, masks: np.ndarray) -> dict:
    agree = 0
    kl_sum = 0.0
    n = fp_l.shape[0]
    for i in range(n):
        m = masks[i]
        lr = fp_l[i, m].astype(np.float64)
        lv = vr_l[i, m].astype(np.float64)
        if lr.size == 0:
            continue
        if int(np.argmax(lr)) == int(np.argmax(lv)):
            agree += 1
        lrs = lr - np.max(lr)
        lvs = lv - np.max(lv)
        log_pr = lrs - np.log(np.sum(np.exp(lrs)))
        log_pv = lvs - np.log(np.sum(np.exp(lvs)))
        pr = np.exp(log_pr)
        kl = float(np.sum(pr * (log_pr - log_pv)))
        kl_sum += max(0.0, kl)
    vdiff = np.abs(fp_v.astype(np.float64) - vr_v.astype(np.float64))
    return {
        "top1_agreement_pct": round(100.0 * agree / n, 2),
        "mean_policy_kl": round(kl_sum / n, 6),
        "max_value_diff": round(float(np.max(vdiff)), 5),
        "mean_value_diff": round(float(np.mean(vdiff)), 5),
    }


def _measure_in_engine_fast_tree_nps_with_models(
    root_m: NPUModel,
    batch_m: NPUModel,
    num_plies: int = 12,
    movetime_ms: int = 400,
) -> dict:
    assert_ac_power()

    def _make():
        return NPUBatchedPUCTSearch(
            root_model=root_m,
            batch_model=batch_m,
            batch_size=16,
            c_puct=1.5,
            use_fpu_reduction=True,
            fpu_reduction=0.20,
            root_fpu_reduction=0.10,
            t_root=1.0,
            use_exact_solver=True,
            use_async_overlap=False,
            use_tree_reuse=True,
            use_nn_cache=True,
            use_fast_tree=True,
        )

    warm = _make()
    wb = chess.Board()
    warm.search(wb, movetime_ms=150)
    wb.push_uci("e2e4")
    warm.search(wb, movetime_ms=150)
    del warm

    sw = _make()
    sb = _make()
    board = chess.Board()
    for u in ("e2e4", "c7c5", "g1f3", "d7d6"):
        board.push_uci(u)

    total_wall = 0.0
    total_added = 0
    total_nodes = 0
    total_cpu = 0.0
    plies = 0
    for _ in range(num_plies):
        if board.is_game_over(claim_draw=True):
            break
        active = sw if board.turn == chess.WHITE else sb
        t0 = time.perf_counter()
        _, uci, _, nodes, _, _ = active.search(board, movetime_ms=movetime_ms)
        total_wall += time.perf_counter() - t0
        p = active.last_profile or {}
        total_added += int(p.get("added_root_visits", nodes))
        total_nodes += int(nodes)
        total_cpu += float(
            p.get("select_s", 0.0)
            + p.get("pack_s", 0.0)
            + p.get("dispatch_s", 0.0)
            + p.get("backup_s", 0.0)
            + p.get("pre_loop_s", 0.0)
        )
        plies += 1
        board.push_uci(uci)

    return {
        "plies": plies,
        "in_engine_added_visits_s": round(total_added / max(1e-6, total_wall), 1),
        "in_engine_total_visits_s": round(total_nodes / max(1e-6, total_wall), 1),
        "cpu_ms_per_move": round((total_cpu * 1000.0) / max(1, plies), 2),
    }


def _measure_in_engine_fast_tree_nps(ov_model: ov.Model, cache_dir: str, num_plies: int = 12, movetime_ms: int = 400) -> dict:
    assert_ac_power()
    root_m = NPUModel(batch_size=1, cache_dir=cache_dir, device="NPU", use_turbo=True, performance_hint="LATENCY", ov_model_override=ov_model, verbose=False)
    batch_m = NPUModel(batch_size=16, cache_dir=cache_dir, device="NPU", use_turbo=True, performance_hint="LATENCY", ov_model_override=ov_model, verbose=False)
    return _measure_in_engine_fast_tree_nps_with_models(root_m, batch_m, num_plies=num_plies, movetime_ms=movetime_ms)


def run_spike_3(private_root: Path) -> dict:
    import nncf

    print("\n" + "=" * 80, flush=True)
    print("SPIKE 3: BLOCK-WISE QUANTIZATION SENSITIVITY PROFILE ON NPU 3720 (346M 10x192)", flush=True)
    print("=" * 80, flush=True)
    assert_ac_power()

    with np.load(DEFAULT_WEIGHTS_NPZ) as npz:
        weights_dict = {k: npz[k] for k in npz.files}
    base_ov = build_openvino_se_resnet(batch_size=-1, num_blocks=10, channels=192, in_channels=18, weights_dict=weights_dict)

    print("[Spike 3] Loading 2,000 real training calibration positions + 5,000 Far-Test positions...", flush=True)
    calib_2000 = _load_2000_calib(private_root)
    x_5000, masks_5000 = _load_5000_far_test(private_root)
    calib_items_2000 = [calib_2000[i : i + 1] for i in range(2000)]
    calib_ds_2000 = nncf.Dataset(calib_items_2000, lambda a: {"board_state": a})
    calib_items_500 = [calib_2000[i : i + 1] for i in range(500)]
    calib_ds_500 = nncf.Dataset(calib_items_500, lambda a: {"board_state": a})

    tmp_ir_dir = Path(tempfile.mkdtemp(prefix="nc_spike3_ir_"))
    tmp_cache_dir = Path(tempfile.mkdtemp(prefix="nc_spike3_cache_"))
    try:
        fp16_xml = tmp_ir_dir / "fp16.xml"
        ov.save_model(base_ov.clone(), str(fp16_xml), compress_to_fp16=True)
        fp16_size_mb = round((fp16_xml.with_suffix(".bin").stat().st_size) / 1048576.0, 2)
        fp16_model = ov.Core().read_model(str(fp16_xml))

        fp_l, fp_v, fp_raw_pos_s = _eval_5000_on_npu(fp16_model, x_5000, str(tmp_cache_dir))
        print(f"[Spike 3] Pure FP16 baseline: {fp16_size_mb} MB | Raw NPU B=16: {fp_raw_pos_s} pos/s", flush=True)

        # 3A. Single-block leave-one-block-quantized sensitivity scan (block_0 .. block_9)
        per_block_profile = []
        base_ignore = ["^init_.*", ".*_se_.*", "^pol_.*", "^val_.*", "^(value|wdl)$"]
        for k in range(10):
            assert_ac_power()
            ignored_blocks = [f"^res_{j}_.*" for j in range(10) if j != k]
            ig = nncf.IgnoredScope(patterns=base_ignore + ignored_blocks, validate=True)
            t0 = time.perf_counter()
            qm = nncf.quantize(
                base_ov.clone(),
                calib_ds_500,
                target_device=nncf.TargetDevice.NPU,
                subset_size=500,
                fast_bias_correction=True,
                ignored_scope=ig,
            )
            q_s = round(time.perf_counter() - t0, 2)
            kx = tmp_ir_dir / f"only_b{k}.xml"
            ov.save_model(qm, str(kx), compress_to_fp16=True)
            km = ov.Core().read_model(str(kx))
            kl, kv, k_raw = _eval_5000_on_npu(km, x_5000, str(tmp_cache_dir))
            fid = _compute_fidelity(fp_l, fp_v, kl, kv, masks_5000)
            row = {"block_index": k, "quant_time_s": q_s, "raw_npu_b16_pos_s": k_raw, **fid}
            per_block_profile.append(row)
            print(
                f"[Spike 3A] Only Block {k}: Top-1={fid['top1_agreement_pct']:6.2f}% | "
                f"KL={fid['mean_policy_kl']:.6f} | MaxValDiff={fid['max_value_diff']:.5f} | RawNPS={k_raw}",
                flush=True,
            )

        ranked = sorted(per_block_profile, key=lambda r: (r["mean_policy_kl"], -r["top1_agreement_pct"], r["max_value_diff"]))
        least_sensitive_6 = sorted(r["block_index"] for r in ranked[:6])
        print(f"[Spike 3A] Empirically least-sensitive 6 blocks by Mean KL: {least_sensitive_6}", flush=True)

        # 3B. Full 2,000-sample NNCF comparisons on NPU 3720 + In-Engine fast_tree NPS
        variants_spec: list[tuple[str, list[int] | None, list[str]]] = [
            ("V0_Pure_FP16", None, []),
            ("Full_10Block_W8A8", list(range(10)), base_ignore),
            (
                "Hybrid_Middle_2_to_7_W8A8",
                [2, 3, 4, 5, 6, 7],
                base_ignore + [f"^res_{j}_.*" for j in (0, 1, 8, 9)],
            ),
            (
                "Hybrid_Middle_2_to_7_ConvOnly",
                [2, 3, 4, 5, 6, 7],
                base_ignore + [f"^res_{j}_.*" for j in (0, 1, 8, 9)] + [".*_out_.*"],
            ),
        ]
        if least_sensitive_6 != [2, 3, 4, 5, 6, 7]:
            keep_fp16 = [j for j in range(10) if j not in least_sensitive_6]
            variants_spec.append(
                (
                    f"Hybrid_LeastSens6_{''.join(str(x) for x in least_sensitive_6)}_W8A8",
                    least_sensitive_6,
                    base_ignore + [f"^res_{j}_.*" for j in keep_fp16],
                )
            )

        fp16_engine = _measure_in_engine_fast_tree_nps(fp16_model, str(tmp_cache_dir))
        variants_results = {}
        for vname, quant_blocks, ig_patterns in variants_spec:
            assert_ac_power()
            if quant_blocks is None:
                variants_results[vname] = {
                    "quantized_blocks": [],
                    "model_size_mb": fp16_size_mb,
                    "quant_time_s": 0.0,
                    "top1_agreement_pct": 100.0,
                    "mean_policy_kl": 0.0,
                    "max_value_diff": 0.0,
                    "mean_value_diff": 0.0,
                    "raw_npu_b16_pos_s": fp_raw_pos_s,
                    **fp16_engine,
                }
                continue

            ig = nncf.IgnoredScope(patterns=ig_patterns, validate=True)
            t0 = time.perf_counter()
            qm = nncf.quantize(
                base_ov.clone(),
                calib_ds_2000,
                target_device=nncf.TargetDevice.NPU,
                subset_size=2000,
                fast_bias_correction=True,
                ignored_scope=ig,
            )
            q_s = round(time.perf_counter() - t0, 2)
            vx = tmp_ir_dir / f"{vname}.xml"
            ov.save_model(qm, str(vx), compress_to_fp16=True)
            sz_mb = round(vx.with_suffix(".bin").stat().st_size / 1048576.0, 2)
            vm = ov.Core().read_model(str(vx))
            vl, vv, v_raw = _eval_5000_on_npu(vm, x_5000, str(tmp_cache_dir))
            fid = _compute_fidelity(fp_l, fp_v, vl, vv, masks_5000)
            eng = _measure_in_engine_fast_tree_nps(vm, str(tmp_cache_dir))
            variants_results[vname] = {
                "quantized_blocks": quant_blocks,
                "model_size_mb": sz_mb,
                "quant_time_s": q_s,
                **fid,
                "raw_npu_b16_pos_s": v_raw,
                **eng,
            }
            print(
                f"[Spike 3B] {vname:30s}: Size={sz_mb:5.2f}MB | Top-1={fid['top1_agreement_pct']:6.2f}% | "
                f"KL={fid['mean_policy_kl']:.6f} | MaxValDiff={fid['max_value_diff']:.5f} | "
                f"AddedNPS={eng['in_engine_added_visits_s']} | TotalNPS={eng['in_engine_total_visits_s']}",
                flush=True,
            )

        return {
            "calib_positions": 2000,
            "far_test_positions": 5000,
            "per_block_sensitivity_500calib": per_block_profile,
            "least_sensitive_6_blocks": least_sensitive_6,
            "variants_2000calib": variants_results,
            "reference_published": {
                "qat_10block_w8a8_top1_pct": 92.52,
                "qat_10block_w8a8_max_value_diff": 0.95,
                "ptq_res3x3_top1_pct": 90.96,
                "ptq_res3x3_max_value_diff": 0.9751,
            },
        }
    finally:
        shutil.rmtree(tmp_ir_dir, ignore_errors=True)
        shutil.rmtree(tmp_cache_dir, ignore_errors=True)


def _make_prod_searcher(
    root_model: NPUModel,
    batch_model: NPUModel,
    use_dynamic_batch_shift: bool = False,
    batch_models: dict[int, NPUModel] | None = None,
) -> NPUBatchedPUCTSearch:
    return NPUBatchedPUCTSearch(
        root_model=root_model,
        batch_model=batch_model,
        batch_size=16,
        c_puct=1.5,
        use_fpu_reduction=True,
        fpu_reduction=0.20,
        root_fpu_reduction=0.10,
        t_root=1.0,
        use_exact_solver=True,
        use_nn_cache=True,
        use_tree_reuse=False,
        use_fast_tree=True,
        use_dynamic_batch_shift=use_dynamic_batch_shift,
        batch_models=batch_models,
    )


def run_phase2_step1(private_root: Path, sf_path: str | None) -> dict:
    """Phase 2 Step 1: Strict Collision & Cold-Start Gear-Shifter evaluation vs `baseline_b16_prod_db`."""
    print("\n" + "=" * 80, flush=True)
    print("PHASE 2 STEP 1: COLLISION-BOUNDED & COLD-START GEAR-SHIFTER (B=4/8/16)", flush=True)
    print("=" * 80, flush=True)
    assert_ac_power()

    base_ov = ov.Core().read_model(DEFAULT_MODEL_XML)
    co_models: dict[int, NPUModel] = {}
    for b in (1, 4, 8, 16):
        co_models[b] = NPUModel(
            batch_size=b,
            cache_dir=DEFAULT_CACHE_DIR,
            device="NPU",
            use_turbo=True,
            performance_hint="LATENCY",
            ov_model_override=base_ov,
            verbose=False,
        )

    cat_c, cat_a = load_loss_anatomy_subsets(private_root)
    puzzles_500 = load_clean_500_puzzles(private_root)
    grader = FenStockfishGrader(sf_path, depth=14)

    try:
        arms = {
            "baseline_b16_prod_db": _make_prod_searcher(co_models[1], co_models[16], use_dynamic_batch_shift=False),
            "gear_shift_v2_collision_cold": _make_prod_searcher(
                co_models[1],
                co_models[16],
                use_dynamic_batch_shift=True,
                batch_models={4: co_models[4], 8: co_models[8], 16: co_models[16]},
            ),
        }

        res: dict = {
            "tactical_horizon_cat_c_19_400ms": {},
            "value_search_cat_a_19_400ms": {},
            "clean_500_puzzles_50ms": {},
            "clean_500_puzzles_200ms": {},
        }
        for name, srch in arms.items():
            assert_ac_power()
            srch.search(chess.Board(), movetime_ms=80, max_nodes=32)

            rc = _eval_loss_fens(srch, grader, cat_c, max_nodes=None, movetime_ms=400)
            res["tactical_horizon_cat_c_19_400ms"][name] = rc
            print(
                f"[P2 Step 1] Cat(c) 19 @ 400ms | {name:28s}: SF={rc['orig_sf_best_matches']}/19 | "
                f"Bl>=150={rc['blunders_ge_150cp']}/19 MeanDrop={rc['mean_cp_drop']:.1f}cp | "
                f"Nodes={rc['total_nodes']} avg_ply={rc['mean_avg_ply']:.2f} max_ply={rc['mean_max_ply']:.2f} | "
                f"Shares%={rc['batch_share_pct']} Reasons={rc['gear_reasons']}",
                flush=True,
            )

            ra = _eval_loss_fens(srch, grader, cat_a, max_nodes=None, movetime_ms=400)
            res["value_search_cat_a_19_400ms"][name] = ra
            print(
                f"[P2 Step 1] Cat(a) 19 @ 400ms | {name:28s}: SF={ra['orig_sf_best_matches']}/19 | "
                f"Bl>=150={ra['blunders_ge_150cp']}/19 MeanDrop={ra['mean_cp_drop']:.1f}cp | "
                f"Nodes={ra['total_nodes']} avg_ply={ra['mean_avg_ply']:.2f} max_ply={ra['mean_max_ply']:.2f} | "
                f"Shares%={ra['batch_share_pct']} Reasons={ra['gear_reasons']}",
                flush=True,
            )

            rp50 = _eval_500_puzzles(srch, puzzles_500, max_nodes=None, movetime_ms=50)
            res["clean_500_puzzles_50ms"][name] = rp50
            print(
                f"[P2 Step 1] 500 Pz   @  50ms | {name:28s}: {rp50['solved']}/500 ({rp50['accuracy_pct']:.2f}%) | "
                f"Nodes={rp50['total_nodes']} (mean={rp50['mean_nodes']:.1f}) avg_ply={rp50['mean_avg_ply']:.2f} "
                f"max_ply={rp50['mean_max_ply']:.2f} | Shares%={rp50['batch_share_pct']} Reasons={rp50['gear_reasons']}",
                flush=True,
            )

            rp200 = _eval_500_puzzles(srch, puzzles_500, max_nodes=None, movetime_ms=200)
            res["clean_500_puzzles_200ms"][name] = rp200
            print(
                f"[P2 Step 1] 500 Pz   @ 200ms | {name:28s}: {rp200['solved']}/500 ({rp200['accuracy_pct']:.2f}%) | "
                f"Nodes={rp200['total_nodes']} (mean={rp200['mean_nodes']:.1f}) avg_ply={rp200['mean_max_ply']:.2f} | "
                f"Shares%={rp200['batch_share_pct']} Reasons={rp200['gear_reasons']}",
                flush=True,
            )

        return res
    finally:
        grader.close()


def run_phase2_step2(private_root: Path) -> dict:
    """Phase 2 Step 2: Ground-Truth Stockfish Eval + Clean Puzzles for Safe-4 vs Middle-6 vs Pure FP16."""
    import gc
    import nncf

    print("\n" + "=" * 80, flush=True)
    print("PHASE 2 STEP 2: GROUND-TRUTH STOCKFISH EVAL & SAFE-4 HYBRID QUANTIZATION", flush=True)
    print("=" * 80, flush=True)
    assert_ac_power()

    with np.load(DEFAULT_WEIGHTS_NPZ) as npz:
        weights_dict = {k: npz[k] for k in npz.files}
    base_ov = build_openvino_se_resnet(batch_size=-1, num_blocks=10, channels=192, in_channels=18, weights_dict=weights_dict)

    print("[P2 Step 2] Loading 2,000 train calib + 5,000 Far-Test (with Stockfish GT) + 500 Clean Puzzles...", flush=True)
    calib_2000 = _load_2000_calib(private_root)
    calib_ds_2000 = nncf.Dataset([calib_2000[i : i + 1] for i in range(2000)], lambda a: {"board_state": a})
    x_5000, masks_5000, acts_5000, val_targets_5000, cps_5000 = _load_5000_far_test_with_targets(private_root)
    puzzles_500 = load_clean_500_puzzles(private_root)

    base_ignore = ["^init_.*", ".*_se_.*", "^pol_.*", "^val_.*", "^(value|wdl)$"]
    variants_spec: list[tuple[str, list[int], list[str]]] = [
        ("V0_Pure_FP16", [], []),
        (
            "Hybrid_Middle_2_to_7_ConvOnly",
            [2, 3, 4, 5, 6, 7],
            base_ignore + [f"^res_{j}_.*" for j in (0, 1, 8, 9)] + [".*_out_.*"],
        ),
        (
            "Hybrid_Safe4_2356_ConvOnly",
            [2, 3, 5, 6],
            base_ignore + [f"^res_{j}_.*" for j in (0, 1, 4, 7, 8, 9)] + [".*_out_.*"],
        ),
    ]

    tmp_ir_dir = Path(tempfile.mkdtemp(prefix="nc_p2s2_ir_"))
    tmp_cache_dir = Path(tempfile.mkdtemp(prefix="nc_p2s2_cache_"))
    try:
        fp_l = fp_v = None
        results: dict = {}

        for vname, quant_blocks, ig_patterns in variants_spec:
            assert_ac_power()
            if not quant_blocks:
                vx = tmp_ir_dir / f"{vname}.xml"
                ov.save_model(base_ov.clone(), str(vx), compress_to_fp16=True)
                q_s = 0.0
            else:
                ig = nncf.IgnoredScope(patterns=ig_patterns, validate=True)
                t0 = time.perf_counter()
                qm = nncf.quantize(
                    base_ov.clone(),
                    calib_ds_2000,
                    target_device=nncf.TargetDevice.NPU,
                    subset_size=2000,
                    fast_bias_correction=True,
                    ignored_scope=ig,
                )
                q_s = round(time.perf_counter() - t0, 2)
                vx = tmp_ir_dir / f"{vname}.xml"
                ov.save_model(qm, str(vx), compress_to_fp16=True)

            sz_mb = round(vx.with_suffix(".bin").stat().st_size / 1048576.0, 2)
            vm = ov.Core().read_model(str(vx))
            root_m = NPUModel(batch_size=1, cache_dir=str(tmp_cache_dir), device="NPU", use_turbo=True, performance_hint="LATENCY", ov_model_override=vm, verbose=False)
            batch_m = NPUModel(batch_size=16, cache_dir=str(tmp_cache_dir), device="NPU", use_turbo=True, performance_hint="LATENCY", ov_model_override=vm, verbose=False)

            vl, vv, v_raw = _eval_5000_on_npu_model(batch_m, x_5000)
            if fp_l is None:
                fp_l, fp_v = vl, vv
                fid = {
                    "top1_agreement_pct": 100.0,
                    "mean_policy_kl": 0.0,
                    "max_value_diff": 0.0,
                    "mean_value_diff": 0.0,
                }
            else:
                fid = _compute_fidelity(fp_l, fp_v, vl, vv, masks_5000)

            gt = _compute_ground_truth_metrics(vl, vv, masks_5000, acts_5000, val_targets_5000, cps_5000)
            eng = _measure_in_engine_fast_tree_nps_with_models(root_m, batch_m, num_plies=12, movetime_ms=400)

            srch = _make_prod_searcher(root_m, batch_m, use_dynamic_batch_shift=False)
            srch.search(chess.Board(), movetime_ms=80, max_nodes=32)

            pz_100ms = _eval_500_puzzles(srch, puzzles_500, max_nodes=None, movetime_ms=100)
            pz_200n = _eval_500_puzzles(srch, puzzles_500, max_nodes=200, movetime_ms=None)

            results[vname] = {
                "quantized_blocks": quant_blocks,
                "model_size_mb": sz_mb,
                "quant_time_s": q_s,
                "raw_npu_b16_pos_s": v_raw,
                "fp16_agreement": fid,
                "ground_truth_far_test_5000": gt,
                "in_engine_fast_tree_12ply": eng,
                "clean_500_puzzles_100ms": pz_100ms,
                "clean_500_puzzles_200nodes": pz_200n,
            }
            print(
                f"[P2 Step 2] {vname:30s} ({sz_mb:5.2f}MB) | Far5k GT Top1={gt['gt_masked_top1_pct']:5.2f}% "
                f"CE={gt['gt_masked_policy_ce']:.4f} MSE={gt['gt_value_mse']:.5f} | "
                f"vsFP16: Top1={fid['top1_agreement_pct']:5.2f}% KL={fid['mean_policy_kl']:.6f} MaxV={fid['max_value_diff']:.4f} | "
                f"Pz100ms={pz_100ms['solved']}/500 ({pz_100ms['accuracy_pct']:.2f}%, {pz_100ms['total_nodes']}n, "
                f"{pz_100ms['nps_search']:.0f}nps) | Pz200n={pz_200n['solved']}/500 ({pz_200n['accuracy_pct']:.2f}%) | "
                f"AddedNPS={eng['in_engine_added_visits_s']}",
                flush=True,
            )
            del srch, root_m, batch_m, vm
            gc.collect()

        return {
            "calib_positions": 2000,
            "far_test_positions": 5000,
            "clean_puzzles_total": len(puzzles_500),
            "variants": results,
        }
    finally:
        shutil.rmtree(tmp_ir_dir, ignore_errors=True)
        shutil.rmtree(tmp_cache_dir, ignore_errors=True)


def _gather_data_readiness(private_root: Path) -> dict:
    data_dir = private_root / "data"
    local_bins = []
    sidecars_found = []
    if data_dir.exists():
        for p in sorted(data_dir.rglob("*.bin")):
            sz = p.stat().st_size
            rel = str(p.relative_to(data_dir)).replace("\\", "/")
            fmt = (
                "691B_v2"
                if (sz > 0 and sz % RECORD_SIZE_V2 == 0 and "v2" in rel.lower())
                else ("107B_v1" if (sz > 0 and sz % RECORD_SIZE_V1 == 0) else "other")
            )
            est_recs = sz // (RECORD_SIZE_V2 if fmt == "691B_v2" else RECORD_SIZE_V1) if fmt != "other" else None
            local_bins.append({"file": rel, "size_mb": round(sz / 1048576.0, 2), "format": fmt, "records": est_recs})
        sidecars_found = [str(p) for p in data_dir.rglob("*sidecar*")]

    s3_prefixes: dict = {}
    bucket = "s3://neurachess-spot-162290422654-us-east-1"
    aws_bin = shutil.which("aws") or "aws"
    for prefix in ("data/", "data/shards/", "data/shards_s1/", "checkpoints/", "manifests/"):
        try:
            proc = subprocess.run(
                [aws_bin, "s3", "ls", f"{bucket}/{prefix}", "--recursive", "--human-readable", "--summarize"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            s3_prefixes[prefix] = lines[-3:] if len(lines) >= 2 else (lines or [f"exit={proc.returncode}"])
        except Exception as exc:
            s3_prefixes[prefix] = [f"error: {exc}"]

    return {
        "local_data_dir": str(data_dir),
        "local_bin_files": local_bins,
        "local_sidecar_files": sidecars_found,
        "s3_bucket": bucket,
        "s3_read_only_summary": s3_prefixes,
        "multipv_sidecars_available": len(sidecars_found) > 0,
        "notes": {
            "hl_gauss_64_ready": "YES - eval_cp is stored in every 107B v1 and 691B v2 record (1M v2 shards locally & on S3; 346M v1 shards require re-upload or regenerating from lichess_db_eval).",
            "moves_left_head_ready": "NO - fixed position records (107B/691B) lack remaining game-length/ply labels; requires PGN/self-play game trajectories.",
            "conv73_action_q_head_ready": "NO - 0 .sidecar.bin MultiPV files exist locally or on S3 (shards_s1/ is empty); training dense Q(s,a) requires either MultiPV Stockfish sidecars or search-tree leaf distillation.",
        },
    }


def run_phase2_step3(private_root: Path) -> dict:
    """Phase 2 Step 3: NPU 3720 Latency Benchmark for Stage 4 Experimental Heads + Data Readiness Report."""
    import gc

    print("\n" + "=" * 80, flush=True)
    print("PHASE 2 STEP 3: NPU 3720 HEAD LATENCY BENCHMARKS (B=16) & DATA READINESS", flush=True)
    print("=" * 80, flush=True)
    assert_ac_power()

    readiness = _gather_data_readiness(private_root)
    with np.load(DEFAULT_WEIGHTS_NPZ) as npz:
        weights_dict = {k: npz[k] for k in npz.files}

    x_full = _load_5000_far_test_with_targets(private_root)[0][:16]
    x_sample = np.ascontiguousarray(x_full, dtype=np.float32)

    specs = [
        ("baseline_10x192_scalar", "baseline_scalar"),
        ("hl_gauss_64_plus_mlh", "hl_gauss_mlh"),
        ("dual_conv73_policy_q_plus_val", "dual_conv73_with_val"),
        ("dual_conv73_policy_q_only", "dual_conv73_only"),
    ]

    proc = psutil.Process(os.getpid())
    core = ov.Core()
    tmp_ir_dir = Path(tempfile.mkdtemp(prefix="nc_p2s3_ir_"))
    tmp_cache_dir = Path(tempfile.mkdtemp(prefix="nc_p2s3_cache_"))

    benchmarks: dict = {}
    base_med_ms: float | None = None
    try:
        for label, variant in specs:
            assert_ac_power()
            gc.collect()
            rss_before = proc.memory_info().rss / 1048576.0
            try:
                npu_before = int(core.get_property("NPU", "NPU_DEVICE_ALLOC_MEM_SIZE")) / 1048576.0
            except Exception:
                npu_before = None

            ov_m = build_openvino_research_head_variant(
                variant=variant,
                batch_size=16,
                num_blocks=10,
                channels=192,
                in_channels=18,
                weights_dict=weights_dict,
            )
            vx = tmp_ir_dir / f"{label}.xml"
            ov.save_model(ov_m, str(vx), compress_to_fp16=True)
            ir_bin_mb = round(vx.with_suffix(".bin").stat().st_size / 1048576.0, 2)
            vm = core.read_model(str(vx))

            npu_m = NPUModel(
                batch_size=16,
                cache_dir=str(tmp_cache_dir),
                device="NPU",
                use_turbo=True,
                performance_hint="LATENCY",
                ov_model_override=vm,
                verbose=False,
            )
            req = npu_m.infer_request
            in_t = req.get_input_tensor().data
            in_t[...] = x_sample

            for _ in range(25):
                req.infer()

            lats_ms = []
            for _ in range(200):
                t0 = time.perf_counter()
                req.infer()
                lats_ms.append((time.perf_counter() - t0) * 1000.0)

            out_info = []
            all_finite = True
            for oi in range(len(npu_m.compiled_model.outputs)):
                arr = req.get_output_tensor(oi).data
                fin = bool(np.all(np.isfinite(arr)))
                if not fin:
                    all_finite = False
                out_info.append({"name": npu_m.compiled_model.outputs[oi].any_name, "shape": list(arr.shape), "finite": fin})

            med = float(np.median(lats_ms))
            mean_v = float(np.mean(lats_ms))
            p10 = float(np.percentile(lats_ms, 10))
            p90 = float(np.percentile(lats_ms, 90))
            pos_s = round(16.0 * 1000.0 / max(1e-6, med), 1)
            if base_med_ms is None:
                base_med_ms = med
            delta_ms = round(med - base_med_ms, 3)
            delta_pct = round(100.0 * (med - base_med_ms) / max(1e-6, base_med_ms), 2)

            rss_after = proc.memory_info().rss / 1048576.0
            try:
                npu_after = int(core.get_property("NPU", "NPU_DEVICE_ALLOC_MEM_SIZE")) / 1048576.0
            except Exception:
                npu_after = None

            benchmarks[label] = {
                "variant": variant,
                "ir_bin_mb": ir_bin_mb,
                "compile_time_s": round(npu_m.compile_time_s, 3),
                "median_ms": round(med, 3),
                "mean_ms": round(mean_v, 3),
                "p10_ms": round(p10, 3),
                "p90_ms": round(p90, 3),
                "raw_b16_pos_per_sec": pos_s,
                "delta_ms_vs_baseline": delta_ms,
                "delta_pct_vs_baseline": delta_pct,
                "delta_rss_mb": round(rss_after - rss_before, 2),
                "npu_alloc_after_mb": round(npu_after, 2) if npu_after is not None else None,
                "npu_alloc_delta_mb": round(npu_after - npu_before, 2) if (npu_after is not None and npu_before is not None) else None,
                "outputs": out_info,
                "all_outputs_finite": all_finite,
            }
            print(
                f"[P2 Step 3] {label:32s}: Size={ir_bin_mb:5.2f}MB | med={med:6.3f}ms (p10={p10:.3f}, p90={p90:.3f}) | "
                f"delta={delta_ms:+.3f}ms ({delta_pct:+.2f}%) | raw={pos_s:7.1f} pos/s | outputs={len(out_info)} finite={all_finite}",
                flush=True,
            )
            del npu_m, req, vm, ov_m
            gc.collect()

        return {
            "batch_size": 16,
            "warmup_iters": 25,
            "timed_iters": 200,
            "npu_config": {"device": "NPU", "performance_hint": "LATENCY", "NPU_TURBO": True, "precision": "fp16"},
            "head_latency_benchmarks": benchmarks,
            "data_readiness": readiness,
        }
    finally:
        shutil.rmtree(tmp_ir_dir, ignore_errors=True)
        shutil.rmtree(tmp_cache_dir, ignore_errors=True)


def build_hybrid_mid27_convonly_ir(
    out_xml_path: Path,
    private_root: Path,
    weights_dict: dict[str, np.ndarray] | None = None,
) -> Path:
    """Build or load cached `Hybrid_Middle_2_to_7_ConvOnly` OpenVINO FP16/INT8 hybrid IR."""
    import nncf

    out_xml_path.parent.mkdir(parents=True, exist_ok=True)
    if out_xml_path.exists() and out_xml_path.with_suffix(".bin").exists():
        return out_xml_path

    if weights_dict is None:
        with np.load(DEFAULT_WEIGHTS_NPZ) as npz:
            weights_dict = {k: npz[k] for k in npz.files}

    base_ov = build_openvino_se_resnet(
        batch_size=-1, num_blocks=10, channels=192, in_channels=18, weights_dict=weights_dict
    )
    calib_2000 = _load_2000_calib(private_root)
    calib_ds_2000 = nncf.Dataset([calib_2000[i : i + 1] for i in range(2000)], lambda a: {"board_state": a})
    ig_patterns = [
        "^init_.*",
        ".*_se_.*",
        "^pol_.*",
        "^val_.*",
        "^(value|wdl)$",
        "^res_0_.*",
        "^res_1_.*",
        "^res_8_.*",
        "^res_9_.*",
        ".*_out_.*",
    ]
    ig = nncf.IgnoredScope(patterns=ig_patterns, validate=True)
    qm = nncf.quantize(
        base_ov,
        calib_ds_2000,
        target_device=nncf.TargetDevice.NPU,
        subset_size=2000,
        fast_bias_correction=True,
        ignored_scope=ig,
    )
    ov.save_model(qm, str(out_xml_path), compress_to_fp16=True)
    return out_xml_path


def _measure_coresident_shape_latencies(co_models: dict[int, NPUModel]) -> dict[str, dict]:
    shape_profile: dict[str, dict] = {}
    for b in sorted(co_models.keys()):
        req = co_models[b].infer_request
        in_arr = req.get_input_tensor().data
        in_arr[...] = 0.0
        for _ in range(10):
            req.infer()
        lats_ms = []
        for _ in range(80):
            t0 = time.perf_counter()
            req.infer()
            lats_ms.append((time.perf_counter() - t0) * 1000.0)
        med = float(np.median(lats_ms))
        shape_profile[f"B={b}"] = {
            "batch_size": b,
            "compile_time_s": round(co_models[b].compile_time_s, 3),
            "turnaround_median_ms": round(med, 3),
            "turnaround_mean_ms": round(float(np.mean(lats_ms)), 3),
            "turnaround_p10_ms": round(float(np.percentile(lats_ms, 10)), 3),
            "turnaround_p90_ms": round(float(np.percentile(lats_ms, 90)), 3),
            "raw_pos_per_sec": round(b * 1000.0 / max(1e-6, med), 1),
        }
    return shape_profile


def _run_step1_arm_suite(
    arm_name: str,
    srch: NPUBatchedPUCTSearch,
    grader: FenStockfishGrader,
    puzzles_500: list[dict],
    cat_a: list[dict],
    cat_c: list[dict],
    results: dict,
) -> None:
    assert_ac_power()
    srch.search(chess.Board(), movetime_ms=80, max_nodes=32)

    for label, mt in (("clean_500_puzzles_50ms", 50), ("clean_500_puzzles_100ms", 100), ("clean_500_puzzles_200ms", 200)):
        rp = _eval_500_puzzles(srch, puzzles_500, max_nodes=None, movetime_ms=mt)
        results[label][arm_name] = rp
        print(
            f"[P3 Step 1] 500 Pz @ {mt:3d}ms | {arm_name:34s}: {rp['solved']}/500 ({rp['accuracy_pct']:.2f}%) | "
            f"Nodes={rp['total_nodes']} (mean={rp['mean_nodes']:.1f}) NPS={rp['nps_search']:.0f} "
            f"avg_ply={rp['mean_avg_ply']:.2f} max_ply={rp['mean_max_ply']:.2f} | "
            f"Shares%={rp['batch_share_pct']} Reasons={rp['gear_reasons']}",
            flush=True,
        )

    ra = _eval_loss_fens(srch, grader, cat_a, max_nodes=None, movetime_ms=400)
    results["value_search_cat_a_19_400ms"][arm_name] = ra
    print(
        f"[P3 Step 1] Cat(a) 19 @400ms | {arm_name:34s}: SF={ra['orig_sf_best_matches']}/19 (live={ra['live_sf_d14_matches']}/19) | "
        f"Bl>=150={ra['blunders_ge_150cp']}/19 MeanDrop={ra['mean_cp_drop']:.2f}cp | "
        f"Nodes={ra['total_nodes']} ({ra['mean_nodes']:.1f}) avg_ply={ra['mean_avg_ply']:.2f} max_ply={ra['mean_max_ply']:.2f} | "
        f"Shares%={ra['batch_share_pct']} Reasons={ra['gear_reasons']}",
        flush=True,
    )

    rc = _eval_loss_fens(srch, grader, cat_c, max_nodes=None, movetime_ms=400)
    results["tactical_horizon_cat_c_19_400ms"][arm_name] = rc
    print(
        f"[P3 Step 1] Cat(c) 19 @400ms | {arm_name:34s}: SF={rc['orig_sf_best_matches']}/19 (live={rc['live_sf_d14_matches']}/19) | "
        f"Bl>=150={rc['blunders_ge_150cp']}/19 MeanDrop={rc['mean_cp_drop']:.2f}cp | "
        f"Nodes={rc['total_nodes']} ({rc['mean_nodes']:.1f}) avg_ply={rc['mean_avg_ply']:.2f} max_ply={rc['mean_max_ply']:.2f} | "
        f"Shares%={rc['batch_share_pct']} Reasons={rc['gear_reasons']}",
        flush=True,
    )


def run_phase3_step1(private_root: Path, sf_path: str | None) -> dict:
    """Phase 3 Step 1: gear_shift_v3 (check < 32 + nireq=2 B=16 async DB) + Hybrid_Middle_2_to_7_ConvOnly."""
    import gc

    print("\n" + "=" * 80, flush=True)
    print("PHASE 3 STEP 1: GEAR_SHIFT_V3 (ASYNC B=16 DB + CHECK<32) + HYBRID_MIDDLE_2_TO_7", flush=True)
    print("=" * 80, flush=True)
    assert_ac_power()

    exp_dir = ROOT_DIR / "weights" / "experimental"
    exp_cache = str(exp_dir / "ov_cache")
    hybrid_xml = build_hybrid_mid27_convonly_ir(exp_dir / "neurachess-10x192-hybrid-mid27-int8.xml", private_root)

    cat_c, cat_a = load_loss_anatomy_subsets(private_root)
    puzzles_500 = load_clean_500_puzzles(private_root)
    grader = FenStockfishGrader(sf_path, depth=14)

    res: dict = {
        "hybrid_ir_xml": str(hybrid_xml),
        "hybrid_ir_size_mb": round(hybrid_xml.with_suffix(".bin").stat().st_size / 1048576.0, 2),
        "coresident_latencies": {},
        "clean_500_puzzles_50ms": {},
        "clean_500_puzzles_100ms": {},
        "clean_500_puzzles_200ms": {},
        "value_search_cat_a_19_400ms": {},
        "tactical_horizon_cat_c_19_400ms": {},
    }

    try:
        # 1. Pure FP16 arms (baseline static B=16 prod_db + FP16 gear_shift_v3 ablation)
        fp16_ov = ov.Core().read_model(DEFAULT_MODEL_XML)
        fp16_models = {
            b: NPUModel(
                batch_size=b,
                cache_dir=DEFAULT_CACHE_DIR,
                device="NPU",
                use_turbo=True,
                performance_hint="LATENCY",
                ov_model_override=fp16_ov,
                verbose=False,
            )
            for b in (1, 4, 8, 16)
        }
        res["coresident_latencies"]["V0_Pure_FP16"] = _measure_coresident_shape_latencies(fp16_models)

        srch_base = _make_prod_searcher(fp16_models[1], fp16_models[16], use_dynamic_batch_shift=False)
        _run_step1_arm_suite("V0_Pure_FP16_baseline_b16_prod_db", srch_base, grader, puzzles_500, cat_a, cat_c, res)

        srch_fp16_v3 = _make_prod_searcher(
            fp16_models[1],
            fp16_models[16],
            use_dynamic_batch_shift=True,
            batch_models={4: fp16_models[4], 8: fp16_models[8], 16: fp16_models[16]},
        )
        _run_step1_arm_suite("V0_Pure_FP16_gear_shift_v3", srch_fp16_v3, grader, puzzles_500, cat_a, cat_c, res)
        del srch_base, srch_fp16_v3, fp16_models, fp16_ov
        gc.collect()

        # 2. Combined Hybrid_Middle_2_to_7_ConvOnly + gear_shift_v3
        hyb_ov = ov.Core().read_model(str(hybrid_xml))
        hyb_models = {
            b: NPUModel(
                batch_size=b,
                cache_dir=exp_cache,
                device="NPU",
                use_turbo=True,
                performance_hint="LATENCY",
                ov_model_override=hyb_ov,
                verbose=False,
            )
            for b in (1, 4, 8, 16)
        }
        res["coresident_latencies"]["Hybrid_Middle_2_to_7_ConvOnly"] = _measure_coresident_shape_latencies(hyb_models)
        print(
            "[P3 Step 1] Hybrid B=1/4/8/16 med(ms):",
            {k: v["turnaround_median_ms"] for k, v in res["coresident_latencies"]["Hybrid_Middle_2_to_7_ConvOnly"].items()},
            flush=True,
        )

        srch_hyb_v3 = _make_prod_searcher(
            hyb_models[1],
            hyb_models[16],
            use_dynamic_batch_shift=True,
            batch_models={4: hyb_models[4], 8: hyb_models[8], 16: hyb_models[16]},
        )
        _run_step1_arm_suite("Hybrid_Mid27_ConvOnly_gear_shift_v3", srch_hyb_v3, grader, puzzles_500, cat_a, cat_c, res)
        del srch_hyb_v3, hyb_models, hyb_ov
        gc.collect()
        return res
    finally:
        grader.close()


def run_phase3_step2(private_root: Path, sf_path: str | None) -> dict:
    """Phase 3 Step 2: Train frozen-backbone HL-Gauss-64 value head (1 pass) and run GT + Cat(a) gate."""
    import gc
    from neurachess.training.train_hl_gauss_value import train_frozen_hl_gauss_64

    print("\n" + "=" * 80, flush=True)
    print("PHASE 3 STEP 2: FROZEN-BACKBONE HL-GAUSS-64 VALUE HEAD ($0, 1 SHUFFLED PASS)", flush=True)
    print("=" * 80, flush=True)
    assert_ac_power()

    exp_dir = ROOT_DIR / "weights" / "experimental"
    exp_cache = str(exp_dir / "ov_cache")
    hl_npz = exp_dir / "neurachess-10x192-hl-gauss64-fp16.npz"
    hl_xml = exp_dir / "neurachess-10x192-hl-gauss64-fp16.xml"

    train_res = train_frozen_hl_gauss_64(
        private_root=private_root,
        out_npz_path=hl_npz,
        out_xml_path=hl_xml,
        val_channels=16,
        batch_size=256,
        chunk_size=16384,
    )

    assert_ac_power()
    x_5000, masks_5000, acts_5000, val_targets_5000, cps_5000 = _load_5000_far_test_with_targets(private_root)
    _, cat_a = load_loss_anatomy_subsets(private_root)
    grader = FenStockfishGrader(sf_path, depth=14)

    try:
        # Compare scalar V0_Pure_FP16 (built identically from DEFAULT_WEIGHTS_NPZ) vs HL-Gauss-64 FP16 on NPU 3720
        with np.load(DEFAULT_WEIGHTS_NPZ) as npz:
            scalar_weights = {k: npz[k] for k in npz.files}
        scalar_xml = exp_dir / "neurachess-10x192-scalar-fp16-from-npz.xml"
        if not scalar_xml.exists():
            ov_sc = build_openvino_se_resnet(
                batch_size=-1, num_blocks=10, channels=192, in_channels=18, weights_dict=scalar_weights
            )
            ov.save_model(ov_sc, str(scalar_xml), compress_to_fp16=True)

        eval_arms = [
            ("scalar_fp16_from_npz", scalar_xml, DEFAULT_CACHE_DIR),
            ("hl_gauss64_fp16", hl_xml, exp_cache),
        ]

        far_test_res: dict = {}
        cat_a_res: dict = {}
        ref_logits = None

        for label, xml_p, cdir in eval_arms:
            vm = ov.Core().read_model(str(xml_p))
            root_m = NPUModel(
                batch_size=1, cache_dir=cdir, device="NPU", use_turbo=True, performance_hint="LATENCY", ov_model_override=vm, verbose=False
            )
            batch_m = NPUModel(
                batch_size=16, cache_dir=cdir, device="NPU", use_turbo=True, performance_hint="LATENCY", ov_model_override=vm, verbose=False
            )
            l_out, v_out, raw_pos_s = _eval_5000_on_npu_model(batch_m, x_5000)
            gt = _compute_ground_truth_metrics(l_out, v_out, masks_5000, acts_5000, val_targets_5000, cps_5000)
            if ref_logits is None:
                ref_logits = l_out
                max_pol_diff = 0.0
            else:
                max_pol_diff = float(np.max(np.abs(l_out.astype(np.float64) - ref_logits.astype(np.float64))))

            far_test_res[label] = {
                "raw_npu_b16_pos_s": raw_pos_s,
                "max_policy_logit_diff_vs_scalar": round(max_pol_diff, 6),
                **gt,
            }

            srch = _make_prod_searcher(root_m, batch_m, use_dynamic_batch_shift=False)
            srch.search(chess.Board(), movetime_ms=80, max_nodes=32)
            ra = _eval_loss_fens(srch, grader, cat_a, max_nodes=None, movetime_ms=400)
            cat_a_res[label] = ra
            print(
                f"[P3 Step 2] {label:24s} | Far5k ValueMSE={gt['gt_value_mse']:.5f} DecSign={gt['gt_decisive_sign_pct']:.2f}% "
                f"Top1={gt['gt_masked_top1_pct']:.2f}% MaxPolDiff={max_pol_diff:.6f} | "
                f"Cat(a)@400ms: SF={ra['orig_sf_best_matches']}/19 (live={ra['live_sf_d14_matches']}/19) "
                f"Bl>=150={ra['blunders_ge_150cp']}/19 MeanDrop={ra['mean_cp_drop']:.2f}cp",
                flush=True,
            )
            del srch, root_m, batch_m, vm
            gc.collect()

        # Check DEFAULT_MODEL_XML baseline Cat(a) & Far5k for strict comparison parity too
        base_sc = far_test_res["scalar_fp16_from_npz"]
        hl_sc = far_test_res["hl_gauss64_fp16"]
        base_ca = cat_a_res["scalar_fp16_from_npz"]
        hl_ca = cat_a_res["hl_gauss64_fp16"]

        mse_improved = hl_sc["gt_value_mse"] < base_sc["gt_value_mse"]
        cat_a_no_reg = (
            hl_ca["orig_sf_best_matches"] >= base_ca["orig_sf_best_matches"]
            and hl_ca["blunders_ge_150cp"] <= base_ca["blunders_ge_150cp"]
            and hl_ca["mean_cp_drop"] <= (base_ca["mean_cp_drop"] + 1.0)
        )
        keep_hl = bool(mse_improved and cat_a_no_reg)

        winning_hyb_xml = exp_dir / "neurachess-10x192-hybrid-mid27-int8.xml"
        if keep_hl:
            with np.load(hl_npz) as npz_hl:
                hl_weights = {k: npz_hl[k] for k in npz_hl.files}
            winning_hyb_xml = build_hybrid_mid27_convonly_ir(
                exp_dir / "neurachess-10x192-hl-gauss64-hybrid-mid27-int8.xml",
                private_root,
                weights_dict=hl_weights,
            )

        print(
            f"[P3 Step 2 Gate] MSE improved ({hl_sc['gt_value_mse']:.5f} < {base_sc['gt_value_mse']:.5f})={mse_improved} | "
            f"Cat(a) non-regressing={cat_a_no_reg} => KEEP_HL_GAUSS_64={keep_hl} | Phase3 challenger IR={winning_hyb_xml.name}",
            flush=True,
        )
        return {
            "training": train_res,
            "far_test_5000": far_test_res,
            "value_search_cat_a_19_400ms": cat_a_res,
            "gate": {
                "far5k_mse_improved": mse_improved,
                "cat_a_no_regression": cat_a_no_reg,
                "keep_hl_gauss_64": keep_hl,
                "winning_hybrid_ir_xml": str(winning_hyb_xml),
            },
        }
    finally:
        grader.close()


def run_phase3_step3(private_root: Path, sf_path: str | None, existing_payload: dict) -> dict:
    """Phase 3 Step 3: 60-game H2H vs v1.0.2 and conditional 50-game SF UCI_Elo=3000 validation match."""
    assert_ac_power()
    exp_dir = ROOT_DIR / "weights" / "experimental"
    exp_cache = str(exp_dir / "ov_cache")

    step2_gate = (existing_payload.get("phase3_step2_hl_gauss_64") or {}).get("gate") or {}
    challenger_xml = Path(
        step2_gate.get("winning_hybrid_ir_xml") or (exp_dir / "neurachess-10x192-hybrid-mid27-int8.xml")
    )
    if not challenger_xml.exists():
        challenger_xml = build_hybrid_mid27_convonly_ir(
            exp_dir / "neurachess-10x192-hybrid-mid27-int8.xml",
            private_root,
        )

    h2h_out = ROOT_DIR / "eval_results" / "phase3_h2h_60g.json"
    sf_out = ROOT_DIR / "eval_results" / "phase3_sf3000_50g.json"

    print("\n" + "=" * 80, flush=True)
    print(f"PHASE 3 STEP 3A: 60-GAME H2H ({challenger_xml.name} + gear_shift_v3 vs v1.0.2 B=16 FP16)", flush=True)
    print("=" * 80, flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT_DIR / "src")
    env["PYTHONUNBUFFERED"] = "1"
    h2h_cmd = [
        sys.executable,
        "-m",
        "neurachess.scripts.match_runner",
        "h2h",
        "--xml-a",
        str(challenger_xml),
        "--xml-b",
        str(DEFAULT_MODEL_XML),
        "--pairs",
        "30",
        "--movetime-ms",
        "400",
        "--max-plies",
        "220",
        "--dynamic-gear-a",
        "--cache-dir-a",
        exp_cache,
        "--cache-dir-b",
        DEFAULT_CACHE_DIR,
        "--output",
        str(h2h_out),
    ]
    subprocess.run(h2h_cmd, check=True, env=env)
    h2h_summary = json.loads(h2h_out.read_text(encoding="utf-8"))
    passed_h2h_gate = float(h2h_summary.get("score_pct", 0.0)) >= 53.0 or float(h2h_summary.get("points", 0.0)) >= 32.0

    result: dict = {
        "challenger_xml": str(challenger_xml),
        "baseline_xml": str(DEFAULT_MODEL_XML),
        "h2h_60_games_400ms_no_ponder": h2h_summary,
        "h2h_gate_passed_ge_53pct": passed_h2h_gate,
        "stockfish_3000_50_games_400ms_ponder": None,
    }

    if passed_h2h_gate:
        assert_ac_power()
        print("\n" + "=" * 80, flush=True)
        print("PHASE 3 STEP 3B: H2H GATE PASSED (>= 53.0%) -> 50-GAME VS STOCKFISH 19 UCI_Elo 3000 (PONDER=ON)", flush=True)
        print("=" * 80, flush=True)
        sf_cmd = [
            sys.executable,
            "-m",
            "neurachess.scripts.match_runner",
            "vs-stockfish",
            "--xml",
            str(challenger_xml),
            "--sf-elo",
            "3000",
            "--pairs",
            "25",
            "--movetime-ms",
            "400",
            "--max-plies",
            "220",
            "--ponder",
            "--dynamic-gear",
            "--cache-dir",
            exp_cache,
            "--output",
            str(sf_out),
        ]
        if sf_path:
            sf_cmd.extend(["--stockfish-path", sf_path])
        subprocess.run(sf_cmd, check=True, env=env)
        sf_summary = json.loads(sf_out.read_text(encoding="utf-8"))
        result["stockfish_3000_50_games_400ms_ponder"] = sf_summary
    else:
        print(
            f"[P3 Step 3] Challenger scored {h2h_summary.get('score_pct')}% ({h2h_summary.get('points')}/60) < 53.0% threshold; "
            "skipping 50-game Stockfish validation match per pre-registered rule.",
            flush=True,
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeuraChess local NPU research spikes (v1.1)")
    parser.add_argument("--spike", choices=["1", "2", "3", "all", "none"], default="all")
    parser.add_argument("--phase2", choices=["1", "2", "3", "all"], default=None)
    parser.add_argument("--phase3", choices=["1", "2", "3", "all"], default=None)
    parser.add_argument("--private-root", type=Path, default=ROOT_DIR.parent / "NeuraChess-Private")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stockfish-path", type=str, default=None)
    args = parser.parse_args()

    out_path = args.output
    if out_path is None:
        if args.phase3:
            out_path = ROOT_DIR / "eval_results" / "research_spikes_phase3.json"
        elif args.phase2:
            out_path = ROOT_DIR / "eval_results" / "research_spikes_phase2.json"
        else:
            out_path = ROOT_DIR / "eval_results" / "research_spikes.json"

    ac = assert_ac_power()
    payload: dict = {}
    if out_path.exists():
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    payload["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload["model"] = "neurachess-10x192-fp16"
    payload["ac_power"] = ac

    if args.phase3:
        if args.phase3 in ("1", "all"):
            payload["phase3_step1_hybrid_gear_shift_v3"] = run_phase3_step1(args.private_root, args.stockfish_path)
            _save_report(out_path, payload)
        if args.phase3 in ("2", "all"):
            payload["phase3_step2_hl_gauss_64"] = run_phase3_step2(args.private_root, args.stockfish_path)
            _save_report(out_path, payload)
        if args.phase3 in ("3", "all"):
            payload["phase3_step3_validation_matches"] = run_phase3_step3(args.private_root, args.stockfish_path, payload)
            _save_report(out_path, payload)
    elif args.phase2:
        if args.phase2 in ("1", "all"):
            payload["phase2_step1_collision_cold_start_gear_shift"] = run_phase2_step1(args.private_root, args.stockfish_path)
            _save_report(out_path, payload)
        if args.phase2 in ("2", "all"):
            payload["phase2_step2_ground_truth_and_safe4_quant"] = run_phase2_step2(args.private_root)
            _save_report(out_path, payload)
        if args.phase2 in ("3", "all"):
            payload["phase2_step3_head_latency_and_data_readiness"] = run_phase2_step3(args.private_root)
            _save_report(out_path, payload)
    else:
        if args.spike in ("1", "all"):
            payload["spike_1_see_puct"] = run_spike_1(args.private_root, args.stockfish_path)
            _save_report(out_path, payload)

        if args.spike in ("2", "all"):
            payload["spike_2_dual_shape_gear_shifting"] = run_spike_2(args.private_root, args.stockfish_path)
            _save_report(out_path, payload)

        if args.spike in ("3", "all"):
            payload["spike_3_blockwise_quantization"] = run_spike_3(args.private_root)
            _save_report(out_path, payload)

    print(f"\n[Done] Research spike telemetry written -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
