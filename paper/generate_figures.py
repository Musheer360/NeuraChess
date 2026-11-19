#!/usr/bin/env python3
"""Generate publication-grade vector (PDF) and 300 DPI raster (PNG) figures
for the NeuraChess arXiv manuscript directly from raw ``eval_results/*.json``.

Usage:
    python paper/generate_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths & Global Academic Styling (Colorblind-Safe Okabe-Ito Palette)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "eval_results"
FIG_DIR = REPO_ROOT / "paper" / "figures"

COLORS = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "charcoal": "#2B2B2B",
    "gray": "#7F7F7F",
    "lightgray": "#E5E7EB",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 10.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.3,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "legend.fontsize": 8.0,
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.edgecolor": "#CCCCCC",
            "grid.color": "#D1D5DB",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(filename: str) -> Any:
    path = EVAL_DIR / filename
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=300)
    plt.close(fig)
    print(
        f"[OK] Saved {pdf_path.relative_to(REPO_ROOT)} ({pdf_path.stat().st_size:,} B) "
        f"& {png_path.relative_to(REPO_ROOT)} ({png_path.stat().st_size:,} B)"
    )


def _tilted_pentanomial_probs(p_emp: list[float], target_mean: float) -> list[float]:
    vals = (0.0, 0.25, 0.50, 0.75, 1.0)
    beta = 0.0
    for _ in range(7):
        w = [p_emp[j] * math.exp(beta * vals[j]) for j in range(5)]
        z = sum(w)
        q = [wi / z for wi in w]
        mean = sum(vals[j] * q[j] for j in range(5))
        diff = mean - target_mean
        if abs(diff) < 1e-10:
            return q
        var = sum(q[j] * ((vals[j] - mean) ** 2) for j in range(5))
        if var < 1e-12:
            break
        beta = max(-40.0, min(40.0, beta - diff / var))
    w = [p_emp[j] * math.exp(beta * vals[j]) for j in range(5)]
    z = sum(w)
    return [wi / z for wi in w]


def compute_pentanomial_sprt_llr(
    pentanomial: list[int],
    elo0: float = 0.0,
    elo1: float = 20.0,
) -> float:
    """Exact exponentially-tilted MLE pentanomial SPRT LLR (Jeffreys prior 0.5)
    matching ``src/neurachess/scripts/match_runner.py``."""
    m_pairs = sum(pentanomial)
    if m_pairs == 0:
        return 0.0
    prior = 0.5
    p_emp = [(pentanomial[j] + prior) / (m_pairs + 5.0 * prior) for j in range(5)]
    s0 = 1.0 / (1.0 + 10.0 ** (-elo0 / 400.0))
    s1 = 1.0 / (1.0 + 10.0 ** (-elo1 / 400.0))
    q0 = _tilted_pentanomial_probs(p_emp, s0)
    q1 = _tilted_pentanomial_probs(p_emp, s1)
    return float(
        sum(
            pentanomial[j] * math.log(max(1e-15, q1[j]) / max(1e-15, q0[j]))
            for j in range(5)
        )
    )


# ---------------------------------------------------------------------------
# Figure 1: Per-Block W8A8 Sensitivity Scan & Variant Comparison
# ---------------------------------------------------------------------------
def make_fig1() -> None:
    spikes1 = load_json("research_spikes.json")
    spikes3 = load_json("research_spikes_phase3.json")
    synth = load_json("v1_1_step1_synthesis.json")
    h2h180 = load_json("phase4_h2h_a2int8_vs_346m_180g_combined.json")
    ladder = load_json("v1_1_stockfish_ladder_150g.json")

    scan = spikes1["spike_3_blockwise_quantization"]["per_block_sensitivity_500calib"]
    variants = spikes1["spike_3_blockwise_quantization"]["variants_2000calib"]
    soup_row = next(r for r in synth if r["name"] == "A2_soup75")

    blocks = [int(r["block_index"]) for r in scan]
    kls = [float(r["mean_policy_kl"]) for r in scan]
    max_vdiffs = [float(r["max_value_diff"]) for r in scan]

    fig = plt.figure(figsize=(11.2, 4.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.16, 1.0], wspace=0.33)

    # --- Panel (a): Per-block sensitivity ---
    ax1 = fig.add_subplot(gs[0])
    ax1_r = ax1.twinx()

    ax1.axvspan(-0.5, 1.5, color="#FDECEC", alpha=0.65, zorder=0)
    ax1.axvspan(7.5, 9.5, color="#FDECEC", alpha=0.65, zorder=0)
    ax1.axvspan(1.5, 7.5, color="#E8F6F1", alpha=0.70, zorder=0)

    bar_colors = [
        COLORS["vermillion"] if b == 9 else (COLORS["blue"] if 2 <= b <= 7 else "#6B9AC4")
        for b in blocks
    ]
    ax1.bar(
        blocks,
        [k * 1e3 for k in kls],
        width=0.56,
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.6,
        alpha=0.90,
        zorder=3,
        label="Mean Policy KL ($\\times 10^{-3}$)",
    )
    ax1_r.plot(
        blocks,
        max_vdiffs,
        color=COLORS["vermillion"],
        marker="D",
        markersize=5.6,
        linewidth=1.8,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=5,
        label="Max $|\\Delta V|$ vs. FP16",
    )

    ax1.set_xticks(blocks)
    ax1.set_xticklabels([f"B{b}" for b in blocks])
    ax1.set_xlim(-0.55, 9.55)
    ax1.set_ylim(0, 17.6)
    ax1_r.set_ylim(0, 1.24)
    ax1.set_xlabel("Single Quantized Residual Block (W8A8 PTQ, $N=5{,}000$ Far-Test)")
    ax1.set_ylabel("Mean Policy KL Divergence ($\\times 10^{-3}$)", color=COLORS["blue"])
    ax1_r.set_ylabel("Max Value Error $|V_{\\mathrm{INT8}} - V_{\\mathrm{FP16}}|$", color=COLORS["vermillion"])
    ax1.tick_params(axis="y", colors=COLORS["blue"])
    ax1_r.tick_params(axis="y", colors=COLORS["vermillion"])
    ax1.grid(axis="y", zorder=1)
    ax1.spines["top"].set_visible(False)
    ax1_r.spines["top"].set_visible(False)
    ax1.set_title("(a) Layerwise W8A8 Sensitivity Scan (Blocks 0–9)", pad=8)

    # Zone headers cleanly along the top margin (above y=15.5)
    ax1.text(4.5, 16.9, "Middle Band (B2–B7): Quantized W8A8",
             ha="center", va="top", fontsize=7.8, fontweight="bold", color="#006B4F")
    ax1.text(0.5, 16.9, "FP16\nGuard", ha="center", va="top", fontsize=7.3,
             fontweight="bold", color="#9B1C1C")
    ax1.text(8.5, 16.9, "FP16\nGuard", ha="center", va="top", fontsize=7.3,
             fontweight="bold", color="#9B1C1C")

    # Legend cleanly in the upper-center open band (below zone header, above callouts)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(
        h1 + h2, l1 + l2,
        loc="upper center", bbox_to_anchor=(0.50, 0.90),
        ncol=2, fontsize=7.3, columnspacing=1.0, handletextpad=0.5,
    )

    # Callout annotations placed cleanly in open space with zero overlap
    ax1_r.annotate(
        f"B1 Value Spike\n$|\\Delta V|_{{\\max}} = {max_vdiffs[1]:.3f}$",
        xy=(1, max_vdiffs[1]),
        xytext=(2.55, 0.80),
        fontsize=7.6,
        fontweight="bold",
        color=COLORS["vermillion"],
        arrowprops=dict(arrowstyle="->", color=COLORS["vermillion"], lw=1.2, shrinkA=2, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec=COLORS["vermillion"], lw=0.8),
        zorder=6,
    )
    ax1.annotate(
        f"B9 Policy KL Spike\n$0.01304$ ($>9\\times$)",
        xy=(9, kls[9] * 1e3),
        xytext=(5.15, 9.6),
        fontsize=7.6,
        fontweight="bold",
        color="#9B1C1C",
        arrowprops=dict(arrowstyle="->", color="#9B1C1C", lw=1.2, shrinkA=2, shrinkB=4),
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#9B1C1C", lw=0.8),
        zorder=6,
    )

    # --- Panel (b): 4-Variant Quantization Comparison ---
    ax2 = fig.add_subplot(gs[1])
    labels = [
        "Pure\nFP16",
        "Full 10-Blk\nW8A8 PTQ",
        "Hybrid Mid\n2..7 PTQ",
        "A2 Mid 2..7\nW8A8 QAT",
    ]
    raw_b16 = [
        float(spikes3["phase3_step1_hybrid_gear_shift_v3"]["coresident_latencies"]["V0_Pure_FP16"]["B=16"]["raw_pos_per_sec"]),
        float(variants["Full_10Block_W8A8"]["raw_npu_b16_pos_s"]),
        float(spikes3["phase3_step1_hybrid_gear_shift_v3"]["coresident_latencies"]["Hybrid_Middle_2_to_7_ConvOnly"]["B=16"]["raw_pos_per_sec"]),
        float(soup_row["throughput"]["b16_raw_pos_s"]),
    ]
    eng_nps = [
        float(h2h180["nps_b"]),
        float(variants["Full_10Block_W8A8"]["in_engine_total_visits_s"]),
        float(variants["Hybrid_Middle_2_to_7_ConvOnly"]["in_engine_total_visits_s"]),
        float(ladder["mean_effective_nps"]),
    ]
    verdicts = [
        ("Baseline\nCE 1.618", COLORS["charcoal"], "#F3F4F6"),
        ("FAIL PTQ\n91.0% agr\n$|\\Delta V|$ 0.975", "#9B1C1C", "#FDECEC"),
        ("PTQ Drop\n95.5% agr\nSF3000 33%", "#9A5B00", "#FEF3C7"),
        ("QAT PASS\nCE 1.491\n+56.5 Elo", "#006B4F", "#E6F7F1"),
    ]

    x = np.arange(len(labels))
    w = 0.34
    b_raw = ax2.bar(
        x - w / 2, raw_b16, width=w, color=COLORS["sky"], edgecolor="#222222",
        linewidth=0.6, label="Raw NPU $B{=}16$ (pos/s)", zorder=3,
    )
    b_eng = ax2.bar(
        x + w / 2, eng_nps, width=w,
        color=[COLORS["blue"], COLORS["vermillion"], COLORS["orange"], COLORS["green"]],
        edgecolor="#222222", linewidth=0.6, label="In-Engine PUCT NPS", zorder=3,
    )

    for rect in b_raw:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 65, f"{h:,.0f}",
                 ha="center", va="bottom", fontsize=7.2, color="#374151")
    for rect in b_eng:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() / 2, h + 65, f"{h:,.0f}",
                 ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#111111")

    for xi, (txt, tc, fc) in enumerate(verdicts):
        ax2.text(
            xi, 5260, txt, ha="center", va="top", fontsize=7.1, fontweight="bold", color=tc,
            bbox=dict(boxstyle="round,pad=0.25", fc=fc, ec=tc, lw=0.75),
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 5550)
    ax2.set_ylabel("Intel NPU 3720 Throughput (pos/s or NPS)")
    ax2.set_title("(b) PTQ vs. Sensitivity-Guided Mid-Block QAT", pad=8)
    ax2.grid(axis="y", zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=7.6)

    save_figure(fig, "fig1_block_sensitivity")


# ---------------------------------------------------------------------------
# Figure 2: NPU 3720 Latency Scaling & Compute-Optimal Pareto Frontier
# ---------------------------------------------------------------------------
def make_fig2() -> None:
    beast = load_json("phase4_beast_archs.json")
    spikes3 = load_json("research_spikes_phase3.json")
    h2h_180 = load_json("phase4_h2h_a2int8_vs_346m_180g_combined.json")
    h2h_b_346 = load_json("phase4_h2h_jobB_ema95_vs_346m_50g.json")
    h2h_bq_a2 = load_json("phase4_h2h_jobB_qat_int8_vs_a2int8_50g.json")
    ladder = load_json("v1_1_stockfish_ladder_150g.json")

    cores = spikes3["phase3_step1_hybrid_gear_shift_v3"]["coresident_latencies"]
    beast_by_name = {a["name"]: a for a in beast["architectures"]}

    # 10x128 FP16 (3.20M params) latencies measured directly on Intel NPU 3720
    # (LATENCY, NPU_TURBO=YES); in-engine NPS (4,810.4) from docs/EXPERIMENTS.md Sec 2.3.
    archs = [
        {
            "label": "10x128\n(FP16)", "params_m": 3.20,
            "b1": 0.611, "b4": 1.434, "b8": 2.434, "b16": 4.523,
            "nps": 4810.4,
        },
        {
            "label": "10x192\n(FP16)", "params_m": 7.01,
            "b1": float(cores["V0_Pure_FP16"]["B=1"]["turnaround_median_ms"]),
            "b4": float(cores["V0_Pure_FP16"]["B=4"]["turnaround_median_ms"]),
            "b8": float(cores["V0_Pure_FP16"]["B=8"]["turnaround_median_ms"]),
            "b16": float(cores["V0_Pure_FP16"]["B=16"]["turnaround_median_ms"]),
            "nps": float(h2h_180["nps_b"]),
        },
        {
            "label": "10x192\n(QAT INT8)", "params_m": 7.19,
            "b1": float(cores["Hybrid_Middle_2_to_7_ConvOnly"]["B=1"]["turnaround_median_ms"]),
            "b4": float(cores["Hybrid_Middle_2_to_7_ConvOnly"]["B=4"]["turnaround_median_ms"]),
            "b8": float(cores["Hybrid_Middle_2_to_7_ConvOnly"]["B=8"]["turnaround_median_ms"]),
            "b16": float(cores["Hybrid_Middle_2_to_7_ConvOnly"]["B=16"]["turnaround_median_ms"]),
            "nps": float(ladder["mean_effective_nps"]),
        },
        {
            "label": "12x192\n(QAT INT8)", "params_m": float(beast_by_name["12x192_mid2_9_hl64"]["param_millions"]),
            "b1": float(beast_by_name["12x192_mid2_9_hl64"]["b1_ms"]),
            "b4": float(beast_by_name["12x192_mid2_9_hl64"]["b4_ms"]),
            "b8": float(beast_by_name["12x192_mid2_9_hl64"]["b8_ms"]),
            "b16": float(beast_by_name["12x192_mid2_9_hl64"]["b16_ms"]),
            "nps": float(h2h_bq_a2["nps_a"]),
        },
        {
            "label": "10x224\n(INT8)", "params_m": float(beast_by_name["10x224_mid2_7_hl64"]["param_millions"]),
            "b1": float(beast_by_name["10x224_mid2_7_hl64"]["b1_ms"]),
            "b4": float(beast_by_name["10x224_mid2_7_hl64"]["b4_ms"]),
            "b8": float(beast_by_name["10x224_mid2_7_hl64"]["b8_ms"]),
            "b16": float(beast_by_name["10x224_mid2_7_hl64"]["b16_ms"]),
            "nps": float(beast_by_name["10x224_mid2_7_hl64"]["measured_in_engine_gear_shift_v3_nps"]),
        },
        {
            "label": "14x192\n(INT8)", "params_m": float(beast_by_name["14x192_mid2_11_hl64"]["param_millions"]),
            "b1": float(beast_by_name["14x192_mid2_11_hl64"]["b1_ms"]),
            "b4": float(beast_by_name["14x192_mid2_11_hl64"]["b4_ms"]),
            "b8": float(beast_by_name["14x192_mid2_11_hl64"]["b8_ms"]),
            "b16": float(beast_by_name["14x192_mid2_11_hl64"]["b16_ms"]),
            "nps": float(beast_by_name["14x192_mid2_11_hl64"]["measured_in_engine_gear_shift_v3_nps"]),
        },
        {
            "label": "12x224\n(INT8)", "params_m": float(beast_by_name["12x224_mid2_9_hl64"]["param_millions"]),
            "b1": float(beast_by_name["12x224_mid2_9_hl64"]["b1_ms"]),
            "b4": float(beast_by_name["12x224_mid2_9_hl64"]["b4_ms"]),
            "b8": float(beast_by_name["12x224_mid2_9_hl64"]["b8_ms"]),
            "b16": float(beast_by_name["12x224_mid2_9_hl64"]["b16_ms"]),
            "nps": float(beast_by_name["12x224_mid2_9_hl64"]["measured_in_engine_gear_shift_v3_nps"]),
        },
    ]

    fig = plt.figure(figsize=(11.3, 4.25))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.12], wspace=0.27)

    # --- Panel (a): B=1, 4, 8, 16 latency curves ---
    ax1 = fig.add_subplot(gs[0])
    curve_archs = [a for a in archs if a["label"] != "10x192\n(FP16)"]
    fp16_10x192 = next(a for a in archs if a["label"] == "10x192\n(FP16)")
    px = [a["params_m"] for a in curve_archs]

    ax1.axvspan(7.85, 12.1, color="#FEF2F2", alpha=0.80, zorder=0)
    ax1.axvline(7.85, color=COLORS["vermillion"], linestyle=":", linewidth=1.1, zorder=1)
    ax1.text(10.0, 9.02, "On-Chip ~4 MB CMX SRAM\nSpillover Cliff", ha="center", va="top",
             fontsize=7.6, fontweight="bold", color="#9B1C1C")

    batch_specs = [
        ("b16", "$B=16$ (Search)", COLORS["vermillion"], "o"),
        ("b8", "$B=8$ (Mid Gear)", COLORS["orange"], "s"),
        ("b4", "$B=4$ (Low Gear)", COLORS["blue"], "^"),
        ("b1", "$B=1$ (Root)", COLORS["green"], "D"),
    ]
    for key, lbl, col, mk in batch_specs:
        ys = [a[key] for a in curve_archs]
        ax1.plot(px, ys, color=col, marker=mk, markersize=5.4, linewidth=1.75,
                 markeredgecolor="white", markeredgewidth=0.6, label=lbl, zorder=4)
        ax1.scatter([fp16_10x192["params_m"]], [fp16_10x192[key]],
                    facecolors="white", edgecolors=col, marker=mk, s=42, linewidths=1.35, zorder=5)

    # Clean non-overlapping callouts
    ax1.annotate("10x128\n4.52 ms", xy=(3.20, 4.523), xytext=(0, 10), textcoords="offset points",
                 fontsize=7.1, ha="center", color="#1F2937")
    ax1.annotate("10x192 FP16\n6.26 ms", xy=(7.01, 6.262), xytext=(-38, -5), textcoords="offset points",
                 fontsize=7.1, ha="center", color="#9A3412",
                 arrowprops=dict(arrowstyle="->", color=COLORS["vermillion"], lw=0.8))
    ax1.annotate("10x192 QAT\n5.41 ms", xy=(7.19, 5.412), xytext=(-14, -27), textcoords="offset points",
                 fontsize=7.2, ha="center", color="#111111", fontweight="bold")
    ax1.annotate("12x192\n6.54 ms", xy=(8.52, 6.540), xytext=(4, -26), textcoords="offset points",
                 fontsize=7.1, ha="center", color="#1F2937")
    ax1.annotate("12x224\n7.81 ms", xy=(11.48, 7.813), xytext=(-10, -26), textcoords="offset points",
                 fontsize=7.1, ha="center", color="#1F2937")

    ax1.set_xlim(2.5, 12.1)
    ax1.set_ylim(0.0, 9.35)
    ax1.set_xlabel("Model Parameter Count (Millions)")
    ax1.set_ylabel("Intel NPU 3720 Median Turnaround Latency (ms)")
    ax1.set_title("(a) Static-Graph Batch Latency Scaling (NPU 3720)", pad=8)
    ax1.grid(True, zorder=1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper left", fontsize=7.3, title="Batch (open = 10x192 FP16)", title_fontsize=7.1)

    # --- Panel (b): Compute-Optimal NPU Pareto Frontier & H2H Elo ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axvspan(7.85, 12.1, color="#FEF2F2", alpha=0.80, zorder=0)
    ax2.axvline(7.85, color=COLORS["vermillion"], linestyle=":", linewidth=1.1, zorder=1)
    ax2.axhline(3250.0, color=COLORS["gray"], linestyle="--", linewidth=1.1, zorder=2)
    ax2.text(11.95, 3295, "3,250 NPS Gate", ha="right", va="bottom",
             fontsize=7.5, color="#4B5563", fontweight="bold")

    cold_pts = [
        (7.19, float(beast["baseline_10x192_hybrid_gear_shift_v3_nps"])),
        (8.52, float(beast_by_name["12x192_mid2_9_hl64"]["measured_in_engine_gear_shift_v3_nps"])),
        (9.62, float(beast_by_name["10x224_mid2_7_hl64"]["measured_in_engine_gear_shift_v3_nps"])),
        (9.88, float(beast_by_name["14x192_mid2_11_hl64"]["measured_in_engine_gear_shift_v3_nps"])),
        (11.48, float(beast_by_name["12x224_mid2_9_hl64"]["measured_in_engine_gear_shift_v3_nps"])),
    ]
    ax2.plot(
        [p[0] for p in cold_pts], [p[1] for p in cold_pts],
        color=COLORS["gray"], linestyle="--", linewidth=1.2, alpha=0.85, zorder=2,
        label="Cold-Start Gear-Shift Sweep (INT8)",
    )
    ax2.scatter(
        [p[0] for p in cold_pts[1:]], [p[1] for p in cold_pts[1:]],
        facecolors="white", edgecolors=COLORS["charcoal"], marker="o", s=46, linewidths=1.3, zorder=4,
    )

    ax2.scatter([3.20], [4810.4], color=COLORS["sky"], edgecolor="#111111", s=72, marker="s", zorder=5,
                label="FP16 Baselines (H2H / Prod)")
    ax2.scatter([7.01], [float(h2h_180["nps_b"])], color=COLORS["blue"], edgecolor="#111111", s=78, marker="s", zorder=5)
    nps_12_fp16 = float(h2h_b_346["nps_a"])
    ax2.scatter([8.55], [nps_12_fp16], color=COLORS["purple"], edgecolor="#111111", s=78, marker="s", zorder=5)
    nps_12_int8 = float(h2h_bq_a2["nps_a"])
    ax2.scatter([8.52], [nps_12_int8], color=COLORS["orange"], edgecolor="#111111", s=85, marker="D", zorder=6,
                label="12x192 Mid-2..9 QAT INT8 (H2H)")
    nps_10_qat = float(ladder["mean_effective_nps"])
    ax2.scatter([7.19], [nps_10_qat], color=COLORS["green"], edgecolor="#111111", s=210, marker="*", zorder=7,
                label="10x192 Mid-2..7 QAT INT8 (Flagship)")

    ax2.annotate("10x128 FP16\n(3.20M, 4,810 NPS)\n45.5% vs 10x192",
                 xy=(3.20, 4810.4), xytext=(3.60, 4390), fontsize=7.1, color="#1F2937",
                 arrowprops=dict(arrowstyle="->", color="#6B7280", lw=0.9))
    ax2.annotate("10x192 QAT INT8\n(7.19M, 3,988 NPS)\nPareto Optimum",
                 xy=(7.02, nps_10_qat), xytext=(3.10, 3830), fontsize=7.4, fontweight="bold",
                 color="#006B4F",
                 bbox=dict(boxstyle="round,pad=0.22", fc="#E6F7F1", ec=COLORS["green"], lw=0.8),
                 arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.2))
    ax2.annotate("10x192 FP16\n(3,254 NPS)", xy=(7.01, 3253.9), xytext=(4.85, 2900),
                 fontsize=7.2, color=COLORS["blue"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=1.0))
    ax2.annotate("12x192 QAT INT8\n(3,324 NPS)\n-13.9 Elo vs A2",
                 xy=(8.52, nps_12_int8), xytext=(9.15, 3720), fontsize=7.1, color="#9A5B00",
                 arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=1.0))
    ax2.annotate("12x192 FP16 (2,698 NPS)\n+48.9 Elo vs 10x192 FP16\n-77.7 Elo vs 10x192 QAT",
                 xy=(8.55, nps_12_fp16), xytext=(8.65, 1955), fontsize=7.0, color="#6B2158",
                 bbox=dict(boxstyle="round,pad=0.22", fc="#FDF2F8", ec=COLORS["purple"], lw=0.75),
                 arrowprops=dict(arrowstyle="->", color=COLORS["purple"], lw=1.0))

    for lbl, (x0, y0), (dx, dy) in [
        ("10x224\n2,671", (9.62, 2670.5), (-4, 14)),
        ("14x192\n2,533", (9.88, 2533.0), (26, 20)),
        ("12x224\n2,362", (11.48, 2362.3), (-2, 16)),
    ]:
        ax2.annotate(lbl, xy=(x0, y0), xytext=(dx, dy), textcoords="offset points",
                     fontsize=6.9, color="#4B5563", ha="center")

    ax2.annotate("", xy=(7.19, nps_10_qat - 95), xytext=(7.01, 3253.9 + 95),
                 arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.5))
    ax2.text(6.68, 3535, "+56.5\nElo", fontsize=7.2, fontweight="bold", color="#006B4F", ha="right")

    ax2.set_xlim(2.5, 12.1)
    ax2.set_ylim(1800, 5150)
    ax2.set_xlabel("Model Parameter Count (Millions)")
    ax2.set_ylabel("In-Engine Search Throughput (NPS, 400 ms/move)")
    ax2.set_title("(b) Compute-Optimal NPU Pareto Frontier & H2H Elo", pad=8)
    ax2.grid(True, zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower left", fontsize=7.0)

    save_figure(fig, "fig2_npu_pareto_frontier")


# ---------------------------------------------------------------------------
# Figure 3: 2x2 QAT Causal Factorial & 180-Game Pentanomial SPRT Trajectory
# ---------------------------------------------------------------------------
def make_fig3() -> None:
    a1_fp16 = load_json("phase4_a1_fp16_hl64_syzygy_sf3000_50g.json")
    ptq_mid = load_json("phase4_sf3000_openings320_apples_50g.json")
    a2_fp16 = load_json("phase4_a2_fp16_hl64_syzygy_sf3000_50g.json")
    a2_int8 = load_json("phase4_a2_mid27_qat_hl64_syzygy_sf3000_50g.json")
    base_fp16 = load_json("phase4_fp16_syzygy_sf3000_50g.json")
    h2h_comb = load_json("phase4_h2h_a2int8_vs_346m_180g_combined.json")

    fig = plt.figure(figsize=(11.3, 4.25))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)

    # --- Panel (a): 2x2 Causal Factorial vs Stockfish 19 UCI_Elo 3000 ---
    ax1 = fig.add_subplot(gs[0])
    groups = ["Frozen Trunk\n(A1 / PTQ, No QAT)", "Joint Trunk\n(A2, 40M Finetune)"]
    fp16_scores = [float(a1_fp16["score_pct"]), float(a2_fp16["score_pct"])]
    int8_scores = [float(ptq_mid["score_pct"]), float(a2_int8["score_pct"])]
    fp16_nps = [float(a1_fp16["effective_nps"]), float(a2_fp16["effective_nps"])]
    int8_nps = [float(ptq_mid["effective_nps"]), float(a2_int8["effective_nps"])]
    fp16_eg = [int(a1_fp16["late_endgame_losses"]), int(a2_fp16["late_endgame_losses"])]
    int8_eg = [int(ptq_mid["late_endgame_losses"]), int(a2_int8["late_endgame_losses"])]

    x = np.arange(len(groups))
    w = 0.32

    ref_score = float(base_fp16["score_pct"])
    ax1.axhline(ref_score, color=COLORS["charcoal"], linestyle="--", linewidth=1.2, zorder=2)
    ax1.text(-0.50, ref_score + 0.9, f"v1.0.2 Scalar FP16 Ref: {ref_score:.1f}% (3,293 NPS)",
             ha="left", va="bottom", fontsize=7.3, fontweight="bold", color=COLORS["charcoal"])

    b1 = ax1.bar(
        x - w / 2, fp16_scores, width=w, color=COLORS["blue"], edgecolor="#222222",
        linewidth=0.7, label="Pure FP16", zorder=3,
    )
    b2 = ax1.bar(
        x + w / 2, int8_scores, width=w, color=[COLORS["vermillion"], COLORS["green"]],
        edgecolor="#222222", linewidth=0.7, label="Mid-2..7 INT8 (Red=PTQ, Green=QAT)", zorder=3,
    )

    for rect, sc, nps, eg in zip(b1, fp16_scores, fp16_nps, fp16_eg):
        cx = rect.get_x() + rect.get_width() / 2
        ax1.text(cx, sc + 0.8, f"{sc:.1f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=COLORS["blue"])
        ax1.text(cx, sc * 0.62, f"{nps:,.0f}\nNPS\n({eg} EG L)", ha="center", va="center",
                 fontsize=7.2, color="white", fontweight="bold")

    for i, (rect, sc, nps, eg) in enumerate(zip(b2, int8_scores, int8_nps, int8_eg)):
        cx = rect.get_x() + rect.get_width() / 2
        col = COLORS["vermillion"] if i == 0 else "#006B4F"
        ax1.text(cx, sc + 0.8, f"{sc:.1f}%", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=col)
        ax1.text(cx, sc * 0.62, f"{nps:,.0f}\nNPS\n({eg} EG L)", ha="center", va="center",
                 fontsize=7.2, color="white", fontweight="bold")

    # Non-overlapping callouts placed above y=52, arrows ending cleanly above bar labels
    ax1.annotate(
        "PTQ Drop vs. FP16 Ref:\n$-12.0\\%$ (25% on [75:85])",
        xy=(0 + w / 2, 37.5), xytext=(-0.02, 54.0),
        ha="center", fontsize=7.3, fontweight="bold", color="#9B1C1C",
        bbox=dict(boxstyle="round,pad=0.24", fc="#FDECEC", ec=COLORS["vermillion"], lw=0.8),
        arrowprops=dict(arrowstyle="->", color=COLORS["vermillion"], lw=1.2),
    )
    ax1.annotate(
        "Joint QAT Reversal:\n$+2.0\\%$ Score, $+16.2\\%$ NPS",
        xy=(1 + w / 2, 44.7), xytext=(1.02, 54.0),
        ha="center", fontsize=7.3, fontweight="bold", color="#006B4F",
        bbox=dict(boxstyle="round,pad=0.24", fc="#E6F7F1", ec=COLORS["green"], lw=0.8),
        arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.2),
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups)
    ax1.set_xlim(-0.55, 1.55)
    ax1.set_ylim(0, 62.5)
    ax1.set_ylabel("Score vs. Stockfish 19 UCI_Elo 3000 (%)")
    ax1.set_title("(a) 2×2 Trunk × Precision Causal Matrix (50g/Arm)", pad=8)
    ax1.grid(axis="y", zorder=1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.50, 0.01), fontsize=7.3)

    # --- Panel (b): Cumulative 180-Game Pentanomial SPRT LLR Curve ---
    ax2 = fig.add_subplot(gs[1])

    pair_scores: dict[int, list[float]] = {}
    pair_order: list[int] = []
    for sname in h2h_comb["slices"]:
        sd = load_json(sname)
        for g in sd["games"]:
            pid = int(g["pair"])
            if pid not in pair_scores:
                pair_scores[pid] = []
                pair_order.append(pid)
            pair_scores[pid].append(float(g["a_score"]))

    counts = [0, 0, 0, 0, 0]
    games_x = [0]
    llrs = [0.0]
    cross_x, cross_y = None, None
    upper = float(h2h_comb["sprt"]["upper"])
    lower = float(h2h_comb["sprt"]["lower"])

    for idx_p, pid in enumerate(pair_order, start=1):
        sc_sum = sum(pair_scores[pid])
        bucket = int(round(sc_sum * 2))
        counts[bucket] += 1
        llr = compute_pentanomial_sprt_llr(counts, elo0=0.0, elo1=20.0)
        games_x.append(idx_p * 2)
        llrs.append(llr)
        if cross_x is None and llr >= upper:
            cross_x, cross_y = idx_p * 2, llr

    assert counts == h2h_comb["pentanomial"], f"Pentanomial mismatch: {counts}"
    assert abs(llrs[-1] - float(h2h_comb["sprt"]["llr"])) < 1e-3, f"LLR mismatch: {llrs[-1]}"

    ax2.axhspan(upper, 4.95, color="#E6F7F1", alpha=0.80, zorder=0)
    ax2.axhspan(-3.60, lower, color="#FDECEC", alpha=0.75, zorder=0)
    ax2.axhline(upper, color=COLORS["green"], linestyle="--", linewidth=1.3, zorder=2)
    ax2.axhline(lower, color=COLORS["vermillion"], linestyle="--", linewidth=1.1, zorder=2)
    ax2.axhline(0.0, color=COLORS["gray"], linestyle=":", linewidth=0.9, zorder=1)

    for b_g in (50, 100):
        ax2.axvline(b_g, color="#9CA3AF", linestyle=":", linewidth=0.9, zorder=1)
    ax2.text(25, -2.55, "Slice 1 (50g)", ha="center", fontsize=7.2, color="#4B5563")
    ax2.text(75, -2.55, "Slice 2 (50g)", ha="center", fontsize=7.2, color="#4B5563")
    ax2.text(140, -2.55, "Slice 3 (80g)", ha="center", fontsize=7.2, color="#4B5563")

    ax2.plot(games_x, llrs, color=COLORS["blue"], linewidth=2.1, zorder=4,
             label="Cumulative Pentanomial LLR")
    ax2.fill_between(games_x, 0, llrs, where=[v >= 0 for v in llrs],
                     color=COLORS["blue"], alpha=0.10, zorder=3)

    if cross_x is not None and cross_y is not None:
        ax2.scatter([cross_x], [cross_y], color=COLORS["green"], edgecolor="#111111",
                    s=120, marker="*", zorder=6, label=f"ACCEPT $H_1$ @ Game {cross_x}")
    ax2.scatter([games_x[-1]], [llrs[-1]], color=COLORS["blue"], edgecolor="white",
                s=52, zorder=6)

    # Clear non-overlapping threshold labels
    ax2.text(4, upper + 0.16, f"ACCEPT $H_1$ (+20 Elo): LLR $\\geq +{upper:.2f}$",
             fontsize=7.5, fontweight="bold", color="#006B4F")
    ax2.text(4, lower - 0.42, f"REJECT $H_1$ (0 Elo): LLR $\\leq {lower:.2f}$",
             fontsize=7.5, fontweight="bold", color="#9B1C1C")

    summary_txt = (
        f"A2 Mid-2..7 QAT INT8 vs. 346M FP16\n"
        f"180 Games: {h2h_comb['wins']}W–{h2h_comb['draws']}D–{h2h_comb['losses']}L "
        f"({h2h_comb['score_pct']:.2f}%, +{h2h_comb['elo_diff']:.1f} Elo)\n"
        f"Pentanomial: {h2h_comb['pentanomial']}\n"
        f"Final LLR = +{llrs[-1]:.4f} (PASS)  |  +19.6% NPS"
    )
    ax2.text(
        0.03, 0.19, summary_txt, transform=ax2.transAxes,
        fontsize=7.2, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.33", fc="white", ec=COLORS["blue"], lw=0.9, alpha=0.95),
        zorder=7,
    )

    ax2.set_xlim(0, 184)
    ax2.set_ylim(-3.60, 4.90)
    ax2.set_xlabel("Paired NPU Games Completed (400 ms/move, Static $B=16$, Syzygy)")
    ax2.set_ylabel("Pentanomial SPRT Log-Likelihood Ratio (LLR)")
    ax2.set_title("(b) 180-Game Paired NPU Championship SPRT Trajectory", pad=8)
    ax2.grid(axis="x", alpha=0.35, zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    # Place legend cleanly in open upper-left BELOW the ACCEPT H1 green text
    ax2.legend(loc="upper left", bbox_to_anchor=(0.01, 0.74), fontsize=7.4)

    save_figure(fig, "fig3_qat_factorial_and_sprt")


# ---------------------------------------------------------------------------
# Figure 4: Clean 1,979 Puzzles, Co-Resident Gear-Shifting & Stockfish Ladder
# ---------------------------------------------------------------------------
def make_fig4() -> None:
    pz1979 = load_json("v1_1_puzzles_clean1979.json")
    spikes3 = load_json("research_spikes_phase3.json")
    ladder = load_json("v1_1_stockfish_ladder_150g.json")
    nodes3500 = load_json("v1_1_sf_nodes3500_40g.json")

    fig = plt.figure(figsize=(12.0, 4.25))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.12, 0.88, 1.35], wspace=0.34)

    # --- Panel (a): 1,979 Clean Lichess Puzzles Node Scaling ---
    ax1 = fig.add_subplot(gs[0])
    budgets = ["policy_top1_0n", "50n", "200n", "800n"]
    blabels = ["0n\n(Policy)", "50n", "200n", "800n"]
    overall = [float(pz1979["results"][b]["pct"]) for b in budgets]
    lt1200 = [float(pz1979["results"][b]["by_bucket"]["<1200"]["pct"]) for b in budgets]
    ge2100 = [float(pz1979["results"][b]["by_bucket"]["2100+"]["pct"]) for b in budgets]

    x1 = np.arange(len(budgets))
    ax1.bar(x1, overall, width=0.52, color=COLORS["blue"], alpha=0.84,
            edgecolor="#222222", linewidth=0.65, label="Overall ($n=1{,}979$)", zorder=3)
    ax1.plot(x1, lt1200, color=COLORS["green"], marker="^", markersize=5.6, linewidth=1.65,
             linestyle="-", label="Rating $<1200$", zorder=5)
    ax1.plot(x1, ge2100, color=COLORS["vermillion"], marker="s", markersize=5.2, linewidth=1.65,
             linestyle="--", label="Rating $2100+$", zorder=5)

    for xi, (ov, hi, lo) in enumerate(zip(overall, lt1200, ge2100)):
        if xi < 3:
            ax1.text(xi, ov + 1.3, f"{ov:.2f}%", ha="center", va="bottom",
                     fontsize=7.4, fontweight="bold", color=COLORS["blue"])
        else:
            # Place 95.65% cleanly inside the top of the 800n bar so it never collides with 98.48% triangle
            ax1.text(xi, ov - 2.2, f"{ov:.2f}%", ha="center", va="top",
                     fontsize=7.5, fontweight="bold", color="white", zorder=6)
            ax1.text(xi, hi + 1.4, f"{hi:.1f}%", ha="center", va="bottom",
                     fontsize=7.0, fontweight="bold", color="#006B4F")
        # High-contrast pill label for 2100+ points
        ax1.text(
            xi, lo - 4.0, f"{lo:.1f}%", ha="center", va="top",
            fontsize=6.9, color="#9B1C1C", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=COLORS["vermillion"], lw=0.5, alpha=0.92),
            zorder=6,
        )

    ax1.set_xticks(x1)
    ax1.set_xticklabels(blabels)
    ax1.set_ylim(42, 106.5)
    ax1.set_xlabel("PUCT Search Budget (A2_soup75 INT8)")
    ax1.set_ylabel("Solve Accuracy (%)")
    ax1.set_title("(a) 1,979 Clean Lichess Puzzles", pad=8)
    ax1.grid(axis="y", zorder=1)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower right", fontsize=7.1)

    # --- Panel (b): Co-Resident B=4/8/16 Gear-Shifting ---
    ax2 = fig.add_subplot(gs[1])
    s1 = spikes3["phase3_step1_hybrid_gear_shift_v3"]
    gs_arms = [
        ("V0_Pure_FP16_baseline_b16_prod_db", "Static $B{=}16$ FP16", COLORS["gray"]),
        ("V0_Pure_FP16_gear_shift_v3", "Gear $B{=}4/8/16$ FP16", COLORS["sky"]),
        ("Hybrid_Mid27_ConvOnly_gear_shift_v3", "Gear + Mid2..7 INT8", COLORS["green"]),
    ]
    x2 = np.arange(2)
    w2 = 0.25
    for idx_a, (akey, albl, acol) in enumerate(gs_arms):
        vals = [
            float(s1["clean_500_puzzles_50ms"][akey]["accuracy_pct"]),
            float(s1["clean_500_puzzles_100ms"][akey]["accuracy_pct"]),
        ]
        rects = ax2.bar(x2 + (idx_a - 1) * w2, vals, width=w2, color=acol,
                        edgecolor="#222222", linewidth=0.6, label=albl, zorder=3)
        for r, v in zip(rects, vals):
            ax2.text(r.get_x() + r.get_width() / 2, v + 0.55, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=7.0,
                     fontweight="bold" if idx_a == 2 else "normal", color="#1F2937")

    ax2.annotate("+8.0%", xy=(0 + w2, 89.8), xytext=(0, 96.6), ha="center",
                 fontsize=7.5, fontweight="bold", color="#006B4F",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#E6F7F1", ec=COLORS["green"], lw=0.7),
                 arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.0))
    ax2.annotate("+7.0%", xy=(1 + w2, 94.3), xytext=(1, 99.6), ha="center",
                 fontsize=7.5, fontweight="bold", color="#006B4F",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#E6F7F1", ec=COLORS["green"], lw=0.7),
                 arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.0))

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["50 ms / move\n($n=500$)", "100 ms / move\n($n=500$)"])
    ax2.set_ylim(72, 104.5)
    ax2.set_ylabel("Solve Accuracy (%)")
    ax2.set_title("(b) Co-Resident Gear-Shifting", pad=8)
    ax2.grid(axis="y", zorder=1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", fontsize=6.9)

    # --- Panel (c): Stockfish 19 Ladder & go nodes 3500 Crossover ---
    ax3 = fig.add_subplot(gs[2])
    cats = ["SF19\nElo 2800\n(50g)", "SF19\nElo 3000\n(50g)", "SF19\nElo 3190\n(50g)", "SF19\nNodes 3500\n(40g)"]
    v10_scores = [62.0, 49.0, 39.0, 37.50]
    v11_scores = [
        float(ladder["tiers"]["2800"]["score_pct"]),
        float(ladder["tiers"]["3000"]["score_pct"]),
        float(ladder["tiers"]["3190"]["score_pct"]),
        float(nodes3500["score_pct"]),
    ]
    v11_cis = [
        ladder["tiers"]["2800"]["wilson_95_ci_pct"],
        ladder["tiers"]["3000"]["wilson_95_ci_pct"],
        ladder["tiers"]["3190"]["wilson_95_ci_pct"],
        nodes3500["wilson_95_ci_pct"],
    ]
    yerr_lo = [s - ci[0] for s, ci in zip(v11_scores, v11_cis)]
    yerr_hi = [ci[1] - s for s, ci in zip(v11_scores, v11_cis)]

    x3 = np.arange(len(cats))
    w3 = 0.34
    ax3.axvline(2.5, color="#9CA3AF", linestyle="--", linewidth=1.0, zorder=2)
    ax3.axhline(50.0, color=COLORS["charcoal"], linestyle=":", linewidth=0.9, zorder=1)

    r_old = ax3.bar(x3 - w3 / 2, v10_scores, width=w3, color="#CBD5E1", edgecolor="#333333",
                    linewidth=0.65, label="v1.0.x Pure FP16*", zorder=3)
    r_new = ax3.bar(x3 + w3 / 2, v11_scores, width=w3,
                    color=[COLORS["blue"], COLORS["blue"], COLORS["blue"], COLORS["green"]],
                    edgecolor="#111111", linewidth=0.65,
                    yerr=[yerr_lo, yerr_hi], capsize=3.0, error_kw=dict(lw=1.0, ecolor="#111111"),
                    label="v1.1.0 A2_soup75 INT8 (95% CI)", zorder=4)

    for r, v in zip(r_old, v10_scores):
        ax3.text(r.get_x() + r.get_width() / 2, v + 1.4, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=7.0, color="#4B5563")
    for idx_b, (r, v, hi) in enumerate(zip(r_new, v11_scores, yerr_hi)):
        ax3.text(r.get_x() + r.get_width() / 2, v + hi + 1.6, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=7.4, fontweight="bold",
                 color="#006B4F" if idx_b == 3 else COLORS["blue"])

    # Clean non-overlapping boxes in upper space
    ax3.text(
        1.15, 113.0,
        f"Mean Implied Elo: {ladder['mean_implied_elo']:.0f}\n(3,988 NPS, 150g unseen book)",
        ha="center", va="top", fontsize=7.1, color="#1F2937",
        bbox=dict(boxstyle="round,pad=0.24", fc="#F8FAFC", ec="#CBD5E1", lw=0.75),
    )
    ax3.annotate(
        "+46.25%\nCrossover!",
        xy=(3 + w3 / 2 - 0.14, 83.0), xytext=(2.18, 76.0),
        ha="center", fontsize=7.4, fontweight="bold", color="#006B4F",
        bbox=dict(boxstyle="round,pad=0.22", fc="#E6F7F1", ec=COLORS["green"], lw=0.8),
        arrowprops=dict(arrowstyle="->", color=COLORS["green"], lw=1.15),
    )

    ax3.set_xticks(x3)
    ax3.set_xticklabels(cats, fontsize=7.7)
    ax3.set_ylim(0, 117.0)
    ax3.set_ylabel("Match Score vs. Stockfish 19 (%)")
    ax3.set_title("(c) Stockfish 19 Ladder & Crossover", pad=8)
    ax3.grid(axis="y", zorder=1)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="lower left", fontsize=6.9)

    save_figure(fig, "fig4_puzzles_and_ladder")


def main() -> None:
    configure_style()
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print("[DONE] Generated all 4 publication figures in paper/figures/.")


if __name__ == "__main__":
    main()
