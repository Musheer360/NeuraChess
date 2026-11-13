# NeuraChess Experimental Log & Negative Results

This file summarizes every major empirical evaluation, hardware benchmark, search ablation, and negative result recorded during the development of NeuraChess. All win-rates report 95% Wilson confidence intervals, and all offline cross-entropies mask illegal move logits to `-10000.0`.

---

## 1. Training Formulation & Policy Head Ablations (5M Slice, `6x128`)

Before scaling training data, controlled 5M-position ablations compared the legacy dense linear policy head (`bias=False`) against `Conv73` + BatchNorm and soft-target Boltzmann legal-masked cross-entropy (`T=0.5`):

| Arm | Configuration | `far_test` CE | `far_test` Top-1 | Puzzle Solve % |
| --- | --- | ---: | ---: | ---: |
| Arm 1 | Legacy dense policy head, no BN, hard targets | 2.4539 | 32.47% | — |
| Arm 3 | `Conv73` + BN, hard targets | 2.2400 | 35.01% | — |
| Arm 4 | `Conv73` + BN, unmasked soft targets | 2.9642 | 17.27% | — |
| **Arm 5** | **`Conv73` + BN, legal-masked Boltzmann soft targets** | **2.2184** | **35.12%** | **37.65%** |

Leaving illegal logits unmasked under soft-target distributions (Arm 4) severely degraded policy calibration (`2.9642` CE); legal masking combined with `Conv73` and BatchNorm (Arm 5) achieved the lowest cross-entropy and served as the permanent training objective.

---

## 2. NPU Throughput & Architecture Sweep (`Intel AI Boost NPU`, `B=16`)

### 2.1 Convolutional vs Transformer Cost Model (`npu_cost_model`)

On the Intel NPU 3720 (static graph compilation under OpenVINO `FP16`), scaled dot-product attention and softmax layers incurred significant execution stalls relative to depthwise/spatial `3x3` convolutions:

| Architecture Variant | Params | `B=16` Async NPU (`pos/s`) | Effective TFLOPS | Relative to `10x128` CNN |
| --- | ---: | ---: | ---: | ---: |
| `1_cnn_6x128_legacy_dense_pol` | 11.55M | 5,080.7 | 1.267 | 1.33x faster (large weights) |
| `2_cnn_6x128_conv73_pol_q` | 2.14M | 5,985.2 | 1.502 | +17.8% vs dense policy |
| `3_cnn_10x128_conv73_pol_q` | 3.20M | **3,831.2** | **1.541** | **1.00x (baseline)** |
| `5_tf_6x192_h6_relbias_pol_q` | 2.00M | 1,351.0 | 0.336 | 2.84x slower |
| `6_tf_6x192_h6_smolgen_pol_q` | — | 1,164.0 | 0.290 | 3.29x slower |
| `7_tf_8x256_h8_relbias_pol_q` | — | 722.9 | — | 5.30x slower |
| `8c_tf_loop_d192_L2_K4` | — | 1,004.2 | — | 3.82x slower |
| `9a_hybrid_cnn4_tfloop_d192_K2` | — | 1,647.1 | — | 2.33x slower |

Because pure SE-ResNet backbones achieved 3x–5x higher NPU compute utilization than transformers/hybrids at equivalent parameter counts, transformers were ruled out for NPU search.

### 2.2 Depth vs Width Sweep (Equal Wall-Clock & 70M Stage 2)

At equal GPU training wall-clock (`790.2s`), `10x128` (`17,959.1` train pos/s) outperformed `6x128` (`25,296.0` pos/s) on `far_test` (`1.9063` vs `1.9495`, improving further to `1.8650` / `43.68%` at full epoch). A broader Stage 1 / Stage 2 scaling sweep across 5 CNN shapes verified how NPU throughput traded off against policy/value accuracy at 70M positions:

| Model | Params | `B=16` NPU Throughput (`pos/s`) | `far_test` CE (Top-1) | `in_region_v2` CE (Top-1) | `heldout_400m` CE (Top-1) | `far_test` Val MSE (Sign%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `6x128 (S5a)` | 1.99M | — | 1.7971 (45.47%) | 1.8337 (43.78%) | — | 0.0852 (92.87%) |
| `10x128` | 3.20M | 3,337.8 (100.0%) | — | — | — | — |
| `8x192 (Stage 2)` | 5.64M | 2,863.3 (85.8%) | 1.6766 (48.24%) | 1.7171 (46.64%) | 1.6899 (46.25%) | 0.0710 (94.09%) |
| `14x128 (Stage 2)` | 4.41M | 2,516.8 (75.4%) | 1.6706 (48.18%) | 1.7071 (46.69%) | 1.6796 (46.48%) | 0.0700 (94.14%) |
| **`10x192 (Stage 2)`** | **7.01M** | **2,348.3 (70.4%)** | **1.6504 (49.20%)** | **1.6905 (47.40%)** | **1.6629 (46.89%)** | **0.0676 (94.50%)** |
| `12x192` | 8.37M | 2,004.7 (60.1%) | — | — | — | — |

### 2.3 Full 346M Training (`10x192` vs `10x128`, `ema_09999`)

Both top Pareto shapes were trained across 346M Lichess evaluation positions (`ema_09999`) and evaluated across all 150k benchmark positions (`50k far_test`, `49,998 in_region_v2`, `50k heldout_400m`) plus 100 paired games at `400 ms/move` (Ponder OFF):

| Metric | `Cont 346M 10x128` | `Full 346M 10x192` | Delta |
| --- | ---: | ---: | ---: |
| `far_test` CE / Top-1 / Top-3 / Sign% | 1.6163 / 49.72% / 78.80% / 94.70% | **1.5655 / 51.43% / 80.17% / 95.32%** | -0.0508 CE |
| `in_region_v2` CE / Top-1 / Top-3 / Sign% | 1.6466 / 48.48% / 77.59% / 94.61% | **1.5887 / 50.16% / 79.13% / 95.12%** | -0.0579 CE |
| `heldout_400m` CE / Top-1 / Top-3 / Sign% | 1.6057 / 48.29% / 77.76% / 95.49% | **1.5522 / 49.92% / 79.41% / 96.03%** | -0.0535 CE |
| 1,979 Clean Puzzles (`1n` policy / 50n / 200n / 800n) | 67.61% / 77.36% / 82.82% / 94.34%* | **71.45% / 79.38% / 84.03% / 95.96%** | +3.84% (1n) |
| Equal-time H2H (`400 ms/move`, 100g, Ponder OFF) | 45.50% (4,810.4 eff NPS) | **54.50% (`38W-33D-29L`, CI `[44.75%, 63.91%]`, 3,290.3 eff NPS)** | +9.00% (n.s., CI crosses 50%) |

Across the three 50k evaluation splits, `Full 346M 10x192` lowers legal-masked cross-entropy by `-0.0508`, `-0.0579`, and `-0.0535` (mean `-0.0541` CE). Because the `n=100` equal-time H2H 95% Wilson CI (`[44.75%, 63.91%]`) spans `50.0%`, it does **not** establish a statistically significant playing-strength advantage over `10x128`; `10x192` was selected on consistent offline policy/value improvements and clean puzzle solve rate (`95.96%` at `800n`). On a separate 2,000-puzzle stratified slice, `Full 346M 10x192` achieved **71.45%** policy Top-1 overall (**92.00%** on `<1200` rating, **56.25%** on `2100+` rating) and **79.40%** at 50 PUCT nodes. For `Cont 346M 10x128`, `1n` denotes raw policy top-1 (`67.61%`), and `50n`/`200n` (`77.36%` / `82.82%`) reflect sequential production defaults (`use_async_overlap=False, nireq=1, fpu_reduction=0.20`); `*` `800n` (`94.34%`) is from the earlier `nireq=2` run (Section 3.5).

---

## 3. Search & Infrastructure Ablations

1. **Inference Batch Size (`B=16` vs `B=32` / `B=64`):**
   Although larger batches increase raw pos/s on NPU, batched PUCT suffers selection staleness (virtual-loss collision) and reduces sequential tree depth per millisecond:
   - `400 ms/move`: `B=32` scored **26.56%** (`8.5 / 32`, CI `[14.40%, 43.75%]`) vs `B=16` (early-stopped at 32 games).
   - `200 ms/move`: `B=32` scored **27.78%** (`18g`) and `B=64` scored **0.00%** (`16g`) against `B=16`.

2. **Subtree Reuse & 64 MB Zobrist NN Cache (Pilot Model):**
   - At `200 ms/move` (24 games): Tree Reuse + NN Cache scored **70.83%** (`15W-4D-5L`, estimated `+154.1` Elo).
   - At `400 ms/move` (24 games): **47.92%** (`9W-5D-10L`). Both optimizations remain active by default with zero regression on longer controls and major gains on fast search/ponder hits.

3. **Pondering A/B vs Stockfish 19 `UCI_Elo 2800` (`400 ms/move`, 50 games each):**
   - `Ponder ON`: **62.00%** (`31.0/50`, CI `[48.15%, 74.14%]`), `785.7` mean starting retained nodes from opponent-turn pondering (`2,896.5` ponder NPS, `45.07%` top-1 / `74.78%` top-3 opponent-move ponder hit rate).
   - `Ponder OFF`: **64.00%** (`32.0/50`, CI `[50.14%, 75.86%]`), `305.6` retained nodes from prior own search. Within statistical noise at 50 games; UCI exposes `Ponder` as an explicit runtime toggle.

4. **Opening Root Top-K Guard (`use_opening_root_guard`, Rejected):**
   Gating opening root expansion to raw Policy Top-4 + `prior >= 0.08` recovered only **22 / 28** opening blunder positions (below the pre-registered `24 / 28` threshold; both arms scored `100 / 100` on Mate-in-1/2 and `291 / 300` on Mate-in-3). Per decision rule, `use_opening_root_guard` remains `False` by default.

5. **Root-Cause Diagnosis of Archived `Cont 346M 10x128` 50-Node Dip (`58.16%` -> `77.36%`):**
   Comparing exact archived benchmark configs showed why only `Cont 346M 10x128` dipped at `50n` (`58.16%`, `200n=74.89%`) while `6x128` (`70.54%`) and `Full 346M 10x192` (`79.38%`) did not: both `6x128` and `10x192` ran sequentially with `nireq=1` (`use_async_overlap=False`), whereas the archived `10x128` run used Step-3 autotuned `nireq=2` (`use_async_overlap=True, async_inflight=2`) with `fpu_reduction=0.25`. Controlled single-parameter ablation on the 1,979 clean puzzles (`50n`, `batch_size=16`) isolated the cause:
   - Sequential production defaults (`use_async_overlap=False, async_inflight=1, fpu_reduction=0.20`): **77.36%** (`1,531 / 1,979`, `avg_nodes=74.87`, `4.40` batches, `40.09s`), rising to **82.82%** at `200n` (`1,639 / 1,979`, `99.69s`).
   - Sequential control with `fpu_reduction=0.25` (`use_async_overlap=False, async_inflight=1`): **77.36%** (`1,531 / 1,979`, `avg_nodes=74.85`, `4.40` batches, `39.56s`) — zero regression from `fpu_reduction=0.25`.
   - Single-parameter flip to async double-buffering (`use_async_overlap=True, async_inflight=2, fpu_reduction=0.20`): **58.67%** (`1,161 / 1,979`, `avg_nodes=61.57`, `3.57` batches, `33.03s`).
   - Archived config (`use_async_overlap=True, async_inflight=2, fpu_reduction=0.25`): **58.16%** (`1,151 / 1,979`, `avg_nodes=61.57`, `3.57` batches, `33.01s`), reproducing the published figure exactly.

   Under `max_nodes=50`, `use_async_overlap=True` (`nireq=2`) alone accounts for the ~19-point drop (`avg_nodes=61.57` over `3.57` async batches vs `74.87` over `4.40` sequential batches). Sequential execution (`use_async_overlap=False, nireq=1`) is the uniform production default.

---

## 4. Negative Results: Post-Training Quantization, QAT W8A8, and Hyperparameter Drift

### 4.1 Post-Training Quantization Matrix (`10x192`, `V0_FP16` vs `W8A16` / `W8A8`)

| Config | OpenVINO Weight Size | Top-1 Agreement vs FP16 | Max Value Diff | Added Visits/s (`B=16`) | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **`V0_FP16 [LATENCY]`** | **13.36 MB** | **100.00%** | **0.0** | **2,592.2** (`3,028.6` total) | **Selected Default** |
| `A2_W8A16_ASYM` | 6.71 MB | 97.28% | — | 2,479.0 | Slower & below 98.5% gate |
| `B2_W8A8_StemRes` (PTQ) | 7.66 MB | 91.18% | 0.99182 | 2,960.4 | Failed `>=98.5%` agreement gate |

### 4.2 Quantization-Aware Training (QAT W8A8 + Value-Weighted Fine-Tuning) Ablation

To attempt recovering W8A8 accuracy while gaining NPU speed (`4,105.3` vs `3,451.6` NPS, `+18.94%`), `Full 346M 10x192` underwent a 20M-position QAT fine-tune with 2x value loss (`value_loss_weight=2.0`) and `ema_09995`, followed by paired Stockfish 19 `UCI_Elo 3000` causal isolation matches (`50` games each at `400 ms/move`, Ponder ON):

- **150K Offline Eval Drift (`Post-QAT FP16` vs `Full 346M 10x192`):** Mean 150K CE regressed slightly from `1.5688 -> 1.5710` (`+0.0022`), Top-1 `50.50% -> 50.28%` (`-0.22%`), Value MSE `0.0530 -> 0.0536`.
- **QAT W8A8 Acceptance Gate:** Top-1 agreement vs FP16 was `92.52%` (below the `97.5%` acceptance gate); in a 60-game NPU equal-time match under `(c_puct=1.0, fpu_reduction=0.0, t_root=0.85)`, QAT W8A8 scored `51.67%` (`31.0/60`) against Post-QAT FP16.
- **Causal Decomposition vs Stockfish 19 `UCI_Elo 3000` (50 games/arm):**
  - `Full 346M 10x192 FP16` + Default Search (`c_puct=1.5, fpu=0.20, t_root=1.0`): **49.00%** (`16W-17D-17L`)
  - `QAT W8A8` + Tuned Aggressive Search (`c_puct=1.0, fpu=0.00, t_root=0.85`): **40.00%** (`11W-18D-21L`)
  - `QAT W8A8` + Restored Default Search (`c_puct=1.5, fpu=0.20, t_root=1.0`): **43.00%** (`10W-23D-17L`)

The isolated ablation attributes **-6.0%** score loss directly to `W8A8` residual quantization noise (even with QAT) and **-3.0%** to removing FPU reduction / lowering `c_puct`. Both `Full 346M 10x192 V0_FP16` and `(c_puct=1.5, use_fpu_reduction=True, fpu_reduction=0.20, t_root=1.0)` remain the locked production defaults.

---

## 5. Stage 3.5 Loss Anatomy (44 Losses vs Stockfish 19)

All 44 lost games from the 150-game Stockfish 19 calibration (`2800 / 3000 / 3190`) were analyzed at fixed Stockfish depth 14 using corrected mover-perspective evaluation (`eval_stm_to_nc_drop`, verified `< 30 cp` delta between `1. e4` and `1. d4`, avoiding side-to-move sign errors):

| Error Category / Phase | Count | Share (`n=44`) |
| --- | ---: | ---: |
| (a) Search / Value error (Stockfish best in Policy Top-3) | 19 | 43.18% |
| (b) Policy blindspot (Stockfish best outside Policy Top-3) | 5 | 11.36% |
| (c) Tactical horizon (`< 4` plies) | 19 | 43.18% |
| (d) Late endgame grind | 1 | 2.27% |
| Opening (`ply <= 20`) | 27 | 61.36% |
| Middlegame (`20 < ply <= 60`) | 13 | 29.55% |
| Endgame (`ply > 60`) | 4 | 9.09% |

**Calibration & Interval Notes (`v1.0.0` Baseline):**
- Each 50-game Stockfish `UCI_Elo` ladder rung carries a wide 95% Wilson interval (roughly `±95 Elo` binomial half-width, ~190 Elo CI width at `n=50`) and should be read alongside the cumulative 150-game result (`50.00%`, 95% Wilson CI `[42.10%, 57.90%]`) and fixed-node Stockfish rungs.
- All `v1.0.0` Stockfish 19 calibration matches in this document and `README.md` ran with `OwnBook=False` (no opening book) under fixed `movetime=400 ms` (prior to adding the `v1.0.2` CC0 Lichess opening book and cold-start `1.30x` `wtime`/`btime` buffer).
