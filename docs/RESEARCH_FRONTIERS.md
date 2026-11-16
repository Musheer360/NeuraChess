# NeuraChess Research Frontiers & Architectural Audit (`v1.0.2` -> `v1.1`)

This document audits the current **`NeuraChess v1.0.2`** codebase (`Full 346M 10x192 FP16`, C++23 AVX2/BMI2 `fast_tree`) against five next-generation research directions tailored to heterogeneous **Intel Core Ultra (P-core + Intel AI Boost NPU 3720)** execution.

---

## Baseline Failure & Utilization Profile (`NeuraChess v1.0.2`)

Empirical telemetry from `v1.0.0`/`v1.0.2` establishes two structural bottlenecks driving playing-strength losses against Stockfish 19 (`UCI_Elo` 2800–3190 cumulative `50.00%`, `44W-62D-44L`):

1. **Failure Mode Taxonomy (44 Lost Games, Fixed SF Depth-14 Audit):**
   - **(a) Search / Value Error (`19 / 44 = 43.18%`):** Stockfish's best move was already inside the raw Policy Net's Top-3 logits, yet PUCT + scalar `Tanh` value backup picked an inferior move.
   - **(b) Policy Prior Blindspot (`5 / 44 = 11.36%`):** Stockfish's best move lay outside Policy Top-3.
   - **(c) Tactical Horizon / Hung Material (`19 / 44 = 43.18%`):** Immediate tactical oversight within `< 4` plies causing material loss.
   - **(d) Late Endgame Grind (`1 / 44 = 2.27%`):** Slow conversion or fortress/horizon drift beyond 60 plies.
   - **Phase Distribution:** Opening (`ply <= 20`) `27 (61.36%)`, Middlegame `13 (29.55%)`, Endgame `4 (9.09%)`.

2. **Hardware Asymmetric Idle Telemetry (`10x192` FP16, `400 ms/move`, `B=16` Sequential):**
   - C++23 `fast_tree` reduced host CPU active time to **`34.97 ms/move`** (`8.56%` CPU busy duty cycle: `5.23 ms` pre-loop, `16.57 ms` AVX2 selection, `2.74 ms` dispatch, `10.43 ms` backup).
   - For **`373.64 ms/move` (`91.44%` of wall time)**, the Redwood Cove P-core thread sits idle blocked on OpenVINO NPU inference (`~6.46–6.84 ms` per `B=16` batch).
   - Simply overlapping `nireq=2` async NPU batches (`use_async_overlap=True`) hurt fixed-node tactical accuracy by **`-18.69%`** on `10x128` 50n puzzles (`77.36% -> 58.67%`) due to virtual-loss selection staleness and shallower tree waves. Therefore, idle P-core cycles must be spent **inside each node/wave** (zero-staleness CPU micro-tactics or exact leaf shaping) rather than decoupling NPU queues.

---

## 1. Heterogeneous C++23 Bitboard SEE + PUCT Integration

### (a) What Is Already Implemented in `NeuraChess v1.0.2`
- **BMI2 PEXT & Ray Tables (`src/neurachess/engine/fast_tree/bitboard_chess.hpp`):** Precomputed `rook_pext[102400]` (`_pext_u64`), `bishop_pext[5248]`, `pawn_attacks`, `knight_attacks`, `king_attacks`, `between_bb[64][64]`, `line_ray_bb[64][64]`, `attackers_mask()`, `slider_blockers()`, and `is_ep_move_safe()`.
- **Unused Bonus Plumbing:** `generate_scored_legal_moves()` computes only a naive 1-ply MVV-LVA constant (`+0.35` equal/winning capture, `+0.10` losing capture) for legacy unit-test compatibility; `ft_expand_node_with_logits(..., /*tactical_bonuses_unused_no_pruning*/)` ignores bonuses completely, and production search runs 100% unpruned raw softmax priors.
- **Blind Scalar FPU (`fast_tree.cpp::select_best_child_avx2`):** Every unvisited child (`visit_count + virtual_loss == 0`) at a parent node receives the exact same scalar FPU value (`Q_parent - c_fpu * sqrt(visited_mass)`).

### (b) What Is Genuinely Novel
- Compute **exact bitboard Static Exchange Evaluation (`CBoard::see(mv)`)** directly in C++23 during node expansion (`< 1–2 us` per node, fitting cleanly inside the host gap or `91.4%` idle NPU budget), accounting for **X-ray slider discovery**, **absolutely pinned defenders (`slider_blockers` + `line_ray_bb`)**, **en passant (`is_ep` dual-square removal)**, **promotion/recapture promotions**, and **king capture into defended squares**.
- Wire a **per-child `AlignedSlab<float> fpu_bias` lane** into `ChildSoAArena` (evaluated in 1 AVX2 `_mm256_add_ps` instruction inside `select_best_child_avx2`):
  - If a capture hangs material (`SEE < -100 cp`) and raw prior `P(s, a) < 0.15` (so genuine high-prior positional sacrifices remain untouched), apply an FPU penalty (`fpu_bias = -0.15`), preventing first-visit wasting of NPU slots on unsound grabs while leaving visited `Q` unbiased.
  - If a capture wins material cleanly (`SEE >= +100 cp`), apply a small FPU boost (`fpu_bias = +0.05`), accelerating tactical threat discovery in the first 1–2 waves.

### (c) Expected Impact on Failures (`43.18%` Value/Search, `43.18%` Tactical Horizon)
- **Tactical Horizon (`43.18%`):** High direct impact at shallow node budgets (`50–200n`) and early search waves—immediately suppresses foolish `QxP`-into-pawn/battery captures with low policy priors (`< 0.15`) and prioritizes `SEE >= +100` tactical captures for first-wave NPU verification.
- **Value/Search (`43.18%`):** Moderate positive secondary impact by shifting fixed-budget `NPU` batch slots away from refutable recapture blunders toward sound positional/tactical candidates in Policy Top-3 (though multi-move quiet tactical forks or non-capture blunders still require deeper lookahead or better value heads).

### (d) Exact Implementation Cost
- **Training Cost:** `$0` / Zero retraining (pure inference-time C++ search change).
- **Engineering & Runtime Cost:** ~150 lines C++23 in `bitboard_chess.hpp` + `fast_tree.cpp`, 1 extra float array (`4 bytes/edge`) in `ChildSoAArena`, `< 15 us` CPU overhead per expanded node (negligible inside the `373.6 ms` NPU wait window; zero impact when `use_see_prior_shaping=False`).

---

## 2. Dual-Shape Dynamic NPU Gear-Shifting (`B=4` / `B=8` + `B=16`)

### (a) What Is Already Implemented in `NeuraChess v1.0.2`
- **`NPUModel` Static Shapes (`src/neurachess/models/npu_compiler.py`):** Compiles one static root shape (`B=1`) and one static search shape (`B=16`) with `PERFORMANCE_HINT="LATENCY"`, `NPU_TURBO=True`, and `.blob` disk caching under `weights/cache`.
- **`NPUAutoTuner` (`src/neurachess/engine/npu_autotuner.py`):** Sweeps `B in {8, 16, 32, 64}` once offline/at startup to pick a single fixed production batch size (`B=16` sequential).
- **Single-Request Zero-Copy Loop (`src/neurachess/engine/fast_tree/__init__.py`):** `FastTreeSearchBackend.search()` only binds and loops over a single `batch_model` (`B=16`), padding under-filled forced trees (`n_eval < 16`) with duplicate slot-0 copies (`ft_select_and_pack_native`).

### (b) What Is Genuinely Novel
- Keep **`B=1`, `B=4`, `B=8`, and `B=16`** OpenVINO graphs compiled **simultaneously co-resident** on Intel NPU 3720 (`Total NPU alloc + host RSS ~160–215 MB` total across all 4 graphs).
- Exploit the steep sub-linear latency curve of NPU 3720 (verified hardware numbers: `B=1` `0.81 ms`, `B=4` `1.94 ms`, `B=8` `3.46 ms`, `B=16` `6.84 ms`): `B=4` delivers **3.53x faster sequential turnaround latency** at `88.2%` of `B=16` throughput (`2,062 pos/s`), while `B=8` delivers **1.98x faster turnaround** at `98.8%` of `B=16` throughput (`2,310` vs `2,338 pos/s`).
- Implement **entropy / tactical state-gated dynamic batch selection** in `fast_tree`:
  - **Low gear (`B=4` or `B=8`):** When the root/active subtree is **in check**, **few legal/forcing moves exist**, **policy entropy is low** (`H(P) < tau`), or `select_and_pack_native` encounters high collision/underfill (`n_eval <= 4` or `<= 8`), dispatch the `B=4`/`B=8` graph instead of burning `6.84 ms` on virtual-loss-diluted or padded `B=16` batches.
  - **High gear (`B=16`):** In broad, high-entropy middlegame positions with high leaf diversity (`n_eval > 8`), use `B=16` for peak raw NPS.

### (c) Expected Impact on Failures (`43.18%` Value/Search, `43.18%` Tactical Horizon)
- **Tactical Horizon (`43.18%`):** Direct positive impact on narrow forcing lines (checks, recaptures, mates): within a fixed `400 ms` budget, shifting from `6.84 ms` (`~58` sequential waves max) to `1.94–3.46 ms` (`~115–206` waves) increases achievable **sequential ply depth** and eliminates virtual-loss distortion on unique forcing moves.
- **Value/Search (`43.18%`):** Moderate positive impact in sharp positions with 2–4 candidate moves, where `B=8` or `B=4` concentrates visits cleanly without forced over-exploration of noise moves via `16`-slot virtual loss. Under strict fixed-node limits (`max_nodes`), smaller `B` also cuts wasted virtual-loss diversity overhead at zero wall-time penalty when latency scales sub-linearly.

### (d) Exact Implementation Cost
- **Training Cost:** `$0` / Zero retraining (uses existing `neurachess-10x192-fp16.xml`).
- **Engineering & Memory Cost:** Compiles 2 extra static graphs (`B=4`, `B=8`), adding `~110 MB` host RSS (cached `.blob` warm startup `< 1.5s`), plus a ~60-line policy scheduler and multi-request zero-copy binding in `FastTreeSearchBackend`.

---

## 3. Layerwise Sensitivity-Guided Hybrid Quantization (`Blocks 2..7` INT8 + `Blocks 0,1,8,9` FP16)

### (a) What Is Already Implemented in `NeuraChess v1.0.2`
- **All-or-Nothing Quantization (`src/neurachess/models/quantize_nncf.py` & `scripts/export_openvino.py`):** Exposes NNCF `compress_weights` (`W8A16_ASYM`) and uniform `nncf.quantize(..., fast_bias_correction=True)` without per-block sensitivity selection.
- **Documented Negative Results (`docs/EXPERIMENTS.md` §4):**
  - `B1_W8A8_Res3x3` (quantizing all 10 blocks `res_0..res_9` 3x3 convs + residual stream while ignoring stem/SE/heads): **90.96%** Top-1 agreement vs FP16, `Mean KL = 0.02198`, **`Max Value Diff = 0.9751`** (almost full `[-1, +1]` inversion!), failing the `>= 98.5%` gate.
  - `B2_W8A8_StemRes`: `91.18%` agreement, `Max Value Diff = 0.99182`.
  - 20M QAT fine-tuning (`W8A8`): reached only `92.52%` agreement and lost **`-6.0%` score** vs FP16 against Stockfish 3000 despite `+18.94%` raw NPU speedup (`4,105` vs `3,452` NPS).

### (b) What Is Genuinely Novel
- Measure **exact block-by-block quantization sensitivity** across all 10 residual blocks (`res_0`..`res_9`) independently via `nncf.IgnoredScope` on 5,000 held-out `Far-Test` positions.
- **Hypothesis & Selective Boundary Preservation:**
  - Early blocks (`block_0, block_1`) process sharp sparse binary piece-plane features (`0.0`/`1.0` bitboards + halfmove clock); late blocks (`block_8, block_9`) feed both the unbounded `Conv73` policy logits (`4,672` outputs) and pre-`Tanh` value projection (`val_dense2`), where small activation clipping causes `~0.98` value flips.
  - Keeping `{stem, SE gates, heads, blocks 0, 1, 8, 9}` (and optionally residual skip adds) in **FP16** while quantizing only the **6 middle / least-sensitive residual blocks (`blocks 2..7` or empirically selected top-6)** to W8A8 (`fast_bias_correction=True`, 2,000 real training calibration states) targets `>= 98.5%` Top-1 agreement, `Mean KL <= 0.015`, `Max Value Diff << 0.95`, and partial NPU speedup on 60% of backbone MACs.

### (c) Expected Impact on Failures (`43.18%` Value/Search, `43.18%` Tactical Horizon)
- **Tactical & Value Preservation vs Previous W8A8:** Does *not* exceed FP16 accuracy directly, but seeks to **recover the `-6.0%` W8A8 strength regression** while harvesting higher NPU `B=16` NPS (`> 2,592` added visits/s) -> translating extra NPS into `+10–20%` search nodes per 400 ms if fidelity gates pass (`>= 98.5%` Top-1, `KL <= 0.015`).

### (d) Exact Implementation Cost
- **Training Cost:** `$0` (pure post-training NNCF graph quantization on existing `neurachess-10x192-fp16.npz` + `2,000` local shard positions).
- **Engineering Cost:** ~100 lines using `nncf.quantize(..., ignored_scope=...)` with regex scopes targeting exact block indices in `se_resnet_backbone.py`.

---

## 4. Single-Pass `Conv73` Action-Value $Q(s, a)$ Head (Learned First-Play Urgency)

### (a) What Is Already Implemented in `NeuraChess v1.0.2`
- **Spatial `Conv73` Architecture (`src/neurachess/models/pytorch_se_resnet.py`, `se_resnet_backbone.py`):** Already maps the trunk feature tensor `h [B, C, 8, 8]` via a single `1x1` conv (`C -> 73`) to `[B, 4672]` logits using `(from_sq * 73 + slot)` canonical indexing.
- **MultiPV Sidecar Infrastructure (`src/neurachess/training/build_dataset.py` & `training/train.py`):** The dataset pipeline *already records* up to 4 alternate top engine moves (`extra_acts`, `extra_cps` in `.sidecar.bin`, `17 bytes/pos`), currently used *only* to build Boltzmann soft targets `probs = softmax(all_cps / (100 * T))`.

### (b) What Is Genuinely Novel
- Attach a **parallel `Conv73` Action-Value Head $Q_\theta(s, a)$** (`Conv2d(C, 73, 1x1) -> Tanh -> [B, 4672]`) directly alongside the policy `Conv73` head inside the same forward pass (`+~14K` params, `< 0.5%` FLOPs on `10x192`).
- Train $Q_\theta(s, a)$ with masked MSE against MultiPV targets $\tanh(\text{cp}_k \cdot 0.00287823)$ already stored for up to 5 legal moves per training position in `.sidecar.bin`, plus optional 1-ply value-network distillation for remaining legal moves.
- **Replacing Blind FPU in `fast_tree`:** Instead of setting FPU identically for every unvisited child at a node (`Q_parent - c_fpu * sqrt(visited_mass)`), initialize each edge `(s, a)` with `fpu_i = mix(scalar_fpu, Q_theta(s, a))` directly inside `ChildSoAArena` (reusing the exact same `fpu_bias` lane added in Direction 1!). Wait-free **1-ply tactical lookahead across all legal moves in a single NPU pass**, costing 0 extra NPU inferences!

### (c) Expected Impact on Failures (`43.18%` Value/Search, `43.18%` Tactical Horizon)
- **Tactical Horizon (`43.18%`):** **Very High impact** — if Policy Net mistakenly ranks a poisoned capture or tactical blunder #1/#2 (`prior > 0.15`), a learned `Q(s, a)` head directly sees the refutation evaluation (`Q(s, a_bad) = -0.80` vs `Q(s, a_best) = +0.10`) *before* spending tree visits, preventing tactical traps even under strict `50n` budgets.
- **Value/Search (`43.18%`):** **High direct impact** — cures the "Top-3 prior exists, but PUCT wastes visits exploring wrong candidate or miscalibrated FPU fails to switch moves cleanly" failure mode cleanly in Category (a).

### (d) Exact Implementation Cost
- **Training Cost:** Requires training/fine-tuning a new checkpoint with the 2nd `Conv73` head (~$100–$300 cloud GPU fine-tune over sidecar-equipped 346M/subset shards, NOT executed in this `$0` local spike).
- **Inference / C++ Cost:** `+4,672` float16 outputs per leaf (`+9.3 KB` DMA transfer), negligible `< 2%` NPU latency increase, zero extra tree traversal passes.

---

## 5. HL-Gauss Categorical Distributional Value Head + Moves-Left Head (MLH)

### (a) What Is Already Implemented in `NeuraChess v1.0.2`
- **Scalar `Tanh` MSE Value Head (`training/train.py`, `models/pytorch_se_resnet.py`):** `Full 346M 10x192` uses `value_dim=1` (`Linear(128, 1) -> Tanh`), predicting `v_target = tanh(cp * 0.00287823)` via `F.mse_loss` (`heldout_400m` MSE = `0.0402`, Sign = `96.03%`).
- **Legacy WDL Placeholder:** Code supports `value_dim=3` (`WDL` softmax) and stores `self.contempt = 0.02`, neither active on the `10x192` scalar-tanh model. No Moves-Left Head exists, though `halfmove_clock / 100.0` is supplied on input plane `17`.

### (b) What Is Genuinely Novel
- **HL-Gauss Categorical Value Head (Imani & White 2018; Farebrother et al. ICML 2024):** Replace scalar MSE regression (`1` output -> `Tanh`) with **`K = 51` or `101` histogram bins** partitioning `[-1, +1]` (Gaussian-smoothed soft targets with $\sigma = 0.75 \cdot \Delta_{\text{bin}}$, trained with cross-entropy loss). On NPU 3720, `Linear(128, K) -> Softmax -> fixed dot product` (or host expectation) fits completely within L1/CMX cache with ~`13K` parameters (`< 0.1%` NPU latency cost). Cross-entropy over HL-Gauss avoids MSE's "shrinkage toward zero" under tactical evaluation uncertainty and preserves sharper bimodal probability mass.
- **Auxiliary Moves-Left Head (MLH):** Add a small dense head (`Linear(128, 1) -> Softplus`) predicting remaining plies `m(s)` (distilled from game length / mate distance), enabling distance-weighted draw contempt and tie-breaking ("win fastest, delay losses") without brittle scalar heuristics.

### (c) Expected Impact on Failures (`43.18%` Value/Search, `43.18%` Tactical Horizon)
- **Value/Search (`43.18%`):** **High impact** on sharp tactical positions where Stockfish eval flips discontinuously between `+200 cp` and `-400 cp` across minor subtleties: categorical HL-Gauss cross-entropy produces cleaner gradients and higher sensitivity to sharp tactical swings without scalar MSE mean-collapse. MLH directly targets the remaining `2.27%` endgame grind and forcing mate conversion speed.
- **Tactical Horizon (`43.18%`):** **Low-to-moderate indirect impact** — improves static leaf calibration, though single-position evaluation still requires search depth (`Spike 2`) and capture/tactical urgency (`Spike 1`, `Direction 4`) to look past quiet horizon tactics.

### (d) Exact Implementation Cost
- **Training Cost:** Requires retraining or warm-start head distillation on the 346M Lichess shards + game-length labels (`> $0`, deferred past local NPU spikes).
- **Inference Cost:** Replacing `Linear(128, 1) -> Tanh` with `Linear(128, 65) -> Softmax` adds ~`8K` MACs (`< 0.01 ms` on NPU 3720).

---

## Summary Comparison Matrix

| Direction | Repo Status (`v1.0.2`) | Target Loss Class | Retraining Required? | Expected Cost |
| :--- | :--- | :--- | :--- | :--- |
| **1. C++23 Bitboard SEE + PUCT** | BMI2 PEXT tables exist; `see()` & per-child `fpu_bias` missing | Cat (c) `43.18%` + Cat (a) `43.18%` | **No ($0)** | Spike 1 (Implemented locally) |
| **2. Dual-Shape Gear-Shifting (`B=4/8/16`)** | Single static `B=16`; no multi-shape runtime dispatch | Cat (c) `43.18%` + Cat (a) `43.18%` | **No ($0)** | Spike 2 (Implemented locally) |
| **3. Blockwise Hybrid W8A8 (`2..7`)** | Uniform W8A8/Res3x3 failed (`90.96%` / `-6%` score); no block profile | Compute / NPS (`v1.0.2` W8A8 recovery) | **No ($0)** | Spike 3 (Implemented locally) |
| **4. Conv73 Action-Value $Q(s,a)$** | `Conv73` policy + `17B` MultiPV sidecars exist; $Q$ head missing | Cat (a) `43.18%` + Cat (c) `43.18%` | **Yes** (fine-tune) | Deferred to `v1.1` GPU phase |
| **5. HL-Gauss Value + Moves-Left** | Scalar `Tanh` MSE (`value_dim=1`); no MLH / HL-Gauss bins | Cat (a) `43.18%` + Cat (d) `2.27%` | **Yes** (retrain/distill) | Deferred to `v1.1` GPU phase |

---

## Observed Empirical Outcomes (`research/v1.1-spikes`, Local NPU 3720)

All raw measurements are archived in [`eval_results/research_spikes.json`](../eval_results/research_spikes.json) (`$0` cloud spend):

1. **Spike 1 — C++23 Bitboard SEE (`use_see_prior_shaping`):**
   - `CBoard::see()` verified **56 / 56** tactical exchange positions at **`37.85 ns/call`** (`0.038 us`).
   - **Negative Result on FPU Shaping Rule:** Enabling `use_see_prior_shaping=True` (`-0.15` FPU penalty on `SEE < -100, P < 0.15`; `+0.05` FPU boost on `SEE >= +100`) slightly regressed 500 stratified Lichess puzzles (**`78.20%` vs `78.80%`** at `50n`; **`82.00%` vs `82.20%`** at `200n`), increased 200n Category (c) `>= 150 cp` blunders (`6/19 -> 7/19`, mean drop `127.16 -> 134.42 cp`), and increased Category (a) mean drop (`33.63 -> 38.79 cp`). Flag stays **`False`**.

2. **Spike 2 — Dual-Shape NPU Gear-Shifting (`B=1, 4, 8, 16` Co-Resident):**
   - Co-resident footprint: **`59.09 MB`** NPU alloc / **`172.38 MB`** host delta RSS across all 4 graphs; turnaround latencies **`0.737 ms`** (`B=1`, `1,357 pos/s`), **`1.887 ms`** (`B=4`, `2,120 pos/s`), **`3.379 ms`** (`B=8`, `2,367 pos/s`), **`6.282 ms`** (`B=16`, `2,547 pos/s`).
   - At `400 ms/move`, Mate-in-3 (`n=100`) held **`97/100 (97.00%)`** with `30.7%` fewer batches (`11.68 -> 8.10`), higher `avg_ply` (`1.739 -> 2.192`), and global max ply `14 -> 15`. On Category (c) (`n=19`), wave count jumped **`2.54x`** (`67.32 -> 171.11` batches/move), `avg_ply` rose (`6.465 -> 6.703`), and `>= 150 cp` blunders fell (`5/19 -> 4/19`), though over-aggressive `B=4` triggering (`3,069 / 3,251` batches = `94.4%`) lowered total evaluated nodes (`1,187.58 -> 838.53`) and `SF_best` matches (`11/19 -> 10/19`). Needs tighter entropy/check thresholds before enabling by default (`use_dynamic_batch_shift=False`).

3. **Spike 3 — Block-Wise Quantization Sensitivity Profile (NPU 3720):**
   - Single-block scan (`500` calib) pinpointed the two failure sources in prior uniform W8A8: **`Block 9`** alone causes **`0.013042`** policy KL (vs `0.000462–0.001436` for blocks `0–8`), and **`Block 1`** alone triggers the **`0.92645`** `Max Value Diff` outlier spike.
   - Keeping blocks `0, 1, 8, 9` and residual skip adds `.*_out_.*` in FP16 (`Hybrid_Middle_2_to_7_ConvOnly`, `2,000` calib) raised 5,000-pos Far-Test Top-1 agreement from **`90.96%`** (Full PTQ) / **`92.52%`** (QAT) to **`95.52%`**, cut Mean Policy KL **4.5x** (`0.021977 -> 0.004891`), cut Max Value Diff **41.6%** (`0.97510 -> 0.56921`), and delivered **`2,997.1`** added visits/s (**`+16.2%`** vs FP16 `2,578.8`)—yet still fell short of the historical `>= 98.5%` Top-1 gate.

---

## Phase 2 Empirical Follow-Ups ($0 Cloud Spend, NPU 3720)

All raw Phase 2 telemetry is recorded in [`eval_results/research_spikes_phase2.json`](../eval_results/research_spikes_phase2.json). `use_see_prior_shaping=False` remains the default production setting ("Bitter Lesson": learned 346M priors outperform static SEE shaping).

### 1. Spike 2 Refinement — Strict Collision-Bounded & Cold-Start Gear-Shifting
Replaced all entropy / concentration triggers in [fast_tree.cpp](../src/neurachess/engine/fast_tree/fast_tree.cpp) (`ft_select_dynamic_native`, `ft_pack_inflight_batch_native`) with three strict triggers:
- **(a) Cold-Start Ramp (`reason=1`):** `root.visit_count < 16` → `B=4`
- **(b) Narrow Root (`reason=2`):** root is in check or `<= 5` legal moves → `B=4`
- **(c) Collision-Bounded Flush (`reason=3, 4`):** during `select(16)`, once `>= 6` virtual-loss collisions / terminal / cache hits occur with `1..4` unique uncached leaves (`B=4`) or `5..8` unique uncached leaves (`B=8`), flush immediately instead of padding up to `16`.

| Benchmark | Arm | Accuracy / SF Best | Blunders >=150cp | Mean Drop | Total Nodes | Avg / Max Ply | Batch Shares (B16 / B8 / B4) |
|---|---|---|---|---|---|---|---|
| Cat (c) 19 @ `400 ms` | `baseline_b16_prod_db` | `11/19` (`12/19` live) | `5/19` | `84.26 cp` | `22,350` (`1,176.3`) | `6.45 / 13.42` | `100% B=16` (double-buffered) |
| Cat (c) 19 @ `400 ms` | `gear_shift_v2_collision_cold` | `10/19` (`11/19` live) | `6/19` | `99.63 cp` | `20,411` (`1,074.3`) | `6.83 / 13.47` | `69.50% / 7.71% / 22.79%` |
| Cat (a) 19 @ `400 ms` | `baseline_b16_prod_db` | `15/19` (`15/19` live) | `1/19` | `27.21 cp` | `23,390` (`1,231.1`) | `6.51 / 13.58` | `100% B=16` (double-buffered) |
| Cat (a) 19 @ `400 ms` | `gear_shift_v2_collision_cold` | `15/19` (`15/19` live) | `1/19` | `34.74 cp` | `21,093` (`1,110.2`) | `6.92 / 13.63` | `68.88% / 8.13% / 22.98%` |
| 500 Puzzles @ `50 ms` | `baseline_b16_prod_db` | `405/500 (81.00%)` | — | — | `67,614` (`135.2`) | `2.21 / 4.33` | `100% B=16` |
| 500 Puzzles @ `50 ms` | `gear_shift_v2_collision_cold` | **`450/500 (90.00%)`** | — | — | `50,014` (`100.0`) | **`3.76 / 7.58`** | `43.29% / 3.59% / 53.12%` |
| 500 Puzzles @ `200 ms` | `baseline_b16_prod_db` | `471/500 (94.20%)` | — | — | `229,565` (`459.1`) | `4.84 / 10.08` | `100% B=16` |
| 500 Puzzles @ `200 ms` | `gear_shift_v2_collision_cold` | **`478/500 (95.60%)`** | — | — | `207,190` (`414.4`) | **`5.86 / 10.60`** | `58.51% / 6.03% / 35.45%` |

**Diagnosis:**
- **Ultra-short / forced-tactics regime wins decisively:** Cold-start `B=4` waves lift `50 ms` puzzle accuracy by **`+9.00%`** (`405 -> 450 / 500`) and push average search depth from `2.21` to `3.76` plies (`4.33 -> 7.58` max ply), and still improve `200 ms` solve rate from `94.20%` to **`95.60%`**.
- **Why `B=16` landed at `69.50%` (Cat c) / `58.51%` (200 ms puzzles) instead of `>= 80%`:** Rule (b) (`reason=2`) locks any root-in-check position at `B=4` permanently across the entire 400 ms search (`198/1413 = 14.01%` batches from a single check FEN #17 in Cat (c), `197/1414 = 13.93%` from FEN #6 in Cat (a), and `3,026/16,343 = 18.52%` in 200 ms puzzles). Across the 18 non-check FENs in Cat (c), `B=16` share was `80.8%`. Furthermore, `gear_shift_v2` runs synchronous `zc_cur[0].infer()`, forfeiting the ~6% double-buffering throughput gain of `baseline_b16_prod_db`. Future v1.1 optimization should gate Narrow Root `in_check` to `root.visit_count < 32` and retain async double-buffering on steady `B=16` waves.

---

### 2. Spike 3 Ground-Truth Eval & `Hybrid_Safe4_2356_ConvOnly`
Evaluated against actual Stockfish ground-truth labels on `5,000` `test_far_50k.bin` positions and equal wall-clock (`100 ms/puzzle`) / equal node (`200 n/puzzle`) runs on 500 Stratified Clean Puzzles:

| Variant | Blocks INT8 | Far5k GT Top-1 | Far5k GT CE | Far5k GT Value MSE | vs FP16 Top-1 / KL / MaxV | Pz `100 ms` Equal Time | Pz `200n` Equal Nodes | Added NPS (12p) |
|---|---|---|---|---|---|---|---|---|
| `V0_Pure_FP16` (`13.36 MB` IR) | None | `50.46%` | `1.6180` | `0.06117` | `100% / 0.0 / 0.0` | `428/500 (85.60%)`, `118,665n`, `2,683.4 nps` | `411/500 (82.20%)` | `2,588.0` |
| `Hybrid_Middle_2_to_7_ConvOnly` (`15.33 MB` IR) | `[2,3,4,5,6,7]` | **`50.54%`** | `1.6215` | **`0.06085`** | `95.52% / 0.004891 / 0.56921` | **`447/500 (89.40%)`**, **`139,248n`**, **`3,212.8 nps`** | `413/500 (82.60%)` | **`3,098.5`** (`+19.7%`) |
| `Hybrid_Safe4_2356_ConvOnly` (`19.13 MB` IR) | `[2,3,5,6]` | **`50.54%`** | `1.6190` | `0.06143` | `96.40% / 0.003075 / 0.54297` | `441/500 (88.20%)`, `132,160n`, `3,016.1 nps` | **`414/500 (82.80%)`** | `2,920.7` (`+12.9%`) |

**Key Conclusion:** Ground-truth evaluation overturns the FP16-agreement proxy objection: **`Hybrid_Middle_2_to_7_ConvOnly`** matches or slightly beats `V0_Pure_FP16` on actual Stockfish Top-1 (`50.54%` vs `50.46%`), Top-3 (`79.54%` vs `79.44%`), Decisive Sign (`93.67%` vs `93.49%`), and Value MSE (`0.06085` vs `0.06117`), while delivering **`+19.7%` higher added NPS (`3,098.5` vs `2,588.0`)** and solving **`+19` more clean puzzles (`85.60% -> 89.40%`)** at fixed `100 ms/puzzle` (`139,248` vs `118,665` total nodes, `mean_max_ply 7.07 -> 8.04`).

---

### 3. Stage 4 Architecture NPU 3720 Latency (`B=16`) & Data Readiness
Compiled & benchmarked experimental heads (`B=16`, `LATENCY`, `NPU_TURBO=True`, `25` warmup + `200` timed iterations, all outputs verified finite):

| Graph Architecture | IR `.bin` | Outputs | Median Latency | p10 / p90 | Delta vs Baseline | Raw `B=16` Throughput |
|---|---|---|---|---|---|---|
| `baseline_10x192_scalar` | `13.36 MB` | `Policy [16,4672], Value [16,1]` | `6.386 ms` | `6.257 / 6.532 ms` | `+0.000 ms (+0.00%)` | `2,505.4 pos/s` |
| `hl_gauss_64_plus_mlh` | `13.44 MB` | `Policy [16,4672], HL64 [16,64], MLH [16,1]` | **`6.399 ms`** | `6.319 / 6.543 ms` | **`+0.012 ms (+0.19%)`** | `2,500.5 pos/s` |
| `dual_conv73_policy_q_plus_val` | `13.38 MB` | `Policy [16,4672], Q [16,4672], Value [16,1]` | **`6.473 ms`** | `6.371 / 6.657 ms` | **`+0.087 ms (+1.36%)`** | `2,471.9 pos/s` |
| `dual_conv73_policy_q_only` | `13.13 MB` | `Policy [16,4672], Q [16,4672]` | `6.437 ms` | `6.321 / 6.600 ms` | `+0.050 ms (+0.79%)` | `2,485.8 pos/s` |

**Training Data Readiness (Verified Local Disk + Read-Only S3 Audit):**
- **HL-Gauss-64 Value Head:** **Ready (`eval_cp` present in all 107B v1 & 691B v2 shards)** — `1M` `691B` v2 positions (`v2_shards_1m/shard_{000,001}.bin`, `659.0 MiB` total) exist both locally and at `s3://neurachess-spot-162290422654-us-east-1/data/shards/`, plus `107B` v1 eval sets & local training samples (`sample_shard_{000,040}.bin`, `overnight_shards/`, `train_300k.bin`). Note `s3://.../data/shards_s1/` has `0` objects (`0 Bytes`).
- **Moves-Left Head (MLH):** **Not in position shards** — fixed 107B/691B records have no remaining game-length/ply labels; requires PGN or self-play trajectory extraction.
- **Conv73 Action-Value $Q(s,a)$ Head:** **Requires MultiPV sidecar generation** — **zero `.sidecar.bin` files** exist locally or in `s3://neurachess-spot-162290422654-us-east-1/`. Supervising all legal moves per position requires running Stockfish MultiPV sidecar generation or MCTS leaf-value distillation.

---

## Phase 3 Empirical Lock-In ($0 Cloud Spend, NPU 3720)

All raw Phase 3 telemetry is stored in [`eval_results/research_spikes_phase3.json`](../eval_results/research_spikes_phase3.json), [`eval_results/phase3_h2h_60g.json`](../eval_results/phase3_h2h_60g.json), [`eval_results/phase3_sf3000_50g.json`](../eval_results/phase3_sf3000_50g.json), and [`eval_results/phase3_control_fp16_sf3000_20g.json`](../eval_results/phase3_control_fp16_sf3000_20g.json).

### 1. `gear_shift_v3` (`check < 32` Cap + `nireq=2` Async `B=16`) + `Hybrid_Middle_2_to_7_ConvOnly`
Upgraded [fast_tree.cpp](../src/neurachess/engine/fast_tree/fast_tree.cpp) (`ft_select_dynamic_native`, `ft_recommend_batch_size`) to cap Narrow-Root Rule (b) (`reason=2`, in check or `<= 5` legal moves) at `root.visit_count < 32` before escalating to `B=16`, and added `nireq=2` async double-buffering for `B=16` waves inside `FastTreeSearchBackend.search` ([`__init__.py`](../src/neurachess/engine/fast_tree/__init__.py)).

- **Co-resident Hybrid IR (`neurachess-10x192-hybrid-mid27-int8.xml`, `15.33 MB` `.bin`):** `B=1`: `0.708 ms` (`1,412.5 pos/s`), `B=4`: `1.603 ms` (`2,495.6 pos/s`), `B=8`: `2.859 ms` (`2,797.8 pos/s`), `B=16`: `5.412 ms` (`2,956.6 pos/s`) vs Pure FP16 `0.739 / 1.894 / 3.360 / 6.262 ms`.

| Suite | Arm | Solve / SF Best | Blunders >=150cp / Drop | Total Nodes | Avg / Max Ply | `B=16` Share % |
|---|---|---|---|---|---|---|
| 500 Clean Puzzles @ `50 ms` | `V0_Pure_FP16_baseline_b16_prod_db` | `400/500 (80.00%)` | — | `67,401` (`2,661 nps`) | `2.21 / 4.33` | `100%` |
| 500 Clean Puzzles @ `50 ms` | `V0_Pure_FP16_gear_shift_v3` | `436/500 (87.20%)` | — | `57,983` (`2,562 nps`) | `3.12 / 6.68` | `54.73%` |
| 500 Clean Puzzles @ `50 ms` | `Hybrid_Mid27_ConvOnly_gear_shift_v3` | **`440/500 (88.00%)`** | — | `67,426` (`3,007 nps`) | **`3.35 / 7.12`** | `58.71%` |
| 500 Clean Puzzles @ `100 ms` | `V0_Pure_FP16_baseline_b16_prod_db` | `428/500 (85.60%)` | — | `118,324` (`2,679 nps`) | `3.32 / 7.05` | `100%` |
| 500 Clean Puzzles @ `100 ms` | `V0_Pure_FP16_gear_shift_v3` | `459/500 (91.80%)` | — | `111,480` (`2,661 nps`) | `4.26 / 8.81` | `66.80%` |
| 500 Clean Puzzles @ `100 ms` | `Hybrid_Mid27_ConvOnly_gear_shift_v3` | **`463/500 (92.60%)`** | — | **`129,411` (`3,139 nps`)** | **`4.50 / 9.23`** | `69.42%` |
| 500 Clean Puzzles @ `200 ms` | `V0_Pure_FP16_baseline_b16_prod_db` | `471/500 (94.20%)` | — | `229,266` (`2,788 nps`) | `4.84 / 10.07` | `100%` |
| 500 Clean Puzzles @ `200 ms` | `V0_Pure_FP16_gear_shift_v3` | **`478/500 (95.60%)`** | — | `223,738` (`2,814 nps`) | `5.45 / 10.44` | `73.05%` |
| 500 Clean Puzzles @ `200 ms` | `Hybrid_Mid27_ConvOnly_gear_shift_v3` | **`478/500 (95.60%)`** | — | **`260,544` (`3,300 nps`)** | **`5.65 / 10.57`** | `73.65%` |
| Cat (a) 19 @ `400 ms` | `V0_Pure_FP16_baseline_b16_prod_db` | `15/19` (`16/19` live) | `1/19` (`32.32 cp`) | `23,355` (`1,229.2`) | `6.51 / 13.58` | `100%` |
| Cat (a) 19 @ `400 ms` | `V0_Pure_FP16_gear_shift_v3` | `15/19` (`16/19` live) | `1/19` (`33.47 cp`) | `23,607` (`1,242.5`) | `6.75 / 13.63` | **`81.30%`** |
| Cat (a) 19 @ `400 ms` | `Hybrid_Mid27_ConvOnly_gear_shift_v3` | `15/19` (`14/19` live) | `1/19` (`39.32 cp`) | **`27,588` (`+18.1%`)** | **`6.94 / 13.95`** | **`80.62%`** |
| Cat (c) 19 @ `400 ms` | `V0_Pure_FP16_baseline_b16_prod_db` | `11/19` (`11/19` live) | `5/19` (`88.79 cp`) | `22,500` (`1,184.2`) | `6.46 / 13.42` | `100%` |
| Cat (c) 19 @ `400 ms` | `V0_Pure_FP16_gear_shift_v3` | `10/19` (`11/19` live) | `6/19` (`90.79 cp`) | `22,476` (`1,183.0`) | `6.60 / 13.32` | **`82.17%`** |
| Cat (c) 19 @ `400 ms` | `Hybrid_Mid27_ConvOnly_gear_shift_v3` | `9/19` (`11/19` live) | `6/19` (`97.21 cp`) | **`26,463` (`+17.6%`)** | **`6.84 / 13.47`** | **`80.95%`** |

---

### 2. Frozen-Backbone `HL-Gauss-64` Value Head (`$0`, 1 Pass over 2,299,414 Positions)
Using [`train_hl_gauss_value.py`](../src/neurachess/training/train_hl_gauss_value.py), froze 100% of the stem, 10 SE-ResNet blocks, and `Conv73` policy head, extracted frozen `[B, 192, 8, 8]` features via dual `OV NPU + Arc GPU` throughput pipelines (`3,316 pos/s` end-to-end training throughput), excluded `586` holdout-overlap positions (`test_far_50k`, `test_in_region_50k_v2`, puzzles, Cat (a)/(c)), and trained `val_conv(16) -> val_fc1(256) -> val_fc2(64)` with HL-Gauss CE (`sigma = 1.5 * bin_width`), folding `E[V] = sum_i softmax(logits)_i * z_i` directly to `[B, 1]` inside OpenVINO FP16 IR ([`se_resnet_backbone.py`](../src/neurachess/models/se_resnet_backbone.py)).

| Variant | Far5k Value MSE | Far5k Decisive Sign | Far5k Top-1 / MaxPolDiff | Cat (a) `400 ms` (`orig` / `live` / `Bl>=150` / `MeanDrop`) |
|---|---|---|---|---|
| `scalar_fp16_from_npz` (`346M` head) | `0.06117` | `93.49%` | `50.46%` / `0.0` (ref) | `15/19` / `16/19` / `1/19` / `32.32 cp` (med `1.0`) |
| `hl_gauss64_fp16` (`1`-pass head) | **`0.06037`** | **`95.15%`** | `50.46%` / `0.000000` | `15/19` / `14/19` / `1/19` / `33.47 cp` (med `0.0`) |
| `hl_gauss64_hybrid_mid27_int8` (`+ v3`) | **`0.05954`** | **`95.18%`** | `50.54%` / — | `15/19` / `13/19` / `1/19` / `34.05 cp` (`27,681` n) |

- **Gate Outcome (`KEEP_HL_GAUSS_64=False`):** While `HL-Gauss-64` kept policy outputs bit-for-bit identical (`MaxPolDiff=0.000000`) and strongly improved offline value calibration (`MSE 0.06117 -> 0.06037` FP16 / `0.05954` Hybrid; `DecSign 93.49% -> 95.18%`), Category (a) `400 ms` dipped slightly (`live_sf 16/19 -> 14/19`, `MeanDrop 32.32 -> 33.47 cp`), likely reflecting 1 pass over `2.3M` positions (`0.66%` of the 346M corpus) and `sigma=1.5*w` tail smoothing on tactical extremes. Per the pre-registered rule, the 346M scalar head (`neurachess-10x192-hybrid-mid27-int8.xml` + `gear_shift_v3`) advanced to Step 3.

---

### 3. Validation Matches (`400 ms/move`, AC Power)
- **Step 3A — `60`-Game H2H (`OPENINGS_50[:30]`, `use_ponder=False`):**
  - `Hybrid_Mid27_ConvOnly + gear_shift_v3` vs `v1.0.2 (V0_Pure_FP16, B=16)`: **`35.5 / 60` (`59.17%`, `W=21 D=29 L=10`)**, Wilson 95% CI `[46.55%, 70.68%]`, Pentanomial `[2, 1, 14, 10, 3]`, SPRT LLR `+0.9349` (`[0, 20]` Elo bounds), **`4,092.4 NPS`** vs **`3,314.4 NPS`** (`+23.5%`). Passed the `>= 53.0%` gate.
- **Step 3B — `50`-Game vs Stockfish 19 `UCI_Elo 3000` (`OPENINGS_50[:25]`, `use_ponder=True`, `max_plies=220`):**
  - `Hybrid_Mid27_ConvOnly + gear_shift_v3`: **`15.0 / 50` (`30.00%`, `W=3 D=24 L=23`)**, Wilson 95% CI `[19.10%, 43.75%]`, **`effective_nps = 4,108.0`** (vs historical `49.00%`, `24.5/50` in `step4_stockfish_ladder_final.json`).
- **Opening-Suite Control Check (`20` games, `OPENINGS_50[:10]`, `400 ms/move`, `use_ponder=True` vs SF 19 `UCI_Elo 3000`):**
  - `v1.0.2` (`V0_Pure_FP16`, `B=16`) scored **`5.5 / 20` (`27.50%`, `W=0 D=11 L=9`, `3,340.8 NPS`)** on the exact same first 10 `OPENINGS_50` pairs where Phase 3 scored **`5.0 / 20` (`25.00%`, `W=0 D=10 L=10`)**. This isolates the `30.0%` vs `49.0%` difference primarily to `OPENINGS_50` GM mainline openings + `220`-ply play vs `step4`'s randomized `openings_320[75:100]` (`max_plies=150` + material adjudication), while Phase 3 demonstrated **`+59.17%` H2H (`+64.5` Elo)** over `v1.0.2` and `+8.0% / +7.0%` gains on `50/100 ms` tactics.



