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

