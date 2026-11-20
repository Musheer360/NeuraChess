# NeuraChess

NeuraChess is a neural chess engine targeting Intel Core Ultra **Intel AI Boost NPU** acceleration through OpenVINO, paired with an AVX2/BMI2 C++23 Structure-of-Arrays (SoA) batched PUCT search tree (`fast_tree`).

NeuraChess trains 18-plane Squeeze-and-Excitation residual networks (`ChessSEResNet`) with a 73-plane spatial policy head (`Conv73`, mapping directly to the 4,672 AlphaZero move encodings) on packed bitboard datasets, folds BatchNorm affine parameters statically into FP16/INT8 OpenVINO IR weights, and runs CPU tree traversal alongside batched NPU inference (`B=16`).

---

## Release Milestones (`v1.0.2` $\to$ `v1.1.0` $\to$ `v1.2.0`)

| Release | Model / Role | Cumulative Data | Raw NPU `B=16` | Real-Time Play (`400 ms/move`, Syzygy) | Clean Lichess Puzzles (`n=1,979`) |
| --- | --- | ---: | ---: | --- | --- |
| [`v1.0.2`](https://github.com/Musheer360/NeuraChess/releases/tag/v1.0.0) | `10x192-Scalar` (Pure FP16 Baseline) | 346M | `6.39 ms` | `3,275` eff. NPS; `37.50%` vs SF19 `3,500`n | `71.45%` / `79.38%` / `84.03%` / `95.96%`* |
| **[`v1.1.0`](https://github.com/Musheer360/NeuraChess/releases/tag/v1.1.0)** | **`10x192-QAT-786M` (Default Real-Time Match Model)** | **786M** | **`5.34 ms`** | **`3,006.3` implied SF19 `UCI_Elo`** (`50.67%`, `46W–60D–44L`, `150g`), **`4,013.3` eff. NPS**, **`+107.6` Elo** (`65.0%`) vs `v1.0.2` FP16 | `67.00%` / `77.01%` / `83.63%` / `95.35%` |
| **[`v1.2.0`](https://github.com/Musheer360/NeuraChess/releases/tag/v1.2.0)** | **`10x192-Q-1.00B` (Action-Value $Q$-FPU & Uncertainty Release)** | **1.00B** | **`5.57 ms`** (Lean 3-Out) | **`0.04578`** 150K GPU MSE (`-4.5%`); `+7.0` Elo (`51.0%`) single-variable $Q$-FPU control; `50.0%` natural parity vs `v1.1.0` | `67.96%` / `76.81%` / **`85.19%`** / **`95.76%`** (`2100+`: **`87.31%`**) |

*(Research paper: [`paper/neurachess_npu_paper.pdf`](paper/neurachess_npu_paper.pdf), "Mixed-Precision Neural MCTS and Multi-Shape Batching on a 3W Edge NPU"; live rated NPU games: [`lichess.org/@/NeuraChess-AI`](https://lichess.org/@/NeuraChess-AI). \*`v1.0.2` puzzle row ran on an un-decontaminated 2,000-puzzle pool; `v1.1.0` & `v1.2.0` use the leakage-free 1,979-puzzle benchmark.)*

## Summary

- **`v1.1.0` Default Real-Time Match Model (`10x192-QAT-786M`, Middle `2..7` W8A8 QAT INT8):**
  - Combines sensitivity-guided Middle-Block (`2..7`) W8A8 QAT, 64-bin `HL-Gauss` distributional value supervision, and balanced 50/50 Lc0 T80 + Stockfish Multi-PV scaling to `786M` positions (`5.34 ms` raw `B=16` on Intel AI Boost NPU 3720).
  - Beats `v1.0.2` Pure FP16 by **`+107.6` Elo** (`65.00%`, `54W–22D–24L`, `0` double losses across 50 pairs, SPRT `ACCEPT_H1` **`LLR = +3.5739`**, `3,851.0` vs `3,251.9` unpondered NPS).
  - Scores **`50.67%`** (`46W–60D–44L`, 95% CI `[42.75%, 58.55%]`) across the **150-game Stockfish 19 `UCI_Elo` 2800 / 3000 / 3190 ladder** (**`3,006.3` mean implied `UCI_Elo`**, **`4,013.3` effective pondered NPS**), plus **`83.75%`** (`30W–7D–3L`) against uncapped Stockfish 19 at `go nodes 3500`.
- **`v1.2.0` 1.00-Billion-Position Research Release (`10x192-Q-1.00B`, Action-Value $Q$-FPU & Uncertainty):**
  - Scales `10x192` to **1.00 Billion cumulative positions** with a parallel single-pass `Conv73` action-value head $Q_\theta(s,a)$ supervised in $\mathrm{atanh}$ logit space and gathered zero-copy in C++23 strictly over legal moves.
  - Cuts unified 150K GPU Value MSE by **`-4.5%`** (`0.04794` $\to$ **`0.04578`**) and reaches **`85.19%`** (`1,686/1,979`) at `200n` and **`95.76%`** (`1,895/1,979`, `2100+` **`87.31%`**) at `800n` on the clean Lichess puzzle suite (`+1.11`–`+1.56 pp` from $Q$-FPU).
  - Adds **`+0.0001 ms`** graph-folded 64-bin distributional uncertainty ($\sigma_V$, raising 500 clean `200n` puzzles by `+2.00 pp` when median-centered at `0.35`) and Moves-Left (`MLH`) tie-breaking (**`-9.5` plies** faster average win conversion at `50.0%` H2H parity).
- **C++23 `fast_tree` Search (`src/neurachess/engine/fast_tree`):** Zero-allocation cache-aligned SoA bitboard node arena with AVX2/BMI2 legal move generation (81/81 unit tests including CPW Perft & 2,000-position differential fuzzer), 5-piece Syzygy WDL/DTZ tablebase probing, Polyglot legal-EP 64-bit Zobrist canonical transposition neural cache (`64 MB`), subtree reuse, and exact minimax mate-distance propagation.
- **Documentation and Logs:** Architectural and search ablations (`Conv73` vs dense policy, CNN vs transformer latency on NPU 3720, `B=16` vs `B=32`/`64`, FP16 vs PTQ/QAT W8A8, `10x192` vs `12x192`, `346M -> 786M -> 1.00B` scaling, and Stockfish 19 calibration) are logged with 95% Wilson CIs in [`paper/neurachess_npu_paper.pdf`](paper/neurachess_npu_paper.pdf) and [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## System Architecture

```text
+-----------------------------------------------------------------------+
|                       UCI / CLI / FastAPI Frontend                     |
|   neurachess.uci  |  play_terminal.py  |  neurachess.web.server       |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|       Search Controller (src/neurachess/search/npu_batched_puct.py)   |
|  - Dynamic UCI time & PonderManager (search/ponder_manager.py)        |
|  - Exact minimax solver (mate_plies tracking, draw-safe propagation)  |
|  - 64 MB Zobrist canonical NN cache (FPU=0.20, c_puct=1.5, T=1.0)     |
+-----------------------------------+-----------------------------------+
                                    |
          +-------------------------+-------------------------+
          | Leaf selection & bitboard movegen (CPU P-cores)   | Batched tensors
          v                                                   v
+-----------------------------------+   +-------------------------------+
|  C++23 fast_tree (AVX2/BMI2 SoA)  |   |  OpenVINO NPUModel (B=1 / 16) |
|  src/.../fast_tree/fast_tree.cpp  |   |  src/.../npu_compiler.py      |
|  src/.../fast_tree/bitboard_chess |   |  SE-ResNet (FP16 IR) + Cache  |
+-----------------------------------+   +-------------------------------+
```

For full details on board tensor layouts (`gpu_unpack` vs `legacy_flip`), Conv73 indexing, C++ SoA memory layout, and OpenVINO NPU compilation hints, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick Start

### 1. Environment & Dependencies

NeuraChess requires Python 3.10+ and bundles OpenVINO Runtime plus `ziglang` in its base install so `fast_tree` compiles automatically on first launch without external C++ toolchain setup (`NEURACHESS_ARCH=native` by default, or `NEURACHESS_ARCH=x86-64-v3` for a portable AVX2+BMI2+FMA build without AVX-512). PyTorch and NNCF are optional and only needed for training, dataset generation, and IR export:

```bash
# Lightweight UCI / search / OpenVINO inference install (no PyTorch required)
pip install -e .

# Optional extras for training/export (`[train]`) or the FastAPI web UI (`[web]`)
pip install -e ".[train]"
pip install -e ".[web]"

# Or install all dependencies together
pip install -r requirements.txt
```

### 2. Download Verified Pretrained Weights

Model weights are tracked via GitHub Releases (not committed to Git) and verified via SHA-256 checksums:

```bash
# Download v1.1.0 default 10x192 Mid-2..7 QAT INT8 OpenVINO IR + folded NumPy weights
python -m neurachess.scripts.download_weights --model 10x192

# Optional: deep 12x192 INT8, v1.0.2 10x192 FP16, compact 10x128 FP16, or all bundles
python -m neurachess.scripts.download_weights --model 12x192
python -m neurachess.scripts.download_weights --model all
```

| Model Asset | Release / Purpose | Size (Bytes) | SHA-256 |
| --- | --- | ---: | --- |
| `neurachess-10x192-qat-int8.xml` | **Default `v1.1.0`** `10x192-QAT-786M` Mid-`2..7` INT8 IR | 156,225 | `4c95e88c22b5bb0df084db2aab08f4a040b82f0eb0cc6551b017f503382adefc` |
| `neurachess-10x192-qat-int8.bin` | **Default `v1.1.0`** `10x192-QAT-786M` INT8+FP16 weights | 16,669,672 | `3b5b4c63f0c161d6baaa505705d0595481b5d8d5618901388e30a4a4ecad86e1` |
| `neurachess-10x192-qat-int8.npz` | Folded `10x192-QAT-786M` + `HL-Gauss-64` weights | 26,702,217 | `72423a8979fdb1d049e22fb72746d0298a6b5eb75832463e14bbbdd988f65edc` |
| `10x192_1B_qhead_qat_mid2_7_int8.xml` | **`v1.2.0`** `10x192-Q-1.00B` 4-Head (`+Q`, $\sigma_V$, `MLH`) INT8 IR | 171,530 | `1acddbe0fe5783e5b2b464ea56698d0d115c64aa932ab1a1fae1ec2cc1f6ac02` |
| `10x192_1B_qhead_qat_mid2_7_int8.bin` | **`v1.2.0`** `10x192-Q-1.00B` 4-Head INT8 weights | 16,864,068 | `d173432c922c2f9becd0a005393a40348c58e6f78db7709a4eb0afaa218fba10` |
| `1B_soup35_ema_qat_lean3out_q_hybrid_int8.xml` | **`v1.2.0`** `10x192-Q-1.00B-Soup` Lean 3-Out (`+Q`) INT8 IR | 166,522 | `1edcaa50e5cb56aaab1bd954a494d84a29acbf8541a8aa1e6f2bcf04d4b2a438` |
| `1B_soup35_ema_qat_lean3out_q_hybrid_int8.bin` | **`v1.2.0`** `10x192-Q-1.00B-Soup` Lean 3-Out INT8 weights | 16,726,028 | `99838bc59e622480369961a4e81b625988d92f8c4b253e3f6fcd9e052c9a9c98` |
| `neurachess-12x192-qat-int8.xml` | Deep `v1.1.0` `12x192` Mid-`2..9` INT8 IR | 189,780 | `975589f19b1fb721b4c5eca966eec997c589df2685bbcd3f17e6ac809fc1e52b` |
| `neurachess-12x192-qat-int8.bin` | Deep `v1.1.0` `12x192` INT8+FP16 weights | 18,150,392 | `eceadf14d505efeb81d94d0bfa99cefbc6aa702ac2b16839993ab07ff0bdf1fb` |
| `neurachess-12x192-qat-int8.npz` | Folded `12x192` + `HL-Gauss-64` weights | 31,626,990 | `149f17785aac782e697e8c5744372765e62c559af2f7d9ec485bff323ef82d3a` |
| `neurachess-10x192-fp16.xml` | `v1.0.x` `10x192` OpenVINO FP16 graph | 156,496 | `f70fbd9923c678af0436f2272d4ad962e66b543ee844d620813846c5f7c3b698` |
| `neurachess-10x192-fp16.bin` | `v1.0.x` `10x192` OpenVINO FP16 weights | 14,006,754 | `1ef15c58aedddbe2562a59373c37c5933a02b9371530f0c725b023b17dc7f02e` |
| `neurachess-10x192-fp16.npz` | Folded `10x192` FP16 weights (`346M`) | 25,912,231 | `52805a44b0670aae020bee26c0bc05e9ee6cb9835b374759ff1cd5f0ddb8f049` |
| `neurachess-10x128-fp16.xml` | Compact `10x128` OpenVINO FP16 graph | 156,453 | `2075a2dbaf51147d48c68190658a277368388d17766d601196f91a40846d60b5` |
| `neurachess-10x128-fp16.bin` | Compact `10x128` OpenVINO FP16 weights | 6,394,338 | `a5a5b302b091acb7cc67ff81d03e074ff4d11ec0143942547bd8a30a22119c66` |
| `neurachess-10x128-fp16.npz` | Folded `10x128` FP16 weights | 11,801,621 | `2f993da939572cc02c7962e6b69761706229b2331434d44e887369804cf65137` |

*(If no NPU device is detected at runtime, `NPUModel` automatically falls back to `CPU` execution.)*

### 3. Play or Connect via UCI

- **UCI Protocol (`lichess-bot`, `python-chess`, Arena, Cutechess, BanksiaGUI):**
  ```bash
  neurachess-uci
  # or: python -m neurachess.uci
  # or Windows unbuffered UCI wrapper (recommended for lichess-bot with concurrency=1):
  scripts\run_neurachess_uci.bat
  ```
  Supported UCI options: `Hash` (default `64` MB), `Ponder` (`true`/`false`, default `true`), `OwnBook` (`true`/`false`, default `true`), `SyzygyPath` (`auto` or tablebase directory path), `Move Overhead` (default `150` ms, min `0`, max `2000`; clock allocation subtracts `Move Overhead` with a `25` ms floor), `Threads` (default `1`, `1..16`), `BatchSize` (default `16`, `1..32`), and `EncoderLayout` (`gpu_unpack`/`legacy_flip`). Emits full `info depth ... seldepth ... score cp|mate ... nodes ... nps ... time ... pv ...` telemetry and `OwnBook` moves via [`assets/opening_book.json`](assets/opening_book.json) (`1,877` EPDs from **CC0-1.0** [`lichess-org/chess-openings`](https://github.com/lichess-org/chess-openings)), with a `1.30x` cold-start clock buffer when `retained_visits < 100`.
- **Interactive Terminal:**
  ```bash
  python play_terminal.py
  ```
- **Web Match UI (FastAPI, requires `[web]`):**
  ```bash
  uvicorn neurachess.web.server:app --host 127.0.0.1 --port 8000
  ```
  *(Note: interactive Stockfish web sparring looks for `STOCKFISH_PATH` environment variable, `stockfish` on `PATH`, or `bin/stockfish/`.)*

### 4. Evaluation, Benchmarking & Training CLI

```bash
# Offline dataset benchmark on packed bitboard shards
python -m neurachess.scripts.evaluate dataset --checkpoint path/to/ckpt.pt --datasets test_split.bin --layout gpu_unpack

# Tactical puzzle benchmark across node budgets
python -m neurachess.scripts.evaluate puzzles --puzzles-json puzzles.json --device NPU --budgets 1 50 200 800

# Stockfish paired match & bidirectional pentanomial SPRT (`--own-book` defaults to False)
python -m neurachess.scripts.match_runner vs-stockfish --sf-elo 3000 --pairs 25 --movetime-ms 400 --ponder

# Rebuild the bundled CC0 Lichess ECO opening book
python -m neurachess.scripts.build_opening_book

# Export PyTorch checkpoint or folded NPZ to OpenVINO FP16 / W8A16 IR (requires `[train]`)
python -m neurachess.scripts.export_openvino --pt ckpt.pt --blocks 10 --channels 192 --out-xml weights/model.xml --verify-device NPU

# Build bitboard shards from Lichess eval dumps or Stockfish (requires `[train]`)
python -m neurachess.training.build_dataset from-lichess --input lichess_db_eval.jsonl.zst --out-dir data/shards --write-sidecar

# Train Conv73 SE-ResNet with GPU VRAM shard mixing and Boltzmann soft targets (requires `[train]`)
python -m neurachess.training.train --shards-dir data/shards --blocks 10 --channels 192 --out-dir weights/runs/exp1
```

### 5. Run Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Benchmark Summary

### Paired NPU Head-to-Head Progression (`400 ms/move`, static `B=16`, Syzygy ON)

| Matchup (`openings_320`, `max_plies=150`) | Games | Score (W-D-L) | Score % | 95% Wilson CI | Elo | SPRT LLR | NPS (A / B) |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| **`v1.1.0` (`10x192-QAT-786M`)** vs `v1.0.2` (`346M` FP16) | **100** | **`54W - 22D - 24L`** | **65.00%** | `[55.25%, 73.64%]` | **`+107.6`** | **`+3.57` (PASS)** | `3,851 / 3,252` |
| **`v1.1.0` (`10x192-QAT-786M`)** vs `10x192-QAT-386M` | 100 | `36W - 32D - 32L` | **52.00%** | `[42.32%, 61.54%]` | `+13.9` | `+0.20` | `3,835 / 3,871` |
| `10x192-QAT-386M` INT8 vs `v1.0.2` FP16 (Combined) | 180 | `77W - 55D - 48L` | **58.06%** | `[50.75%, 65.02%]` | `+56.5` | `+3.83` (PASS) | `3,891 / 3,254` |
| `10x192-QAT-386M` INT8 vs FP16 (Single-Variable Ctrl) | 100 | `43W - 22D - 35L` | **54.00%** | `[44.26%, 63.44%]` | `+27.9` | `+0.99` | `3,821 / 3,225` |
| **`v1.2.0` (`10x192-Q-1.00B`, $Q$-FPU `0.40`)** vs 3-Head Ctrl | 100 | `40W - 22D - 38L` | **51.00%** | `[41.35%, 60.58%]` | **`+7.0`** | `-0.18` | `3,665 / 3,727` |
| `v1.2.0` (`10x192-Q-1.00B-Soup`, Lean 3-Out) vs `v1.1.0` | 100 | `35W - 25D - 40L` | 47.50% | `[37.99%, 57.20%]` | `-17.4` | `-1.44` | `3,718 / 3,775` |
| `v1.2.0` (`10x192-Q-1.00B-EMA`, Lean 2-Out PTQ) vs `v1.1.0` | 100 | `30W - 31D - 39L` | 45.50% | `[36.09%, 55.25%]` | `-31.3` | `-3.59` (H0) | `3,769 / 3,785` |

### Stockfish 19 Calibration Ladder (`400 ms/move`, `Ponder=True`, `fast_tree`, Syzygy ON)

| Engine / Opponent Configuration | Opening Slice | Games | Score (W-D-L) | Score % | 95% Wilson CI | Eff NPS | Implied `UCI_Elo` |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: |
| **`v1.1.0` (`10x192-QAT-786M`)** vs SF19 `UCI_Elo 2800` | `openings_320[0:25]` | 50 | `34W - 9D - 7L` | **77.00%** | `[63.67%, 86.48%]` | 3,975.6 | 3,009.9 |
| **`v1.1.0` (`10x192-QAT-786M`)** vs SF19 `UCI_Elo 3000` | `openings_320[25:50]` | 50 | `6W - 32D - 12L` | **44.00%** | `[31.16%, 57.69%]` | **4,075.1** | 2,958.1 |
| **`v1.1.0` (`10x192-QAT-786M`)** vs SF19 `UCI_Elo 3190` | `openings_320[50:75]` | 50 | `6W - 19D - 25L` | **31.00%** | `[19.93%, 44.78%]` | 3,989.3 | 3,051.0 |
| **`v1.1.0` (`10x192-QAT-786M`) Combined Ladder** | `openings_320[0:75]` | **150** | **`46W - 60D - 44L`** | **50.67%** | **`[42.75%, 58.55%]`** | **4,013.3** | **`3,006.3`** |
| `10x192-QAT-386M` (Soup) Combined Ladder | `openings_320[0:75]` | 150 | `38W - 71D - 41L` | 49.00% | `[41.13%, 56.92%]` | 3,987.9 | 2,992.2 |
| `10x192-QAT-386M` vs SF19 **Fixed `3,500` nodes** | `openings_320[165:185]` | **40** | **`30W - 7D - 3L`** | **83.75%** | `[69.48%, 92.11%]` | 3,771.0 | — |
| `v1.0.0` (`346M FP16`) Combined Ladder (no Syzygy) | `OPENINGS_50` | 150 | `44W - 62D - 44L` | 50.00% | `[42.10%, 57.90%]` | 3,275.0 | — |
| `v1.0.0` (`346M FP16`) vs SF19 Fixed `3,500` nodes | `OPENINGS_50` | 40 | `9W - 12D - 19L` | 37.50% | `[24.22%, 52.97%]` | 3,264.6 | — |

### Unified 150K GPU Evaluation & Clean Lichess Puzzles (`n = 1,979`, Syzygy OFF)

| Model Milestone | OOD 50K MSE | In-Reg 50K MSE | Held-Out 50K MSE | Mean 150K MSE | OOD Top-1 | Held-Out Sign | Clean Puzzles `0n / 50n / 200n / 800n` (`2100+` @ `800n`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `10x192-QAT-386M` (Soup INT8) | 0.06101 | 0.05460 | 0.04649 | 0.05403 | 51.43% | 96.20% | `68.72% / 78.27% / 83.07% / 95.65%` (`86.55%`) |
| **`v1.1.0` (`10x192-QAT-786M`, 50/50 QAT)** | 0.05384 | 0.04729 | 0.04270 | 0.04794 | 51.50% | 96.87% | `67.00% / 77.01% / 83.63% / 95.35%` (`85.53%`) |
| **`v1.2.0` (`10x192-Q-1.00B-EMA`, Pre-QAT)** | **0.05333** | **0.04694** | 0.03706 | **0.04578** | **51.89%** | 96.91% | `67.21% / — / 85.09% / —` (INT8 $Q$-FPU `0.40`) |
| **`v1.2.0` (`10x192-Q-1.00B`, 4-Head $Q$-FPU)** | 0.05751 | 0.05076 | **0.03372** | 0.04733 | 51.74% | **97.03%** | `67.96% / 76.81% / 85.09% /` **`95.76%`** (**`87.31%`**) |
| **`v1.2.0` (`10x192-Q-1.00B-Soup`, Lean 3-Out)** | — | — | — | — | — | — | `67.96% / — /` **`85.19%`** `/ —` (`5.57 ms` `B=16`) |

*(All `v1.1.0` Stockfish 19 matches ran at `400 ms/move`, static `B=16`, `--ponder`, `--fpu-reduction 0.20`, `--max-plies 150 --adjudicate-material`, and 5-piece Syzygy tablebases on `openings_320`; note archived [`phase4_sf3000_openings320_apples_50g.json`](eval_results/phase4_sf3000_openings320_apples_50g.json) scoring `33.0%` was `Hybrid_Mid27 PTQ (no QAT)`, not `v1.0.2 Pure FP16`. All dataset splits contain 50,000 positions except `test_in_region_v2` with 49,998 and the 5,000-position `test_far_50k` NPU INT8 eval. Cross-entropy masks illegal move logits to `-10000.0`.)*

**Scope & Caveat:** Stockfish `UCI_Elo` reflects Stockfish's internal calibrated node/time limiter, not human tournament ratings or external CCRL pools, and individual 50-game rungs carry wide 95% Wilson intervals (`~±95 Elo` binomial half-width, ~190 Elo CI width at `n=50`). All `v1.0.0` Stockfish 19 matches above reflect `OwnBook=False` under fixed `400 ms/move` without Syzygy or `v1.0.2` cold-start `wtime`/`btime` buffering.

---

## License & Attribution

Released under the **GNU General Public License v3.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).

**GPL-3.0 Section 7(b) Attribution Notice:** Any derivative work, redistribution, or use of the source code or pretrained weights must retain explicit attribution to **Musheer Alam (https://github.com/Musheer360/NeuraChess)**.
