# NeuraChess

NeuraChess is a neural chess engine targeting Intel Core Ultra **Intel AI Boost NPU** acceleration through OpenVINO, paired with an AVX2/BMI2 C++23 Structure-of-Arrays (SoA) batched PUCT search tree (`fast_tree`).

Instead of relying on traditional alpha-beta hand-crafted heuristics, NeuraChess trains 18-plane Squeeze-and-Excitation residual networks (`ChessSEResNet`) with a parameter-efficient 73-plane spatial policy head (`Conv73`, mapping directly to the 4,672 AlphaZero move encodings) on packed bitboard datasets, folds BatchNorm affine parameters statically into FP16 OpenVINO IR weights, and executes asynchronous CPU tree traversal alongside batched NPU inference (`B=16`).

---

## Key Highlights

- **Flagship `v1.1.0` `10x192 Mid-2..7 QAT INT8` (`A2_soup75`) & Deep `12x192` Architecture:** Combines sensitivity-guided middle-block `W8A8` QAT, an `HL-Gauss-64` distributional value head, `MLH` auxiliary training, and a `0.75 * A2 + 0.25 * 346M` trunk/policy weight soup recalibrated on clean `gpu_unpack` boards (`neurachess-10x192-qat-int8`), plus an optional 12-block `300M`-token `12x192` INT8 bundle (`neurachess-12x192-qat-int8`).
- **Verified Playing-Strength Gain (`+56.5` Elo vs `v1.0.2` Pure FP16 & `+46.25%` at SF `3,500` Nodes):**
  - `A2_base` (`Mid-2..7 QAT INT8`) beat `v1.0.2 Pure FP16` (`neurachess-10x192-fp16`) by **`+56.5` Elo** across `180` head-to-head games (**`58.06%`**, `77W-55D-48L`, SPRT `ACCEPT_H1` **`LLR = +3.8313`**, `3,890.7` vs `3,253.9` NPS), and `A2_soup75` beat `A2_base` **`54.00%`** (`16W-22D-12L`) over `50` paired games.
  - Against **Stockfish 19 (`go nodes 3500`, uncapped, `40` games)** at `400 ms/move` with Syzygy, `A2_soup75` jumped from `v1.0.2`'s **`37.50%`** (`9W-12D-19L`) to **`83.75%`** (**`30W-7D-3L`**, Wilson 95% CI `[69.48%, 92.11%]`).
  - On **Stockfish 19 `UCI_Elo`** (`400 ms/move`, `--ponder`, Syzygy, `openings_320[0:75]`), `A2_soup75` jumped `2800` from `62.00%` to **`74.00%`** (`29W-16D-5L`), cut `3000` losses down to `14` (**`42.00%`**, `6W-30D-14L` at **`4,025.1` eff NPS**), and averaged **`3,987.9` eff NPS** across `150` games (`49.00%`, `38W-71D-41L`).
- **C++23 `fast_tree` Search (`src/neurachess/engine/fast_tree`):** Zero-allocation cache-aligned SoA bitboard node arena with AVX2/BMI2 legal move generation (CPW Perft & 2,000-position differential fuzzer verified), 5-piece Syzygy tablebase probing, Polyglot legal-EP 64-bit Zobrist canonical transposition neural cache (`64 MB`), safe subtree reuse across moves (with path-dependent draw invalidation), and exact minimax mate-distance propagation.
- **Empirical Rigor:** Every architectural choice (`Conv73` vs dense policy, CNN vs transformer attention latency on NPU 3720, `B=16` vs `B=32`/`64`, FP16 vs PTQ/QAT W8A8, Phase 4 Job A/B scaling, and Stockfish 19 calibration) is documented with confidence intervals and ablation data in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

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
|       Search Orchestrator (src/neurachess/search/npu_batched_puct.py) |
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

| Model Asset | Purpose | Size (Bytes) | SHA-256 |
| --- | --- | ---: | --- |
| `neurachess-10x192-qat-int8.xml` | Default `v1.1.0` 10x192 Mid-`2..7` INT8 IR | 162,064 | `86fce43ef5a89580f548cf56fa79079b055afdf1da8eb0903ff6537874df5bcb` |
| `neurachess-10x192-qat-int8.bin` | Default `v1.1.0` 10x192 INT8+FP16 weights (`A2_soup75`) | 16,669,672 | `9db3c4d6e1f45a07a4cc0cd6cfc5898edc1c3ffdc8171137a754de0b7f3db260` |
| `neurachess-10x192-qat-int8.npz` | Folded `A2_soup75` `10x192` + `HL-Gauss-64` weights | 28,762,840 | `99d2689b76cec74927f425d121cddc6a9e2b6360704a721ba0b0f67cc30c5df2` |
| `neurachess-12x192-qat-int8.xml` | Deep `v1.1.0` 12x192 Mid-`2..9` INT8 IR (`Job B`) | 189,780 | `975589f19b1fb721b4c5eca966eec997c589df2685bbcd3f17e6ac809fc1e52b` |
| `neurachess-12x192-qat-int8.bin` | Deep `v1.1.0` 12x192 INT8+FP16 weights | 18,150,392 | `eceadf14d505efeb81d94d0bfa99cefbc6aa702ac2b16839993ab07ff0bdf1fb` |
| `neurachess-12x192-qat-int8.npz` | Folded `Job B` `12x192` + `HL-Gauss-64` weights | 31,626,990 | `149f17785aac782e697e8c5744372765e62c559af2f7d9ec485bff323ef82d3a` |
| `neurachess-10x192-fp16.xml` | `v1.0.x` 10x192 OpenVINO FP16 graph | 156,496 | `f70fbd9923c678af0436f2272d4ad962e66b543ee844d620813846c5f7c3b698` |
| `neurachess-10x192-fp16.bin` | `v1.0.x` 10x192 OpenVINO FP16 weights | 14,006,754 | `1ef15c58aedddbe2562a59373c37c5933a02b9371530f0c725b023b17dc7f02e` |
| `neurachess-10x192-fp16.npz` | Folded `10x192` FP16 weights (`346M ema_09999`) | 25,912,231 | `52805a44b0670aae020bee26c0bc05e9ee6cb9835b374759ff1cd5f0ddb8f049` |
| `neurachess-10x128-fp16.xml` | Compact `10x128` OpenVINO FP16 graph | 156,453 | `2075a2dbaf51147d48c68190658a277368388d17766d601196f91a40846d60b5` |
| `neurachess-10x128-fp16.bin` | Compact `10x128` OpenVINO FP16 weights | 6,394,338 | `a5a5b302b091acb7cc67ff81d03e074ff4d11ec0143942547bd8a30a22119c66` |
| `neurachess-10x128-fp16.npz` | Folded `10x128` FP16 weights (`ema_09999`) | 11,801,621 | `2f993da939572cc02c7962e6b69761706229b2331434d44e887369804cf65137` |

*(If no NPU device is detected at runtime, `NPUModel` automatically falls back to `CPU` execution.)*

### 3. Play or Connect via UCI

- **UCI Protocol (Arena, Cutechess, BanksiaGUI):**
  ```bash
  neurachess-uci
  # or: python -m neurachess.uci
  ```
  Supported UCI options: `Hash` (default `64` MB), `Ponder` (`false`/`true`), `EncoderLayout` (`gpu_unpack`/`legacy_flip`), `OwnBook` (`true`/`false`, default `true`). `OwnBook` uses [`assets/opening_book.json`](assets/opening_book.json) (`1,877` EPDs built cleanly via `python -m neurachess.scripts.build_opening_book` from **CC0-1.0** [`lichess-org/chess-openings`](https://github.com/lichess-org/chess-openings), `a.tsv`–`e.tsv`), and dynamic `wtime`/`btime` search applies a `1.30x` cold-start buffer when `retained_visits < 100`.
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

### `v1.1.0` Flagship (`A2_soup75` `10x192 Mid-2..7 QAT INT8`) vs `v1.0.2` (`346M 10x192 FP16`)

- **Head-to-Head (`400 ms/move`, static `B=16`, Syzygy):**
  - `A2_base` (`Mid-2..7 QAT INT8`) beat `v1.0.2 Pure FP16` (`neurachess-10x192-fp16`) by **`+56.5` Elo** across **`180` paired games** (`openings_320[75:165]`): **`58.06%`** (`77W - 55D - 48L`, Wilson 95% CI `[50.75%, 65.02%]`, pentanomial `[2, 6, 49, 27, 6]`, SPRT `(0, 20)` **`ACCEPT_H1` `LLR = +3.8313 > 2.9444`**), NPS **`3,890.7` vs `3,253.9` (`+19.6%`)**.
  - `A2_soup75` (`neurachess-10x192-qat-int8`, `0.75*A2 + 0.25*346M` trunk+`Conv73` + `HL-Gauss-64` + recalibrated `Mid-2..7` INT8) beat `A2_base` **`54.00%`** (`16W - 22D - 12L`, `50` games, `openings_320[75:100]`, pentanomial `[0, 6, 11, 6, 2]`).
  - `A2_soup75` Clean `1,979` Lichess Puzzles (`21` contaminated IDs excluded): `0n` (Policy Top-1): **`68.72%`** (`1,360`), `50n`: **`78.27%`** (`1,549`, `3,064.2` NPS), `200n`: **`83.07%`** (`1,644`, `3,133.5` NPS), `800n`: **`95.65%`** (`1,893`, `3,387.1` NPS; `2100+`: `86.55%`).

### Stockfish 19 Calibration (`400 ms/move`, `Ponder=True`, `fast_tree`)

| Engine / Opponent Configuration | Opening Slice | Games | Score (W-D-L) | Score % | 95% Wilson CI | Eff NPS |
| --- | --- | ---: | --- | ---: | --- | ---: |
| **`v1.1.0` (`A2_soup75`)** vs Stockfish 19 `UCI_Elo 2800` | `openings_320[0:25]` | 50 | `29W - 16D - 5L` | **74.00%** | `[60.45%, 84.13%]` | 3,898.9 |
| **`v1.1.0` (`A2_soup75`)** vs Stockfish 19 `UCI_Elo 3000` | `openings_320[25:50]` | 50 | `6W - 30D - 14L` | **42.00%** | `[29.37%, 55.77%]` | **4,025.1** |
| **`v1.1.0` (`A2_soup75`)** vs Stockfish 19 `UCI_Elo 3190` | `openings_320[50:75]` | 50 | `3W - 25D - 22L` | **31.00%** | `[19.93%, 44.78%]` | **4,039.6** |
| **`v1.1.0` Cumulative (`UCI_Elo` 2800–3190)** | `openings_320[0:75]` | **150** | **`38W - 71D - 41L`** | **49.00%** | — | **3,987.9** |
| **`v1.1.0` (`A2_soup75`)** vs Stockfish 19 **Fixed `3,500` nodes** | `openings_320[165:185]` | **40** | **`30W - 7D - 3L`** | **83.75%** | `[69.48%, 92.11%]` | 3,771.0 |
| `v1.0.0` (`346M FP16`) vs Stockfish 19 `UCI_Elo 2800` | `OPENINGS_50` | 50 | `20W - 22D - 8L` | 62.00% | `[48.15%, 74.14%]` | 3,247.7 |
| `v1.0.0` (`346M FP16`) vs Stockfish 19 `UCI_Elo 3000` | `OPENINGS_50` | 50 | `16W - 17D - 17L` | 49.00% | `[35.72%, 62.42%]` | 3,276.4 |
| `v1.0.0` (`346M FP16`) vs Stockfish 19 `UCI_Elo 3190` | `OPENINGS_50` | 50 | `8W - 23D - 19L` | 39.00% | `[26.73%, 52.84%]` | 3,301.0 |
| `v1.0.0` Cumulative (`UCI_Elo` 2800–3190) | `OPENINGS_50` | 150 | `44W - 62D - 44L` | 50.00% | `[42.10%, 57.90%]` | 3,275.0 |
| `v1.0.0` (`346M FP16`) vs Stockfish 19 Fixed `1,000` nodes | `OPENINGS_50` | 40 | `23W - 15D - 2L` | 76.25% | `[61.14%, 86.75%]` | 3,217.7 |
| `v1.0.0` (`346M FP16`) vs Stockfish 19 Fixed `3,500` nodes | `OPENINGS_50` | 40 | `9W - 12D - 19L` | 37.50% | `[24.22%, 52.97%]` | 3,264.6 |
| `v1.0.0` (`346M FP16`) vs Stockfish 19 Fixed `10,000` nodes | `OPENINGS_50` | 40 | `1W - 12D - 27L` | 17.50% | `[8.75%, 31.95%]` | 3,260.2 |
| `v1.0.0` (`346M FP16`) vs Stockfish 19 Fixed `100,000` nodes | `OPENINGS_50` | 40 | `0W - 6D - 34L` | 7.50% | `[2.58%, 19.86%]` | 3,215.6 |

### Offline 150k Unified Evaluation & Clean Tactical Puzzles (`10x192` & `10x128` History)

| Model | Params | `far_test` CE (Top-1) | `in_region_v2` CE (Top-1) | `heldout_400m` CE (Top-1) | Held-Out Val MSE (Sign%) | Puzzles 1n / 50n / 200n / 800n (n=1,979) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1M Pilot (6x128)` | 11.55M | 2.2476 (33.15%) | — | — | — | 43.15% / 56.44% / 68.22% / 82.16% |
| `S5a (6x128, 70M)` | 1.99M | 1.7971 (45.47%) | 1.8337 (43.78%) | — | — | 57.10% / 70.54% / 78.73% / 91.81% |
| `Cont 346M (10x128)` | 3.20M | 1.6163 (49.72%) | 1.6466 (48.48%) | 1.6057 (48.29%) | 0.0457 (95.49%) | 67.61% / 77.36% / 82.82% / 94.34%* |
| `Full 346M (10x192 FP16)` | 7.01M | 1.5655 (51.43%) | 1.5887 (50.16%) | 1.5522 (49.92%) | 0.0402 (96.03%) | 71.45% / 79.38% / 84.03% / 95.96% |
| **`v1.1.0` (`A2_soup75` QAT INT8)** | **7.15M** | **1.4911** (50.78%, 5k) | — | — | **95.12% dec. sign** (5k) | **68.72% / 78.27% / 83.07% / 95.65%** |

*(All `v1.1.0` Stockfish 19 matches ran at `400 ms/move`, static `B=16`, `--ponder`, `--fpu-reduction 0.20`, `--max-plies 150 --adjudicate-material`, and 5-piece Syzygy tablebases on `openings_320`; note archived [`phase4_sf3000_openings320_apples_50g.json`](eval_results/phase4_sf3000_openings320_apples_50g.json) scoring `33.0%` was `Hybrid_Mid27 PTQ (no QAT)`, not `v1.0.2 Pure FP16`. All dataset splits contain 50,000 positions except `test_in_region_v2` with 49,998 and the 5,000-position `test_far_50k` NPU INT8 eval. Cross-entropy masks illegal move logits to `-10000.0`.)*

**Scope & Caveat:** Stockfish `UCI_Elo` reflects Stockfish's internal calibrated node/time limiter rather than human tournament ratings or external CCRL pools, and individual 50-game rungs carry wide 95% Wilson intervals (`~±95 Elo` binomial half-width, ~190 Elo CI width at `n=50`). All `v1.0.0` Stockfish 19 matches above reflect `OwnBook=False` under fixed `400 ms/move` without Syzygy or `v1.0.2` cold-start `wtime`/`btime` buffering.

---

## License & Attribution

Released under the **GNU General Public License v3.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).

**GPL-3.0 Section 7(b) Attribution Notice:** Any derivative work, redistribution, or use of the source code or pretrained weights must retain explicit attribution to **Musheer Alam (https://github.com/Musheer360/NeuraChess)**.
