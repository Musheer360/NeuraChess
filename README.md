# NeuraChess

NeuraChess is a neural chess engine targeting Intel Core Ultra **Intel AI Boost NPU** acceleration through OpenVINO, paired with an AVX2/BMI2 C++23 Structure-of-Arrays (SoA) batched PUCT search tree (`fast_tree`).

Instead of relying on traditional alpha-beta hand-crafted heuristics, NeuraChess trains 18-plane Squeeze-and-Excitation residual networks (`ChessSEResNet`) with a parameter-efficient 73-plane spatial policy head (`Conv73`, mapping directly to the 4,672 AlphaZero move encodings) on packed bitboard datasets, folds BatchNorm affine parameters statically into FP16 OpenVINO IR weights, and executes asynchronous CPU tree traversal alongside batched NPU inference (`B=16`).

---

## Key Highlights

- **Hardware-Matched Backbone (`10x192` & `10x128` FP16):** Trained on ~346M Lichess evaluation positions (`ema_09999`), achieving **1.5522** legal-masked cross-entropy (**49.92% Top-1 / 79.41% Top-3 / 96.03% decisive value sign agreement**) on the 50k held-out split (`heldout_400m`), and **95.96%** solve rate at 800 search nodes on 1,979 clean tactical puzzles.
- **C++23 `fast_tree` Search (`engine/fast_tree`):** Zero-allocation cache-aligned SoA bitboard node arena with AVX2/BMI2 legal move generation, 64-bit Zobrist canonical transposition neural cache (`64 MB`), safe subtree reuse across moves (with path-dependent draw invalidation), and exact minimax mate-distance propagation.
- **Sustained NPU Throughput:** **~3,275 effective NPS** (`10x192`, `400 ms/move` with internal pondering enabled) and **~4,810 effective NPS** (`10x128`, Ponder OFF) under OpenVINO `V0_FP16 [LATENCY]` compilation.
- **Empirical Rigor:** Every architectural choice (Conv73 vs dense policy, CNN vs transformer attention latency on NPU 3720, `B=16` vs `B=32`/`64`, FP16 vs PTQ/QAT W8A8, and Stockfish 19 calibration) is documented with confidence intervals and ablation data in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## System Architecture

```text
+-----------------------------------------------------------------------+
|                       UCI / CLI / FastAPI Frontend                     |
|    uci/engine_uci.py  |  play_terminal.py  |  web/server.py           |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|              Search Orchestrator (search/npu_batched_puct.py)         |
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
|  engine/fast_tree/fast_tree.cpp   |   |  models/npu_compiler.py       |
|  engine/fast_tree/bitboard_chess  |   |  SE-ResNet (FP16 IR) + Cache  |
+-----------------------------------+   +-------------------------------+
```

For full details on board tensor layouts (`gpu_unpack` vs `legacy_flip`), Conv73 indexing, C++ SoA memory layout, and OpenVINO NPU compilation hints, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quick Start

### 1. Environment & Dependencies

NeuraChess requires Python 3.10+, OpenVINO Runtime, PyTorch (for training/export), and a C++23 compiler (`g++` / MinGW-w64 / Clang) to build `fast_tree` on first launch.

```bash
pip install -r requirements.txt
```

### 2. Download Verified Pretrained Weights

Model weights are tracked via GitHub Releases (not committed to Git) and verified via SHA-256 checksums:

```bash
# Download default 10x192 FP16 OpenVINO IR + folded NumPy weights
python scripts/download_weights.py --model 10x192

# Or download both 10x192 and high-speed 10x128
python scripts/download_weights.py --model all
```

| Model Asset | Purpose | Size (Bytes) | SHA-256 |
| --- | --- | ---: | --- |
| `neurachess-10x192-fp16.xml` | Default 10x192 OpenVINO FP16 graph | 156,496 | `f70fbd9923c678af0436f2272d4ad962e66b543ee844d620813846c5f7c3b698` |
| `neurachess-10x192-fp16.bin` | Default 10x192 OpenVINO FP16 weights | 14,006,754 | `1ef15c58aedddbe2562a59373c37c5933a02b9371530f0c725b023b17dc7f02e` |
| `neurachess-10x192-fp16.npz` | Folded 10x192 weights (`ema_09999`) | 25,912,231 | `52805a44b0670aae020bee26c0bc05e9ee6cb9835b374759ff1cd5f0ddb8f049` |
| `neurachess-10x128-fp16.xml` | Compact 10x128 OpenVINO FP16 graph | 156,453 | `2075a2dbaf51147d48c68190658a277368388d17766d601196f91a40846d60b5` |
| `neurachess-10x128-fp16.bin` | Compact 10x128 OpenVINO FP16 weights | 6,394,338 | `a5a5b302b091acb7cc67ff81d03e074ff4d11ec0143942547bd8a30a22119c66` |
| `neurachess-10x128-fp16.npz` | Folded 10x128 weights (`ema_09999`) | 11,801,621 | `2f993da939572cc02c7962e6b69761706229b2331434d44e887369804cf65137` |

*(If no NPU device is detected at runtime, `NPUModel` automatically falls back to `CPU` execution.)*

### 3. Play or Connect via UCI

- **UCI Protocol (Arena, Cutechess, BanksiaGUI):**
  ```bash
  python uci/engine_uci.py
  ```
  Supported UCI options: `Hash` (default `64` MB), `Ponder` (`false`/`true`), `EncoderLayout` (`gpu_unpack`/`legacy_flip`).
- **Interactive Terminal:**
  ```bash
  python play_terminal.py
  ```
- **Web Match UI (FastAPI):**
  ```bash
  uvicorn web.server:app --host 127.0.0.1 --port 8000
  ```
  *(Note: interactive Stockfish web sparring looks for `STOCKFISH_PATH` environment variable, `stockfish` on `PATH`, or `bin/stockfish/`.)*

### 4. Evaluation, Benchmarking & Training CLI

```bash
# Offline dataset benchmark on packed bitboard shards
python scripts/evaluate.py dataset --checkpoint path/to/ckpt.pt --datasets test_split.bin --layout gpu_unpack

# Tactical puzzle benchmark across node budgets
python scripts/evaluate.py puzzles --puzzles-json puzzles.json --device NPU --budgets 1 50 200 800

# Stockfish paired match & bidirectional pentanomial SPRT
python scripts/match_runner.py vs-stockfish --sf-elo 3000 --pairs 25 --movetime-ms 400 --ponder

# Export PyTorch checkpoint or folded NPZ to OpenVINO FP16 / W8A16 IR
python scripts/export_openvino.py --pt ckpt.pt --blocks 10 --channels 192 --out-xml weights/model.xml --verify-device NPU

# Build bitboard shards from Lichess eval dumps or Stockfish
python training/build_dataset.py from-lichess --input lichess_db_eval.jsonl.zst --out-dir data/shards --write-sidecar

# Train Conv73 SE-ResNet with GPU VRAM shard mixing and Boltzmann soft targets
python training/train.py --shards-dir data/shards --blocks 10 --channels 192 --out-dir weights/runs/exp1
```

### 5. Run Unit Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Benchmark Summary (`10x192` FP16 Default)

### Offline 150k Unified Evaluation & Clean Tactical Puzzles

| Model | Params | `far_test` CE (Top-1) | `in_region_v2` CE (Top-1) | `heldout_400m` CE (Top-1) | Held-Out Val MSE (Sign%) | Puzzles 1n / 50n / 200n / 800n (n=1,979) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `1M Pilot (6x128)` | 11.55M | 2.2476 (33.15%) | — | — | — | 43.15% / 56.44% / 68.22% / 82.16% |
| `S5a (6x128, 70M)` | 1.99M | 1.7971 (45.47%) | 1.8337 (43.78%) | — | — | 57.10% / 70.54% / 78.73% / 91.81% |
| `Cont 346M (10x128)` | 3.20M | 1.6163 (49.72%) | 1.6466 (48.48%) | 1.6057 (48.29%) | 0.0457 (95.49%) | 67.61% / 58.16% / 74.89% / 94.34% |
| **`Full 346M (10x192)`** | **7.01M** | **1.5655 (51.43%)** | **1.5887 (50.16%)** | **1.5522 (49.92%)** | **0.0402 (96.03%)** | **71.45% / 79.38% / 84.03% / 95.96%** |

*(All dataset splits contain 50,000 positions except `test_in_region_v2` with 49,998. Cross-entropy masks illegal move logits to `-10000.0`. Notice the clean head-to-head match between `10x192` and `10x128` at equal `400 ms/move` without pondering favored `10x192` by **54.50%**, `38W-33D-29L`, 95% Wilson CI `[44.75%, 63.91%]` over 100 paired games despite `10x128` searching at 4,810 vs 3,290 eff NPS.)*

### Stockfish 19 Calibration (`400 ms/move`, `Ponder=True`, `fast_tree`)

| Opponent Configuration | Games | Score (W-D-L) | Score % | 95% Wilson CI | Eff NPS |
| --- | ---: | --- | ---: | --- | ---: |
| Stockfish 19 `UCI_Elo 2800` | 50 | `20W - 22D - 8L` | **62.00%** | `[48.15%, 74.14%]` | 3,247.7 |
| Stockfish 19 `UCI_Elo 3000` | 50 | `16W - 17D - 17L` | **49.00%** | `[35.72%, 62.42%]` | 3,276.4 |
| Stockfish 19 `UCI_Elo 3190` | 50 | `8W - 23D - 19L` | **39.00%** | `[26.73%, 52.84%]` | 3,301.0 |
| **Cumulative (`UCI_Elo` 2800–3190)** | **150** | **`44W - 62D - 44L`** | **50.00%** | **`[42.10%, 57.90%]`** | **3,275.0** |
| Stockfish 19 Fixed `1,000` nodes (uncapped) | 40 | `23W - 15D - 2L` | **76.25%** | `[61.14%, 86.75%]` | 3,217.7 |
| Stockfish 19 Fixed `3,500` nodes (uncapped) | 40 | `9W - 12D - 19L` | **37.50%** | `[24.22%, 52.97%]` | 3,264.6 |
| Stockfish 19 Fixed `10,000` nodes (uncapped) | 40 | `1W - 12D - 27L` | **17.50%** | `[8.75%, 31.95%]` | 3,260.2 |
| Stockfish 19 Fixed `100,000` nodes (uncapped) | 40 | `0W - 6D - 34L` | **7.50%** | `[2.58%, 19.86%]` | 3,215.6 |

**Scope & Caveat:** Stockfish `UCI_Elo` reflects Stockfish's internal calibrated node/time limiter rather than human tournament ratings or external CCRL pools. Under fixed-node uncapped Stockfish 19 (`Threads=1, Hash=16`), 400 ms NeuraChess sits between the 1,000-node and 3,500-node Stockfish strength bands (~1,300 effective NPU tree visits per move).

---

## License & Attribution

Released under the **GNU General Public License v3.0** ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)).

**GPL-3.0 Section 7(b) Attribution Notice:** Any derivative work, redistribution, or use of the source code or pretrained weights must retain explicit attribution to **Musheer Alam (https://github.com/Musheer360/NeuraChess)**.
