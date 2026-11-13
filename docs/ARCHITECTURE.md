# NeuraChess Architecture

This document covers the neural representation, OpenVINO NPU compilation pipeline, and C++23 batched PUCT search subsystem in NeuraChess.

---

## 1. Board Tensor Encoding (`src/neurachess/core/board_tensor.py`)

Positions are encoded as `(18, 8, 8)` `float32` spatial tensors (`NUM_PLANES = 18`). NeuraChess supports two explicit layouts controlled via `encoder_layout` (`weights/*.npz` metadata or UCI `EncoderLayout`):

1. **`gpu_unpack` (Default Production Layout):**
   - Planes `0..5`: White `P, N, B, R, Q, K` bitboards mapped without vertical rank flip.
   - Planes `6..11`: Black `p, n, b, r, q, k` bitboards.
   - Plane `12`: Broadcast scalar turn indicator (`0.0` for White, `1.0` for Black).
   - Planes `13..16`: Broadcast castling rights (`WK, WQ, BK, BQ`).
   - Plane `17`: Normalized half-move clock (`halfmove_clock / 100.0`).
   - Rationale: Matches bitboard GPU chunk unpacking directly (`src/neurachess/training/train.py::gpu_unpack_v1`), eliminating host-side bitboard decoding overhead during training.

2. **`legacy_flip` (Canonical Side-to-Move Flip):**
   - Ranks and friendly/enemy piece order flip vertically whenever `board.turn == chess.BLACK`. Retained for pilot compatibility and `src/neurachess/core/tensor_transitions.py` unit validation.

---

## 2. Action Space & Conv73 Policy Head (`src/neurachess/core/constants.py`, `src/neurachess/models/pytorch_se_resnet.py`)

NeuraChess uses the AlphaZero canonical **4,672-action encoding** (`64` source squares x `73` move planes):

- Planes `0..55`: Queen-like sliding rays across 8 compass directions x distances `1..7`.
- Planes `56..63`: Knight jumps (8 offsets).
- Planes `64..72`: Pawn underpromotions (`N, B, R` x left-capture / push / right-capture); queen promotions share the directional sliding planes.
- **Canonical Action Flip:** Black moves reflect vertically (`sq ^ 56`) so all policy heads learn White-oriented source/target geometry.

**Conv73 vs Dense Policy Head:**
Rather than flattening `C x 8 x 8 -> 4,672` with a giant fully connected projection (~9.5M parameters on `6x128` alone), `ChessSEResNet(policy_head="conv73")` emits a single `1x1` convolution from `C -> 73` channels followed by `(B, 73, 8, 8) -> permute(0, 2, 3, 1) -> reshape(B, 4672)`. This cut `6x128` parameter count by **5.39x** (`11.55M -> 2.14M`) and improved `B=16` NPU throughput by **+17.8%** (`5,080.7 -> 5,985.2 pos/s`).

---

## 3. Backbone & OpenVINO NPU Compilation (`src/neurachess/models/se_resnet_backbone.py`, `src/neurachess/models/npu_compiler.py`)

Each backbone is a pre-folded Squeeze-and-Excitation ResNet (`10x192` or `10x128`):
- Initial `3x3` stem convolution (`18 -> C`) + folded BatchNorm bias + ReLU.
- `B` residual blocks, each with:
  - `Conv2d(C, C, 3x3)` + folded BN bias + ReLU
  - `Conv2d(C, C, 3x3)` + folded BN bias
  - Squeeze-and-Excitation gate (`GlobalAvgPool -> Conv1x1(C, C//8) -> ReLU -> Conv1x1(C//8, 2*C)` generating scale + shift modulation before residual addition and ReLU)
- `Conv73` policy head + Value head (`Conv1x1(C, 1) -> ReLU -> Flatten -> Linear(64, 128) -> ReLU -> Linear(128, 1) -> Tanh`).

**Static BatchNorm Folding & NPU Properties:**
During `.npz` export (`ChessSEResNet.export_to_npz`), running `BatchNorm2d` mean/variance and affine `(gamma, beta)` parameters are analytically absorbed into convolution weights/biases (`fold_conv_bn`), eliminating normalization layers inside the OpenVINO graph. Static `B=1` (root/ponder) and `B=16` (leaf search batches) graphs compile under `PERFORMANCE_HINT="LATENCY"` and `NPU_TURBO=True`, with compiled OpenVINO blobs cached in `weights/cache` for fast restarts.

---

## 4. C++23 `fast_tree` Batched PUCT Engine (`src/neurachess/engine/fast_tree/`, `src/neurachess/search/npu_batched_puct.py`)

To saturate batched NPU execution without Python `chess.Board` allocation bottlenecks:

1. **Bitboard State Representation & Zobrist Convention (`bitboard_chess.hpp`):**
   - Header-only C++23 bitboard engine using BMI2 PEXT/AVX2 intrinsics when supported, generating legal moves and incremental Zobrist keys (`zobrist_hi` / `zobrist_lo`) in sub-microsecond per-node time.
   - **Polyglot Legal En-Passant Zobrist Convention:** `ep_square` is XORed into the Zobrist key **only when a legal en-passant capture actually exists** (`board_legal_ep_square()`), not unconditionally after every double pawn push. Verified across the 6 canonical CPW Perft positions and a 2,000-position fixed-seed differential fuzzer (`tests/test_perft_and_fuzz.py`).
   - **Keyed Zig C++23 Compilation:** Compiled via bundled `ziglang` (`-ffast-math -fno-finite-math-only` to keep `std::isnan`/`std::isinf` guards active), with DLL filename and `.stamp` keyed on the full flag string and `NEURACHESS_ARCH` (`native` default vs `x86-64-v3` portable AVX2+BMI2+FMA baseline with zero AVX-512 instructions).
2. **Cache-Aligned Structure-of-Arrays (SoA) Tree (`fast_tree.cpp`):**
   - Fixed contiguous node arena (`alignas(64)`) storing child slices, prior probabilities, visit counts (`N`), cumulative values (`W`), virtual losses, exact solver tags (`mate_plies`), and edge moves.
   - Single C-ABI `select_leaf_batch` call gathers up to `B=16` unexpanded leaf states and encodes their 18-plane tensors straight into a caller-owned contiguous float32 staging buffer passed to OpenVINO zero-copy (`ov.Tensor(..., shared_memory=True)`).
3. **Production Search Configuration:**
   - **Batch size:** `B=16` (empirical A/B vs `B=32`/`B=64` favored `B=16`; sequential `use_async_overlap=False, async_inflight=1` avoids fixed-node search-quality degradation).
   - **PUCT exploration & FPU:** `c_puct = 1.5`, `use_fpu_reduction = True` (`fpu_reduction = 0.20`, `root_fpu_reduction = 0.10`), `t_root = 1.0`, `contempt = 0.02`.
   - **Canonical 64 MB Zobrist NN Cache:** Stores quantized `(logits, value)` keyed by Zobrist hash inside `CanonicalNNCache` to short-circuit transpositions without redundant NPU calls.
   - **Tree Reuse & Exact Solver:** Preserves valid subtrees after opponent replies while resetting `path_dep_draw` flags and pruning stale path-dependent repetition/50-move tags (`tests/test_pondering.py`, `tests/test_tree_reuse_and_nn_cache.py`). Forced mates propagate bottom-up prioritizing shortest `mate_plies` over longer mates or PUCT visits.

---

## 5. Runtime Auto-Tuner & Core Pinning (`src/neurachess/engine/npu_autotuner.py`)

At startup, `NPUAutoTuner` can probe OpenVINO runtime scheduling, pin worker threads to performance CPU cores (avoiding cross-CCX/E-core jitter on hybrid Intel Core Ultra processors) when HighQoS affinity is enabled, and calibrate batch-16 throughput against device thermal/power states.

---

## 6. UCI Opening Book & Time Management (`src/neurachess/uci/engine_uci.py`)

- **CC0 ECO Opening Book (`assets/opening_book.json`):** Built reproducibly by `src/neurachess/scripts/build_opening_book.py` from `lichess-org/chess-openings` (`a.tsv`–`e.tsv`, **CC0-1.0** public domain, `1,877` EPDs, `<= 16` plies, weighted by ECO frequency support `>= 2`). Enabled by default in UCI via `OwnBook = true` (`false` in `neurachess.scripts.match_runner` unless `--own-book` is passed).
- **Cold-Start Clock Buffer:** When moving out of an uncached or freshly initialized root (`retained_visits < 100`) under `wtime`/`btime` controls, time management applies a `1.30x` search time boost (capped at `0.25 * remaining_time`) to compensate for un-primed tree expansion.
