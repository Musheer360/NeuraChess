"""Frozen-backbone HL-Gauss-64 Distributional Value Head Trainer ($0 Local NPU/GPU).

Freezes 100% of the 10x192 stem, all 10 SE-ResNet residual blocks, and the
Conv73 policy head from `neurachess-10x192-fp16.npz`, keeping policy outputs
bit-for-bit identical while replacing only the value head:
  `val_conv (192 -> val_ch, 1x1) -> val_fc1 (val_ch*64 -> 256) -> val_fc2 (256 -> 64 bins)`

Trains for 1 global pass over local Stockfish-evaluated shards with leakage-free
board deduplication against `test_far_50k`, `test_in_region_50k_v2`, and puzzles,
then exports OpenVINO FP16 IR with folded linear expectation `E[V] = sum p_i * z_i`.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import time
from pathlib import Path

import chess
import numpy as np
import openvino as ov
import openvino.opset13 as ops
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurachess.models.npu_compiler import DEFAULT_WEIGHTS_NPZ
from neurachess.models.se_resnet_backbone import _build_se_resnet_core, build_openvino_se_resnet
from neurachess.scripts.evaluate import unpack_raw_to_gpu_inputs
from neurachess.training.bitboard_dataset import RECORD_SIZE_V1, RECORD_SIZE_V2, pack_position

CP_TANH_SCALE = math.log(10.0) / 800.0  # 0.0028782313662425573


class HLGaussValueHead(nn.Module):
    """Lightweight HL-Gauss-64 value head operating on frozen `[B, 192, 8, 8]` backbone features."""

    def __init__(self, in_channels: int = 192, val_channels: int = 16, hidden_dim: int = 256, num_bins: int = 64):
        super().__init__()
        self.val_channels = int(val_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_bins = int(num_bins)
        self.val_conv = nn.Conv2d(in_channels, self.val_channels, kernel_size=1, bias=True)
        self.val_fc1 = nn.Linear(self.val_channels * 8 * 8, self.hidden_dim, bias=True)
        self.val_fc2 = nn.Linear(self.hidden_dim, self.num_bins, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.val_conv(feat))
        h = F.relu(self.val_fc1(h.flatten(1)))
        return self.val_fc2(h)


def hl_gauss_centers_and_edges(num_bins: int = 64) -> tuple[np.ndarray, np.ndarray, float]:
    z = np.linspace(-1.0, 1.0, num_bins, dtype=np.float32)
    bin_width = 2.0 / float(num_bins - 1)
    edges = np.linspace(-1.0 - 0.5 * bin_width, 1.0 + 0.5 * bin_width, num_bins + 1, dtype=np.float32)
    sigma = 1.5 * bin_width
    return z, edges, sigma


def compute_hl_gauss_targets(eval_cp: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """Convert raw centipawns (`B,`) to normalized HL-Gauss probabilities (`B, 64`)."""
    v = torch.tanh(eval_cp.float() * CP_TANH_SCALE).clamp(-1.0, 1.0)
    denom = math.sqrt(2.0) * sigma
    cdf = 0.5 * (1.0 + torch.erf((edges.unsqueeze(0) - v.unsqueeze(1)) / denom))
    probs = (cdf[:, 1:] - cdf[:, :-1]).clamp_min(0.0)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _hash98_u64(arr_u8: np.ndarray) -> np.ndarray:
    """Vectorized 64-bit signature of the first 98 canonical board bytes (12 bitboards + turn + castling)."""
    # View first 96 bytes as 12 uint64 words, combine with turn (byte 96) and castling (byte 97)
    words = np.ascontiguousarray(arr_u8[:, :96]).view(np.uint64)
    mult = np.array(
        [
            0x9E3779B97F4A7C15,
            0xBF58476D1CE4E5B9,
            0x94D049BB133111EB,
            0xD6E8FEB86659FD93,
            0xA0761D6478BD642F,
            0xE7037ED1A0B428DB,
            0x8EBC6AF09C88C6E3,
            0x589965CC75374CC3,
            0x1D8E4E27C47D124F,
            0x2545F4914F6CDD1D,
            0x6C62272E07BB0142,
            0xB2B5C735F98D9F1B,
        ],
        dtype=np.uint64,
    )
    with np.errstate(over="ignore"):
        acc = (words * mult).sum(axis=1, dtype=np.uint64)
        acc ^= (arr_u8[:, 96].astype(np.uint64) << np.uint64(1)) | (arr_u8[:, 97].astype(np.uint64) << np.uint64(9))
        acc ^= acc >> np.uint64(30)
        acc *= np.uint64(0xBF58476D1CE4E5B9)
        acc ^= acc >> np.uint64(27)
    return acc


def build_forbidden_board_set(private_root: Path) -> tuple[set[bytes], np.ndarray]:
    """Collect 98-byte canonical board keys from holdout test binaries and puzzle suites."""
    forbidden: set[bytes] = set()
    data_dir = private_root / "data"

    for fname in ("test_far_50k.bin", "test_in_region_50k_v2.bin"):
        p = data_dir / fname
        if p.exists():
            raw = p.read_bytes()
            n = len(raw) // RECORD_SIZE_V1
            arr = np.frombuffer(raw[: n * RECORD_SIZE_V1], dtype=np.uint8).reshape(n, RECORD_SIZE_V1)
            for row in arr[:, :98]:
                forbidden.add(row.tobytes())

    def _add_fen(fen: str, push_first: str | None = None) -> None:
        b = chess.Board(fen)
        forbidden.add(pack_position(b)[:98])
        if push_first:
            b.push_uci(push_first)
            forbidden.add(pack_position(b)[:98])

    strat_p = private_root / "scoreboard" / "puzzles_2000_stratified.json"
    if strat_p.exists():
        for plist in json.loads(strat_p.read_text(encoding="utf-8")).values():
            for p in plist:
                _add_fen(p["fen"])

    mate_p = private_root / "eval_results" / "puzzles_mate_suite.json"
    if mate_p.exists():
        for plist in json.loads(mate_p.read_text(encoding="utf-8")).values():
            if isinstance(plist, list):
                for p in plist:
                    m0 = p["moves"][0] if p.get("moves") else None
                    _add_fen(p["fen"], m0)

    loss_p = private_root / "eval_results" / "step4_loss_anatomy.json"
    if loss_p.exists():
        for entry in json.loads(loss_p.read_text(encoding="utf-8")).get("losses", []):
            _add_fen(entry["blunder_analysis"]["fen"])

    if not forbidden:
        return forbidden, np.empty((0,), dtype=np.uint64)

    f_arr = np.frombuffer(b"".join(forbidden), dtype=np.uint8).reshape(len(forbidden), 98)
    return forbidden, np.unique(_hash98_u64(f_arr))


def load_filtered_training_records(private_root: Path) -> tuple[np.ndarray, dict]:
    """Load all 5 Stockfish shards cleanly into uint8[N, 107] while removing any holdout overlap."""
    t0 = time.perf_counter()
    forbidden_98, forbidden_h64 = build_forbidden_board_set(private_root)
    data_dir = private_root / "data"
    shard_specs = [
        (data_dir / "v2_shards_1m" / "shard_000.bin", RECORD_SIZE_V2),
        (data_dir / "v2_shards_1m" / "shard_001.bin", RECORD_SIZE_V2),
        (data_dir / "sample_shard_000.bin", RECORD_SIZE_V1),
        (data_dir / "sample_shard_040.bin", RECORD_SIZE_V1),
        (data_dir / "train_300k.bin", RECORD_SIZE_V1),
    ]

    kept_chunks: list[np.ndarray] = []
    per_file_stats = []
    total_raw = 0
    total_excluded = 0

    for sp, stride in shard_specs:
        if not sp.exists():
            raise FileNotFoundError(f"Missing required training shard: {sp}")
        raw_bytes = sp.read_bytes()
        n = len(raw_bytes) // stride
        total_raw += n
        view = np.frombuffer(raw_bytes[: n * stride], dtype=np.uint8).reshape(n, stride)[:, :RECORD_SIZE_V1].copy()
        hashes = _hash98_u64(view[:, :98])
        cand_mask = np.isin(hashes, forbidden_h64)
        excluded = 0
        if np.any(cand_mask):
            keep_flags = np.ones(n, dtype=np.bool_)
            for idx in np.nonzero(cand_mask)[0]:
                if view[idx, :98].tobytes() in forbidden_98:
                    keep_flags[idx] = False
                    excluded += 1
            if excluded > 0:
                view = view[keep_flags]
        total_excluded += excluded
        kept_chunks.append(view)
        per_file_stats.append({"file": str(sp.name), "stride": stride, "raw": n, "excluded": excluded, "kept": len(view)})

    all_records = np.concatenate(kept_chunks, axis=0)
    return all_records, {
        "forbidden_unique_boards": len(forbidden_98),
        "total_raw_records": total_raw,
        "total_excluded_overlap": total_excluded,
        "total_training_records": int(len(all_records)),
        "load_filter_s": round(time.perf_counter() - t0, 2),
        "shards": per_file_stats,
    }


def _extract_features_on_queue(q: ov.AsyncInferQueue, compiled: ov.CompiledModel, x_np: np.ndarray, bs: int) -> np.ndarray:
    n = x_np.shape[0]
    out = np.empty((n, 192, 8, 8), dtype=np.float32)
    in_name = compiled.input(0).get_any_name()
    n_batches = (n + bs - 1) // bs

    def _cb(req, userdata):
        s, k = userdata
        out[s : s + k] = req.get_output_tensor(0).data[:k]

    q.set_callback(_cb)
    for bi in range(n_batches):
        s = bi * bs
        k = min(bs, n - s)
        if k == bs:
            buf = x_np[s : s + k]
        else:
            buf = np.zeros((bs, 18, 8, 8), dtype=np.float32)
            buf[:k] = x_np[s : s + k]
        q.start_async({in_name: buf}, userdata=(s, k))
    q.wait_all()
    return out


class _FrozenExtractor:
    """Extracts frozen `[N, 192, 8, 8]` features concurrently on Intel NPU + Arc GPU."""

    def __init__(self, weights_dict: dict[str, np.ndarray]):
        core = ov.Core()
        core.set_property(
            "NPU",
            {
                "PERFORMANCE_HINT": ov.properties.hint.PerformanceMode.THROUGHPUT,
                "NPU_TURBO": True,
                "INFERENCE_PRECISION_HINT": ov.Type.f16,
            },
        )
        core.set_property(
            "GPU",
            {
                "PERFORMANCE_HINT": ov.properties.hint.PerformanceMode.THROUGHPUT,
                "INFERENCE_PRECISION_HINT": ov.Type.f16,
            },
        )

        inp_n, feat_n, _, _, _, _, _ = _build_se_resnet_core(
            batch_size=16, num_blocks=10, channels=192, weights_dict=weights_dict
        )
        m_npu = ov.Model([ops.result(feat_n)], [inp_n], "fb_npu_16")
        self.c_npu = core.compile_model(m_npu, "NPU")
        self.nireq_npu = max(4, int(self.c_npu.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
        self.q_npu = ov.AsyncInferQueue(self.c_npu, self.nireq_npu)

        self.has_gpu = "GPU" in core.available_devices
        if self.has_gpu:
            inp_g, feat_g, _, _, _, _, _ = _build_se_resnet_core(
                batch_size=64, num_blocks=10, channels=192, weights_dict=weights_dict
            )
            m_gpu = ov.Model([ops.result(feat_g)], [inp_g], "fb_gpu_64")
            self.c_gpu = core.compile_model(m_gpu, "GPU")
            self.nireq_gpu = max(4, int(self.c_gpu.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")))
            self.q_gpu = ov.AsyncInferQueue(self.c_gpu, self.nireq_gpu)

    def extract_batch(self, x_np: np.ndarray) -> np.ndarray:
        n = x_np.shape[0]
        if not self.has_gpu or n < 512:
            return _extract_features_on_queue(self.q_npu, self.c_npu, x_np, 16)
        split = int(round(n * 0.53))
        split = max(16, (split // 16) * 16)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_npu = pool.submit(_extract_features_on_queue, self.q_npu, self.c_npu, x_np[:split], 16)
            f_gpu = pool.submit(_extract_features_on_queue, self.q_gpu, self.c_gpu, x_np[split:], 64)
            return np.concatenate([f_npu.result(), f_gpu.result()], axis=0)


def train_frozen_hl_gauss_64(
    private_root: Path,
    out_npz_path: Path,
    out_xml_path: Path,
    val_channels: int = 16,
    batch_size: int = 256,
    chunk_size: int = 16384,
    lr: float = 1e-3,
    min_lr: float = 1e-5,
    weight_decay: float = 1e-4,
    max_records: int | None = None,
    seed: int = 42,
) -> dict:
    """Train only the HL-Gauss-64 value head for 1 shuffled pass over local shards and export FP16 OpenVINO IR."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    with np.load(DEFAULT_WEIGHTS_NPZ) as npz:
        base_weights = {k: npz[k] for k in npz.files}

    print("[HL-Gauss-64] Loading and leakage-filtering local training shards...", flush=True)
    records_u8, data_stats = load_filtered_training_records(private_root)
    if max_records is not None and len(records_u8) > max_records:
        perm_sub = rng.permutation(len(records_u8))[:max_records]
        records_u8 = records_u8[perm_sub]
    else:
        records_u8 = records_u8[rng.permutation(len(records_u8))]

    n_train = len(records_u8)
    print(
        f"[HL-Gauss-64] Shuffled train set: {n_train:,} positions "
        f"(excluded {data_stats['total_excluded_overlap']} overlap records in {data_stats['load_filter_s']}s)",
        flush=True,
    )

    # Disjoint validation monitor (`val_100k.bin` first 20k)
    val_path = private_root / "data" / "val_100k.bin"
    val_raw = val_path.read_bytes()[: 20000 * RECORD_SIZE_V1]
    val_u8 = np.frombuffer(val_raw, dtype=np.uint8).reshape(20000, RECORD_SIZE_V1)
    val_x = unpack_raw_to_gpu_inputs(val_raw, 20000).numpy().astype(np.float32)
    val_cps = val_u8[:, 102:104].copy().view(np.int16).reshape(-1)
    val_v_targets = np.tanh(val_cps.astype(np.float64) * CP_TANH_SCALE)

    extractor = _FrozenExtractor(base_weights)
    print("[HL-Gauss-64] Extracting disjoint val_20k frozen features...", flush=True)
    val_feats = extractor.extract_batch(val_x)

    z_np, edges_np, sigma = hl_gauss_centers_and_edges(64)
    z_t = torch.from_numpy(z_np)
    edges_t = torch.from_numpy(edges_np)

    head = HLGaussValueHead(in_channels=192, val_channels=val_channels, hidden_dim=256, num_bins=64)
    if val_channels == 16 and "val_conv_w" in base_weights:
        head.val_conv.weight.data.copy_(torch.from_numpy(base_weights["val_conv_w"].astype(np.float32)))
        nn.init.zeros_(head.val_conv.bias)

    total_steps = max(1, (n_train + batch_size - 1) // batch_size)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

    def _eval_val() -> tuple[float, float, float]:
        head.eval()
        with torch.no_grad():
            feats_t = torch.from_numpy(val_feats)
            logits = head(feats_t)
            t_probs = compute_hl_gauss_targets(torch.from_numpy(val_cps), edges_t, sigma)
            hl_ce = float(-(t_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean().item())
            e_v = (F.softmax(logits, dim=-1) @ z_t).numpy().astype(np.float64)
            mse = float(np.mean((e_v - val_v_targets) ** 2))
            dec = np.abs(val_cps) >= 50
            sign_acc = float(100.0 * np.mean(((val_cps[dec] > 0) & (e_v[dec] > 0)) | ((val_cps[dec] < 0) & (e_v[dec] < 0))))
        head.train()
        return hl_ce, mse, sign_acc

    init_ce, init_mse, init_sign = _eval_val()
    print(f"[HL-Gauss-64] Init Val20k: HL-CE={init_ce:.4f} | MSE={init_mse:.5f} | DecSign={init_sign:.2f}%", flush=True)

    t_train0 = time.perf_counter()
    running_ce = 0.0
    running_n = 0
    history = []

    for c_idx, start in enumerate(range(0, n_train, chunk_size)):
        end = min(n_train, start + chunk_size)
        k = end - start
        chunk_u8 = records_u8[start:end]
        x_np = unpack_raw_to_gpu_inputs(chunk_u8.tobytes(), k).numpy().astype(np.float32)
        cps_np = chunk_u8[:, 102:104].copy().view(np.int16).reshape(-1)
        feats_np = extractor.extract_batch(x_np)

        perm_c = torch.randperm(k)
        feats_t = torch.from_numpy(feats_np)[perm_c]
        cps_t = torch.from_numpy(cps_np)[perm_c]
        targets_t = compute_hl_gauss_targets(cps_t, edges_t, sigma)

        head.train()
        for b0 in range(0, k, batch_size):
            b1 = min(k, b0 + batch_size)
            logits = head(feats_t[b0:b1])
            loss = -(targets_t[b0:b1] * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_ce += float(loss.item()) * (b1 - b0)
            running_n += b1 - b0

        if (c_idx + 1) % 20 == 0 or end == n_train:
            v_ce, v_mse, v_sign = _eval_val()
            elapsed = time.perf_counter() - t_train0
            avg_ce = running_ce / max(1, running_n)
            row = {
                "seen": end,
                "elapsed_s": round(elapsed, 1),
                "pos_per_s": round(end / max(1e-6, elapsed), 0),
                "train_hl_ce": round(avg_ce, 4),
                "val20k_hl_ce": round(v_ce, 4),
                "val20k_mse": round(v_mse, 5),
                "val20k_dec_sign_pct": round(v_sign, 2),
                "lr": round(scheduler.get_last_lr()[0], 7),
            }
            history.append(row)
            print(
                f"[HL-Gauss-64] {end:7d}/{n_train} ({100.0*end/n_train:5.1f}%) | "
                f"{row['pos_per_s']:.0f} pos/s | tr_CE={avg_ce:.4f} | val_CE={v_ce:.4f} "
                f"val_MSE={v_mse:.5f} val_Sign={v_sign:.2f}% | lr={row['lr']:.2e}",
                flush=True,
            )
            running_ce = 0.0
            running_n = 0

    train_s = round(time.perf_counter() - t_train0, 2)
    del extractor
    import gc
    gc.collect()

    # Export updated weights (.npz + OpenVINO FP16 .xml with untouched stem/blocks/pol73!)
    new_weights = dict(base_weights)
    new_weights["val_conv_w"] = head.val_conv.weight.detach().cpu().numpy().astype(np.float32)
    new_weights["val_conv_b"] = head.val_conv.bias.detach().cpu().numpy().astype(np.float32)
    new_weights["val_dense1_w"] = head.val_fc1.weight.detach().cpu().numpy().astype(np.float32)
    new_weights["val_dense1_b"] = head.val_fc1.bias.detach().cpu().numpy().astype(np.float32)
    new_weights["val_dense2_w"] = head.val_fc2.weight.detach().cpu().numpy().astype(np.float32)
    new_weights["val_dense2_b"] = head.val_fc2.bias.detach().cpu().numpy().astype(np.float32)
    new_weights["val_hl_z"] = z_np.astype(np.float32)

    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz_path, **new_weights)

    ov_model = build_openvino_se_resnet(
        batch_size=-1,
        num_blocks=10,
        channels=192,
        in_channels=18,
        policy_head="conv73",
        weights_dict=new_weights,
        model_name="NeuraChess_10x192_HLGauss64",
    )
    out_xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(out_xml_path), compress_to_fp16=True)

    return {
        "val_channels": val_channels,
        "hidden_dim": 256,
        "num_bins": 64,
        "sigma": round(sigma, 6),
        "train_seconds": train_s,
        "throughput_pos_per_s": round(n_train / max(1e-6, train_s), 1),
        "data_stats": data_stats,
        "val20k_init": {"hl_ce": round(init_ce, 4), "mse": round(init_mse, 5), "dec_sign_pct": round(init_sign, 2)},
        "val20k_final": history[-1] if history else {},
        "history": history,
        "out_npz": str(out_npz_path),
        "out_xml": str(out_xml_path),
    }
