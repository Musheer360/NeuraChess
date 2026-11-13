"""
NeuraChess SE-ResNet + Conv73 Trainer.

Trains SE-ResNet models with the Conv73 spatial policy head on packed bitboard shards
(`107B` v1 or `691B` v2, with optional `17B` top-K sidecar files for Boltzmann soft targets).
Features GPU-resident chunk interleaving, channels-last AMP FP16, step-driven cosine LR,
dual EMA tracking (`0.9995`, `0.9999`), periodic held-out evaluation, and BN-folded `.npz` export.
"""

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.pytorch_se_resnet import ChessSEResNet
from training.bitboard_dataset import RECORD_SIZE_V1, RECORD_SIZE_V2, unpack_position

_DEVICE_CONST_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_EVAL_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, int, int]] = {}


def _get_device_consts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    if key not in _DEVICE_CONST_CACHE:
        _DEVICE_CONST_CACHE[key] = (
            torch.arange(64, dtype=torch.int64, device=device),
            torch.arange(5, dtype=torch.long, device=device).unsqueeze(0),
        )
    return _DEVICE_CONST_CACHE[key]


class FastModelEMA:
    """Fused multi-tensor EMA shadow parameter tracker."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._fp_shadow: list[torch.Tensor] = []
        self._fp_src: list[torch.Tensor] = []
        self._int_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.reset_from_model(model)

    @staticmethod
    def _raw_model(model: nn.Module) -> nn.Module:
        return model._orig_mod if hasattr(model, "_orig_mod") else model

    def _rebuild_lists(self, model: nn.Module) -> None:
        raw_m = self._raw_model(model)
        live_tensors = {
            **dict(raw_m.named_parameters()),
            **dict(raw_m.named_buffers()),
        }
        self._fp_shadow = []
        self._fp_src = []
        self._int_pairs = []
        for name, src in live_tensors.items():
            if name in self.shadow:
                sh = self.shadow[name]
                if src.is_floating_point():
                    self._fp_shadow.append(sh)
                    self._fp_src.append(src)
                else:
                    self._int_pairs.append((sh, src))

    def reset_from_model(self, model: nn.Module) -> None:
        raw_m = self._raw_model(model)
        self.shadow = {name: t.clone().detach() for name, t in raw_m.state_dict().items()}
        self._rebuild_lists(model)

    def load_shadow_dict(self, state_dict: dict, model: nn.Module, device: torch.device) -> None:
        self.shadow = {k: v.clone().detach().to(device) for k, v in state_dict.items()}
        self._rebuild_lists(model)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            if self._fp_shadow:
                torch._foreach_mul_(self._fp_shadow, self.decay)
                torch._foreach_add_(self._fp_shadow, self._fp_src, alpha=1.0 - self.decay)
            for sh, src in self._int_pairs:
                sh.copy_(src)


def gpu_unpack_v1(raw_batch: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack 107B records into `gpu_unpack` `(B, 18, 8, 8)` tensor, action index, and tanh value."""
    batch_n = raw_batch.shape[0]
    shifts, _ = _get_device_consts(device)

    bbs = raw_batch[:, :96].contiguous().view(torch.int64)
    planes_12 = ((bbs.unsqueeze(-1) >> shifts) & 1).float().reshape(batch_n, 12, 8, 8)

    turn = raw_batch[:, 96].float().view(batch_n, 1, 1, 1).expand(batch_n, 1, 8, 8)
    castling = raw_batch[:, 97].long()
    c_wk = ((castling >> 0) & 1).float().view(batch_n, 1, 1, 1).expand(batch_n, 1, 8, 8)
    c_wq = ((castling >> 1) & 1).float().view(batch_n, 1, 1, 1).expand(batch_n, 1, 8, 8)
    c_bk = ((castling >> 2) & 1).float().view(batch_n, 1, 1, 1).expand(batch_n, 1, 8, 8)
    c_bq = ((castling >> 3) & 1).float().view(batch_n, 1, 1, 1).expand(batch_n, 1, 8, 8)
    halfmove = (raw_batch[:, 99].float() / 100.0).view(batch_n, 1, 1, 1).expand(batch_n, 1, 8, 8)

    inputs = torch.cat([planes_12, turn, c_wk, c_wq, c_bk, c_bq, halfmove], dim=1)

    act = raw_batch[:, 100:102].contiguous().view(torch.int16).squeeze(-1).long()
    act = torch.where(act < 0, act + 65536, act)

    cp = raw_batch[:, 102:104].contiguous().view(torch.int16).squeeze(-1).float()
    v_target = torch.tanh(cp * 0.00287823)
    return inputs, act, v_target


def compute_soft_targets(
    acts: torch.Tensor,
    cp_scores: torch.Tensor,
    sidecar_raw: torch.Tensor,
    device: torch.device,
    temperature: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build top-K Boltzmann target indices and probabilities from best action + sidecar."""
    batch_n = acts.shape[0]
    _, slot_idx = _get_device_consts(device)

    num_extra = sidecar_raw[:, 0].long()
    extra_acts = sidecar_raw[:, 1:9].contiguous().view(torch.int16).view(batch_n, 4).long()
    extra_acts = torch.where(extra_acts < 0, extra_acts + 65536, extra_acts)
    extra_cps = sidecar_raw[:, 9:17].contiguous().view(torch.int16).view(batch_n, 4).float()

    all_acts = torch.cat([acts.unsqueeze(-1), extra_acts], dim=1)
    all_cps = torch.cat([cp_scores.unsqueeze(-1), extra_cps], dim=1)

    mask = slot_idx <= num_extra.unsqueeze(1)
    scaled_cps = (all_cps / (100.0 * temperature)).masked_fill(~mask, -1e9)
    probs = F.softmax(scaled_cps, dim=-1)
    return all_acts, probs


def get_cosine_lr(
    step: int,
    total_steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
) -> float:
    if step < warmup_steps:
        return 1e-6 + (peak_lr - 1e-6) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    clamped = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * clamped))


def discover_shards(shards_dir: Path) -> list[tuple[Path, Path | None, int]]:
    """Find `.bin` shards in `shards_dir` along with optional `.sidecar.bin` and record size."""
    shards: list[tuple[Path, Path | None, int]] = []
    for p in sorted(shards_dir.glob("*.bin")):
        if p.name.endswith(".sidecar.bin") or p.name.endswith(".masks.bin"):
            continue
        sz = p.stat().st_size
        if sz == 0:
            continue
        side_p = p.with_name(p.name.replace(".bin", ".sidecar.bin"))
        side = side_p if side_p.exists() else None
        if side is not None and sz % RECORD_SIZE_V1 == 0 and side.stat().st_size == (sz // RECORD_SIZE_V1) * 17:
            shards.append((p, side, RECORD_SIZE_V1))
        elif sz % RECORD_SIZE_V2 == 0:
            shards.append((p, None, RECORD_SIZE_V2))
        elif sz % RECORD_SIZE_V1 == 0:
            shards.append((p, None, RECORD_SIZE_V1))
    return shards


def load_chunk_tensors(
    chunk: list[tuple[Path, Path | None, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Load a chunk of shards into contiguous CPU/GPU tensors."""
    counts = []
    rec_size = chunk[0][2]
    has_sidecar = all(s is not None for _, s, _ in chunk)
    for main_p, side_p, r_sz in chunk:
        if r_sz != rec_size:
            raise ValueError("Mixed v1/v2 shard formats within one training directory are unsupported.")
        n_rec = main_p.stat().st_size // rec_size
        counts.append(n_rec)

    total_records = sum(counts)
    main_cpu = torch.empty((total_records, rec_size), dtype=torch.uint8)
    side_cpu = torch.empty((total_records, 17), dtype=torch.uint8) if has_sidecar else None

    offset = 0
    for (main_p, side_p, _), n_rec in zip(chunk, counts):
        m_arr = np.fromfile(main_p, dtype=np.uint8, count=n_rec * rec_size).reshape(n_rec, rec_size)
        main_cpu[offset : offset + n_rec].copy_(torch.from_numpy(m_arr))
        if has_sidecar and side_p is not None and side_cpu is not None:
            s_arr = np.fromfile(side_p, dtype=np.uint8, count=n_rec * 17).reshape(n_rec, 17)
            side_cpu[offset : offset + n_rec].copy_(torch.from_numpy(s_arr))
        offset += n_rec

    if device.type == "cuda":
        main_t = main_cpu.to(device, non_blocking=False)
        side_t = side_cpu.to(device, non_blocking=False) if side_cpu is not None else None
        del main_cpu, side_cpu
        return main_t, side_t, total_records

    return main_cpu, side_cpu, total_records


def _get_cached_eval_data(test_path: Path, max_eval: int):
    key = (str(test_path), max_eval)
    if key in _EVAL_CACHE:
        return _EVAL_CACHE[key]

    from core.tensor_movegen import get_legal_action_mask

    f_size = test_path.stat().st_size
    rec_size = RECORD_SIZE_V2 if (f_size % RECORD_SIZE_V2 == 0 and f_size % RECORD_SIZE_V1 != 0) else RECORD_SIZE_V1
    num_pos = min(f_size // rec_size, max_eval)
    raw_arr = np.fromfile(test_path, dtype=np.uint8, count=num_pos * rec_size).reshape(num_pos, rec_size)
    masks_bool = np.empty((num_pos, 4672), dtype=np.bool_)

    if rec_size == RECORD_SIZE_V2:
        packed = raw_arr[:, RECORD_SIZE_V1:RECORD_SIZE_V2]
        masks_bool[:] = np.unpackbits(packed, axis=1)[:, :4672].astype(np.bool_)
    else:
        raw_bytes = raw_arr.tobytes()
        for idx in range(num_pos):
            off = idx * RECORD_SIZE_V1
            rec = unpack_position(raw_bytes[off : off + RECORD_SIZE_V1])
            masks_bool[idx] = get_legal_action_mask(rec["board"]) > 0.5

    _EVAL_CACHE[key] = (raw_arr, masks_bool, num_pos, rec_size)
    return _EVAL_CACHE[key]


def evaluate_dataset(
    model: nn.Module,
    test_path: Path,
    device: torch.device,
    batch_size: int = 512,
    max_eval: int = 50000,
) -> dict[str, float]:
    model.eval()
    raw_arr, masks_bool, num_pos, _ = _get_cached_eval_data(test_path, max_eval)
    amp_enabled = device.type == "cuda"

    total_ce = 0.0
    correct_top1 = 0
    total_val_mse = 0.0
    decisive_count = 0
    sign_correct = 0

    with torch.no_grad():
        for i in range(0, num_pos, batch_size):
            b_sz = min(batch_size, num_pos - i)
            raw_t = torch.from_numpy(raw_arr[i : i + b_sz, :RECORD_SIZE_V1]).to(device)
            inputs, acts, v_targets = gpu_unpack_v1(raw_t, device)
            inputs = inputs.to(memory_format=torch.channels_last)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                logits, v_pred = model(inputs)

            mask_t = torch.from_numpy(masks_bool[i : i + b_sz]).to(device=device)
            masked_logits = logits.float() + (~mask_t).float() * -10000.0

            total_ce += F.cross_entropy(masked_logits, acts, reduction="sum").item()
            correct_top1 += (masked_logits.argmax(dim=-1) == acts).sum().item()

            vp = v_pred.squeeze(-1).float()
            total_val_mse += ((vp - v_targets) ** 2).sum().item()
            dec_mask = v_targets.abs() >= 0.1429
            if dec_mask.any():
                sign_correct += (vp[dec_mask] * v_targets[dec_mask] > 0).sum().item()
                decisive_count += int(dec_mask.sum().item())

    return {
        "masked_ce": round(total_ce / max(1, num_pos), 4),
        "top1_pct": round(100.0 * correct_top1 / max(1, num_pos), 2),
        "val_mse": round(total_val_mse / max(1, num_pos), 4),
        "sign_agreement_pct": round(100.0 * sign_correct / max(1, decisive_count), 2),
    }


def save_checkpoint_and_folded(
    model: nn.Module,
    ema_1: FastModelEMA,
    ema_2: FastModelEMA,
    optimizer: torch.optim.Optimizer,
    step: int,
    out_dir: Path,
    tag: str,
    arch_kwargs: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_m = FastModelEMA._raw_model(model)
    pt_path = out_dir / f"{tag}.pt"
    torch.save(
        {
            "step": step,
            "arch": arch_kwargs,
            "model_state_dict": raw_m.state_dict(),
            "ema_09995": ema_1.shadow,
            "ema_09999": ema_2.shadow,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        pt_path,
    )

    folded_net = ChessSEResNet(
        num_blocks=arch_kwargs["blocks"],
        channels=arch_kwargs["channels"],
        value_dim=1,
        policy_head="conv73",
        use_bn=True,
    )
    folded_net.load_state_dict({k: v.detach().cpu() for k, v in ema_2.shadow.items()})
    folded_net.eval()
    npz_path = out_dir / f"{tag}_ema09999_folded.npz"
    folded_net.export_to_npz(str(npz_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeuraChess SE-ResNet + Conv73 on bitboard shards")
    parser.add_argument("--shards-dir", type=Path, required=True, help="Directory of packed .bin shards")
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "weights" / "runs" / "default")
    parser.add_argument("--eval-data", type=Path, default=None, help="Optional .bin test split for periodic eval")
    parser.add_argument("--init-weights", type=Path, default=None, help="Optional .pt checkpoint for warm start")
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-shards", type=int, default=4, help="Shards loaded simultaneously for mixing")
    parser.add_argument("--peak-lr", type=float, default=2e-3)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--ema-decay-1", type=float, default=0.9995)
    parser.add_argument("--ema-decay-2", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shards = discover_shards(args.shards_dir)
    if not shards:
        raise FileNotFoundError(f"No valid 107B/691B .bin shards found in {args.shards_dir}")

    total_positions_per_epoch = sum(p.stat().st_size // r for p, _, r in shards)
    total_steps = max(1, (total_positions_per_epoch * args.epochs) // args.batch_size)
    print(
        f"[Train] Device={device} | Arch={args.blocks}x{args.channels} | "
        f"Shards={len(shards)} ({total_positions_per_epoch:,} pos/epoch) | TotalSteps={total_steps:,}"
    )

    model = ChessSEResNet(
        num_blocks=args.blocks,
        channels=args.channels,
        value_dim=1,
        policy_head="conv73",
        use_bn=True,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-6, weight_decay=args.weight_decay, fused=(device.type == "cuda")
        )
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=args.weight_decay)

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    ema_1 = FastModelEMA(model, decay=args.ema_decay_1)
    ema_2 = FastModelEMA(model, decay=args.ema_decay_2)

    if args.init_weights is not None:
        ckpt = torch.load(args.init_weights, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt.get("ema_09999", ckpt.get("ema_state_dict", ckpt)))
        model.load_state_dict(sd, strict=False)
        ema_1.reset_from_model(model)
        ema_2.reset_from_model(model)
        print(f"[Train] Loaded initial weights from {args.init_weights}")

    arch_kwargs = {"blocks": args.blocks, "channels": args.channels}
    rng = random.Random(args.seed)
    global_step = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        epoch_shards = list(shards)
        rng.shuffle(epoch_shards)
        for chunk_idx in range(0, len(epoch_shards), args.chunk_shards):
            chunk = epoch_shards[chunk_idx : chunk_idx + args.chunk_shards]
            main_t, side_t, n_rec = load_chunk_tensors(chunk, device)
            perm = torch.randperm(n_rec, device=device)

            for pos in range(0, n_rec - args.batch_size + 1, args.batch_size):
                model.train()
                idx = perm[pos : pos + args.batch_size]
                raw_b = main_t[idx]

                lr = get_cosine_lr(global_step, total_steps, args.peak_lr, args.min_lr, args.warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                inputs, acts, v_targets = gpu_unpack_v1(raw_b[:, :RECORD_SIZE_V1], device)
                inputs = inputs.to(memory_format=torch.channels_last)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    logits, v_pred = model(inputs)
                    logits_f = logits.float()
                    if side_t is not None:
                        cp_raw = raw_b[:, 102:104].contiguous().view(torch.int16).squeeze(-1).float()
                        top_acts, top_probs = compute_soft_targets(
                            acts, cp_raw, side_t[idx], device, temperature=args.temperature
                        )
                        log_probs = F.log_softmax(logits_f, dim=-1)
                        picked = torch.gather(log_probs, dim=-1, index=top_acts)
                        pol_loss = -(top_probs * picked).sum(dim=-1).mean()
                    else:
                        pol_loss = F.cross_entropy(logits_f, acts)
                    val_loss = F.mse_loss(v_pred.squeeze(-1).float(), v_targets)
                    loss = pol_loss + args.value_loss_weight * val_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                ema_1.update(model)
                ema_2.update(model)
                global_step += 1

                if global_step % args.log_every == 0:
                    elapsed = time.time() - t_start
                    pos_s = (global_step * args.batch_size) / max(1e-6, elapsed)
                    print(
                        f"epoch={epoch+1} step={global_step}/{total_steps} lr={lr:.2e} "
                        f"loss={loss.item():.4f} (pol={pol_loss.item():.4f}, val={val_loss.item():.4f}) "
                        f"speed={pos_s:,.0f} pos/s",
                        flush=True,
                    )

                if args.eval_data and global_step % args.eval_every == 0:
                    metrics = evaluate_dataset(model, args.eval_data, device, batch_size=args.batch_size)
                    print(f"[Eval @ step {global_step}] {json.dumps(metrics)}", flush=True)

                if global_step % args.save_every == 0:
                    save_checkpoint_and_folded(
                        model, ema_1, ema_2, optimizer, global_step, args.out_dir, f"step_{global_step}", arch_kwargs
                    )

            del main_t, side_t, perm
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    save_checkpoint_and_folded(
        model, ema_1, ema_2, optimizer, global_step, args.out_dir, f"neurachess_{args.blocks}x{args.channels}_final", arch_kwargs
    )
    print(f"[Train] Complete at step {global_step}. Checkpoints written to {args.out_dir}")


if __name__ == "__main__":
    main()
