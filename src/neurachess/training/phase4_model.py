"""Phase 4 Neural Architectures, Middle-Block W8A8 Fake-Quantization, Depth Expansion & Export.

Supports both Cloud Spot jobs without duplicating backbone logic:
  - Job A (`10x192` + `HL-Gauss-64` + `MLH`):
      * `ckpt_A1_frozen_trunk`: 100% frozen 346M stem/blocks/Conv73 policy; trains HL-Gauss-64 + MLH.
      * `ckpt_A2_joint_middle_qat`: Folds trunk BatchNorms cleanly and applies CUDA-native W8A8 ConvOnly
        fake-quantization on middle residual blocks `2..7`, keeping blocks `0,1,8,9`, SE gates,
        residual skips, Policy head, HL-Gauss-64, and MLH in FP16/BF16.
  - Job B (`12x192_mid2_9_hl64` + `MLH`):
      * Function-preserving depth expansion from 346M `10x192 (ema09999)`:
        copies stem, Conv73 policy, `val_conv` (16ch), and blocks `0..9` -> `[0,1,2,3,4,6,7,8,9,10]`,
        initializes inserted blocks `5` and `11` (`conv1` Kaiming, `conv2=0`, `bn2=I`) so `x + SE(0) = x`,
        and verifies at Step 0 that policy logits match 346M `10x192` within `< 1e-6`.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import openvino as ov
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurachess.models.pytorch_se_resnet import ChessSEResNet, SEBlock, SEResidualBlock, fold_conv_bn
from neurachess.models.se_resnet_backbone import build_openvino_se_resnet
from neurachess.training.train_hl_gauss_value import (
    CP_TANH_SCALE,
    HLGaussValueHead,
    compute_hl_gauss_targets,
    hl_gauss_centers_and_edges,
)

AUTHOR_META = "Musheer Alam (https://github.com/Musheer360/NeuraChess)"
LICENSE_META = "GPL-3.0"


class MovesLeftHead(nn.Module):
    """Moves-Left auxiliary regression head matching `hl_gauss_mlh` OpenVINO spec."""

    def __init__(self, in_channels: int = 192, mlh_channels: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.mlh_conv = nn.Conv2d(in_channels, mlh_channels, kernel_size=1, bias=True)
        self.mlh_d1 = nn.Linear(mlh_channels * 8 * 8, hidden_dim, bias=True)
        self.mlh_d2 = nn.Linear(hidden_dim, 1, bias=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        m = F.relu(self.mlh_conv(feat))
        m = F.relu(self.mlh_d1(m.flatten(1)))
        return F.relu(self.mlh_d2(m))


class W8A8FakeQuantConv2d(nn.Module):
    """Zero-sync CUDA-native W8A8 Conv2d fake-quant wrapper (per-channel symmetric W, EMA uint8 A)."""

    def __init__(self, conv: nn.Conv2d, act_momentum: float = 0.05, unsigned_act: bool = True):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.act_momentum = float(act_momentum)
        self.unsigned_act = bool(unsigned_act)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        self.register_buffer("act_max", torch.tensor(6.0, dtype=torch.float32, device=conv.weight.device))
        self.register_buffer("act_initialized", torch.tensor(0, dtype=torch.uint8, device=conv.weight.device))

    def calibrate(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            cur_max = x.detach().float().abs().amax().clamp(min=1e-5)
            ema_val = self.act_max * (1.0 - self.act_momentum) + cur_max * self.act_momentum
            self.act_max.copy_(torch.where(self.act_initialized == 0, cur_max, ema_val))
            self.act_initialized.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.calibrate(x)
        amax = self.act_max.clamp(min=1e-5).to(x.dtype)
        if self.unsigned_act:
            s_a = amax / 255.0
            x_q = torch.round(x / s_a).clamp(0.0, 255.0) * s_a
        else:
            s_a = amax / 127.0
            x_q = torch.round(x / s_a).clamp(-127.0, 127.0) * s_a
        x_fq = x + (x_q - x).detach()

        # Per-output-channel symmetric int8 weights [-127, 127]
        w = self.weight
        s_w = (w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6) / 127.0).to(w.dtype)
        w_q = torch.round(w / s_w).clamp(-127.0, 127.0) * s_w
        w_fq = w + (w_q - w).detach()

        return F.conv2d(
            x_fq,
            w_fq,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def to_plain_conv2d(self) -> nn.Conv2d:
        plain = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=(self.bias is not None),
        ).to(device=self.weight.device, dtype=self.weight.dtype)
        with torch.no_grad():
            plain.weight.copy_(self.weight)
            if self.bias is not None and plain.bias is not None:
                plain.bias.copy_(self.bias)
        return plain


class ChessSEResNetPhase4(nn.Module):
    """SE-ResNet with Conv73 policy head, HL-Gauss-64 value head, and Moves-Left auxiliary head."""

    def __init__(
        self,
        num_blocks: int = 10,
        channels: int = 192,
        se_reduction: int = 4,
        use_bn: bool = True,
        val_channels: int = 16,
        val_hidden: int = 256,
        num_bins: int = 64,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.num_blocks = int(num_blocks)
        self.channels = int(channels)
        self.se_reduction = int(se_reduction)
        self.use_bn = bool(use_bn)
        self.conv_bias = bool(conv_bias)
        self.num_bins = int(num_bins)

        folded_bias = (not self.use_bn) and self.conv_bias
        self.init_conv = nn.Conv2d(18, self.channels, kernel_size=3, padding=1, bias=folded_bias)
        self.init_bn = nn.BatchNorm2d(self.channels) if self.use_bn else nn.Identity()

        self.res_blocks = nn.ModuleList(
            [SEResidualBlock(self.channels, reduction=self.se_reduction, use_bn=self.use_bn) for _ in range(self.num_blocks)]
        )
        if folded_bias:
            for blk in self.res_blocks:
                blk.conv1 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=True)
                blk.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=True)

        self.pol_conv = nn.Conv2d(self.channels, 73, kernel_size=1, bias=not self.use_bn)
        self.pol_bn = nn.BatchNorm2d(73) if self.use_bn else nn.Identity()

        self.hl_head = HLGaussValueHead(
            in_channels=self.channels,
            val_channels=val_channels,
            hidden_dim=val_hidden,
            num_bins=num_bins,
        )
        self.ml_head = MovesLeftHead(in_channels=self.channels, mlh_channels=8, hidden_dim=64)

        z_np, edges_np, sigma = hl_gauss_centers_and_edges(num_bins)
        self.hl_sigma = float(sigma)
        self.register_buffer("hl_z", torch.from_numpy(z_np))
        self.register_buffer("hl_edges", torch.from_numpy(edges_np))

    def forward_trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.init_bn(self.init_conv(x)))
        for block in self.res_blocks:
            h = block(h)
        return h

    def forward_policy_and_heads(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.pol_bn(self.pol_conv(feat))
        policy_logits = p.permute(0, 2, 3, 1).contiguous().reshape(feat.shape[0], 4672)
        hl_logits = self.hl_head(feat)
        val_exp = torch.matmul(F.softmax(hl_logits.float(), dim=-1), self.hl_z.float()).unsqueeze(-1)
        mlh_pred = self.ml_head(feat)
        return policy_logits, hl_logits, val_exp, mlh_pred

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.forward_trunk(x)
        return self.forward_policy_and_heads(feat)

    def forward_eval_only(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility helper returning `(policy_logits, val_expectation)` for standard 2-output evals."""
        pol, _, val_exp, _ = self.forward(x)
        return pol, val_exp


def _fold_conv_bn_inplace(conv: nn.Conv2d, bn: nn.Module) -> nn.Conv2d:
    w_f, b_f = fold_conv_bn(conv, bn)
    folded = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        folded.weight.copy_(torch.from_numpy(w_f).to(device=conv.weight.device, dtype=conv.weight.dtype))
        if b_f is not None:
            folded.bias.copy_(torch.from_numpy(b_f).to(device=conv.weight.device, dtype=conv.weight.dtype))
        else:
            folded.bias.zero_()
    return folded


def apply_middle_w8a8_qat(model: ChessSEResNetPhase4, qat_blocks: tuple[int, ...] = (2, 3, 4, 5, 6, 7)) -> ChessSEResNetPhase4:
    """Fold trunk BNs in middle blocks cleanly and replace `conv1`/`conv2` in `qat_blocks` with W8A8 FakeQuant."""
    model.eval()
    for b_idx in qat_blocks:
        blk = model.res_blocks[b_idx]
        if isinstance(blk.bn1, nn.BatchNorm2d):
            blk.conv1 = _fold_conv_bn_inplace(blk.conv1, blk.bn1)
            blk.bn1 = nn.Identity()
        if isinstance(blk.bn2, nn.BatchNorm2d):
            blk.conv2 = _fold_conv_bn_inplace(blk.conv2, blk.bn2)
            blk.bn2 = nn.Identity()
        if not isinstance(blk.conv1, W8A8FakeQuantConv2d):
            blk.conv1 = W8A8FakeQuantConv2d(blk.conv1)
        if not isinstance(blk.conv2, W8A8FakeQuantConv2d):
            blk.conv2 = W8A8FakeQuantConv2d(blk.conv2)

    # Freeze remaining trunk BatchNorms so running stats do not drift at low QAT LR
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)
    return model


def enforce_frozen_bn_eval(model: nn.Module) -> None:
    """Keep BatchNorm2d layers with frozen parameters strictly in eval() mode."""
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    for m in raw.modules():
        if isinstance(m, nn.BatchNorm2d):
            has_trainable = any(p.requires_grad for p in m.parameters())
            if not has_trainable:
                m.eval()


def extract_10x192_state_dict(ckpt_obj: dict) -> dict[str, torch.Tensor]:
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj
    for k in ("ema_09999_state_dict", "ema_state_dict", "ema_09995_state_dict", "model_state_dict", "state_dict"):
        if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
            sd = ckpt_obj[k]
            return {kk.replace("_orig_mod.", ""): vv for kk, vv in sd.items()}
    return {kk.replace("_orig_mod.", ""): vv for kk, vv in ckpt_obj.items() if isinstance(vv, torch.Tensor)}


def load_346m_into_10x192_phase4(model: ChessSEResNetPhase4, ckpt_path: str | Path) -> dict[str, torch.Tensor]:
    """Load 346M 10x192 stem, 10 res blocks, Conv73 policy, and `val_conv` (16ch) into `ChessSEResNetPhase4`."""
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    src_sd = extract_10x192_state_dict(raw)
    dst_sd = model.state_dict()
    copied = []
    for k, dst_t in dst_sd.items():
        if k.startswith(("init_conv.", "init_bn.", "res_blocks.", "pol_conv.", "pol_bn.")) and k in src_sd:
            dst_t.copy_(src_sd[k].to(dtype=dst_t.dtype))
            copied.append(k)
    if "val_conv.weight" in src_sd:
        model.hl_head.val_conv.weight.data.copy_(src_sd["val_conv.weight"].to(dtype=model.hl_head.val_conv.weight.dtype))
        if "val_conv.bias" in src_sd:
            model.hl_head.val_conv.bias.data.copy_(src_sd["val_conv.bias"].to(dtype=model.hl_head.val_conv.bias.dtype))
        else:
            model.hl_head.val_conv.bias.data.zero_()
    return src_sd


def load_folded_npz_into_phase4(model: ChessSEResNetPhase4, npz_path: str | Path) -> None:
    """Load BN-folded `.npz` (e.g., `A2_soup75.npz`) into `ChessSEResNetPhase4(..., use_bn=False, conv_bias=True)`."""
    with np.load(str(npz_path)) as z, torch.no_grad():
        model.init_conv.weight.copy_(torch.from_numpy(z["init_conv_w"]).to(dtype=model.init_conv.weight.dtype))
        if model.init_conv.bias is not None and "init_conv_b" in z:
            model.init_conv.bias.copy_(torch.from_numpy(z["init_conv_b"]).to(dtype=model.init_conv.bias.dtype))
        for i, blk in enumerate(model.res_blocks):
            blk.conv1.weight.copy_(torch.from_numpy(z[f"res_{i}_conv1_w"]).to(dtype=blk.conv1.weight.dtype))
            if blk.conv1.bias is not None and f"res_{i}_conv1_b" in z:
                blk.conv1.bias.copy_(torch.from_numpy(z[f"res_{i}_conv1_b"]).to(dtype=blk.conv1.bias.dtype))
            blk.conv2.weight.copy_(torch.from_numpy(z[f"res_{i}_conv2_w"]).to(dtype=blk.conv2.weight.dtype))
            if blk.conv2.bias is not None and f"res_{i}_conv2_b" in z:
                blk.conv2.bias.copy_(torch.from_numpy(z[f"res_{i}_conv2_b"]).to(dtype=blk.conv2.bias.dtype))
            blk.se.squeeze.weight.copy_(torch.from_numpy(z[f"res_{i}_se_sq_w"]).to(dtype=blk.se.squeeze.weight.dtype))
            blk.se.excite.weight.copy_(torch.from_numpy(z[f"res_{i}_se_ex_w"]).to(dtype=blk.se.excite.weight.dtype))
        model.pol_conv.weight.copy_(torch.from_numpy(z["pol_73_w"]).to(dtype=model.pol_conv.weight.dtype))
        if model.pol_conv.bias is not None and "pol_73_b" in z:
            model.pol_conv.bias.copy_(torch.from_numpy(z["pol_73_b"]).to(dtype=model.pol_conv.bias.dtype))
        if "val_conv_w" in z:
            model.hl_head.val_conv.weight.copy_(torch.from_numpy(z["val_conv_w"]).to(dtype=model.hl_head.val_conv.weight.dtype))
            model.hl_head.val_conv.bias.copy_(torch.from_numpy(z["val_conv_b"]).to(dtype=model.hl_head.val_conv.bias.dtype))
            model.hl_head.val_fc1.weight.copy_(torch.from_numpy(z["val_dense1_w"]).to(dtype=model.hl_head.val_fc1.weight.dtype))
            model.hl_head.val_fc1.bias.copy_(torch.from_numpy(z["val_dense1_b"]).to(dtype=model.hl_head.val_fc1.bias.dtype))
            model.hl_head.val_fc2.weight.copy_(torch.from_numpy(z["val_dense2_w"]).to(dtype=model.hl_head.val_fc2.weight.dtype))
            model.hl_head.val_fc2.bias.copy_(torch.from_numpy(z["val_dense2_b"]).to(dtype=model.hl_head.val_fc2.bias.dtype))
        if "mlh_conv_w" in z:
            model.ml_head.mlh_conv.weight.copy_(torch.from_numpy(z["mlh_conv_w"]).to(dtype=model.ml_head.mlh_conv.weight.dtype))
            model.ml_head.mlh_conv.bias.copy_(torch.from_numpy(z["mlh_conv_b"]).to(dtype=model.ml_head.mlh_conv.bias.dtype))
            model.ml_head.mlh_d1.weight.copy_(torch.from_numpy(z["mlh_d1_w"]).to(dtype=model.ml_head.mlh_d1.weight.dtype))
            model.ml_head.mlh_d1.bias.copy_(torch.from_numpy(z["mlh_d1_b"]).to(dtype=model.ml_head.mlh_d1.bias.dtype))
            model.ml_head.mlh_d2.weight.copy_(torch.from_numpy(z["mlh_d2_w"]).to(dtype=model.ml_head.mlh_d2.weight.dtype))
            model.ml_head.mlh_d2.bias.copy_(torch.from_numpy(z["mlh_d2_b"]).to(dtype=model.ml_head.mlh_d2.bias.dtype))


def transplant_hl_mlh_heads(dst_model: ChessSEResNetPhase4, src_ckpt_or_sd: str | Path | dict[str, torch.Tensor]) -> int:
    """Transplant `hl_head.*` and `ml_head.*` tensors from a matching-channel checkpoint into `dst_model`."""
    if isinstance(src_ckpt_or_sd, (str, Path)):
        raw = torch.load(str(src_ckpt_or_sd), map_location="cpu", weights_only=False)
        src_sd = extract_10x192_state_dict(raw)
    else:
        src_sd = extract_10x192_state_dict(src_ckpt_or_sd)
    copied = 0
    with torch.no_grad():
        for k, dst_t in dst_model.state_dict().items():
            if k.startswith(("hl_head.", "ml_head.")) and k in src_sd:
                if tuple(src_sd[k].shape) != tuple(dst_t.shape):
                    raise ValueError(f"Head shape mismatch for {k}: src={tuple(src_sd[k].shape)} vs dst={tuple(dst_t.shape)}")
                dst_t.copy_(src_sd[k].to(device=dst_t.device, dtype=dst_t.dtype))
                copied += 1
    if copied == 0:
        raise RuntimeError("No hl_head.*/ml_head.* tensors found in source checkpoint for transplant")
    return copied


def expand_10x192_to_12x192_identity(
    ckpt_10x192_path_or_sd: str | Path | dict[str, torch.Tensor],
    seed: int = 42,
    verify_tolerance: float = 1e-6,
) -> tuple[ChessSEResNetPhase4, float]:
    """Function-preserving 10x192 -> 12x192 depth expansion with identity blocks at indices 5 & 11.

    Block mapping:
      10x192 blocks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -> 12x192 blocks [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
      12x192 blocks [5, 11]: identity (`conv1` Kaiming-normal, `conv2` weights & bias = 0 => x + SE(0) = x)
    Verifies on Step 0 that policy logits match 346M 10x192 within `max_diff < verify_tolerance`.
    """
    if isinstance(ckpt_10x192_path_or_sd, (str, Path)):
        raw = torch.load(str(ckpt_10x192_path_or_sd), map_location="cpu", weights_only=False)
        src_sd = extract_10x192_state_dict(raw)
    else:
        src_sd = extract_10x192_state_dict(ckpt_10x192_path_or_sd)

    m12 = ChessSEResNetPhase4(num_blocks=12, channels=192, use_bn=True)
    block_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}

    with torch.no_grad():
        for k, dst_t in m12.state_dict().items():
            if k.startswith(("init_conv.", "init_bn.", "pol_conv.", "pol_bn.")) and k in src_sd:
                dst_t.copy_(src_sd[k].to(dtype=dst_t.dtype))

        for src_i, dst_i in block_map.items():
            pref_s = f"res_blocks.{src_i}."
            src_blk = {k[len(pref_s) :]: v for k, v in src_sd.items() if k.startswith(pref_s)}
            m12.res_blocks[dst_i].load_state_dict(src_blk, strict=True)

        gen = torch.Generator(device="cpu").manual_seed(seed)
        for new_i in (5, 11):
            blk = m12.res_blocks[new_i]
            nn.init.kaiming_normal_(blk.conv1.weight, mode="fan_out", nonlinearity="relu", generator=gen)
            if blk.conv1.bias is not None:
                blk.conv1.bias.zero_()
            if isinstance(blk.bn1, nn.BatchNorm2d):
                blk.bn1.weight.fill_(1.0)
                blk.bn1.bias.zero_()
                blk.bn1.running_mean.zero_()
                blk.bn1.running_var.fill_(1.0)
            blk.conv2.weight.zero_()
            if blk.conv2.bias is not None:
                blk.conv2.bias.zero_()
            if isinstance(blk.bn2, nn.BatchNorm2d):
                blk.bn2.weight.fill_(1.0)
                blk.bn2.bias.zero_()
                blk.bn2.running_mean.zero_()
                blk.bn2.running_var.fill_(1.0)

        if "val_conv.weight" in src_sd:
            m12.hl_head.val_conv.weight.copy_(src_sd["val_conv.weight"])
            if "val_conv.bias" in src_sd:
                m12.hl_head.val_conv.bias.copy_(src_sd["val_conv.bias"])
            else:
                m12.hl_head.val_conv.bias.zero_()

        if "val_dense1.weight" in src_sd:
            w_d1 = src_sd["val_dense1.weight"]
            rows = min(m12.hl_head.val_fc1.weight.shape[0], w_d1.shape[0])
            cols = min(m12.hl_head.val_fc1.weight.shape[1], w_d1.shape[1])
            m12.hl_head.val_fc1.weight[:rows, :cols].copy_(w_d1[:rows, :cols])
            if m12.hl_head.val_fc1.bias is not None:
                m12.hl_head.val_fc1.bias[:rows].zero_()

    # Step 0 function-preservation verification against exact 10x192 reference
    m10 = ChessSEResNet(num_blocks=10, channels=192, value_dim=1, policy_head="conv73", use_bn=True)
    m10.load_state_dict(src_sd, strict=True)
    m10.eval()
    m12.eval()
    rng_t = torch.Generator(device="cpu").manual_seed(seed + 101)
    x_test = (torch.rand((8, 18, 8, 8), generator=rng_t) > 0.7).float()
    x_test[:, 12:18] = torch.rand((8, 6, 8, 8), generator=rng_t)
    with torch.no_grad():
        pol10, _ = m10(x_test)
        pol12, _, _, _ = m12(x_test)
        max_diff = float((pol10 - pol12).abs().max().item())
    if not (max_diff < verify_tolerance):
        raise RuntimeError(
            f"10x192 -> 12x192 Step-0 policy preservation FAILED: max_diff={max_diff:.3e} >= {verify_tolerance}"
        )
    return m12, max_diff


def unwrap_qat_conv(conv: nn.Module) -> nn.Conv2d:
    return conv.to_plain_conv2d() if isinstance(conv, W8A8FakeQuantConv2d) else conv


def export_phase4_to_folded_npz_and_ov(
    model: ChessSEResNetPhase4,
    out_dir: str | Path,
    prefix: str,
    calib_inputs: list[np.ndarray] | None = None,
    int8_blocks: tuple[int, ...] | None = None,
) -> dict[str, str]:
    """Fold BatchNorms cleanly into `.npz`, save FP16 OpenVINO IR, and optionally export Middle-INT8 Hybrid IR."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw.eval()

    weights: dict[str, np.ndarray] = {}
    w_init, b_init = fold_conv_bn(unwrap_qat_conv(raw.init_conv), raw.init_bn)
    weights["init_conv_w"] = w_init.astype(np.float32)
    if b_init is not None:
        weights["init_conv_b"] = b_init.astype(np.float32)

    for i, blk in enumerate(raw.res_blocks):
        w1, b1 = fold_conv_bn(unwrap_qat_conv(blk.conv1), blk.bn1)
        w2, b2 = fold_conv_bn(unwrap_qat_conv(blk.conv2), blk.bn2)
        weights[f"res_{i}_conv1_w"] = w1.astype(np.float32)
        if b1 is not None:
            weights[f"res_{i}_conv1_b"] = b1.astype(np.float32)
        weights[f"res_{i}_conv2_w"] = w2.astype(np.float32)
        if b2 is not None:
            weights[f"res_{i}_conv2_b"] = b2.astype(np.float32)
        weights[f"res_{i}_se_sq_w"] = blk.se.squeeze.weight.detach().cpu().numpy().astype(np.float32)
        weights[f"res_{i}_se_ex_w"] = blk.se.excite.weight.detach().cpu().numpy().astype(np.float32)

    wp, bp = fold_conv_bn(unwrap_qat_conv(raw.pol_conv), raw.pol_bn)
    weights["pol_73_w"] = wp.astype(np.float32)
    if bp is not None:
        weights["pol_73_b"] = bp.astype(np.float32)

    weights["val_conv_w"] = raw.hl_head.val_conv.weight.detach().cpu().numpy().astype(np.float32)
    weights["val_conv_b"] = raw.hl_head.val_conv.bias.detach().cpu().numpy().astype(np.float32)
    weights["val_dense1_w"] = raw.hl_head.val_fc1.weight.detach().cpu().numpy().astype(np.float32)
    weights["val_dense1_b"] = raw.hl_head.val_fc1.bias.detach().cpu().numpy().astype(np.float32)
    weights["val_dense2_w"] = raw.hl_head.val_fc2.weight.detach().cpu().numpy().astype(np.float32)
    weights["val_dense2_b"] = raw.hl_head.val_fc2.bias.detach().cpu().numpy().astype(np.float32)
    weights["val_hl_z"] = raw.hl_z.detach().cpu().numpy().astype(np.float32)

    weights["mlh_conv_w"] = raw.ml_head.mlh_conv.weight.detach().cpu().numpy().astype(np.float32)
    weights["mlh_conv_b"] = raw.ml_head.mlh_conv.bias.detach().cpu().numpy().astype(np.float32)
    weights["mlh_d1_w"] = raw.ml_head.mlh_d1.weight.detach().cpu().numpy().astype(np.float32)
    weights["mlh_d1_b"] = raw.ml_head.mlh_d1.bias.detach().cpu().numpy().astype(np.float32)
    weights["mlh_d2_w"] = raw.ml_head.mlh_d2.weight.detach().cpu().numpy().astype(np.float32)
    weights["mlh_d2_b"] = raw.ml_head.mlh_d2.bias.detach().cpu().numpy().astype(np.float32)
    weights["encoder_layout"] = np.array("gpu_unpack")
    weights["author"] = np.array(AUTHOR_META)
    weights["license"] = np.array(LICENSE_META)

    npz_p = out_p / f"{prefix}_folded.npz"
    np.savez_compressed(npz_p, **weights)

    ov_fp16 = build_openvino_se_resnet(
        batch_size=-1,
        num_blocks=raw.num_blocks,
        channels=raw.channels,
        in_channels=18,
        se_reduction=raw.se_reduction,
        policy_head="conv73",
        weights_dict=weights,
        model_name=f"NeuraChess_{prefix}_FP16",
    )
    fp16_xml = out_p / f"{prefix}_fp16.xml"
    ov.save_model(ov_fp16, str(fp16_xml), compress_to_fp16=True)

    outputs = {
        "folded_npz": str(npz_p),
        "fp16_xml": str(fp16_xml),
        "fp16_bin": str(fp16_xml.with_suffix(".bin")),
    }

    if calib_inputs and int8_blocks:
        try:
            import nncf

            fp16_blocks = [j for j in range(raw.num_blocks) if j not in int8_blocks]
            ig_patterns = [
                "^init_.*",
                ".*_se_.*",
                "^pol_.*",
                "^val_.*",
                "^(value|wdl)$",
                ".*_out_.*",
            ] + [f"^res_{j}_.*" for j in fp16_blocks]
            ig = nncf.IgnoredScope(patterns=ig_patterns, validate=True)
            calib_ds = nncf.Dataset(calib_inputs, lambda a: {"board_state": a})
            q_ov = nncf.quantize(
                ov_fp16,
                calib_ds,
                target_device=nncf.TargetDevice.NPU,
                subset_size=len(calib_inputs),
                fast_bias_correction=True,
                ignored_scope=ig,
            )
            hyb_xml = out_p / f"{prefix}_hybrid_int8.xml"
            ov.save_model(q_ov, str(hyb_xml), compress_to_fp16=True)
            outputs["hybrid_int8_xml"] = str(hyb_xml)
            outputs["hybrid_int8_bin"] = str(hyb_xml.with_suffix(".bin"))
        except Exception as e:
            outputs["hybrid_int8_error"] = f"{type(e).__name__}: {e}"

    return outputs
