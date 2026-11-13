"""
NeuraChess PyTorch Squeeze-and-Excitation ResNet Module (v2.5)

PyTorch implementation with SE blocks and 3-way WDL value head.
Allows full training in PyTorch and direct weight export to OpenVINO NPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.shadow = {name: p.clone().detach() for name, p in raw_m.state_dict().items()}

    def update(self, model: nn.Module):
        raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
        with torch.no_grad():
            for name, p in raw_m.state_dict().items():
                if name in self.shadow:
                    if p.is_floating_point():
                        self.shadow[name].mul_(self.decay).add_(p, alpha=1.0 - self.decay)
                    else:
                        self.shadow[name].copy_(p)


def fold_conv_bn(conv: nn.Conv2d, bn: nn.Module | None) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Analytically folds BatchNorm2d parameters into Conv2d weights and biases:
      W_folded = W * (gamma / sqrt(var + eps))
      B_folded = (B - mean) * (gamma / sqrt(var + eps)) + beta
    Achieves zero inference compute overhead on OpenVINO NPU.
    """
    w = conv.weight.detach().cpu().numpy()
    if bn is None or isinstance(bn, nn.Identity):
        b = conv.bias.detach().cpu().numpy() if conv.bias is not None else None
        return w, b

    gamma = bn.weight.detach().cpu().numpy()
    beta = bn.bias.detach().cpu().numpy()
    mean = bn.running_mean.detach().cpu().numpy()
    var = bn.running_var.detach().cpu().numpy()
    eps = bn.eps

    scale = gamma / np.sqrt(var + eps)
    w_folded = w * scale[:, None, None, None]
    b = conv.bias.detach().cpu().numpy() if conv.bias is not None else np.zeros(conv.out_channels, dtype=np.float32)
    b_folded = (b - mean) * scale + beta
    return w_folded, b_folded


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        se_ch = max(8, channels // reduction)
        self.squeeze = nn.Conv2d(channels, se_ch, kernel_size=1, bias=False)
        self.excite = nn.Conv2d(se_ch, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global Average Pooling
        gap = torch.mean(x, dim=(2, 3), keepdim=True)
        sq = F.relu(self.squeeze(gap))
        ex = torch.sigmoid(self.excite(sq))
        return x * ex


class SEResidualBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, use_bn: bool = False):
        super().__init__()
        self.use_bn = use_bn
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.se = SEBlock(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + res)


class ChessSEResNet(nn.Module):
    def __init__(
        self,
        num_blocks: int = 6,
        channels: int = 128,
        se_reduction: int = 4,
        value_dim: int = 1,
        use_bn: bool = False,
        policy_head: str = "legacy",
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        self.se_reduction = se_reduction
        self.value_dim = value_dim
        self.use_bn = use_bn
        self.policy_head = policy_head.lower()

        # Initial conv (18 canonical planes)
        self.init_conv = nn.Conv2d(18, channels, kernel_size=3, padding=1, bias=False)
        self.init_bn = nn.BatchNorm2d(channels) if use_bn else nn.Identity()

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            SEResidualBlock(channels, reduction=se_reduction, use_bn=use_bn) for _ in range(num_blocks)
        ])

        # Policy Head
        if self.policy_head == "conv73":
            # AlphaZero-style 73x8x8 spatial conv head (channels -> 73 channels)
            self.pol_conv = nn.Conv2d(channels, 73, kernel_size=1, bias=not use_bn)
            self.pol_bn = nn.BatchNorm2d(73) if use_bn else nn.Identity()
            self.pol_dense = None
        else:
            # Legacy Head: Conv 1x1 (channels -> 32) -> Dense 4672
            self.pol_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
            self.pol_bn = nn.Identity()
            self.pol_dense = nn.Linear(32 * 8 * 8, 4672, bias=False)

        # Value Head: Conv 1x1 (channels -> 16) -> Dense 128 -> Dense value_dim (1=Tanh scalar, 3=WDL)
        self.val_conv = nn.Conv2d(channels, 16, kernel_size=1, bias=False)
        self.val_dense1 = nn.Linear(16 * 8 * 8, 128, bias=False)
        self.val_dense2 = nn.Linear(128, value_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            policy_logits: [B, 4672]
            value_out: [B, 1] (if value_dim=1, Tanh in [-1, +1]) or [B, 3] (if value_dim=3, Softmax WDL)
        """
        h = F.relu(self.init_bn(self.init_conv(x)))
        for block in self.res_blocks:
            h = block(h)

        # Policy Head
        if self.policy_head == "conv73":
            p = self.pol_bn(self.pol_conv(h))
            # Permute from [B, 73, 8, 8] to [B, 8, 8, 73]
            # where dim 1 is from_r (0..7), dim 2 is from_c (0..7), dim 3 is slot (0..72)
            # Row-major flat index = (from_r * 8 + from_c) * 73 + slot == from_sq * 73 + slot
            policy_logits = p.permute(0, 2, 3, 1).contiguous().reshape(x.shape[0], 4672)
        else:
            p = F.relu(self.pol_conv(h))
            p = p.flatten(start_dim=1)
            policy_logits = self.pol_dense(p)

        # Value Head
        v = F.relu(self.val_conv(h))
        v = v.flatten(start_dim=1)
        v = F.relu(self.val_dense1(v))
        val_logits = self.val_dense2(v)

        if self.value_dim == 1:
            val_out = torch.tanh(val_logits)
        else:
            val_out = F.softmax(val_logits, dim=1)

        return policy_logits, val_out

    def export_to_npz(self, path: str, encoder_layout: str = "gpu_unpack") -> str:
        """Export model weights to .npz for OpenVINO NPU compilation, folding BatchNorm if present."""
        self.eval()
        weights = {}

        w_init, b_init = fold_conv_bn(self.init_conv, self.init_bn)
        weights["init_conv_w"] = w_init
        if b_init is not None:
            weights["init_conv_b"] = b_init

        for i, block in enumerate(self.res_blocks):
            w1, b1 = fold_conv_bn(block.conv1, block.bn1)
            w2, b2 = fold_conv_bn(block.conv2, block.bn2)
            weights[f"res_{i}_conv1_w"] = w1
            if b1 is not None:
                weights[f"res_{i}_conv1_b"] = b1
            weights[f"res_{i}_conv2_w"] = w2
            if b2 is not None:
                weights[f"res_{i}_conv2_b"] = b2

            weights[f"res_{i}_se_sq_w"] = block.se.squeeze.weight.detach().cpu().numpy()
            weights[f"res_{i}_se_ex_w"] = block.se.excite.weight.detach().cpu().numpy()

        if self.policy_head == "conv73":
            wp, bp = fold_conv_bn(self.pol_conv, self.pol_bn)
            weights["pol_73_w"] = wp
            if bp is not None:
                weights["pol_73_b"] = bp
        else:
            weights["pol_conv_w"] = self.pol_conv.weight.detach().cpu().numpy()
            weights["pol_dense_w"] = self.pol_dense.weight.detach().cpu().numpy()

        weights["val_conv_w"] = self.val_conv.weight.detach().cpu().numpy()
        weights["val_dense1_w"] = self.val_dense1.weight.detach().cpu().numpy()
        weights["val_dense2_w"] = self.val_dense2.weight.detach().cpu().numpy()
        if encoder_layout:
            weights["encoder_layout"] = np.array(encoder_layout)
        weights["author"] = np.array("Musheer Alam (https://github.com/Musheer360/NeuraChess)")
        weights["license"] = np.array("GPL-3.0")

        np.savez_compressed(path, **weights)
        return path

    def load_from_npz(self, path: str):
        """Load weights from .npz."""
        npz = np.load(path)
        with torch.no_grad():
            if "init_conv_w" in npz:
                self.init_conv.weight.copy_(torch.from_numpy(npz["init_conv_w"]))
            for i, block in enumerate(self.res_blocks):
                k1 = f"res_{i}_conv1_w"
                k2 = f"res_{i}_conv2_w"
                ksq = f"res_{i}_se_sq_w"
                kex = f"res_{i}_se_ex_w"
                if k1 in npz:
                    block.conv1.weight.copy_(torch.from_numpy(npz[k1]))
                if k2 in npz:
                    block.conv2.weight.copy_(torch.from_numpy(npz[k2]))
                if ksq in npz:
                    block.se.squeeze.weight.copy_(torch.from_numpy(npz[ksq]))
                if kex in npz:
                    block.se.excite.weight.copy_(torch.from_numpy(npz[kex]))

            if "pol_conv_w" in npz:
                self.pol_conv.weight.copy_(torch.from_numpy(npz["pol_conv_w"]))
            if "pol_dense_w" in npz:
                self.pol_dense.weight.copy_(torch.from_numpy(npz["pol_dense_w"]))
            if "val_conv_w" in npz:
                self.val_conv.weight.copy_(torch.from_numpy(npz["val_conv_w"]))
            if "val_dense1_w" in npz:
                self.val_dense1.weight.copy_(torch.from_numpy(npz["val_dense1_w"]))
            if "val_dense2_w" in npz:
                w2 = npz["val_dense2_w"]
                if w2.shape == self.val_dense2.weight.shape:
                    self.val_dense2.weight.copy_(torch.from_numpy(w2))
                elif w2.shape == (3, 128) and self.value_dim == 1:
                    # Legacy 3-way to 1-way: init with (w_win - w_loss)
                    adapted = (w2[0:1] - w2[2:3]) * 0.5
                    self.val_dense2.weight.copy_(torch.from_numpy(adapted))
