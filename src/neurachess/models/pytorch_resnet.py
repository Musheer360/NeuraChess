"""
NeuraChess PyTorch ResNet Implementation (v2.0)
Mathematically identical to the OpenVINO NPU computation graph.
Used for backpropagation and incremental local training.
"""

import os
import torch
import torch.nn as nn
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        out = self.relu2(out + residual)
        return out


class ChessResNet(nn.Module):
    def __init__(self, in_channels: int = 55, num_blocks: int = 6, channels: int = 128, num_actions: int = 4672):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        # 1. Initial Conv
        self.init_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.init_relu = nn.ReLU(inplace=True)

        # 2. Residual Tower
        self.blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # 3. Policy Head
        self.pol_conv = nn.Conv2d(channels, 32, kernel_size=1, bias=False)
        self.pol_relu = nn.ReLU(inplace=True)
        self.pol_linear = nn.Linear(32 * 8 * 8, num_actions, bias=False)

        # 4. Value Head
        self.val_conv = nn.Conv2d(channels, 16, kernel_size=1, bias=False)
        self.val_relu = nn.ReLU(inplace=True)
        self.val_linear1 = nn.Linear(16 * 8 * 8, 128, bias=False)
        self.val_relu2 = nn.ReLU(inplace=True)
        self.val_linear2 = nn.Linear(128, 1, bias=False)
        self.val_tanh = nn.Tanh()

    def forward(self, x):
        x = self.init_relu(self.init_conv(x))
        for block in self.blocks:
            x = block(x)

        # Policy
        p = self.pol_relu(self.pol_conv(x))
        p_flat = p.reshape(p.size(0), -1)
        policy_logits = self.pol_linear(p_flat)

        # Value
        v = self.val_relu(self.val_conv(x))
        v_flat = v.reshape(v.size(0), -1)
        v_hidden = self.val_relu2(self.val_linear1(v_flat))
        value = self.val_tanh(self.val_linear2(v_hidden))

        return policy_logits, value

    def export_to_npz(self, out_path: str):
        """Exports weights dictionary directly compatible with OpenVINO NPU compiler."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        w_dict = {
            "init_conv_w": self.init_conv.weight.detach().cpu().numpy(),
            "pol_conv_w": self.pol_conv.weight.detach().cpu().numpy(),
            "pol_dense_w": self.pol_linear.weight.detach().cpu().numpy(),
            "val_conv_w": self.val_conv.weight.detach().cpu().numpy(),
            "val_dense1_w": self.val_linear1.weight.detach().cpu().numpy(),
            "val_dense2_w": self.val_linear2.weight.detach().cpu().numpy(),
        }
        for i, block in enumerate(self.blocks):
            w_dict[f"res_{i}_conv1_w"] = block.conv1.weight.detach().cpu().numpy()
            w_dict[f"res_{i}_conv2_w"] = block.conv2.weight.detach().cpu().numpy()

        np.savez_compressed(out_path, **w_dict)
        print(f"Exported PyTorch weights to OpenVINO format at: {out_path}")

    def load_from_npz(self, npz_path: str):
        """Loads weights from OpenVINO-format npz dictionary."""
        npz = np.load(npz_path)
        with torch.no_grad():
            self.init_conv.weight.copy_(torch.from_numpy(npz["init_conv_w"]))
            self.pol_conv.weight.copy_(torch.from_numpy(npz["pol_conv_w"]))
            self.pol_linear.weight.copy_(torch.from_numpy(npz["pol_dense_w"]))
            self.val_conv.weight.copy_(torch.from_numpy(npz["val_conv_w"]))
            self.val_linear1.weight.copy_(torch.from_numpy(npz["val_dense1_w"]))
            self.val_linear2.weight.copy_(torch.from_numpy(npz["val_dense2_w"]))

            for i, block in enumerate(self.blocks):
                block.conv1.weight.copy_(torch.from_numpy(npz[f"res_{i}_conv1_w"]))
                block.conv2.weight.copy_(torch.from_numpy(npz[f"res_{i}_conv2_w"]))
        print(f"Successfully loaded weights from {npz_path}")
