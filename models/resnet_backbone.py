"""
NeuraChess Parametric ResNet Architecture (v2.0)
Constructed natively using OpenVINO opset13 for direct NPU 3720 compilation.

Input:  [B, 55, 8, 8] float32
Output:
  - policy_logits: [B, 4672] float32
  - value:         [B, 1] float32 in [-1.0, +1.0]
"""

import openvino as ov
import openvino.opset13 as ops
import numpy as np


def build_openvino_resnet(
    batch_size: int = 1,
    num_blocks: int = 6,
    channels: int = 128,
    weights_dict: dict[str, np.ndarray] | None = None,
    model_name: str = "NeuraChess_ResNet",
) -> ov.Model:
    """
    Constructs a parametric OpenVINO Model strictly compliant with Intel NPU 3720 constraints:
      - Strictly static shape [B, 55, 8, 8]
      - Zero dynamic dimensions or unsupported integer casts
      - FP16/FP32 tensor operations mapped to NCE hardware tiles
    """
    rng = np.random.RandomState(42)

    def get_w(name, shape, scale=0.05):
        if weights_dict and name in weights_dict:
            return ops.constant(weights_dict[name].astype(np.float32), name=name)
        # Kaiming-like initialization
        fan_in = np.prod(shape[1:])
        std = np.sqrt(2.0 / fan_in) if fan_in > 0 else scale
        w = (rng.randn(*shape) * std).astype(np.float32)
        return ops.constant(w, ov.Type.f32, name=name)

    # 1. Input Parameter [B, 55, 8, 8]
    inp = ops.parameter([batch_size, 55, 8, 8], ov.Type.f32, name="board_state")

    # 2. Initial Convolution 55 -> channels (3x3, pad 1)
    w_init = get_w("init_conv_w", (channels, 55, 3, 3))
    x = ops.convolution(inp, w_init, strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1])
    x = ops.relu(x)

    # 3. Residual Blocks
    for i in range(num_blocks):
        w1 = get_w(f"res_{i}_conv1_w", (channels, channels, 3, 3))
        c1 = ops.convolution(x, w1, strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1])
        r1 = ops.relu(c1)

        w2 = get_w(f"res_{i}_conv2_w", (channels, channels, 3, 3))
        c2 = ops.convolution(r1, w2, strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1])
        x = ops.relu(ops.add(x, c2))

    # 4. Policy Head: 1x1 Conv (channels -> 32) -> Flatten -> Dense 4672
    w_pol_conv = get_w("pol_conv_w", (32, channels, 1, 1))
    pol_conv = ops.relu(ops.convolution(x, w_pol_conv, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1]))
    pol_flat = ops.reshape(pol_conv, ops.constant(np.array([batch_size, 32 * 8 * 8], dtype=np.int64)), False)
    w_pol_dense = get_w("pol_dense_w", (4672, 32 * 8 * 8))
    policy_logits = ops.matmul(pol_flat, w_pol_dense, transpose_a=False, transpose_b=True, name="policy_logits")

    # 5. Value Head: 1x1 Conv (channels -> 16) -> Flatten -> Dense 128 -> Dense 1 -> Tanh
    w_val_conv = get_w("val_conv_w", (16, channels, 1, 1))
    val_conv = ops.relu(ops.convolution(x, w_val_conv, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1]))
    val_flat = ops.reshape(val_conv, ops.constant(np.array([batch_size, 16 * 8 * 8], dtype=np.int64)), False)
    w_val_d1 = get_w("val_dense1_w", (128, 16 * 8 * 8))
    val_d1 = ops.relu(ops.matmul(val_flat, w_val_d1, transpose_a=False, transpose_b=True))
    w_val_d2 = get_w("val_dense2_w", (1, 128))
    value = ops.tanh(ops.matmul(val_d1, w_val_d2, transpose_a=False, transpose_b=True), name="value")

    return ov.Model([ops.result(policy_logits), ops.result(value)], [inp], f"{model_name}_B{batch_size}")
