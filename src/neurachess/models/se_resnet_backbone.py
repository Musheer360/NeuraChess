"""
NeuraChess Squeeze-and-Excitation ResNet Architecture (v2.5)
Constructed natively using OpenVINO opset13 for direct Intel AI Boost NPU compilation.

Features:
  - Input:  [B, 55, 8, 8] float32 canonical board tensor
  - Backbone: 6 Residual Blocks with Squeeze-and-Excitation (SE) channel attention
  - Policy Head: [B, 4672] raw logits over discrete action space
  - Value Head: 3-Way WDL [B, 3] Softmax probabilities [P_win, P_draw, P_loss]
"""

import openvino as ov
import openvino.opset13 as ops
import numpy as np


def _build_se_resnet_core(
    batch_size: int = 1,
    num_blocks: int = 6,
    channels: int = 128,
    in_channels: int = 18,
    se_reduction: int = 4,
    policy_head: str = "conv73",
    weights_dict: dict[str, np.ndarray] | None = None,
    model_name: str = "NeuraChess_SE_ResNet",
):
    """Build shared OpenVINO SE-ResNet backbone and primary policy head."""
    rng = np.random.RandomState(42)

    def get_w(name, shape, scale=0.05):
        if weights_dict and name in weights_dict:
            w_arr = weights_dict[name].astype(np.float32)
            if tuple(w_arr.shape) != tuple(shape):
                w_arr = np.resize(w_arr, shape)
            return ops.constant(w_arr, ov.Type.f32, name=name)
        fan_in = np.prod(shape[1:])
        std = np.sqrt(2.0 / fan_in) if fan_in > 0 else scale
        w = (rng.randn(*shape) * std).astype(np.float32)
        return ops.constant(w, ov.Type.f32, name=name)

    def maybe_add_bias(node, b_name, num_channels, op_name=None):
        if weights_dict and b_name in weights_dict:
            b_arr = weights_dict[b_name].astype(np.float32).reshape(1, num_channels, 1, 1)
            b_const = ops.constant(b_arr, ov.Type.f32, name=b_name)
            return ops.add(node, b_const, name=op_name or f"{b_name}_add")
        return node

    def maybe_add_dense_bias(node, b_name, out_dim, op_name=None):
        if weights_dict and b_name in weights_dict:
            b_arr = weights_dict[b_name].astype(np.float32).reshape(1, out_dim)
            b_const = ops.constant(b_arr, ov.Type.f32, name=b_name)
            return ops.add(node, b_const, name=op_name or f"{b_name}_add")
        return node

    # 1. Input Parameter [B, in_channels, 8, 8] (supports dynamic batch_size <= 0 via PartialShape)
    if batch_size is None or int(batch_size) <= 0:
        inp_shape = ov.PartialShape([-1, in_channels, 8, 8])
        model_tag = f"{model_name}_Bdyn"
    else:
        inp_shape = [int(batch_size), in_channels, 8, 8]
        model_tag = f"{model_name}_B{batch_size}"
    inp = ops.parameter(inp_shape, ov.Type.f32, name="board_state")

    # 2. Initial Convolution in_channels -> channels (3x3, pad 1)
    w_init = get_w("init_conv_w", (channels, in_channels, 3, 3))
    x = ops.convolution(
        inp, w_init, strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1], name="init_conv"
    )
    x = maybe_add_bias(x, "init_conv_b", channels, op_name="init_conv_add")
    x = ops.relu(x, name="init_relu")

    # Reduction channels for SE
    se_ch = max(8, channels // se_reduction)

    # 3. Residual Blocks with Squeeze-and-Excitation
    for i in range(num_blocks):
        w1 = get_w(f"res_{i}_conv1_w", (channels, channels, 3, 3))
        c1 = ops.convolution(
            x, w1, strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1], name=f"res_{i}_conv1"
        )
        c1 = maybe_add_bias(c1, f"res_{i}_conv1_b", channels, op_name=f"res_{i}_conv1_add")
        r1 = ops.relu(c1, name=f"res_{i}_relu1")

        w2 = get_w(f"res_{i}_conv2_w", (channels, channels, 3, 3))
        c2 = ops.convolution(
            r1, w2, strides=[1, 1], pads_begin=[1, 1], pads_end=[1, 1], dilations=[1, 1], name=f"res_{i}_conv2"
        )
        c2 = maybe_add_bias(c2, f"res_{i}_conv2_b", channels, op_name=f"res_{i}_conv2_add")

        gap = ops.reduce_mean(
            c2, ops.constant(np.array([2, 3], dtype=np.int64), name=f"res_{i}_se_axes"), keep_dims=True, name=f"res_{i}_se_gap"
        )

        w_sq = get_w(f"res_{i}_se_sq_w", (se_ch, channels, 1, 1))
        sq = ops.convolution(
            gap, w_sq, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name=f"res_{i}_se_sq"
        )
        sq = maybe_add_bias(sq, f"res_{i}_se_sq_b", se_ch, op_name=f"res_{i}_se_sq_add")
        sq = ops.relu(sq, name=f"res_{i}_se_relu")

        w_ex = get_w(f"res_{i}_se_ex_w", (channels, se_ch, 1, 1))
        ex = ops.convolution(
            sq, w_ex, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name=f"res_{i}_se_ex"
        )
        ex = maybe_add_bias(ex, f"res_{i}_se_ex_b", channels, op_name=f"res_{i}_se_ex_add")
        ex = ops.sigmoid(ex, name=f"res_{i}_se_sig")

        c2_scaled = ops.multiply(c2, ex, name=f"res_{i}_se_mul")
        res_sum = ops.add(x, c2_scaled, name=f"res_{i}_out_add")
        x = ops.relu(res_sum, name=f"res_{i}_out_relu")

    # 4. Policy Head: Conv73 or Legacy Dense
    has_conv73_weights = bool(weights_dict and "pol_73_w" in weights_dict)
    has_legacy_weights = bool(weights_dict and "pol_conv_w" in weights_dict)
    if has_conv73_weights or (policy_head == "conv73" and not has_legacy_weights):
        w_pol_73 = get_w("pol_73_w", (73, channels, 1, 1))
        pol_conv = ops.convolution(
            x, w_pol_73, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name="pol_73_conv"
        )
        pol_conv = maybe_add_bias(pol_conv, "pol_73_b", 73, op_name="pol_73_add")
        pol_trans = ops.transpose(
            pol_conv, ops.constant(np.array([0, 2, 3, 1], dtype=np.int64), name="pol_perm"), name="pol_trans"
        )
        policy_logits = ops.reshape(
            pol_trans, ops.constant(np.array([0, 4672], dtype=np.int64), name="pol_shape"), True, name="pol_reshape"
        )
    else:
        w_pol_conv = get_w("pol_conv_w", (32, channels, 1, 1))
        pol_conv = ops.convolution(
            x, w_pol_conv, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name="pol_conv"
        )
        pol_conv = maybe_add_bias(pol_conv, "pol_conv_b", 32, op_name="pol_conv_add")
        pol_conv = ops.relu(pol_conv, name="pol_conv_relu")
        pol_flat = ops.reshape(
            pol_conv, ops.constant(np.array([0, 32 * 8 * 8], dtype=np.int64), name="pol_flat_shape"), True, name="pol_flat"
        )
        w_pol_dense = get_w("pol_dense_w", (4672, 32 * 8 * 8))
        policy_logits = ops.matmul(pol_flat, w_pol_dense, transpose_a=False, transpose_b=True, name="pol_dense")
        policy_logits = maybe_add_dense_bias(policy_logits, "pol_dense_b", 4672, op_name="pol_dense_add")

    return inp, x, policy_logits, model_tag, get_w, maybe_add_bias, maybe_add_dense_bias


def _build_standard_value_head(
    x,
    channels,
    weights_dict,
    get_w,
    maybe_add_bias,
    maybe_add_dense_bias,
    export_uncertainty: bool = False,
):
    val_ch = int(weights_dict["val_conv_w"].shape[0]) if (weights_dict and "val_conv_w" in weights_dict) else 16
    val_h = int(weights_dict["val_dense1_w"].shape[0]) if (weights_dict and "val_dense1_w" in weights_dict) else 128
    val_dim = int(weights_dict["val_dense2_w"].shape[0]) if (weights_dict and "val_dense2_w" in weights_dict) else 1

    w_val_conv = get_w("val_conv_w", (val_ch, channels, 1, 1))
    val_conv = ops.convolution(
        x, w_val_conv, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name="val_conv"
    )
    val_conv = maybe_add_bias(val_conv, "val_conv_b", val_ch, op_name="val_conv_add")
    val_conv = ops.relu(val_conv, name="val_conv_relu")
    val_flat = ops.reshape(
        val_conv,
        ops.constant(np.array([0, val_ch * 8 * 8], dtype=np.int64), name="val_flat_shape"),
        True,
        name="val_flat",
    )
    w_val_d1 = get_w("val_dense1_w", (val_h, val_ch * 8 * 8))
    val_d1 = ops.matmul(val_flat, w_val_d1, transpose_a=False, transpose_b=True, name="val_dense1")
    val_d1 = maybe_add_dense_bias(val_d1, "val_dense1_b", val_h, op_name="val_dense1_add")
    val_d1 = ops.relu(val_d1, name="val_dense1_relu")
    w_val_d2 = get_w("val_dense2_w", (val_dim, val_h))
    val_logits = ops.matmul(val_d1, w_val_d2, transpose_a=False, transpose_b=True, name="val_dense2")
    val_logits = maybe_add_dense_bias(val_logits, "val_dense2_b", val_dim, op_name="val_dense2_add")

    val_sigma = None
    if val_dim == 1:
        val_out = ops.tanh(val_logits, name="value")
    elif weights_dict and "val_hl_z" in weights_dict:
        z_arr = np.asarray(weights_dict["val_hl_z"], dtype=np.float32).reshape(val_dim, 1)
        val_probs = ops.softmax(val_logits, axis=1, name="val_hl_probs")
        z_const = ops.constant(z_arr, ov.Type.f32, name="val_hl_z")
        val_out = ops.matmul(val_probs, z_const, transpose_a=False, transpose_b=False, name="value")
        if export_uncertainty:
            z2_arr = (z_arr * z_arr).astype(np.float32)
            z2_const = ops.constant(z2_arr, ov.Type.f32, name="val_hl_z2")
            ez2 = ops.matmul(val_probs, z2_const, transpose_a=False, transpose_b=False, name="val_hl_ez2")
            v_sq = ops.multiply(val_out, val_out, name="val_hl_v_sq")
            var = ops.subtract(ez2, v_sq, name="val_hl_var")
            eps = ops.constant(np.array([[1e-6]], dtype=np.float32), ov.Type.f32, name="val_hl_eps")
            var_clamped = ops.maximum(var, eps, name="val_hl_var_clamped")
            val_sigma = ops.sqrt(var_clamped, name="val_sigma")
    else:
        val_out = ops.softmax(val_logits, axis=1, name="wdl")
    return val_d1, val_out, val_sigma


def _build_mlh_head(x, channels, weights_dict, get_w, maybe_add_bias, maybe_add_dense_bias):
    mlh_ch = int(weights_dict["mlh_conv_w"].shape[0]) if (weights_dict and "mlh_conv_w" in weights_dict) else 8
    mlh_h = int(weights_dict["mlh_d1_w"].shape[0]) if (weights_dict and "mlh_d1_w" in weights_dict) else 64
    w_mc = get_w("mlh_conv_w", (mlh_ch, channels, 1, 1))
    mc = ops.convolution(
        x, w_mc, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name="mlh_conv"
    )
    mc = maybe_add_bias(mc, "mlh_conv_b", mlh_ch, op_name="mlh_conv_add")
    mc = ops.relu(mc, name="mlh_conv_relu")
    m_flat = ops.reshape(
        mc,
        ops.constant(np.array([0, mlh_ch * 8 * 8], dtype=np.int64), name="mlh_flat_shape"),
        True,
        name="mlh_flat",
    )
    w_md1 = get_w("mlh_d1_w", (mlh_h, mlh_ch * 8 * 8))
    md1 = ops.matmul(m_flat, w_md1, transpose_a=False, transpose_b=True, name="mlh_d1")
    md1 = maybe_add_dense_bias(md1, "mlh_d1_b", mlh_h, op_name="mlh_d1_add")
    md1 = ops.relu(md1, name="mlh_d1_relu")
    w_md2 = get_w("mlh_d2_w", (1, mlh_h))
    md2 = ops.matmul(md1, w_md2, transpose_a=False, transpose_b=True, name="mlh_d2")
    md2 = maybe_add_dense_bias(md2, "mlh_d2_b", 1, op_name="mlh_d2_add")
    return ops.relu(md2, name="moves_left")


def _build_action_q_head(x, channels, get_w, maybe_add_bias):
    w_q = get_w("q_73_w", (73, channels, 1, 1))
    qc = ops.convolution(
        x, w_q, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name="q_73_conv"
    )
    qc = maybe_add_bias(qc, "q_73_b", 73, op_name="q_73_add")
    qt = ops.transpose(
        qc, ops.constant(np.array([0, 2, 3, 1], dtype=np.int64), name="q_perm"), name="q_trans"
    )
    qf = ops.reshape(
        qt, ops.constant(np.array([0, 4672], dtype=np.int64), name="q_shape"), True, name="q_reshape"
    )
    return ops.tanh(qf, name="action_q")


def build_openvino_se_resnet(
    batch_size: int = 1,
    num_blocks: int = 6,
    channels: int = 128,
    in_channels: int = 18,
    se_reduction: int = 4,
    policy_head: str = "conv73",
    weights_dict: dict[str, np.ndarray] | None = None,
    model_name: str = "NeuraChess_SE_ResNet",
    export_uncertainty: bool = False,
    export_mlh: bool = False,
    export_q_head: bool = False,
) -> ov.Model:
    """
    Constructs an OpenVINO Model with SE blocks and scalar/WDL/HL-Gauss-64 value head,
    strictly compliant with Intel NPU 3720 constraints.

    Optional auxiliary outputs (preserving [policy_logits, value] as outputs 0 and 1):
      - `export_uncertainty=True`: appends `val_sigma` [B, 1] from HL-Gauss-64 second moment.
      - `export_mlh=True`: appends `moves_left` [B, 1] from the trained Moves-Left head.
      - `export_q_head=True`: appends `action_q` [B, 4672] from the parallel Conv73 Q(s,a) head.
    """
    inp, x, policy_logits, model_tag, get_w, maybe_add_bias, maybe_add_dense_bias = _build_se_resnet_core(
        batch_size=batch_size,
        num_blocks=num_blocks,
        channels=channels,
        in_channels=in_channels,
        se_reduction=se_reduction,
        policy_head=policy_head,
        weights_dict=weights_dict,
        model_name=model_name,
    )
    _, val_out, val_sigma = _build_standard_value_head(
        x,
        channels,
        weights_dict,
        get_w,
        maybe_add_bias,
        maybe_add_dense_bias,
        export_uncertainty=export_uncertainty,
    )
    policy_logits.output(0).set_names({"policy_logits"})
    val_out.output(0).set_names({"value"})
    results = [ops.result(policy_logits, name="policy_logits_out"), ops.result(val_out, name="value_out")]
    if export_uncertainty and val_sigma is not None:
        val_sigma.output(0).set_names({"val_sigma"})
        results.append(ops.result(val_sigma, name="val_sigma_out"))
    if export_mlh and weights_dict and "mlh_conv_w" in weights_dict:
        mlh_out = _build_mlh_head(x, channels, weights_dict, get_w, maybe_add_bias, maybe_add_dense_bias)
        mlh_out.output(0).set_names({"moves_left"})
        results.append(ops.result(mlh_out, name="moves_left_out"))
    if export_q_head and weights_dict and "q_73_w" in weights_dict:
        q_out = _build_action_q_head(x, channels, get_w, maybe_add_bias)
        q_out.output(0).set_names({"action_q"})
        results.append(ops.result(q_out, name="action_q_out"))
    return ov.Model(results, [inp], model_tag)


def build_openvino_research_head_variant(
    variant: str,
    batch_size: int = 16,
    num_blocks: int = 10,
    channels: int = 192,
    in_channels: int = 18,
    se_reduction: int = 4,
    weights_dict: dict[str, np.ndarray] | None = None,
) -> ov.Model:
    """Construct experimental multi-head graphs on top of the SE-ResNet backbone for NPU latency benchmarking."""
    inp, x, policy_logits, model_tag, get_w, maybe_add_bias, maybe_add_dense_bias = _build_se_resnet_core(
        batch_size=batch_size,
        num_blocks=num_blocks,
        channels=channels,
        in_channels=in_channels,
        se_reduction=se_reduction,
        policy_head="conv73",
        weights_dict=weights_dict,
        model_name=f"NeuraChess_{variant}",
    )
    rng_h = np.random.default_rng(1234)

    def rand_w(name: str, shape: tuple[int, ...], std: float = 0.02):
        return ops.constant((rng_h.standard_normal(shape) * std).astype(np.float32), ov.Type.f32, name=name)

    def rand_b(name: str, shape: tuple[int, ...], std: float = 0.01):
        return ops.constant((rng_h.standard_normal(shape) * std).astype(np.float32), ov.Type.f32, name=name)

    if variant == "baseline_scalar":
        _, val_out, _ = _build_standard_value_head(x, channels, weights_dict, get_w, maybe_add_bias, maybe_add_dense_bias)
        return ov.Model([ops.result(policy_logits), ops.result(val_out)], [inp], model_tag)

    if variant == "hl_gauss_mlh":
        # Shared value trunk (val_conv 16ch -> val_dense1 128) feeding 64-bin HL-Gauss softmax
        w_val_conv = get_w("val_conv_w", (16, channels, 1, 1))
        val_conv = ops.convolution(
            x, w_val_conv, strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name="val_conv"
        )
        val_conv = maybe_add_bias(val_conv, "val_conv_b", 16, op_name="val_conv_add")
        val_conv = ops.relu(val_conv, name="val_conv_relu")
        val_flat = ops.reshape(
            val_conv, ops.constant(np.array([0, 16 * 8 * 8], dtype=np.int64), name="val_flat_shape"), True, name="val_flat"
        )
        w_val_d1 = get_w("val_dense1_w", (128, 16 * 8 * 8))
        val_d1 = ops.matmul(val_flat, w_val_d1, transpose_a=False, transpose_b=True, name="val_dense1")
        val_d1 = maybe_add_dense_bias(val_d1, "val_dense1_b", 128, op_name="val_dense1_add")
        val_d1 = ops.relu(val_d1, name="val_dense1_relu")
        hl_d2 = ops.matmul(val_d1, rand_w("val_hl64_w", (64, 128)), transpose_a=False, transpose_b=True, name="val_hl64_dense")
        hl_d2 = ops.add(hl_d2, rand_b("val_hl64_b", (1, 64)), name="val_hl64_add")
        hl_probs = ops.softmax(hl_d2, axis=1, name="hl_gauss_64")

        # Moves-Left Head: Conv(192 -> 8, 1x1) -> ReLU -> Flat(512) -> Dense(64) -> ReLU -> Dense(1) -> ReLU
        mlh_c = ops.convolution(
            x, rand_w("mlh_conv_w", (8, channels, 1, 1)), strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name="mlh_conv"
        )
        mlh_c = ops.relu(ops.add(mlh_c, rand_b("mlh_conv_b", (1, 8, 1, 1)), name="mlh_conv_add"), name="mlh_conv_relu")
        mlh_flat = ops.reshape(mlh_c, ops.constant(np.array([0, 8 * 8 * 8], dtype=np.int64), name="mlh_flat_shape"), True, name="mlh_flat")
        mlh_d1 = ops.matmul(mlh_flat, rand_w("mlh_d1_w", (64, 8 * 8 * 8)), transpose_a=False, transpose_b=True, name="mlh_d1")
        mlh_d1 = ops.relu(ops.add(mlh_d1, rand_b("mlh_d1_b", (1, 64)), name="mlh_d1_add"), name="mlh_d1_relu")
        mlh_d2 = ops.matmul(mlh_d1, rand_w("mlh_d2_w", (1, 64)), transpose_a=False, transpose_b=True, name="mlh_d2")
        mlh_out = ops.relu(ops.add(mlh_d2, rand_b("mlh_d2_b", (1, 1)), name="mlh_d2_add"), name="moves_left")
        return ov.Model([ops.result(policy_logits), ops.result(hl_probs), ops.result(mlh_out)], [inp], model_tag)

    if variant in ("dual_conv73_with_val", "dual_conv73_only"):
        # Action-Value Q(s, a) Conv73 head parallel to Policy Conv73
        q_conv = ops.convolution(
            x, rand_w("q_73_w", (73, channels, 1, 1)), strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1], name="q_73_conv"
        )
        q_conv = ops.add(q_conv, rand_b("q_73_b", (1, 73, 1, 1)), name="q_73_add")
        q_trans = ops.transpose(q_conv, ops.constant(np.array([0, 2, 3, 1], dtype=np.int64), name="q_perm"), name="q_trans")
        q_flat = ops.reshape(q_trans, ops.constant(np.array([0, 4672], dtype=np.int64), name="q_shape"), True, name="q_reshape")
        q_out = ops.tanh(q_flat, name="action_q")
        if variant == "dual_conv73_only":
            return ov.Model([ops.result(policy_logits), ops.result(q_out)], [inp], model_tag)
        _, val_out, _ = _build_standard_value_head(x, channels, weights_dict, get_w, maybe_add_bias, maybe_add_dense_bias)
        return ov.Model([ops.result(policy_logits), ops.result(q_out), ops.result(val_out)], [inp], model_tag)

    raise ValueError(f"Unknown research head variant: {variant}")
