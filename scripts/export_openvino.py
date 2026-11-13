"""
NeuraChess OpenVINO IR Exporter & NPU Parity Verifier (`scripts/export_openvino.py`).

Exports PyTorch SE-ResNet checkpoints (`.pt`) or pre-folded `.npz` weights to OpenVINO IR
(`FP16`, or optional `W8A16` / `W8A8` via NNCF PTQ) and optionally verifies output parity
and batch throughput on the target OpenVINO device.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.npu_compiler import infer_arch_from_npz
from models.pytorch_se_resnet import ChessSEResNet
from models.se_resnet_backbone import build_openvino_se_resnet


def verify_ir_parity(
    ov_model: ov.Model,
    pt_model: torch.nn.Module | None,
    device: str = "NPU",
    batch_size: int = 16,
    bench_iters: int = 40,
) -> dict:
    core = ov.Core()
    if device == "NPU":
        core.set_property("NPU", {"PERFORMANCE_HINT": ov.properties.hint.PerformanceMode.LATENCY, "NPU_TURBO": True})

    m = ov_model.clone()
    ps = m.input(0).get_partial_shape()
    in_ch = int(ps[1].get_length()) if ps[1].is_static else 18
    m.reshape({m.input(0).get_any_name(): [batch_size, in_ch, 8, 8]})

    compiled = core.compile_model(m, device)
    req = compiled.create_infer_request()

    rng = np.random.default_rng(42)
    sample = rng.integers(0, 2, size=(batch_size, in_ch, 8, 8)).astype(np.float32)

    out_ov = list(req.infer({"board_state": sample}).values())
    metrics = {"device": device, "batch_size": batch_size}

    if pt_model is not None:
        pt_model.eval()
        with torch.no_grad():
            pol_pt, val_pt = pt_model(torch.from_numpy(sample))
        pol_np, val_np = pol_pt.numpy(), val_pt.numpy()
        top1_agree = float(np.mean(np.argmax(pol_np, axis=1) == np.argmax(out_ov[0], axis=1)) * 100.0)
        metrics["top1_agreement_pct"] = round(top1_agree, 2)
        metrics["max_logit_diff"] = round(float(np.max(np.abs(pol_np - out_ov[0]))), 6)
        metrics["max_value_diff"] = round(float(np.max(np.abs(val_np - out_ov[1]))), 6)

    for _ in range(10):
        req.infer({"board_state": sample})
    t0 = time.perf_counter()
    for _ in range(bench_iters):
        req.infer({"board_state": sample})
    dt = time.perf_counter() - t0
    metrics["pos_per_sec"] = round((bench_iters * batch_size) / max(1e-9, dt), 1)
    metrics["ms_per_batch"] = round((dt / bench_iters) * 1000.0, 3)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NeuraChess checkpoint or folded NPZ to OpenVINO IR")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pt", type=Path, help="PyTorch checkpoint path (.pt)")
    group.add_argument("--npz", type=Path, help="Pre-folded NumPy archive path (.npz)")
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--channels", type=int, default=192)
    parser.add_argument("--no-bn", action="store_true")
    parser.add_argument("--batch-size", type=int, default=0, help="0 exports dynamic batch (-1), >0 exports static B")
    parser.add_argument("--out-xml", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, default=None, help="Optional path to save BN-folded .npz when exporting .pt")
    parser.add_argument("--quant", choices=["fp16", "w8a16_asym"], default="fp16")
    parser.add_argument("--verify-device", type=str, default=None, help="Optional device (NPU or CPU) to benchmark/verify parity")
    args = parser.parse_args()

    pt_model = None
    if args.pt is not None:
        ckpt = torch.load(args.pt, map_location="cpu", weights_only=False)
        sd = ckpt.get("ema_09999", ckpt.get("ema_09995", ckpt.get("model_state_dict", ckpt)))
        pt_model = ChessSEResNet(
            num_blocks=args.blocks, channels=args.channels, value_dim=1, policy_head="conv73", use_bn=not args.no_bn
        )
        pt_model.load_state_dict(sd, strict=True)
        pt_model.eval()

        npz_out = args.out_npz or args.out_xml.with_suffix(".npz")
        npz_out.parent.mkdir(parents=True, exist_ok=True)
        pt_model.export_to_npz(str(npz_out))
        with np.load(npz_out) as npz:
            weights_dict = {k: npz[k] for k in npz.files}
        blocks, channels = args.blocks, args.channels
    else:
        with np.load(args.npz) as npz:
            weights_dict = {k: npz[k] for k in npz.files}
        blocks, channels, _, _ = infer_arch_from_npz(weights_dict)

    ov_model = build_openvino_se_resnet(
        batch_size=args.batch_size if args.batch_size > 0 else -1,
        num_blocks=blocks,
        channels=channels,
        in_channels=18,
        weights_dict=weights_dict,
        model_name=f"NeuraChess_{blocks}x{channels}",
    )

    if args.quant == "w8a16_asym":
        import nncf

        ov_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)

    ov_model.set_rt_info("Musheer Alam (https://github.com/Musheer360/NeuraChess)", "author")
    ov_model.set_rt_info("GPL-3.0", "license")

    args.out_xml.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(args.out_xml), compress_to_fp16=(args.quant == "fp16"))
    print(f"[Export] Saved OpenVINO IR ({args.quant}) -> {args.out_xml}")

    if args.verify_device:
        bench_b = args.batch_size if args.batch_size > 0 else 16
        report = verify_ir_parity(ov_model, pt_model, device=args.verify_device, batch_size=bench_b)
        print(f"[Verify] {report}")


if __name__ == "__main__":
    main()
