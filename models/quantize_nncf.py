"""
NNCF Quantization Helper for OpenVINO SE-ResNet Models.

Supports INT8 weight compression and calibration using bitboard shards or binary
datasets, targeting OpenVINO NPU/CPU execution. For full IR export + parity checks,
see `scripts/export_openvino.py`.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import openvino as ov
import nncf

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.npu_compiler import DEFAULT_WEIGHTS_NPZ, infer_arch_from_npz
from models.se_resnet_backbone import build_openvino_se_resnet


def quantize_se_resnet(
    weights_path: str,
    calibration_data_path: str | None = None,
    batch_size: int = 16,
    subset_size: int = 20,
    out_dir: str | None = None,
    device: str = "NPU",
) -> tuple[ov.CompiledModel, dict]:
    """Quantize a folded `.npz` SE-ResNet model via NNCF and benchmark against FP16."""
    os.environ["NNCF_TELEM_OPT_OUT"] = "1"
    os.environ["OPENVINO_TELEMETRY_OPTOUT"] = "1"

    out_path = Path(out_dir) if out_dir else (ROOT_DIR / "weights" / "quantized")
    out_path.mkdir(parents=True, exist_ok=True)

    with np.load(weights_path) as npz:
        weights_dict = {k: npz[k] for k in npz.files}
    num_blocks, channels, in_ch, _ = infer_arch_from_npz(weights_dict)

    fp_model = build_openvino_se_resnet(
        batch_size=batch_size,
        num_blocks=num_blocks,
        channels=channels,
        in_channels=in_ch,
        weights_dict=weights_dict,
    )

    calibration_batches: list[np.ndarray] = []
    if calibration_data_path and Path(calibration_data_path).is_file():
        from training.bitboard_dataset import unpack_position
        from core.board_tensor import board_to_tensor

        raw = Path(calibration_data_path).read_bytes()
        rec_size = 107 if len(raw) % 107 == 0 and len(raw) % 691 != 0 else 691
        n_total = len(raw) // rec_size
        n_calib = min(subset_size, max(1, n_total // batch_size))
        for i in range(n_calib):
            batch = np.zeros((batch_size, in_ch, 8, 8), dtype=np.float32)
            for j in range(batch_size):
                idx = (i * batch_size + j) % n_total
                rec = unpack_position(raw[idx * rec_size : idx * rec_size + 107])
                batch[j] = board_to_tensor(rec["board"], layout="gpu_unpack")
            calibration_batches.append(batch)
    else:
        rng = np.random.default_rng(42)
        for _ in range(subset_size):
            calibration_batches.append(rng.random((batch_size, in_ch, 8, 8), dtype=np.float32))

    calibration_dataset = nncf.Dataset(calibration_batches, lambda item: {"board_state": item})

    t0 = time.time()
    int8_model = nncf.quantize(
        fp_model,
        calibration_dataset,
        subset_size=len(calibration_batches),
        fast_bias_correction=True,
    )
    quant_time = time.time() - t0

    stem = Path(weights_path).stem
    ir_xml = out_path / f"{stem}_b{batch_size}_w8a8.xml"
    ov.save_model(int8_model, str(ir_xml))

    core = ov.Core()
    comp_fp = core.compile_model(fp_model, device)
    comp_int8 = core.compile_model(int8_model, device)

    dummy = np.zeros((batch_size, in_ch, 8, 8), dtype=np.float32)
    comp_fp([dummy])
    comp_int8([dummy])

    iters = 30
    t0 = time.perf_counter()
    for _ in range(iters):
        comp_fp([dummy])
    fp_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iters):
        comp_int8([dummy])
    int8_dt = time.perf_counter() - t0

    fp_pos_s = (iters * batch_size) / max(1e-9, fp_dt)
    int8_pos_s = (iters * batch_size) / max(1e-9, int8_dt)
    metrics = {
        "batch_size": batch_size,
        "quant_time_s": round(quant_time, 2),
        "fp16_ms": round((fp_dt / iters) * 1000.0, 2),
        "fp16_pos_s": round(fp_pos_s, 1),
        "int8_ms": round((int8_dt / iters) * 1000.0, 2),
        "int8_pos_s": round(int8_pos_s, 1),
        "speedup_pct": round(((int8_pos_s - fp_pos_s) / fp_pos_s) * 100.0, 2),
        "ir_xml": str(ir_xml),
    }
    return comp_int8, metrics


def main():
    parser = argparse.ArgumentParser(description="Quantize a folded NeuraChess .npz model with NNCF")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_NPZ, help="Path to folded .npz weights")
    parser.add_argument("--calib-data", type=str, default=None, help="Optional .bin calibration file (107B or 691B)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--subset-size", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="NPU")
    args = parser.parse_args()

    _, metrics = quantize_se_resnet(
        weights_path=args.weights,
        calibration_data_path=args.calib_data,
        batch_size=args.batch_size,
        subset_size=args.subset_size,
        out_dir=args.out_dir,
        device=args.device,
    )
    print(metrics)


if __name__ == "__main__":
    main()
