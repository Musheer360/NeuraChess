# Copyright (c) 2026 Musheer Alam. Licensed under GPL-3.0.
"""
NeuraChess OpenVINO NPU Compilation & Inference Wrapper.

Targets Intel AI Boost NPU (via OpenVINO Runtime) with fallback support for CPU/GPU,
persistent compiled-blob caching, LATENCY/THROUGHPUT execution hints, and thread-safe
`InferRequest` cloning.
"""

import logging
import time
from pathlib import Path
import numpy as np
import openvino as ov

logger = logging.getLogger(__name__)

from .resnet_backbone import build_openvino_resnet
from .se_resnet_backbone import build_openvino_se_resnet

from neurachess._paths import find_repo_root

ROOT_DIR = find_repo_root()
WEIGHTS_DIR = ROOT_DIR / "weights"
DEFAULT_CACHE_DIR = str(WEIGHTS_DIR / "cache")
DEFAULT_MODEL_XML = str(WEIGHTS_DIR / "neurachess-10x192-fp16.xml")
DEFAULT_WEIGHTS_NPZ = str(WEIGHTS_DIR / "neurachess-10x192-fp16.npz")
FALLBACK_MODEL_XML = str(WEIGHTS_DIR / "neurachess-10x128-fp16.xml")
FALLBACK_WEIGHTS_NPZ = str(WEIGHTS_DIR / "neurachess-10x128-fp16.npz")


def infer_arch_from_npz(weights_dict: dict[str, np.ndarray]) -> tuple[int, int, int, bool]:
    """Infer `(num_blocks, channels, input_planes, use_se)` from folded `.npz` keys."""
    init_w = weights_dict["init_conv_w"]
    channels = int(init_w.shape[0])
    input_planes = int(init_w.shape[1])
    block_indices = {
        int(k.split("_")[1])
        for k in weights_dict
        if k.startswith("res_") and k.endswith("_conv1_w")
    }
    num_blocks = (max(block_indices) + 1) if block_indices else 10
    use_se = any(k.startswith("res_") and "_se_" in k for k in weights_dict)
    return num_blocks, channels, input_planes, use_se


def resolve_default_model_paths() -> tuple[str | None, str | None]:
    """Return `(xml_path, npz_path)` for the highest-priority available local model."""
    for xml_p, npz_p in (
        (DEFAULT_MODEL_XML, DEFAULT_WEIGHTS_NPZ),
        (FALLBACK_MODEL_XML, FALLBACK_WEIGHTS_NPZ),
    ):
        xml_exists = Path(xml_p).exists()
        npz_exists = Path(npz_p).exists()
        if xml_exists or npz_exists:
            return (xml_p if xml_exists else None), (npz_p if npz_exists else None)
    return None, None


class NPUModel:
    """
    OpenVINO Model compiled to Intel AI Boost NPU (or CPU/GPU).

    When invoked with no explicit weights/IR/architecture arguments, resolves
    `weights/neurachess-10x192-fp16.{xml,npz}` (falling back to
    `weights/neurachess-10x128-fp16.{xml,npz}`) in `LATENCY` mode with
    `NPU_TURBO` enabled.
    """

    def __init__(
        self,
        batch_size: int = 16,
        cache_dir: str | None = None,
        weights_dict: dict | None = None,
        device: str = "NPU",
        use_se: bool = True,
        num_blocks: int | None = None,
        channels: int | None = None,
        use_turbo: bool = True,
        ov_model_override=None,
        performance_hint: str = "LATENCY",
        ov_ir_xml_path: str | None = None,
        verbose: bool = True,
    ):
        use_default_model = (
            weights_dict is None
            and ov_model_override is None
            and ov_ir_xml_path is None
            and num_blocks is None
            and channels is None
        )
        if use_default_model:
            default_xml, default_npz = resolve_default_model_paths()
            if default_xml is not None:
                ov_ir_xml_path = default_xml
            elif default_npz is not None:
                with np.load(default_npz) as npz:
                    weights_dict = {k: npz[k] for k in npz.files}
                inferred_blocks, inferred_ch, _, inferred_se = infer_arch_from_npz(weights_dict)
                num_blocks = inferred_blocks
                channels = inferred_ch
                use_se = inferred_se

        if weights_dict is not None and (num_blocks is None or channels is None):
            inferred_blocks, inferred_ch, _, inferred_se = infer_arch_from_npz(weights_dict)
            if num_blocks is None:
                num_blocks = inferred_blocks
            if channels is None:
                channels = inferred_ch
            use_se = inferred_se

        num_blocks = 10 if num_blocks is None else int(num_blocks)
        channels = 192 if channels is None else int(channels)
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR

        self.batch_size = batch_size
        self.device = device
        self.cache_dir = cache_dir
        self.use_se = use_se
        self.use_turbo = use_turbo
        self.performance_hint = str(performance_hint).upper()
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

        self.core = ov.Core()
        if self.cache_dir:
            self.core.set_property(self.device, {"CACHE_DIR": self.cache_dir})

        hint_enum = (
            ov.properties.hint.PerformanceMode.LATENCY
            if self.performance_hint == "LATENCY"
            else ov.properties.hint.PerformanceMode.THROUGHPUT
        )
        self.core.set_property(self.device, {"PERFORMANCE_HINT": hint_enum})
        if self.device == "NPU":
            try:
                self.core.set_property(self.device, {"NPU_TURBO": bool(self.use_turbo)})
            except (RuntimeError, ValueError):
                logger.debug("NPU_TURBO property not accepted by OpenVINO device; continuing without override", exc_info=True)
        self.core.set_property(self.device, {"INFERENCE_PRECISION_HINT": ov.Type.f16})

        if ov_model_override is not None:
            m = ov_model_override.clone()
            ps = m.input(0).get_partial_shape()
            in_ch = int(ps[1].get_length()) if ps[1].is_static else 18
            m.reshape({m.input(0).get_any_name(): [self.batch_size, in_ch, 8, 8]})
            self.ov_model = m
        elif ov_ir_xml_path is not None:
            m = self.core.read_model(str(ov_ir_xml_path))
            ps = m.input(0).get_partial_shape()
            in_ch = int(ps[1].get_length()) if ps[1].is_static else 18
            m.reshape({m.input(0).get_any_name(): [self.batch_size, in_ch, 8, 8]})
            self.ov_model = m
        else:
            if verbose:
                arch_kind = "SE-ResNet" if self.use_se else "ResNet"
                print(f"Building NeuraChess {arch_kind} (Batch {self.batch_size}, {num_blocks}x{channels})...")
            if self.use_se:
                self.ov_model = build_openvino_se_resnet(
                    batch_size=self.batch_size,
                    num_blocks=num_blocks,
                    channels=channels,
                    weights_dict=weights_dict,
                )
            else:
                self.ov_model = build_openvino_resnet(
                    batch_size=self.batch_size,
                    num_blocks=num_blocks,
                    channels=channels,
                    weights_dict=weights_dict,
                )

        t0 = time.perf_counter()
        self.compiled_model = self.core.compile_model(self.ov_model, self.device)
        self.compile_time_s = time.perf_counter() - t0
        if verbose:
            print(f"Compilation to {self.device} complete in {self.compile_time_s:.2f}s.")

        self.infer_request = self.compiled_model.create_infer_request()
        self.infer_request_async = self.compiled_model.create_infer_request()

    def clone(self) -> "NPUModel":
        """Create a lightweight worker instance with independent InferRequests sharing one compiled model."""
        clone_m = NPUModel.__new__(NPUModel)
        clone_m.batch_size = self.batch_size
        clone_m.device = self.device
        clone_m.cache_dir = self.cache_dir
        clone_m.use_se = self.use_se
        clone_m.use_turbo = getattr(self, "use_turbo", True)
        clone_m.performance_hint = getattr(self, "performance_hint", "LATENCY")
        clone_m.compile_time_s = 0.0
        clone_m.core = self.core
        clone_m.ov_model = self.ov_model
        clone_m.compiled_model = self.compiled_model
        clone_m.infer_request = self.compiled_model.create_infer_request()
        clone_m.infer_request_async = self.compiled_model.create_infer_request()
        return clone_m

    def infer(self, board_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference on compiled OpenVINO model.

        Args:
            board_tensor: `np.ndarray` of shape `(B, 18, 8, 8)` or `(18, 8, 8)`.
        Returns:
            `(policy_logits, value)` of shapes `(B, 4672)` and `(B, 1)`.
        """
        if board_tensor.ndim == 3:
            board_tensor = np.expand_dims(board_tensor, axis=0)

        if board_tensor.shape[0] != self.batch_size:
            raise ValueError(f"Expected batch size {self.batch_size}, got {board_tensor.shape[0]}")

        res = self.infer_request.infer({"board_state": np.ascontiguousarray(board_tensor, dtype=np.float32)})
        outputs = list(res.values())
        return outputs[0], outputs[1]

    def __call__(self, board_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.infer(board_tensor)
