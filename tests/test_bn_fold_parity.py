"""Unit tests verifying BatchNorm-folded NPZ export parity against OpenVINO IR."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openvino as ov
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.pytorch_se_resnet import ChessSEResNet
from models.se_resnet_backbone import build_openvino_se_resnet


class TestBNFoldParity(unittest.TestCase):
    def test_bn_folded_npz_matches_openvino_cpu(self) -> None:
        torch.manual_seed(123)
        pt_model = ChessSEResNet(num_blocks=2, channels=32, value_dim=1, policy_head="conv73", use_bn=True)

        # Populate non-trivial running BN stats and affine parameters before folding.
        pt_model.train()
        with torch.no_grad():
            for _ in range(4):
                pt_model(torch.randn(4, 18, 8, 8) * 1.5 + 0.3)
            for mod in pt_model.modules():
                if isinstance(mod, torch.nn.BatchNorm2d):
                    mod.weight.uniform_(0.6, 1.4)
                    mod.bias.uniform_(-0.3, 0.3)
        pt_model.eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            npz_path = Path(tmp_dir) / "folded.npz"
            pt_model.export_to_npz(str(npz_path))
            with np.load(npz_path) as npz:
                weights_dict = {k: npz[k] for k in npz.files}

        self.assertIn("init_conv_b", weights_dict)
        self.assertIn("res_0_conv1_b", weights_dict)
        self.assertIn("pol_73_b", weights_dict)

        ov_model = build_openvino_se_resnet(
            batch_size=4,
            num_blocks=2,
            channels=32,
            in_channels=18,
            weights_dict=weights_dict,
            model_name="TestFoldParity",
        )
        compiled = ov.Core().compile_model(ov_model, "CPU")

        rng = np.random.default_rng(7)
        x_np = rng.normal(size=(4, 18, 8, 8)).astype(np.float32)
        with torch.no_grad():
            pol_pt, val_pt = pt_model(torch.from_numpy(x_np))

        outputs = compiled({"board_state": x_np})
        pol_ov = np.asarray(outputs[compiled.output(0)], dtype=np.float32)
        val_ov = np.asarray(outputs[compiled.output(1)], dtype=np.float32)

        np.testing.assert_allclose(pol_pt.numpy(), pol_ov, atol=2e-4)
        np.testing.assert_allclose(val_pt.numpy(), val_ov, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
