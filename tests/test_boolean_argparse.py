"""Unit tests verifying CLI argparse flags and boolean semantics across entrypoints."""

import subprocess
import sys
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch

ROOT_DIR = Path(__file__).resolve().parent.parent


class _DummyModel:
    batch_size = 1

    def infer(self, x):
        import numpy as np

        b = x.shape[0]
        return np.zeros((b, 4672), dtype=np.float32), np.zeros((b, 1), dtype=np.float32)


class TestBooleanArgparse(unittest.TestCase):
    def test_cli_entrypoints_help_exit_zero(self) -> None:
        cli_scripts = [
            ROOT_DIR / "src" / "neurachess" / "scripts" / "evaluate.py",
            ROOT_DIR / "src" / "neurachess" / "scripts" / "match_runner.py",
            ROOT_DIR / "src" / "neurachess" / "scripts" / "export_openvino.py",
            ROOT_DIR / "src" / "neurachess" / "scripts" / "download_weights.py",
            ROOT_DIR / "src" / "neurachess" / "training" / "build_dataset.py",
        ]
        for script_path in cli_scripts:
            proc = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"{script_path.name} --help failed:\n{proc.stderr}")

        for mod_name in ("neurachess.scripts.evaluate", "neurachess.scripts.export_openvino", "neurachess.uci"):
            proc = subprocess.run(
                [sys.executable, "-m", mod_name, "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"-m {mod_name} --help failed:\n{proc.stderr}")

    @unittest.skipIf(torch is None, "PyTorch not installed")
    def test_train_cli_help_exit_zero(self) -> None:
        train_script = ROOT_DIR / "src" / "neurachess" / "training" / "train.py"
        proc = subprocess.run(
            [sys.executable, str(train_script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, f"{train_script.name} --help failed:\n{proc.stderr}")

    def test_search_boolean_flags_do_not_coerce_truthy_strings(self) -> None:
        s = NPUBatchedPUCTSearch(
            root_model=_DummyModel(),
            batch_model=_DummyModel(),
            batch_size=16,
            use_fpu_reduction=False,
            use_exact_solver=False,
            use_tree_reuse=False,
            use_nn_cache=False,
            use_fast_tree=False,
            use_opening_root_guard=False,
        )
        self.assertIs(s.use_fpu_reduction, False)
        self.assertIs(s.use_exact_solver, False)
        self.assertIs(s.use_tree_reuse, False)
        self.assertIs(s.use_nn_cache, False)
        self.assertIs(s.use_fast_tree, False)
        self.assertIs(s.use_opening_root_guard, False)


if __name__ == "__main__":
    unittest.main()
