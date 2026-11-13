"""Unit tests verifying CLI argparse flags and boolean semantics across entrypoints."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from search.npu_batched_puct import NPUBatchedPUCTSearch


class _DummyModel:
    batch_size = 1

    def infer(self, x):
        import numpy as np

        b = x.shape[0]
        return np.zeros((b, 4672), dtype=np.float32), np.zeros((b, 1), dtype=np.float32)


class TestBooleanArgparse(unittest.TestCase):
    def test_cli_entrypoints_help_exit_zero(self) -> None:
        cli_scripts = [
            ROOT_DIR / "scripts" / "evaluate.py",
            ROOT_DIR / "scripts" / "match_runner.py",
            ROOT_DIR / "scripts" / "export_openvino.py",
            ROOT_DIR / "scripts" / "download_weights.py",
            ROOT_DIR / "training" / "train.py",
            ROOT_DIR / "training" / "build_dataset.py",
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
