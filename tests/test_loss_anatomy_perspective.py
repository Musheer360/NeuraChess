"""Unit tests verifying side-to-move perspective conversion in loss anatomy analysis."""

import shutil
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate import SF_PATH, StockfishAnalyzer, eval_stm_to_nc_drop, verify_perspective_sanity


def _has_stockfish_binary() -> bool:
    p = Path(SF_PATH)
    return p.exists() or shutil.which(SF_PATH) is not None


class TestLossAnatomyPerspective(unittest.TestCase):
    def test_perspective_sign_formula_synthetic(self) -> None:
        # Equal alternatives: White +15 cp before move, Black -10 cp after -> +10 cp White -> 5 cp drop.
        before, after, drop = eval_stm_to_nc_drop(15, -10)
        self.assertEqual(before, 15)
        self.assertEqual(after, 10)
        self.assertEqual(drop, 5)

        # Genuine blunder: White +50 cp before move, Black +250 cp after -> -250 cp White -> 300 cp drop.
        before_b, after_b, drop_b = eval_stm_to_nc_drop(50, 250)
        self.assertEqual(before_b, 50)
        self.assertEqual(after_b, -250)
        self.assertEqual(drop_b, 300)

    @unittest.skipUnless(_has_stockfish_binary(), "Stockfish binary not found on PATH or STOCKFISH_PATH")
    def test_startpos_d4_e4_perspective_delta(self) -> None:
        sf = StockfishAnalyzer(SF_PATH)
        try:
            res = verify_perspective_sanity(sf, depth=14)
        finally:
            sf.close()
        self.assertLess(res["d4_vs_e4_delta_cp"], 30)
        self.assertLess(res["d4_drop_vs_best_cp"], 30)
        self.assertLess(res["c5_drop_cp"], 30)


if __name__ == "__main__":
    unittest.main()
