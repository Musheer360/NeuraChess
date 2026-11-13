"""Monte-Carlo validation for bidirectional pair-level pentanomial SPRT."""

import sys
import unittest
from pathlib import Path

import numpy as np

from neurachess.scripts.match_runner import compute_pentanomial_sprt_llr, elo_to_score


def _pentanomial_probs(true_elo: float, draw_rate: float = 0.40) -> tuple[np.ndarray, float]:
    p = elo_to_score(true_elo)
    effective_draw = min(draw_rate, 2.0 * min(p, 1.0 - p) * 0.90)
    w = p - 0.5 * effective_draw
    d = effective_draw
    l = (1.0 - p) - 0.5 * effective_draw
    probs = np.array(
        [
            l * l,
            2.0 * l * d,
            d * d + 2.0 * w * l,
            2.0 * w * d,
            w * w,
        ],
        dtype=np.float64,
    )
    probs /= probs.sum()
    return probs, float(np.sum(probs * np.array([0.0, 0.25, 0.50, 0.75, 1.0])))


def _run_mc(
    num_trials: int,
    true_elo: float,
    elo0: float = 0.0,
    elo1: float = 20.0,
    max_pairs: int = 300,
    seed: int = 101,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    probs, true_mean = _pentanomial_probs(true_elo)
    draws = rng.choice(5, size=(num_trials, max_pairs), p=probs)

    accept_h1 = accept_h0 = inconclusive = 0
    for row in draws:
        counts = [0, 0, 0, 0, 0]
        for idx in range(9):
            counts[int(row[idx])] += 1
        decided = False
        for idx in range(9, max_pairs):
            counts[int(row[idx])] += 1
            res = compute_pentanomial_sprt_llr(
                counts, elo0=elo0, elo1=elo1, alpha=0.05, beta=0.05, min_pairs_for_decision=10
            )
            if res["status"] == "ACCEPT_H1":
                accept_h1 += 1
                decided = True
                break
            if res["status"] == "ACCEPT_H0":
                accept_h0 += 1
                decided = True
                break
        if not decided:
            inconclusive += 1

    return {
        "true_mean": true_mean,
        "accept_h1_rate": accept_h1 / num_trials,
        "accept_h0_rate": accept_h0 / num_trials,
        "inconclusive_rate": inconclusive / num_trials,
    }


class TestSprtMonteCarlo(unittest.TestCase):
    def test_elo_to_score_symmetry(self) -> None:
        self.assertAlmostEqual(elo_to_score(0.0), 0.5, places=9)
        self.assertAlmostEqual(elo_to_score(20.0) + elo_to_score(-20.0), 1.0, places=9)

    def test_type1_error_under_null(self) -> None:
        res = _run_mc(num_trials=1500, true_elo=0.0, elo0=0.0, elo1=20.0, max_pairs=300, seed=101)
        self.assertLessEqual(res["accept_h1_rate"], 0.065)

    def test_asymptotic_power_under_alternative(self) -> None:
        res = _run_mc(num_trials=1000, true_elo=20.0, elo0=0.0, elo1=20.0, max_pairs=1500, seed=303)
        self.assertGreaterEqual(res["accept_h1_rate"], 0.88)
        self.assertLessEqual(res["accept_h0_rate"], 0.065)


if __name__ == "__main__":
    unittest.main()
