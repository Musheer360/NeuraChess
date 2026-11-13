"""Unit tests for Conv73 policy head, BatchNorm folding, Boltzmann soft loss, EMA, and checkpoints."""

import sys
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    nn = None
    F = None

from neurachess.core.constants import move_to_action
if torch is not None:
    from neurachess.models.pytorch_se_resnet import ChessSEResNet, ModelEMA, fold_conv_bn
else:
    ChessSEResNet = ModelEMA = fold_conv_bn = None


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestConv73Trainer(unittest.TestCase):
    def test_conv73_action_mapping_10000(self) -> None:
        t = torch.zeros(1, 73, 8, 8, dtype=torch.float32)
        for from_r in range(8):
            for from_c in range(8):
                from_sq = from_r * 8 + from_c
                for slot in range(73):
                    t[0, slot, from_r, from_c] = from_sq * 73 + slot

        flat = t.permute(0, 2, 3, 1).contiguous().reshape(1, 4672)[0]
        for idx in range(4672):
            self.assertEqual(int(flat[idx].item()), idx)

        rng = np.random.default_rng(42)
        verified_moves = 0
        board = chess.Board()
        while verified_moves < 10000:
            if board.is_game_over():
                board = chess.Board()
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                board = chess.Board()
                continue
            for m in legal_moves:
                act_idx = move_to_action(m, canonical_flip=(board.turn == chess.BLACK))
                self.assertGreaterEqual(act_idx, 0)
                self.assertLess(act_idx, 4672)
                from_sq = act_idx // 73
                slot = act_idx % 73
                self.assertEqual(int(t[0, slot, from_sq // 8, from_sq % 8].item()), act_idx)
                verified_moves += 1
                if verified_moves >= 10000:
                    break
            board.push(legal_moves[rng.integers(len(legal_moves))])

    def test_conv73_forward_backward(self) -> None:
        for use_bn in (False, True):
            model = ChessSEResNet(num_blocks=2, channels=32, policy_head="conv73", use_bn=use_bn)
            model.train()
            x = torch.randn(4, 18, 8, 8, requires_grad=True)
            pol, val = model(x)
            self.assertEqual(pol.shape, (4, 4672))
            self.assertEqual(val.shape, (4, 1))
            target_act = torch.tensor([10, 200, 1500, 4000], dtype=torch.long)
            target_val = torch.tensor([[0.5], [-0.2], [0.0], [0.9]], dtype=torch.float32)
            loss = F.cross_entropy(pol, target_act) + F.mse_loss(val, target_val)
            loss.backward()
            self.assertIsNotNone(model.pol_conv.weight.grad)
            self.assertIsNotNone(x.grad)

    def test_conv73_batchnorm_folding(self) -> None:
        model = ChessSEResNet(num_blocks=2, channels=32, policy_head="conv73", use_bn=True)
        model.eval()
        w_folded, b_folded = fold_conv_bn(model.pol_conv, model.pol_bn)
        self.assertEqual(w_folded.shape, (73, 32, 1, 1))
        self.assertEqual(b_folded.shape, (73,))

        h = torch.randn(2, 32, 8, 8)
        with torch.no_grad():
            out_orig = model.pol_bn(model.pol_conv(h))
            conv_folded = nn.Conv2d(32, 73, kernel_size=1, bias=True)
            conv_folded.weight.data = torch.from_numpy(w_folded)
            conv_folded.bias.data = torch.from_numpy(b_folded)
            out_folded = conv_folded(h)
            diff = float((out_orig - out_folded).abs().max().item())
        self.assertLess(diff, 1e-4)

    def test_topk_boltzmann_soft_loss(self) -> None:
        moves = [100, 250, 400]
        cps = [120, 80, -30]
        temperature = 0.5
        q_vals = [1.0 / (1.0 + 10.0 ** (-cp / 400.0)) for cp in cps]
        exp_q = np.exp(np.array(q_vals) / temperature)
        soft_probs = exp_q / np.sum(exp_q)

        logits = torch.randn(1, 4672)
        log_probs = F.log_softmax(logits, dim=-1)
        soft_loss = -sum(soft_probs[i] * log_probs[0, moves[i]] for i in range(len(moves)))
        self.assertGreater(float(soft_loss.item()), 0.0)

        ce_k1 = F.cross_entropy(logits, torch.tensor([moves[0]], dtype=torch.long))
        diff = float((-log_probs[0, moves[0]] - ce_k1).abs().item())
        self.assertLess(diff, 1e-5)

    def test_model_ema_and_resume(self) -> None:
        model = ChessSEResNet(num_blocks=2, channels=32, policy_head="conv73", use_bn=False)
        ema = ModelEMA(model, decay=0.9)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        ema.update(model)
        diff = float((ema.shadow["init_conv.weight"] - model.init_conv.weight).abs().max().item())
        self.assertGreater(diff, 0.8)
        self.assertLess(diff, 1.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "checkpoint.pt"
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
            x = torch.randn(2, 18, 8, 8)
            pol, val = model(x)
            (pol.sum() + val.sum()).backward()
            opt.step()
            torch.save(
                {"step": 42, "model_state": model.state_dict(), "opt_state": opt.state_dict()},
                ckpt_path,
            )

            model2 = ChessSEResNet(num_blocks=2, channels=32, policy_head="conv73", use_bn=False)
            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model2.load_state_dict(loaded["model_state"])
            model.eval()
            model2.eval()
            with torch.no_grad():
                p1, v1 = model(x)
                p2, v2 = model2(x)
            self.assertTrue(torch.allclose(p1, p2, atol=1e-6))
            self.assertTrue(torch.allclose(v1, v2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
