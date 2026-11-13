"""
NeuraChess Unified Evaluation CLI (`neurachess.scripts.evaluate`).

Subcommands:
  - `dataset`: Offline evaluation on 107B or 691B `.bin` test splits (masked CE,
    legal-only CE, Top-1, Top-3, Value MSE, decisive sign agreement).
  - `puzzles`: Tactical puzzle evaluation across policy-only (1-node) and PUCT node budgets.
  - `loss-anatomy`: First-blunder game analysis via fixed-depth Stockfish + policy prior rank.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

import chess
import numpy as np
import openvino as ov

if TYPE_CHECKING:
    import torch

from neurachess._paths import find_repo_root

ROOT_DIR = find_repo_root()

from neurachess.core.board_tensor import board_to_tensor
from neurachess.core.constants import action_to_uci, uci_to_action
from neurachess.core.tensor_movegen import get_legal_action_mask
from neurachess.models.npu_compiler import NPUModel, resolve_default_model_paths
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch
from neurachess.training.bitboard_dataset import RECORD_SIZE_V1, unpack_position

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def resolve_stockfish_path(explicit_path: str | None = None) -> str:
    if explicit_path and Path(explicit_path).exists():
        return explicit_path
    env_p = os.environ.get("STOCKFISH_PATH")
    if env_p and Path(env_p).exists():
        return env_p
    which_p = shutil.which("stockfish")
    if which_p:
        return which_p
    for cand in (
        ROOT_DIR / "bin" / "stockfish" / "stockfish-windows-x86-64-universal.exe",
        ROOT_DIR / "bin" / "stockfish" / "stockfish",
    ):
        if cand.exists():
            return str(cand)
    return "stockfish"


SF_PATH = resolve_stockfish_path()


def compute_material(board: chess.Board, color: chess.Color) -> int:
    return sum(PIECE_VALUES.get(p.piece_type, 0) for p in board.piece_map().values() if p.color == color)


class StockfishAnalyzer:
    """Thin persistent UCI interface for fixed-depth positional evaluation."""

    def __init__(self, sf_path: str | None = None):
        binary = resolve_stockfish_path(sf_path)
        self.proc = subprocess.Popen(
            [binary],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        while "uciok" not in self.proc.stdout.readline():
            pass
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 32")
        self._send("setoption name UCI_LimitStrength value false")
        self._send("isready")
        while "readyok" not in self.proc.stdout.readline():
            pass

    def _send(self, cmd: str) -> None:
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def eval_position(self, moves_history: list[str], depth: int = 14) -> tuple[str | None, int]:
        """Return `(best_move_uci, eval_cp_stm)` from the side-to-move perspective."""
        move_str = " ".join(moves_history)
        self._send(f"position startpos moves {move_str}" if move_str else "position startpos")
        self._send(f"go depth {depth}")

        best_m = None
        score_cp = 0
        while True:
            line = self.proc.stdout.readline().strip()
            if "score cp" in line:
                parts = line.split()
                if "cp" in parts:
                    score_cp = int(parts[parts.index("cp") + 1])
            elif "score mate" in line:
                parts = line.split()
                if "mate" in parts:
                    m_val = int(parts[parts.index("mate") + 1])
                    score_cp = 30000 if m_val > 0 else -30000
            if line.startswith("bestmove"):
                parts = line.split()
                best_m = parts[1] if len(parts) > 1 else None
                break
        return best_m, score_cp

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=2)
        except (OSError, subprocess.SubprocessError, BrokenPipeError, ValueError):
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                logger.debug("Stockfish process termination fallback wait failed", exc_info=True)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                logger.debug("Stockfish stream close failed", exc_info=True)


def eval_stm_to_nc_drop(eval_stm_before_cp: int, eval_stm_after_cp: int) -> tuple[int, int, int]:
    """Convert side-to-move Stockfish evaluations before and after a move to mover perspective."""
    eval_nc_before = int(eval_stm_before_cp)
    eval_nc_after = -int(eval_stm_after_cp)
    return eval_nc_before, eval_nc_after, eval_nc_before - eval_nc_after


def verify_perspective_sanity(sf: StockfishAnalyzer, depth: int = 14) -> dict[str, int | str]:
    """Assert book moves from startpos (`1. d4` vs `1. e4`, `1...c5`) have < 30 cp eval drop."""
    best_start, eval_stm_start = sf.eval_position([], depth=depth)
    _, eval_stm_after_e4 = sf.eval_position(["e2e4"], depth=depth)
    _, eval_stm_after_d4 = sf.eval_position(["d2d4"], depth=depth)
    eval_white_e4 = eval_stm_to_nc_drop(eval_stm_start, eval_stm_after_e4)[1]
    eval_white_d4 = eval_stm_to_nc_drop(eval_stm_start, eval_stm_after_d4)[1]
    d4_vs_e4_delta = abs(eval_white_e4 - eval_white_d4)
    d4_drop_vs_best = abs(eval_stm_to_nc_drop(eval_stm_start, eval_stm_after_d4)[2])
    assert d4_vs_e4_delta < 30, f"Perspective bug: 1. d4 vs 1. e4 delta={d4_vs_e4_delta} cp"
    assert d4_drop_vs_best < 30, f"Perspective bug: 1. d4 drop={d4_drop_vs_best} cp"

    _, eval_stm_b_before = sf.eval_position(["e2e4"], depth=depth)
    _, eval_stm_b_after = sf.eval_position(["e2e4", "c7c5"], depth=depth)
    c5_drop = abs(eval_stm_to_nc_drop(eval_stm_b_before, eval_stm_b_after)[2])
    assert c5_drop < 30, f"Perspective bug on 1...c5: drop={c5_drop} cp"
    return {
        "best_start_move": best_start or "",
        "eval_white_e4_cp": eval_white_e4,
        "eval_white_d4_cp": eval_white_d4,
        "d4_vs_e4_delta_cp": d4_vs_e4_delta,
        "d4_drop_vs_best_cp": d4_drop_vs_best,
        "c5_drop_cp": c5_drop,
    }


def unpack_raw_to_gpu_inputs(raw_bytes: bytes, num_records: int) -> torch.Tensor:
    import torch

    raw_arr = np.frombuffer(raw_bytes[: num_records * RECORD_SIZE_V1], dtype=np.uint8).reshape(num_records, RECORD_SIZE_V1)
    bbs = torch.from_numpy(raw_arr[:, :96].copy().view(np.int64))
    shifts = torch.arange(64, dtype=torch.int64)
    planes_12 = ((bbs.unsqueeze(-1) >> shifts) & 1).float().reshape(num_records, 12, 8, 8)
    turn = torch.from_numpy(raw_arr[:, 96].astype(np.float32)).view(num_records, 1, 1, 1).expand(num_records, 1, 8, 8)
    castling = raw_arr[:, 97].astype(np.int64)
    c_wk = torch.from_numpy(((castling >> 0) & 1).astype(np.float32)).view(num_records, 1, 1, 1).expand(num_records, 1, 8, 8)
    c_wq = torch.from_numpy(((castling >> 1) & 1).astype(np.float32)).view(num_records, 1, 1, 1).expand(num_records, 1, 8, 8)
    c_bk = torch.from_numpy(((castling >> 2) & 1).astype(np.float32)).view(num_records, 1, 1, 1).expand(num_records, 1, 8, 8)
    c_bq = torch.from_numpy(((castling >> 3) & 1).astype(np.float32)).view(num_records, 1, 1, 1).expand(num_records, 1, 8, 8)
    halfmove = (torch.from_numpy(raw_arr[:, 99].astype(np.float32)) / 100.0).view(num_records, 1, 1, 1).expand(num_records, 1, 8, 8)
    return torch.cat([planes_12, turn, c_wk, c_wq, c_bk, c_bq, halfmove], dim=1)


def load_bitboard_eval_dataset(bin_path: Path, max_records: int = 50000, cache_dir: Path | None = None) -> dict:
    import torch

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{bin_path.stem}_{max_records}.pt"
        if cache_path.exists():
            return torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        cache_path = None

    raw_data = bin_path.read_bytes()[: max_records * RECORD_SIZE_V1]
    n_rec = len(raw_data) // RECORD_SIZE_V1
    canonical_tensors, masks, acts, vals, raw_cps = [], [], [], [], []
    for i in range(n_rec):
        raw_b = raw_data[i * RECORD_SIZE_V1 : (i + 1) * RECORD_SIZE_V1]
        rec = unpack_position(raw_b)
        b = rec["board"]
        is_black = b.turn == chess.BLACK
        act = rec["action_index"]
        mask = get_legal_action_mask(b)
        if mask[act] < 0.5:
            try:
                m = b.parse_uci(action_to_uci(act, canonical_flip=is_black))
                if m in b.legal_moves:
                    act = uci_to_action(m.uci(), canonical_flip=is_black)
            except (ValueError, KeyError, IndexError):
                logger.debug("Action fallback remapping skipped for action index %s", act, exc_info=True)
        canonical_tensors.append(board_to_tensor(b, layout="legacy_flip"))
        masks.append(mask)
        acts.append(act)
        cp = rec["eval_cp"]
        raw_cps.append(cp)
        vals.append(float(np.tanh(cp * 0.00287823)))

    ds = {
        "count": n_rec,
        "canonical_inputs": torch.from_numpy(np.array(canonical_tensors)),
        "gpu_inputs": unpack_raw_to_gpu_inputs(raw_data, n_rec),
        "masks": torch.from_numpy(np.array(masks)),
        "acts": torch.tensor(acts, dtype=torch.long),
        "vals": torch.tensor(vals, dtype=torch.float32),
        "cps": np.array(raw_cps),
    }
    if cache_path is not None:
        torch.save(ds, cache_path)
    return ds


def evaluate_model_on_data(
    model: torch.nn.Module,
    ds: dict,
    input_repr: str = "gpu_unpack",
    batch_size: int = 512,
    device: torch.device | None = None,
) -> dict:
    import torch
    import torch.nn.functional as F

    if device is None:
        device = torch.device("cpu")
    model.eval()
    inputs_all = ds["canonical_inputs"] if input_repr == "canonical" else ds["gpu_inputs"]
    n = ds["count"]

    total_masked_ce = total_unmasked_ce = total_legal_masked_ce = total_val_mse = 0.0
    legal_target_count = correct_top1 = correct_top3 = decisive_total = decisive_sign_correct = 0

    with torch.no_grad():
        for i in range(0, n, batch_size):
            b_sz = min(batch_size, n - i)
            b_inp = inputs_all[i : i + b_sz].to(device)
            b_mask = ds["masks"][i : i + b_sz].to(device)
            b_act = ds["acts"][i : i + b_sz].to(device)
            b_val = ds["vals"][i : i + b_sz].to(device)
            b_cps = ds["cps"][i : i + b_sz]

            logits, v_pred = model(b_inp)
            total_unmasked_ce += F.cross_entropy(logits, b_act, reduction="sum").item()
            masked_logits = logits + (1.0 - b_mask) * -10000.0
            total_masked_ce += F.cross_entropy(masked_logits, b_act, reduction="sum").item()

            is_legal = b_mask[torch.arange(b_sz, device=device), b_act] > 0.5
            n_leg = int(is_legal.sum().item())
            if n_leg > 0:
                total_legal_masked_ce += F.cross_entropy(masked_logits[is_legal], b_act[is_legal], reduction="sum").item()
                legal_target_count += n_leg

            correct_top1 += int((torch.argmax(masked_logits, dim=1) == b_act).sum().item())
            _, top3 = torch.topk(masked_logits, k=3, dim=1)
            correct_top3 += int((top3 == b_act.unsqueeze(1)).any(dim=1).sum().item())

            v_scalar = v_pred.squeeze(-1) if v_pred.shape[-1] == 1 else (v_pred[:, 0] - v_pred[:, 2])
            total_val_mse += ((v_scalar - b_val) ** 2).sum().item()
            v_p_np = v_scalar.cpu().numpy()
            for idx in range(b_sz):
                if abs(b_cps[idx]) >= 50:
                    decisive_total += 1
                    if (b_cps[idx] > 0 and v_p_np[idx] > 0) or (b_cps[idx] < 0 and v_p_np[idx] < 0):
                        decisive_sign_correct += 1

    return {
        "positions": n,
        "masked_ce": round(total_masked_ce / max(1, n), 4),
        "masked_ce_legal_only": round(total_legal_masked_ce / max(1, legal_target_count), 4),
        "illegal_targets": n - legal_target_count,
        "unmasked_ce": round(total_unmasked_ce / max(1, n), 4),
        "masked_top1_pct": round(100.0 * correct_top1 / max(1, n), 2),
        "masked_top3_pct": round(100.0 * correct_top3 / max(1, n), 2),
        "value_mse": round(total_val_mse / max(1, n), 4),
        "sign_agreement_pct": round(100.0 * decisive_sign_correct / max(1, decisive_total), 2),
        "decisive_count": decisive_total,
    }


def _load_torch_model_from_pt_or_npz(weights_path: Path, blocks: int, channels: int, use_bn: bool, device: torch.device) -> torch.nn.Module:
    import torch
    from neurachess.models.pytorch_se_resnet import ChessSEResNet

    if weights_path.suffix == ".npz":
        raise ValueError("Offline PyTorch dataset evaluation requires a .pt checkpoint; pass .xml/.npz for OpenVINO evaluation.")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    sd = ckpt.get("ema_09999", ckpt.get("ema_09995", ckpt.get("model_state_dict", ckpt)))
    model = ChessSEResNet(num_blocks=blocks, channels=channels, value_dim=1, policy_head="conv73", use_bn=use_bn)
    model.load_state_dict(sd, strict=True)
    return model.to(device)


def evaluate_puzzle_suite(
    puzzles: list[dict],
    xml_path: str | None = None,
    device: str = "NPU",
    node_budgets: list[int] | None = None,
    encoder_layout: str = "gpu_unpack",
) -> dict:
    if node_budgets is None:
        node_budgets = [1, 50, 200, 800]
    if xml_path is None:
        xml_path, _ = resolve_default_model_paths()

    root_m = NPUModel(batch_size=1, device=device, performance_hint="LATENCY", ov_ir_xml_path=xml_path, verbose=False)
    batch_m = NPUModel(batch_size=16, device=device, performance_hint="LATENCY", ov_ir_xml_path=xml_path, verbose=False)
    searcher = NPUBatchedPUCTSearch(
        root_model=root_m,
        batch_model=batch_m,
        batch_size=16,
        encoder_layout=encoder_layout,
        use_fast_tree=True,
        use_fpu_reduction=True,
        use_exact_solver=True,
        use_tree_reuse=False,
        use_nn_cache=True,
    )

    summary = {}
    for budget in node_budgets:
        solved = 0
        t0 = time.perf_counter()
        for pz in puzzles:
            board = chess.Board(pz["fen"])
            board.push_uci(pz["moves"][0])
            expected = pz["moves"][1]
            tensor = board_to_tensor(board, layout=encoder_layout)
            if budget <= 1:
                logits, _ = root_m.infer(tensor)
                legals = list(board.legal_moves)
                is_black = board.turn == chess.BLACK
                best_m = max(legals, key=lambda m: float(logits[0, uci_to_action(m.uci(), canonical_flip=is_black)]))
                uci = best_m.uci()
            else:
                searcher.clear_state()
                _, uci, _, _, _, _ = searcher.search(board, tensor, movetime_ms=10_000, max_nodes=budget)
            if uci == expected:
                solved += 1
        summary[f"nodes_{budget}"] = {
            "solved": solved,
            "total": len(puzzles),
            "accuracy_pct": round(100.0 * solved / max(1, len(puzzles)), 2),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }
    return summary


def cmd_dataset(args: argparse.Namespace) -> None:
    import torch

    device = torch.device(args.device)
    model = _load_torch_model_from_pt_or_npz(args.checkpoint, args.blocks, args.channels, not args.no_bn, device)
    cache_dir = ROOT_DIR / "eval_results" / "cache" if args.use_cache else None
    results = {}
    for ds_path in args.datasets:
        ds = load_bitboard_eval_dataset(ds_path, max_records=args.max_records, cache_dir=cache_dir)
        results[ds_path.name] = evaluate_model_on_data(model, ds, input_repr=args.layout, batch_size=args.batch_size, device=device)
    print(json.dumps(results, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


def cmd_puzzles(args: argparse.Namespace) -> None:
    raw = json.loads(args.puzzles_json.read_text(encoding="utf-8"))
    puzzles = raw if isinstance(raw, list) else raw.get("puzzles", [])
    if args.max_puzzles:
        puzzles = puzzles[: args.max_puzzles]
    res = evaluate_puzzle_suite(
        puzzles=puzzles,
        xml_path=str(args.xml) if args.xml else None,
        device=args.device,
        node_budgets=args.budgets,
        encoder_layout=args.layout,
    )
    print(json.dumps(res, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(res, indent=2), encoding="utf-8")


def cmd_loss_anatomy(args: argparse.Namespace) -> None:
    sf = StockfishAnalyzer(args.stockfish_path)
    try:
        sanity = verify_perspective_sanity(sf, depth=args.depth)
        print(f"[Sanity OK] {sanity}")
        games_payload = json.loads(args.games_json.read_text(encoding="utf-8"))
        games = games_payload.get("games", []) if isinstance(games_payload, dict) else games_payload
        losses = [g for g in games if g.get("nc_score", 1.0) == 0.0]

        xml_p = str(args.xml) if args.xml else resolve_default_model_paths()[0]
        ov_core = ov.Core()
        compiled = ov_core.compile_model(ov_core.read_model(xml_p), "CPU")
        out_layer = compiled.output(0)
        infer_req = compiled.create_infer_request()

        counts = {"a_search_value": 0, "b_policy_blindspot": 0, "c_tactical_horizon": 0, "d_endgame_grind": 0}
        analyzed = []
        for g in losses:
            moves = g.get("moves", [])
            nc_is_white = g.get("nc_color", "white") == "white"
            board = chess.Board()
            hist: list[str] = []
            for ply_idx, mv_uci in enumerate(moves):
                is_nc_turn = (board.turn == chess.WHITE) == nc_is_white
                if is_nc_turn:
                    sf_best, eval_b = sf.eval_position(hist, depth=args.depth)
                    _, eval_a = sf.eval_position(hist + [mv_uci], depth=args.depth)
                    _, _, drop = eval_stm_to_nc_drop(eval_b, eval_a)
                    if drop >= args.blunder_cp:
                        tensor = board_to_tensor(board, layout="gpu_unpack")[np.newaxis, ...]
                        logits = infer_req.infer([tensor])[out_layer][0]
                        is_blk = board.turn == chess.BLACK
                        top3 = sorted(
                            [(m.uci(), float(logits[uci_to_action(m.uci(), canonical_flip=is_blk)])) for m in board.legal_moves],
                            key=lambda x: x[1],
                            reverse=True,
                        )[:3]
                        in_top3 = sf_best in {m for m, _ in top3}
                        cat = "a_search_value" if in_top3 else "b_policy_blindspot"
                        counts[cat] += 1
                        analyzed.append({"ply": ply_idx + 1, "fen": board.fen(), "played": mv_uci, "best": sf_best, "drop_cp": drop, "category": cat})
                        break
                hist.append(mv_uci)
                board.push_uci(mv_uci)

        report = {"losses_analyzed": len(analyzed), "counts": counts, "examples": analyzed[:15]}
        print(json.dumps(report, indent=2))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    finally:
        sf.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuraChess Unified Evaluation CLI")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_ds = sub.add_parser("dataset", help="Evaluate PyTorch checkpoint on packed .bin test splits")
    p_ds.add_argument("--checkpoint", type=Path, required=True)
    p_ds.add_argument("--datasets", type=Path, nargs="+", required=True)
    p_ds.add_argument("--blocks", type=int, default=10)
    p_ds.add_argument("--channels", type=int, default=192)
    p_ds.add_argument("--no-bn", action="store_true")
    p_ds.add_argument("--layout", choices=["gpu_unpack", "canonical"], default="gpu_unpack")
    p_ds.add_argument("--batch-size", type=int, default=512)
    p_ds.add_argument("--max-records", type=int, default=50000)
    p_ds.add_argument("--device", type=str, default="cpu")
    p_ds.add_argument("--use-cache", action="store_true")
    p_ds.add_argument("--output", type=Path, default=None)

    p_pz = sub.add_parser("puzzles", help="Run tactical puzzle benchmark on NPU/CPU")
    p_pz.add_argument("--puzzles-json", type=Path, required=True)
    p_pz.add_argument("--xml", type=Path, default=None)
    p_pz.add_argument("--device", type=str, default="NPU")
    p_pz.add_argument("--budgets", type=int, nargs="+", default=[1, 50, 200, 800])
    p_pz.add_argument("--max-puzzles", type=int, default=None)
    p_pz.add_argument("--layout", choices=["gpu_unpack", "legacy_flip"], default="gpu_unpack")
    p_pz.add_argument("--output", type=Path, default=None)

    p_la = sub.add_parser("loss-anatomy", help="Analyze lost games at fixed Stockfish depth")
    p_la.add_argument("--games-json", type=Path, required=True)
    p_la.add_argument("--xml", type=Path, default=None)
    p_la.add_argument("--stockfish-path", type=str, default=None)
    p_la.add_argument("--depth", type=int, default=14)
    p_la.add_argument("--blunder-cp", type=int, default=150)
    p_la.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.mode == "dataset":
        cmd_dataset(args)
    elif args.mode == "puzzles":
        cmd_puzzles(args)
    else:
        cmd_loss_anatomy(args)


if __name__ == "__main__":
    main()
