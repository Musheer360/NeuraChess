"""
NeuraChess Match Runner & Pentanomial SPRT Harness (`neurachess.scripts.match_runner`).

Subcommands:
  - `vs-stockfish`: Balanced colour-reversed opening pairs against an external UCI engine
    at `UCI_Elo` or fixed node limits (`--sf-elo` or `--sf-nodes`).
  - `h2h`: Balanced paired head-to-head match between two NeuraChess OpenVINO IR configurations
    with bidirectional pair-level pentanomial SPRT and Wilson 95% confidence intervals.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine

from neurachess._paths import find_repo_root

ROOT_DIR = find_repo_root()

from neurachess.core.board_tensor import board_to_tensor
from neurachess.models.npu_compiler import DEFAULT_CACHE_DIR, NPUModel, resolve_default_model_paths
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch
from neurachess.search.ponder_manager import PonderManager
from neurachess.uci.engine_uci import UCIEngine

OPENINGS_50 = [
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4"],
    ["e2e4", "e7e5", "g1f3", "g8f6", "f3e5", "d7d6"],
    ["e2e4", "e7e5", "g1f3", "b8c6", "b1c3", "g8f6"],
    ["e2e4", "e7e5", "b1c3", "g8f6", "f2f4", "d7d5"],
    ["e2e4", "e7e5", "f2f4", "e5f4", "g1f3", "g7g5"],
    ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4"],
    ["e2e4", "c7c5", "g1f3", "b8c6", "d2d4", "c5d4"],
    ["e2e4", "c7c5", "g1f3", "e7e6", "d2d4", "c5d4"],
    ["e2e4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6"],
    ["e2e4", "c7c5", "c2c3", "g8f6", "e4e5", "f6d5"],
    ["e2e4", "c7c5", "c2c3", "d7d5", "e4d5", "d8d5"],
    ["e2e4", "c7c5", "g1f3", "d7d6", "f1b5", "b8d7"],
    ["e2e4", "c7c5", "g1f3", "b8c6", "f1b5", "g7g6"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "f8b4"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "b1d2", "c7c5"],
    ["e2e4", "e7e6", "d2d4", "d7d5", "e4e5", "c7c5"],
    ["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"],
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4e5", "c8f5"],
    ["e2e4", "c7c6", "d2d4", "d7d5", "e4d5", "c6d5"],
    ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5"],
    ["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6"],
    ["e2e4", "g7g6", "d2d4", "f8g7", "b1c3", "d7d6"],
    ["e2e4", "g8f6", "e4e5", "f6d5", "d2d4", "d7d6"],
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"],
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6"],
    ["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "e7e6"],
    ["d2d4", "d7d5", "c2c4", "d5c4", "g1f3", "g8f6"],
    ["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "c7c5"],
    ["d2d4", "d7d5", "c1f4", "g8f6", "e2e3", "c7c5"],
    ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"],
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "b7b6"],
    ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3", "f8b4"],
    ["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5"],
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7"],
    ["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5"],
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "e7e6"],
    ["d2d4", "g8f6", "c2c4", "c7c5", "d4d5", "b7b5"],
    ["d2d4", "f7f5", "g2g3", "g8f6", "f1g2", "e7e6"],
    ["d2d4", "g8f6", "c1g5", "e7e6", "e2e4", "h7h6"],
    ["c2c4", "e7e5", "b1c3", "g8f6", "g1f3", "b8c6"],
    ["c2c4", "e7e5", "b1c3", "b8c6", "g2g3", "g7g6"],
    ["c2c4", "c7c5", "b1c3", "b8c6", "g2g3", "g7g6"],
    ["c2c4", "g8f6", "b1c3", "e7e6", "g1f3", "d7d5"],
    ["g1f3", "d7d5", "c2c4", "e7e6", "g2g3", "g8f6"],
    ["g1f3", "g8f6", "g2g3", "g7g6", "f1g2", "f8g7"],
    ["b2b3", "e7e5", "c1b2", "b8c6", "e2e3", "g8f6"],
]


def wilson_interval(wins: float, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = wins / total
    denom = 1.0 + (z ** 2) / total
    center = (p + (z ** 2) / (2.0 * total)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z ** 2) / (4.0 * (total ** 2))))
    return max(0.0, center - spread), min(1.0, center + spread)


def score_to_elo_diff(score: float) -> float:
    s = min(0.999, max(0.001, score))
    return -400.0 * math.log10(1.0 / s - 1.0)


def elo_to_score(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def _tilted_pentanomial_probs(p_emp: list[float], target_mean: float) -> list[float]:
    vals = (0.0, 0.25, 0.50, 0.75, 1.0)
    beta = 0.0
    for _ in range(7):
        w = [p_emp[j] * math.exp(beta * vals[j]) for j in range(5)]
        z = sum(w)
        q = [wi / z for wi in w]
        mean = sum(vals[j] * q[j] for j in range(5))
        diff = mean - target_mean
        if abs(diff) < 1e-10:
            return q
        var = sum(q[j] * ((vals[j] - mean) ** 2) for j in range(5))
        if var < 1e-12:
            break
        beta = max(-40.0, min(40.0, beta - diff / var))
    w = [p_emp[j] * math.exp(beta * vals[j]) for j in range(5)]
    z = sum(w)
    return [wi / z for wi in w]


def compute_pentanomial_sprt_llr(
    pent_counts: list[int],
    elo0: float = 0.0,
    elo1: float = 20.0,
    alpha: float = 0.05,
    beta: float = 0.05,
    min_pairs_for_decision: int = 10,
) -> dict:
    m_pairs = sum(pent_counts)
    lower = math.log(beta / (1.0 - alpha))
    upper = math.log((1.0 - beta) / alpha)
    if m_pairs == 0:
        return {"llr": 0.0, "lower": round(lower, 4), "upper": round(upper, 4), "elo0": elo0, "elo1": elo1, "pentanomial": list(pent_counts), "pairs": 0, "status": "CONTINUE"}

    prior = 0.5
    p_emp = [(pent_counts[j] + prior) / (m_pairs + 5.0 * prior) for j in range(5)]
    q0 = _tilted_pentanomial_probs(p_emp, elo_to_score(elo0))
    q1 = _tilted_pentanomial_probs(p_emp, elo_to_score(elo1))
    llr = sum(pent_counts[j] * math.log(max(1e-15, q1[j]) / max(1e-15, q0[j])) for j in range(5))
    if m_pairs < min_pairs_for_decision:
        status = "CONTINUE"
    else:
        status = "ACCEPT_H1" if llr >= upper else ("ACCEPT_H0" if llr <= lower else "CONTINUE")
    return {"llr": round(llr, 4), "lower": round(lower, 4), "upper": round(upper, 4), "elo0": elo0, "elo1": elo1, "pentanomial": list(pent_counts), "pairs": m_pairs, "status": status}


def _resolve_sf(explicit_path: str | None) -> str:
    if explicit_path and Path(explicit_path).exists():
        return explicit_path
    env_p = os.environ.get("STOCKFISH_PATH")
    if env_p and Path(env_p).exists():
        return env_p
    w = shutil.which("stockfish")
    if w:
        return w
    for cand in (
        ROOT_DIR / "bin" / "stockfish" / "stockfish-windows-x86-64-universal.exe",
        ROOT_DIR / "bin" / "stockfish" / "stockfish",
    ):
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("Stockfish binary not found; pass --stockfish-path or set STOCKFISH_PATH.")


def load_openings_file(path: Path | None) -> list[list[str]]:
    if path is None:
        return OPENINGS_50
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("openings", raw) if isinstance(raw, dict) else raw
    return [it["moves"] if isinstance(it, dict) else list(it) for it in items]


def adjudicate_unfinished(board: chess.Board) -> str:
    """Adjudicate unfinished games by v1.0.2 material difference (>200 cp threshold)."""
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return outcome.result()
    val_map = {chess.PAWN: 100, chess.KNIGHT: 305, chess.BISHOP: 333, chess.ROOK: 563, chess.QUEEN: 950}
    w_mat = sum(len(board.pieces(pt, chess.WHITE)) * val_map[pt] for pt in val_map)
    b_mat = sum(len(board.pieces(pt, chess.BLACK)) * val_map[pt] for pt in val_map)
    mat_diff = w_mat - b_mat
    if mat_diff > 200:
        return "1-0"
    if mat_diff < -200:
        return "0-1"
    return "1/2-1/2"


def _build_engine(
    xml_path: str | None,
    batch_size: int = 16,
    device: str = "NPU",
    use_fast_tree: bool = True,
    encoder_layout: str = "gpu_unpack",
    use_dynamic_batch_shift: bool = False,
    use_ponder: bool = False,
    movetime_ms: int = 400,
    cache_dir: str | None = None,
    fpu_reduction: float | None = None,
    syzygy_path: str | None = None,
) -> NPUBatchedPUCTSearch:
    resolved_xml = xml_path or resolve_default_model_paths()[0]
    eff_cache = cache_dir or DEFAULT_CACHE_DIR
    root_m = NPUModel(
        batch_size=1,
        cache_dir=eff_cache,
        device=device,
        use_turbo=True,
        performance_hint="LATENCY",
        ov_ir_xml_path=resolved_xml,
        verbose=False,
    )
    batch_m = NPUModel(
        batch_size=batch_size,
        cache_dir=eff_cache,
        device=device,
        use_turbo=True,
        performance_hint="LATENCY",
        ov_ir_xml_path=resolved_xml,
        verbose=False,
    )
    need_gear_shapes = bool(use_dynamic_batch_shift) and ((not use_ponder) or int(movetime_ms) <= 150)
    batch_models = None
    if need_gear_shapes:
        batch_models = {
            b: (
                batch_m
                if b == batch_size
                else NPUModel(
                    batch_size=b,
                    cache_dir=eff_cache,
                    device=device,
                    use_turbo=True,
                    performance_hint="LATENCY",
                    ov_ir_xml_path=resolved_xml,
                    verbose=False,
                )
            )
            for b in (4, 8, 16)
        }
    kw: dict = {
        "root_model": root_m,
        "batch_model": batch_m,
        "batch_size": batch_size,
        "use_fast_tree": use_fast_tree,
        "encoder_layout": encoder_layout,
        "use_dynamic_batch_shift": bool(use_dynamic_batch_shift),
        "use_ponder": bool(use_ponder),
        "batch_models": batch_models,
        "syzygy_path": syzygy_path,
    }
    if fpu_reduction is not None:
        kw["fpu_reduction"] = float(fpu_reduction)
    return NPUBatchedPUCTSearch(**kw)


def _select_engine_move(
    board: chess.Board,
    searcher: NPUBatchedPUCTSearch,
    movetime_ms: int,
    book_driver: UCIEngine | None,
) -> tuple[str, int, list[str], float]:
    """Return `(uci, nodes, pv, elapsed_s)` from the CC0 opening book if enabled and present, else PUCT."""
    if book_driver is not None:
        book_uci, book_ponder = book_driver.lookup_book_move(board)
        if book_uci is not None:
            pv = [book_uci, book_ponder] if book_ponder else [book_uci]
            return book_uci, 0, pv, 0.0
    t0 = time.perf_counter()
    _, best_uci, _, nodes, pv, _ = searcher.search(
        board,
        board_to_tensor(board, layout=searcher.encoder_layout),
        movetime_ms=movetime_ms,
    )
    return best_uci, int(nodes), list(pv), time.perf_counter() - t0


def run_stockfish_match(args: argparse.Namespace) -> dict:
    sf_bin = _resolve_sf(args.stockfish_path)
    openings = load_openings_file(getattr(args, "openings_file", None))
    start_pair = int(getattr(args, "start_pair", 0) or 0)
    adjudicate_mat = bool(getattr(args, "adjudicate_material", False))
    searcher = _build_engine(
        str(args.xml) if args.xml else None,
        batch_size=args.batch_size,
        device=args.device,
        use_fast_tree=not args.python_tree,
        use_dynamic_batch_shift=getattr(args, "dynamic_gear", False),
        use_ponder=bool(args.ponder),
        movetime_ms=int(args.movetime_ms),
        cache_dir=getattr(args, "cache_dir", None),
        fpu_reduction=getattr(args, "fpu_reduction", None),
        syzygy_path=getattr(args, "syzygy_path", None),
    )
    ponder = PonderManager(searcher, use_ponder=args.ponder)
    book_driver = UCIEngine(own_book=True, book_seed=42) if getattr(args, "own_book", False) else None

    sf_opts = {"Threads": 1, "Hash": 16}
    if args.sf_elo is not None:
        sf_opts["UCI_LimitStrength"] = True
        sf_opts["UCI_Elo"] = args.sf_elo
    else:
        sf_opts["UCI_LimitStrength"] = False

    sf_engine = chess.engine.SimpleEngine.popen_uci(sf_bin)
    sf_engine.configure(sf_opts)

    wins = draws = losses = 0
    adjudicated_wins = adjudicated_draws = adjudicated_losses = 0
    late_endgame_losses = 0
    total_nodes = 0
    total_search_s = 0.0
    games_log = []

    try:
        searcher.search(chess.Board(), board_to_tensor(chess.Board(), layout=searcher.encoder_layout), movetime_ms=50, max_nodes=32)
        searcher.clear_state()
        for pair_offset in range(args.pairs):
            pair_idx = start_pair + pair_offset
            opening = openings[pair_idx % len(openings)]
            for nc_color in (chess.WHITE, chess.BLACK):
                ponder.stop()
                searcher.clear_state()
                board = chess.Board()
                for uci in opening:
                    board.push_uci(uci)

                while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
                    if board.turn == nc_color:
                        best_uci, nodes, pv, dt = _select_engine_move(board, searcher, args.movetime_ms, book_driver)
                        total_search_s += dt
                        total_nodes += nodes
                        mv = chess.Move.from_uci(best_uci)
                        board.push(mv)
                        if args.ponder and not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
                            exp = chess.Move.from_uci(pv[1]) if len(pv) >= 2 else None
                            ponder.start_internal_ponder(board.copy(), expected_reply=exp)
                    else:
                        limit = chess.engine.Limit(nodes=args.sf_nodes) if args.sf_nodes else chess.engine.Limit(time=args.movetime_ms / 1000.0)
                        res = sf_engine.play(board, limit)
                        opp_mv = res.move
                        if args.ponder:
                            ponder.on_opponent_move(opp_mv)
                        board.push(opp_mv)

                ponder.stop()
                outcome = board.outcome(claim_draw=True)
                adj = False
                if outcome is not None:
                    res_str = outcome.result()
                elif adjudicate_mat:
                    res_str = adjudicate_unfinished(board)
                    adj = True
                else:
                    res_str = "1/2-1/2"
                    adj = True

                if res_str == "1-0":
                    score = 1.0 if nc_color == chess.WHITE else 0.0
                elif res_str == "0-1":
                    score = 0.0 if nc_color == chess.WHITE else 1.0
                else:
                    score = 0.5

                if score == 1.0:
                    wins += 1
                    if adj:
                        adjudicated_wins += 1
                elif score == 0.0:
                    losses += 1
                    if adj:
                        adjudicated_losses += 1
                    if board.ply() > 120 or chess.popcount(board.occupied) <= 7:
                        late_endgame_losses += 1
                else:
                    draws += 1
                    if adj:
                        adjudicated_draws += 1

                games_log.append(
                    {
                        "pair": pair_idx,
                        "nc_color": "white" if nc_color == chess.WHITE else "black",
                        "result": res_str,
                        "adjudicated": adj,
                        "end_pieces": int(chess.popcount(board.occupied)),
                        "nc_score": score,
                        "plies": board.ply(),
                    }
                )
                played = wins + draws + losses
                print(
                    f"[Game {played}/{args.pairs * 2} | pair={pair_idx} plies={board.ply()} res={res_str}{' (adj)' if adj else ''}] "
                    f"W={wins} D={draws} L={losses} ({100.0*(wins+0.5*draws)/played:.1f}%)",
                    flush=True,
                )
                if args.output:
                    pts_now = wins + 0.5 * draws
                    lo_now, hi_now = wilson_interval(pts_now, played)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps(
                            {
                                "total_games": played,
                                "wins": wins,
                                "draws": draws,
                                "losses": losses,
                                "points": pts_now,
                                "score_pct": round(100.0 * pts_now / max(1, played), 2),
                                "wilson_95_ci_pct": [round(lo_now * 100.0, 2), round(hi_now * 100.0, 2)],
                                "effective_nps": round(total_nodes / max(1e-6, total_search_s), 1),
                                "adjudicated": {"wins": adjudicated_wins, "draws": adjudicated_draws, "losses": adjudicated_losses},
                                "late_endgame_losses": late_endgame_losses,
                                "games": games_log,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
    finally:
        ponder.stop()
        sf_engine.quit()

    total_g = wins + draws + losses
    pts = wins + 0.5 * draws
    lo, hi = wilson_interval(pts, total_g)
    summary = {
        "total_games": total_g,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": pts,
        "score_pct": round(100.0 * pts / max(1, total_g), 2),
        "wilson_95_ci_pct": [round(lo * 100.0, 2), round(hi * 100.0, 2)],
        "effective_nps": round(total_nodes / max(1e-6, total_search_s), 1),
        "adjudicated": {"wins": adjudicated_wins, "draws": adjudicated_draws, "losses": adjudicated_losses},
        "late_endgame_losses": late_endgame_losses,
        "games": games_log,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_h2h_match(args: argparse.Namespace) -> dict:
    openings = load_openings_file(getattr(args, "openings_file", None))
    start_pair = int(getattr(args, "start_pair", 0) or 0)
    adjudicate_mat = bool(getattr(args, "adjudicate_material", False))
    fpu_red = getattr(args, "fpu_reduction", None)
    syz_path = getattr(args, "syzygy_path", None)
    eng_a = _build_engine(
        str(args.xml_a),
        batch_size=args.batch_size,
        device=args.device,
        use_dynamic_batch_shift=getattr(args, "dynamic_gear_a", False),
        movetime_ms=int(args.movetime_ms),
        cache_dir=getattr(args, "cache_dir_a", None),
        fpu_reduction=fpu_red,
        syzygy_path=syz_path,
    )
    eng_b = _build_engine(
        str(args.xml_b),
        batch_size=args.batch_size,
        device=args.device,
        use_dynamic_batch_shift=getattr(args, "dynamic_gear_b", False),
        movetime_ms=int(args.movetime_ms),
        cache_dir=getattr(args, "cache_dir_b", None),
        fpu_reduction=fpu_red,
        syzygy_path=syz_path,
    )
    book_driver = UCIEngine(own_book=True, book_seed=42) if getattr(args, "own_book", False) else None
    warm_t = board_to_tensor(chess.Board(), layout=eng_a.encoder_layout)
    eng_a.search(chess.Board(), warm_t, movetime_ms=50, max_nodes=32)
    eng_b.search(chess.Board(), warm_t, movetime_ms=50, max_nodes=32)
    eng_a.clear_state()
    eng_b.clear_state()

    wins_a = draws = losses_a = 0
    pent = [0, 0, 0, 0, 0]
    nodes_a = nodes_b = 0
    time_a = time_b = 0.0
    games_log = []

    def _summarize_h2h() -> dict:
        total_g = wins_a + draws + losses_a
        pts = wins_a + 0.5 * draws
        lo, hi = wilson_interval(pts, total_g)
        return {
            "total_games": total_g,
            "wins": wins_a,
            "draws": draws,
            "losses": losses_a,
            "points": pts,
            "score_pct": round(100.0 * pts / max(1, total_g), 2),
            "wilson_95_ci_pct": [round(lo * 100.0, 2), round(hi * 100.0, 2)],
            "pentanomial": list(pent),
            "sprt": compute_pentanomial_sprt_llr(pent),
            "nps_a": round(nodes_a / max(1e-6, time_a), 1),
            "nps_b": round(nodes_b / max(1e-6, time_b), 1),
            "games": games_log,
        }

    for pair_offset in range(args.pairs):
        pair_idx = start_pair + pair_offset
        opening = openings[pair_idx % len(openings)]
        pair_score = 0.0
        for a_color in (chess.WHITE, chess.BLACK):
            eng_a.clear_state()
            eng_b.clear_state()
            board = chess.Board()
            for uci in opening:
                board.push_uci(uci)

            while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
                is_a = board.turn == a_color
                active = eng_a if is_a else eng_b
                best_uci, mv_nodes, _, mv_dt = _select_engine_move(board, active, args.movetime_ms, book_driver)
                if is_a:
                    nodes_a += mv_nodes
                    time_a += mv_dt
                else:
                    nodes_b += mv_nodes
                    time_b += mv_dt
                board.push_uci(best_uci)

            outcome = board.outcome(claim_draw=True)
            adj = False
            if outcome is not None:
                res_str = outcome.result()
            elif adjudicate_mat:
                res_str = adjudicate_unfinished(board)
                adj = True
            else:
                res_str = "1/2-1/2"
                adj = True

            if res_str == "1-0":
                g_score = 1.0 if a_color == chess.WHITE else 0.0
            elif res_str == "0-1":
                g_score = 0.0 if a_color == chess.WHITE else 1.0
            else:
                g_score = 0.5

            if g_score == 1.0:
                wins_a += 1
            elif g_score == 0.0:
                losses_a += 1
            else:
                draws += 1
            pair_score += g_score
            games_log.append(
                {
                    "pair": pair_idx,
                    "a_color": "white" if a_color == chess.WHITE else "black",
                    "result": res_str,
                    "adjudicated": adj,
                    "a_score": g_score,
                    "plies": board.ply(),
                }
            )

        pent[int(round(pair_score * 2))] += 1
        sprt = compute_pentanomial_sprt_llr(pent)
        pts_now = wins_a + 0.5 * draws
        g_now = wins_a + draws + losses_a
        print(
            f"[Pair {pair_offset+1}/{args.pairs} (idx={pair_idx})] W={wins_a} D={draws} L={losses_a} "
            f"({100.0 * pts_now / max(1, g_now):.1f}%) | Pent={pent} | SPRT={sprt['status']} ({sprt['llr']})",
            flush=True,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(_summarize_h2h(), indent=2), encoding="utf-8")

    summary = _summarize_h2h()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuraChess UCI Match & Pentanomial SPRT Runner")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_sf = sub.add_parser("vs-stockfish", help="Play balanced pairs against Stockfish")
    p_sf.add_argument("--xml", type=Path, default=None)
    p_sf.add_argument("--stockfish-path", type=str, default=None)
    p_sf.add_argument("--sf-elo", type=int, default=3000)
    p_sf.add_argument("--sf-nodes", type=int, default=None)
    p_sf.add_argument("--pairs", type=int, default=25)
    p_sf.add_argument("--movetime-ms", type=int, default=400)
    p_sf.add_argument("--batch-size", type=int, default=16)
    p_sf.add_argument("--max-plies", type=int, default=220)
    p_sf.add_argument("--device", type=str, default="NPU")
    p_sf.add_argument("--ponder", action="store_true")
    p_sf.add_argument("--dynamic-gear", action="store_true", help="Enable co-resident B=4/8/16 dynamic gear-shifting")
    p_sf.add_argument("--cache-dir", type=str, default=None)
    p_sf.add_argument("--openings-file", type=Path, default=None, help="Optional JSON openings file (e.g., openings_320.json)")
    p_sf.add_argument("--start-pair", type=int, default=0, help="Starting pair index into openings list")
    p_sf.add_argument("--adjudicate-material", action="store_true", help="Adjudicate max_plies unfinished games by v1.0.2 material difference (>200cp)")
    p_sf.add_argument("--fpu-reduction", type=float, default=None, help="Override PUCT fpu_reduction (e.g., 0.25 for v1.0.2 ladder parity)")
    p_sf.add_argument("--syzygy-path", type=str, default=None, help="Syzygy tablebase directory or 'auto'/'none' (default: disabled unless NEURACHESS_SYZYGY_PATH set)")
    p_sf.add_argument("--own-book", action="store_true", help="Use bundled CC0 Lichess opening book up to ply 15")
    p_sf.add_argument("--python-tree", action="store_true")
    p_sf.add_argument("--output", type=Path, default=None)

    p_h2h = sub.add_parser("h2h", help="Play paired head-to-head match between two OpenVINO IR models")
    p_h2h.add_argument("--xml-a", type=Path, required=True)
    p_h2h.add_argument("--xml-b", type=Path, required=True)
    p_h2h.add_argument("--pairs", type=int, default=50)
    p_h2h.add_argument("--movetime-ms", type=int, default=400)
    p_h2h.add_argument("--batch-size", type=int, default=16)
    p_h2h.add_argument("--max-plies", type=int, default=220)
    p_h2h.add_argument("--device", type=str, default="NPU")
    p_h2h.add_argument("--dynamic-gear-a", action="store_true", help="Enable B=4/8/16 dynamic gear-shifting on Engine A")
    p_h2h.add_argument("--dynamic-gear-b", action="store_true", help="Enable B=4/8/16 dynamic gear-shifting on Engine B")
    p_h2h.add_argument("--cache-dir-a", type=str, default=None)
    p_h2h.add_argument("--cache-dir-b", type=str, default=None)
    p_h2h.add_argument("--openings-file", type=Path, default=None, help="Optional JSON openings file (e.g., openings_320.json)")
    p_h2h.add_argument("--start-pair", type=int, default=0, help="Starting pair index into openings list")
    p_h2h.add_argument("--adjudicate-material", action="store_true", help="Adjudicate max_plies unfinished games by v1.0.2 material difference (>200cp)")
    p_h2h.add_argument("--fpu-reduction", type=float, default=None, help="Override PUCT fpu_reduction")
    p_h2h.add_argument("--syzygy-path", type=str, default=None, help="Syzygy tablebase directory")
    p_h2h.add_argument("--own-book", action="store_true", help="Use bundled CC0 Lichess opening book up to ply 15")
    p_h2h.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    if args.mode == "vs-stockfish":
        res = run_stockfish_match(args)
    else:
        res = run_h2h_match(args)
    print(json.dumps({k: v for k, v in res.items() if k != "games"}, indent=2))


if __name__ == "__main__":
    main()
