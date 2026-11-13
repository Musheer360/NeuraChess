"""
NeuraChess Candidate Selector (v3.0)

Selects candidate moves for tree expansion:
  - Ranks legal moves by policy logit plus a small capture bonus (MVV-LVA style: +0.35 for
    equal-or-winning captures, +0.10 for captures that lose material).
  - No check bonus and no draw-claim detection (v2.6 had both; they were removed because they
    cost ~7x more time per call; the effect on playing strength has not been isolated).
  - If all legal moves fit in top_k, returns them all unscored (no ranking needed).
  - The bonus only affects which moves are proposed; values/priors come from the network.
"""

import chess
import numpy as np
from neurachess.core.constants import TOTAL_ACTIONS, uci_to_action, move_to_action


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


def select_top_candidates(
    board: chess.Board,
    policy_logits: np.ndarray,
    top_k: int = 16,
) -> tuple[list[int], list[chess.Move], dict]:
    """
    Selects top-K candidate moves using fast policy prior + MVV-LVA capture bonus for beam inclusion.
    Zero push/pop movegen overhead and zero string allocations.
    
    Returns:
        candidate_actions: list[int]
        candidate_moves: list[chess.Move]
        meta: dict with terminal state information
    """
    is_black = (board.turn == chess.BLACK)
    legal_moves = list(board.legal_moves)
    n_legal = len(legal_moves)

    # 1. Terminal State Check
    if n_legal == 0:
        in_check = board.is_check()
        return [], [], {
            "is_terminal": True,
            "is_checkmate": in_check,
            "is_stalemate": not in_check,
            "is_draw_claimable": False,
            "n_legal": 0,
        }

    # Fast short-circuit if all legal moves fit within top_k
    if n_legal <= top_k:
        candidate_actions = [move_to_action(m, canonical_flip=is_black) for m in legal_moves]
        return candidate_actions, legal_moves, {
            "is_terminal": False,
            "is_checkmate": False,
            "is_stalemate": False,
            "is_draw_claimable": False,
            "n_legal": n_legal,
        }

    scored_candidates = []
    for move in legal_moves:
        act = move_to_action(move, canonical_flip=is_black)
        base_logit = float(policy_logits[act])

        # Fast MVV-LVA capture bonus (no push/pop movegen)
        tactical_bonus = 0.0
        if board.is_capture(move):
            attacker_type = board.piece_type_at(move.from_square) or chess.PAWN
            attacker_val = PIECE_VALUES.get(attacker_type, 100)
            if board.is_en_passant(move):
                victim_val = PIECE_VALUES[chess.PAWN]
            else:
                victim_type = board.piece_type_at(move.to_square) or chess.PAWN
                victim_val = PIECE_VALUES.get(victim_type, 100)

            if victim_val >= attacker_val:
                tactical_bonus = 0.35  # Winning or equal trade
            else:
                tactical_bonus = 0.10  # Potential sacrifice, minor proposal boost

        proposal_score = base_logit + tactical_bonus
        scored_candidates.append((proposal_score, act, move))

    # Sort descending by proposal score and select top_k
    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = scored_candidates[:top_k]

    candidate_actions = [c[1] for c in top_candidates]
    candidate_moves = [c[2] for c in top_candidates]

    return candidate_actions, candidate_moves, {
        "is_terminal": False,
        "is_checkmate": False,
        "is_stalemate": False,
        "is_draw_claimable": False,
        "n_legal": n_legal,
    }


def select_opening_guarded_root_act(
    candidates: list[tuple[int, float, int, float, bool, bool]],
    top_k: int = 4,
    min_prior: float = 0.08,
    q_override_margin: float = 0.15,
) -> int:
    """Select root action in opening positions (fullmove <= 12) with prior guard + Q override.

    Each tuple in `candidates` is:
        (act, prior, visits, q_from_root, is_proven_win, is_proven_loss)
    Restricts selection to top-`top_k` raw policy priors (plus any move with `prior >= min_prior`),
    unless an outside move is a proven mate win or (visited, non-losing) exceeds the Top-3 best Q
    by at least `q_override_margin`.
    """
    by_prior = sorted(candidates, key=lambda c: c[1], reverse=True)
    guard_acts = {c[0] for c in by_prior[:top_k]} | {c[0] for c in by_prior if c[1] >= min_prior}

    top3 = by_prior[:3]
    visited_top3_q = [c[3] for c in top3 if c[2] > 0]
    if visited_top3_q:
        top3_best_q: float | None = max(visited_top3_q)
    else:
        visited_guard_q = [c[3] for c in by_prior if c[0] in guard_acts and c[2] > 0]
        top3_best_q = max(visited_guard_q) if visited_guard_q else None

    eligible = [
        c
        for c in by_prior
        if (c[0] in guard_acts)
        or c[4]
        or (not c[5] and c[2] > 0 and top3_best_q is not None and c[3] >= top3_best_q + q_override_margin)
    ]
    if not eligible:
        eligible = by_prior

    best = max(eligible, key=lambda c: (c[2], c[3], c[1]))
    return best[0]
