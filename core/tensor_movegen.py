"""
NeuraChess Legal Action Mask Generator (v2.0)

Generates the exact boolean / float32 legal move mask:
    M in {0, 1}^4672 (or [B, 4672])
where M[a] = 1.0 if and only if action slot a is a strictly legal chess move.

Complies with all FIDE chess rules:
  - Knight, Bishop, Rook, Queen, King moves
  - Pawn single push, double push, diagonal captures, en-passant captures
  - Queen promotions and Underpromotions (Knight, Bishop, Rook)
  - Castling rights (through-check and into-check validation)
  - Absolute pins, discovered checks, double checks
"""

import numpy as np
import chess
from .constants import (
    TOTAL_ACTIONS,
    encode_action,
    decode_action,
    uci_to_action,
    action_to_uci,
)


def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """
    Generate the 4,672-dimensional legal action mask for the given position.
    canonical_flip is applied automatically if it is Black's turn.
    Returns:
        mask: np.ndarray of shape (4672,), dtype float32 (1.0 for legal, 0.0 for illegal).
    """
    mask = np.zeros(TOTAL_ACTIONS, dtype=np.float32)
    is_black = (board.turn == chess.BLACK)

    for move in board.legal_moves:
        uci = move.uci()
        act = uci_to_action(uci, canonical_flip=is_black)
        mask[act] = 1.0

    return mask


def get_legal_action_indices(board: chess.Board) -> list[int]:
    """Return the list of legal action indices in [0, 4671]."""
    is_black = (board.turn == chess.BLACK)
    return [uci_to_action(move.uci(), canonical_flip=is_black) for move in board.legal_moves]


def action_to_move(board: chess.Board, action_idx: int) -> chess.Move:
    """
    Convert an action index in [0, 4671] back to a valid chess.Move for the given board.
    Properly handles pawn promotions vs non-pawn moves to the 8th rank.
    """
    is_black = (board.turn == chess.BLACK)
    from_sq_canon, to_sq_canon, promo = decode_action(action_idx)
    
    # Unflip if Black to get board coordinates
    from_file = from_sq_canon % 8
    from_rank = from_sq_canon // 8
    to_file = to_sq_canon % 8
    to_rank = to_sq_canon // 8

    if is_black:
        from_rank = 7 - from_rank
        to_rank = 7 - to_rank

    board_from_sq = from_rank * 8 + from_file
    board_to_sq = to_rank * 8 + to_file
    
    piece = board.piece_at(board_from_sq)
    is_pawn = piece is not None and piece.piece_type == chess.PAWN
    is_promo_rank = (to_rank == 7 if board.turn == chess.WHITE else to_rank == 0)

    if is_pawn and is_promo_rank:
        if promo is None:
            promo = "q"  # Default queen promotion
        promo_type = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}[promo.lower()]
    else:
        promo_type = None

    return chess.Move(board_from_sq, board_to_sq, promotion=promo_type)
