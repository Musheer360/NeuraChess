"""
NeuraChess 18-Plane Representation (v3.1)
Supports explicit encoder layouts:
1. 'legacy_flip': Side-to-move friendly/opponent perspective with ranks flipped for Black.
   Used by 1M Pilot, EXP-011.
2. 'gpu_unpack': Dataset unpack representation. White pieces in planes 0..5, Black in 6..11,
   Turn in plane 12, White/Black castling in 13..16, Halfmove in 17. No rank flipping.
   Used by S3, S4, S4-Fix, S5a.
"""

import numpy as np
import chess
from .constants import NUM_PLANES

ENCODER_LAYOUT_LEGACY_FLIP = "legacy_flip"
ENCODER_LAYOUT_GPU_UNPACK = "gpu_unpack"
SUPPORTED_ENCODER_LAYOUTS = (ENCODER_LAYOUT_LEGACY_FLIP, ENCODER_LAYOUT_GPU_UNPACK)

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def encode_single_position(board: chess.Board, is_black: bool) -> np.ndarray:
    """Encode current piece placements into 12 planes (6 friendly, 6 opponent)."""
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    friendly_color = chess.BLACK if is_black else chess.WHITE
    opponent_color = chess.WHITE if is_black else chess.BLACK

    for p_idx, piece_type in enumerate(PIECE_TYPES):
        # Friendly pieces
        for sq in board.pieces(piece_type, friendly_color):
            r = chess.square_rank(sq)
            c = chess.square_file(sq)
            if is_black:
                r = 7 - r
            planes[p_idx, r, c] = 1.0

        # Opponent pieces
        for sq in board.pieces(piece_type, opponent_color):
            r = chess.square_rank(sq)
            c = chess.square_file(sq)
            if is_black:
                r = 7 - r
            planes[6 + p_idx, r, c] = 1.0

    return planes


def board_to_tensor_legacy_flip(board: chess.Board) -> np.ndarray:
    """Canonical representation (pilot, EXP-011): friendly/opponent perspective, ranks flipped for Black."""
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    is_black = (board.turn == chess.BLACK)

    # 1. Piece placements (Planes 0..11)
    tensor[0:12] = encode_single_position(board, is_black)

    # 2. Castling rights (Planes 12..15)
    friendly_color = chess.BLACK if is_black else chess.WHITE
    opponent_color = chess.WHITE if is_black else chess.BLACK

    if board.has_kingside_castling_rights(friendly_color):
        tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(friendly_color):
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(opponent_color):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(opponent_color):
        tensor[15, :, :] = 1.0

    # 3. En-passant target square (Plane 16)
    if board.ep_square is not None:
        ep_r = chess.square_rank(board.ep_square)
        ep_c = chess.square_file(board.ep_square)
        if is_black:
            ep_r = 7 - ep_r
        tensor[16, ep_r, ep_c] = 1.0

    # 4. Halfmove clock normalized (Plane 17)
    tensor[17, :, :] = min(1.0, board.halfmove_clock / 100.0)

    return tensor


def board_to_tensor_gpu_unpack(board: chess.Board) -> np.ndarray:
    """Dataset unpack representation (S3, S4, S4-fix, S5a): White/Black absolute coordinates, turn in plane 12."""
    tensor = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # Planes 0..5: White pieces
    # Planes 6..11: Black pieces
    for p_idx, piece_type in enumerate(PIECE_TYPES):
        for sq in board.pieces(piece_type, chess.WHITE):
            tensor[p_idx, sq // 8, sq % 8] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            tensor[6 + p_idx, sq // 8, sq % 8] = 1.0

    # Plane 12: Turn (1.0 for Black, 0.0 for White)
    if board.turn == chess.BLACK:
        tensor[12, :, :] = 1.0

    # Planes 13..16: Castling rights (WK, WQ, BK, BQ)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # Plane 17: Halfmove clock normalized
    tensor[17, :, :] = min(1.0, board.halfmove_clock / 100.0)

    return tensor


def board_to_tensor(
    board: chess.Board,
    history_boards: list[chess.Board] | None = None,
    layout: str = ENCODER_LAYOUT_LEGACY_FLIP
) -> np.ndarray:
    """
    Convert a chess.Board into an 18-plane float32 tensor of shape (18, 8, 8).
    layout must be 'legacy_flip' or 'gpu_unpack'.
    """
    if isinstance(history_boards, str):
        layout = history_boards
        history_boards = None

    if layout == ENCODER_LAYOUT_LEGACY_FLIP:
        return board_to_tensor_legacy_flip(board)
    elif layout == ENCODER_LAYOUT_GPU_UNPACK:
        return board_to_tensor_gpu_unpack(board)
    else:
        raise ValueError(f"Unsupported encoder_layout: '{layout}'. Must be one of {SUPPORTED_ENCODER_LAYOUTS}")
