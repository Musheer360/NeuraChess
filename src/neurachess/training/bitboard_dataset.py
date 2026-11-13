"""
Bitboard Dataset Packer & Unpacker (Part D4)

Packs chess board positions and evaluation targets into a compact ~107-byte binary record:
- 12 piece bitboards (12 * 8 = 96 bytes)
- Side to move (1 byte)
- Castling rights (1 byte)
- En passant square (1 byte)
- Halfmove clock (1 byte)
- Best move action index (2 bytes, uint16)
- Eval centipawns (2 bytes, int16)
- WDL probabilities (3 bytes, uint8 * 3)
Total: 107 bytes per position.
"""

import struct
import chess
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Format string for struct:
# 12 uint64 (Q) = 96 bytes
# 4 uint8 (B) = 4 bytes (turn, castling, ep_square, halfmove)
# 1 uint16 (H) = 2 bytes (action index)
# 1 int16 (h) = 2 bytes (eval cp, side-to-move perspective)
# 3 uint8 (B) = 3 bytes (W, D, L * 255)
# Total = 96 + 4 + 2 + 2 + 3 = 107 bytes
PACK_FORMAT = "<12Q 4B H h 3B"
RECORD_SIZE = struct.calcsize(PACK_FORMAT)
assert RECORD_SIZE == 107, f"Expected 107 bytes, got {RECORD_SIZE}"
PACKED_MASK_SIZE = 584  # 4672 // 8
RECORD_SIZE_V1 = 107
RECORD_SIZE_V2 = 107 + PACKED_MASK_SIZE  # 691 bytes

PIECE_ORDER = [
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
]

def pack_position(
    board: chess.Board,
    action_index: int = 0,
    eval_cp: int = 0,
    wdl: Tuple[float, float, float] = (0.5, 0.0, 0.5)
) -> bytes:
    """Pack a chess.Board and targets into 107 bytes."""
    # 12 piece bitboards
    bbs = []
    for piece_type, color in PIECE_ORDER:
        bbs.append(int(board.pieces_mask(piece_type, color)))
        
    turn = 1 if board.turn == chess.BLACK else 0
    
    # Castling rights: 4 bits
    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8
        
    ep_square = board.ep_square if board.ep_square is not None else 255
    halfmove = min(100, board.halfmove_clock)
    
    # Clamped eval cp
    eval_clamped = max(-32767, min(32767, int(eval_cp)))
    
    w_byte = int(round(np.clip(wdl[0], 0.0, 1.0) * 255))
    d_byte = int(round(np.clip(wdl[1], 0.0, 1.0) * 255))
    l_byte = int(round(np.clip(wdl[2], 0.0, 1.0) * 255))
    
    return struct.pack(
        PACK_FORMAT,
        *bbs,
        turn,
        castling,
        ep_square,
        halfmove,
        int(action_index),
        eval_clamped,
        w_byte,
        d_byte,
        l_byte
    )

def unpack_position(raw_bytes: bytes) -> Dict[str, Any]:
    """Unpack a 107-byte record into chess.Board and metadata/targets."""
    unpacked = struct.unpack(PACK_FORMAT, raw_bytes)
    bbs = unpacked[:12]
    turn, castling, ep_square, halfmove, action_index, eval_cp, w_byte, d_byte, l_byte = unpacked[12:]
    
    board = chess.Board.empty()
    for (piece_type, color), bb in zip(PIECE_ORDER, bbs):
        for sq in range(64):
            if (bb >> sq) & 1:
                board.set_piece_at(sq, chess.Piece(piece_type, color))
                
    board.turn = chess.BLACK if turn == 1 else chess.WHITE
    
    # Restore castling
    castling_fen = ""
    if castling & 1: castling_fen += "K"
    if castling & 2: castling_fen += "Q"
    if castling & 4: castling_fen += "k"
    if castling & 8: castling_fen += "q"
    board.set_castling_fen(castling_fen if castling_fen else "-")
    
    board.ep_square = ep_square if ep_square != 255 else None
    board.halfmove_clock = halfmove
    
    wdl = (w_byte / 255.0, d_byte / 255.0, l_byte / 255.0)
    
    return {
        "board": board,
        "action_index": action_index,
        "eval_cp": eval_cp,
        "wdl": wdl
    }


def pack_position_v2(
    board: chess.Board,
    action_index: int = 0,
    eval_cp: int = 0,
    wdl: Tuple[float, float, float] = (0.5, 0.0, 0.5),
    legal_mask: Optional[np.ndarray] = None
) -> bytes:
    """Pack board, metadata, targets, and 584-byte packed legal move bitmask into 691 bytes."""
    base_107 = pack_position(board, action_index, eval_cp, wdl)
    if legal_mask is not None:
        if legal_mask.dtype != np.uint8 and legal_mask.dtype != bool:
            mask_bool = (legal_mask > 0.5)
        else:
            mask_bool = legal_mask
        packed_mask = np.packbits(mask_bool).tobytes()
    else:
        # Generate mask if not provided
        from neurachess.core.tensor_movegen import get_legal_action_mask
        m = get_legal_action_mask(board)
        packed_mask = np.packbits(m > 0.5).tobytes()
        
    assert len(packed_mask) == PACKED_MASK_SIZE, f"Expected {PACKED_MASK_SIZE} bytes, got {len(packed_mask)}"
    return base_107 + packed_mask


def unpack_position_v2(raw_bytes: bytes) -> Dict[str, Any]:
    """Unpack a 691-byte record into board, metadata, targets, and unpacked legal mask."""
    rec = unpack_position(raw_bytes[:RECORD_SIZE_V1])
    packed_mask = raw_bytes[RECORD_SIZE_V1:RECORD_SIZE_V2]
    mask = np.unpackbits(np.frombuffer(packed_mask, dtype=np.uint8)).astype(np.float32)
    rec["legal_mask"] = mask
    return rec


def fast_tensor_from_unpacked(bbs: tuple, turn: int, castling: int, ep_square: int, halfmove: int) -> np.ndarray:
    """
    Construct canonical 18-plane Markovian float32 tensor directly from bitboards in microseconds.
    Zero python-chess Board object construction overhead.
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    is_black = (turn == 1)

    for idx, bb in enumerate(bbs):
        if bb == 0:
            continue
        b_bytes = np.frombuffer(np.uint64(bb).tobytes(), dtype=np.uint8)
        bits = np.unpackbits(b_bytes, bitorder="little").reshape(8, 8)
        
        if not is_black:
            plane_idx = idx
        else:
            plane_idx = (idx - 6) if idx >= 6 else (idx + 6)
            bits = bits[::-1, :]  # Invert ranks for Black
            
        tensor[plane_idx] = bits.astype(np.float32)

    if not is_black:
        if castling & 1: tensor[12, :, :] = 1.0
        if castling & 2: tensor[13, :, :] = 1.0
        if castling & 4: tensor[14, :, :] = 1.0
        if castling & 8: tensor[15, :, :] = 1.0
    else:
        if castling & 4: tensor[12, :, :] = 1.0
        if castling & 8: tensor[13, :, :] = 1.0
        if castling & 1: tensor[14, :, :] = 1.0
        if castling & 2: tensor[15, :, :] = 1.0

    if ep_square != 255:
        r = ep_square // 8
        c = ep_square % 8
        if is_black:
            r = 7 - r
        tensor[16, r, c] = 1.0

    tensor[17, :, :] = float(halfmove) / 100.0
    return tensor

