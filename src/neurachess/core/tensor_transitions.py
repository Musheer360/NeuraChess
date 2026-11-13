"""
NeuraChess Tensor State Transition Operator T(S, a) -> S' (v3.0)

Computes the exact next 18-plane canonical board state tensor S' given
the current state tensor S and action index a in [0, 4671].

Handles:
  - Piece moves and captures
  - Castling rook displacements
  - En-passant captures and en-passant square creation
  - Pawn promotions (Queen, Knight, Bishop, Rook)
  - Castling rights invalidation
  - 50-move halfmove clock updates
  - Perspective flipping (canonical White/Black symmetry)
"""

import numpy as np
from .constants import NUM_PLANES, decode_action


def apply_tensor_transition(state: np.ndarray, action_idx: int) -> np.ndarray:
    """
    Given canonical state tensor S of shape (18, 8, 8) and action index a,
    returns next canonical state tensor S' of shape (18, 8, 8).
    """
    next_state = np.zeros_like(state)
    from_sq, to_sq, promo = decode_action(action_idx)

    from_r, from_c = from_sq // 8, from_sq % 8
    to_r, to_c = to_sq // 8, to_sq % 8

    # 1. Identify which friendly piece is moving (Planes 0..5: P, N, B, R, Q, K)
    moving_piece_idx = -1
    for p in range(6):
        if state[p, from_r, from_c] == 1.0:
            moving_piece_idx = p
            break

    if moving_piece_idx == -1:
        moving_piece_idx = 0

    # Copy current piece placements
    new_cur_pieces = np.copy(state[0:12])  # 6 friendly, 6 opponent

    # Remove moving piece from source
    new_cur_pieces[moving_piece_idx, from_r, from_c] = 0.0

    # Check for capture: remove any opponent piece at to_sq
    is_capture = False
    for opp_p in range(6):
        if new_cur_pieces[6 + opp_p, to_r, to_c] == 1.0:
            new_cur_pieces[6 + opp_p, to_r, to_c] = 0.0
            is_capture = True

    # Place moving piece at destination (handling promotion if applicable)
    if moving_piece_idx == 0 and to_r == 7:  # Pawn reaching 8th rank
        promo_piece = promo.lower() if promo else "q"
        promo_map = {"q": 4, "r": 3, "b": 2, "n": 1}
        new_piece_idx = promo_map.get(promo_piece, 4)
        new_cur_pieces[new_piece_idx, to_r, to_c] = 1.0
    else:
        new_cur_pieces[moving_piece_idx, to_r, to_c] = 1.0

    # Special Case: Castling (King moving 2 squares horizontally on rank 0)
    if moving_piece_idx == 5 and from_r == 0 and to_r == 0:
        if to_c == 6:  # Kingside (e1 -> g1): move Rook from h1 (0,7) to f1 (0,5)
            new_cur_pieces[3, 0, 7] = 0.0
            new_cur_pieces[3, 0, 5] = 1.0
        elif to_c == 2:  # Queenside (e1 -> c1): move Rook from a1 (0,0) to d1 (0,3)
            new_cur_pieces[3, 0, 0] = 0.0
            new_cur_pieces[3, 0, 3] = 1.0

    # Special Case: En-Passant capture
    if moving_piece_idx == 0 and from_c != to_c and state[16, to_r, to_c] == 1.0:
        new_cur_pieces[6, from_r, to_c] = 0.0
        is_capture = True

    temp_state = np.zeros_like(state)
    temp_state[0:12] = new_cur_pieces

    # 2. Castling rights update (Planes 12..15)
    f_oo = state[12, 0, 0]
    f_ooo = state[13, 0, 0]
    o_oo = state[14, 0, 0]
    o_ooo = state[15, 0, 0]

    # Friendly King moved
    if moving_piece_idx == 5:
        f_oo = 0.0
        f_ooo = 0.0

    # Friendly Rook moved
    if moving_piece_idx == 3:
        if from_r == 0 and from_c == 7:
            f_oo = 0.0
        elif from_r == 0 and from_c == 0:
            f_ooo = 0.0

    # Opponent Rook captured on initial squares (rank 7 in current perspective)
    if to_r == 7 and to_c == 7:
        o_oo = 0.0
    elif to_r == 7 and to_c == 0:
        o_ooo = 0.0

    temp_state[12, :, :] = f_oo
    temp_state[13, :, :] = f_ooo
    temp_state[14, :, :] = o_oo
    temp_state[15, :, :] = o_ooo

    # 3. En-Passant target plane (Plane 16)
    if moving_piece_idx == 0 and from_r == 1 and to_r == 3:
        temp_state[16, 2, from_c] = 1.0
    else:
        temp_state[16, :, :] = 0.0

    # 4. Halfmove clock (Plane 17)
    if moving_piece_idx == 0 or is_capture:
        temp_state[17, :, :] = 0.0
    else:
        temp_state[17, :, :] = min(1.0, state[17, 0, 0] + 0.01)

    # 5. CANONICAL INVERSION (Flip perspective for opponent's turn)
    # Flip piece planes: swap friendly (0..5) <-> opponent (6..11) and invert ranks
    next_state[0:6, :, :] = np.flip(temp_state[6:12, :, :], axis=1)
    next_state[6:12, :, :] = np.flip(temp_state[0:6, :, :], axis=1)

    # Swap castling rights: old opponent rights become new friendly rights
    next_state[12, :, :] = temp_state[14, 0, 0]
    next_state[13, :, :] = temp_state[15, 0, 0]
    next_state[14, :, :] = temp_state[12, 0, 0]
    next_state[15, :, :] = temp_state[13, 0, 0]

    # Flip en-passant plane vertically
    next_state[16, :, :] = np.flip(temp_state[16, :, :], axis=0)

    # Halfmove clock remains unchanged
    next_state[17, :, :] = temp_state[17, :, :]

    return next_state


def apply_batched_tensor_transitions(state_batch: np.ndarray, action_indices: list[int] | np.ndarray) -> np.ndarray:
    """Apply transitions for a batch of B positions and B actions."""
    B = len(action_indices)
    next_states = np.zeros((B, NUM_PLANES, 8, 8), dtype=np.float32)
    for b in range(B):
        next_states[b] = apply_tensor_transition(state_batch[b], int(action_indices[b]))
    return next_states
