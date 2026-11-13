"""
NeuraChess Core Constants and Action Space Mapping (v2.0)

Action space: 4,672 discrete slots (AlphaZero convention):
  64 from-squares x 73 action types per square:
    0..55:  Queen slides (8 compass directions x 7 distances)
    56..63: Knight jumps (8 coordinate offsets)
    64..72: Underpromotions (3 file directions x 3 promo pieces: N, B, R)
            (Queen promotions are standard distance-1 queen moves to 8th rank)
"""

import numpy as np

# Total dimensions
NUM_SQUARES = 64
NUM_ACTIONS_PER_SQUARE = 73
TOTAL_ACTIONS = NUM_SQUARES * NUM_ACTIONS_PER_SQUARE  # 4672
NUM_PLANES = 18

# Compass directions: (dr, dc) from friendly perspective (advancing towards +rank)
DIRECTIONS = [
    (+1,  0),  # N
    (+1, +1),  # NE
    ( 0, +1),  # E
    (-1, +1),  # SE
    (-1,  0),  # S
    (-1, -1),  # SW
    ( 0, -1),  # W
    (+1, -1),  # NW
]
DIR_NAME = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Knight jumps: (dr, dc)
KNIGHT_OFFSETS = [
    (+2, +1),
    (+1, +2),
    (-1, +2),
    (-2, +1),
    (-2, -1),
    (-1, -2),
    (+1, -2),
    (+2, -1),
]

# Underpromotions: direction index (0: left capture dc=-1, 1: straight push dc=0, 2: right capture dc=+1)
PROMO_DIR_DC = [-1, 0, +1]
PROMO_PIECES = ["n", "b", "r"]  # 0: Knight, 1: Bishop, 2: Rook

# Precomputed lookup tables for O(1) encoding / decoding
_ACTION_TO_MOVE = {}
_MOVE_TO_ACTION = {}


def _init_tables():
    for from_sq in range(64):
        from_r = from_sq // 8
        from_c = from_sq % 8

        # 1. Queen slides (0..55)
        for d_idx, (dr, dc) in enumerate(DIRECTIONS):
            for dist in range(1, 8):
                to_r = from_r + dr * dist
                to_c = from_c + dc * dist
                slot = d_idx * 7 + (dist - 1)
                action_idx = from_sq * 73 + slot

                if 0 <= to_r < 8 and 0 <= to_c < 8:
                    to_sq = to_r * 8 + to_c
                    # If this is a pawn moving from rank 6 (7th rank) to rank 7 (8th rank),
                    # it could be a Queen promotion. Queen promo shares the same slot as a normal move.
                    promo = "q" if from_r == 6 and to_r == 7 and dist == 1 and abs(dc) <= 1 else None
                    _ACTION_TO_MOVE[action_idx] = (from_sq, to_sq, promo)
                    _MOVE_TO_ACTION[(from_sq, to_sq, promo)] = action_idx
                    if promo == "q":
                        # Also register without promo for non-pawn moves (e.g. Rook, Queen)
                        _MOVE_TO_ACTION[(from_sq, to_sq, None)] = action_idx

        # 2. Knight jumps (56..63)
        for k_idx, (dr, dc) in enumerate(KNIGHT_OFFSETS):
            to_r = from_r + dr
            to_c = from_c + dc
            slot = 56 + k_idx
            action_idx = from_sq * 73 + slot

            if 0 <= to_r < 8 and 0 <= to_c < 8:
                to_sq = to_r * 8 + to_c
                _ACTION_TO_MOVE[action_idx] = (from_sq, to_sq, None)
                _MOVE_TO_ACTION[(from_sq, to_sq, None)] = action_idx

        # 3. Underpromotions (64..72)
        if from_r == 6:  # 7th rank in 0-indexed
            to_r = 7      # 8th rank
            for dir_idx, dc in enumerate(PROMO_DIR_DC):
                to_c = from_c + dc
                if 0 <= to_c < 8:
                    to_sq = to_r * 8 + to_c
                    for p_idx, piece in enumerate(PROMO_PIECES):
                        slot = 64 + 3 * dir_idx + p_idx
                        action_idx = from_sq * 73 + slot
                        _ACTION_TO_MOVE[action_idx] = (from_sq, to_sq, piece)
                        _MOVE_TO_ACTION[(from_sq, to_sq, piece)] = action_idx


_init_tables()


def encode_action(from_sq: int, to_sq: int, promo: str | None = None) -> int:
    """Map a (from_sq, to_sq, promo) tuple to action index in [0, 4671]."""
    key = (from_sq, to_sq, promo.lower() if promo else None)
    if key in _MOVE_TO_ACTION:
        return _MOVE_TO_ACTION[key]
    raise ValueError(f"Illegal move geometry: from={from_sq}, to={to_sq}, promo={promo}")


def decode_action(action_idx: int) -> tuple[int, int, str | None]:
    """Map action index in [0, 4671] to (from_sq, to_sq, promo)."""
    if action_idx in _ACTION_TO_MOVE:
        return _ACTION_TO_MOVE[action_idx]
    raise ValueError(f"Action index {action_idx} out of range or off-board.")


def uci_to_action(uci_str: str, canonical_flip: bool = False) -> int:
    """
    Convert a UCI move string (e.g. 'e2e4', 'e7e8q') to an action index.
    If canonical_flip is True (Black's turn), coordinates are flipped vertically.
    """
    from_file = ord(uci_str[0]) - ord('a')
    from_rank = int(uci_str[1]) - 1
    to_file = ord(uci_str[2]) - ord('a')
    to_rank = int(uci_str[3]) - 1
    promo = uci_str[4].lower() if len(uci_str) > 4 else None

    if canonical_flip:
        from_rank = 7 - from_rank
        to_rank = 7 - to_rank

    from_sq = from_rank * 8 + from_file
    to_sq = to_rank * 8 + to_file
    return encode_action(from_sq, to_sq, promo)


_PROMO_MAP = {2: 'n', 3: 'b', 4: 'r', 5: 'q'}  # chess.KNIGHT..QUEEN


def move_to_action(move: "chess.Move", canonical_flip: bool = False) -> int:
    """
    Directly converts a chess.Move object to an action index in [0, 4671],
    bypassing intermediate string formatting for zero-allocation performance.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promo = _PROMO_MAP.get(move.promotion, None) if move.promotion else None

    if canonical_flip:
        from_sq = (7 - (from_sq >> 3)) * 8 + (from_sq & 7)
        to_sq = (7 - (to_sq >> 3)) * 8 + (to_sq & 7)

    key = (from_sq, to_sq, promo)
    return _MOVE_TO_ACTION[key]


def action_to_uci(action_idx: int, canonical_flip: bool = False) -> str:
    """
    Convert an action index in [0, 4671] to a UCI string (e.g. 'e2e4').
    If canonical_flip is True (Black's turn), coordinates are flipped vertically back to standard UCI.
    """
    from_sq, to_sq, promo = decode_action(action_idx)
    from_file = from_sq % 8
    from_rank = from_sq // 8
    to_file = to_sq % 8
    to_rank = to_sq // 8

    if canonical_flip:
        from_rank = 7 - from_rank
        to_rank = 7 - to_rank

    res = f"{chr(ord('a') + from_file)}{from_rank + 1}{chr(ord('a') + to_file)}{to_rank + 1}"
    if promo:
        res += promo.lower()
    return res


# ---------------------------------------------------------------------------
# Canonical Value Mapping: cp <-> Expected Score V in [-1.0, +1.0]
# ---------------------------------------------------------------------------
K_VAL = np.log(10) / 800.0  # ~0.0028782313662
INV_K_VAL = 800.0 / np.log(10)  # ~347.4355855226


def centipawns_to_value(cp: float | int) -> float:
    """Converts Stockfish centipawns to side-to-move expected score V in [-1.0, +1.0]."""
    return float(np.tanh(float(cp) * K_VAL))


def value_to_centipawns(v: float, max_cp: int = 10000) -> int:
    """
    Exact mathematical inverse of V = tanh(cp * ln(10) / 800):
      cp = arctanh(clamp(V, -0.999999, 0.999999)) * (800 / ln(10))
    Clipped to [-max_cp, max_cp] to avoid infinities on decisive/terminal values.
    """
    v_clamped = np.clip(float(v), -0.999999, 0.999999)
    cp = np.arctanh(v_clamped) * INV_K_VAL
    return int(np.clip(round(cp), -max_cp, max_cp))

