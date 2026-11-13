"""
Compiled Zero-GIL C++23 PUCT Tree, 64B-Aligned SoA AVX2 Selector, P-Core Pinning &
Zero-Copy OpenVINO Staging Wrapper (engine/fast_tree/__init__.py)
==================================================================================
Compiles `fast_tree.cpp` + `bitboard_chess.hpp` with Zig/Clang 22 C++23:
  `-std=c++23 -O3 -ffast-math -fno-finite-math-only -march=native -shared`
(or `-mcpu=x86_64_v3` when `NEURACHESS_ARCH=x86-64-v3`)
and exposes `FastTreeSearchBackend` with:
  - 100% unpruned legal move expansion at root and all internal nodes
  - 64-byte cache-line aligned Struct-of-Arrays (`alignas(64)`) AVX2 selection
  - Core Ultra 5 125U Hybrid P-core thread pinning + HighQoS (`SetThreadInformation`)
  - Zero-copy OpenVINO `InferRequest` input/output tensor memory staging
  - Native C++23 background pondering loop for `PonderManager`
"""

import os
import sys
import time
import math
import ctypes
import threading
import subprocess
import numpy as np
import chess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.constants import TOTAL_ACTIONS, NUM_PLANES, uci_to_action, action_to_uci, move_to_action
from core.board_tensor import board_to_tensor, ENCODER_LAYOUT_GPU_UNPACK
from search.candidate_selector import PIECE_VALUES, select_opening_guarded_root_act

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CPP_PATH = os.path.join(_THIS_DIR, "fast_tree.cpp")
_HPP_PATH = os.path.join(_THIS_DIR, "bitboard_chess.hpp")
def _resolve_compile_spec() -> tuple[str, str, list[str], str]:
    import hashlib
    import json

    arch_env = os.environ.get("NEURACHESS_ARCH", "native").strip() or "native"
    if arch_env in ("x86-64-v3", "x86_64_v3"):
        arch_flag = "-mcpu=x86_64_v3"
    elif arch_env == "native":
        arch_flag = "-march=native"
    else:
        arch_flag = f"-mcpu={arch_env}"

    flags = ["-std=c++23", "-O3", "-ffast-math", "-fno-finite-math-only", arch_flag]
    sig_payload = json.dumps(
        {
            "arch_env": arch_env,
            "flags": flags,
            "platform": sys.platform,
        },
        sort_keys=True,
    )
    sig_hash = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:10]
    safe_arch = "".join(c if (c.isalnum() or c in "-_") else "_" for c in arch_env)
    ext = ".dll" if os.name == "nt" else ".so"
    prefix = "fast_tree" if os.name == "nt" else "libfast_tree"
    dll_path = os.path.join(_THIS_DIR, f"{prefix}_{safe_arch}_{sig_hash}{ext}")
    stamp_path = f"{dll_path}.stamp"
    cmd = [
        sys.executable,
        "-m",
        "ziglang",
        "c++",
        *flags,
        "-shared",
        "-o",
        dll_path,
        _CPP_PATH,
    ]
    return dll_path, stamp_path, cmd, sig_payload


_DLL_PATH, _STAMP_PATH, _, _ = _resolve_compile_spec()
_DLL_NAME = os.path.basename(_DLL_PATH)

_LIB = None
_PINNED_THREADS: dict[int, dict] = {}
_PIN_LOCK = threading.Lock()


def ensure_compiled(force: bool = False) -> str:
    dll_path, stamp_path, cmd, sig_payload = _resolve_compile_spec()
    src_mtimes = [
        os.path.getmtime(p)
        for p in (_CPP_PATH, _HPP_PATH)
        if os.path.exists(p)
    ]
    max_src_mtime = max(src_mtimes) if src_mtimes else 0.0

    if not force and os.path.exists(dll_path) and os.path.exists(stamp_path):
        try:
            with open(stamp_path, "r", encoding="utf-8") as sf:
                recorded_sig = sf.read().strip()
        except OSError:
            recorded_sig = ""
        dll_mtime = os.path.getmtime(dll_path)
        if recorded_sig == sig_payload and (not src_mtimes or dll_mtime >= max_src_mtime):
            return dll_path

    # Single-TU unity build (fast_tree.cpp includes bitboard_chess.hpp directly),
    # giving Clang 22 full whole-module IPO/inlining at -O3.
    # Keep -fno-finite-math-only so std::isnan()/std::isinf() guards remain active under -ffast-math.
    subprocess.check_call(cmd)
    with open(stamp_path, "w", encoding="utf-8") as sf:
        sf.write(sig_payload)
    return dll_path


def get_lib() -> ctypes.CDLL:
    global _LIB
    dll_path = ensure_compiled(force=False)
    if _LIB is not None and getattr(_LIB, "_name", None) == dll_path:
        return _LIB

    lib = ctypes.CDLL(dll_path)

    # P-Core Pinning & HighQoS API
    lib.ft_pin_current_thread_to_pcores.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ft_pin_current_thread_to_pcores.restype = ctypes.c_int32

    # Cache API
    lib.ft_cache_create.argtypes = [ctypes.c_uint64]
    lib.ft_cache_create.restype = ctypes.c_void_p
    lib.ft_cache_destroy.argtypes = [ctypes.c_void_p]
    lib.ft_cache_destroy.restype = None
    lib.ft_cache_clear.argtypes = [ctypes.c_void_p]
    lib.ft_cache_clear.restype = None
    lib.ft_cache_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ft_cache_stats.restype = None
    lib.ft_cache_contains.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    lib.ft_cache_contains.restype = ctypes.c_int32

    # Tree API
    lib.ft_tree_create.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
    ]
    lib.ft_tree_create.restype = ctypes.c_void_p
    lib.ft_tree_destroy.argtypes = [ctypes.c_void_p]
    lib.ft_tree_destroy.restype = None
    lib.ft_tree_reset_root.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
    lib.ft_tree_reset_root.restype = ctypes.c_int32
    lib.ft_tree_reset_stats.argtypes = [ctypes.c_void_p]
    lib.ft_tree_reset_stats.restype = None
    lib.ft_tree_num_nodes.argtypes = [ctypes.c_void_p]
    lib.ft_tree_num_nodes.restype = ctypes.c_int32
    lib.ft_tree_set_root_fpu_visit.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.ft_tree_set_root_fpu_visit.restype = None
    lib.ft_clear_all_vl.argtypes = [ctypes.c_void_p]
    lib.ft_clear_all_vl.restype = None

    lib.ft_expand_root_from_cache.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ft_expand_root_from_cache.restype = ctypes.c_int32

    lib.ft_expand_node_with_logits.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.ft_expand_node_with_logits.restype = ctypes.c_double

    lib.ft_select_begin.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.ft_select_begin.restype = ctypes.c_int32

    lib.ft_select_provide_instantiation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.ft_select_provide_instantiation.restype = ctypes.c_int32

    lib.ft_pop_inflight_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ft_pop_inflight_batch.restype = ctypes.c_void_p

    lib.ft_revert_batch_vl_and_clear_pending.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.ft_revert_batch_vl_and_clear_pending.restype = None

    lib.ft_backup_single_leaf_from_batch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.ft_backup_single_leaf_from_batch.restype = None

    lib.ft_destroy_inflight_batch.argtypes = [ctypes.c_void_p]
    lib.ft_destroy_inflight_batch.restype = None

    lib.ft_get_stats.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ft_get_stats.restype = None

    lib.ft_get_node_children_info.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ft_get_node_children_info.restype = ctypes.c_int32

    lib.ft_reroot_compact.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
    ]
    lib.ft_reroot_compact.restype = ctypes.c_int32

    lib.ft_get_instantiated_preorder.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
    ]
    lib.ft_get_instantiated_preorder.restype = ctypes.c_int32

    lib.ft_invalidate_reused_node.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
    ]
    lib.ft_invalidate_reused_node.restype = None

    lib.ft_finalize_reused_subtree.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.ft_finalize_reused_subtree.restype = None

    lib.ft_encode_gpu_unpack_single.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ft_encode_gpu_unpack_single.restype = None

    # Pure C++23 Zero-GIL Native Bitboard Pipeline
    lib.ft_init_action_table.argtypes = [ctypes.POINTER(ctypes.c_uint16)]
    lib.ft_init_action_table.restype = None

    lib.ft_clear_seen_keys.argtypes = [ctypes.c_void_p]
    lib.ft_clear_seen_keys.restype = None

    lib.ft_compute_cboard_zobrist.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ft_compute_cboard_zobrist.restype = None

    lib.ft_set_root_native.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ft_set_root_native.restype = None

    lib.ft_encode_root_native.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
    lib.ft_encode_root_native.restype = None

    lib.ft_try_reuse_root_native.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.ft_try_reuse_root_native.restype = ctypes.c_int32

    lib.ft_record_seen_root_native.argtypes = [ctypes.c_void_p]
    lib.ft_record_seen_root_native.restype = ctypes.c_int32

    lib.ft_expand_root_native.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    lib.ft_expand_root_native.restype = ctypes.c_double

    lib.ft_select_and_pack_native.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ft_select_and_pack_native.restype = ctypes.c_void_p

    lib.ft_backup_batch_native.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ft_backup_batch_native.restype = None

    lib.ft_test_generate_moves_native.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.ft_test_generate_moves_native.restype = ctypes.c_int32

    lib.ft_perft_native.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.ft_perft_native.restype = ctypes.c_uint64

    lib.ft_test_push_chain_native.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.ft_test_push_chain_native.restype = ctypes.c_int32

    lib.ft_test_math_guards_active.argtypes = [ctypes.c_double, ctypes.c_float]
    lib.ft_test_math_guards_active.restype = ctypes.c_int32

    # Populate 2 x 64 x 64 x 6 action lookup table once on DLL load
    act_arr = np.zeros((2, 64, 64, 6), dtype=np.uint16)
    for is_blk in (0, 1):
        cflip = bool(is_blk)
        for f_sq in range(64):
            for t_sq in range(64):
                if f_sq == t_sq:
                    continue
                for promo in (0, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
                    mv = chess.Move(f_sq, t_sq, promo if promo else None)
                    try:
                        a = move_to_action(mv, canonical_flip=cflip)
                        act_arr[is_blk, f_sq, t_sq, promo] = a
                    except KeyError:
                        pass
    lib.ft_init_action_table(act_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)))

    _LIB = lib
    return lib


def pin_current_thread_to_pcores(force: bool = False) -> dict:
    """
    Pins the calling OS thread to the fastest P-cores (highest Win32 `EfficiencyClass`)
    and enables HighQoS (`SetThreadInformation` `ThreadPowerThrottling` `StateMask=0`)
    so Windows 11 never migrates search/ponder workers onto slow LP-E cores.
    """
    tid = threading.get_ident()
    if not force:
        cached = _PINNED_THREADS.get(tid)
        if cached is not None:
            return cached

    lib = get_lib()
    mask_u64 = ctypes.c_uint64(0)
    max_eff = ctypes.c_int32(0)
    qos_ok = ctypes.c_int32(0)
    pinned = lib.ft_pin_current_thread_to_pcores(
        ctypes.byref(mask_u64),
        ctypes.byref(max_eff),
        ctypes.byref(qos_ok),
    )
    info = {
        "thread_id": tid,
        "pinned_ok": bool(pinned),
        "pcore_affinity_mask": int(mask_u64.value),
        "pcore_affinity_hex": hex(int(mask_u64.value)),
        "max_efficiency_class": int(max_eff.value),
        "high_qos_enabled": bool(qos_ok.value),
    }
    with _PIN_LOCK:
        _PINNED_THREADS[tid] = info
    return info


def pack_move(m: chess.Move) -> int:
    return m.from_square | (m.to_square << 6) | ((m.promotion or 0) << 12)


def unpack_move(mp: int) -> chess.Move:
    promo = (mp >> 12) & 0xF
    return chess.Move(mp & 0x3F, (mp >> 6) & 0x3F, promo if promo else None)


def c_perft(board_fen: str | chess.Board, depth: int) -> int:
    """Runs native C++23 bitboard perft(depth) from a FEN string or chess.Board."""
    board = board_fen if isinstance(board_fen, chess.Board) else chess.Board(board_fen)
    lib = get_lib()
    bbs = (ctypes.c_uint64 * 8)(
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
    )
    ep_sq = -1 if board.ep_square is None else int(board.ep_square)
    turn_w = 1 if board.turn == chess.WHITE else 0
    return int(lib.ft_perft_native(bbs, ctypes.c_uint64(board.castling_rights), ep_sq, turn_w, int(depth)))


def compute_zobrist_pair(board: chess.Board) -> tuple[int, int, tuple]:
    t_key = board._transposition_key()
    h1 = hash(t_key) & 0xFFFFFFFFFFFFFFFF
    h1 = ((h1 ^ (h1 >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    h1 = ((h1 ^ (h1 >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    h1 = (h1 ^ (h1 >> 31)) & 0xFFFFFFFFFFFFFFFF

    occ_w = board.occupied_co[chess.WHITE]
    occ_b = board.occupied_co[chess.BLACK]
    h2 = (
        occ_w
        ^ ((occ_b * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
        ^ (board.pawns << 1)
        ^ (board.kings << 7)
    ) & 0xFFFFFFFFFFFFFFFF
    if board.turn:
        h2 ^= 0xF00DBAADBEEF1234
    return h1, h2, t_key


def evaluate_terminal_flags_fast(b: chess.Board) -> tuple[int, float, int, float, int]:
    if b.halfmove_clock >= 100:
        return 1, 0.0, 0, 0.0, 1
    if b.is_insufficient_material():
        return 1, 0.0, 0, 0.0, 0
    if b.is_repetition(2):
        return 1, 0.0, 0, 0.0, 1
    if not any(b.generate_legal_moves()):
        if b.is_check():
            return 1, -1.0, 1, -1.0, 0
        return 1, 0.0, 0, 0.0, 0
    return 0, 0.0, 0, 0.0, 0


class FastLruEvalCache:
    """Python wrapper around native C++23 64-bit Zobrist LruEvalCache."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024):
        self.lib = get_lib()
        self.max_bytes = int(max_bytes)
        self._ptr = self.lib.ft_cache_create(ctypes.c_uint64(self.max_bytes))
        self._u64_a = ctypes.c_uint64(0)
        self._u64_b = ctypes.c_uint64(0)
        self._u64_c = ctypes.c_uint64(0)
        self._u64_d = ctypes.c_uint64(0)
        self._u64_e = ctypes.c_uint64(0)

    def __del__(self):
        if getattr(self, "_ptr", None):
            try:
                self.lib.ft_cache_destroy(self._ptr)
            except Exception:
                pass
            self._ptr = None

    def clear(self):
        if self._ptr:
            self.lib.ft_cache_clear(self._ptr)

    def stats(self) -> tuple[int, int, int, int, int]:
        self.lib.ft_cache_stats(
            self._ptr,
            ctypes.byref(self._u64_a),
            ctypes.byref(self._u64_b),
            ctypes.byref(self._u64_c),
            ctypes.byref(self._u64_d),
            ctypes.byref(self._u64_e),
        )
        return (
            int(self._u64_a.value),
            int(self._u64_b.value),
            int(self._u64_c.value),
            int(self._u64_d.value),
            int(self._u64_e.value),
        )

    @property
    def used_bytes(self) -> int:
        return self.stats()[1]

    @property
    def hits(self) -> int:
        return self.stats()[2]

    @property
    def misses(self) -> int:
        return self.stats()[3]

    @property
    def evictions(self) -> int:
        return self.stats()[4]

    def __len__(self) -> int:
        return self.stats()[0]


class FastTreeSearchBackend:
    """
    Native C++23 PUCT Search Arena + 64B-Aligned SoA AVX2 + Zero-Copy OpenVINO Backend.
    """

    MAX_MOVES = 256
    MAX_BATCH = 128

    def __init__(
        self,
        batch_size: int = 16,
        c_puct: float = 1.5,
        use_fpu_reduction: bool = True,
        fpu_reduction: float = 0.20,
        root_fpu_reduction: float = 0.10,
        use_exact_solver: bool = True,
        use_gumbel_root: bool = False,
        use_async_overlap: bool = False,
        async_inflight: int = 2,
        use_tree_reuse: bool = True,
        use_nn_cache: bool = True,
        nn_cache_max_bytes: int = 64 * 1024 * 1024,
        encoder_layout: str = ENCODER_LAYOUT_GPU_UNPACK,
        t_root: float = 1.0,
        use_opening_root_guard: bool = False,
    ):
        self.lib = get_lib()
        pin_current_thread_to_pcores()

        self.batch_size = batch_size
        self.c_puct = c_puct
        self.t_root = float(t_root)
        self.use_opening_root_guard = bool(use_opening_root_guard)
        self.use_fpu_reduction = use_fpu_reduction
        self.fpu_reduction = fpu_reduction
        self.root_fpu_reduction = root_fpu_reduction
        self.use_exact_solver = use_exact_solver
        self.use_gumbel_root = use_gumbel_root
        self.use_async_overlap = use_async_overlap
        self.async_inflight = max(1, int(async_inflight))
        self.use_tree_reuse = use_tree_reuse
        self.use_nn_cache = use_nn_cache
        self.encoder_layout = encoder_layout

        self.cache = FastLruEvalCache(max_bytes=nn_cache_max_bytes)
        self._tree_ptr = self.lib.ft_tree_create(
            self.cache._ptr,
            ctypes.c_double(c_puct),
            ctypes.c_int32(1 if use_fpu_reduction else 0),
            ctypes.c_double(fpu_reduction),
            ctypes.c_double(root_fpu_reduction),
            ctypes.c_int32(1 if use_exact_solver else 0),
        )

        self._boards: dict[int, chess.Board] = {}
        self._keys: dict[int, tuple[int, int, tuple]] = {}
        self._tensors: dict[int, np.ndarray] = {}
        self._has_saved_tree = False
        self._seen_npu_keys: set[tuple] = set()
        self._extra_requests: list = []
        # Zero-copy OpenVINO InferRequest pointer cache keyed by id(req)
        self._zc_req_cache: dict[int, tuple] = {}

        # Pre-allocated reusable ctypes buffers (zero per-step allocations)
        self._c_node = ctypes.c_int32(0)
        self._c_parent = ctypes.c_int32(0)
        self._c_move = ctypes.c_uint16(0)
        self._c_n_eval = ctypes.c_int32(0)
        self._c_max_ply = ctypes.c_int32(0)
        self._c_root_val = ctypes.c_double(0.0)

        self._leaves_buf = (ctypes.c_int32 * self.MAX_BATCH)()
        self._acts_buf = (ctypes.c_uint16 * self.MAX_MOVES)()
        self._moves_buf = (ctypes.c_uint16 * self.MAX_MOVES)()
        self._bonuses_buf = (ctypes.c_double * self.MAX_MOVES)()
        self._bbs_buf = (ctypes.c_uint64 * 12)()
        self._bbs8_buf = (ctypes.c_uint64 * 8)()
        self._tmp_bbs8_buf = (ctypes.c_uint64 * 8)()
        self._rev_hi_buf = (ctypes.c_uint64 * 104)()
        self._rev_lo_buf = (ctypes.c_uint64 * 104)()
        self._delta_mp_buf = (ctypes.c_uint16 * 8)()
        self._c_u64_a = ctypes.c_uint64(0)
        self._c_u64_b = ctypes.c_uint64(0)
        self._c_i32_a = ctypes.c_int32(0)
        self._c_i32_b = ctypes.c_int32(0)
        self._c_i32_c = ctypes.c_int32(0)
        self._saved_move_stack: list[chess.Move] | None = None
        self._bt_ring = [np.zeros((self.batch_size, NUM_PLANES, 8, 8), dtype=np.float32) for _ in range(8)]
        self._bt_ring_idx = 0
        self.last_root_visits: dict[int, int] = {}

        self._ch_idx_buf = (ctypes.c_int32 * self.MAX_MOVES)()
        self._ch_act_buf = (ctypes.c_int32 * self.MAX_MOVES)()
        self._ch_mp_buf = (ctypes.c_uint16 * self.MAX_MOVES)()
        self._ch_vis_buf = (ctypes.c_int32 * self.MAX_MOVES)()
        self._ch_q_buf = (ctypes.c_double * self.MAX_MOVES)()
        self._ch_pr_buf = (ctypes.c_double * self.MAX_MOVES)()
        self._ch_slv_buf = (ctypes.c_uint8 * self.MAX_MOVES)()
        self._ch_sval_buf = (ctypes.c_double * self.MAX_MOVES)()
        self._ch_mplies_buf = (ctypes.c_int32 * self.MAX_MOVES)()

        self._st_i64 = [ctypes.c_int64(0) for _ in range(6)]
        self._st_i32 = [ctypes.c_int32(0) for _ in range(2)]

    def __del__(self):
        if getattr(self, "_tree_ptr", None):
            try:
                self.lib.ft_tree_destroy(self._tree_ptr)
            except Exception:
                pass
            self._tree_ptr = None

    def reconfigure(
        self,
        c_puct: float | None = None,
        use_fpu_reduction: bool | None = None,
        fpu_reduction: float | None = None,
        root_fpu_reduction: float | None = None,
        t_root: float | None = None,
        use_opening_root_guard: bool | None = None,
    ):
        if c_puct is not None:
            self.c_puct = float(c_puct)
        if use_fpu_reduction is not None:
            self.use_fpu_reduction = bool(use_fpu_reduction)
        if fpu_reduction is not None:
            self.fpu_reduction = float(fpu_reduction)
        if root_fpu_reduction is not None:
            self.root_fpu_reduction = float(root_fpu_reduction)
        if t_root is not None:
            self.t_root = float(t_root)
        if use_opening_root_guard is not None:
            self.use_opening_root_guard = bool(use_opening_root_guard)

        if getattr(self, "_tree_ptr", None):
            try:
                self.lib.ft_tree_destroy(self._tree_ptr)
            except Exception:
                pass
            self._tree_ptr = None

        self._tree_ptr = self.lib.ft_tree_create(
            self.cache._ptr,
            ctypes.c_double(self.c_puct),
            ctypes.c_int32(1 if self.use_fpu_reduction else 0),
            ctypes.c_double(self.fpu_reduction),
            ctypes.c_double(self.root_fpu_reduction),
            ctypes.c_int32(1 if self.use_exact_solver else 0),
        )
        self.clear_state()

    def clear_state(self):
        self._boards.clear()
        self._keys.clear()
        self._tensors.clear()
        self._has_saved_tree = False
        self._saved_move_stack = None
        self.cache.clear()
        self._seen_npu_keys.clear()
        if self._tree_ptr:
            self.lib.ft_clear_seen_keys(self._tree_ptr)

    def _bind_zero_copy_req(self, req):
        """
        Extracts and caches stable C pointers directly into OpenVINO `InferRequest`'s
        64-byte aligned unified host input & output tensors, eliminating all NumPy copies
        and Python dict allocations on dispatch and output retrieval.
        """
        rid = id(req)
        entry = self._zc_req_cache.get(rid)
        if entry is not None and entry[0] is req:
            return entry

        try:
            in_t = req.get_input_tensor()
            out0_t = req.get_output_tensor(0)
            out1_t = req.get_output_tensor(1)
            in_arr = in_t.data
            out0_arr = out0_t.data
            out1_arr = out1_t.data
            if (
                in_arr.dtype == np.float32
                and in_arr.flags["C_CONTIGUOUS"]
                and out0_arr.dtype == np.float32
                and out0_arr.flags["C_CONTIGUOUS"]
                and out1_arr.dtype == np.float32
                and out1_arr.flags["C_CONTIGUOUS"]
            ):
                in_ptr = in_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                out0_ptr = out0_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                out1_ptr = out1_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                val_cols = int(out1_arr.shape[1]) if out1_arr.ndim == 2 else 1
                entry = (req, in_ptr, out0_ptr, out1_ptr, val_cols, in_arr, out0_arr, out1_arr)
                self._zc_req_cache[rid] = entry
                return entry
        except Exception:
            pass
        return None

    def _extract_cboard_args(self, board: chess.Board) -> tuple[int, int, int, int, int]:
        bbs = self._bbs8_buf
        bbs[0] = board.pawns
        bbs[1] = board.knights
        bbs[2] = board.bishops
        bbs[3] = board.rooks
        bbs[4] = board.queens
        bbs[5] = board.kings
        bbs[6] = board.occupied_co[chess.WHITE]
        bbs[7] = board.occupied_co[chess.BLACK]

        rev_len = 0
        if board.halfmove_clock > 0 and len(board.move_stack) > 0:
            tmp = board.copy()
            tbbs = self._tmp_bbs8_buf
            while len(tmp.move_stack) > 0 and rev_len < 104:
                mv = tmp.pop()
                if tmp.is_irreversible(mv):
                    break
                tbbs[0] = tmp.pawns
                tbbs[1] = tmp.knights
                tbbs[2] = tmp.bishops
                tbbs[3] = tmp.rooks
                tbbs[4] = tmp.queens
                tbbs[5] = tmp.kings
                tbbs[6] = tmp.occupied_co[chess.WHITE]
                tbbs[7] = tmp.occupied_co[chess.BLACK]
                self.lib.ft_compute_cboard_zobrist(
                    tbbs,
                    ctypes.c_uint64(tmp.castling_rights),
                    -1 if tmp.ep_square is None else int(tmp.ep_square),
                    1 if tmp.turn == chess.WHITE else 0,
                    ctypes.byref(self._c_u64_a),
                    ctypes.byref(self._c_u64_b),
                )
                self._rev_hi_buf[rev_len] = self._c_u64_a.value
                self._rev_lo_buf[rev_len] = self._c_u64_b.value
                rev_len += 1
            for i in range(rev_len // 2):
                j = rev_len - 1 - i
                self._rev_hi_buf[i], self._rev_hi_buf[j] = self._rev_hi_buf[j], self._rev_hi_buf[i]
                self._rev_lo_buf[i], self._rev_lo_buf[j] = self._rev_lo_buf[j], self._rev_lo_buf[i]

        return (
            int(board.castling_rights),
            -1 if board.ep_square is None else int(board.ep_square),
            1 if board.turn == chess.WHITE else 0,
            int(board.halfmove_clock),
            rev_len,
        )

    def select_and_pack_native(
        self,
        max_depth: int,
        use_pending: bool = False,
        target_in_ptr=None,
    ) -> tuple[ctypes.c_void_p, np.ndarray | None, int, int]:
        if target_in_ptr is not None:
            bt = None
            dst_ptr = target_in_ptr
        else:
            bt = self._bt_ring[self._bt_ring_idx]
            self._bt_ring_idx = (self._bt_ring_idx + 1) & 7
            dst_ptr = bt.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        handle = self.lib.ft_select_and_pack_native(
            self._tree_ptr,
            self.batch_size,
            max_depth,
            1 if self.use_nn_cache else 0,
            1 if use_pending else 0,
            dst_ptr,
            ctypes.byref(self._c_n_eval),
            ctypes.byref(self._c_max_ply),
        )
        return handle, bt, int(self._c_n_eval.value), int(self._c_max_ply.value)

    def backup_batch_native_ptrs(
        self,
        handle: ctypes.c_void_p,
        logits_ptr,
        vals_ptr,
        val_cols: int,
        max_depth: int,
        prof: dict,
    ):
        store_cache = 1 if (self.use_nn_cache or self.use_tree_reuse) else 0
        self.lib.ft_backup_batch_native(
            self._tree_ptr,
            handle,
            logits_ptr,
            vals_ptr,
            val_cols,
            max_depth,
            store_cache,
            ctypes.byref(self._c_i32_a),
            ctypes.byref(self._c_i32_b),
        )
        prof["novel_npu_evals"] += int(self._c_i32_a.value)
        prof["wasted_dup_key_evals"] += int(self._c_i32_b.value)

    def backup_batch_native(
        self,
        handle: ctypes.c_void_p,
        p_logits_batch: np.ndarray,
        val_batch: np.ndarray,
        max_depth: int,
        prof: dict,
    ):
        logits_c = (
            p_logits_batch
            if (p_logits_batch.flags["C_CONTIGUOUS"] and p_logits_batch.dtype == np.float32)
            else np.ascontiguousarray(p_logits_batch, dtype=np.float32)
        )
        vals_c = (
            val_batch
            if (val_batch.flags["C_CONTIGUOUS"] and val_batch.dtype == np.float32)
            else np.ascontiguousarray(val_batch, dtype=np.float32)
        )
        val_cols = int(vals_c.shape[1]) if vals_c.ndim == 2 else 1
        self.backup_batch_native_ptrs(
            handle,
            logits_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vals_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            val_cols,
            max_depth,
            prof,
        )

    def _encode_board_fast(self, board: chess.Board) -> np.ndarray:
        if self.encoder_layout != ENCODER_LAYOUT_GPU_UNPACK:
            return board_to_tensor(board, layout=self.encoder_layout)

        arr = np.empty((NUM_PLANES, 8, 8), dtype=np.float32)
        occ_w = board.occupied_co[chess.WHITE]
        occ_b = board.occupied_co[chess.BLACK]
        bbs = self._bbs_buf
        bbs[0] = occ_w & board.pawns
        bbs[1] = occ_w & board.knights
        bbs[2] = occ_w & board.bishops
        bbs[3] = occ_w & board.rooks
        bbs[4] = occ_w & board.queens
        bbs[5] = occ_w & board.kings
        bbs[6] = occ_b & board.pawns
        bbs[7] = occ_b & board.knights
        bbs[8] = occ_b & board.bishops
        bbs[9] = occ_b & board.rooks
        bbs[10] = occ_b & board.queens
        bbs[11] = occ_b & board.kings

        if not board.chess960:
            cr = board.castling_rights
            cmask = ((cr >> 7) & 1) | ((cr & 1) << 1) | (((cr >> 63) & 1) << 2) | (((cr >> 56) & 1) << 3)
        else:
            cmask = (
                (1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
                | (2 if board.has_queenside_castling_rights(chess.WHITE) else 0)
                | (4 if board.has_kingside_castling_rights(chess.BLACK) else 0)
                | (8 if board.has_queenside_castling_rights(chess.BLACK) else 0)
            )

        self.lib.ft_encode_gpu_unpack_single(
            bbs,
            1 if board.turn == chess.BLACK else 0,
            int(cmask),
            int(board.halfmove_clock),
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return arr

    def _fill_legal_move_buffers(self, board: chess.Board, legals: list[chess.Move] | None = None) -> int:
        if legals is None:
            legals = list(board.legal_moves)
        n_leg = len(legals)
        if n_leg == 0:
            return 0

        is_blk = board.turn == chess.BLACK
        acts_b = self._acts_buf
        moves_b = self._moves_buf
        bon_b = self._bonuses_buf

        for idx, mv in enumerate(legals):
            acts_b[idx] = move_to_action(mv, canonical_flip=is_blk)
            moves_b[idx] = pack_move(mv)
            bon_b[idx] = 0.0
        return n_leg

    def _get_children_snapshot(self, node_idx: int) -> list[dict]:
        nc = self.lib.ft_get_node_children_info(
            self._tree_ptr,
            node_idx,
            self._ch_idx_buf,
            self._ch_act_buf,
            self._ch_mp_buf,
            self._ch_vis_buf,
            self._ch_q_buf,
            self._ch_pr_buf,
            self._ch_slv_buf,
            self._ch_sval_buf,
            self._ch_mplies_buf,
        )
        out = []
        for i in range(nc):
            out.append(
                {
                    "idx": int(self._ch_idx_buf[i]),
                    "act": int(self._ch_act_buf[i]),
                    "mp": int(self._ch_mp_buf[i]),
                    "visits": int(self._ch_vis_buf[i]),
                    "q": float(self._ch_q_buf[i]),
                    "prior": float(self._ch_pr_buf[i]),
                    "is_solved": bool(self._ch_slv_buf[i]),
                    "solved_val": float(self._ch_sval_buf[i]),
                    "mate_plies": int(self._ch_mplies_buf[i]),
                }
            )
        return out

    def _try_reuse_root(self, board: chess.Board, board_tensor: np.ndarray, max_depth: int, prof: dict) -> bool:
        if not self._has_saved_tree or 0 not in self._boards:
            return False

        old_board = self._boards[0]
        target_hi, target_lo, target_tkey = compute_zobrist_pair(board)
        found_idx = -1

        n_old = len(old_board.move_stack)
        n_new = len(board.move_stack)
        if n_new > n_old and (n_new - n_old) <= 4 and board.move_stack[:n_old] == old_board.move_stack:
            curr_idx = 0
            ok = True
            for mv in board.move_stack[n_old:]:
                if curr_idx not in self._boards:
                    ok = False
                    break
                cb = self._boards[curr_idx]
                act = move_to_action(mv, canonical_flip=(cb.turn == chess.BLACK))
                children = self._get_children_snapshot(curr_idx)
                matched = next((c for c in children if c["act"] == act), None)
                if matched is None:
                    ok = False
                    break
                next_idx = matched["idx"]
                if next_idx not in self._boards:
                    nb = cb.copy()
                    nb.push(unpack_move(matched["mp"]))
                    self._boards[next_idx] = nb
                    self._keys[next_idx] = compute_zobrist_pair(nb)
                curr_idx = next_idx
            if ok and curr_idx in self._boards:
                cb = self._boards[curr_idx]
                if cb._transposition_key() == target_tkey and cb.halfmove_clock == board.halfmove_clock:
                    found_idx = curr_idx

        if found_idx < 0:
            ch1_list = self._get_children_snapshot(0)
            for c1 in ch1_list:
                m1 = unpack_move(c1["mp"])
                b1 = old_board.copy(stack=False)
                b1.push(m1)
                if b1.turn == board.turn and b1._transposition_key() == target_tkey and b1.halfmove_clock == board.halfmove_clock:
                    found_idx = c1["idx"]
                    break
                ch2_list = self._get_children_snapshot(c1["idx"])
                for c2 in ch2_list:
                    m2 = unpack_move(c2["mp"])
                    b2 = b1.copy(stack=False)
                    b2.push(m2)
                    if b2.turn == board.turn and b2._transposition_key() == target_tkey and b2.halfmove_clock == board.halfmove_clock:
                        found_idx = c2["idx"]
                        break
                if found_idx >= 0:
                    break

        if found_idx < 0:
            return False

        old_total = self.lib.ft_tree_num_nodes(self._tree_ptr)
        old_map_buf = (ctypes.c_int32 * max(16, old_total))()
        n_surv = self.lib.ft_reroot_compact(self._tree_ptr, found_idx, old_map_buf, max(16, old_total))
        if n_surv <= 0:
            return False

        new_boards: dict[int, chess.Board] = {}
        new_keys: dict[int, tuple[int, int, tuple]] = {}
        new_tensors: dict[int, np.ndarray] = {}
        for new_i in range(n_surv):
            old_i = int(old_map_buf[new_i])
            if old_i in self._boards:
                new_boards[new_i] = self._boards[old_i]
            if old_i in self._keys:
                new_keys[new_i] = self._keys[old_i]
            if old_i in self._tensors:
                new_tensors[new_i] = self._tensors[old_i]

        stack_matches = (
            0 in new_boards
            and len(new_boards[0].move_stack) == len(board.move_stack)
            and new_boards[0].move_stack == board.move_stack
        )

        new_boards[0] = board.copy()
        new_keys[0] = (target_hi, target_lo, target_tkey)
        new_tensors[0] = board_tensor

        pre_idx_buf = (ctypes.c_int32 * max(16, n_surv))()
        pre_par_buf = (ctypes.c_int32 * max(16, n_surv))()
        pre_mp_buf = (ctypes.c_uint16 * max(16, n_surv))()
        pre_pd_buf = (ctypes.c_uint8 * max(16, n_surv))()

        n_pre = self.lib.ft_get_instantiated_preorder(
            self._tree_ptr, pre_idx_buf, pre_par_buf, pre_mp_buf, pre_pd_buf, max(16, n_surv)
        )
        invalidated_count = 0
        for k in range(n_pre):
            ni = int(pre_idx_buf[k])
            if ni == 0:
                continue
            pi = int(pre_par_buf[k])
            mp = int(pre_mp_buf[k])
            old_pd = bool(pre_pd_buf[k])

            if not stack_matches or ni not in new_boards:
                pb = new_boards[pi]
                b = pb.copy()
                b.push(unpack_move(mp))
                new_boards[ni] = b
                new_keys[ni] = compute_zobrist_pair(b)
                if ni in new_tensors:
                    new_tensors[ni][17, :, :] = min(1.0, b.halfmove_clock / 100.0)
                is_50m = b.halfmove_clock >= 100
                is_insuff = (not is_50m) and b.is_insufficient_material()
                is_rep = (not is_50m) and (not is_insuff) and b.is_repetition(2)
                new_pd = is_50m or is_rep
            else:
                b = new_boards[ni]
                new_pd = False

            if old_pd or new_pd:
                invalidated_count += 1
                iterm, tval, isolv, sval, pdep = evaluate_terminal_flags_fast(b)
                self.lib.ft_invalidate_reused_node(self._tree_ptr, ni, iterm, tval, isolv, sval, pdep)

        self.lib.ft_finalize_reused_subtree(self._tree_ptr, pre_idx_buf, n_pre, max_depth)

        post_cap = max(16, self.lib.ft_tree_num_nodes(self._tree_ptr))
        post_idx_buf = (ctypes.c_int32 * post_cap)()
        post_par_buf = (ctypes.c_int32 * post_cap)()
        post_mp_buf = (ctypes.c_uint16 * post_cap)()
        post_pd_buf = (ctypes.c_uint8 * post_cap)()
        n_post = self.lib.ft_get_instantiated_preorder(
            self._tree_ptr, post_idx_buf, post_par_buf, post_mp_buf, post_pd_buf, post_cap
        )
        for k in range(n_post):
            ni = int(post_idx_buf[k])
            if ni == 0 or ni in new_boards:
                continue
            pi = int(post_par_buf[k])
            mp = int(post_mp_buf[k])
            b = new_boards[pi].copy()
            b.push(unpack_move(mp))
            new_boards[ni] = b
            new_keys[ni] = compute_zobrist_pair(b)

        self._boards = new_boards
        self._keys = new_keys
        self._tensors = new_tensors

        self.lib.ft_get_stats(
            self._tree_ptr,
            ctypes.byref(self._st_i64[0]),
            ctypes.byref(self._st_i64[1]),
            ctypes.byref(self._st_i64[2]),
            ctypes.byref(self._st_i64[3]),
            ctypes.byref(self._st_i64[4]),
            ctypes.byref(self._st_i64[5]),
            ctypes.byref(self._st_i32[0]),
            ctypes.byref(self._st_i32[1]),
        )
        prof["tree_reused"] = True
        prof["reused_root_visits"] = int(self._st_i32[1].value)
        prof["reused_subtree_nodes"] = n_pre
        prof["invalidated_path_dep_nodes"] = invalidated_count
        return True

    def select_leaf_batch(self, max_depth: int, use_pending: bool = False) -> tuple[ctypes.c_void_p, list[int], int]:
        status = self.lib.ft_select_begin(
            self._tree_ptr,
            self.batch_size,
            max_depth,
            1 if self.use_nn_cache else 0,
            1 if use_pending else 0,
            ctypes.byref(self._c_node),
            ctypes.byref(self._c_parent),
            ctypes.byref(self._c_move),
        )

        while status == 1:
            ni = int(self._c_node.value)
            pi = int(self._c_parent.value)
            mp = int(self._c_move.value)

            b = self._boards[pi].copy()
            b.push(unpack_move(mp))
            self._boards[ni] = b

            iterm, tval, isolv, sval, pdep = evaluate_terminal_flags_fast(b)
            if not iterm:
                khi, klo, tkey = compute_zobrist_pair(b)
                self._keys[ni] = (khi, klo, tkey)
            else:
                khi, klo = 0, 0

            status = self.lib.ft_select_provide_instantiation(
                self._tree_ptr,
                ni,
                iterm,
                tval,
                isolv,
                sval,
                pdep,
                ctypes.c_uint64(khi),
                ctypes.c_uint64(klo),
                ctypes.byref(self._c_node),
                ctypes.byref(self._c_parent),
                ctypes.byref(self._c_move),
            )

        handle = self.lib.ft_pop_inflight_batch(
            self._tree_ptr,
            self._leaves_buf,
            ctypes.byref(self._c_n_eval),
            ctypes.byref(self._c_max_ply),
        )
        n_eval = int(self._c_n_eval.value)
        leaves = [int(self._leaves_buf[i]) for i in range(n_eval)]
        return handle, leaves, int(self._c_max_ply.value)

    def pack_batch_tensors(self, leaves: list[int]) -> np.ndarray:
        n_eval = len(leaves)
        bt = np.zeros((self.batch_size, NUM_PLANES, 8, 8), dtype=np.float32)
        for i in range(n_eval):
            li = leaves[i]
            t = self._tensors.get(li)
            if t is None:
                t = self._encode_board_fast(self._boards[li])
                self._tensors[li] = t
            bt[i] = t
        if 0 < n_eval < self.batch_size:
            bt[n_eval:] = bt[0]
        return bt

    def backup_batch(
        self,
        handle: ctypes.c_void_p,
        leaves: list[int],
        p_logits_batch: np.ndarray,
        val_batch: np.ndarray,
        max_depth: int,
        prof: dict,
    ):
        self.lib.ft_revert_batch_vl_and_clear_pending(self._tree_ptr, handle)
        store_cache = 1 if (self.use_nn_cache or self.use_tree_reuse) else 0

        for i, li in enumerate(leaves):
            b = self._boards[li]
            khi, klo, tkey = self._keys[li]
            if tkey in self._seen_npu_keys:
                prof["wasted_dup_key_evals"] += 1
            else:
                self._seen_npu_keys.add(tkey)
                prof["novel_npu_evals"] += 1

            v_out = val_batch[i]
            if v_out.ndim == 1 and len(v_out) == 3:
                leaf_val = float(v_out[0] - v_out[2])
            else:
                leaf_val = float(v_out[0])

            logits_row = np.ascontiguousarray(p_logits_batch[i], dtype=np.float32)
            n_leg = self._fill_legal_move_buffers(b)

            self.lib.ft_backup_single_leaf_from_batch(
                self._tree_ptr,
                handle,
                i,
                ctypes.c_uint64(khi),
                ctypes.c_uint64(klo),
                ctypes.c_double(leaf_val),
                n_leg,
                self._acts_buf,
                self._moves_buf,
                logits_row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self._bonuses_buf,
                max_depth,
                store_cache,
            )

        self.lib.ft_destroy_inflight_batch(handle)

    def _infer_root_native_zero_copy(self, root_model, board_tensor: np.ndarray | None) -> tuple[float, ctypes.POINTER(ctypes.c_float) | None, np.ndarray | None]:
        """Runs root (B=1) inference using zero-copy OpenVINO tensor staging when supported."""
        zc_root = self._bind_zero_copy_req(getattr(root_model, "infer_request", None))
        if zc_root is not None:
            req_r, r_in_ptr, r_out0_ptr, r_out1_ptr, r_val_cols, _, r_out0_arr, r_out1_arr = zc_root
            self.lib.ft_encode_root_native(self._tree_ptr, r_in_ptr)
            req_r.infer()
            v0 = r_out1_arr[0]
            root_v = float(v0[0] - v0[2]) if (r_val_cols == 3) else float(v0[0])
            if self.t_root != 1.0 and self.t_root > 0:
                r_out0_arr[:] /= self.t_root
            return root_v, r_out0_ptr, None

        if board_tensor is None:
            tmp_arr = np.empty((NUM_PLANES, 8, 8), dtype=np.float32)
            self.lib.ft_encode_root_native(
                self._tree_ptr,
                tmp_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            board_tensor = tmp_arr
        p_logits_root, v_root_out = root_model.infer(board_tensor)
        root_v = (
            float(v_root_out[0][0] - v_root_out[0][2])
            if (v_root_out.ndim == 2 and v_root_out.shape[1] == 3)
            else float(v_root_out[0][0])
        )
        logits_r = np.ascontiguousarray(p_logits_root[0], dtype=np.float32)
        if self.t_root != 1.0 and self.t_root > 0:
            logits_r = logits_r / self.t_root
        return root_v, logits_r.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), logits_r

    def _select_best_move_from_root_snapshot(
        self,
        root_ch: list[dict],
        apply_opening_guard: bool = False,
    ) -> tuple[int, str, float, list[str], str | None]:
        self.last_root_visits = {ch["act"]: ch["visits"] for ch in root_ch}
        best_child = None

        if self.use_exact_solver:
            best_mate_plies = None
            best_mate_prior = -1.0
            for ch in root_ch:
                if ch["is_solved"] and ch["solved_val"] <= -1.0:
                    if (
                        best_mate_plies is None
                        or ch["mate_plies"] < best_mate_plies
                        or (ch["mate_plies"] == best_mate_plies and ch["prior"] > best_mate_prior)
                    ):
                        best_mate_plies = ch["mate_plies"]
                        best_mate_prior = ch["prior"]
                        best_child = ch

        if best_child is None and self.use_gumbel_root and len(root_ch) > 0:
            max_v = max(c["visits"] for c in root_ch)
            if max_v > 0:
                c_visit = 50.0
                c_scale = 1.0
                best_g_score = -float("inf")
                visit_floor = max(1, int(0.25 * max_v))
                for ch in root_ch:
                    if ch["visits"] < visit_floor:
                        continue
                    if ch["is_solved"] and ch["solved_val"] >= 1.0:
                        continue
                    child_q = -ch["q"]
                    g_score = math.log(max(1e-8, ch["prior"])) + (c_visit + max_v) * c_scale * child_q
                    if g_score > best_g_score:
                        best_g_score = g_score
                        best_child = ch

        if best_child is None:
            if apply_opening_guard and len(root_ch) > 0:
                candidates = [
                    (
                        int(ch["act"]),
                        float(ch["prior"]),
                        int(ch["visits"]),
                        float(-ch["q"]),
                        bool(ch["is_solved"] and ch["solved_val"] <= -1.0),
                        bool(ch["is_solved"] and ch["solved_val"] >= 1.0),
                    )
                    for ch in root_ch
                ]
                chosen_act = select_opening_guarded_root_act(candidates)
                for ch in root_ch:
                    if ch["act"] == chosen_act:
                        best_child = ch
                        break
            else:
                best_visits = -1
                best_q = -float("inf")
                for ch in root_ch:
                    child_q = -ch["q"]
                    if ch["visits"] > best_visits or (ch["visits"] == best_visits and child_q > best_q):
                        best_visits = ch["visits"]
                        best_q = child_q
                        best_child = ch

        if best_child is None:
            best_child = root_ch[0]

        best_act = best_child["act"]
        best_eval = -best_child["q"]
        best_uci = unpack_move(best_child["mp"]).uci()

        pv = [best_uci]
        curr_idx = best_child["idx"]
        while len(pv) < 8:
            sub_ch = self._get_children_snapshot(curr_idx)
            if not sub_ch:
                break
            nxt = max(sub_ch, key=lambda c: c["visits"])
            if nxt["visits"] <= 0 and len(pv) > 1:
                break
            pv.append(unpack_move(nxt["mp"]).uci())
            curr_idx = nxt["idx"]

        ponder_uci = pv[1] if len(pv) >= 2 else None
        return best_act, best_uci, best_eval, pv, ponder_uci

    def run_ponder_session(
        self,
        root_model,
        batch_model,
        board: chess.Board,
        stop_event: threading.Event,
        ponderhit_event: threading.Event,
        default_movetime_ms: int | None = None,
        max_depth: int = 14,
    ) -> dict:
        """
        Executes P-core-pinned C++23 + SoA AVX2 + zero-copy double-buffered pondering on `board`
        until `stop_event` (or `ponderhit_event` deadline) fires, draining any in-flight NPU batch
        cleanly so zero virtual loss leaks and the C++23 arena + Zobrist cache remain ready for
        instant `ft_try_reuse_root_native` re-rooting on the opponent's move!
        """
        pin_current_thread_to_pcores()

        root_legals = list(board.legal_moves)
        if len(root_legals) == 0:
            return {"ponder_nodes": 0, "best_uci": None, "ponder_uci": None, "root_visits": 0, "best_eval": 0.0}

        self.lib.ft_tree_reset_stats(self._tree_ptr)
        prof = {"novel_npu_evals": 0, "wasted_dup_key_evals": 0}
        ponder_nodes = 0

        cr_val, ep_val, turn_w, hm_val, rev_len = self._extract_cboard_args(board)
        reused = False
        if self.use_tree_reuse and self._has_saved_tree:
            n_delta = 0
            stack_ext = 0
            if self._saved_move_stack is not None:
                n_old = len(self._saved_move_stack)
                n_new = len(board.move_stack)
                if n_new > n_old and (n_new - n_old) <= 4 and board.move_stack[:n_old] == self._saved_move_stack:
                    n_delta = n_new - n_old
                    stack_ext = 1
                    for di, mv in enumerate(board.move_stack[n_old:]):
                        self._delta_mp_buf[di] = pack_move(mv)
            ok_reuse = self.lib.ft_try_reuse_root_native(
                self._tree_ptr,
                self._bbs8_buf,
                ctypes.c_uint64(cr_val),
                ep_val,
                turn_w,
                hm_val,
                rev_len,
                self._rev_hi_buf,
                self._rev_lo_buf,
                n_delta,
                self._delta_mp_buf,
                stack_ext,
                max_depth,
                ctypes.byref(self._c_i32_a),
                ctypes.byref(self._c_i32_b),
                ctypes.byref(self._c_i32_c),
                ctypes.byref(self._c_u64_a),
                ctypes.byref(self._c_u64_b),
            )
            if ok_reuse:
                reused = True
                root_hi = int(self._c_u64_a.value)
                root_lo = int(self._c_u64_b.value)

        if not reused:
            self.lib.ft_set_root_native(
                self._tree_ptr,
                self._bbs8_buf,
                ctypes.c_uint64(cr_val),
                ep_val,
                turn_w,
                hm_val,
                rev_len,
                self._rev_hi_buf,
                self._rev_lo_buf,
                ctypes.byref(self._c_u64_a),
                ctypes.byref(self._c_u64_b),
            )
            root_hi = int(self._c_u64_a.value)
            root_lo = int(self._c_u64_b.value)

        root_children = self._get_children_snapshot(0)
        if len(root_children) < len(root_legals):
            hit_ok = 0
            if self.use_nn_cache:
                hit_ok = self.lib.ft_expand_root_from_cache(
                    self._tree_ptr,
                    ctypes.c_uint64(root_hi),
                    ctypes.c_uint64(root_lo),
                    len(root_legals),
                    ctypes.byref(self._c_root_val),
                )
            if hit_ok:
                root_v = float(self._c_root_val.value)
                ponder_nodes += 1
            else:
                root_v, r_logits_ptr, _keep_r = self._infer_root_native_zero_copy(root_model, None)
                ponder_nodes += 1
                self.lib.ft_record_seen_root_native(self._tree_ptr)
                self.lib.ft_expand_root_native(
                    self._tree_ptr,
                    ctypes.c_double(root_v),
                    r_logits_ptr,
                    1 if (self.use_nn_cache or self.use_tree_reuse) else 0,
                )
            if self.use_fpu_reduction:
                self.lib.ft_tree_set_root_fpu_visit(self._tree_ptr, ctypes.c_double(root_v))

        self._has_saved_tree = True
        self._saved_move_stack = list(board.move_stack)

        has_async = hasattr(batch_model, "infer_request_async") and hasattr(batch_model.infer_request, "start_async")
        _pc = time.perf_counter
        search_deadline = float("inf")

        def _should_stop() -> bool:
            nonlocal search_deadline
            if stop_event.is_set():
                return True
            if ponderhit_event.is_set():
                if search_deadline == float("inf"):
                    mt = default_movetime_ms if (default_movetime_ms and default_movetime_ms > 0) else 600
                    search_deadline = _pc() + (mt / 1000.0)
                if _pc() >= search_deadline:
                    return True
            return False

        if has_async:
            requests = [batch_model.infer_request, batch_model.infer_request_async]
            zc_reqs = [self._bind_zero_copy_req(requests[0]), self._bind_zero_copy_req(requests[1])]
            in_flight_p: dict[int, tuple[ctypes.c_void_p, int]] = {}

            if not _should_stop():
                zc0 = zc_reqs[0]
                h0, bt0, nev0, _ = self.select_and_pack_native(
                    max_depth=max_depth,
                    use_pending=False,
                    target_in_ptr=zc0[1] if zc0 is not None else None,
                )
                if nev0 > 0:
                    if zc0 is not None:
                        requests[0].start_async()
                    else:
                        requests[0].start_async({"board_state": bt0})
                    in_flight_p[0] = (h0, nev0)
                else:
                    self.lib.ft_destroy_inflight_batch(h0)

            buf_idx = 0
            while len(in_flight_p) > 0:
                stopping = _should_stop()
                next_buf = 1 - buf_idx

                if not stopping and next_buf not in in_flight_p:
                    zcn = zc_reqs[next_buf]
                    hn, btn, nevn, _ = self.select_and_pack_native(
                        max_depth=max_depth,
                        use_pending=False,
                        target_in_ptr=zcn[1] if zcn is not None else None,
                    )
                    if nevn > 0:
                        if zcn is not None:
                            requests[next_buf].start_async()
                        else:
                            requests[next_buf].start_async({"board_state": btn})
                        in_flight_p[next_buf] = (hn, nevn)
                    else:
                        self.lib.ft_destroy_inflight_batch(hn)

                if buf_idx in in_flight_p:
                    curr_req = requests[buf_idx]
                    curr_req.wait()
                    curr_h, _ = in_flight_p.pop(buf_idx)
                    ponder_nodes += self.batch_size
                    zcc = zc_reqs[buf_idx]
                    if zcc is not None:
                        self.backup_batch_native_ptrs(curr_h, zcc[2], zcc[3], zcc[4], max_depth, prof)
                    else:
                        outputs = list(curr_req.results.values())
                        self.backup_batch_native(curr_h, outputs[0], outputs[1], max_depth, prof)

                if len(in_flight_p) == 0 and not _should_stop():
                    zcr = zc_reqs[next_buf]
                    hr, btr, nevr, _ = self.select_and_pack_native(
                        max_depth=max_depth,
                        use_pending=False,
                        target_in_ptr=zcr[1] if zcr is not None else None,
                    )
                    if nevr > 0:
                        if zcr is not None:
                            requests[next_buf].start_async()
                        else:
                            requests[next_buf].start_async({"board_state": btr})
                        in_flight_p[next_buf] = (hr, nevr)
                    else:
                        self.lib.ft_destroy_inflight_batch(hr)

                buf_idx = next_buf
        else:
            zc_sync = self._bind_zero_copy_req(getattr(batch_model, "infer_request", None))
            while not _should_stop():
                hs, bt, nevs, _ = self.select_and_pack_native(
                    max_depth=max_depth,
                    use_pending=False,
                    target_in_ptr=zc_sync[1] if zc_sync is not None else None,
                )
                if nevs == 0:
                    self.lib.ft_destroy_inflight_batch(hs)
                    break
                ponder_nodes += self.batch_size
                if zc_sync is not None:
                    zc_sync[0].infer()
                    self.backup_batch_native_ptrs(hs, zc_sync[2], zc_sync[3], zc_sync[4], max_depth, prof)
                else:
                    p_logits_batch, val_batch = batch_model.infer(bt)
                    self.backup_batch_native(hs, p_logits_batch, val_batch, max_depth, prof)

        self.lib.ft_clear_all_vl(self._tree_ptr)
        self.lib.ft_get_stats(
            self._tree_ptr,
            ctypes.byref(self._st_i64[0]),
            ctypes.byref(self._st_i64[1]),
            ctypes.byref(self._st_i64[2]),
            ctypes.byref(self._st_i64[3]),
            ctypes.byref(self._st_i64[4]),
            ctypes.byref(self._st_i64[5]),
            ctypes.byref(self._st_i32[0]),
            ctypes.byref(self._st_i32[1]),
        )
        ponder_nodes += int(self._st_i64[4].value)  # include cache-expanded nodes
        final_root_visits = int(self._st_i32[1].value)

        self._has_saved_tree = True
        self._saved_move_stack = list(board.move_stack)

        root_ch = self._get_children_snapshot(0)
        best_uci = None
        ponder_uci = None
        best_eval = 0.0
        if root_ch:
            apply_guard = bool(self.use_opening_root_guard and board.fullmove_number <= 12)
            _, best_uci, best_eval, _, ponder_uci = self._select_best_move_from_root_snapshot(
                root_ch,
                apply_opening_guard=apply_guard,
            )

        return {
            "ponder_nodes": ponder_nodes,
            "best_uci": best_uci,
            "ponder_uci": ponder_uci,
            "root_visits": final_root_visits,
            "best_eval": best_eval,
        }

    def search(
        self,
        root_model,
        batch_model,
        board: chess.Board,
        board_tensor: np.ndarray | None = None,
        movetime_ms: int = 600,
        max_depth: int = 14,
        max_nodes: int | None = None,
    ) -> tuple[int, str, float, int, list[str], int, dict]:
        pin_current_thread_to_pcores()

        t_start = time.perf_counter()
        t_deadline = t_start + (movetime_ms / 1000.0)
        is_black = board.turn == chess.BLACK
        nodes_evaluated = 0
        max_depth_reached = 0

        root_legals = list(board.legal_moves)
        if len(root_legals) == 0:
            self.last_root_visits = {}
            val = -1.0 if board.is_check() else 0.0
            return -1, "0000", val, 0, [], 0, {}

        if len(root_legals) == 1:
            m = root_legals[0]
            act = uci_to_action(m.uci(), canonical_flip=is_black)
            self.last_root_visits = {act: 1}
            return act, m.uci(), 0.0, 1, [m.uci()], 1, {}

        for m in root_legals:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                act = uci_to_action(m.uci(), canonical_flip=is_black)
                self.last_root_visits = {act: 1}
                return act, m.uci(), 1.0, len(root_legals), [m.uci()], 1, {}
            board.pop()

        use_native = (not board.chess960) and (self.encoder_layout == ENCODER_LAYOUT_GPU_UNPACK)
        if not use_native and board_tensor is None:
            board_tensor = self._encode_board_fast(board)

        has_async = hasattr(batch_model, "infer_request_async") and hasattr(batch_model.infer_request, "start_async")
        _pc = time.perf_counter
        prof = {
            "select_s": 0.0,
            "pack_s": 0.0,
            "dispatch_s": 0.0,
            "wait_s": 0.0,
            "backup_s": 0.0,
            "batches": 0,
            "slots": 0,
            "unique_leaves": 0,
            "root_npu_evals": 0,
            "novel_npu_evals": 0,
            "wasted_dup_key_evals": 0,
            "collisions": 0,
            "dedup_avoided": 0,
            "nn_cache_hits": 0,
            "nn_cache_misses": 0,
            "cache_expanded_nodes": 0,
            "terminal_backups": 0,
            "tree_reused": False,
            "reused_root_visits": 0,
            "reused_subtree_nodes": 0,
            "invalidated_path_dep_nodes": 0,
            "max_depth_cache": 0,
            "mode": "async_overlap"
            if (has_async and self.use_async_overlap)
            else ("double_buffer" if has_async else "sync"),
            "inflight_during_cpu_work": self.async_inflight
            if (has_async and self.use_async_overlap)
            else (1 if has_async else 0),
            "fast_tree": True,
            "zero_copy_staging": False,
        }

        self.lib.ft_tree_reset_stats(self._tree_ptr)

        reused = False
        if use_native:
            cr_val, ep_val, turn_w, hm_val, rev_len = self._extract_cboard_args(board)
            if self.use_tree_reuse and self._has_saved_tree:
                n_delta = 0
                stack_ext = 0
                if self._saved_move_stack is not None:
                    n_old = len(self._saved_move_stack)
                    n_new = len(board.move_stack)
                    if n_new > n_old and (n_new - n_old) <= 4 and board.move_stack[:n_old] == self._saved_move_stack:
                        n_delta = n_new - n_old
                        stack_ext = 1
                        for di, mv in enumerate(board.move_stack[n_old:]):
                            self._delta_mp_buf[di] = pack_move(mv)
                ok_reuse = self.lib.ft_try_reuse_root_native(
                    self._tree_ptr,
                    self._bbs8_buf,
                    ctypes.c_uint64(cr_val),
                    ep_val,
                    turn_w,
                    hm_val,
                    rev_len,
                    self._rev_hi_buf,
                    self._rev_lo_buf,
                    n_delta,
                    self._delta_mp_buf,
                    stack_ext,
                    max_depth,
                    ctypes.byref(self._c_i32_a),
                    ctypes.byref(self._c_i32_b),
                    ctypes.byref(self._c_i32_c),
                    ctypes.byref(self._c_u64_a),
                    ctypes.byref(self._c_u64_b),
                )
                if ok_reuse:
                    reused = True
                    root_hi = int(self._c_u64_a.value)
                    root_lo = int(self._c_u64_b.value)
                    prof["tree_reused"] = True
                    prof["reused_root_visits"] = int(self._c_i32_a.value)
                    prof["reused_subtree_nodes"] = int(self._c_i32_b.value)
                    prof["invalidated_path_dep_nodes"] = int(self._c_i32_c.value)

            if not reused:
                self.lib.ft_set_root_native(
                    self._tree_ptr,
                    self._bbs8_buf,
                    ctypes.c_uint64(cr_val),
                    ep_val,
                    turn_w,
                    hm_val,
                    rev_len,
                    self._rev_hi_buf,
                    self._rev_lo_buf,
                    ctypes.byref(self._c_u64_a),
                    ctypes.byref(self._c_u64_b),
                )
                root_hi = int(self._c_u64_a.value)
                root_lo = int(self._c_u64_b.value)
        else:
            if self.use_tree_reuse:
                reused = self._try_reuse_root(board, board_tensor, max_depth, prof)
            root_hi, root_lo, root_tkey = compute_zobrist_pair(board)
            if not reused:
                self._boards.clear()
                self._keys.clear()
                self._tensors.clear()
                self.lib.ft_tree_reset_root(self._tree_ptr, ctypes.c_uint64(root_hi), ctypes.c_uint64(root_lo))
                self._boards[0] = board.copy()
                self._keys[0] = (root_hi, root_lo, root_tkey)
                self._tensors[0] = board_tensor

        root_children = self._get_children_snapshot(0)
        if len(root_children) < len(root_legals):
            hit_ok = 0
            if self.use_nn_cache:
                hit_ok = self.lib.ft_expand_root_from_cache(
                    self._tree_ptr,
                    ctypes.c_uint64(root_hi),
                    ctypes.c_uint64(root_lo),
                    len(root_legals),
                    ctypes.byref(self._c_root_val),
                )
            if hit_ok:
                root_v = float(self._c_root_val.value)
                nodes_evaluated += 1
            else:
                if use_native:
                    root_v, r_logits_ptr, _keep_r = self._infer_root_native_zero_copy(root_model, board_tensor)
                    nodes_evaluated += 1
                    prof["root_npu_evals"] += 1
                    if self.lib.ft_record_seen_root_native(self._tree_ptr):
                        prof["novel_npu_evals"] += 1
                    else:
                        prof["wasted_dup_key_evals"] += 1
                    self.lib.ft_expand_root_native(
                        self._tree_ptr,
                        ctypes.c_double(root_v),
                        r_logits_ptr,
                        1 if (self.use_nn_cache or self.use_tree_reuse) else 0,
                    )
                else:
                    p_logits_root, v_root_out = root_model.infer(board_tensor)
                    nodes_evaluated += 1
                    prof["root_npu_evals"] += 1
                    if root_tkey in self._seen_npu_keys:
                        prof["wasted_dup_key_evals"] += 1
                    else:
                        self._seen_npu_keys.add(root_tkey)
                        prof["novel_npu_evals"] += 1
                    root_v = (
                        float(v_root_out[0][0] - v_root_out[0][2])
                        if (v_root_out.ndim == 2 and v_root_out.shape[1] == 3)
                        else float(v_root_out[0][0])
                    )
                    logits_r = np.ascontiguousarray(p_logits_root[0], dtype=np.float32)
                    n_leg = self._fill_legal_move_buffers(self._boards[0], legals=root_legals)
                    self.lib.ft_expand_node_with_logits(
                        self._tree_ptr,
                        0,
                        ctypes.c_uint64(root_hi),
                        ctypes.c_uint64(root_lo),
                        ctypes.c_double(root_v),
                        n_leg,
                        self._acts_buf,
                        self._moves_buf,
                        logits_r.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        self._bonuses_buf,
                        n_leg,
                        1 if (self.use_nn_cache or self.use_tree_reuse) else 0,
                    )

            if self.use_fpu_reduction:
                self.lib.ft_tree_set_root_fpu_visit(self._tree_ptr, ctypes.c_double(root_v))

        t_loop0 = _pc()

        if use_native:
            if has_async and self.use_async_overlap:
                from collections import deque

                depth = self.async_inflight + 1
                reqs = [batch_model.infer_request, batch_model.infer_request_async]
                while len(reqs) + len(self._extra_requests) < depth:
                    self._extra_requests.append(batch_model.compiled_model.create_infer_request())
                reqs = (reqs + self._extra_requests)[:depth]
                zc_reqs = [self._bind_zero_copy_req(rq) for rq in reqs]
                prof["zero_copy_staging"] = all(z is not None for z in zc_reqs)
                free = list(range(depth))
                inflight: deque = deque()

                while True:
                    while free:
                        time_up = (_pc() >= t_deadline) or (
                            max_nodes is not None and nodes_evaluated + len(inflight) * self.batch_size >= max_nodes
                        )
                        if time_up:
                            break
                        r_cand = free[-1]
                        zcr = zc_reqs[r_cand]
                        t0 = _pc()
                        handle_n, bt, n_ev, mply_n = self.select_and_pack_native(
                            max_depth=max_depth,
                            use_pending=True,
                            target_in_ptr=zcr[1] if zcr is not None else None,
                        )
                        t1 = _pc()
                        prof["select_s"] += t1 - t0
                        if n_ev == 0:
                            self.lib.ft_destroy_inflight_batch(handle_n)
                            break
                        r = free.pop()
                        if zcr is not None:
                            reqs[r].start_async()
                        else:
                            reqs[r].start_async({"board_state": bt})
                        prof["dispatch_s"] += _pc() - t1
                        inflight.append((r, handle_n, n_ev, mply_n))
                    if not inflight:
                        break
                    r, curr_h, curr_nev, curr_mply = inflight.popleft()
                    t0 = _pc()
                    reqs[r].wait()
                    t1 = _pc()
                    prof["wait_s"] += t1 - t0
                    nodes_evaluated += self.batch_size
                    prof["batches"] += 1
                    prof["slots"] += self.batch_size
                    prof["unique_leaves"] += curr_nev
                    if curr_mply > max_depth_reached:
                        max_depth_reached = curr_mply
                    zcc = zc_reqs[r]
                    if zcc is not None:
                        self.backup_batch_native_ptrs(curr_h, zcc[2], zcc[3], zcc[4], max_depth, prof)
                    else:
                        outputs = list(reqs[r].results.values())
                        self.backup_batch_native(curr_h, outputs[0], outputs[1], max_depth, prof)
                    prof["backup_s"] += _pc() - t1
                    free.append(r)

            elif has_async:
                requests = [batch_model.infer_request, batch_model.infer_request_async]
                zc_reqs = [self._bind_zero_copy_req(requests[0]), self._bind_zero_copy_req(requests[1])]
                prof["zero_copy_staging"] = all(z is not None for z in zc_reqs)
                in_flight_n: dict[int, tuple[ctypes.c_void_p, int, int]] = {}

                while True:
                    zc0 = zc_reqs[0]
                    t0 = _pc()
                    h0, bt0, nev_0, mply_0 = self.select_and_pack_native(
                        max_depth=max_depth,
                        use_pending=False,
                        target_in_ptr=zc0[1] if zc0 is not None else None,
                    )
                    t1 = _pc()
                    prof["select_s"] += t1 - t0
                    if nev_0 > 0:
                        if zc0 is not None:
                            requests[0].start_async()
                        else:
                            requests[0].start_async({"board_state": bt0})
                        prof["dispatch_s"] += _pc() - t1
                        in_flight_n[0] = (h0, nev_0, mply_0)
                        break
                    self.lib.ft_destroy_inflight_batch(h0)
                    self.lib.ft_get_stats(
                        self._tree_ptr,
                        ctypes.byref(self._st_i64[0]),
                        ctypes.byref(self._st_i64[1]),
                        ctypes.byref(self._st_i64[2]),
                        ctypes.byref(self._st_i64[3]),
                        ctypes.byref(self._st_i64[4]),
                        ctypes.byref(self._st_i64[5]),
                        ctypes.byref(self._st_i32[0]),
                        ctypes.byref(self._st_i32[1]),
                    )
                    cur_cache_exp = int(self._st_i64[4].value)
                    if (
                        (_pc() >= t_deadline)
                        or (max_nodes is not None and (nodes_evaluated + cur_cache_exp) >= max_nodes)
                        or cur_cache_exp == 0
                    ):
                        break

                buf_idx = 0
                while len(in_flight_n) > 0:
                    time_up = (_pc() >= t_deadline) or (max_nodes is not None and nodes_evaluated >= max_nodes)
                    next_buf = 1 - buf_idx

                    if not time_up and next_buf not in in_flight_n:
                        zcn = zc_reqs[next_buf]
                        t0 = _pc()
                        hn, btn, nev_next, mply_next = self.select_and_pack_native(
                            max_depth=max_depth,
                            use_pending=False,
                            target_in_ptr=zcn[1] if zcn is not None else None,
                        )
                        t1 = _pc()
                        prof["select_s"] += t1 - t0
                        if nev_next > 0:
                            if zcn is not None:
                                requests[next_buf].start_async()
                            else:
                                requests[next_buf].start_async({"board_state": btn})
                            prof["dispatch_s"] += _pc() - t1
                            in_flight_n[next_buf] = (hn, nev_next, mply_next)
                        else:
                            self.lib.ft_destroy_inflight_batch(hn)

                    curr_req = requests[buf_idx]
                    t0 = _pc()
                    curr_req.wait()
                    t1 = _pc()
                    prof["wait_s"] += t1 - t0
                    nodes_evaluated += self.batch_size

                    curr_h, curr_nev, curr_mply = in_flight_n.pop(buf_idx)
                    prof["batches"] += 1
                    prof["slots"] += self.batch_size
                    prof["unique_leaves"] += curr_nev
                    if curr_mply > max_depth_reached:
                        max_depth_reached = curr_mply

                    zcc = zc_reqs[buf_idx]
                    if zcc is not None:
                        self.backup_batch_native_ptrs(curr_h, zcc[2], zcc[3], zcc[4], max_depth, prof)
                    else:
                        outputs = list(curr_req.results.values())
                        self.backup_batch_native(curr_h, outputs[0], outputs[1], max_depth, prof)
                    prof["backup_s"] += _pc() - t1

                    if len(in_flight_n) == 0 and not (
                        (_pc() >= t_deadline) or (max_nodes is not None and nodes_evaluated >= max_nodes)
                    ):
                        zcr = zc_reqs[next_buf]
                        t0 = _pc()
                        hr, btr, nev_re, mply_re = self.select_and_pack_native(
                            max_depth=max_depth,
                            use_pending=False,
                            target_in_ptr=zcr[1] if zcr is not None else None,
                        )
                        t1_p = _pc()
                        prof["select_s"] += t1_p - t0
                        if nev_re > 0:
                            if zcr is not None:
                                requests[next_buf].start_async()
                            else:
                                requests[next_buf].start_async({"board_state": btr})
                            prof["dispatch_s"] += _pc() - t1_p
                            in_flight_n[next_buf] = (hr, nev_re, mply_re)
                        else:
                            self.lib.ft_destroy_inflight_batch(hr)

                    buf_idx = next_buf
            else:
                zc_sync = self._bind_zero_copy_req(getattr(batch_model, "infer_request", None))
                prof["zero_copy_staging"] = zc_sync is not None
                while _pc() < t_deadline:
                    if max_nodes is not None and nodes_evaluated >= max_nodes:
                        break
                    t0 = _pc()
                    hs, bt, nev_s, mply_s = self.select_and_pack_native(
                        max_depth=max_depth,
                        use_pending=False,
                        target_in_ptr=zc_sync[1] if zc_sync is not None else None,
                    )
                    t1 = _pc()
                    prof["select_s"] += t1 - t0
                    if nev_s == 0:
                        self.lib.ft_destroy_inflight_batch(hs)
                        break
                    if mply_s > max_depth_reached:
                        max_depth_reached = mply_s
                    if zc_sync is not None:
                        zc_sync[0].infer()
                        t3 = _pc()
                        prof["wait_s"] += t3 - t1
                        nodes_evaluated += self.batch_size
                        prof["batches"] += 1
                        prof["slots"] += self.batch_size
                        prof["unique_leaves"] += nev_s
                        self.backup_batch_native_ptrs(hs, zc_sync[2], zc_sync[3], zc_sync[4], max_depth, prof)
                    else:
                        p_logits_batch, val_batch = batch_model.infer(bt)
                        t3 = _pc()
                        prof["wait_s"] += t3 - t1
                        nodes_evaluated += self.batch_size
                        prof["batches"] += 1
                        prof["slots"] += self.batch_size
                        prof["unique_leaves"] += nev_s
                        self.backup_batch_native(hs, p_logits_batch, val_batch, max_depth, prof)
                    prof["backup_s"] += _pc() - t3

        elif has_async and self.use_async_overlap:
            from collections import deque

            depth = self.async_inflight + 1
            reqs = [batch_model.infer_request, batch_model.infer_request_async]
            while len(reqs) + len(self._extra_requests) < depth:
                self._extra_requests.append(batch_model.compiled_model.create_infer_request())
            reqs = (reqs + self._extra_requests)[:depth]
            free = list(range(depth))
            inflight: deque = deque()

            while True:
                while free:
                    time_up = (_pc() >= t_deadline) or (
                        max_nodes is not None and nodes_evaluated + len(inflight) * self.batch_size >= max_nodes
                    )
                    if time_up:
                        break
                    t0 = _pc()
                    handle_n, leaves_n, mply_n = self.select_leaf_batch(max_depth=max_depth, use_pending=True)
                    t1 = _pc()
                    prof["select_s"] += t1 - t0
                    if len(leaves_n) == 0:
                        self.lib.ft_destroy_inflight_batch(handle_n)
                        break
                    bt = self.pack_batch_tensors(leaves_n)
                    t2 = _pc()
                    prof["pack_s"] += t2 - t1
                    r = free.pop()
                    reqs[r].start_async({"board_state": bt})
                    prof["dispatch_s"] += _pc() - t2
                    inflight.append((r, handle_n, leaves_n, mply_n))
                if not inflight:
                    break
                r, curr_h, curr_leaves, curr_mply = inflight.popleft()
                t0 = _pc()
                reqs[r].wait()
                t1 = _pc()
                prof["wait_s"] += t1 - t0
                outputs = list(reqs[r].results.values())
                nodes_evaluated += self.batch_size
                prof["batches"] += 1
                prof["slots"] += self.batch_size
                prof["unique_leaves"] += len(curr_leaves)
                if curr_mply > max_depth_reached:
                    max_depth_reached = curr_mply
                self.backup_batch(curr_h, curr_leaves, outputs[0], outputs[1], max_depth, prof)
                prof["backup_s"] += _pc() - t1
                free.append(r)

        elif has_async:
            requests = [batch_model.infer_request, batch_model.infer_request_async]
            in_flight: dict[int, tuple[ctypes.c_void_p, list[int], int]] = {}

            while True:
                t0 = _pc()
                h0, leaves_0, mply_0 = self.select_leaf_batch(max_depth=max_depth, use_pending=False)
                prof["select_s"] += _pc() - t0
                if len(leaves_0) > 0:
                    t1 = _pc()
                    bt0 = self.pack_batch_tensors(leaves_0)
                    t2 = _pc()
                    prof["pack_s"] += t2 - t1
                    requests[0].start_async({"board_state": bt0})
                    prof["dispatch_s"] += _pc() - t2
                    in_flight[0] = (h0, leaves_0, mply_0)
                    break
                self.lib.ft_destroy_inflight_batch(h0)
                self.lib.ft_get_stats(
                    self._tree_ptr,
                    ctypes.byref(self._st_i64[0]),
                    ctypes.byref(self._st_i64[1]),
                    ctypes.byref(self._st_i64[2]),
                    ctypes.byref(self._st_i64[3]),
                    ctypes.byref(self._st_i64[4]),
                    ctypes.byref(self._st_i64[5]),
                    ctypes.byref(self._st_i32[0]),
                    ctypes.byref(self._st_i32[1]),
                )
                cur_cache_exp = int(self._st_i64[4].value)
                if (
                    (_pc() >= t_deadline)
                    or (max_nodes is not None and (nodes_evaluated + cur_cache_exp) >= max_nodes)
                    or cur_cache_exp == 0
                ):
                    break

            buf_idx = 0
            while len(in_flight) > 0:
                time_up = (_pc() >= t_deadline) or (max_nodes is not None and nodes_evaluated >= max_nodes)
                next_buf = 1 - buf_idx

                if not time_up and next_buf not in in_flight:
                    t0 = _pc()
                    hn, leaves_next, mply_next = self.select_leaf_batch(max_depth=max_depth, use_pending=False)
                    t1 = _pc()
                    prof["select_s"] += t1 - t0
                    if len(leaves_next) > 0:
                        btn = self.pack_batch_tensors(leaves_next)
                        t2 = _pc()
                        prof["pack_s"] += t2 - t1
                        requests[next_buf].start_async({"board_state": btn})
                        prof["dispatch_s"] += _pc() - t2
                        in_flight[next_buf] = (hn, leaves_next, mply_next)
                    else:
                        self.lib.ft_destroy_inflight_batch(hn)

                curr_req = requests[buf_idx]
                t0 = _pc()
                curr_req.wait()
                t1 = _pc()
                prof["wait_s"] += t1 - t0
                outputs = list(curr_req.results.values())
                nodes_evaluated += self.batch_size

                curr_h, curr_leaves, curr_mply = in_flight.pop(buf_idx)
                prof["batches"] += 1
                prof["slots"] += self.batch_size
                prof["unique_leaves"] += len(curr_leaves)
                if curr_mply > max_depth_reached:
                    max_depth_reached = curr_mply

                self.backup_batch(curr_h, curr_leaves, outputs[0], outputs[1], max_depth, prof)
                prof["backup_s"] += _pc() - t1

                if len(in_flight) == 0 and not (
                    (_pc() >= t_deadline) or (max_nodes is not None and nodes_evaluated >= max_nodes)
                ):
                    t0 = _pc()
                    hr, leaves_re, mply_re = self.select_leaf_batch(max_depth=max_depth, use_pending=False)
                    prof["select_s"] += _pc() - t0
                    if len(leaves_re) > 0:
                        t1_p = _pc()
                        btr = self.pack_batch_tensors(leaves_re)
                        t2_p = _pc()
                        prof["pack_s"] += t2_p - t1_p
                        requests[next_buf].start_async({"board_state": btr})
                        prof["dispatch_s"] += _pc() - t2_p
                        in_flight[next_buf] = (hr, leaves_re, mply_re)
                    else:
                        self.lib.ft_destroy_inflight_batch(hr)

                buf_idx = next_buf
        else:
            while _pc() < t_deadline:
                if max_nodes is not None and nodes_evaluated >= max_nodes:
                    break
                t0 = _pc()
                hs, leaves_s, mply_s = self.select_leaf_batch(max_depth=max_depth, use_pending=False)
                t1 = _pc()
                prof["select_s"] += t1 - t0
                if len(leaves_s) == 0:
                    self.lib.ft_destroy_inflight_batch(hs)
                    break
                if mply_s > max_depth_reached:
                    max_depth_reached = mply_s
                bt = self.pack_batch_tensors(leaves_s)
                t2 = _pc()
                prof["pack_s"] += t2 - t1
                p_logits_batch, val_batch = batch_model.infer(bt)
                t3 = _pc()
                prof["wait_s"] += t3 - t2
                nodes_evaluated += self.batch_size
                prof["batches"] += 1
                prof["slots"] += self.batch_size
                prof["unique_leaves"] += len(leaves_s)
                self.backup_batch(hs, leaves_s, p_logits_batch, val_batch, max_depth, prof)
                prof["backup_s"] += _pc() - t3

        self.lib.ft_get_stats(
            self._tree_ptr,
            ctypes.byref(self._st_i64[0]),
            ctypes.byref(self._st_i64[1]),
            ctypes.byref(self._st_i64[2]),
            ctypes.byref(self._st_i64[3]),
            ctypes.byref(self._st_i64[4]),
            ctypes.byref(self._st_i64[5]),
            ctypes.byref(self._st_i32[0]),
            ctypes.byref(self._st_i32[1]),
        )
        prof["terminal_backups"] = int(self._st_i64[0].value)
        prof["collisions"] = int(self._st_i64[1].value)
        prof["nn_cache_hits"] = int(self._st_i64[2].value)
        prof["nn_cache_misses"] = int(self._st_i64[3].value)
        prof["cache_expanded_nodes"] = int(self._st_i64[4].value)
        prof["dedup_avoided"] = int(self._st_i64[5].value)
        prof["max_depth_cache"] = int(self._st_i32[0].value)
        final_root_visits = int(self._st_i32[1].value)

        if prof["max_depth_cache"] > max_depth_reached:
            max_depth_reached = prof["max_depth_cache"]

        nodes_evaluated += prof["cache_expanded_nodes"]
        prof["pre_loop_s"] = t_loop0 - t_start
        prof["loop_s"] = _pc() - t_loop0
        prof["total_s"] = _pc() - t_start
        prof["final_root_visits"] = final_root_visits
        prof["added_root_visits"] = max(0, final_root_visits - prof["reused_root_visits"])
        prof["npu_total_slots"] = prof["root_npu_evals"] + prof["slots"]
        prof["unique_npu_evals"] = prof["root_npu_evals"] + prof["unique_leaves"]
        prof["wasted_pad_slots"] = prof["slots"] - prof["unique_leaves"]
        prof["total_wasted_npu_slots"] = prof["npu_total_slots"] - prof["novel_npu_evals"]
        c_entries, c_bytes, _, _, _ = self.cache.stats()
        prof["nn_cache_used_bytes"] = c_bytes
        prof["nn_cache_entries"] = c_entries

        self._has_saved_tree = bool(self.use_tree_reuse)
        self._saved_move_stack = list(board.move_stack) if self.use_tree_reuse else None
        if not self.use_tree_reuse:
            self._boards.clear()
            self._keys.clear()
            self._tensors.clear()

        root_ch = self._get_children_snapshot(0)
        apply_guard = bool(self.use_opening_root_guard and board.fullmove_number <= 12)
        best_act, best_uci, best_eval, pv, _ = self._select_best_move_from_root_snapshot(
            root_ch,
            apply_opening_guard=apply_guard,
        )
        return best_act, best_uci, best_eval, nodes_evaluated, pv, max_depth_reached, prof
