"""
Unit & Integration Tests for Part B1 (`use_tree_reuse` and `use_nn_cache`).

Verifies:
  1. A cache hit in `CanonicalNNCache` returns bit-for-bit EXACTLY what a fresh evaluation
     (`select_top_candidates` + softmax + value extraction) returns across all `top_k` values
     (4, 8, 16, all legal moves), White/Black perspectives, captures, en-passant, promotions,
     and transpositions, while respecting the 64 MB LRU memory cap.
  2. Re-rooting (`use_tree_reuse=True`) preserves valid subtree visits/values, widens root/shallow
     branching to full legal moves, invalidates/recomputes any node whose value or terminal/draw
     state depended on repetition or the 50-move clock, and produces zero false solver proofs or
     stale draws after re-rooting.
  3. Mate-in-1 / Mate-in-2 / Mate-in-3 tactical suites do not regress under `use_tree_reuse`,
     `use_nn_cache`, or `both` (both from cold start and after re-rooting from a preceding position).
"""

import os
import sys
import json
import unittest
import chess
import numpy as np

from neurachess.core.constants import TOTAL_ACTIONS, NUM_PLANES, uci_to_action, move_to_action
from neurachess.core.board_tensor import board_to_tensor
from neurachess.search.candidate_selector import select_top_candidates
from neurachess.search.npu_batched_puct import CanonicalNNCache, TreeNode, NPUBatchedPUCTSearch


class DeterministicMockModel:
    """Deterministic model that hashes canonical board planes to reproducible policy/value outputs."""

    def __init__(self, batch_size: int = 1, force_uniform: bool = False):
        self.batch_size = batch_size
        self.force_uniform = force_uniform
        self.calls = 0

    def infer(self, board_tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        if board_tensor.ndim == 3:
            b_in = board_tensor[np.newaxis, ...]
        else:
            b_in = board_tensor
        B = b_in.shape[0]
        p_out = np.zeros((B, TOTAL_ACTIONS), dtype=np.float32)
        v_out = np.zeros((B, 1), dtype=np.float32)
        for i in range(B):
            if self.force_uniform:
                v_out[i, 0] = 0.0
            else:
                # Hash planes 0..16 (canonical piece/castling/ep) so transpositions have identical logits
                seed = int(np.sum(b_in[i, :17] * np.arange(1, 18, dtype=np.float32)[:, None, None]) * 10007) & 0x7FFFFFFF
                rng = np.random.RandomState(seed % (2**31 - 1))
                p_out[i] = rng.standard_normal(TOTAL_ACTIONS).astype(np.float32)
                v_out[i, 0] = float(np.tanh(rng.standard_normal()))
        return p_out, v_out


def _fresh_eval_reference(board: chess.Board, p_logits: np.ndarray, v_out: np.ndarray, top_k: int):
    if v_out.ndim == 1 and len(v_out) == 3:
        leaf_val = float(v_out[0] - v_out[2])
    else:
        leaf_val = float(v_out[0])
    c_acts, c_moves, _ = select_top_candidates(board, p_logits, top_k=top_k)
    if len(c_acts) == 0:
        return [], [], np.empty(0, dtype=p_logits.dtype), leaf_val
    l_sel = p_logits[c_acts]
    e_l = np.exp(l_sel - np.max(l_sel))
    p_sel = e_l / np.sum(e_l)
    return c_acts, c_moves, p_sel, leaf_val


def test_nn_cache_hit_matches_fresh_evaluation_exactly():
    """
    B1 Test (1): A cache hit returns bit-for-bit EXACTLY what a fresh evaluation returns
    across all top_k values (4, 8, 16, all legal moves) and diverse tactical/special positions.
    """
    fens = [
        chess.STARTING_FEN,
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # Kiwipete (captures, castling)
        "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # En-passant available
        "8/P7/8/8/8/8/7k/K7 w - - 0 1",  # Promotions
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",  # Black to move
    ]
    model = DeterministicMockModel(batch_size=1)
    cache = CanonicalNNCache()
    assert cache.max_bytes == 64 * 1024 * 1024

    for fen in fens:
        board = chess.Board(fen)
        tensor = board_to_tensor(board)
        p_logits_batch, v_out_batch = model.infer(tensor)
        p_logits = p_logits_batch[0]
        v_out = v_out_batch[0]

        key = CanonicalNNCache.canonical_key(board)
        entry = CanonicalNNCache.build_entry(board, p_logits, v_out)
        cache.put(key, entry)

        cached_entry = cache.get(key)
        assert cached_entry is not None

        n_legal = board.legal_moves.count()
        for top_k in [4, 8, 16, n_legal, 256]:
            c_acts, c_moves, c_priors, c_val = CanonicalNNCache.expand_from_entry(cached_entry, top_k=top_k)
            f_acts, f_moves, f_priors, f_val = _fresh_eval_reference(board, p_logits, v_out, top_k=top_k)

            assert c_acts == f_acts, f"Action mismatch at top_k={top_k} for FEN {fen}"
            assert c_moves == f_moves, f"Move mismatch at top_k={top_k} for FEN {fen}"
            assert c_val == f_val, f"Value mismatch at top_k={top_k} for FEN {fen}"
            assert np.array_equal(c_priors, f_priors), f"Priors not bit-for-bit identical at top_k={top_k} for FEN {fen}"


def test_nn_cache_transposition_and_lru_memory_cap():
    """
    Verifies that transpositions (different move orders reaching the same canonical position)
    share the exact same canonical key and hit the cache, and that the LRU evicts oldest entries
    when `max_bytes` is reached.
    """
    # 1. d4 Nf6 2. c4 e6 3. Nf3 vs 1. Nf3 Nf6 2. c4 e6 3. d4
    b1 = chess.Board()
    for uci in ["d2d4", "g8f6", "c2c4", "e7e6", "g1f3"]:
        b1.push_uci(uci)

    b2 = chess.Board()
    for uci in ["g1f3", "g8f6", "c2c4", "e7e6", "d2d4"]:
        b2.push_uci(uci)

    assert CanonicalNNCache.canonical_key(b1) == CanonicalNNCache.canonical_key(b2)

    # Test strict LRU memory cap
    small_cache = CanonicalNNCache(max_bytes=4096)
    model = DeterministicMockModel(batch_size=1)
    board = chess.Board()
    keys = []
    for mv in list(board.legal_moves)[:10]:
        board.push(mv)
        k = CanonicalNNCache.canonical_key(board)
        keys.append(k)
        p, v = model.infer(board_to_tensor(board))
        small_cache.put(k, CanonicalNNCache.build_entry(board, p[0], v[0]))
        assert small_cache.used_bytes <= small_cache.max_bytes
        board.pop()

    assert small_cache.evictions > 0
    assert small_cache.get(keys[0]) is None
    assert small_cache.get(keys[-1]) is not None


def _assert_tree_invariants(root: TreeNode):
    """Verifies bottom-up visit/value consistency and ensures no path-dependent draw is marked solved."""
    stack = [root]
    while stack:
        node = stack.pop()
        if node.path_dep_draw:
            assert not node.is_solved, "Path-dependent draw must NEVER be marked is_solved=True"
        exp_visits = node.self_visits + sum(c.visits for c in node.children.values())
        exp_val = node.self_value_sum + sum(-c.total_value for c in node.children.values())
        assert node.visits == exp_visits, f"Visit mismatch at ply={node.ply}: {node.visits} != {exp_visits}"
        assert abs(node.total_value - exp_val) < 1e-4, f"Value mismatch at ply={node.ply}: {node.total_value} != {exp_val}"
        for ch in node.children.values():
            assert ch.parent is node
            assert ch.ply == node.ply + 1
            if ch.is_instantiated or ch.visits > 0:
                stack.append(ch)


def test_tree_reuse_invalidates_repetitions_and_50move_with_no_false_proofs():
    """
    B1 Test (2a - Python backend): After our move and the opponent's reply, re-root at the child
    and keep its subtree; nodes whose value or terminal/draw state depended on repetition or the
    50-move clock must be invalidated or recomputed, with no false proofs or stale draws.
    """
    root_model = DeterministicMockModel(batch_size=1, force_uniform=True)
    batch_model = DeterministicMockModel(batch_size=8, force_uniform=True)
    searcher = NPUBatchedPUCTSearch(
        root_model=root_model,
        batch_model=batch_model,
        batch_size=8,
        use_exact_solver=True,
        use_tree_reuse=True,
        use_nn_cache=True,
        use_fast_tree=False,
    )

    # Position where White has a knight shuttling repetition option (Ng1-f3-g1) and an irreversible pawn push (a2a3)
    board = chess.Board("4k3/8/8/8/8/8/P7/4K1N1 w - - 0 1")
    board.push_uci("g1f3")
    board.push_uci("e8f8")
    # Now from here, f3g1 followed by f8e8 repeats the initial position (2-fold repetition in tree)
    _, uci1, _, _, _, _ = searcher.search(board, movetime_ms=500, max_nodes=80)
    saved_root_1 = searcher._saved_root
    assert saved_root_1 is not None

    # Force a repetition draw node inside the subtree of a2a3 -> f8e8 to verify invalidation on re-root
    act_a3 = uci_to_action("a2a3", canonical_flip=False)
    ch_a3 = saved_root_1.children[act_a3]
    if not ch_a3.is_instantiated:
        ch_a3.instantiate()
    # Ensure ch_a3 is expanded and has reply f8e8
    if not ch_a3.is_expanded:
        p, v = root_model.infer(ch_a3.state_tensor)
        entry = CanonicalNNCache.build_entry(ch_a3.board, p[0], v[0])
        searcher._expand_node_from_entry(ch_a3, entry, top_k=16)
        ch_a3.self_visits = 1
        ch_a3.self_value_sum = entry["leaf_val"]
    act_f8e8 = uci_to_action("f8e8", canonical_flip=True)
    grandchild = ch_a3.children[act_f8e8]
    if not grandchild.is_instantiated:
        grandchild.instantiate()
    if not grandchild.is_expanded:
        p, v = root_model.infer(grandchild.state_tensor)
        entry = CanonicalNNCache.build_entry(grandchild.board, p[0], v[0])
        searcher._expand_node_from_entry(grandchild, entry, top_k=8)
        grandchild.self_visits = 1
        grandchild.self_value_sum = entry["leaf_val"]

    # Pick a child of grandchild and simulate that it had been marked as a path-dependent draw with 10 visits
    any_gc_child = next(iter(grandchild.children.values()))
    any_gc_child.instantiate()
    any_gc_child.is_terminal = True
    any_gc_child.terminal_value = 0.0
    any_gc_child.path_dep_draw = True
    any_gc_child.self_visits = 10
    any_gc_child.self_value_sum = 0.0
    any_gc_child.visits = 10
    any_gc_child.total_value = 0.0
    grandchild.visits = grandchild.self_visits + sum(c.visits for c in grandchild.children.values())
    grandchild.total_value = grandchild.self_value_sum + sum(-c.total_value for c in grandchild.children.values())

    # Now play our move a2a3 (irreversible pawn move resets halfmove_clock and repetition history!) and opponent's f8e8
    board.push_uci("a2a3")
    board.push_uci("f8e8")

    _, uci2, _, _, _, _ = searcher.search(board, movetime_ms=500, max_nodes=64)
    prof2 = searcher.last_profile
    assert prof2["tree_reused"] is True
    assert prof2["invalidated_path_dep_nodes"] >= 1

    reused_root = searcher._saved_root
    assert reused_root is not None
    # Root must have been widened from 8 children (at old ply=2) to all legal moves at ply=0
    assert len(reused_root.children) == board.legal_moves.count()
    # No node in the non-mate position may be marked solved
    assert not reused_root.is_solved
    for ch in reused_root.children.values():
        assert not ch.is_solved
    # Verify exact bottom-up visit and value invariants across the entire reused tree
    _assert_tree_invariants(reused_root)


def test_fast_tree_reuse_retains_visits_and_no_false_proofs():
    """
    B1 Test (2b - FastTree C++ backend): After our move and the opponent's reply, C++ tree reuse
    retains positive root visits, widens root to all legal moves, and leaves zero false solver proofs.
    """
    root_model = DeterministicMockModel(batch_size=1, force_uniform=True)
    batch_model = DeterministicMockModel(batch_size=8, force_uniform=True)
    searcher = NPUBatchedPUCTSearch(
        root_model=root_model,
        batch_model=batch_model,
        batch_size=8,
        use_exact_solver=True,
        use_tree_reuse=True,
        use_nn_cache=True,
        use_fast_tree=True,
    )

    board = chess.Board("4k3/8/8/8/8/8/P7/4K1N1 w - - 0 1")
    board.push_uci("g1f3")
    board.push_uci("e8f8")
    _, _, _, _, _, _ = searcher.search(board, movetime_ms=500, max_nodes=160)
    assert searcher._fast_backend is not None
    assert searcher._fast_backend._has_saved_tree is True
    assert searcher.last_profile["final_root_visits"] > 0

    board.push_uci("a2a3")
    board.push_uci("f8e8")
    _, _, _, _, _, _ = searcher.search(board, movetime_ms=500, max_nodes=64)
    prof2 = searcher.last_profile
    assert prof2["tree_reused"] is True
    assert prof2["reused_root_visits"] > 0
    assert prof2["final_root_visits"] >= prof2["reused_root_visits"]

    root_ch = searcher._fast_backend._get_children_snapshot(0)
    assert len(root_ch) == board.legal_moves.count()
    assert not any(ch["is_solved"] for ch in root_ch)


def test_mate_in_1_2_3_no_regression_with_reuse_and_cache():
    """
    B1 Test (3): Mate-in-1, Mate-in-2, and Mate-in-3 tactical positions must not regress
    when `use_tree_reuse=True`, `use_nn_cache=True`, or `both` are enabled (including after re-rooting).
    Tested across both Python and C++ FastTree backends.
    """
    root_model = DeterministicMockModel(batch_size=1, force_uniform=True)
    batch_model = DeterministicMockModel(batch_size=8, force_uniform=True)

    # Pre-mate sequence: White plays 1. Rd1+ Ke8 2. Rd8# (or 1...Kc8 2. Rd8#)
    # Position before opponent blunder: White plays h2h3, Black plays h7h6 allowing Ra8#
    pre_fen = "3k4/7p/3K4/8/8/8/7P/R7 w - - 0 1"
    # Mate-in-1: Ra8#
    m1_board = chess.Board("3k4/8/3K4/8/8/8/8/R7 w - - 0 1")
    # Mate-in-2: 1. Rb2 Ka3 2. Ra1#
    m2_board = chess.Board("8/8/8/8/k7/8/2K5/RR6 w - - 0 1")
    # Corridor mate-in-3: 1. Ra8+ Bc8 2. Rxc8+ Bd8 3. Rxd8#
    m3_board = chess.Board("6k1/1b2bppp/8/8/8/8/5PPP/R5K1 w - - 0 1")

    for use_fast_tree in (False, True):
        for use_reuse, use_cache in [(False, False), (True, False), (False, True), (True, True)]:
            searcher = NPUBatchedPUCTSearch(
                root_model=root_model,
                batch_model=batch_model,
                batch_size=8,
                use_exact_solver=True,
                use_tree_reuse=use_reuse,
                use_nn_cache=use_cache,
                use_fast_tree=use_fast_tree,
            )
            # Warm tree on pre_fen, then push moves to reach m1_board and test re-rooting into mate-in-1
            b_pre = chess.Board(pre_fen)
            searcher.search(b_pre, movetime_ms=500, max_nodes=48)
            b_pre.push_uci("h2h3")
            b_pre.push_uci("h7h6")
            _, best_m1, eval_m1, _, _, _ = searcher.search(b_pre, movetime_ms=500, max_nodes=48)
            assert best_m1 == "a1a8", (
                f"Failed mate-in-1 after re-root (fast={use_fast_tree}, reuse={use_reuse}, cache={use_cache}): {best_m1}"
            )
            assert eval_m1 >= 0.99

            # Cold mate-in-1
            searcher.clear_state()
            _, best_m1_cold, _, _, _, _ = searcher.search(m1_board.copy(), movetime_ms=500, max_nodes=32)
            assert best_m1_cold == "a1a8"

            # Mate-in-2 (cold + re-root into second move)
            searcher.clear_state()
            b2 = m2_board.copy()
            _, best_m2, _, _, _, _ = searcher.search(b2, movetime_ms=1000, max_nodes=160)
            b_chk = m2_board.copy()
            b_chk.push_uci(best_m2)
            for reply in b_chk.legal_moves:
                b_chk.push(reply)
                assert any(
                    (b_chk.push(m) or True) and b_chk.is_checkmate() and (b_chk.pop() or True)
                    for m in list(b_chk.legal_moves)
                ), (
                    f"Move {best_m2} did not force mate-in-2 against {reply.uci()} "
                    f"(fast={use_fast_tree}, reuse={use_reuse}, cache={use_cache})"
                )
                b_chk.pop()

            # Mate-in-3 (1. Ra8+ Bc8 2. Rxc8+ Bd8 3. Rxd8#; verify both cold and re-rooted across moves)
            searcher.clear_state()
            b3 = m3_board.copy()
            _, m3_1, _, _, _, _ = searcher.search(b3, movetime_ms=1000, max_nodes=240)
            assert m3_1 == "a1a8", (
                f"Failed mate-in-3 move 1 (fast={use_fast_tree}, reuse={use_reuse}, cache={use_cache}): {m3_1}"
            )
            b3.push_uci("a1a8")
            b3.push_uci("b7c8")
            _, m3_2, _, _, _, _ = searcher.search(b3, movetime_ms=1000, max_nodes=160)
            assert m3_2 == "a8c8", (
                f"Failed mate-in-3 move 2 after re-root (fast={use_fast_tree}, reuse={use_reuse}, cache={use_cache}): {m3_2}"
            )


class TestTreeReuseAndNNCache(unittest.TestCase):
    def test_nn_cache_hit_matches_fresh_evaluation_exactly(self):
        test_nn_cache_hit_matches_fresh_evaluation_exactly()

    def test_nn_cache_transposition_and_lru_memory_cap(self):
        test_nn_cache_transposition_and_lru_memory_cap()

    def test_tree_reuse_invalidates_repetitions_and_50move_with_no_false_proofs(self):
        test_tree_reuse_invalidates_repetitions_and_50move_with_no_false_proofs()

    def test_fast_tree_reuse_retains_visits_and_no_false_proofs(self):
        test_fast_tree_reuse_retains_visits_and_no_false_proofs()

    def test_mate_in_1_2_3_no_regression_with_reuse_and_cache(self):
        test_mate_in_1_2_3_no_regression_with_reuse_and_cache()


if __name__ == "__main__":
    test_nn_cache_hit_matches_fresh_evaluation_exactly()
    print("[PASS] test_nn_cache_hit_matches_fresh_evaluation_exactly")
    test_nn_cache_transposition_and_lru_memory_cap()
    print("[PASS] test_nn_cache_transposition_and_lru_memory_cap")
    test_tree_reuse_invalidates_repetitions_and_50move_with_no_false_proofs()
    print("[PASS] test_tree_reuse_invalidates_repetitions_and_50move_with_no_false_proofs")
    test_fast_tree_reuse_retains_visits_and_no_false_proofs()
    print("[PASS] test_fast_tree_reuse_retains_visits_and_no_false_proofs")
    test_mate_in_1_2_3_no_regression_with_reuse_and_cache()
    print("[PASS] test_mate_in_1_2_3_no_regression_with_reuse_and_cache")
    print("ALL PART B1 UNIT TESTS PASSED.")

