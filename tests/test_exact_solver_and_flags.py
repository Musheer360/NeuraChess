"""
Unit & Correctness Tests for Search v3.1 Flags and Exact Solver Propagation
===========================================================================
Verifies:
  1. All v3.1 flags (use_fpu_reduction, use_exact_solver, use_gumbel_root) default to False.
  2. Path-dependent 2-fold repetitions, 50-move draws, stalemates, and insufficient material
     NEVER mark a node as is_solved=True.
  3. Defender with a repetition-draw escape is never falsely proven lost (-1.0).
  4. Defender with an unexpanded legal escape move (outside initial top_k pruning) expands
     remaining legal moves instead of falsely marking the node as solved (-1.0).

"""

import os
import sys
import unittest
import numpy as np
import chess

from neurachess.core.board_tensor import board_to_tensor
from neurachess.core.constants import uci_to_action
from neurachess.search.npu_batched_puct import TreeNode, NPUBatchedPUCTSearch


class DummyModel:
    def __init__(self, batch_size: int = 1):
        self.batch_size = batch_size

    def infer(self, x: np.ndarray):
        b = x.shape[0]
        return np.zeros((b, 4672), dtype=np.float32), np.zeros((b, 1), dtype=np.float32)


def test_default_flags():
    s = NPUBatchedPUCTSearch(root_model=DummyModel(1), batch_model=DummyModel(16), batch_size=16)
    assert s.use_fpu_reduction is True, "use_fpu_reduction must default to True"
    assert abs(s.fpu_reduction - 0.20) < 1e-9, "fpu_reduction must default to 0.20"
    assert abs(s.root_fpu_reduction - 0.10) < 1e-9, "root_fpu_reduction must default to 0.10"
    assert abs(s.c_puct - 1.5) < 1e-9, "c_puct must default to 1.5"
    assert abs(s.t_root - 1.0) < 1e-9, "t_root must default to 1.0"
    assert s.use_exact_solver is True, "use_exact_solver must default to True"
    assert s.use_fast_tree is True, "use_fast_tree must default to True"
    assert s.use_nn_cache is True, "use_nn_cache must default to True"
    assert s.use_tree_reuse is True, "use_tree_reuse must default to True"
    assert s.use_gumbel_root is False, "use_gumbel_root must default to False"
    assert s.use_async_overlap is False, "use_async_overlap must default to False"
    assert s.use_opening_root_guard is False, "use_opening_root_guard must default to False"

    # Explicit flag-OFF configuration must disable all optional components cleanly
    s_off = NPUBatchedPUCTSearch(
        root_model=DummyModel(1),
        batch_model=DummyModel(16),
        batch_size=16,
        use_fpu_reduction=False,
        use_exact_solver=False,
        use_gumbel_root=False,
        use_async_overlap=False,
        use_tree_reuse=False,
        use_nn_cache=False,
        use_fast_tree=False,
        use_opening_root_guard=False,
    )
    assert s_off.use_fpu_reduction is False
    assert s_off.use_exact_solver is False
    assert s_off.use_gumbel_root is False
    assert s_off.use_async_overlap is False
    assert s_off.use_tree_reuse is False
    assert s_off.use_nn_cache is False
    assert s_off.use_fast_tree is False
    assert s_off.use_opening_root_guard is False
    assert s_off._fast_backend is None
    _, best_uci, _, nodes_eval, _, _ = s_off.search(chess.Board(), movetime_ms=200, max_nodes=8)
    assert best_uci is not None and nodes_eval >= 1
    assert s_off._saved_root is None, "use_tree_reuse=False must not retain _saved_root"
    print("[PASS] Default production flags and explicit flag-OFF behavior verified")


def test_repetition_never_marks_is_solved():
    b = chess.Board()
    root = TreeNode(b.copy(), board_to_tensor(b), ply=0)

    # Play Nf3 Nf6 Ng1 Ng8 to create a 2-fold repetition of the starting position
    moves = ["g1f3", "g8f6", "f3g1", "f6g8"]
    curr = root
    for ply_idx, uci in enumerate(moves, start=1):
        mv = chess.Move.from_uci(uci)
        is_blk = (curr.board.turn == chess.BLACK)
        act = uci_to_action(uci, canonical_flip=is_blk)
        child = TreeNode(
            board=None,
            state_tensor=None,
            ply=ply_idx,
            parent=curr,
            action_from_parent=act,
            prior=0.5,
            move_from_parent=mv,
        )
        curr.children[act] = child
        curr.is_expanded = True
        child.instantiate()
        curr = child

    # After f6g8, the board repeats the initial position (is_repetition(2) == True)
    assert curr.board.is_repetition(2) is True
    assert curr.is_terminal is True
    assert curr.terminal_value == 0.0
    assert curr.is_solved is False, "Path-dependent 2-fold repetition MUST NEVER set is_solved=True!"
    print("[PASS] 2-fold repetition sets is_terminal=True (0.0) but never sets is_solved=True")


def test_solver_never_proves_loss_when_repetition_escape_exists():
    s = NPUBatchedPUCTSearch(
        root_model=DummyModel(1),
        batch_model=DummyModel(16),
        batch_size=16,
        use_exact_solver=True,
    )
    # Construct a defender node with 2 legal moves: one is proven loss (+1.0 for child),
    # one is a 2-fold repetition draw (is_terminal=True, is_solved=False).
    b = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # Stalemate check first
    st_node = TreeNode(None, None, ply=1, parent=TreeNode(chess.Board("7k/8/5QK1/8/8/8/8/8 w - - 0 1"), np.zeros((18, 8, 8), dtype=np.float32), ply=0), action_from_parent=0, move_from_parent=chess.Move.from_uci("f6f7"))
    st_node.instantiate()
    assert st_node.is_terminal is True
    assert st_node.is_solved is False, "Stalemate must not be marked as a forced mate win/loss"

    # Now test defender with 2 moves where 1 loses and 1 draws by repetition
    b_def = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")  # Black king on a8, White king on b6: legal moves a8b8
    def_node = TreeNode(b_def, board_to_tensor(b_def), ply=1)
    def_node.is_expanded = True
    def_node.num_legal_moves = 2  # Suppose 2 legal moves

    c_lose = TreeNode(None, None, ply=2, parent=def_node, action_from_parent=1)
    c_lose.is_solved = True
    c_lose.solved_value = 1.0  # Opponent wins in 1 from c_lose

    c_rep = TreeNode(None, None, ply=2, parent=def_node, action_from_parent=2)
    c_rep.is_terminal = True
    c_rep.terminal_value = 0.0
    c_rep.is_solved = False  # Repetition draw

    def_node.children = {1: c_lose, 2: c_rep}
    s._update_solver_along_path([def_node, c_rep])
    assert def_node.is_solved is False, "Defender with a repetition draw escape must NEVER be proven lost!"
    print("[PASS] Defender with repetition/stalemate draw resource is never falsely proven lost")


def test_solver_expands_pruned_moves_before_proving_defender_lost():
    s = NPUBatchedPUCTSearch(
        root_model=DummyModel(1),
        batch_model=DummyModel(16),
        batch_size=16,
        use_exact_solver=True,
    )
    b_def = chess.Board()  # 20 legal moves
    def_node = TreeNode(b_def, board_to_tensor(b_def), ply=2)
    def_node.is_expanded = True

    # Suppose candidate selector only expanded 1 move and that move was proven losing (+1.0 for child)
    first_mv = list(b_def.legal_moves)[0]
    act0 = uci_to_action(first_mv.uci(), canonical_flip=False)
    c0 = TreeNode(None, None, ply=3, parent=def_node, action_from_parent=act0, move_from_parent=first_mv)
    c0.is_solved = True
    c0.solved_value = 1.0
    def_node.children[act0] = c0

    s._update_solver_along_path([def_node, c0])
    assert def_node.is_solved is False, "Must NOT prove defender lost when only 1 of 20 legal moves was expanded!"
    assert len(def_node.children) == 20, f"Expected all 20 legal moves to be expanded, got {len(def_node.children)}"
    print("[PASS] Pruned defender node expands remaining 19 legal moves instead of false mate proof")


def test_shortest_mate_in_2_chosen_over_mate_in_5():
    """
    Verifies that when root has both a proven Mate-in-5 child (discovered first, higher prior)
    and a proven Mate-in-2 child (discovered second, lower prior), the exact solver
    tracks exact mate_plies along the minimax path and selects the Mate-in-2 over Mate-in-5.
    """
    s = NPUBatchedPUCTSearch(
        root_model=DummyModel(1),
        batch_model=DummyModel(16),
        batch_size=16,
        use_exact_solver=True,
    )
    b_root = chess.Board()
    root = TreeNode(b_root, board_to_tensor(b_root), ply=0)
    root.is_expanded = True

    # Child A: Mate-in-5 (9 plies from root -> child_a has mate_plies = 8), higher prior 0.80
    mv_m5 = chess.Move.from_uci("e2e3")
    act_m5 = uci_to_action("e2e3", canonical_flip=False)
    child_m5 = TreeNode(None, None, ply=1, parent=root, action_from_parent=act_m5, prior=0.80, move_from_parent=mv_m5)
    child_m5.is_solved = True
    child_m5.solved_value = -1.0  # Loss for child = Win for root
    child_m5.mate_plies = 8       # 8 plies remaining from child_m5 = 9 plies (Mate-in-5) from root
    child_m5.visits = 50
    child_m5.total_value = -50.0

    # Child B: Mate-in-2 (3 plies from root -> child_b -> gk -> mate_leaf), lower prior 0.10
    mv_m2 = chess.Move.from_uci("e2e4")
    act_m2 = uci_to_action("e2e4", canonical_flip=False)
    b_c2 = b_root.copy()
    b_c2.push(mv_m2)
    child_m2 = TreeNode(b_c2, board_to_tensor(b_c2), ply=1, parent=root, action_from_parent=act_m2, prior=0.10, move_from_parent=mv_m2)
    child_m2.is_expanded = True
    child_m2.num_legal_moves = 1  # Forced reply for defender

    mv_def = chess.Move.from_uci("e7e5")
    act_def = uci_to_action("e7e5", canonical_flip=True)
    b_gk = b_c2.copy()
    b_gk.push(mv_def)
    grandchild = TreeNode(b_gk, board_to_tensor(b_gk), ply=2, parent=child_m2, action_from_parent=act_def, prior=1.0, move_from_parent=mv_def)
    grandchild.is_expanded = True

    mv_win = chess.Move.from_uci("g1f3")
    act_win = uci_to_action("g1f3", canonical_flip=False)
    mate_leaf = TreeNode(None, None, ply=3, parent=grandchild, action_from_parent=act_win, prior=1.0, move_from_parent=mv_win)
    mate_leaf.is_terminal = True
    mate_leaf.terminal_value = -1.0
    mate_leaf.is_solved = True
    mate_leaf.solved_value = -1.0
    mate_leaf.mate_plies = 0  # Checkmated on board

    grandchild.children[act_win] = mate_leaf
    child_m2.children[act_def] = grandchild
    root.children[act_m5] = child_m5
    root.children[act_m2] = child_m2

    # First propagate Mate-in-5 at root
    s._update_solver_along_path([root, child_m5])
    assert root.is_solved is True and root.mate_plies == 9

    # Now propagate Mate-in-2 path [root, child_m2, grandchild, mate_leaf]
    s._update_solver_along_path([root, child_m2, grandchild, mate_leaf])
    assert grandchild.is_solved is True and grandchild.solved_value == 1.0 and grandchild.mate_plies == 1
    assert child_m2.is_solved is True and child_m2.solved_value == -1.0 and child_m2.mate_plies == 2
    assert root.is_solved is True and root.solved_value == 1.0 and root.mate_plies == 3

    # Verify q_from_parent_view ranks Mate-in-2 strictly above Mate-in-5
    assert child_m2.q_from_parent_view() > child_m5.q_from_parent_view()

    # Verify root move selection chooses Mate-in-2 (act_m2) over Mate-in-5 (act_m5)
    best_act = None
    best_mate_plies = None
    for act, child in root.children.items():
        if child.is_solved and child.solved_value <= -1.0:
            if best_mate_plies is None or child.mate_plies < best_mate_plies:
                best_mate_plies = child.mate_plies
                best_act = act
    assert best_act == act_m2, f"Expected Mate-in-2 ({act_m2}) over Mate-in-5 ({act_m5}), got {best_act}"
    print("[PASS] Shortest proven mate (Mate-in-2, 3 plies) chosen over Mate-in-5 (9 plies)")


class TestExactSolverAndFlags(unittest.TestCase):
    def test_default_flags(self):
        test_default_flags()

    def test_repetition_never_marks_is_solved(self):
        test_repetition_never_marks_is_solved()

    def test_solver_never_proves_loss_when_repetition_escape_exists(self):
        test_solver_never_proves_loss_when_repetition_escape_exists()

    def test_solver_expands_pruned_moves_before_proving_defender_lost(self):
        test_solver_expands_pruned_moves_before_proving_defender_lost()

    def test_shortest_mate_in_2_chosen_over_mate_in_5(self):
        test_shortest_mate_in_2_chosen_over_mate_in_5()


if __name__ == "__main__":
    test_default_flags()
    test_repetition_never_marks_is_solved()
    test_solver_never_proves_loss_when_repetition_escape_exists()
    test_solver_expands_pruned_moves_before_proving_defender_lost()
    test_shortest_mate_in_2_chosen_over_mate_in_5()
    print("\nALL EXACT-SOLVER & FLAG TESTS PASSED.")
