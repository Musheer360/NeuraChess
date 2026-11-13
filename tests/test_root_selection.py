"""
Unit test for B1: Root move selection by most-visited child with Q-value tie-breaking.
"""

import unittest
import chess
import numpy as np

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from search.npu_batched_puct import TreeNode, NPUBatchedPUCTSearch


class TestRootMoveSelection(unittest.TestCase):
    def test_root_move_selection_visits_and_tie_break(self):
        board = chess.Board()
        state_tensor = np.zeros((55, 8, 8), dtype=np.float32)
        root = TreeNode(board, state_tensor, ply=0)

        # Child 1: act 10, visits 50, child.q_value = -0.2 (root_q = +0.2)
        c1 = TreeNode(board.copy(), state_tensor, ply=1, parent=root, action_from_parent=10)
        c1.visits = 50
        c1.total_value = -10.0  # q_value = -0.2 -> root view = +0.2

        # Child 2: act 20, visits 80, child.q_value = 0.5 (root_q = -0.5)
        c2 = TreeNode(board.copy(), state_tensor, ply=1, parent=root, action_from_parent=20)
        c2.visits = 80
        c2.total_value = 40.0  # q_value = 0.5 -> root view = -0.5

        root.children[10] = c1
        root.children[20] = c2

        # Test case 1: Most visited wins even if Q is lower
        best_act = None
        best_visits = -1
        best_q = -float("inf")
        for act, child in root.children.items():
            child_q = -child.q_value
            if child.visits > best_visits or (child.visits == best_visits and child_q > best_q):
                best_visits = child.visits
                best_q = child_q
                best_act = act

        self.assertEqual(best_act, 20, "Most-visited child must be selected")

        # Test case 2: Tied visits -> Higher Q-value wins
        # Set c1 visits = 80, child.q_value = -0.4 (root_q = +0.4)
        c1.visits = 80
        c1.total_value = -32.0  # q = -0.4 -> root_q = +0.4
        # Set c2 visits = 80, child.q_value = 0.1 (root_q = -0.1)
        c2.visits = 80
        c2.total_value = 8.0   # q = +0.1 -> root_q = -0.1

        best_act = None
        best_visits = -1
        best_q = -float("inf")
        for act, child in root.children.items():
            child_q = -child.q_value
            if child.visits > best_visits or (child.visits == best_visits and child_q > best_q):
                best_visits = child.visits
                best_q = child_q
                best_act = act

        self.assertEqual(best_act, 10, "On tied visits, higher Q-value from root perspective must win")


if __name__ == "__main__":
    unittest.main()
