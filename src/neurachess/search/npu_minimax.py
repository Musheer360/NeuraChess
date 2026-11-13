"""
NeuraChess Vectorized NPU Minimax Baseline Search (legacy fixed-horizon search).
Retained for reference comparison against asynchronous batched PUCT.
"""

import time
from pathlib import Path
import sys
import numpy as np
import chess

from neurachess.core.constants import action_to_uci
from neurachess.core.tensor_movegen import get_legal_action_mask
from neurachess.core.tensor_transitions import apply_tensor_transition
from neurachess.models.npu_compiler import NPUModel


class NPUMinimaxSearch:
    """Vectorized 2-ply NPU Minimax Search Engine."""

    def __init__(self, root_model: NPUModel | None = None, batch_k_model: NPUModel | None = None, k: int = 16):
        self.k = k
        self.root_model = root_model if root_model is not None else NPUModel(batch_size=1)
        self.batch_model = batch_k_model if batch_k_model is not None else NPUModel(batch_size=self.k)

    def search_deep(
        self,
        board: chess.Board,
        board_tensor: np.ndarray,
        movetime_ms: int = 600,
        alpha_policy: float = 0.3,
        beta_value: float = 0.7,
    ) -> tuple[int, str, float, int]:
        """Evaluate root and top-K candidate replies until `movetime_ms` expires."""
        is_black = (board.turn == chess.BLACK)
        legal_mask = get_legal_action_mask(board)
        n_legal = int(legal_mask.sum())

        if n_legal == 0:
            return -1, "0000", 0.0, 0

        if n_legal == 1:
            best_act = int(np.argmax(legal_mask))
            return best_act, action_to_uci(best_act, canonical_flip=is_black), 0.0, 1

        t_deadline = time.time() + (movetime_ms / 1000.0)

        policy_logits, _ = self.root_model.infer(board_tensor)
        p_logits = policy_logits[0]
        masked_logits = p_logits + (legal_mask - 1.0) * 1e4

        cand_count = min(self.k, n_legal)
        sorted_indices = np.argsort(masked_logits)[::-1]
        top_k_actions = sorted_indices[:cand_count]

        child_states = np.zeros((self.k,) + board_tensor.shape, dtype=np.float32)
        for i, act in enumerate(top_k_actions):
            child_states[i] = apply_tensor_transition(board_tensor, act)
        for i in range(cand_count, self.k):
            child_states[i] = child_states[0]

        nodes_evaluated = 1 + self.k
        accumulated_child_values = np.zeros(self.k, dtype=np.float32)
        rollout_iterations = 0

        while time.time() < t_deadline or rollout_iterations < 5:
            _, child_values = self.batch_model.infer(child_states)
            accumulated_child_values += -child_values[:, 0]
            nodes_evaluated += self.k
            rollout_iterations += 1

        avg_child_values = accumulated_child_values / rollout_iterations
        norm_p = np.exp(masked_logits[top_k_actions] - np.max(masked_logits[top_k_actions]))
        norm_p = norm_p / np.sum(norm_p)

        scores = alpha_policy * norm_p + beta_value * avg_child_values[:cand_count]
        best_k_idx = int(np.argmax(scores))
        best_action = int(top_k_actions[best_k_idx])
        best_uci = action_to_uci(best_action, canonical_flip=is_black)
        best_eval = float(avg_child_values[best_k_idx])
        return best_action, best_uci, best_eval, nodes_evaluated

    search = search_deep
