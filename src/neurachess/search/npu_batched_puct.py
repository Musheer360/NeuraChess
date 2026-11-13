# Copyright (c) 2026 Musheer Alam. Licensed under GPL-3.0.
"""
NeuraChess Vectorized Batched PUCT Search Engine (v3.0 - Lazy Evaluation)

Implements mathematically sound batched tree search on Intel AI Boost NPU:
  - Lazy Child Instantiation: Unvisited candidates avoid board copies, move pushes,
    tensor transitions, and terminal checks entirely.
  - Fast O(1) Terminal Detection: Replaces slow full-movegen simulation with fast check/generator tests.
  - Exact Terminal Solver Propagation: Proven checkmates (+1.0 / -1.0) propagate with infinite priority.
  - Vectorized PUCT Selection: Q(s, a) + c_puct * P(s, a) * sqrt(sum(N)) / (1 + N(s, a)) with virtual loss.
  - Multi-worker cloning: Up to 32 concurrent games saturate the Intel NPU.
"""

import logging
import time
import math
from collections import OrderedDict
import numpy as np
import chess

logger = logging.getLogger(__name__)

from neurachess.core.constants import TOTAL_ACTIONS, NUM_PLANES, uci_to_action, action_to_uci, move_to_action
from neurachess.core.board_tensor import board_to_tensor
from neurachess.core.tensor_transitions import apply_tensor_transition
from neurachess.models.npu_compiler import NPUModel
from neurachess.search.candidate_selector import select_top_candidates, select_opening_guarded_root_act, PIECE_VALUES


class CanonicalNNCache:
    """
    LRU neural-network evaluation cache keyed by canonical position hash (`board._transposition_key()`).
    Stores all legal-move logits + unsorted/sorted candidate orders + leaf value so that a cache hit
    at any branching_k (root all-moves, 16, 8, or 4) returns bit-for-bit identical priors and value
    to a fresh evaluation. Enforces a strict memory cap (default 64 MB).
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024):
        self.max_bytes = int(max_bytes)
        self.used_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._store: OrderedDict[tuple, dict] = OrderedDict()

    @staticmethod
    def canonical_key(board: chess.Board) -> tuple:
        return board._transposition_key()

    @staticmethod
    def build_entry(board: chess.Board, p_logits: np.ndarray, v_out: np.ndarray) -> dict:
        logits_1d = p_logits[0] if p_logits.ndim == 2 else p_logits
        if v_out.ndim == 2:
            v_1d = v_out[0]
        else:
            v_1d = v_out
        if len(v_1d) == 3:
            leaf_val = float(v_1d[0] - v_1d[2])
        else:
            leaf_val = float(v_1d[0])

        is_black = (board.turn == chess.BLACK)
        legal_moves = list(board.legal_moves)
        n_legal = len(legal_moves)
        if n_legal == 0:
            return {
                "unsorted_acts": (),
                "unsorted_moves": (),
                "unsorted_logits": np.empty(0, dtype=logits_1d.dtype),
                "sorted_order": (),
                "leaf_val": leaf_val,
                "bytes": 256,
            }

        unsorted_acts = []
        scored = []
        for idx, move in enumerate(legal_moves):
            act = move_to_action(move, canonical_flip=is_black)
            unsorted_acts.append(act)
            base_logit = float(logits_1d[act])
            tactical_bonus = 0.0
            if board.is_capture(move):
                attacker_type = board.piece_type_at(move.from_square) or chess.PAWN
                attacker_val = PIECE_VALUES.get(attacker_type, 100)
                if board.is_en_passant(move):
                    victim_val = PIECE_VALUES[chess.PAWN]
                else:
                    victim_type = board.piece_type_at(move.to_square) or chess.PAWN
                    victim_val = PIECE_VALUES.get(victim_type, 100)
                tactical_bonus = 0.35 if victim_val >= attacker_val else 0.10
            scored.append((base_logit + tactical_bonus, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        sorted_order = tuple(x[1] for x in scored)
        unsorted_acts_t = tuple(unsorted_acts)
        unsorted_moves_t = tuple(legal_moves)
        unsorted_logits = np.array(logits_1d[list(unsorted_acts_t)], copy=True)
        est_bytes = 256 + n_legal * 48 + int(unsorted_logits.nbytes)
        return {
            "unsorted_acts": unsorted_acts_t,
            "unsorted_moves": unsorted_moves_t,
            "unsorted_logits": unsorted_logits,
            "sorted_order": sorted_order,
            "leaf_val": leaf_val,
            "bytes": est_bytes,
        }

    @staticmethod
    def expand_from_entry(entry: dict, top_k: int) -> tuple[list[int], list[chess.Move], np.ndarray, float]:
        u_acts = entry["unsorted_acts"]
        n_legal = len(u_acts)
        leaf_val = entry["leaf_val"]
        if n_legal == 0:
            return [], [], np.empty(0, dtype=np.float32), leaf_val
        if n_legal <= top_k:
            c_acts = list(u_acts)
            c_moves = list(entry["unsorted_moves"])
            l_sel = entry["unsorted_logits"]
        else:
            idxs = entry["sorted_order"][:top_k]
            u_moves = entry["unsorted_moves"]
            u_logits = entry["unsorted_logits"]
            c_acts = [u_acts[i] for i in idxs]
            c_moves = [u_moves[i] for i in idxs]
            l_sel = u_logits[list(idxs)]
        e_l = np.exp(l_sel - np.max(l_sel))
        p_sel = e_l / np.sum(e_l)
        return c_acts, c_moves, p_sel, leaf_val

    def get(self, key: tuple) -> dict | None:
        entry = self._store.get(key)
        if entry is not None:
            self._store.move_to_end(key)
            self.hits += 1
            return entry
        self.misses += 1
        return None

    def put(self, key: tuple, entry: dict):
        eb = entry["bytes"]
        if eb > self.max_bytes:
            return
        old = self._store.pop(key, None)
        if old is not None:
            self.used_bytes -= old["bytes"]
        while self.used_bytes + eb > self.max_bytes and self._store:
            _, evicted = self._store.popitem(last=False)
            self.used_bytes -= evicted["bytes"]
            self.evictions += 1
        self._store[key] = entry
        self.used_bytes += eb

    def __len__(self) -> int:
        return len(self._store)

    def clear(self):
        self._store.clear()
        self.used_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0


class TreeNode:
    __slots__ = (
        "board",
        "state_tensor",
        "ply",
        "parent",
        "action_from_parent",
        "move_from_parent",
        "children",
        "is_expanded",
        "is_terminal",
        "terminal_value",
        "visits",
        "total_value",
        "virtual_loss",
        "prior",
        "is_instantiated",
        "is_solved",
        "solved_value",
        "mate_plies",
        "num_legal_moves",
        "self_visits",
        "self_value_sum",
        "path_dep_draw",
        "nn_entry",
        "encoder_layout",
    )

    def __init__(
        self,
        board: chess.Board | None,
        state_tensor: np.ndarray | None,
        ply: int = 0,
        parent: "TreeNode | None" = None,
        action_from_parent: int | None = None,
        prior: float = 0.0,
        move_from_parent: chess.Move | None = None,
        encoder_layout: str = "gpu_unpack",
    ):
        self.board = board
        self.state_tensor = state_tensor
        self.ply = ply
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.move_from_parent = move_from_parent
        self.encoder_layout = parent.encoder_layout if parent is not None else encoder_layout
        self.children: dict[int, TreeNode] = {}
        self.is_expanded = False
        self.is_terminal = False
        self.terminal_value = 0.0
        self.visits = 0
        self.total_value = 0.0
        self.virtual_loss = 0
        self.prior = prior
        self.is_instantiated = (board is not None)
        self.is_solved = False
        self.solved_value = 0.0
        self.mate_plies = 0
        self.num_legal_moves: int | None = None
        self.self_visits = 0
        self.self_value_sum = 0.0
        self.path_dep_draw = False
        self.nn_entry: dict | None = None

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def q_from_parent_view(self, virtual_loss_penalty: float = 1.0, fpu_value: float = 0.0) -> float:
        if self.is_solved:
            if self.solved_value <= -1.0:
                return 1000.0 - float(self.mate_plies)  # Shorter proven mate has strictly higher score
            if self.solved_value >= 1.0:
                return -1000.0 + float(self.mate_plies)  # Defender prefers delaying mate longer

        eff_visits = self.visits + self.virtual_loss
        if eff_visits == 0:
            return fpu_value
        # From child perspective, value sum is self.total_value.
        # From parent perspective, normal value sum is -self.total_value.
        # Virtual loss treats in-flight evaluations as temporary losses (-1.0) from parent view.
        parent_total = -self.total_value - (self.virtual_loss * virtual_loss_penalty)
        return parent_total / eff_visits

    def _evaluate_terminal_flags(self, b: chess.Board):
        self.is_terminal = False
        self.terminal_value = 0.0
        self.path_dep_draw = False
        self.is_solved = False
        self.solved_value = 0.0
        self.mate_plies = 0
        # 1. 50-move rule (path-dependent; never mark is_solved)
        if b.halfmove_clock >= 100:
            self.is_terminal = True
            self.terminal_value = 0.0
            self.path_dep_draw = True
        # 2. Insufficient material
        elif b.is_insufficient_material():
            self.is_terminal = True
            self.terminal_value = 0.0
        # 3. Twofold repetition (path-dependent; never mark is_solved)
        elif b.is_repetition(2):
            self.is_terminal = True
            self.terminal_value = 0.0
            self.path_dep_draw = True
        # 4. Checkmate / Stalemate (fast exit without generating full move list)
        elif not any(b.generate_legal_moves()):
            self.is_terminal = True
            if b.is_check():
                # Child is checkmated: from child perspective it is -1.0 (exact proven mate)
                self.terminal_value = -1.0
                self.is_solved = True
                self.solved_value = -1.0
            else:
                # Stalemate (draw, not a forced mate win/loss)
                self.terminal_value = 0.0

    def instantiate(self):
        if self.is_instantiated:
            return
        assert self.parent is not None and self.parent.board is not None
        b = self.parent.board.copy()
        b.push(self.move_from_parent)
        self.board = b
        self.state_tensor = board_to_tensor(b, layout=self.encoder_layout)
        self._evaluate_terminal_flags(b)
        self.is_instantiated = True


class NPUBatchedPUCTSearch:
    def __init__(
        self,
        root_model: NPUModel | None = None,
        batch_model: NPUModel | None = None,
        batch_size: int = 16,
        c_puct: float = 1.5,
        contempt: float = 0.02,
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
        encoder_layout: str = "gpu_unpack",
        use_fast_tree: bool = True,
        t_root: float = 1.0,
        use_opening_root_guard: bool = False,
    ):
        self.encoder_layout = encoder_layout
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.t_root = float(t_root)
        self.use_opening_root_guard = bool(use_opening_root_guard)
        self.contempt = contempt
        self.use_fpu_reduction = use_fpu_reduction
        self.fpu_reduction = fpu_reduction
        self.root_fpu_reduction = root_fpu_reduction
        self.use_exact_solver = use_exact_solver
        self.use_gumbel_root = use_gumbel_root
        # Async-overlap spike (default OFF): keep `async_inflight` NPU batches in flight while the CPU
        # selects / backs up the next one. Uses async_inflight + 1 infer requests and a pending-leaf set
        # so a leaf already in flight is never sent twice (collisions get temporary virtual loss).
        self.use_async_overlap = use_async_overlap
        self.async_inflight = max(1, int(async_inflight))
        self.use_tree_reuse = use_tree_reuse
        self.use_nn_cache = use_nn_cache
        self.use_fast_tree = use_fast_tree
        self.nn_cache = CanonicalNNCache(max_bytes=nn_cache_max_bytes)
        self._saved_root: TreeNode | None = None
        self._seen_npu_keys: set[tuple] = set()
        self._extra_requests: list = []
        # Per-search CPU/NPU time breakdown (seconds and counts), filled by search().
        self.last_profile: dict = {}
        self.last_root_visits: dict[int, int] = {}

        if self.use_fast_tree:
            from neurachess.engine.fast_tree import FastTreeSearchBackend

            self._fast_backend = FastTreeSearchBackend(
                batch_size=self.batch_size,
                c_puct=self.c_puct,
                use_fpu_reduction=self.use_fpu_reduction,
                fpu_reduction=self.fpu_reduction,
                root_fpu_reduction=self.root_fpu_reduction,
                use_exact_solver=self.use_exact_solver,
                use_gumbel_root=self.use_gumbel_root,
                use_async_overlap=self.use_async_overlap,
                async_inflight=self.async_inflight,
                use_tree_reuse=self.use_tree_reuse,
                use_nn_cache=self.use_nn_cache,
                nn_cache_max_bytes=nn_cache_max_bytes,
                encoder_layout=self.encoder_layout,
                t_root=self.t_root,
                use_opening_root_guard=self.use_opening_root_guard,
            )
        else:
            self._fast_backend = None

        # Root model for batch 1 (initial expansion)
        self.root_model = root_model if root_model is not None else NPUModel(batch_size=1)
        # Batched model for batch B (leaf evaluations)
        self.batch_model = batch_model if batch_model is not None else NPUModel(batch_size=self.batch_size)
        if not hasattr(self.batch_model, "infer_request_async") and hasattr(self.batch_model, "compiled_model"):
            try:
                self.batch_model.infer_request_async = self.batch_model.compiled_model.create_infer_request()
            except (RuntimeError, AttributeError):
                logger.debug("Async OpenVINO InferRequest creation skipped; using sync request", exc_info=True)

    def clear_state(self):
        """Resets saved search tree, NN transposition cache, and seen-key tracker between games."""
        self._saved_root = None
        self.nn_cache.clear()
        self._seen_npu_keys.clear()
        self.last_root_visits = {}
        if self._fast_backend is not None:
            self._fast_backend.clear_state()

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
        if self._fast_backend is not None:
            self._fast_backend.reconfigure(
                c_puct=c_puct,
                use_fpu_reduction=use_fpu_reduction,
                fpu_reduction=fpu_reduction,
                root_fpu_reduction=root_fpu_reduction,
                t_root=t_root,
                use_opening_root_guard=use_opening_root_guard,
            )
        self.clear_state()

    @staticmethod
    def _branching_k_for_ply(ply: int, n_legal: int | None = None) -> int:
        # Zero Legal-Move Pruning: 100% of legal moves expanded at root and all internal nodes.
        return n_legal if n_legal is not None else 256

    def _expand_node_from_entry(self, node: TreeNode, entry: dict, top_k: int) -> float:
        c_acts, c_moves, p_sel, leaf_val = CanonicalNNCache.expand_from_entry(entry, top_k=top_k)
        if node.ply == 0 and self.t_root != 1.0 and self.t_root > 0 and len(p_sel) > 0:
            p_sel = np.power(np.clip(p_sel, 1e-12, 1.0), 1.0 / self.t_root)
            s = float(np.sum(p_sel))
            if s > 0:
                p_sel = p_sel / s
        node.nn_entry = entry
        if len(c_acts) > 0:
            for act, m, pr in zip(c_acts, c_moves, p_sel):
                existing = node.children.get(act)
                if existing is not None:
                    existing.prior = float(pr)
                    existing.ply = node.ply + 1
                    existing.parent = node
                else:
                    node.children[act] = TreeNode(
                        board=None,
                        state_tensor=None,
                        ply=node.ply + 1,
                        parent=node,
                        action_from_parent=act,
                        prior=float(pr),
                        move_from_parent=m,
                    )
            node.is_expanded = True
        return leaf_val

    def _find_reused_root(self, board: chess.Board) -> TreeNode | None:
        if self._saved_root is None or self._saved_root.board is None:
            return None
        old_root = self._saved_root
        old_board = old_root.board
        target_key = CanonicalNNCache.canonical_key(board)

        # 1. Exact move_stack prefix continuation (1..4 plies, typically 2 plies: our move + opponent reply)
        n_old = len(old_board.move_stack)
        n_new = len(board.move_stack)
        if n_new > n_old and (n_new - n_old) <= 4 and board.move_stack[:n_old] == old_board.move_stack:
            curr = old_root
            ok = True
            for mv in board.move_stack[n_old:]:
                if not curr.is_instantiated:
                    curr.instantiate()
                if curr.board is None:
                    ok = False
                    break
                act = move_to_action(mv, canonical_flip=(curr.board.turn == chess.BLACK))
                ch = curr.children.get(act)
                if ch is None:
                    ok = False
                    break
                if not ch.is_instantiated:
                    ch.instantiate()
                curr = ch
            if ok and curr.board is not None:
                if (
                    CanonicalNNCache.canonical_key(curr.board) == target_key
                    and curr.board.halfmove_clock == board.halfmove_clock
                ):
                    return curr

        # 2. Fallback depth-1 / depth-2 search for callers that construct a fresh chess.Board(fen)
        for ch1 in old_root.children.values():
            if ch1.move_from_parent is None:
                continue
            b1 = old_board.copy(stack=False)
            b1.push(ch1.move_from_parent)
            if b1.turn == board.turn and CanonicalNNCache.canonical_key(b1) == target_key and b1.halfmove_clock == board.halfmove_clock:
                if not ch1.is_instantiated:
                    ch1.instantiate()
                return ch1
            if ch1.is_expanded:
                for ch2 in ch1.children.values():
                    if ch2.move_from_parent is None:
                        continue
                    b2 = b1.copy(stack=False)
                    b2.push(ch2.move_from_parent)
                    if (
                        b2.turn == board.turn
                        and CanonicalNNCache.canonical_key(b2) == target_key
                        and b2.halfmove_clock == board.halfmove_clock
                    ):
                        if not ch1.is_instantiated:
                            ch1.instantiate()
                        if not ch2.is_instantiated:
                            ch2.instantiate()
                        return ch2
        return None

    def _prepare_reused_root(
        self,
        new_root: TreeNode,
        board: chess.Board,
        board_tensor: np.ndarray,
        max_depth: int,
        prof: dict,
    ) -> TreeNode:
        stack_matches = (
            new_root.board is not None
            and len(new_root.board.move_stack) == len(board.move_stack)
            and new_root.board.move_stack == board.move_stack
        )
        new_root.parent = None
        new_root.action_from_parent = None
        new_root.move_from_parent = None
        new_root.board = board.copy()
        new_root.state_tensor = board_tensor
        new_root.ply = 0
        new_root.is_instantiated = True
        new_root.virtual_loss = 0
        new_root.is_terminal = False
        new_root.terminal_value = 0.0
        new_root.path_dep_draw = False
        new_root.is_solved = False
        new_root.solved_value = 0.0
        new_root.mate_plies = 0

        # Pre-order traversal over instantiated nodes to update board stacks, plies,
        # invalidate/recompute path-dependent repetition & 50-move draws, and widen branching_k.
        preorder: list[TreeNode] = []
        stack: list[TreeNode] = [new_root]
        invalidated_count = 0

        while stack:
            node = stack.pop()
            preorder.append(node)
            node.virtual_loss = 0

            if node is not new_root:
                assert node.parent is not None and node.parent.board is not None
                if not stack_matches or node.board is None:
                    b = node.parent.board.copy()
                    b.push(node.move_from_parent)
                    node.board = b
                    if node.state_tensor is not None:
                        node.state_tensor[17, :, :] = min(1.0, b.halfmove_clock / 100.0)
                    is_50m = (b.halfmove_clock >= 100)
                    is_insuff = (not is_50m) and b.is_insufficient_material()
                    is_rep = (not is_50m) and (not is_insuff) and b.is_repetition(2)
                    new_path_dep = is_50m or is_rep
                else:
                    b = node.board
                    new_path_dep = False

                if node.path_dep_draw or new_path_dep:
                    # Node's value or terminal/draw state depended on repetition or 50-move clock:
                    # invalidate its stored evaluations/subtree and recompute terminal state on the new path.
                    invalidated_count += 1
                    node.self_visits = 0
                    node.self_value_sum = 0.0
                    node.children.clear()
                    node.is_expanded = False
                    node._evaluate_terminal_flags(b)
                elif not node.is_terminal:
                    node.is_solved = False
                    node.solved_value = 0.0
                    node.mate_plies = 0
                    # If an unexpanded leaf was previously capped at old max_depth, reset self_visits so it can expand
                    if not node.is_expanded and node.self_visits > 0 and node.ply < max_depth:
                        node.self_visits = 0
                        node.self_value_sum = 0.0

            if not node.is_terminal and node.is_expanded and node.nn_entry is not None and node.ply < max_depth:
                n_leg = len(node.nn_entry["unsorted_acts"])
                target_k = self._branching_k_for_ply(node.ply, n_legal=n_leg)
                if len(node.children) < min(target_k, n_leg):
                    self._expand_node_from_entry(node, node.nn_entry, top_k=target_k)

            for ch in node.children.values():
                ch.ply = node.ply + 1
                ch.parent = node
                ch.virtual_loss = 0
                if ch.is_instantiated:
                    stack.append(ch)

        # Post-order (bottom-up) recomputation of visits, total_value, and exact-solver status
        for node in reversed(preorder):
            node.visits = node.self_visits + sum(c.visits for c in node.children.values())
            node.total_value = node.self_value_sum + sum(-c.total_value for c in node.children.values())
            if self.use_exact_solver and not node.is_terminal and node.is_expanded and node.children:
                self._update_solver_along_path([node])

        prof["reused_subtree_nodes"] = len(preorder)
        prof["invalidated_path_dep_nodes"] = invalidated_count
        return new_root

    def _compute_fpu(self, node: TreeNode) -> float:
        if not self.use_fpu_reduction or node.visits == 0:
            return 0.0
        visited_mass = 0.0
        for child in node.children.values():
            if child.visits + child.virtual_loss > 0:
                visited_mass += child.prior
        c_fpu = self.root_fpu_reduction if node.ply == 0 else self.fpu_reduction
        return max(-1.0, min(1.0, node.q_value - c_fpu * math.sqrt(visited_mass)))

    def _update_solver_along_path(self, path: list[TreeNode]):
        if not self.use_exact_solver:
            return
        for node in reversed(path):
            if node.is_terminal or not node.is_expanded or not node.children:
                continue
            min_win_plies = None
            max_loss_plies = 0
            all_losing_children = True
            for ch in node.children.values():
                # A child that is a path-dependent repetition/50-move draw (is_terminal with terminal_value==0)
                # is NEVER a proven win or proven loss.
                if ch.is_solved and ch.solved_value <= -1.0:
                    cand_plies = 1 + ch.mate_plies
                    if min_win_plies is None or cand_plies < min_win_plies:
                        min_win_plies = cand_plies
                if not (ch.is_solved and ch.solved_value >= 1.0):
                    all_losing_children = False
                else:
                    cand_loss = 1 + ch.mate_plies
                    if cand_loss > max_loss_plies:
                        max_loss_plies = cand_loss
            if min_win_plies is not None:
                node.is_solved = True
                node.solved_value = 1.0
                if node.mate_plies == 0 or min_win_plies < node.mate_plies:
                    node.mate_plies = min_win_plies
            elif not node.is_solved and all_losing_children and node.board is not None:
                # Only prove a defender node lost (-1.0) if ALL legal moves in node.board
                # have been expanded and every single legal move is proven to lose (+1.0).
                if node.num_legal_moves is None:
                    node.num_legal_moves = node.board.legal_moves.count()
                if len(node.children) >= node.num_legal_moves:
                    node.is_solved = True
                    node.solved_value = -1.0
                    node.mate_plies = max_loss_plies
                else:
                    # Expand remaining unpruned legal moves so defender can test escapes
                    is_blk = (node.board.turn == chess.BLACK)
                    for m in node.board.legal_moves:
                        act = uci_to_action(m.uci(), canonical_flip=is_blk)
                        if act not in node.children:
                            node.children[act] = TreeNode(
                                board=None,
                                state_tensor=None,
                                ply=node.ply + 1,
                                parent=node,
                                action_from_parent=act,
                                prior=1e-4,
                                move_from_parent=m,
                            )

    def _collect_batch(
        self,
        root: TreeNode,
        pending: set | None = None,
        stats: dict | None = None,
        max_depth: int = 14,
        t_deadline: float | None = None,
        pending_keys: set | None = None,
        max_cache_expansions: int | None = None,
    ) -> tuple[list[TreeNode], list[list[TreeNode]], list[list[TreeNode]]]:
        """
        Traverses tree from root using PUCT + virtual loss to collect up to self.batch_size unexpanded leaves.
        If `self.use_nn_cache` is True, cache hits are expanded and backed up immediately in-place so the
        NPU batch slots are reserved for genuine cache misses.
        """
        leaves_to_eval: list[TreeNode] = []
        paths_to_leaves: list[list[TreeNode]] = []
        vl_applied_paths: list[list[TreeNode]] = []
        collision_paths: list[list[TreeNode]] = []
        cache_hits_to_process: list[tuple[TreeNode, list[TreeNode], dict]] = []
        batch_keys: set[tuple] = set()

        for _ in range(self.batch_size):
            if self.use_exact_solver and root.is_solved and root.mate_plies <= 3:
                break

            curr = root
            path = [curr]

            while curr.is_expanded and not curr.is_terminal and not (self.use_exact_solver and curr.is_solved):
                sqrt_n = math.sqrt(curr.visits) if curr.visits > 0 else 1.0
                fpu_val = self._compute_fpu(curr)
                best_score = -float("inf")
                best_child = None

                for act, child in curr.children.items():
                    q_from_parent_view = child.q_from_parent_view(virtual_loss_penalty=1.0, fpu_value=fpu_val)
                    eff_visits = child.visits + child.virtual_loss
                    u = self.c_puct * child.prior * (sqrt_n / (1 + eff_visits))
                    score = q_from_parent_view + u

                    if score > best_score:
                        best_score = score
                        best_child = child

                if best_child is None:
                    break

                curr = best_child
                path.append(curr)

            # Lazy instantiation if selected for the first time
            if not curr.is_instantiated:
                curr.instantiate()

            if curr.is_terminal or (self.use_exact_solver and curr.is_solved):
                # Terminal or solved leaf: backpropagate immediately
                val = curr.solved_value if (self.use_exact_solver and curr.is_solved) else curr.terminal_value
                curr.self_visits += 1
                curr.self_value_sum += val
                for node in reversed(path):
                    node.visits += 1
                    node.total_value += val
                    val = -val
                self._update_solver_along_path(path)
                if stats is not None:
                    stats["terminal_backups"] = stats.get("terminal_backups", 0) + 1
                continue

            if pending is not None and id(curr) in pending:
                # Collision with a leaf already in flight on the NPU.
                for node in path[1:]:
                    node.virtual_loss += 1
                collision_paths.append(path)
                if stats is not None:
                    stats["collisions"] = stats.get("collisions", 0) + 1
                continue

            if self.use_nn_cache and not curr.is_expanded and curr.self_visits == 0 and curr.board is not None:
                c_key = CanonicalNNCache.canonical_key(curr.board)
                entry = self.nn_cache.get(c_key)
                if entry is not None:
                    # Genuine transposition or cross-move cache hit on an unexpanded leaf:
                    # apply temporary virtual loss during this batch collection so sibling descents diversify,
                    # then expand and backpropagate at the end of _collect_batch.
                    cache_hits_to_process.append((curr, path, entry))
                    for node in path[1:]:
                        node.virtual_loss += 1
                    collision_paths.append(path)
                    if stats is not None:
                        stats["nn_cache_hits"] = stats.get("nn_cache_hits", 0) + 1
                        stats["cache_expanded_nodes"] = stats.get("cache_expanded_nodes", 0) + 1
                        if curr.ply > stats.get("max_depth_cache", 0):
                            stats["max_depth_cache"] = curr.ply
                    continue
                else:
                    if stats is not None:
                        stats["nn_cache_misses"] = stats.get("nn_cache_misses", 0) + 1
                    if c_key in batch_keys:
                        if stats is not None:
                            stats["dedup_avoided"] = stats.get("dedup_avoided", 0) + 1
                        continue
                    batch_keys.add(c_key)

            if curr not in leaves_to_eval:
                leaves_to_eval.append(curr)
                paths_to_leaves.append(path)

                # Apply virtual loss to children along the path
                for node in path[1:]:
                    node.virtual_loss += 1
                vl_applied_paths.append(path)
                if pending is not None:
                    pending.add(id(curr))

        for path in collision_paths:
            for node in path[1:]:
                node.virtual_loss -= 1

        # Expand and backpropagate all cache hits collected during this batch
        for c_leaf, c_path, c_entry in cache_hits_to_process:
            if c_leaf.ply < max_depth and not c_leaf.is_expanded:
                b_k = self._branching_k_for_ply(c_leaf.ply)
                leaf_val = self._expand_node_from_entry(c_leaf, c_entry, top_k=b_k)
            else:
                leaf_val = c_entry["leaf_val"]
            c_leaf.self_visits += 1
            c_leaf.self_value_sum += leaf_val
            val = leaf_val
            for node in reversed(c_path):
                node.visits += 1
                node.total_value += val
                val = -val
            self._update_solver_along_path(c_path)

        return leaves_to_eval, paths_to_leaves, vl_applied_paths

    def _pack_batch_tensors(self, leaves_to_eval: list[TreeNode]) -> np.ndarray:
        """
        Packs leaf state tensors into a (batch_size, NUM_PLANES, 8, 8) float32 batch for NPU inference.
        """
        n_eval = len(leaves_to_eval)
        batch_tensors = np.zeros((self.batch_size, NUM_PLANES, 8, 8), dtype=np.float32)
        for i in range(n_eval):
            batch_tensors[i] = leaves_to_eval[i].state_tensor

        if 0 < n_eval < self.batch_size:
            for i in range(n_eval, self.batch_size):
                batch_tensors[i] = batch_tensors[0]
        return batch_tensors

    def _process_batch_results(
        self,
        leaves_to_eval: list[TreeNode],
        paths_to_leaves: list[list[TreeNode]],
        vl_applied_paths: list[list[TreeNode]],
        p_logits_batch: np.ndarray,
        val_batch: np.ndarray,
        max_depth: int,
        stats: dict | None = None,
    ):
        """
        Reverts virtual loss, expands leaves lazily with adaptive branching, and backpropagates values.
        """
        # Revert virtual losses
        for path in vl_applied_paths:
            for node in path[1:]:
                node.virtual_loss -= 1

        n_eval = len(leaves_to_eval)
        for i in range(n_eval):
            leaf = leaves_to_eval[i]
            path = paths_to_leaves[i]
            v_out = val_batch[i]
            p_logits = p_logits_batch[i]

            c_key = CanonicalNNCache.canonical_key(leaf.board) if leaf.board is not None else None
            if c_key is not None and stats is not None:
                if c_key in self._seen_npu_keys:
                    stats["wasted_dup_key_evals"] = stats.get("wasted_dup_key_evals", 0) + 1
                else:
                    self._seen_npu_keys.add(c_key)
                    stats["novel_npu_evals"] = stats.get("novel_npu_evals", 0) + 1

            if self.use_nn_cache or self.use_tree_reuse:
                entry = CanonicalNNCache.build_entry(leaf.board, p_logits, v_out)
                if self.use_nn_cache and c_key is not None:
                    self.nn_cache.put(c_key, entry)
                leaf_val = entry["leaf_val"]
                if leaf.ply < max_depth and not leaf.is_expanded:
                    branching_k = self._branching_k_for_ply(leaf.ply)
                    self._expand_node_from_entry(leaf, entry, top_k=branching_k)
            else:
                # Leaf value from leaf's perspective
                if v_out.ndim == 1 and len(v_out) == 3:
                    leaf_val = float(v_out[0] - v_out[2])
                else:
                    leaf_val = float(v_out[0])

                # Expand leaf if depth limit not reached (100% legal moves expanded, zero pruning)
                if leaf.ply < max_depth and not leaf.is_expanded:
                    branching_k = self._branching_k_for_ply(leaf.ply)
                    c_acts, c_moves, _ = select_top_candidates(leaf.board, p_logits, top_k=branching_k)
                    if len(c_acts) > 0:
                        l_sel = p_logits[c_acts]
                        e_l = np.exp(l_sel - np.max(l_sel))
                        p_sel = e_l / np.sum(e_l)

                        for act, m, pr in zip(c_acts, c_moves, p_sel):
                            c_node = TreeNode(
                                board=None,
                                state_tensor=None,
                                ply=leaf.ply + 1,
                                parent=leaf,
                                action_from_parent=act,
                                prior=float(pr),
                                move_from_parent=m,
                            )
                            leaf.children[act] = c_node
                        leaf.is_expanded = True

            # Negamax Backpropagation
            leaf.self_visits += 1
            leaf.self_value_sum += leaf_val
            val = leaf_val
            for node in reversed(path):
                node.visits += 1
                node.total_value += val
                val = -val
            self._update_solver_along_path(path)

    def search(
        self,
        board: chess.Board,
        board_tensor: np.ndarray | None = None,
        movetime_ms: int = 600,
        max_depth: int = 14,
        max_nodes: int | None = None,
    ) -> tuple[int, str, float, int, list[str], int]:
        """
        Executes Vectorized Batched PUCT search with Lazy Child Evaluation on Intel AI Boost NPU.

        Returns:
            best_action: int
            best_uci: str
            best_eval: float (side-to-move view)
            nodes_evaluated: int
            pv: list[str]
            max_depth_reached: int
        """
        if self._fast_backend is not None:
            best_act, best_uci, best_eval, nodes_evaluated, pv, max_depth_reached, prof = self._fast_backend.search(
                self.root_model,
                self.batch_model,
                board,
                board_tensor=board_tensor,
                movetime_ms=movetime_ms,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
            self.last_profile = prof
            self.last_root_visits = self._fast_backend.last_root_visits
            return best_act, best_uci, best_eval, nodes_evaluated, pv, max_depth_reached

        t_start = time.perf_counter()
        t_deadline = t_start + (movetime_ms / 1000.0)
        is_black = (board.turn == chess.BLACK)
        nodes_evaluated = 0
        max_depth_reached = 0

        # 1. Root terminal check
        root_legals = list(board.legal_moves)
        if len(root_legals) == 0:
            self.last_root_visits = {}
            val = -1.0 if board.is_check() else 0.0
            return -1, "0000", val, 0, [], 0

        if len(root_legals) == 1:
            m = root_legals[0]
            act = uci_to_action(m.uci(), canonical_flip=is_black)
            self.last_root_visits = {act: 1}
            return act, m.uci(), 0.0, 1, [m.uci()], 1

        # Check immediate root mate-in-1 (guaranteed instant tactical victory)
        for m in root_legals:
            board.push(m)
            if board.is_checkmate():
                board.pop()
                act = uci_to_action(m.uci(), canonical_flip=is_black)
                self.last_root_visits = {act: 1}
                return act, m.uci(), 1.0, len(root_legals), [m.uci()], 1
            board.pop()

        if board_tensor is None:
            board_tensor = board_to_tensor(board, layout=self.encoder_layout)

        has_async = hasattr(self.batch_model, "infer_request_async") and hasattr(
            self.batch_model.infer_request, "start_async"
        )
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
            "mode": "async_overlap" if (has_async and self.use_async_overlap) else ("double_buffer" if has_async else "sync"),
            "inflight_during_cpu_work": self.async_inflight if (has_async and self.use_async_overlap) else (1 if has_async else 0),
        }

        # 2. Root setup (Tree Reuse if enabled and child/grandchild found, otherwise fresh root)
        root: TreeNode | None = None
        if self.use_tree_reuse:
            reused_cand = self._find_reused_root(board)
            if reused_cand is not None:
                root = self._prepare_reused_root(reused_cand, board, board_tensor, max_depth, prof)
                prof["tree_reused"] = True
                prof["reused_root_visits"] = root.visits

        if root is None:
            root = TreeNode(board.copy(), board_tensor, ply=0, encoder_layout=self.encoder_layout)

        # Ensure root is expanded with all legal moves
        if not root.is_expanded or len(root.children) < len(root_legals):
            root_key = CanonicalNNCache.canonical_key(root.board)
            root_entry = self.nn_cache.get(root_key) if self.use_nn_cache else None
            if root_entry is not None:
                prof["nn_cache_hits"] += 1
                prof["cache_expanded_nodes"] += 1
                root_v = self._expand_node_from_entry(root, root_entry, top_k=len(root_legals))
                nodes_evaluated += 1
            else:
                if self.use_nn_cache:
                    prof["nn_cache_misses"] += 1
                p_logits_root, v_root_out = self.root_model.infer(board_tensor)
                nodes_evaluated += 1
                prof["root_npu_evals"] += 1
                if root_key in self._seen_npu_keys:
                    prof["wasted_dup_key_evals"] += 1
                else:
                    self._seen_npu_keys.add(root_key)
                    prof["novel_npu_evals"] += 1

                if self.use_nn_cache or self.use_tree_reuse:
                    root_entry = CanonicalNNCache.build_entry(root.board, p_logits_root[0], v_root_out[0])
                    if self.use_nn_cache:
                        self.nn_cache.put(root_key, root_entry)
                    root_v = self._expand_node_from_entry(root, root_entry, top_k=len(root_legals))
                else:
                    root_v = (
                        float(v_root_out[0][0] - v_root_out[0][2])
                        if (v_root_out.ndim == 2 and v_root_out.shape[1] == 3)
                        else float(v_root_out[0][0])
                    )
                    cand_actions, cand_moves, _ = select_top_candidates(
                        root.board, p_logits_root[0], top_k=len(root_legals)
                    )
                    logits_selected = p_logits_root[0][cand_actions]
                    if self.t_root != 1.0 and self.t_root > 0:
                        logits_selected = logits_selected / self.t_root
                    exp_logits = np.exp(logits_selected - np.max(logits_selected))
                    priors = exp_logits / np.sum(exp_logits)
                    for act, m, prior in zip(cand_actions, cand_moves, priors):
                        root.children[act] = TreeNode(
                            board=None,
                            state_tensor=None,
                            ply=1,
                            parent=root,
                            action_from_parent=act,
                            prior=float(prior),
                            move_from_parent=m,
                        )
                    root.is_expanded = True

            if self.use_fpu_reduction and root.visits == 0:
                root.self_visits = 1
                root.self_value_sum = root_v
                root.visits = 1
                root.total_value = root_v

        # 3. Batched Search Loop (Pipelined Double-Buffering)
        t_loop0 = _pc()

        if has_async and self.use_async_overlap:
            from collections import deque

            depth = self.async_inflight + 1
            reqs = [self.batch_model.infer_request, self.batch_model.infer_request_async]
            while len(reqs) + len(self._extra_requests) < depth:
                self._extra_requests.append(self.batch_model.compiled_model.create_infer_request())
            reqs = (reqs + self._extra_requests)[:depth]
            free = list(range(depth))
            inflight: deque = deque()
            pending: set = set()
            pending_keys: set | None = set() if self.use_nn_cache else None

            while True:
                # Top up: keep `depth` batches in flight (so `async_inflight` remain on the NPU while we work).
                while free:
                    time_up = (_pc() >= t_deadline) or (
                        max_nodes is not None and nodes_evaluated + len(inflight) * self.batch_size >= max_nodes
                    )
                    if time_up:
                        break
                    t0 = _pc()
                    leaves_n, paths_n, vl_n = self._collect_batch(
                        root,
                        pending=pending,
                        stats=prof,
                        max_depth=max_depth,
                        t_deadline=t_deadline,
                        pending_keys=pending_keys,
                    )
                    t1 = _pc()
                    prof["select_s"] += t1 - t0
                    if len(leaves_n) == 0:
                        break
                    bt = self._pack_batch_tensors(leaves_n)
                    t2 = _pc()
                    prof["pack_s"] += t2 - t1
                    r = free.pop()
                    reqs[r].start_async({"board_state": bt})
                    prof["dispatch_s"] += _pc() - t2
                    inflight.append((r, leaves_n, paths_n, vl_n))
                if not inflight:
                    break
                r, curr_leaves, curr_paths, curr_vl = inflight.popleft()
                t0 = _pc()
                reqs[r].wait()
                t1 = _pc()
                prof["wait_s"] += t1 - t0
                outputs = list(reqs[r].results.values())
                nodes_evaluated += self.batch_size
                prof["batches"] += 1
                prof["slots"] += self.batch_size
                prof["unique_leaves"] += len(curr_leaves)
                for leaf in curr_leaves:
                    pending.discard(id(leaf))
                    if pending_keys is not None and leaf.board is not None:
                        pending_keys.discard(CanonicalNNCache.canonical_key(leaf.board))
                    if leaf.ply > max_depth_reached:
                        max_depth_reached = leaf.ply
                self._process_batch_results(
                    curr_leaves, curr_paths, curr_vl, outputs[0], outputs[1], max_depth, stats=prof
                )
                prof["backup_s"] += _pc() - t1
                free.append(r)

        elif has_async:
            requests = [self.batch_model.infer_request, self.batch_model.infer_request_async]
            in_flight: dict[int, tuple[list[TreeNode], list[list[TreeNode]], list[list[TreeNode]]]] = {}

            # Prime pipeline with initial batch
            while True:
                t0 = _pc()
                leaves_0, paths_0, vl_0 = self._collect_batch(
                    root,
                    stats=prof,
                    max_depth=max_depth,
                    t_deadline=t_deadline,
                )
                prof["select_s"] += _pc() - t0
                if len(leaves_0) > 0:
                    t1 = _pc()
                    batch_tensors_0 = self._pack_batch_tensors(leaves_0)
                    t2 = _pc()
                    prof["pack_s"] += t2 - t1
                    requests[0].start_async({"board_state": batch_tensors_0})
                    prof["dispatch_s"] += _pc() - t2
                    in_flight[0] = (leaves_0, paths_0, vl_0)
                    break
                if (_pc() >= t_deadline) or (max_nodes is not None and nodes_evaluated >= max_nodes) or prof["cache_expanded_nodes"] == 0:
                    break

            buf_idx = 0

            while len(in_flight) > 0:
                time_up = (time.perf_counter() >= t_deadline) or (
                    max_nodes is not None and nodes_evaluated >= max_nodes
                )
                next_buf = 1 - buf_idx

                # Overlap: dispatch next batch while current batch is computing on NPU
                if not time_up and next_buf not in in_flight:
                    t0 = _pc()
                    leaves_next, paths_next, vl_next = self._collect_batch(
                        root,
                        stats=prof,
                        max_depth=max_depth,
                        t_deadline=t_deadline,
                    )
                    t1 = _pc()
                    prof["select_s"] += t1 - t0
                    if len(leaves_next) > 0:
                        batch_tensors_next = self._pack_batch_tensors(leaves_next)
                        t2 = _pc()
                        prof["pack_s"] += t2 - t1
                        requests[next_buf].start_async({"board_state": batch_tensors_next})
                        prof["dispatch_s"] += _pc() - t2
                        in_flight[next_buf] = (leaves_next, paths_next, vl_next)

                # Wait for current buffer's NPU computation
                curr_req = requests[buf_idx]
                t0 = _pc()
                curr_req.wait()
                t1 = _pc()
                prof["wait_s"] += t1 - t0
                outputs = list(curr_req.results.values())
                p_logits_batch = outputs[0]
                val_batch = outputs[1]
                nodes_evaluated += self.batch_size

                curr_leaves, curr_paths, curr_vl = in_flight.pop(buf_idx)
                prof["batches"] += 1
                prof["slots"] += self.batch_size
                prof["unique_leaves"] += len(curr_leaves)
                for leaf in curr_leaves:
                    if leaf.ply > max_depth_reached:
                        max_depth_reached = leaf.ply

                # Unpack, expand, and backpropagate
                self._process_batch_results(
                    curr_leaves,
                    curr_paths,
                    curr_vl,
                    p_logits_batch,
                    val_batch,
                    max_depth,
                    stats=prof,
                )
                prof["backup_s"] += _pc() - t1

                if len(in_flight) == 0 and not (
                    (time.perf_counter() >= t_deadline)
                    or (max_nodes is not None and nodes_evaluated >= max_nodes)
                ):
                    t0 = _pc()
                    leaves_re, paths_re, vl_re = self._collect_batch(
                        root,
                        stats=prof,
                        max_depth=max_depth,
                        t_deadline=t_deadline,
                    )
                    prof["select_s"] += _pc() - t0
                    if len(leaves_re) > 0:
                        batch_tensors_re = self._pack_batch_tensors(leaves_re)
                        t2 = _pc()
                        prof["pack_s"] += t2 - t1
                        requests[next_buf].start_async({"board_state": batch_tensors_re})
                        prof["dispatch_s"] += _pc() - t2
                        in_flight[next_buf] = (leaves_re, paths_re, vl_re)

                buf_idx = next_buf
        else:
            # Fallback Synchronous Batched PUCT Search
            while time.perf_counter() < t_deadline:
                if max_nodes is not None and nodes_evaluated >= max_nodes:
                    break

                t0 = _pc()
                leaves_to_eval, paths_to_leaves, vl_applied_paths = self._collect_batch(
                    root,
                    stats=prof,
                    max_depth=max_depth,
                    t_deadline=t_deadline,
                )
                t1 = _pc()
                prof["select_s"] += t1 - t0
                if len(leaves_to_eval) == 0:
                    break

                for leaf in leaves_to_eval:
                    if leaf.ply > max_depth_reached:
                        max_depth_reached = leaf.ply

                batch_tensors = self._pack_batch_tensors(leaves_to_eval)
                t2 = _pc()
                prof["pack_s"] += t2 - t1
                p_logits_batch, val_batch = self.batch_model.infer(batch_tensors)
                t3 = _pc()
                prof["wait_s"] += t3 - t2
                nodes_evaluated += self.batch_size
                prof["batches"] += 1
                prof["slots"] += self.batch_size
                prof["unique_leaves"] += len(leaves_to_eval)

                self._process_batch_results(
                    leaves_to_eval,
                    paths_to_leaves,
                    vl_applied_paths,
                    p_logits_batch,
                    val_batch,
                    max_depth,
                    stats=prof,
                )  # Negamax flip
                prof["backup_s"] += _pc() - t3

        if prof["max_depth_cache"] > max_depth_reached:
            max_depth_reached = prof["max_depth_cache"]

        nodes_evaluated += prof["cache_expanded_nodes"]
        prof["pre_loop_s"] = t_loop0 - t_start
        prof["loop_s"] = _pc() - t_loop0
        prof["total_s"] = _pc() - t_start
        prof["final_root_visits"] = root.visits
        prof["added_root_visits"] = max(0, root.visits - prof["reused_root_visits"])
        prof["npu_total_slots"] = prof["root_npu_evals"] + prof["slots"]
        prof["unique_npu_evals"] = prof["root_npu_evals"] + prof["unique_leaves"]
        prof["wasted_pad_slots"] = prof["slots"] - prof["unique_leaves"]
        prof["total_wasted_npu_slots"] = prof["npu_total_slots"] - prof["novel_npu_evals"]
        prof["nn_cache_used_bytes"] = self.nn_cache.used_bytes
        prof["nn_cache_entries"] = len(self.nn_cache)
        self.last_profile = prof
        self.last_root_visits = {act: ch.visits for act, ch in root.children.items()}

        if self.use_tree_reuse:
            self._saved_root = root
        else:
            self._saved_root = None

        # 4. Extract Principal Variation & Best Move from Root
        best_act = None
        if self.use_exact_solver:
            # First check if any root child is a proven checkmate win; always pick the SHORTEST proven mate!
            best_mate_plies = None
            best_mate_prior = -1.0
            for act, child in root.children.items():
                if child.is_solved and child.solved_value <= -1.0:
                    if (
                        best_mate_plies is None
                        or child.mate_plies < best_mate_plies
                        or (child.mate_plies == best_mate_plies and child.prior > best_mate_prior)
                    ):
                        best_mate_plies = child.mate_plies
                        best_mate_prior = child.prior
                        best_act = act

        if best_act is None and self.use_gumbel_root and len(root.children) > 0:
            max_v = max(c.visits for c in root.children.values())
            if max_v > 0:
                # Gumbel AlphaZero Policy Improvement (Danihelka et al., ICLR 2022):
                # Among moves with meaningful visit share (>= 25% of max_visits), rank by
                # log(prior) + (c_visit + max_v) * c_scale * completed_Q
                c_visit = 50.0
                c_scale = 1.0
                best_gumbel_score = -float("inf")
                visit_floor = max(1, int(0.25 * max_v))
                for act, child in root.children.items():
                    if child.visits < visit_floor:
                        continue
                    if child.is_solved and child.solved_value >= 1.0:
                        continue  # Avoid proven losses
                    child_q = -child.q_value
                    g_score = math.log(max(1e-8, child.prior)) + (c_visit + max_v) * c_scale * child_q
                    if g_score > best_gumbel_score:
                        best_gumbel_score = g_score
                        best_act = act

        if best_act is None:
            if self.use_opening_root_guard and board.fullmove_number <= 12 and len(root.children) > 0:
                candidates = [
                    (
                        act,
                        float(child.prior),
                        int(child.visits),
                        float(-child.q_value),
                        bool(child.is_solved and child.solved_value <= -1.0),
                        bool(child.is_solved and child.solved_value >= 1.0),
                    )
                    for act, child in root.children.items()
                ]
                best_act = select_opening_guarded_root_act(candidates)
            else:
                best_visits = -1
                best_q = -float("inf")
                for act, child in root.children.items():
                    child_q = -child.q_value  # Parent (root) perspective
                    if child.visits > best_visits or (child.visits == best_visits and child_q > best_q):
                        best_visits = child.visits
                        best_q = child_q
                        best_act = act

        if best_act is None:
            best_act = list(root.children.keys())[0]

        best_child = root.children[best_act]
        best_eval = -best_child.q_value  # From root view
        if best_child.move_from_parent is not None:
            best_uci = best_child.move_from_parent.uci()
        else:
            best_uci = action_to_uci(best_act, canonical_flip=is_black)

        # Extract PV trajectory
        pv = [best_uci]
        curr = best_child
        while curr.is_expanded and len(curr.children) > 0:
            next_act = max(curr.children.keys(), key=lambda a: curr.children[a].visits)
            next_child = curr.children[next_act]
            if next_child.move_from_parent is not None:
                pv_uci = next_child.move_from_parent.uci()
            else:
                pv_uci = action_to_uci(next_act, canonical_flip=(curr.board.turn == chess.BLACK if curr.board else not is_black))
            pv.append(pv_uci)
            curr = next_child
            if len(pv) >= 8:
                break

        return best_act, best_uci, best_eval, nodes_evaluated, pv, max_depth_reached
