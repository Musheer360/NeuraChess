"""
search/ponder_manager.py
========================
NeuraChess Pondering & Permanent Brain Manager (P1 Implementation).

Implements:
1. Internal Permanent Brain: After our move is selected, keeps PUCT search running
   from the position after our move in a background worker thread.
2. Strict Standard UCI:
   - "bestmove X ponder Y" (Y = most visited reply to X)
   - "go ponder"
   - "ponderhit"
   - "stop"
3. Clean Stop & Safe Re-rooting:
   - Awaits in-flight NPU/CPU batch and finishes backprop before re-rooting
   - Re-rooting obeys repetition and 50-move invalidation rules
   - Zero virtual loss leakage between moves
4. Telemetry Logging (P2):
   - ponder duration (ms)
   - ponder nodes
   - ponder hit (actual == expected)
   - ponder in top-3
   - visits retained
   - nodes at move start (with vs without ponder)
"""

import time
import threading
import chess
import numpy as np

try:
    from .npu_batched_puct import NPUBatchedPUCTSearch, TreeNode, CanonicalNNCache
    from ..core.constants import uci_to_action, action_to_uci, move_to_action
    from ..core.board_tensor import board_to_tensor
except (ImportError, ValueError):
    from search.npu_batched_puct import NPUBatchedPUCTSearch, TreeNode, CanonicalNNCache
    from core.constants import uci_to_action, action_to_uci, move_to_action
    from core.board_tensor import board_to_tensor


class PonderManager:
    def __init__(self, search_engine: NPUBatchedPUCTSearch, use_ponder: bool = False):
        self.search_engine = search_engine
        self.use_ponder = use_ponder

        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._ponderhit_event = threading.Event()
        self._is_pondering = False
        self._is_active_search = False

        self._ponder_board: chess.Board | None = None
        self._expected_move: chess.Move | None = None
        self._ponder_start_time = 0.0
        self._ponder_nodes = 0
        self._last_result: tuple | None = None
        self._uci_output_callback = None

        self.last_telemetry: dict = {
            "ponder_duration_ms": 0.0,
            "ponder_nodes": 0,
            "ponder_hit": False,
            "ponder_in_top3": False,
            "visits_retained": 0,
            "nodes_at_move_start_with_ponder": 0,
            "nodes_at_move_start_without_ponder": 0,
        }

    def start_internal_ponder(self, board_after_our_move: chess.Board, expected_reply: chess.Move | None = None):
        """Starts background thinking while the opponent is thinking (Internal Permanent Brain)."""
        if not self.use_ponder:
            return

        self.stop()  # Ensure any prior worker is stopped cleanly

        with self._lock:
            self._stop_event.clear()
            self._ponderhit_event.clear()
            self._is_pondering = True
            self._is_active_search = False
            self._ponder_board = board_after_our_move.copy()
            self._expected_move = expected_reply
            self._ponder_start_time = time.perf_counter()
            self._ponder_nodes = 0

            self._worker_thread = threading.Thread(
                target=self._ponder_worker_loop,
                args=(self._ponder_board, None, 0),
                daemon=True,
                name="NeuraChess-PonderWorker"
            )
            self._worker_thread.start()

    def start_external_uci_ponder(
        self,
        board: chess.Board,
        movetime_ms: int = 600,
        uci_callback=None
    ):
        """Handles standard UCI 'go ponder' command."""
        self.stop()

        with self._lock:
            self._stop_event.clear()
            self._ponderhit_event.clear()
            self._is_pondering = True
            self._is_active_search = False
            self._ponder_board = board.copy()
            self._ponder_start_time = time.perf_counter()
            self._ponder_nodes = 0
            self._uci_output_callback = uci_callback

            self._worker_thread = threading.Thread(
                target=self._ponder_worker_loop,
                args=(self._ponder_board, movetime_ms, 0),
                daemon=True,
                name="NeuraChess-UCIPonderWorker"
            )
            self._worker_thread.start()

    def ponderhit(self, movetime_ms: int = 600):
        """Called when GUI sends 'ponderhit' (the opponent played the predicted move)."""
        with self._lock:
            if not self._is_pondering:
                return
            self._is_pondering = False
            self._is_active_search = True
            self._ponderhit_event.set()

    def stop(self) -> float:
        """
        Signals stop and awaits completion of the in-flight batch.
        Returns the stop latency in milliseconds.
        """
        t0 = time.perf_counter()
        self._stop_event.set()

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        with self._lock:
            self._is_pondering = False
            self._is_active_search = False
            self._worker_thread = None

        return latency_ms

    def on_opponent_move(self, actual_move: chess.Move) -> dict:
        """
        Called when the opponent's move actually arrives on the board.
        Re-roots the search tree to the actual move's node if available.
        Returns P2 telemetry.
        """
        duration_ms = (time.perf_counter() - self._ponder_start_time) * 1000.0 if self._ponder_start_time > 0 else 0.0
        self.stop()

        ponder_hit = False
        ponder_top3 = False
        visits_retained = 0

        fb = getattr(self.search_engine, "_fast_backend", None)
        if fb is not None and fb._has_saved_tree:
            if self._expected_move is not None:
                ponder_hit = (actual_move == self._expected_move)

            root_ch = fb._get_children_snapshot(0)
            if root_ch:
                from engine.fast_tree import pack_move

                actual_mp = pack_move(actual_move)
                sorted_ch = sorted(root_ch, key=lambda c: c["visits"], reverse=True)
                top3_mps = [c["mp"] for c in sorted_ch[:3]]
                if actual_mp in top3_mps:
                    ponder_top3 = True
                for c in root_ch:
                    if c["mp"] == actual_mp:
                        visits_retained = int(c["visits"])
                        break
        else:
            saved_root = self.search_engine._saved_root
            if saved_root is not None and saved_root.board is not None:
                # Check if actual_move matches expected_move or is among children
                is_blk = (saved_root.board.turn == chess.BLACK)
                act = move_to_action(actual_move, canonical_flip=is_blk)
                if self._expected_move is not None:
                    ponder_hit = (actual_move == self._expected_move)

                # Check top 3 visits
                sorted_children = sorted(saved_root.children.values(), key=lambda c: c.visits, reverse=True)
                top3_moves = [c.move_from_parent for c in sorted_children[:3] if c.move_from_parent is not None]
                if actual_move in top3_moves:
                    ponder_top3 = True

                target_child = saved_root.children.get(act)
                if target_child is not None:
                    visits_retained = target_child.visits

        self.last_telemetry = {
            "ponder_duration_ms": round(duration_ms, 2),
            "ponder_nodes": self._ponder_nodes,
            "ponder_hit": ponder_hit,
            "ponder_in_top3": ponder_top3,
            "visits_retained": visits_retained,
            "nodes_at_move_start_with_ponder": visits_retained,
            "nodes_at_move_start_without_ponder": 0,
        }
        return self.last_telemetry

    def _ponder_worker_loop(self, board: chess.Board, default_movetime_ms: int | None, max_nodes: int = 0):
        """Worker thread executing batched PUCT search while checking stop/ponderhit events."""
        fb = getattr(self.search_engine, "_fast_backend", None)
        if fb is not None and (not board.chess960) and self.search_engine.encoder_layout == "gpu_unpack":
            res = fb.run_ponder_session(
                root_model=self.search_engine.root_model,
                batch_model=self.search_engine.batch_model,
                board=board,
                stop_event=self._stop_event,
                ponderhit_event=self._ponderhit_event,
                default_movetime_ms=default_movetime_ms,
                max_depth=14,
            )
            self._ponder_nodes = int(res["ponder_nodes"])
            if self._ponderhit_event.is_set() and self._uci_output_callback is not None and res["best_uci"] is not None:
                self._uci_output_callback(
                    res["best_uci"],
                    res["ponder_uci"],
                    int(res["root_visits"]),
                    float(res["best_eval"]),
                )
            return

        tensor = board_to_tensor(board, layout=self.search_engine.encoder_layout)
        is_black = (board.turn == chess.BLACK)

        # 1. Setup or reuse root
        root: TreeNode | None = None
        if self.search_engine.use_tree_reuse:
            cand = self.search_engine._find_reused_root(board)
            if cand is not None:
                prof = {}
                root = self.search_engine._prepare_reused_root(cand, board, tensor, 14, prof)

        if root is None:
            root = TreeNode(board.copy(), tensor, ply=0, encoder_layout=self.search_engine.encoder_layout)

        root_legals = list(board.legal_moves)
        if len(root_legals) == 0:
            return

        # Ensure root expanded
        if not root.is_expanded:
            p_logits_root, v_root_out = self.search_engine.root_model.infer(tensor)
            self._ponder_nodes += 1
            root_entry = CanonicalNNCache.build_entry(root.board, p_logits_root[0], v_root_out[0])
            self.search_engine._expand_node_from_entry(root, root_entry, top_k=len(root_legals))

        self.search_engine._saved_root = root

        # Pondering loop: runs until stop_event or ponderhit
        search_deadline = float("inf")

        while not self._stop_event.is_set():
            if self._ponderhit_event.is_set():
                if search_deadline == float("inf"):
                    movetime = default_movetime_ms if default_movetime_ms and default_movetime_ms > 0 else 600
                    search_deadline = time.perf_counter() + (movetime / 1000.0)
                if time.perf_counter() >= search_deadline:
                    break

            # Execute exactly one batch using search engine's _collect_batch
            leaves_to_eval, paths_to_leaves, vl_applied_paths = self.search_engine._collect_batch(
                root,
                stats={},
                max_depth=14,
                t_deadline=search_deadline if search_deadline != float("inf") else time.perf_counter() + 10.0,
            )

            if len(leaves_to_eval) > 0:
                batch_tensors = self.search_engine._pack_batch_tensors(leaves_to_eval)
                p_logits_batch, val_batch = self.search_engine.batch_model.infer(batch_tensors)
                self._ponder_nodes += self.search_engine.batch_size
                self.search_engine._process_batch_results(
                    leaves_to_eval,
                    paths_to_leaves,
                    vl_applied_paths,
                    p_logits_batch,
                    val_batch,
                    14,
                    stats={}
                )

        # On stop/exit: ensure all virtual losses are 0
        def clear_vl(node: TreeNode):
            node.virtual_loss = 0
            for ch in node.children.values():
                if ch.is_expanded or ch.virtual_loss > 0:
                    clear_vl(ch)

        clear_vl(root)
        self.search_engine._saved_root = root

        # If ponderhit triggered and search completed, output bestmove via callback
        if self._ponderhit_event.is_set() and self._uci_output_callback is not None:
            # Extract PV
            best_act = max(root.children.keys(), key=lambda a: root.children[a].visits)
            best_child = root.children[best_act]
            best_uci = best_child.move_from_parent.uci() if best_child.move_from_parent else action_to_uci(best_act, canonical_flip=is_black)
            
            # Predict opponent's reply (ponder move)
            ponder_uci = None
            if best_child.is_expanded and len(best_child.children) > 0:
                p_act = max(best_child.children.keys(), key=lambda a: best_child.children[a].visits)
                p_child = best_child.children[p_act]
                ponder_uci = p_child.move_from_parent.uci() if p_child.move_from_parent else None

            self._uci_output_callback(best_uci, ponder_uci, root.visits, -best_child.q_value)
