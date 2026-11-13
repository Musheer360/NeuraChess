"""
NeuraChess Interactive Terminal Play
Play directly against NeuraChess running on Intel AI Boost NPU.
"""

import sys
import time
from pathlib import Path
import chess

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.board_tensor import board_to_tensor
from search.npu_batched_puct import NPUBatchedPUCTSearch

UNICODE_PIECES = {
    "R": "♖", "N": "♘", "B": "♗", "Q": "♕", "K": "♔", "P": "♙",
    "r": "♜", "n": "♞", "b": "♝", "q": "♛", "k": "♚", "p": "♟",
    ".": "·",
}


def print_board(board: chess.Board):
    print("\n    a b c d e f g h")
    print("  +-----------------+")
    for r in range(7, -1, -1):
        row_str = f"{r + 1} | "
        for f in range(8):
            sq = r * 8 + f
            p = board.piece_at(sq)
            sym = UNICODE_PIECES[p.symbol()] if p else "·"
            row_str += sym + " "
        row_str += f"| {r + 1}"
        print(row_str)
    print("  +-----------------+")
    print("    a b c d e f g h\n")


def main():
    print("=" * 60)
    print("  NEURACHESS: NPU CHESS ENGINE")
    print("  OpenVINO Runtime / Intel AI Boost NPU")
    print("=" * 60)

    color_choice = input("Play as White (w) or Black (b)? [default: w]: ").strip().lower()
    user_is_white = color_choice != "b"

    print("\nInitializing NPU Search Engine...")
    engine = NPUBatchedPUCTSearch(batch_size=16)
    board = chess.Board()
    print_board(board)

    while not board.is_game_over():
        user_turn = (board.turn == chess.WHITE and user_is_white) or (board.turn == chess.BLACK and not user_is_white)

        if user_turn:
            while True:
                move_str = input("Your move (e.g. e2e4, quit): ").strip()
                if move_str.lower() in ("quit", "exit"):
                    print("Game aborted.")
                    return
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in board.legal_moves:
                        board.push(move)
                        break
                    print("Illegal move! Legal moves:", [m.uci() for m in board.legal_moves])
                except Exception:
                    print("Invalid move format! Use standard UCI (e.g. e2e4, g1f3).")
            print_board(board)
        else:
            print("NeuraChess calculating on NPU...")
            tensor = board_to_tensor(board, layout=engine.encoder_layout)
            t0 = time.time()
            _, best_uci, score, nodes, pv, depth = engine.search(board, tensor, movetime_ms=600)
            dt = time.time() - t0

            board.push(chess.Move.from_uci(best_uci))
            pv_str = " ".join(pv) if pv else best_uci
            print(
                f"NeuraChess played: {best_uci} (Eval: {score:+.2f}, Nodes: {nodes}, "
                f"Depth: {depth}, Time: {dt*1000:.1f}ms, PV: {pv_str})"
            )
            print_board(board)

    print("=" * 60)
    print("GAME OVER:", board.result())
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        print(f"Checkmate! Winner: {winner}")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    print("=" * 60)


if __name__ == "__main__":
    main()
