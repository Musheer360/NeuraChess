"""
NeuraChess Live Match Server & Web Interface
FastAPI-based server managing real-time matches between NeuraChess (NPU) and Stockfish (CPU).
"""

import os
import sys
import time
import asyncio
import logging
import chess
import subprocess

logger = logging.getLogger(__name__)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from neurachess._paths import find_repo_root

pkg_root = str(find_repo_root())

from neurachess.core.board_tensor import board_to_tensor
from neurachess.search.npu_batched_puct import NPUBatchedPUCTSearch
from neurachess.core.tensor_movegen import action_to_move

app = FastAPI(title="NeuraChess NPU Arena")

# Static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

import shutil

def resolve_stockfish_path() -> str:
    env_p = os.environ.get("STOCKFISH_PATH")
    if env_p and os.path.exists(env_p):
        return env_p
    which_p = shutil.which("stockfish")
    if which_p:
        return which_p
    for cand in (
        os.path.join(pkg_root, "bin", "stockfish", "stockfish-windows-x86-64-universal.exe"),
        os.path.join(pkg_root, "bin", "stockfish", "stockfish"),
        "stockfish",
    ):
        if os.path.exists(cand):
            return cand
    return "stockfish"

STOCKFISH_PATH = resolve_stockfish_path()


class MatchState:
    def __init__(self):
        self.board = chess.Board()
        self.history: list[chess.Board] = []
        self.moves_san: list[str] = []
        self.moves_uci: list[str] = []
        self.last_move: dict | None = None
        self.is_running = False
        self.is_paused = False
        self.game_over = False
        self.result = "*"
        self.termination = ""
        self.npu_eval = 0.0
        self.npu_latency_ms = 0.0
        self.npu_nodes = 0
        self.npu_nps = 0
        self.npu_active = False
        self.cpu_eval = 0.0
        self.cpu_latency_ms = 0.0
        self.cpu_usage_pct = 0.0
        self.ram_usage_pct = 0.0
        self.move_delay = 0.4  # Snappier delay
        self.think_time_ms = 600  # 600ms sustained NPU search per move
        self.stockfish_skill = 3  # Default skill level (0 to 20)
        self.npu_engine: NPUBatchedPUCTSearch | None = None
        self.stockfish_process: subprocess.Popen | None = None

    def reset(self, stockfish_skill: int = 3, move_delay: float = 0.4, think_time_ms: int = 600):
        self.board = chess.Board()
        self.history = []
        self.moves_san = []
        self.moves_uci = []
        self.last_move = None
        self.is_running = False
        self.is_paused = False
        self.game_over = False
        self.result = "*"
        self.termination = ""
        self.npu_eval = 0.0
        self.npu_latency_ms = 0.0
        self.npu_nodes = 0
        self.npu_nps = 0
        self.npu_active = False
        self.cpu_eval = 0.0
        self.cpu_latency_ms = 0.0
        self.stockfish_skill = stockfish_skill
        self.move_delay = move_delay
        self.think_time_ms = think_time_ms

    def get_dict(self):
        import psutil
        try:
            self.cpu_usage_pct = psutil.cpu_percent(interval=None)
            self.ram_usage_pct = psutil.virtual_memory().percent
        except (OSError, RuntimeError):
            logger.debug("psutil CPU/RAM probe unavailable", exc_info=True)

        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "move_number": self.board.fullmove_number,
            "moves_san": self.moves_san,
            "moves_uci": self.moves_uci,
            "last_move": self.last_move,
            "is_running": self.is_running,
            "is_paused": self.is_paused,
            "game_over": self.game_over,
            "result": self.result,
            "termination": self.termination,
            "npu_eval": self.npu_eval,
            "npu_latency_ms": round(self.npu_latency_ms, 1),
            "npu_nodes": self.npu_nodes,
            "npu_nps": self.npu_nps,
            "npu_active": self.npu_active,
            "cpu_eval": self.cpu_eval,
            "cpu_latency_ms": round(self.cpu_latency_ms, 1),
            "cpu_usage_pct": round(self.cpu_usage_pct, 1),
            "ram_usage_pct": round(self.ram_usage_pct, 1),
            "stockfish_skill": self.stockfish_skill,
            "move_delay": self.move_delay,
            "think_time_ms": self.think_time_ms,
        }


match = MatchState()
connected_websockets: list[WebSocket] = []


async def broadcast_state():
    state_data = match.get_dict()
    to_remove = []
    for ws in connected_websockets:
        try:
            await ws.send_json(state_data)
        except (OSError, RuntimeError):
            to_remove.append(ws)
    for ws in to_remove:
        if ws in connected_websockets:
            connected_websockets.remove(ws)


def ensure_npu_engine():
    if match.npu_engine is None:
        match.npu_engine = NPUBatchedPUCTSearch(batch_size=16)


def get_stockfish_move(board: chess.Board, skill_level: int = 3, movetime_ms: int = 100) -> tuple[chess.Move, float, float]:
    """Queries Stockfish CPU binary via UCI pipe and standardizes score to White's perspective."""
    t0 = time.time()
    p = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    cmds = [
        "uci",
        f"setoption name Skill Level value {skill_level}",
        "isready",
        f"position fen {board.fen()}",
        f"go movetime {movetime_ms}",
    ]
    p.stdin.write("\n".join(cmds) + "\n")
    p.stdin.flush()

    bestmove_str = None
    side_to_move_score = 0.0

    while True:
        line = p.stdout.readline()
        if not line:
            break
        line = line.strip()
        if "score mate" in line:
            parts = line.split()
            if "mate" in parts:
                idx = parts.index("mate")
                try:
                    m_dist = int(parts[idx + 1])
                    side_to_move_score = 100.0 if m_dist > 0 else -100.0
                except (ValueError, IndexError):
                    logger.debug("Ignoring malformed Stockfish mate line: %r", line, exc_info=True)
        elif "score cp" in line:
            parts = line.split()
            if "cp" in parts:
                idx = parts.index("cp")
                try:
                    side_to_move_score = int(parts[idx + 1]) / 100.0
                except (ValueError, IndexError):
                    logger.debug("Ignoring malformed Stockfish cp line: %r", line, exc_info=True)
        if line.startswith("bestmove"):
            parts = line.split()
            if len(parts) >= 2:
                bestmove_str = parts[1]
            break

    p.terminate()
    dt = (time.time() - t0) * 1000.0

    # Fallback if no move returned
    if not bestmove_str or bestmove_str == "(none)":
        move = list(board.legal_moves)[0] if len(list(board.legal_moves)) > 0 else chess.Move.null()
    else:
        move = chess.Move.from_uci(bestmove_str)

    # Standardize to White-relative perspective:
    # If Black to move, side_to_move_score > 0 means Black is ahead -> White score is negative
    white_relative_eval = side_to_move_score if board.turn == chess.WHITE else -side_to_move_score
    return move, white_relative_eval, dt


async def match_loop():
    ensure_npu_engine()

    while match.is_running and not match.board.is_game_over():
        if match.is_paused:
            await asyncio.sleep(0.2)
            continue

        turn = match.board.turn

        if turn == chess.WHITE:
            # White: NeuraChess on Intel AI Boost NPU
            match.npu_active = True
            await broadcast_state()

            tensor = board_to_tensor(match.board, layout=match.npu_engine.encoder_layout)
            t0 = time.time()
            best_act, best_uci, score, nodes, pv, depth = await asyncio.to_thread(
                match.npu_engine.search,
                match.board,
                tensor,
                match.think_time_ms
            )
            dt = (time.time() - t0) * 1000.0

            match.npu_active = False
            # Standardize score to White's perspective
            white_eval = score if turn == chess.WHITE else -score
            match.npu_eval = round(white_eval, 2)
            match.npu_latency_ms = dt
            match.npu_nodes += nodes
            match.npu_nps = int((nodes / (dt / 1000.0)) if dt > 0 else 0)

            move = chess.Move.from_uci(best_uci)
            san = match.board.san(move)

            match.last_move = {
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "uci": move.uci(),
                "san": san,
                "player": "NeuraChess (NPU)",
            }

            match.history.append(match.board.copy())
            match.board.push(move)
            match.moves_san.append(san)
            match.moves_uci.append(move.uci())

        else:
            # Black: Stockfish on CPU
            move, white_eval, dt = await asyncio.to_thread(get_stockfish_move, match.board, match.stockfish_skill, 100)
            san = match.board.san(move)

            match.cpu_eval = round(white_eval, 2)
            match.cpu_latency_ms = dt
            match.last_move = {
                "from": chess.square_name(move.from_square),
                "to": chess.square_name(move.to_square),
                "uci": move.uci(),
                "san": san,
                "player": "Stockfish (CPU)",
            }

            match.history.append(match.board.copy())
            match.board.push(move)
            match.moves_san.append(san)
            match.moves_uci.append(move.uci())

        await broadcast_state()
        await asyncio.sleep(match.move_delay)

    # Game over handling
    if match.board.is_game_over():
        match.game_over = True
        match.result = match.board.result()
        if match.board.is_checkmate():
            winner = "Stockfish (CPU)" if match.board.turn == chess.WHITE else "NeuraChess (NPU)"
            match.termination = f"Checkmate! {winner} wins ({match.result})"
        elif match.board.is_stalemate():
            match.termination = "Draw by Stalemate (1/2-1/2)"
        elif match.board.is_insufficient_material():
            match.termination = "Draw by Insufficient Material (1/2-1/2)"
        elif match.board.is_fivefold_repetition():
            match.termination = "Draw by Repetition (1/2-1/2)"
        elif match.board.is_seventyfive_moves():
            match.termination = "Draw by 75-move rule (1/2-1/2)"
        else:
            match.termination = f"Game Over ({match.result})"

        match.is_running = False
        await broadcast_state()


@app.get("/")
def get_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/state")
def get_state():
    return match.get_dict()


class ControlRequest(BaseModel):
    action: str
    stockfish_skill: int | None = 3
    move_delay: float | None = 0.4
    think_time_ms: int | None = 600


@app.post("/api/control")
async def control_match(req: ControlRequest):
    global match
    if req.action == "start":
        if not match.is_running:
            match.reset(
                stockfish_skill=req.stockfish_skill or 3,
                move_delay=req.move_delay or 0.4,
                think_time_ms=req.think_time_ms or 600,
            )
            match.is_running = True
            asyncio.create_task(match_loop())
    elif req.action == "pause":
        match.is_paused = not match.is_paused
    elif req.action == "stop":
        match.is_running = False
    elif req.action == "reset":
        match.reset(
            stockfish_skill=req.stockfish_skill or 3,
            move_delay=req.move_delay or 0.4,
            think_time_ms=req.think_time_ms or 600,
        )

    await broadcast_state()
    return match.get_dict()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    await websocket.send_json(match.get_dict())
    try:
        while True:
            data = await websocket.receive_text()
            # Can receive ping/pong or control commands
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


def start_server(host="127.0.0.1", port=8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
