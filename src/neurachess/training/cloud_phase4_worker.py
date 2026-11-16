"""Unified AWS SageMaker Cloud Spot Training Worker for Phase 4 (Job A & Job B).

Supports two clean execution modes sharing data streaming, holdout filtering, and 150K GPU eval:
  - `--mode job_a`: Streams 40M leakage-free Lc0 T80 + Stockfish blend (`50% / 50%`), trains:
      (1) `ckpt_A1_frozen_trunk`: Frozen 346M 10x192 trunk + Conv73; trains HL-Gauss-64 + MLH.
      (2) `ckpt_A2_joint_middle_qat`: Unfrozen trunk with CUDA-native W8A8 ConvOnly fake-quant
          on blocks 2..7 (blocks 0,1,8,9, SE, skips, Conv73, HL64, MLH in FP16/BF16), low LR.
  - `--mode job_b`: Function-preserving warm start 10x192 -> 12x192 (identity blocks 5 & 11,
      asserts Step-0 policy max_diff < 1e-6), trains ~300M blended Lc0 T80 + Stockfish with
      two-stage LR + Dual EMA (0.9995, 0.9999) + rolling 15m S3 resume.

Always runs full 150K unified GPU eval (`far_50k`, `in_region_50k`, `heldout_400m_50k`),
exports BN-folded `.npz` + OpenVINO FP16 & Middle-INT8 IRs, and uploads all outputs to S3.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import math
import os
import queue
import random
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

for _p in [os.getcwd(), "/opt/ml/code", str(Path(__file__).resolve().parents[2])]:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from neurachess.training.bitboard_dataset import RECORD_SIZE_V1, RECORD_SIZE_V2, pack_position, unpack_position
from neurachess.training.phase4_model import (
    ChessSEResNetPhase4,
    apply_middle_w8a8_qat,
    enforce_frozen_bn_eval,
    expand_10x192_to_12x192_identity,
    export_phase4_to_folded_npz_and_ov,
    extract_10x192_state_dict,
    load_346m_into_10x192_phase4,
)
from neurachess.training.train_hl_gauss_value import CP_TANH_SCALE, compute_hl_gauss_targets

DEFAULT_REGION = "us-east-1"
DEFAULT_BUCKET = "neurachess-spot-162290422654-us-east-1"
LC0_T80_URL = "https://storage.lczero.org/files/training_data/test80/"
LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
SIDECAR_SIZE = 68
UA_HEADER = {"User-Agent": "NeuraChess-Phase4/1.1"}


def _ensure_runtime_deps() -> None:
    """Install lightweight pure/wheel dependencies (`zstandard`, `openvino`, `nncf`, `chess`) if missing in container."""
    import importlib

    needed = [("zstandard", "zstandard>=0.22.0"), ("openvino", "openvino>=2024.6.0"), ("nncf", "nncf>=2.14.0"), ("chess", "python-chess>=1.999")]
    missing = []
    for mod, pkg in needed:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[Setup] Installing missing runtime packages: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def unpack_v1_tensor_to_inputs_and_meta(raw_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode `[B, 107]` uint8 PackedV1 tensor directly on `raw_t.device` into `(inputs, acts, cps)`."""
    b = raw_t.shape[0]
    dev = raw_t.device
    bbs = raw_t[:, :96].contiguous().view(torch.int64).reshape(b, 12)
    shifts = torch.arange(64, dtype=torch.int64, device=dev)
    planes_12 = ((bbs.unsqueeze(-1) >> shifts) & 1).float().reshape(b, 12, 8, 8)
    turn = raw_t[:, 96].float().view(b, 1, 1, 1).expand(b, 1, 8, 8)
    castling = raw_t[:, 97].long()
    c_wk = ((castling >> 0) & 1).float().view(b, 1, 1, 1).expand(b, 1, 8, 8)
    c_wq = ((castling >> 1) & 1).float().view(b, 1, 1, 1).expand(b, 1, 8, 8)
    c_bk = ((castling >> 2) & 1).float().view(b, 1, 1, 1).expand(b, 1, 8, 8)
    c_bq = ((castling >> 3) & 1).float().view(b, 1, 1, 1).expand(b, 1, 8, 8)
    halfmove = (raw_t[:, 99].float() / 100.0).view(b, 1, 1, 1).expand(b, 1, 8, 8)
    inputs = torch.cat([planes_12, turn, c_wk, c_wq, c_bk, c_bq, halfmove], dim=1)

    acts_i16 = raw_t[:, 100:102].contiguous().view(torch.int16).squeeze(-1).long()
    acts = torch.where(acts_i16 < 0, acts_i16 + 65536, acts_i16)
    cps = raw_t[:, 102:104].contiguous().view(torch.int16).squeeze(-1).float()
    return inputs, acts, cps


class FastModelEMA:
    """Fused multi-tensor EMA tracker for Phase 4 BF16/FP32 training."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.shadow = {k: v.detach().clone() for k, v in raw.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        with torch.no_grad():
            for name, src in raw.state_dict().items():
                sh = self.shadow.get(name)
                if sh is None or sh.shape != src.shape:
                    self.shadow[name] = src.detach().clone()
                elif src.is_floating_point():
                    sh.mul_(self.decay).add_(src.detach(), alpha=1.0 - self.decay)
                else:
                    sh.copy_(src.detach())


def ensure_cpp_converters(work_dir: Path) -> tuple[Path, Path]:
    """Locate prebuilt static Linux x86_64 binaries or compile C++ converters cleanly in <3s."""
    code_root = Path(__file__).resolve().parent
    lc0_bin = work_dir / ("lc0_v6_converter.exe" if os.name == "nt" else "lc0_v6_converter")
    sf_bin = work_dir / ("lichess_eval_converter.exe" if os.name == "nt" else "lichess_eval_converter")

    prebuilt_lc0 = code_root / "bin" / "lc0_v6_converter_linux_x86_64"
    prebuilt_sf = code_root / "bin" / "lichess_eval_converter_linux_x86_64"
    if os.name != "nt" and prebuilt_lc0.is_file() and prebuilt_sf.is_file():
        shutil.copy2(prebuilt_lc0, lc0_bin)
        shutil.copy2(prebuilt_sf, sf_bin)
        lc0_bin.chmod(lc0_bin.stat().st_mode | stat.S_IEXEC)
        sf_bin.chmod(sf_bin.stat().st_mode | stat.S_IEXEC)
        return lc0_bin, sf_bin

    def _compile(src_cpp: Path, out_exe: Path) -> None:
        if out_exe.is_file():
            return
        try:
            import ziglang  # noqa: F401
            cmd = [sys.executable, "-m", "ziglang", "c++", "-std=c++23"]
        except ImportError:
            cxx = shutil.which("clang++") or shutil.which("g++") or "g++"
            cmd = [cxx, "-std=c++2b" if "clang" in cxx else "-std=c++20", "-Wno-attributes"]
        flags = ["-O3", "-ffast-math", "-fno-finite-math-only", "-mpopcnt", "-mavx2", "-mbmi2", "-mfma", "-pthread"]
        subprocess.check_call(cmd + flags + ["-o", str(out_exe), str(src_cpp)])

    _compile(code_root / "lc0_v6_converter.cpp", lc0_bin)
    _compile(code_root / "lichess_eval_converter.cpp", sf_bin)
    return lc0_bin, sf_bin


def build_combined_exclude_bin(eval_paths: list[Path], out_bin: Path) -> Path:
    """Concatenate 107B records from all evaluation binaries into one deduplication file."""
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with out_bin.open("wb") as fout:
        for p in eval_paths:
            if p and p.is_file():
                raw = p.read_bytes()
                n = (len(raw) // RECORD_SIZE_V1) * RECORD_SIZE_V1
                fout.write(raw[:n])
    return out_bin


def convert_v2_shards_to_unified(v2_paths: list[Path], out_prefix: Path, temperature: float = 0.5) -> tuple[Path, Path, int]:
    """Convert S3/local 691B V2 Stockfish shards into unified `(107B .bin, 68B .sidecar68.bin)`."""
    if not v2_paths:
        return out_prefix.with_suffix(".bin"), out_prefix.with_suffix(".sidecar68.bin"), 0

    all_v1 = []
    all_sc = []
    import struct

    for p in v2_paths:
        if not p.is_file():
            continue
        raw = p.read_bytes()
        n = len(raw) // RECORD_SIZE_V2
        if n <= 0:
            continue
        arr = np.frombuffer(raw[: n * RECORD_SIZE_V2], dtype=np.uint8).reshape(n, RECORD_SIZE_V2)
        v1 = arr[:, :RECORD_SIZE_V1].copy()
        all_v1.append(v1)

        # V2 stores extra (k: uint8 at 107, acts: 4x u16 at 108..116, cps: 4x i16 at 116..124)
        sc_arr = np.zeros((n, SIDECAR_SIZE), dtype=np.uint8)
        best_act = v1[:, 100:102].copy().view(np.uint16).reshape(n)
        best_cp = v1[:, 102:104].copy().view(np.int16).reshape(n).astype(np.float32)
        n_ext = np.minimum(4, arr[:, 107].astype(np.int32))
        ext_acts = arr[:, 108:116].copy().view(np.uint16).reshape(n, 4)
        ext_cps = arr[:, 116:124].copy().view(np.int16).reshape(n, 4).astype(np.float32)

        acts8 = np.full((n, 8), 65535, dtype=np.uint16)
        acts8[:, 0] = best_act
        acts8[:, 1:5] = ext_acts
        cps5 = np.full((n, 5), -1e9, dtype=np.float32)
        cps5[:, 0] = best_cp
        for j in range(4):
            valid_j = j < n_ext
            cps5[valid_j, 1 + j] = ext_cps[valid_j, j]
        scaled = cps5 / (100.0 * max(1e-3, temperature))
        scaled -= np.max(scaled, axis=1, keepdims=True)
        ex = np.exp(scaled)
        for j in range(4):
            ex[j >= n_ext, 1 + j] = 0.0
        probs5 = ex / np.maximum(1e-8, np.sum(ex, axis=1, keepdims=True))

        sc_arr[:, 0] = (1 + n_ext).astype(np.uint8)
        sc_arr[:, 1] = v1[:, 96] & 1
        sc_arr[:, 2:4].view(np.uint16)[:] = 500
        sc_arr[:, 4:20].view(np.uint16)[:] = acts8
        probs8 = np.zeros((n, 8), dtype=np.float32)
        probs8[:, :5] = probs5
        sc_arr[:, 20:52].view(np.float32)[:] = probs8
        q = np.tanh(best_cp * CP_TANH_SCALE).astype(np.float32)
        sc_arr[:, 52:56].view(np.float32)[:, 0] = q
        sc_arr[:, 60:64].view(np.float32)[:, 0] = q
        sc_arr[:, 64:68].view(np.float32)[:, 0] = -1.0
        all_sc.append(sc_arr)

    out_bin = out_prefix.with_suffix(".bin")
    out_sc = out_prefix.with_suffix(".sidecar68.bin")
    if not all_v1:
        return out_bin, out_sc, 0
    v1_cat = np.concatenate(all_v1, axis=0)
    sc_cat = np.concatenate(all_sc, axis=0)
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(v1_cat.tobytes())
    out_sc.write_bytes(sc_cat.tobytes())
    return out_bin, out_sc, int(v1_cat.shape[0])


def list_t80_nonempty_urls(min_bytes: int = 50_000_000, seed: int = 42) -> list[str]:
    """Fetch `storage.lczero.org/files/training_data/test80/` HTML and filter non-empty `.tar` archives."""
    html = urllib.request.urlopen(urllib.request.Request(LC0_T80_URL, headers=UA_HEADER), timeout=30).read().decode("utf-8", "ignore")
    pat = re.compile(r'href="(training-run1-test80-[0-9]+-[0-9]+\.tar)"[^>]*>[^<]+</a>\s+\S+\s+\S+\s+(\d+)')
    urls = []
    for fname, sz_s in pat.findall(html):
        if int(sz_s) >= min_bytes:
            urls.append(LC0_T80_URL + fname)
    rng = random.Random(seed)
    rng.shuffle(urls)
    return urls


def stream_lc0_t80_chunk(
    lc0_exe: Path,
    tar_url: str,
    out_prefix: Path,
    exclude_bin: Path | None,
    max_records: int = 0,
    threads: int = 4,
) -> tuple[Path, Path, int]:
    """Stream one Lc0 T80 `.tar` URL directly through `lc0_v6_converter` stdout/stdin pipe."""
    cmd = [str(lc0_exe), "--out-prefix", str(out_prefix), "--min-ply", "9", "--threads", str(threads)]
    if max_records > 0:
        cmd += ["--max-records", str(max_records)]
    if exclude_bin and exclude_bin.is_file():
        cmd += ["--exclude-bin", str(exclude_bin)]
    cmd.append("-")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    req = urllib.request.Request(tar_url, headers=UA_HEADER)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            while True:
                buf = resp.read(2 * 1024 * 1024)
                if not buf:
                    break
                try:
                    proc.stdin.write(buf)
                except BrokenPipeError:
                    break
    except Exception:
        pass
    try:
        proc.stdin.close()
    except Exception:
        pass
    out, _ = proc.communicate(timeout=300)
    lines = out.decode("utf-8", "ignore").strip().splitlines()
    stats = json.loads(lines[-1]) if lines else {}
    return Path(stats.get("out_bin", str(out_prefix) + ".bin")), Path(stats.get("out_sidecar", str(out_prefix) + ".lc0_sidecar.bin")), int(stats.get("records_written", 0))


def stream_lc0_t80_parallel(
    lc0_exe: Path,
    tar_urls: list[str],
    start_idx: int,
    out_dir: Path,
    exclude_bin: Path | None,
    target_positions: int,
    *,
    workers: int = 3,
    threads_per_worker: int = 1,
    tag: str = "Lc0",
) -> tuple[list[tuple[Path, Path, int]], int]:
    """Stream multiple Lc0 T80 `.tar` URLs in parallel waves until `target_positions` is reached."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[Path, Path, int]] = []
    got = 0
    idx = start_idx % max(1, len(tar_urls))
    n_urls = len(tar_urls)
    if n_urls == 0 or target_positions <= 0:
        return pairs, idx

    wave = 0
    while got < target_positions:
        rem = target_positions - got
        batch_n = 1 if rem <= 3_500_000 else min(workers, max(1, (rem + 3_500_000) // 4_000_000))
        jobs = []
        for bi in range(batch_n):
            url = tar_urls[(idx + bi) % n_urls]
            jobs.append((wave * workers + bi, url))
        idx = (idx + batch_n) % n_urls
        wave += 1

        per_cap = rem if batch_n == 1 else max(500_000, rem // batch_n + 500_000)
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_n) as ex:
            futs = {
                ex.submit(
                    stream_lc0_t80_chunk,
                    lc0_exe,
                    url,
                    out_dir / f"lc0_w{jid:04d}",
                    exclude_bin,
                    per_cap,
                    threads_per_worker,
                ): url
                for jid, url in jobs
            }
            for fut in concurrent.futures.as_completed(futs):
                url = futs[fut]
                try:
                    bp, sp, cnt = fut.result()
                    if cnt > 0 and bp.is_file():
                        pairs.append((bp, sp, cnt))
                        got += cnt
                        print(f"[{tag}] +{cnt:,} -> {got:,}/{target_positions:,} ({Path(url).name})", flush=True)
                except Exception as exc:
                    print(f"[{tag} Warning] {Path(url).name}: {exc}", flush=True)
    return pairs, idx


class LichessContinuousStream:
    """Single-connection persistent `lichess_db_eval.jsonl.zst` reader yielding chunked `(v1, sc)` shards without resetting."""

    def __init__(
        self,
        sf_exe: Path,
        exclude_hash_file: Path | None,
        exclude_bin: Path | None,
        initial_skip: int = 1_000_000,
        threads: int = 2,
        opener=None,
    ):
        self.sf_exe = sf_exe
        self.exclude_hash_file = exclude_hash_file
        self.exclude_bin = exclude_bin
        self.initial_skip = int(initial_skip)
        self.threads = int(threads)
        self.opener = opener
        self._resp = None
        self._zr = None
        self._byte_buf = b""
        self._skipped_initial = False
        self._chunk_seq = 0

    def _open_stream(self) -> None:
        import zstandard as zstd

        if self._zr is not None:
            return
        if self.opener is not None:
            self._resp = self.opener(LICHESS_EVAL_URL)
        else:
            req = urllib.request.Request(LICHESS_EVAL_URL, headers=UA_HEADER)
            self._resp = urllib.request.urlopen(req, timeout=120)
        dctx = zstd.ZstdDecompressor()
        self._zr = dctx.stream_reader(self._resp, closefd=False)

    def close(self) -> None:
        for obj in (self._zr, self._resp):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._zr = None
        self._resp = None
        self._byte_buf = b""

    def read_chunk(self, out_prefix: Path, target_records: int) -> tuple[Path, Path, int]:
        skip_arg = self.initial_skip if not self._skipped_initial else 0
        cmd = [
            str(self.sf_exe),
            "--out-prefix",
            str(out_prefix),
            "--skip-lines",
            str(skip_arg),
            "--max-records",
            str(target_records),
            "--threads",
            str(self.threads),
        ]
        if self.exclude_hash_file and self.exclude_hash_file.is_file():
            cmd += ["--exclude-hashes", str(self.exclude_hash_file)]
        if self.exclude_bin and self.exclude_bin.is_file():
            cmd += ["--exclude-bin", str(self.exclude_bin)]
        cmd.append("-")

        self._open_stream()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None

        while proc.poll() is None:
            assert self._zr is not None
            raw = self._zr.read(1024 * 1024)
            if not raw:
                if self._byte_buf:
                    tail = self._byte_buf + (b"" if self._byte_buf.endswith(b"\n") else b"\n")
                    self._byte_buf = b""
                    try:
                        proc.stdin.write(tail)
                    except BrokenPipeError:
                        pass
                break
            data = self._byte_buf + raw
            nl = data.rfind(b"\n")
            if nl == -1:
                self._byte_buf = data
                continue
            lines_blk = data[: nl + 1]
            self._byte_buf = data[nl + 1 :]
            try:
                proc.stdin.write(lines_blk)
                proc.stdin.flush()
            except BrokenPipeError:
                break

        self._skipped_initial = True
        self._chunk_seq += 1
        try:
            proc.stdin.close()
        except Exception:
            pass
        out, _ = proc.communicate(timeout=300)
        lines = out.decode("utf-8", "ignore").strip().splitlines()
        stats = json.loads(lines[-1]) if lines else {}
        n_wr = int(stats.get("records_written", 0))
        print(f"[LichessStream] chunk #{self._chunk_seq}: wrote {n_wr:,} records (total_skipped={stats.get('skipped_overlap', 0):,})", flush=True)
        return (
            Path(stats.get("out_bin", str(out_prefix) + ".bin")),
            Path(stats.get("out_sidecar", str(out_prefix) + ".sidecar68.bin")),
            n_wr,
        )


def stream_lichess_sf_chunks(
    sf_exe: Path,
    out_dir: Path,
    exclude_hash_file: Path | None,
    exclude_bin: Path | None,
    target_positions: int,
    skip_lines: int = 10_000_000,
    threads: int = 4,
) -> list[tuple[Path, Path, int]]:
    """Stream `lichess_db_eval.jsonl.zst` through `zstd` -> C++ `lichess_eval_converter` in bounded chunks."""
    streamer = LichessContinuousStream(sf_exe, exclude_hash_file, exclude_bin, initial_skip=skip_lines, threads=threads)
    try:
        bp, sp, n_wr = streamer.read_chunk(out_dir / "sf_stream", target_positions)
        return [(bp, sp, n_wr)]
    finally:
        streamer.close()


def load_unified_chunk(bin_p: Path, sc_p: Path) -> tuple[np.ndarray, np.ndarray]:
    rb = bin_p.read_bytes()
    rs = sc_p.read_bytes()
    n = min(len(rb) // RECORD_SIZE_V1, len(rs) // SIDECAR_SIZE)
    v1 = np.frombuffer(rb[: n * RECORD_SIZE_V1], dtype=np.uint8).reshape(n, RECORD_SIZE_V1).copy()
    sc = np.frombuffer(rs[: n * SIDECAR_SIZE], dtype=np.uint8).reshape(n, SIDECAR_SIZE).copy()
    return v1, sc


def unpack_unified_sidecar_batch(raw_sc: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode `UnifiedSidecar68` batch `[B, 68]` on GPU into `(acts8, probs8, plies_left, has_mlh)`."""
    B = raw_sc.shape[0]
    flags = raw_sc[:, 1]
    has_mlh = ((flags & 8) != 0).float()

    acts_u16 = raw_sc[:, 4:20].contiguous().view(torch.int16).view(B, 8).long()
    acts8 = torch.where(acts_u16 < 0, acts_u16 + 65536, acts_u16).clamp(0, 4671)
    probs8 = raw_sc[:, 20:52].contiguous().view(torch.float32).view(B, 8)
    probs8 = probs8 / probs8.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    plies_left = raw_sc[:, 64:68].contiguous().view(torch.float32).view(B).clamp(min=0.0, max=250.0)
    return acts8, probs8, plies_left, has_mlh


def compute_cosine_lr(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
    prog = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * prog))


_EVAL_SPLIT_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


@torch.no_grad()
def evaluate_unified_150k(
    model: ChessSEResNetPhase4,
    eval_files: dict[str, Path],
    device: torch.device,
    batch_size: int = 512,
    max_eval: int = 50000,
    ref_baseline_pol: dict[str, torch.Tensor] | None = None,
) -> dict:
    """Run full 150K unified evaluation across `far_50k`, `in_region_50k`, and `heldout_400m_50k`."""
    from neurachess.core.tensor_movegen import get_legal_action_mask

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.eval()
    results = {}

    for split_name, path in eval_files.items():
        if not path or not path.is_file():
            results[split_name] = {"error": f"missing {path}"}
            continue
        cache_k = (str(path.resolve()), int(max_eval))
        if cache_k in _EVAL_SPLIT_CACHE:
            arr, masks_bool = _EVAL_SPLIT_CACHE[cache_k]
            n = len(arr)
        else:
            raw = path.read_bytes()
            n = min(max_eval, len(raw) // RECORD_SIZE_V1)
            arr = np.frombuffer(raw[: n * RECORD_SIZE_V1], dtype=np.uint8).reshape(n, RECORD_SIZE_V1)
            masks_bool = np.empty((n, 4672), dtype=np.bool_)
            for i in range(n):
                rec = unpack_position(raw[i * RECORD_SIZE_V1 : (i + 1) * RECORD_SIZE_V1])
                masks_bool[i] = get_legal_action_mask(rec["board"]) > 0.5
            _EVAL_SPLIT_CACHE[cache_k] = (arr, masks_bool)

        tot_ce = 0.0
        tot_hl_ce = 0.0
        top1_ok = 0
        top3_ok = 0
        tot_mse = 0.0
        dec_cnt = 0
        dec_ok = 0
        max_pol_drift = 0.0

        for i in range(0, n, batch_size):
            b1 = min(n, i + batch_size)
            raw_t = torch.from_numpy(arr[i:b1].copy()).to(device)
            inputs, acts, cps = unpack_v1_tensor_to_inputs_and_meta(raw_t)
            inputs = inputs.to(device=device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
            v_targets = torch.tanh(cps * CP_TANH_SCALE)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                logits, hl_logits, val_exp, _ = raw_model(inputs)

            mask_t = torch.from_numpy(masks_bool[i:b1]).to(device)
            masked_logits = logits.float() + (~mask_t).float() * -10000.0
            tot_ce += float(F.cross_entropy(masked_logits, acts, reduction="sum").item())

            top3 = masked_logits.topk(3, dim=-1).indices
            top1_ok += int((top3[:, 0] == acts).sum().item())
            top3_ok += int((top3 == acts.unsqueeze(1)).any(dim=-1).sum().item())

            hl_tgt = compute_hl_gauss_targets(cps, raw_model.hl_edges, raw_model.hl_sigma)
            tot_hl_ce += float(-(hl_tgt * F.log_softmax(hl_logits.float(), dim=-1)).sum(dim=-1).sum().item())

            vp = val_exp.squeeze(-1).float()
            tot_mse += float(((vp - v_targets) ** 2).sum().item())
            dec = v_targets.abs() >= 0.1429
            if dec.any():
                dec_ok += int(((vp[dec] * v_targets[dec]) > 0).sum().item())
                dec_cnt += int(dec.sum().item())

            if ref_baseline_pol is not None and split_name in ref_baseline_pol and i == 0:
                ref_slice = ref_baseline_pol[split_name].to(device)
                k = min(ref_slice.shape[0], logits.shape[0])
                max_pol_drift = float((logits[:k].float() - ref_slice[:k].float()).abs().max().item())

        row = {
            "samples": n,
            "policy_ce": round(tot_ce / max(1, n), 4),
            "top1_pct": round(100.0 * top1_ok / max(1, n), 2),
            "top3_pct": round(100.0 * top3_ok / max(1, n), 2),
            "hl_gauss_ce": round(tot_hl_ce / max(1, n), 4),
            "val_mse": round(tot_mse / max(1, n), 5),
            "decisive_sign_pct": round(100.0 * dec_ok / max(1, dec_cnt), 2),
        }
        if ref_baseline_pol is not None and split_name in ref_baseline_pol:
            row["max_policy_logit_drift_vs_346m"] = round(max_pol_drift, 6)
        results[split_name] = row
    return results


def _download_or_locate(local_dirs: list[Path], filename: str, s3_client, bucket: str, s3_key: str, dst: Path) -> Path:
    for d in local_dirs:
        cand = d / filename
        if cand.is_file():
            return cand
    if not dst.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[S3] Downloading s3://{bucket}/{s3_key} -> {dst}", flush=True)
        s3_client.download_file(bucket, s3_key, str(dst))
    return dst


def _build_mixed_epoch_pool(
    v1_sf: np.ndarray,
    sc_sf: np.ndarray,
    v1_lc0: np.ndarray,
    sc_lc0: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    v1 = np.concatenate([v1_sf, v1_lc0], axis=0) if (len(v1_sf) and len(v1_lc0)) else (v1_lc0 if len(v1_lc0) else v1_sf)
    sc = np.concatenate([sc_sf, sc_lc0], axis=0) if (len(sc_sf) and len(sc_lc0)) else (sc_lc0 if len(sc_lc0) else sc_sf)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(v1))
    return v1[perm], sc[perm]


def _train_step_batch(
    model: ChessSEResNetPhase4,
    optimizer: torch.optim.Optimizer,
    raw_v1: torch.Tensor,
    raw_sc: torch.Tensor,
    device: torch.device,
    *,
    freeze_trunk: bool,
    pol_weight: float,
    val_weight: float,
    mlh_weight: float,
    grad_clip: float = 1.0,
) -> tuple[float, float, float, float]:
    inp, _, cps = unpack_v1_tensor_to_inputs_and_meta(raw_v1)
    inp = inp.to(device=device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
    acts8, probs8, plies_left, has_mlh = unpack_unified_sidecar_batch(raw_sc, device)

    raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
    hl_targets = compute_hl_gauss_targets(cps, raw_m.hl_edges, raw_m.hl_sigma)

    optimizer.zero_grad(set_to_none=True)
    amp_ok = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_ok):
        if freeze_trunk:
            with torch.no_grad():
                feat = raw_m.forward_trunk(inp)
            pol_logits, hl_logits, _, mlh_pred = raw_m.forward_policy_and_heads(feat)
        else:
            pol_logits, hl_logits, _, mlh_pred = model(inp)

        l_hl = -(hl_targets * F.log_softmax(hl_logits.float(), dim=-1)).sum(dim=-1).mean()
        l_pol = torch.tensor(0.0, device=device)
        if pol_weight > 0.0:
            logp = F.log_softmax(pol_logits.float(), dim=-1)
            l_pol = -(probs8 * logp.gather(1, acts8)).sum(dim=-1).mean()

        l_mlh = torch.tensor(0.0, device=device)
        mlh_cnt = has_mlh.sum()
        if mlh_weight > 0.0:
            per_ex = F.smooth_l1_loss(mlh_pred.squeeze(-1).float() / 50.0, plies_left / 50.0, reduction="none")
            l_mlh = (per_ex * has_mlh).sum() / mlh_cnt.clamp(min=1.0)

        loss = pol_weight * l_pol + val_weight * l_hl + mlh_weight * l_mlh

    loss.backward()
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return float(loss.item()), float(l_pol.item()), float(l_hl.item()), float(l_mlh.item())


def _train_on_pool(
    model: ChessSEResNetPhase4,
    v1_np: np.ndarray,
    sc_np: np.ndarray,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    freeze_trunk: bool,
    pol_weight: float,
    val_weight: float,
    mlh_weight: float,
    ema_list: list[FastModelEMA],
    step_state: dict,
    lr_fn,
    max_positions: int,
    s3_ckpt_fn=None,
    stop_deadline_ts: float = 0.0,
) -> bool:
    """Iterate randomized minibatches directly from GPU-pinned uint8 chunk arrays."""
    enforce_frozen_bn_eval(model)
    n = len(v1_np)
    chunk_cap = min(n, 2_000_000)

    for start in range(0, n, chunk_cap):
        if max_positions > 0 and step_state["seen"] >= max_positions:
            return False
        if stop_deadline_ts > 0 and time.time() >= stop_deadline_ts:
            return True
        end = min(n, start + chunk_cap)
        k = end - start
        v1_t = torch.from_numpy(v1_np[start:end].copy()).to(device)
        sc_t = torch.from_numpy(sc_np[start:end].copy()).to(device)
        perm = torch.randperm(k, device=device)

        num_b = k // batch_size
        for bi in range(num_b):
            if max_positions > 0 and step_state["seen"] >= max_positions:
                del v1_t, sc_t, perm
                return False
            if stop_deadline_ts > 0 and (bi % 100 == 0) and time.time() >= stop_deadline_ts:
                del v1_t, sc_t, perm
                return True

            lr_fn(optimizer, step_state["step"], step_state["seen"])
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            rb_v1 = v1_t.index_select(0, idx)
            rb_sc = sc_t.index_select(0, idx)
            loss_v, lp_v, lh_v, lm_v = _train_step_batch(
                model, optimizer, rb_v1, rb_sc, device,
                freeze_trunk=freeze_trunk,
                pol_weight=pol_weight,
                val_weight=val_weight,
                mlh_weight=mlh_weight,
            )
            for ema in ema_list:
                ema.update(model)

            step_state["step"] += 1
            step_state["seen"] += batch_size
            if step_state["step"] % 500 == 0:
                dt = max(0.1, time.time() - step_state["t0"])
                rate = step_state["seen"] / dt
                lrs = [round(pg["lr"], 7) for pg in optimizer.param_groups]
                print(
                    f"  [Step {step_state['step']:6d} | {step_state['seen']:,}/{max_positions:,}] "
                    f"{rate:7.0f} pos/s | loss={loss_v:.4f} (pol={lp_v:.4f}, hl={lh_v:.4f}, mlh={lm_v:.4f}) | lr={lrs}",
                    flush=True,
                )
            if s3_ckpt_fn is not None and (time.time() - step_state["last_ckpt"]) >= 900.0:
                step_state["last_ckpt"] = time.time()
                s3_ckpt_fn(step_state)
        del v1_t, sc_t, perm
    return False


def _capture_first_batch_pol_logits(model: nn.Module, eval_files: dict[str, Path], device: torch.device) -> dict[str, torch.Tensor]:
    raw_m = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_m.eval()
    out = {}
    with torch.no_grad():
        for k, p in eval_files.items():
            if not p or not p.is_file():
                continue
            raw = p.read_bytes()[: 512 * RECORD_SIZE_V1]
            n = len(raw) // RECORD_SIZE_V1
            arr = torch.from_numpy(np.frombuffer(raw[: n * RECORD_SIZE_V1], dtype=np.uint8).reshape(n, RECORD_SIZE_V1).copy()).to(device)
            inp, _, _ = unpack_v1_tensor_to_inputs_and_meta(arr)
            inp = inp.to(device=device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                pol, _, _, _ = raw_m(inp)
            out[k] = pol.detach().cpu()
    return out


def run_job_a(args: argparse.Namespace, device: torch.device, s3_client, dirs: dict) -> None:
    tmp_dir: Path = dirs["tmp"]
    model_dir: Path = dirs["model"]
    eval_files: dict[str, Path] = dirs["eval_files"]

    init_pt = _download_or_locate(
        dirs["search"], "full_346m_10x192_final_model.pt", s3_client, args.bucket,
        "checkpoints/full_346m_10x192/full_346m_10x192_final_model.pt", tmp_dir / "init_346m_10x192.pt"
    )
    lc0_exe, sf_exe = ensure_cpp_converters(tmp_dir)
    excl_bin = dirs["excl_bin"]
    excl_hashes = dirs["excl_hashes"]

    # Stream & cache 40M blend (20M Lc0 T80 + 20M Stockfish = 40M total) cleanly onto NVMe once
    cache_dir = tmp_dir / "job_a_40m_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sf_target = args.total_positions // 2
    lc0_target = args.total_positions - sf_target

    # Stream Lichess SF + parallel Lc0 T80 concurrently for maximum wire throughput
    v2_b, v2_s, v2_n = convert_v2_shards_to_unified(dirs["v2_shards"], cache_dir / "v2_sf")
    sf_pairs = [(v2_b, v2_s, v2_n)] if v2_n > 0 else []
    rem_sf = max(0, sf_target - v2_n)
    t80_urls = list_t80_nonempty_urls(seed=args.seed)

    sf_fut = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as data_ex:
        if rem_sf > 0:
            sf_fut = data_ex.submit(stream_lichess_sf_chunks, sf_exe, cache_dir, excl_hashes, excl_bin, rem_sf, 1_000_000, 2)
        lc0_pairs, _ = stream_lc0_t80_parallel(
            lc0_exe, t80_urls, 0, cache_dir, excl_bin, lc0_target, workers=3, threads_per_worker=1, tag="Job A Data"
        )
        if sf_fut is not None:
            sf_pairs.extend(sf_fut.result())

    # Load 40M into CPU host arrays (pinned-ready ~7GB) for fast shuffling between A1 & A2
    sf_v1_list, sf_sc_list = [], []
    for bp, sp, cnt in sf_pairs:
        if cnt > 0 and bp.is_file():
            v, s = load_unified_chunk(bp, sp)
            sf_v1_list.append(v); sf_sc_list.append(s)
    lc0_v1_list, lc0_sc_list = [], []
    for bp, sp, cnt in lc0_pairs:
        if cnt > 0 and bp.is_file():
            v, s = load_unified_chunk(bp, sp)
            lc0_v1_list.append(v); lc0_sc_list.append(s)

    v1_sf = np.concatenate(sf_v1_list, axis=0) if sf_v1_list else np.empty((0, RECORD_SIZE_V1), dtype=np.uint8)
    sc_sf = np.concatenate(sf_sc_list, axis=0) if sf_sc_list else np.empty((0, SIDECAR_SIZE), dtype=np.uint8)
    v1_lc0 = np.concatenate(lc0_v1_list, axis=0) if lc0_v1_list else np.empty((0, RECORD_SIZE_V1), dtype=np.uint8)
    sc_lc0 = np.concatenate(lc0_sc_list, axis=0) if lc0_sc_list else np.empty((0, SIDECAR_SIZE), dtype=np.uint8)
    del sf_v1_list, sf_sc_list, lc0_v1_list, lc0_sc_list
    gc.collect()

    mix_v1, mix_sc = _build_mixed_epoch_pool(v1_sf, sc_sf, v1_lc0, sc_lc0, seed=args.seed)
    del v1_sf, sc_sf, v1_lc0, sc_lc0
    gc.collect()
    n_pool = len(mix_v1)
    print(f"[Job A Ready] Blended 40M pool size: {n_pool:,} positions", flush=True)

    # Build calib 500 inputs from pool for OpenVINO NNCF export
    calib_x = unpack_v1_tensor_to_inputs_and_meta(torch.from_numpy(mix_v1[:500].copy()))[0].numpy().astype(np.float32)
    calib_inputs = [calib_x[i : i + 1] for i in range(len(calib_x))]

    # -------------------- Checkpoint A1: Frozen 346M Trunk -> Train HL-Gauss-64 + MLH --------------------
    m_a1 = ChessSEResNetPhase4(num_blocks=10, channels=192, use_bn=True).to(device)
    load_346m_into_10x192_phase4(m_a1, init_pt)
    if device.type == "cuda":
        m_a1 = m_a1.to(memory_format=torch.channels_last)
    ref_pol = _capture_first_batch_pol_logits(m_a1, eval_files, device)

    a1_pt = model_dir / "A1" / "ckpt_A1_frozen_trunk.pt"
    a1_resumed = False
    if args.resume and s3_client and args.bucket:
        try:
            a1_pt.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(args.bucket, "checkpoints/phase4_job_a/A1/ckpt_A1_frozen_trunk.pt", str(a1_pt))
            ck1 = torch.load(str(a1_pt), map_location=device, weights_only=False)
            m_a1.load_state_dict(ck1["model_state_dict"])
            eval_a1 = ck1.get("eval_150k") or evaluate_unified_150k(m_a1, eval_files, device, ref_baseline_pol=ref_pol)
            exp_a1 = {"resumed_from_s3": "checkpoints/phase4_job_a/A1/ckpt_A1_frozen_trunk.pt"}
            a1_resumed = True
            print(f"[Job A1 Resumed from S3] seen={ck1.get('seen', 0):,}", flush=True)
        except Exception:
            a1_resumed = False

    if not a1_resumed:
        for p in list(m_a1.init_conv.parameters()) + list(m_a1.init_bn.parameters()) + list(m_a1.res_blocks.parameters()) + list(m_a1.pol_conv.parameters()) + list(m_a1.pol_bn.parameters()):
            p.requires_grad_(False)
        for mod in (m_a1.init_bn, m_a1.res_blocks, m_a1.pol_bn):
            mod.eval()

        head_params = list(m_a1.hl_head.parameters()) + list(m_a1.ml_head.parameters())
        opt_a1 = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4, fused=(device.type == "cuda"))
        tot_steps_a1 = max(1, (args.total_positions + args.batch_size - 1) // args.batch_size)
        st_a1 = {"step": 0, "seen": 0, "seed": args.seed, "t0": time.time(), "last_ckpt": time.time()}

        def _lr_a1(opt, stp, _seen):
            lr = compute_cosine_lr(stp, tot_steps_a1, min(500, tot_steps_a1 // 10), 1e-3, 1e-5)
            for pg in opt.param_groups:
                pg["lr"] = lr

        print("\n=== [Job A1] Training Frozen-Trunk HL-Gauss-64 + MLH (40M pos) ===", flush=True)
        m_a1.hl_head.train(); m_a1.ml_head.train()
        while st_a1["seen"] < args.total_positions:
            _train_on_pool(
                m_a1, mix_v1, mix_sc, device, opt_a1,
                batch_size=args.batch_size, freeze_trunk=True,
                pol_weight=0.0, val_weight=1.0, mlh_weight=0.05,
                ema_list=[], step_state=st_a1, lr_fn=_lr_a1,
                max_positions=args.total_positions,
            )

        eval_a1 = evaluate_unified_150k(m_a1, eval_files, device, ref_baseline_pol=ref_pol)
        print(f"[Job A1 Eval 150K] {json.dumps(eval_a1)}", flush=True)
        exp_a1 = export_phase4_to_folded_npz_and_ov(m_a1, model_dir / "A1", "ckpt_A1_frozen_trunk")
        torch.save({"model_state_dict": m_a1.state_dict(), "eval_150k": eval_a1, "seen": st_a1["seen"]}, a1_pt)
        if s3_client and args.bucket:
            for fn in list((model_dir / "A1").glob("*")):
                s3_client.upload_file(str(fn), args.bucket, f"checkpoints/phase4_job_a/A1/{fn.name}")

    # -------------------- Checkpoint A2: Joint Middle-Block (2..7) W8A8 QAT --------------------
    print("\n=== [Job A2] Unfreezing + W8A8 Middle 2..7 QAT (40M pos, LR 2e-5 -> 1e-6) ===", flush=True)
    m_a2 = ChessSEResNetPhase4(num_blocks=10, channels=192, use_bn=True).to(device)
    m_a2.load_state_dict(m_a1.state_dict())
    for p in m_a2.parameters():
        p.requires_grad_(True)
    apply_middle_w8a8_qat(m_a2, qat_blocks=(2, 3, 4, 5, 6, 7))
    if device.type == "cuda":
        m_a2 = m_a2.to(memory_format=torch.channels_last)

    # Calibrate W8A8 activation ranges cleanly on 500 positions while keeping frozen BNs in eval
    m_a2.train()
    enforce_frozen_bn_eval(m_a2)
    with torch.no_grad():
        cx = torch.from_numpy(calib_x).to(device=device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format)
        m_a2(cx)

    ema_a2 = FastModelEMA(m_a2, decay=0.9995)
    train_p2 = [p for p in m_a2.parameters() if p.requires_grad]
    opt_a2 = torch.optim.AdamW(train_p2, lr=2e-5, weight_decay=1e-4, fused=(device.type == "cuda"))
    tot_steps_a2 = max(1, (args.total_positions + args.batch_size - 1) // args.batch_size)
    st_a2 = {"step": 0, "seen": 0, "seed": args.seed + 99, "t0": time.time(), "last_ckpt": time.time()}

    def _lr_a2(opt, stp, _seen):
        lr = compute_cosine_lr(stp, tot_steps_a2, 500, 2e-5, 1e-6)
        for pg in opt.param_groups:
            pg["lr"] = lr

    while st_a2["seen"] < args.total_positions:
        _train_on_pool(
            m_a2, mix_v1, mix_sc, device, opt_a2,
            batch_size=args.batch_size, freeze_trunk=False,
            pol_weight=1.0, val_weight=1.0, mlh_weight=0.05,
            ema_list=[ema_a2], step_state=st_a2, lr_fn=_lr_a2,
            max_positions=args.total_positions,
        )

    m_a2.load_state_dict(ema_a2.shadow)
    eval_a2 = evaluate_unified_150k(m_a2, eval_files, device, ref_baseline_pol=ref_pol)
    print(f"[Job A2 Eval 150K] {json.dumps(eval_a2)}", flush=True)
    exp_a2 = export_phase4_to_folded_npz_and_ov(
        m_a2, model_dir / "A2", "ckpt_A2_joint_middle_qat",
        calib_inputs=calib_inputs, int8_blocks=(2, 3, 4, 5, 6, 7),
    )
    a2_pt = model_dir / "A2" / "ckpt_A2_joint_middle_qat.pt"
    torch.save({"model_state_dict": m_a2.state_dict(), "ema_09995_state_dict": ema_a2.shadow, "eval_150k": eval_a2, "seen": st_a2["seen"]}, a2_pt)

    rep_a = {"mode": "job_a", "a1_eval_150k": eval_a1, "a2_eval_150k": eval_a2, "a1_export": exp_a1, "a2_export": exp_a2}
    (model_dir / "phase4_job_a_report.json").write_text(json.dumps(rep_a, indent=2), encoding="utf-8")
    if s3_client and args.bucket:
        for fn in list((model_dir / "A2").glob("*")):
            s3_client.upload_file(str(fn), args.bucket, f"checkpoints/phase4_job_a/A2/{fn.name}")
        s3_client.upload_file(str(model_dir / "phase4_job_a_report.json"), args.bucket, "checkpoints/phase4_job_a/phase4_job_a_report.json")


def run_job_b(args: argparse.Namespace, device: torch.device, s3_client, dirs: dict) -> None:
    tmp_dir: Path = dirs["tmp"]
    model_dir: Path = dirs["model"]
    ckpt_dir: Path = dirs["ckpt"]
    eval_files: dict[str, Path] = dirs["eval_files"]

    init_pt = _download_or_locate(
        dirs["search"], "full_346m_10x192_final_model.pt", s3_client, args.bucket,
        "checkpoints/full_346m_10x192/full_346m_10x192_final_model.pt", tmp_dir / "init_346m_10x192.pt"
    )
    lc0_exe, sf_exe = ensure_cpp_converters(tmp_dir)
    excl_bin = dirs["excl_bin"]
    excl_hashes = dirs["excl_hashes"]

    m12, step0_diff = expand_10x192_to_12x192_identity(init_pt, seed=args.seed, verify_tolerance=1e-6)
    print(f"[Job B Step-0 Policy Preservation VERIFIED] max_diff={step0_diff:.3e} (< 1e-6)", flush=True)
    m12 = m12.to(device)
    if device.type == "cuda":
        m12 = m12.to(memory_format=torch.channels_last)
    ref_pol = _capture_first_batch_pol_logits(m12, eval_files, device)

    new_params = (
        list(m12.res_blocks[5].parameters())
        + list(m12.res_blocks[11].parameters())
        + list(m12.hl_head.parameters())
        + list(m12.ml_head.parameters())
    )
    new_ids = {id(p) for p in new_params}
    base_params = [p for p in m12.parameters() if p.requires_grad and id(p) not in new_ids]

    opt = torch.optim.AdamW(
        [{"params": new_params, "lr": 2e-4}, {"params": base_params, "lr": 2e-5}],
        weight_decay=1e-4,
        fused=(device.type == "cuda"),
    )
    ema_9995 = FastModelEMA(m12, decay=0.9995)
    ema_9999 = FastModelEMA(m12, decay=0.9999)

    step_state = {"step": 0, "seen": 0, "seed": args.seed, "t0": time.time(), "last_ckpt": time.time()}
    s3_resume_key = "checkpoints/phase4_job_b/latest.pt"
    local_resume = ckpt_dir / "job_b_latest.pt"
    if args.resume and s3_client and args.bucket:
        try:
            s3_client.download_file(args.bucket, s3_resume_key, str(local_resume))
            ck = torch.load(str(local_resume), map_location=device, weights_only=False)
            m12.load_state_dict(ck["model_state_dict"])
            ema_9995.shadow = ck["ema_09995_state_dict"]
            ema_9999.shadow = ck["ema_09999_state_dict"]
            opt.load_state_dict(ck["optimizer_state_dict"])
            step_state["step"] = int(ck.get("step", 0))
            step_state["seen"] = int(ck.get("seen", 0))
            print(f"[Job B Resume OK] step={step_state['step']:,}, seen={step_state['seen']:,}", flush=True)
        except Exception as e:
            print(f"[Job B Resume Note] Starting fresh ({e})", flush=True)

    stage1_pos = min(20_000_000, args.total_positions // 5 if args.total_positions < 20_000_000 else 20_000_000)
    tot_steps_est = max(1, (args.total_positions + args.batch_size - 1) // args.batch_size)
    stage1_steps = max(1, stage1_pos // args.batch_size)

    def _lr_b(o, stp, seen):
        if seen < stage1_pos:
            lr_new = compute_cosine_lr(stp, stage1_steps, min(500, stage1_steps // 10), 2e-4, 1e-4)
            lr_base = 2e-5
        else:
            lr_c = compute_cosine_lr(stp - stage1_steps, max(1, tot_steps_est - stage1_steps), 0, 1e-4, 5e-6)
            lr_new = lr_c
            lr_base = lr_c
        o.param_groups[0]["lr"] = lr_new
        o.param_groups[1]["lr"] = lr_base

    def _save_s3_ckpt(st):
        ck_p = ckpt_dir / "job_b_latest.pt"
        torch.save({
            "step": st["step"], "seen": st["seen"],
            "model_state_dict": m12.state_dict(),
            "ema_09995_state_dict": ema_9995.shadow,
            "ema_09999_state_dict": ema_9999.shadow,
            "optimizer_state_dict": opt.state_dict(),
        }, ck_p)
        if s3_client and args.bucket:
            try:
                s3_client.upload_file(str(ck_p), args.bucket, s3_resume_key)
                print(f"  [Checkpoint] Uploaded {s3_resume_key} at seen={st['seen']:,}", flush=True)
            except Exception as e:
                print(f"  [Checkpoint Warning] {e}", flush=True)

    # Async rolling 5M mixed pool producer (~70% fresh parallel Lc0 T80 + ~30% non-resetting Stockfish stream)
    t80_urls = list_t80_nonempty_urls(seed=args.seed)
    t80_idx = [(step_state["seen"] // 3_500_000) % max(1, len(t80_urls))]
    pool_q: queue.Queue = queue.Queue(maxsize=3)
    sf_q: queue.Queue = queue.Queue(maxsize=3)
    stop_prod = threading.Event()
    v2_b, v2_s, v2_n = convert_v2_shards_to_unified(dirs["v2_shards"], tmp_dir / "job_b_v2_sf")
    sf_ring: list[tuple[np.ndarray, np.ndarray]] = []
    if v2_n > 0 and v2_b.is_file():
        sf_ring.append(load_unified_chunk(v2_b, v2_s))
        if step_state["seen"] == 0:
            sf_q.put(sf_ring[0])

    def _sf_streamer():
        sf_stream = LichessContinuousStream(sf_exe, excl_hashes, excl_bin, initial_skip=1_000_000, threads=1)
        sf_idx = 0
        sf_dir = tmp_dir / "job_b_sf_stream"
        sf_dir.mkdir(parents=True, exist_ok=True)
        try:
            while not stop_prod.is_set():
                try:
                    bp, sp, cnt = sf_stream.read_chunk(sf_dir / f"sf_{sf_idx:04d}", 1_500_000)
                    if cnt > 0 and bp.is_file():
                        v, s = load_unified_chunk(bp, sp)
                        bp.unlink(missing_ok=True)
                        sp.unlink(missing_ok=True)
                        while not stop_prod.is_set():
                            try:
                                sf_q.put((v, s), timeout=2.0)
                                break
                            except queue.Full:
                                continue
                        sf_idx += 1
                    else:
                        sf_stream.close()
                        time.sleep(2.0)
                except Exception as exc:
                    print(f"[Job B SF Stream Note] Reopening Lichess stream ({exc})", flush=True)
                    sf_stream.close()
                    time.sleep(3.0)
        finally:
            sf_stream.close()
            shutil.rmtree(sf_dir, ignore_errors=True)

    def _producer():
        p_id = 0
        while not stop_prod.is_set() and (p_id * 5_000_000) < (args.total_positions + 5_000_000):
            p_dir = tmp_dir / f"job_b_pool_{p_id:03d}"
            p_dir.mkdir(parents=True, exist_ok=True)
            lc0_pairs, next_idx = stream_lc0_t80_parallel(
                lc0_exe, t80_urls, t80_idx[0], p_dir, excl_bin, 3_500_000, workers=2, threads_per_worker=1, tag="Job B Lc0"
            )
            t80_idx[0] = next_idx
            lc0_v1s, lc0_scs = [], []
            for bp, sp, cnt in lc0_pairs:
                if cnt > 0 and bp.is_file():
                    v, s = load_unified_chunk(bp, sp)
                    lc0_v1s.append(v); lc0_scs.append(s)

            v1_sf, sc_sf = np.empty((0, RECORD_SIZE_V1), dtype=np.uint8), np.empty((0, SIDECAR_SIZE), dtype=np.uint8)
            try:
                v1_sf, sc_sf = sf_q.get(timeout=5.0 if sf_ring else 120.0)
                if len(sf_ring) >= 6:
                    sf_ring.pop(0)
                sf_ring.append((v1_sf, sc_sf))
            except queue.Empty:
                if sf_ring:
                    v1_sf, sc_sf = sf_ring[p_id % len(sf_ring)]

            v1_lc = np.concatenate(lc0_v1s, axis=0) if lc0_v1s else np.empty((0, RECORD_SIZE_V1), dtype=np.uint8)
            sc_lc = np.concatenate(lc0_scs, axis=0) if lc0_scs else np.empty((0, SIDECAR_SIZE), dtype=np.uint8)
            mv1, msc = _build_mixed_epoch_pool(v1_sf, sc_sf, v1_lc, sc_lc, seed=args.seed + p_id)
            shutil.rmtree(p_dir, ignore_errors=True)
            while not stop_prod.is_set():
                try:
                    pool_q.put((mv1, msc), timeout=2.0)
                    break
                except queue.Full:
                    continue
            p_id += 1
        pool_q.put(None)

    sf_th = threading.Thread(target=_sf_streamer, daemon=True)
    sf_th.start()
    prod_th = threading.Thread(target=_producer, daemon=True)
    prod_th.start()

    # Reserve 7 min before hard MaxRuntimeInSeconds for guaranteed 150K GPU eval + IR export + S3 upload
    stop_deadline = time.time() + max(600, args.max_runtime_sec - 420)
    calib_inputs = None

    print(f"\n=== [Job B] Training 12x192_mid2_9_hl64 + MLH (~{args.total_positions:,} pos) ===", flush=True)
    while step_state["seen"] < args.total_positions:
        item = pool_q.get()
        if item is None:
            break
        mv1, msc = item
        if calib_inputs is None and len(mv1) >= 500:
            cx = unpack_v1_tensor_to_inputs_and_meta(torch.from_numpy(mv1[:500].copy()))[0].numpy().astype(np.float32)
            calib_inputs = [cx[i : i + 1] for i in range(len(cx))]
        hit_deadline = _train_on_pool(
            m12, mv1, msc, device, opt,
            batch_size=args.batch_size, freeze_trunk=False,
            pol_weight=1.0, val_weight=1.0, mlh_weight=0.05,
            ema_list=[ema_9995, ema_9999], step_state=step_state, lr_fn=_lr_b,
            max_positions=args.total_positions, s3_ckpt_fn=_save_s3_ckpt,
            stop_deadline_ts=stop_deadline,
        )
        del mv1, msc
        gc.collect()
        if hit_deadline:
            print("[Job B] Reached graceful 7-minute shutdown guard; proceeding to 150K eval + export!", flush=True)
            break

    stop_prod.set()
    _save_s3_ckpt(step_state)

    eval_logs = {}
    for lbl, shadow in [("ema_09995", ema_9995.shadow), ("ema_09999", ema_9999.shadow)]:
        m12.load_state_dict(shadow)
        ev = evaluate_unified_150k(m12, eval_files, device, ref_baseline_pol=ref_pol)
        eval_logs[lbl] = ev
        print(f"[Job B {lbl} 150K Eval] {json.dumps(ev)}", flush=True)
        out_sub = model_dir / lbl
        int8_rng = (2, 3, 4, 5, 6, 7, 8, 9) if lbl == "ema_09999" else None
        export_phase4_to_folded_npz_and_ov(
            m12, out_sub, f"12x192_mid2_9_hl64_{lbl}",
            calib_inputs=calib_inputs, int8_blocks=int8_rng,
        )

    final_pt = model_dir / "12x192_mid2_9_hl64_final.pt"
    torch.save({
        "architecture": "12x192_mid2_9_hl64_mlh",
        "step": step_state["step"],
        "seen": step_state["seen"],
        "step0_policy_max_diff": step0_diff,
        "ema_09995_state_dict": ema_9995.shadow,
        "ema_09999_state_dict": ema_9999.shadow,
        "eval_150k": eval_logs,
    }, final_pt)
    rep_b = {"mode": "job_b", "step0_diff": step0_diff, "seen": step_state["seen"], "eval_150k": eval_logs}
    (model_dir / "phase4_job_b_report.json").write_text(json.dumps(rep_b, indent=2), encoding="utf-8")

    if s3_client and args.bucket:
        for p in model_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(model_dir).as_posix()
                s3_client.upload_file(str(p), args.bucket, f"checkpoints/phase4_job_b/{rel}")
                print(f"  [S3 Upload] checkpoints/phase4_job_b/{rel}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["job_a", "job_b"], required=True)
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--total_positions", type=int, default=40_000_000)
    parser.add_argument("--max_runtime_sec", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True)
    args = parser.parse_args()
    _ensure_runtime_deps()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    ckpt_dir = Path("/opt/ml/checkpoints") if os.path.exists("/opt/ml") else (model_dir / "checkpoints")
    tmp_dir = Path("/opt/ml/phase4_tmp") if os.path.exists("/opt/ml") else Path("/tmp/phase4_work")
    for d in (model_dir, ckpt_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    import boto3
    s3_client = boto3.client("s3", region_name=args.region) if args.bucket else None
    search_dirs = [data_dir, data_dir / "shards", Path("/opt/ml/code"), Path.cwd(), tmp_dir]

    eval_files = {
        "far_test_50k": _download_or_locate(search_dirs, "test_far_50k.bin", s3_client, args.bucket, "data/test_far_50k.bin", tmp_dir / "test_far_50k.bin"),
        "in_region_50k": _download_or_locate(search_dirs, "test_in_region_50k.bin", s3_client, args.bucket, "data/test_in_region_50k.bin", tmp_dir / "test_in_region_50k.bin"),
        "heldout_400m_50k": _download_or_locate(search_dirs, "test_heldout_400m_50k.bin", s3_client, args.bucket, "data/test_heldout_400m_50k.bin", tmp_dir / "test_heldout_400m_50k.bin"),
    }
    excl_bin = build_combined_exclude_bin(list(eval_files.values()), tmp_dir / "eval_exclude_150k.bin")
    excl_hashes = _download_or_locate(search_dirs, "eval_disjoint_hashes_v2.json", s3_client, args.bucket, "deploy/eval_disjoint_hashes_v2.json", tmp_dir / "eval_disjoint_hashes_v2.json")

    v2_shards = []
    for sfn in ("shard_000.bin", "shard_001.bin"):
        v2_shards.append(_download_or_locate(search_dirs, sfn, s3_client, args.bucket, f"data/shards/{sfn}", tmp_dir / f"v2_{sfn}"))

    dirs = {
        "data": data_dir, "model": model_dir, "ckpt": ckpt_dir, "tmp": tmp_dir,
        "search": search_dirs, "eval_files": eval_files, "excl_bin": excl_bin,
        "excl_hashes": excl_hashes, "v2_shards": v2_shards,
    }

    if args.mode == "job_a":
        run_job_a(args, device, s3_client, dirs)
    else:
        run_job_b(args, device, s3_client, dirs)


if __name__ == "__main__":
    main()
