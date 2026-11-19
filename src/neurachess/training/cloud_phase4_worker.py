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
import io
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
    load_folded_npz_into_phase4,
    transplant_hl_mlh_heads,
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
    """Fused multi-tensor EMA tracker using `torch._foreach_mul_` / `torch._foreach_add_`."""

    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.decay = float(decay)
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.shadow = {k: v.detach().clone() for k, v in raw.state_dict().items()}
        self._fp_shadow: list[torch.Tensor] = []
        self._fp_src: list[torch.Tensor] = []
        self._int_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._rebuild_lists(raw)

    def _rebuild_lists(self, model: nn.Module) -> None:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        live: dict[str, torch.Tensor] = {}
        for k, p in raw.named_parameters():
            live[k] = p
        for k, b in raw.named_buffers():
            live[k] = b
        self._fp_shadow = []
        self._fp_src = []
        self._int_pairs = []
        for k, src in live.items():
            sh = self.shadow.get(k)
            if sh is None or sh.shape != src.shape or sh.device != src.device:
                sh = src.detach().clone()
                self.shadow[k] = sh
            if src.is_floating_point():
                self._fp_shadow.append(sh)
                self._fp_src.append(src)
            else:
                self._int_pairs.append((sh, src))

    def load_shadow_dict(self, state_dict: dict[str, torch.Tensor], model: nn.Module) -> None:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        dev = next(raw.parameters()).device
        self.shadow = {k: v.detach().to(device=dev).clone() for k, v in state_dict.items()}
        self._rebuild_lists(raw)

    def update(self, model: nn.Module) -> None:
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        with torch.no_grad():
            if not self._fp_shadow:
                self._rebuild_lists(raw)
            if self._fp_shadow:
                srcs = [t.detach() for t in self._fp_src]
                torch._foreach_mul_(self._fp_shadow, self.decay)
                torch._foreach_add_(self._fp_shadow, srcs, alpha=1.0 - self.decay)
            for sh, src in self._int_pairs:
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


def list_t80_nonempty_urls(min_bytes: int = 50_000_000, seed: int = 42, skip_first: int = 0) -> list[str]:
    """Fetch `storage.lczero.org/files/training_data/test80/` HTML and filter non-empty `.tar` archives."""
    html = urllib.request.urlopen(urllib.request.Request(LC0_T80_URL, headers=UA_HEADER), timeout=30).read().decode("utf-8", "ignore")
    pat = re.compile(r'href="(training-run1-test80-[0-9]+-[0-9]+\.tar)"[^>]*>[^<]+</a>\s+\S+\s+\S+\s+(\d+)')
    urls = []
    for fname, sz_s in pat.findall(html):
        if int(sz_s) >= min_bytes:
            urls.append(LC0_T80_URL + fname)
    rng = random.Random(seed)
    rng.shuffle(urls)
    if skip_first > 0 and skip_first < len(urls):
        urls = urls[skip_first:]
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
        if workers >= 2:
            batch_n = min(workers, max(2, (rem + 999_999) // 1_000_000)) if rem > 400_000 else 1
        else:
            batch_n = 1
        jobs = []
        for bi in range(batch_n):
            url = tar_urls[(idx + bi) % n_urls]
            jobs.append((wave * workers + bi, url))
        idx = (idx + batch_n) % n_urls
        wave += 1

        per_cap = max(250_000, rem + 100_000) if batch_n == 1 else max(600_000, (rem + batch_n - 1) // batch_n + 350_000)
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


def _find_zstd_frame_offset(url: str, start_byte: int, probe_mb: int = 16) -> int:
    """Locate the first valid ZSTD frame header at or after `start_byte` via HTTP Range probe."""
    import zstandard as zstd

    magic = b"\x28\xb5\x2f\xfd"
    probe_bytes = max(4, int(probe_mb)) * 1024 * 1024
    req = urllib.request.Request(
        url,
        headers={**UA_HEADER, "Range": f"bytes={start_byte}-{start_byte + probe_bytes - 1}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    pos = 0
    while True:
        idx = data.find(magic, pos)
        if idx == -1:
            break
        try:
            zr = zstd.ZstdDecompressor().stream_reader(io.BytesIO(data[idx:]))
            sample = zr.read(1024)
            if b"fen" in sample:
                return start_byte + idx
        except Exception:
            pass
        pos = idx + 4
    raise RuntimeError(f"No valid ZSTD frame found within {probe_mb} MB of byte offset {start_byte:,}")


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
        worker_stride: int = 1,
        worker_rem: int = 0,
        byte_offset: int = 0,
    ):
        self.sf_exe = sf_exe
        self.exclude_hash_file = exclude_hash_file
        self.exclude_bin = exclude_bin
        self.initial_skip = int(initial_skip)
        self.threads = int(threads)
        self.opener = opener
        self.worker_stride = max(1, int(worker_stride))
        self.worker_rem = int(worker_rem) % self.worker_stride
        self.byte_offset = max(0, int(byte_offset))
        self._resp = None
        self._zr = None
        self._byte_buf = b""
        self._py_line_idx = 0
        self._chunk_seq = 0

    def _open_stream(self) -> None:
        import zstandard as zstd

        if self._zr is not None:
            return
        if self.opener is not None:
            self._resp = self.opener(LICHESS_EVAL_URL)
        elif self.byte_offset > 0:
            frame_abs = _find_zstd_frame_offset(LICHESS_EVAL_URL, self.byte_offset, probe_mb=16)
            self.initial_skip = max(1, self.initial_skip)
            req = urllib.request.Request(LICHESS_EVAL_URL, headers={**UA_HEADER, "Range": f"bytes={frame_abs}-"})
            self._resp = urllib.request.urlopen(req, timeout=120)
            print(f"[LichessStream] Instant HTTP Range ZSTD seek -> byte {frame_abs:,} (requested {self.byte_offset:,})", flush=True)
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
        self._py_line_idx = 0

    def _filter_lines_block(self, lines_blk: bytes) -> bytes:
        if not lines_blk:
            return b""
        raw_Part = lines_blk[:-1] if lines_blk.endswith(b"\n") else lines_blk
        cur = self._py_line_idx
        # Fast C-level newline counting while skipping large line offsets (e.g., 40M lines)
        if cur < self.initial_skip:
            n_fast = raw_Part.count(b"\n") + 1
            if cur + n_fast <= self.initial_skip:
                self._py_line_idx = cur + n_fast
                return b""
        lines = raw_Part.split(b"\n")
        n_lines = len(lines)
        self._py_line_idx = cur + n_lines
        if cur + n_lines <= self.initial_skip:
            return b""
        start_off = max(0, self.initial_skip - cur)
        rem = lines[start_off:]
        if self.worker_stride > 1:
            rel0 = (cur + start_off - self.initial_skip) % self.worker_stride
            first_k = (self.worker_rem - rel0) % self.worker_stride
            rem = rem[first_k :: self.worker_stride]
        if not rem:
            return b""
        return b"\n".join(rem) + b"\n"

    def read_chunk(self, out_prefix: Path, target_records: int) -> tuple[Path, Path, int]:
        cmd = [
            str(self.sf_exe),
            "--out-prefix",
            str(out_prefix),
            "--skip-lines",
            "0",
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

        stream_err = None
        while proc.poll() is None:
            assert self._zr is not None
            try:
                raw = self._zr.read(1024 * 1024)
            except Exception as exc:
                stream_err = exc
                self.close()
                break
            if not raw:
                if self._byte_buf:
                    tail = self._byte_buf + (b"" if self._byte_buf.endswith(b"\n") else b"\n")
                    self._byte_buf = b""
                    filtered = self._filter_lines_block(tail)
                    if filtered:
                        try:
                            proc.stdin.write(filtered)
                            proc.stdin.flush()
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
            filtered = self._filter_lines_block(lines_blk)
            if not filtered:
                continue
            try:
                proc.stdin.write(filtered)
                proc.stdin.flush()
            except BrokenPipeError:
                break

        self._chunk_seq += 1
        try:
            proc.stdin.close()
        except Exception:
            pass
        out, _ = proc.communicate(timeout=300)
        lines = out.decode("utf-8", "ignore").strip().splitlines()
        stats = json.loads(lines[-1]) if lines else {}
        n_wr = int(stats.get("records_written", 0))
        note = f" (stream interrupted: {stream_err})" if stream_err is not None else ""
        print(f"[LichessStream] chunk #{self._chunk_seq}: wrote {n_wr:,} records (total_skipped={stats.get('skipped_overlap', 0):,}){note}", flush=True)
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
    cps_dev = cps.to(device=device, non_blocking=True)
    hl_targets = compute_hl_gauss_targets(cps_dev, raw_m.hl_edges, raw_m.hl_sigma)
    v_target = torch.tanh(cps_dev.float() * CP_TANH_SCALE).clamp(-1.0, 1.0)

    optimizer.zero_grad(set_to_none=True)
    amp_ok = device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_ok):
        if freeze_trunk:
            with torch.no_grad():
                feat = raw_m.forward_trunk(inp)
            pol_logits, hl_logits, val_exp, mlh_pred = raw_m.forward_policy_and_heads(feat)
        else:
            pol_logits, hl_logits, val_exp, mlh_pred = model(inp)

        l_hl_ce = -(hl_targets * F.log_softmax(hl_logits.float(), dim=-1)).sum(dim=-1).mean()
        l_val_mse = F.mse_loss(val_exp.squeeze(-1).float(), v_target)
        l_hl = l_hl_ce + 2.0 * l_val_mse
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
        if device.type == "cuda":
            v1_t = torch.from_numpy(v1_np[start:end].copy()).pin_memory().to(device, non_blocking=True)
            sc_t = torch.from_numpy(sc_np[start:end].copy()).pin_memory().to(device, non_blocking=True)
        else:
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

    # Fast-path: consume pre-harvested S3 mix shards first (~1.5 GB/s), streaming only any remainder
    s3_pref = getattr(args, "s3_shard_prefix", "data/phase4_mix_shards").strip("/")
    ready_s3_keys = list_ready_s3_mix_shards(s3_client, args.bucket, s3_pref) if (s3_client and args.bucket) else []
    s3_v1_parts: list[np.ndarray] = []
    s3_sc_parts: list[np.ndarray] = []
    s3_got = 0
    if ready_s3_keys:
        rng_a = random.Random(args.seed + 7)
        pick_keys = list(ready_s3_keys)
        rng_a.shuffle(pick_keys)
        dl_a = tmp_dir / "job_a_s3_shards"
        for bkey in pick_keys:
            if s3_got >= args.total_positions:
                break
            try:
                vv, ss = download_s3_mix_shard(s3_client, args.bucket, bkey, dl_a)
                take = min(len(vv), args.total_positions - s3_got)
                s3_v1_parts.append(vv[:take])
                s3_sc_parts.append(ss[:take])
                s3_got += take
            except Exception as exc:
                print(f"[Job A S3 Shard Warning] {bkey}: {exc}", flush=True)
        shutil.rmtree(dl_a, ignore_errors=True)
        print(f"[Job A S3 Fast-Path] Loaded {s3_got:,}/{args.total_positions:,} positions from {len(s3_v1_parts)} S3 shards!", flush=True)

    rem_total = max(0, args.total_positions - s3_got)
    cache_dir = tmp_dir / "job_a_40m_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sf_target = rem_total // 2
    lc0_target = rem_total - sf_target

    sf_pairs: list[tuple[Path, Path, int]] = []
    lc0_pairs: list[tuple[Path, Path, int]] = []
    if rem_total > 0:
        v2_b, v2_s, v2_n = convert_v2_shards_to_unified(dirs["v2_shards"], cache_dir / "v2_sf")
        if v2_n > 0:
            sf_pairs.append((v2_b, v2_s, v2_n))
        rem_sf = max(0, sf_target - v2_n)
        t80_urls = list_t80_nonempty_urls(seed=args.seed)

        sf_fut = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as data_ex:
            if rem_sf > 0:
                sf_fut = data_ex.submit(stream_lichess_sf_chunks, sf_exe, cache_dir, excl_hashes, excl_bin, rem_sf, 1_000_000, 2)
            if lc0_target > 0:
                lc0_pairs, _ = stream_lc0_t80_parallel(
                    lc0_exe, t80_urls, 0, cache_dir, excl_bin, lc0_target, workers=4, threads_per_worker=1, tag="Job A Data"
                )
            if sf_fut is not None:
                sf_pairs.extend(sf_fut.result())

    # Low-RAM chunk-by-chunk scattered shuffle (~7GB peak instead of ~24GB concat+permute copies)
    sf_chunks, lc0_chunks = [], []
    sf_keep = 0
    for bp, sp, cnt in sf_pairs:
        if cnt > 0 and bp.is_file() and sp.is_file() and sf_keep < sf_target:
            avail = min(bp.stat().st_size // RECORD_SIZE_V1, sp.stat().st_size // SIDECAR_SIZE, sf_target - sf_keep)
            if avail > 0:
                sf_chunks.append((bp, sp, avail))
                sf_keep += avail
    lc0_keep = 0
    for bp, sp, cnt in lc0_pairs:
        if cnt > 0 and bp.is_file() and sp.is_file() and lc0_keep < lc0_target:
            avail = min(bp.stat().st_size // RECORD_SIZE_V1, sp.stat().st_size // SIDECAR_SIZE, lc0_target - lc0_keep)
            if avail > 0:
                lc0_chunks.append((bp, sp, avail))
                lc0_keep += avail

    n_pool = s3_got + sf_keep + lc0_keep
    mix_v1 = np.empty((n_pool, RECORD_SIZE_V1), dtype=np.uint8)
    mix_sc = np.empty((n_pool, SIDECAR_SIZE), dtype=np.uint8)
    perm = np.random.default_rng(args.seed).permutation(n_pool)
    off = 0
    for vv, ss in zip(s3_v1_parts, s3_sc_parts):
        k = len(vv)
        dst_idx = perm[off : off + k]
        mix_v1[dst_idx] = vv
        mix_sc[dst_idx] = ss
        off += k
    s3_v1_parts.clear()
    s3_sc_parts.clear()
    for bp, sp, take in sf_chunks + lc0_chunks:
        v, s = load_unified_chunk(bp, sp)
        k = min(take, len(v))
        dst_idx = perm[off : off + k]
        mix_v1[dst_idx] = v[:k]
        mix_sc[dst_idx] = s[:k]
        off += k
        del v, s
        bp.unlink(missing_ok=True)
        sp.unlink(missing_ok=True)
    del perm
    gc.collect()
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"[Job A Ready] Blended pool size: {n_pool:,} positions (s3={s3_got:,}, sf={sf_keep:,}, lc0={lc0_keep:,})", flush=True)

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


def list_ready_s3_mix_shards(s3_client, bucket: str, prefix: str = "data/phase4_mix_shards") -> list[str]:
    """Return sorted base S3 keys (`<prefix>/shard_XXX`) that have both `.bin` and `.sidecar68.bin` uploaded."""
    if s3_client is None or not bucket:
        return []
    norm_pref = prefix.strip("/") + "/"
    bins: set[str] = set()
    scs: set[str] = set()
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=norm_pref):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                sz = int(obj.get("Size", 0))
                if k.endswith(".sidecar68.bin") and sz >= SIDECAR_SIZE * 1000:
                    scs.add(k[: -len(".sidecar68.bin")])
                elif k.endswith(".bin") and not k.endswith(".sidecar68.bin") and sz >= RECORD_SIZE_V1 * 1000:
                    bins.add(k[: -len(".bin")])
    except Exception as exc:
        print(f"[S3 Shards Note] list_objects_v2 ({exc})", flush=True)
        return []
    return sorted(bins & scs)


def download_s3_mix_shard(s3_client, bucket: str, base_key: str, dst_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download a pre-packed `(107B .bin, 68B .sidecar68.bin)` shard pair in parallel over S3."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(base_key).name
    bin_p = dst_dir / f"{stem}.bin"
    sc_p = dst_dir / f"{stem}.sidecar68.bin"
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(s3_client.download_file, bucket, f"{base_key}.bin", str(bin_p))
        f2 = ex.submit(s3_client.download_file, bucket, f"{base_key}.sidecar68.bin", str(sc_p))
        f1.result()
        f2.result()
    v1, sc = load_unified_chunk(bin_p, sc_p)
    bin_p.unlink(missing_ok=True)
    sc_p.unlink(missing_ok=True)
    return v1, sc


def run_harvest_shards(args: argparse.Namespace, s3_client, dirs: dict) -> None:
    """High-throughput CPU Shard Harvester (`ml.c5.4xlarge` 16 vCPUs): streams 50% Lc0 T80 + 50% SF into S3."""
    tmp_dir: Path = dirs["tmp"]
    lc0_exe, sf_exe = ensure_cpp_converters(tmp_dir)
    excl_bin = dirs["excl_bin"]
    excl_hashes = dirs["excl_hashes"]

    shard_sz = max(10_000, int(args.shard_size))
    tot_shards = max(1, (args.total_positions + shard_sz - 1) // shard_sz)
    w_id = int(args.worker_id)
    n_workers = max(1, int(args.num_workers))
    prefix = args.s3_shard_prefix.strip("/")
    my_shard_ids = [sid for sid in range(tot_shards) if (sid % n_workers) == w_id]

    existing = set(list_ready_s3_mix_shards(s3_client, args.bucket, prefix)) if args.resume else set()
    todo_ids = [sid for sid in my_shard_ids if f"{prefix}/shard_{sid:03d}" not in existing]

    sf_ratio = float(getattr(args, "sf_ratio", 0.50))
    sf_per_shard = int(round(shard_sz * sf_ratio))
    lc0_per_shard = shard_sz - sf_per_shard
    t80_skip = max(0, int(getattr(args, "t80_skip", 40)))
    sf_base_offset = max(0, int(getattr(args, "sf_line_offset", 40_000_000)))

    print(
        f"=== [Harvester w={w_id}/{n_workers}] Assigned {len(my_shard_ids)} shards "
        f"({len(todo_ids)} remaining, shard_size={shard_sz:,}, "
        f"mix={int(round((1.0 - sf_ratio) * 100))}% Lc0 / {int(round(sf_ratio * 100))}% SF, "
        f"t80_skip={t80_skip}, sf_offset={sf_base_offset:,}) -> s3://{args.bucket}/{prefix}/ ===",
        flush=True,
    )
    if not todo_ids:
        return

    all_t80 = list_t80_nonempty_urls(seed=args.seed, skip_first=t80_skip)
    my_t80 = all_t80[w_id::n_workers] if all_t80 else []
    t80_idx = (len(my_shard_ids) - len(todo_ids)) % max(1, len(my_t80))

    stop_ev = threading.Event()
    sf_q: queue.Queue = queue.Queue(maxsize=6)
    sf_ring: list[tuple[np.ndarray, np.ndarray]] = []
    v2_seed_v1 = np.empty((0, RECORD_SIZE_V1), dtype=np.uint8)
    v2_seed_sc = np.empty((0, SIDECAR_SIZE), dtype=np.uint8)

    if dirs["v2_shards"] and len(todo_ids) == len(my_shard_ids):
        v2_b, v2_s, v2_n = convert_v2_shards_to_unified(dirs["v2_shards"], tmp_dir / f"harv_v2_sf_w{w_id}")
        if v2_n > 0 and v2_b.is_file():
            v_init, s_init = load_unified_chunk(v2_b, v2_s)
            v2_b.unlink(missing_ok=True)
            v2_s.unlink(missing_ok=True)
            if len(v_init) >= sf_per_shard:
                for off in range(0, len(v_init), sf_per_shard):
                    sub_v = v_init[off : off + sf_per_shard]
                    sub_s = s_init[off : off + sf_per_shard]
                    if len(sub_v) == sf_per_shard:
                        sf_ring.append((sub_v, sub_s))
                        if not sf_q.full():
                            sf_q.put((sub_v, sub_s))
                    elif len(sub_v) > 0:
                        v2_seed_v1, v2_seed_sc = sub_v, sub_s
            else:
                # Top up partial V2 seed (e.g., 500K) with streamed SF so shard 0 is an exact 50/50 2M shard
                v2_seed_v1, v2_seed_sc = v_init, s_init
                print(f"[Harvester w={w_id}] Loaded {len(v_init):,} V2 SF seed records (will top up to {sf_per_shard:,} from Lichess stream)", flush=True)

    streams_per_worker = 2
    base_byte_offset = max(3_000_000_000, int(sf_base_offset * 75))
    region_span_bytes = 3_500_000_000
    done_before = len(my_shard_ids) - len(todo_ids)

    def _harv_sf_worker(stream_idx: int):
        g_idx = w_id * streams_per_worker + stream_idx
        start_byte = base_byte_offset + g_idx * region_span_bytes + (done_before // streams_per_worker) * int(sf_per_shard * 75)
        st = LichessContinuousStream(
            sf_exe, excl_hashes, excl_bin,
            initial_skip=1, threads=2,
            worker_stride=1, worker_rem=0,
            byte_offset=start_byte,
        )
        sf_seq = 0
        sf_streamed_total = 0
        rem_v2_v, rem_v2_s = (v2_seed_v1, v2_seed_sc) if stream_idx == 0 else (
            np.empty((0, RECORD_SIZE_V1), dtype=np.uint8),
            np.empty((0, SIDECAR_SIZE), dtype=np.uint8),
        )
        sf_dir = tmp_dir / f"harv_sf_w{w_id}_s{stream_idx}"
        sf_dir.mkdir(parents=True, exist_ok=True)
        try:
            while not stop_ev.is_set():
                try:
                    need_sf = sf_per_shard - len(rem_v2_v) if len(rem_v2_v) > 0 else sf_per_shard
                    bp, sp, cnt = st.read_chunk(sf_dir / f"sf_{sf_seq:04d}", need_sf)
                    if cnt > 0 and bp.is_file():
                        v, s = load_unified_chunk(bp, sp)
                        bp.unlink(missing_ok=True)
                        sp.unlink(missing_ok=True)
                        sf_streamed_total += len(v)
                        if len(rem_v2_v) > 0:
                            v = np.concatenate([rem_v2_v, v], axis=0)[:sf_per_shard]
                            s = np.concatenate([rem_v2_s, s], axis=0)[:sf_per_shard]
                            rem_v2_v = np.empty((0, RECORD_SIZE_V1), dtype=np.uint8)
                            rem_v2_s = np.empty((0, SIDECAR_SIZE), dtype=np.uint8)
                        while not stop_ev.is_set():
                            try:
                                sf_q.put((v, s), timeout=2.0)
                                break
                            except queue.Full:
                                continue
                        sf_seq += 1
                    else:
                        st.close()
                        st.byte_offset = start_byte + int(sf_streamed_total * 75)
                        time.sleep(2.0)
                except Exception as exc:
                    print(f"[Harvester SF w={w_id} s={stream_idx}] Stream reopen ({exc})", flush=True)
                    st.close()
                    st.byte_offset = start_byte + int(sf_streamed_total * 75)
                    time.sleep(3.0)
        finally:
            st.close()
            shutil.rmtree(sf_dir, ignore_errors=True)

    sf_threads = [threading.Thread(target=_harv_sf_worker, args=(si,), daemon=True) for si in range(streams_per_worker)]
    for th in sf_threads:
        th.start()

    stop_deadline = time.time() + max(300, args.max_runtime_sec - 120)
    lc0_buf_v1 = np.empty((0, RECORD_SIZE_V1), dtype=np.uint8)
    lc0_buf_sc = np.empty((0, SIDECAR_SIZE), dtype=np.uint8)
    cpu_cnt = os.cpu_count() or 8
    par_t80 = max(4, min(8, cpu_cnt - 6))
    t0_all = time.time()
    done_cnt = 0

    try:
        for sid in todo_ids:
            if time.time() >= stop_deadline:
                print(f"[Harvester w={w_id}] Hit graceful deadline before shard_{sid:03d}; exiting cleanly.", flush=True)
                break

            while len(lc0_buf_v1) < lc0_per_shard and my_t80:
                w_dir = tmp_dir / f"harv_lc0_w{w_id}_{sid:03d}"
                rem_shards = max(1, len(todo_ids) - done_cnt)
                need = min(rem_shards * lc0_per_shard, max(lc0_per_shard * 6, lc0_per_shard - len(lc0_buf_v1)))
                pairs, t80_idx = stream_lc0_t80_parallel(
                    lc0_exe, my_t80, t80_idx, w_dir, excl_bin, need,
                    workers=par_t80, threads_per_worker=1, tag=f"Harv-W{w_id}",
                )
                v_add, s_add = [lc0_buf_v1], [lc0_buf_sc]
                for bp, sp, cnt in pairs:
                    if cnt > 0 and bp.is_file():
                        vv, ss = load_unified_chunk(bp, sp)
                        v_add.append(vv)
                        s_add.append(ss)
                shutil.rmtree(w_dir, ignore_errors=True)
                lc0_buf_v1 = np.concatenate(v_add, axis=0)
                lc0_buf_sc = np.concatenate(s_add, axis=0)

            take_lc = min(lc0_per_shard, len(lc0_buf_v1))
            v1_lc, sc_lc = lc0_buf_v1[:take_lc], lc0_buf_sc[:take_lc]
            lc0_buf_v1, lc0_buf_sc = lc0_buf_v1[take_lc:], lc0_buf_sc[take_lc:]

            v1_sf, sc_sf = np.empty((0, RECORD_SIZE_V1), dtype=np.uint8), np.empty((0, SIDECAR_SIZE), dtype=np.uint8)
            while len(v1_sf) == 0 and time.time() < stop_deadline:
                try:
                    v1_sf, sc_sf = sf_q.get(timeout=90.0 if sf_ring else 180.0)
                    if len(sf_ring) >= 8:
                        sf_ring.pop(0)
                    sf_ring.append((v1_sf, sc_sf))
                except queue.Empty:
                    if sf_ring:
                        print(f"[Harvester w={w_id}] SF queue wait timed out on shard_{sid:03d}; using ring fallback", flush=True)
                        v1_sf, sc_sf = sf_ring[sid % len(sf_ring)]
                        break
                    print(f"[Harvester w={w_id}] Waiting for first SF Range chunk on shard_{sid:03d}...", flush=True)
            if len(v1_sf) == 0:
                print(f"[Harvester w={w_id}] Deadline hit while waiting for SF chunk; skipping incomplete shard_{sid:03d}.", flush=True)
                break

            mv1, msc = _build_mixed_epoch_pool(v1_sf[:sf_per_shard], sc_sf[:sf_per_shard], v1_lc, sc_lc, seed=args.seed + sid)
            out_dir = tmp_dir / f"upload_w{w_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            b_file = out_dir / f"shard_{sid:03d}.bin"
            s_file = out_dir / f"shard_{sid:03d}.sidecar68.bin"
            s_file.write_bytes(msc.tobytes())
            b_file.write_bytes(mv1.tobytes())

            if s3_client and args.bucket:
                # Upload sidecar FIRST and main .bin SECOND so list_ready_s3_mix_shards sees atomic pairs
                s3_client.upload_file(str(s_file), args.bucket, f"{prefix}/shard_{sid:03d}.sidecar68.bin")
                s3_client.upload_file(str(b_file), args.bucket, f"{prefix}/shard_{sid:03d}.bin")
            b_file.unlink(missing_ok=True)
            s_file.unlink(missing_ok=True)
            done_cnt += 1
            rate = (done_cnt * len(mv1)) / max(1.0, time.time() - t0_all)
            print(
                f"[Harvester w={w_id}] Uploaded shard_{sid:03d} ({len(mv1):,} pos, "
                f"sf={min(len(v1_sf), sf_per_shard):,}, lc0={len(v1_lc):,}, "
                f"done={done_cnt}/{len(todo_ids)}, avg={rate:,.0f} pos/s)",
                flush=True,
            )
            del mv1, msc
            gc.collect()
    finally:
        stop_ev.set()


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

    lr_scale = math.sqrt(max(1.0, args.batch_size / 512.0))
    opt = torch.optim.AdamW(
        [{"params": new_params, "lr": 2e-4 * lr_scale}, {"params": base_params, "lr": 2e-5 * lr_scale}],
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
            ema_9995.load_shadow_dict(ck["ema_09995_state_dict"], m12)
            ema_9999.load_shadow_dict(ck["ema_09999_state_dict"], m12)
            opt.load_state_dict(ck["optimizer_state_dict"])
            step_state["step"] = int(ck.get("step", 0))
            step_state["seen"] = int(ck.get("seen", 0))
            print(f"[Job B Resume OK] step={step_state['step']:,}, seen={step_state['seen']:,}", flush=True)
        except Exception as e:
            print(f"[Job B Resume Note] Starting fresh ({e})", flush=True)

    fwd_m12: nn.Module = m12
    if device.type == "cuda" and os.name != "nt":
        try:
            fwd_m12 = torch.compile(m12, mode="default")
            print("[Job B] Enabled torch.compile(mode='default') on CUDA L40S!", flush=True)
        except Exception as exc:
            print(f"[Job B Note] Using eager CUDA ({exc})", flush=True)
            fwd_m12 = m12

    stage1_pos = min(20_000_000, args.total_positions // 5 if args.total_positions < 20_000_000 else 20_000_000)
    tot_steps_est = max(1, (args.total_positions + args.batch_size - 1) // args.batch_size)
    stage1_steps = max(1, stage1_pos // args.batch_size)

    def _lr_b(o, stp, seen):
        if seen < stage1_pos:
            lr_new = compute_cosine_lr(stp, stage1_steps, min(500, stage1_steps // 10), 2e-4 * lr_scale, 1e-4 * lr_scale)
            lr_base = 2e-5 * lr_scale
        else:
            lr_c = compute_cosine_lr(stp - stage1_steps, max(1, tot_steps_est - stage1_steps), 0, 1e-4 * lr_scale, 5e-6)
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

    # S3-first high-bandwidth double-buffered shard prefetcher (with seamless live-stream fallback)
    pool_q: queue.Queue = queue.Queue(maxsize=4)
    stop_prod = threading.Event()
    s3_pref = args.s3_shard_prefix.strip("/")

    def _s3_and_fallback_producer():
        dl_dir = tmp_dir / "job_b_s3_prefetch"
        pass_idx = step_state["seen"] // 60_000_000
        seen_in_pass: set[str] = set()
        t80_urls = None
        t80_idx = 0
        v2_pair = None

        while not stop_prod.is_set():
            ready_keys = list_ready_s3_mix_shards(s3_client, args.bucket, s3_pref)
            unseen = [k for k in ready_keys if k not in seen_in_pass]
            if not unseen and ready_keys:
                seen_in_pass.clear()
                pass_idx += 1
                unseen = list(ready_keys)

            if unseen:
                rng_p = random.Random(args.seed + pass_idx * 1009 + len(seen_in_pass))
                rng_p.shuffle(unseen)
                for bkey in unseen:
                    if stop_prod.is_set():
                        break
                    try:
                        mv1, msc = download_s3_mix_shard(s3_client, args.bucket, bkey, dl_dir)
                        seen_in_pass.add(bkey)
                        while not stop_prod.is_set():
                            try:
                                pool_q.put((mv1, msc), timeout=2.0)
                                break
                            except queue.Full:
                                continue
                    except Exception as exc:
                        print(f"[Job B S3 Prefetch Warning] {bkey}: {exc}", flush=True)
                    # Re-check S3 list periodically during Pass 0 so newly harvested shards join immediately
                    if pass_idx == 0 and (len(seen_in_pass) % 4 == 0):
                        break
                continue

            # Fallback if S3 mix shards are still empty at startup: stream 1 mixed pool directly
            if t80_urls is None:
                t80_urls = list_t80_nonempty_urls(seed=args.seed)
                t80_idx = (step_state["seen"] // 3_500_000) % max(1, len(t80_urls))
                vb, vs, vn = convert_v2_shards_to_unified(dirs["v2_shards"], tmp_dir / "job_b_v2_sf")
                if vn > 0 and vb.is_file():
                    v2_pair = load_unified_chunk(vb, vs)
            p_dir = tmp_dir / "job_b_fallback_stream"
            lc0_pairs, t80_idx = stream_lc0_t80_parallel(
                lc0_exe, t80_urls, t80_idx, p_dir, excl_bin, 2_000_000, workers=3, threads_per_worker=1, tag="Job B Fallback"
            )
            lc0_v1s, lc0_scs = [], []
            for bp, sp, cnt in lc0_pairs:
                if cnt > 0 and bp.is_file():
                    vv, ss = load_unified_chunk(bp, sp)
                    lc0_v1s.append(vv); lc0_scs.append(ss)
            shutil.rmtree(p_dir, ignore_errors=True)
            v1_sf, sc_sf = v2_pair if v2_pair is not None else (np.empty((0, RECORD_SIZE_V1), dtype=np.uint8), np.empty((0, SIDECAR_SIZE), dtype=np.uint8))
            v1_lc = np.concatenate(lc0_v1s, axis=0) if lc0_v1s else np.empty((0, RECORD_SIZE_V1), dtype=np.uint8)
            sc_lc = np.concatenate(lc0_scs, axis=0) if lc0_scs else np.empty((0, SIDECAR_SIZE), dtype=np.uint8)
            mv1, msc = _build_mixed_epoch_pool(v1_sf[:600_000], sc_sf[:600_000], v1_lc, sc_lc, seed=args.seed + pass_idx)
            while not stop_prod.is_set():
                try:
                    pool_q.put((mv1, msc), timeout=2.0)
                    break
                except queue.Full:
                    continue
        pool_q.put(None)

    prod_th = threading.Thread(target=_s3_and_fallback_producer, daemon=True)
    prod_th.start()

    # Reserve 14 min before hard MaxRuntimeInSeconds for Stage-3 20M Middle-2..9 QAT tail + 150K GPU eval + IR export
    qat_tail_pos = min(20_000_000, max(0, args.total_positions // 15))
    fp_target_pos = max(args.batch_size, args.total_positions - qat_tail_pos)
    stop_deadline = time.time() + max(600, args.max_runtime_sec - 840)
    calib_inputs = None
    last_pool_pair: tuple[np.ndarray, np.ndarray] | None = None

    print(f"\n=== [Job B Stage 1+2] Pure FP16/BF16 12x192 + HL-Gauss-64 + MLH ({fp_target_pos:,} pos, B={args.batch_size}) ===", flush=True)
    while step_state["seen"] < fp_target_pos:
        item = pool_q.get()
        if item is None:
            break
        mv1, msc = item
        if calib_inputs is None and len(mv1) >= 500:
            cx = unpack_v1_tensor_to_inputs_and_meta(torch.from_numpy(mv1[:500].copy()))[0].numpy().astype(np.float32)
            calib_inputs = [cx[i : i + 1] for i in range(len(cx))]
        hit_deadline = _train_on_pool(
            fwd_m12, mv1, msc, device, opt,
            batch_size=args.batch_size, freeze_trunk=False,
            pol_weight=1.0, val_weight=1.0, mlh_weight=0.05,
            ema_list=[ema_9995, ema_9999], step_state=step_state, lr_fn=_lr_b,
            max_positions=fp_target_pos, s3_ckpt_fn=_save_s3_ckpt,
            stop_deadline_ts=stop_deadline,
        )
        last_pool_pair = (mv1, msc)
        if hit_deadline:
            print("[Job B] Reached Stage 1+2 graceful deadline; proceeding to Pure FP16 export + Stage 3 Middle-2..9 QAT!", flush=True)
            break

    _save_s3_ckpt(step_state)

    eval_logs = {}
    for lbl, shadow in [("ema_09995", ema_9995.shadow), ("ema_09999", ema_9999.shadow)]:
        m12.load_state_dict(shadow)
        ev = evaluate_unified_150k(m12, eval_files, device, ref_baseline_pol=ref_pol)
        eval_logs[lbl] = ev
        print(f"[Job B {lbl} Pure FP16 150K Eval] {json.dumps(ev)}", flush=True)
        out_sub = model_dir / lbl
        export_phase4_to_folded_npz_and_ov(
            m12, out_sub, f"12x192_mid2_9_hl64_{lbl}",
            calib_inputs=calib_inputs, int8_blocks=(2, 3, 4, 5, 6, 7, 8, 9) if lbl == "ema_09999" else None,
        )
        if s3_client and args.bucket:
            for p in out_sub.glob("*"):
                if p.is_file():
                    s3_client.upload_file(str(p), args.bucket, f"checkpoints/phase4_job_b/{lbl}/{p.name}")

    # Stage 3: True W8A8 Middle-Block (2..9) QAT Polish (~20M positions) cloned from ema_09999
    qat_deadline = time.time() + max(180, args.max_runtime_sec - int(time.time() - step_state["t0"]) - 300)
    if qat_tail_pos > 0 and time.time() < qat_deadline:
        print(f"\n=== [Job B Stage 3] W8A8 Middle 2..9 QAT Polish ({qat_tail_pos:,} pos, LR 1.5e-5 -> 1e-6) ===", flush=True)
        m12_qat = ChessSEResNetPhase4(num_blocks=12, channels=192, use_bn=True).to(device)
        m12_qat.load_state_dict(ema_9999.shadow)
        for p in m12_qat.parameters():
            p.requires_grad_(True)
        apply_middle_w8a8_qat(m12_qat, qat_blocks=(2, 3, 4, 5, 6, 7, 8, 9))
        if device.type == "cuda":
            m12_qat = m12_qat.to(memory_format=torch.channels_last)
        m12_qat.train()
        enforce_frozen_bn_eval(m12_qat)
        if calib_inputs:
            with torch.no_grad():
                cx_t = torch.from_numpy(np.concatenate(calib_inputs[:128], axis=0)).to(
                    device=device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format
                )
                m12_qat(cx_t)

        ema_qat = FastModelEMA(m12_qat, decay=0.9995)
        opt_qat = torch.optim.AdamW(
            [p for p in m12_qat.parameters() if p.requires_grad],
            lr=1.5e-5 * lr_scale, weight_decay=1e-4, fused=(device.type == "cuda"),
        )
        tot_qat_steps = max(1, (qat_tail_pos + args.batch_size - 1) // args.batch_size)
        st_qat = {"step": 0, "seen": 0, "seed": args.seed + 777, "t0": time.time(), "last_ckpt": 1e18}

        def _lr_qat(o, stp, _s):
            lr_q = compute_cosine_lr(stp, tot_qat_steps, min(250, tot_qat_steps // 10), 1.5e-5 * lr_scale, 1e-6)
            for pg in o.param_groups:
                pg["lr"] = lr_q

        while st_qat["seen"] < qat_tail_pos and time.time() < qat_deadline:
            try:
                q_item = pool_q.get(timeout=5.0)
            except queue.Empty:
                q_item = last_pool_pair
            if q_item is None:
                q_item = last_pool_pair
            if q_item is None:
                break
            qv1, qsc = q_item
            hit_q = _train_on_pool(
                m12_qat, qv1, qsc, device, opt_qat,
                batch_size=args.batch_size, freeze_trunk=False,
                pol_weight=1.0, val_weight=1.0, mlh_weight=0.05,
                ema_list=[ema_qat], step_state=st_qat, lr_fn=_lr_qat,
                max_positions=qat_tail_pos, stop_deadline_ts=qat_deadline,
            )
            last_pool_pair = (qv1, qsc)
            if hit_q:
                break

        m12_qat.load_state_dict(ema_qat.shadow)
        ev_qat = evaluate_unified_150k(m12_qat, eval_files, device, ref_baseline_pol=ref_pol)
        eval_logs["qat_mid2_9"] = ev_qat
        print(f"[Job B qat_mid2_9 150K Eval] {json.dumps(ev_qat)}", flush=True)
        out_qat = model_dir / "qat_mid2_9"
        export_phase4_to_folded_npz_and_ov(
            m12_qat, out_qat, "12x192_mid2_9_hl64_qat",
            calib_inputs=calib_inputs, int8_blocks=(2, 3, 4, 5, 6, 7, 8, 9),
        )
        torch.save({"ema_09995_state_dict": ema_qat.shadow, "eval_150k": ev_qat, "seen": st_qat["seen"]}, out_qat / "12x192_mid2_9_hl64_qat.pt")

    stop_prod.set()
    last_pool_pair = None
    gc.collect()

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


def _mean_150k_val_mse(ev: dict) -> float:
    vals = [float(ev[k]["val_mse"]) for k in ("far_test_50k", "in_region_50k", "heldout_400m_50k") if isinstance(ev.get(k), dict) and "val_mse" in ev[k]]
    return float(sum(vals) / len(vals)) if vals else float("inf")


def run_job_c_10x192_mastery(args: argparse.Namespace, device: torch.device, s3_client, dirs: dict) -> None:
    """400M-Position 10x192 Mastery Pipeline (360M unquantized BF16 + 40M Middle-2..7 W8A8 QAT)."""
    tmp_dir: Path = dirs["tmp"]
    model_dir: Path = dirs["model"]
    ckpt_dir: Path = dirs["ckpt"]
    eval_files: dict[str, Path] = dirs["eval_files"]
    t0_job = time.time()

    a2_s3_key = getattr(args, "a2_soup75_s3_key", "checkpoints/phase4_job_c/init/A2_soup75_10x192.pt")
    jb_s3_key = getattr(args, "job_b_ema95_s3_key", "checkpoints/phase4_job_c/init/12x192_mid2_9_hl64_ema_09995.pt")
    a2_init_pt = _download_or_locate(
        dirs["search"], "A2_soup75_10x192.pt", s3_client, args.bucket, a2_s3_key, tmp_dir / "A2_soup75_10x192.pt"
    )
    jb_init_pt = _download_or_locate(
        dirs["search"], "12x192_mid2_9_hl64_ema_09995.pt", s3_client, args.bucket, jb_s3_key, tmp_dir / "12x192_mid2_9_hl64_ema_09995.pt"
    )

    m10 = ChessSEResNetPhase4(num_blocks=10, channels=192, use_bn=False, conv_bias=True).to(device)
    a2_raw = torch.load(str(a2_init_pt), map_location="cpu", weights_only=False)
    a2_sd = extract_10x192_state_dict(a2_raw)
    m10.load_state_dict(a2_sd, strict=True)
    if device.type == "cuda":
        m10 = m10.to(memory_format=torch.channels_last)

    ref_pol = _capture_first_batch_pol_logits(m10, eval_files, device)

    # Stage split: 360M unquantized BF16 + 40M Middle-2..7 W8A8 QAT (proportional for small unit-test runs)
    qat_pos = min(40_000_000, max(0, args.total_positions // 10))
    stage1_pos = max(args.batch_size, args.total_positions - qat_pos)

    eval_logs: dict = {}
    step0_choice = "a2_soup75_native"
    stage1_done = False

    # Check if Stage 1 already completed before a Stage 2 Spot preemption
    s3_s1_done_key = "checkpoints/phase4_job_c/stage1_complete.pt"
    s3_resume_key = "checkpoints/phase4_job_c/latest.pt"
    local_s1_done = ckpt_dir / "job_c_stage1_complete.pt"
    local_resume = ckpt_dir / "job_c_latest.pt"

    ema_9995 = FastModelEMA(m10, decay=0.9995)
    ema_9999 = FastModelEMA(m10, decay=0.9999)
    opt = torch.optim.AdamW(m10.parameters(), lr=8e-5, weight_decay=1e-4, fused=(device.type == "cuda"))
    step_state = {"step": 0, "seen": 0, "seed": args.seed, "t0": time.time(), "last_ckpt": time.time()}

    if args.resume and s3_client and args.bucket:
        try:
            s3_client.download_file(args.bucket, s3_s1_done_key, str(local_s1_done))
            ck_done = torch.load(str(local_s1_done), map_location=device, weights_only=False)
            ema_9995.load_shadow_dict(ck_done["ema_09995_state_dict"], m10)
            ema_9999.load_shadow_dict(ck_done["ema_09999_state_dict"], m10)
            step_state["step"] = int(ck_done.get("step", 0))
            step_state["seen"] = int(ck_done.get("seen", stage1_pos))
            eval_logs.update(ck_done.get("eval_logs", {}))
            step0_choice = ck_done.get("step0_head_choice", step0_choice)
            stage1_done = True
            print(f"[Job C Resume] Stage 1 already complete (seen={step_state['seen']:,}); resuming straight into Stage 2 QAT!", flush=True)
        except Exception:
            stage1_done = False

    if not stage1_done and args.resume and s3_client and args.bucket:
        try:
            s3_client.download_file(args.bucket, s3_resume_key, str(local_resume))
            ck = torch.load(str(local_resume), map_location=device, weights_only=False)
            m10.load_state_dict(ck["model_state_dict"])
            ema_9995.load_shadow_dict(ck["ema_09995_state_dict"], m10)
            ema_9999.load_shadow_dict(ck["ema_09999_state_dict"], m10)
            opt.load_state_dict(ck["optimizer_state_dict"])
            step_state["step"] = int(ck.get("step", 0))
            step_state["seen"] = int(ck.get("seen", 0))
            eval_logs.update(ck.get("eval_logs", {}))
            step0_choice = ck.get("step0_head_choice", step0_choice)
            print(f"[Job C Resume OK] Stage 1 step={step_state['step']:,}, seen={step_state['seen']:,}", flush=True)
        except Exception as e:
            print(f"[Job C Resume Note] Starting fresh ({e})", flush=True)

    # Step-0 Cross-Architecture Warm-Start Verification (Policy drift == 0.000e+00 & lowest Step-0 150K MSE)
    if not stage1_done and step_state["seen"] == 0:
        print("\n=== [Job C Step 0] Verifying A2_soup75 trunk + Cross-Architecture Head Selection ===", flush=True)
        ev_native = evaluate_unified_150k(m10, eval_files, device, ref_baseline_pol=ref_pol)
        mse_native = _mean_150k_val_mse(ev_native)
        print(f"[Job C Step-0 A2_soup75 Native Heads] mean_mse={mse_native:.5f} | {json.dumps(ev_native)}", flush=True)

        native_heads = {k: v.detach().cpu().clone() for k, v in m10.state_dict().items() if k.startswith(("hl_head.", "ml_head."))}
        n_copied = transplant_hl_mlh_heads(m10, jb_init_pt)
        ev_trans = evaluate_unified_150k(m10, eval_files, device, ref_baseline_pol=ref_pol)
        mse_trans = _mean_150k_val_mse(ev_trans)
        print(f"[Job C Step-0 JobB-ema95 Transplanted Heads ({n_copied} tensors)] mean_mse={mse_trans:.5f} | {json.dumps(ev_trans)}", flush=True)

        eval_logs["step0_a2_soup75_native"] = ev_native
        eval_logs["step0_job_b_transplanted"] = ev_trans

        if mse_trans <= mse_native:
            step0_choice = "job_b_ema95_transplanted"
            chosen_ev = ev_trans
            print(f"[Job C Step-0 Choice] Keeping Job B ema_09995 transplanted heads (MSE {mse_trans:.5f} <= {mse_native:.5f})", flush=True)
        else:
            step0_choice = "a2_soup75_native"
            with torch.no_grad():
                for k, t_cpu in native_heads.items():
                    m10.state_dict()[k].copy_(t_cpu.to(device=device, dtype=m10.state_dict()[k].dtype))
            chosen_ev = ev_native
            print(
                f"[Job C Step-0 Choice] Keeping A2_soup75 native matched heads (MSE {mse_native:.5f} < Job B transplant {mse_trans:.5f}; avoids cross-depth feature-shift shock)",
                flush=True,
            )

        max_step0_drift = max(
            float(v.get("max_policy_logit_drift_vs_346m", 0.0))
            for v in chosen_ev.values()
            if isinstance(v, dict)
        )
        print(f"[Job C Step-0 Policy Preservation VERIFIED] max_drift_vs_A2_soup75={max_step0_drift:.3e}", flush=True)
        if max_step0_drift > 1e-5:
            raise RuntimeError(f"Job C Step-0 policy drift check FAILED: {max_step0_drift:.3e} > 1e-5")

        # Re-initialize EMAs & optimizer cleanly from chosen Step-0 weights
        ema_9995 = FastModelEMA(m10, decay=0.9995)
        ema_9999 = FastModelEMA(m10, decay=0.9999)
        opt = torch.optim.AdamW(m10.parameters(), lr=8e-5, weight_decay=1e-4, fused=(device.type == "cuda"))

    # Async S3 double-buffered shard prefetcher across the 80 fresh 50/50 shards
    pool_q: queue.Queue = queue.Queue(maxsize=4)
    stop_prod = threading.Event()
    s3_pref = args.s3_shard_prefix.strip("/")

    def _s3_shard_producer():
        dl_dir = tmp_dir / "job_c_s3_prefetch"
        pass_idx = step_state["seen"] // 160_000_000
        seen_in_pass: set[str] = set()
        wait_logged = False
        while not stop_prod.is_set():
            ready_keys = list_ready_s3_mix_shards(s3_client, args.bucket, s3_pref)
            unseen = [k for k in ready_keys if k not in seen_in_pass]
            if not unseen and ready_keys:
                seen_in_pass.clear()
                pass_idx += 1
                unseen = list(ready_keys)
                print(f"[Job C Data] Starting shuffled pass #{pass_idx} over {len(ready_keys)} S3 shards ({len(ready_keys) * 2}M pos)", flush=True)

            if not unseen:
                if not wait_logged:
                    print(f"[Job C Data] Waiting for S3 shards under s3://{args.bucket}/{s3_pref}/ ...", flush=True)
                    wait_logged = True
                time.sleep(5.0)
                continue
            wait_logged = False

            rng_p = random.Random(args.seed + pass_idx * 1009 + len(seen_in_pass))
            rng_p.shuffle(unseen)
            for bkey in unseen:
                if stop_prod.is_set():
                    break
                try:
                    mv1, msc = download_s3_mix_shard(s3_client, args.bucket, bkey, dl_dir)
                    seen_in_pass.add(bkey)
                    while not stop_prod.is_set():
                        try:
                            pool_q.put((mv1, msc), timeout=2.0)
                            break
                        except queue.Full:
                            continue
                except Exception as exc:
                    print(f"[Job C S3 Prefetch Warning] {bkey}: {exc}", flush=True)
                # Poll S3 every 4 shards during Pass 0 so newly harvested shards join immediately
                if pass_idx == 0 and (len(seen_in_pass) % 4 == 0):
                    break
        pool_q.put(None)

    prod_th = threading.Thread(target=_s3_shard_producer, daemon=True)
    prod_th.start()

    calib_inputs = None
    last_pool_pair: tuple[np.ndarray, np.ndarray] | None = None

    def _capture_calib(mv1: np.ndarray) -> None:
        nonlocal calib_inputs
        if calib_inputs is None and len(mv1) >= 500:
            cx = unpack_v1_tensor_to_inputs_and_meta(torch.from_numpy(mv1[:500].copy()))[0].numpy().astype(np.float32)
            calib_inputs = [cx[i : i + 1] for i in range(len(cx))]

    # -------------------- Stage 1: 360M Unquantized BF16 Mastery --------------------
    if not stage1_done:
        fwd_m10: nn.Module = m10
        if device.type == "cuda" and os.name != "nt":
            try:
                fwd_m10 = torch.compile(m10, mode="default")
                print("[Job C Stage 1] Enabled torch.compile(mode='default') on CUDA L40S!", flush=True)
            except Exception as exc:
                print(f"[Job C Stage 1 Note] Using eager CUDA ({exc})", flush=True)
                fwd_m10 = m10

        tot_steps_s1 = max(1, (stage1_pos + args.batch_size - 1) // args.batch_size)
        warmup_s1 = min(300, max(1, tot_steps_s1 // 20))

        def _lr_c1(o, stp, _seen):
            lr = compute_cosine_lr(stp, tot_steps_s1, warmup_s1, 8e-5, 4e-6)
            for pg in o.param_groups:
                pg["lr"] = lr

        def _save_s1_ckpt(st):
            ck_p = ckpt_dir / "job_c_latest.pt"
            torch.save({
                "stage": "stage1",
                "step": st["step"],
                "seen": st["seen"],
                "step0_head_choice": step0_choice,
                "eval_logs": eval_logs,
                "model_state_dict": m10.state_dict(),
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

        # Reserve 40 min before hard MaxRuntimeInSeconds for Stage 1 exports + Stage 2 40M QAT + final eval
        s1_deadline = t0_job + max(600, args.max_runtime_sec - 2400)
        print(
            f"\n=== [Job C Stage 1] Unquantized BF16 10x192 Mastery "
            f"({stage1_pos:,} pos, B={args.batch_size}, cosine LR 8e-5 -> 4e-6) ===",
            flush=True,
        )
        m10.train()
        while step_state["seen"] < stage1_pos:
            item = pool_q.get()
            if item is None:
                break
            mv1, msc = item
            _capture_calib(mv1)
            hit_dl = _train_on_pool(
                fwd_m10, mv1, msc, device, opt,
                batch_size=args.batch_size, freeze_trunk=False,
                pol_weight=1.0, val_weight=1.0, mlh_weight=0.05,
                ema_list=[ema_9995, ema_9999], step_state=step_state, lr_fn=_lr_c1,
                max_positions=stage1_pos, s3_ckpt_fn=_save_s1_ckpt,
                stop_deadline_ts=s1_deadline,
            )
            last_pool_pair = (mv1, msc)
            if hit_dl:
                print("[Job C Stage 1] Reached Stage 1 graceful deadline; exporting Pure FP16 & proceeding to Stage 2 QAT!", flush=True)
                break

        _save_s1_ckpt(step_state)

        s1_dir = model_dir / "stage1_fp16"
        s1_dir.mkdir(parents=True, exist_ok=True)
        for lbl, prefix_lbl, shadow in [
            ("stage1_ema09995", "10x192_mastery_ema09995", ema_9995.shadow),
            ("stage1_ema09999", "10x192_mastery_ema09999", ema_9999.shadow),
        ]:
            m10.load_state_dict(shadow)
            ev = evaluate_unified_150k(m10, eval_files, device, ref_baseline_pol=ref_pol)
            eval_logs[lbl] = ev
            print(f"[Job C {lbl} 150K Eval] mean_mse={_mean_150k_val_mse(ev):.5f} | {json.dumps(ev)}", flush=True)
            exp = export_phase4_to_folded_npz_and_ov(m10, s1_dir, prefix_lbl)
            # Also provide exact `<prefix>_fp16.npz` and `<prefix>_fp16.pt` names
            shutil.copy2(exp["folded_npz"], s1_dir / f"{prefix_lbl}_fp16.npz")
            torch.save({
                "architecture": "10x192_folded_hl64_mlh",
                "label": lbl,
                "seen": step_state["seen"],
                "model_state_dict": shadow,
                "eval_150k": ev,
            }, s1_dir / f"{prefix_lbl}_fp16.pt")

        torch.save({
            "step": step_state["step"],
            "seen": step_state["seen"],
            "step0_head_choice": step0_choice,
            "eval_logs": eval_logs,
            "ema_09995_state_dict": ema_9995.shadow,
            "ema_09999_state_dict": ema_9999.shadow,
        }, local_s1_done)
        if s3_client and args.bucket:
            s3_client.upload_file(str(local_s1_done), args.bucket, s3_s1_done_key)
            for p in s1_dir.glob("*"):
                if p.is_file():
                    s3_client.upload_file(str(p), args.bucket, f"checkpoints/phase4_job_c/stage1_fp16/{p.name}")
                    print(f"  [S3 Upload] checkpoints/phase4_job_c/stage1_fp16/{p.name}", flush=True)

    # -------------------- Stage 2: 40M Middle-Block (2..7) W8A8 QAT Polish --------------------
    mse_95 = _mean_150k_val_mse(eval_logs.get("stage1_ema09995", {}))
    mse_99 = _mean_150k_val_mse(eval_logs.get("stage1_ema09999", {}))
    best_s1_lbl = "stage1_ema09995" if mse_95 <= mse_99 else "stage1_ema09999"
    best_s1_shadow = ema_9995.shadow if mse_95 <= mse_99 else ema_9999.shadow
    print(
        f"\n=== [Job C Stage 2] Selected best 360M EMA: {best_s1_lbl} "
        f"(ema09995 MSE={mse_95:.5f} vs ema09999 MSE={mse_99:.5f}) ===",
        flush=True,
    )

    s2_deadline = t0_job + max(300, args.max_runtime_sec - 360)
    if qat_pos > 0 and time.time() < s2_deadline:
        print(f"=== [Job C Stage 2] Middle-Block (2..7) W8A8 QAT Polish ({qat_pos:,} pos, cosine LR 1.5e-5 -> 1e-6) ===", flush=True)
        m10_qat = ChessSEResNetPhase4(num_blocks=10, channels=192, use_bn=False, conv_bias=True).to(device)
        m10_qat.load_state_dict(best_s1_shadow)
        for p in m10_qat.parameters():
            p.requires_grad_(True)
        apply_middle_w8a8_qat(m10_qat, qat_blocks=(2, 3, 4, 5, 6, 7))
        if device.type == "cuda":
            m10_qat = m10_qat.to(memory_format=torch.channels_last)
        m10_qat.train()
        enforce_frozen_bn_eval(m10_qat)

        if calib_inputs is None:
            first_item = pool_q.get()
            if first_item is not None:
                _capture_calib(first_item[0])
                last_pool_pair = first_item

        if calib_inputs:
            with torch.no_grad():
                cx_t = torch.from_numpy(np.concatenate(calib_inputs[:256], axis=0)).to(
                    device=device, memory_format=torch.channels_last if device.type == "cuda" else torch.contiguous_format
                )
                m10_qat(cx_t)

        ema_qat = FastModelEMA(m10_qat, decay=0.9995)
        opt_qat = torch.optim.AdamW(
            [p for p in m10_qat.parameters() if p.requires_grad],
            lr=1.5e-5, weight_decay=1e-4, fused=(device.type == "cuda"),
        )
        tot_qat_steps = max(1, (qat_pos + args.batch_size - 1) // args.batch_size)
        warmup_qat = min(250, max(1, tot_qat_steps // 10))
        st_qat = {"step": 0, "seen": 0, "seed": args.seed + 777, "t0": time.time(), "last_ckpt": time.time()}
        s3_s2_key = "checkpoints/phase4_job_c/stage2_latest.pt"

        if args.resume and s3_client and args.bucket:
            try:
                loc_s2 = ckpt_dir / "job_c_stage2_latest.pt"
                s3_client.download_file(args.bucket, s3_s2_key, str(loc_s2))
                ck2 = torch.load(str(loc_s2), map_location=device, weights_only=False)
                m10_qat.load_state_dict(ck2["model_state_dict"])
                ema_qat.load_shadow_dict(ck2["ema_09995_state_dict"], m10_qat)
                opt_qat.load_state_dict(ck2["optimizer_state_dict"])
                st_qat["step"] = int(ck2.get("step", 0))
                st_qat["seen"] = int(ck2.get("seen", 0))
                print(f"[Job C Stage 2 Resume OK] step={st_qat['step']:,}, seen={st_qat['seen']:,}", flush=True)
            except Exception:
                pass

        def _lr_c2(o, stp, _seen):
            lr_q = compute_cosine_lr(stp, tot_qat_steps, warmup_qat, 1.5e-5, 1e-6)
            for pg in o.param_groups:
                pg["lr"] = lr_q

        def _save_s2_ckpt(st):
            ck2_p = ckpt_dir / "job_c_stage2_latest.pt"
            torch.save({
                "stage": "stage2_qat",
                "step": st["step"],
                "seen": st["seen"],
                "model_state_dict": m10_qat.state_dict(),
                "ema_09995_state_dict": ema_qat.shadow,
                "optimizer_state_dict": opt_qat.state_dict(),
            }, ck2_p)
            if s3_client and args.bucket:
                try:
                    s3_client.upload_file(str(ck2_p), args.bucket, s3_s2_key)
                    print(f"  [Checkpoint] Uploaded {s3_s2_key} at qat_seen={st['seen']:,}", flush=True)
                except Exception as e:
                    print(f"  [Checkpoint Warning] {e}", flush=True)

        while st_qat["seen"] < qat_pos and time.time() < s2_deadline:
            try:
                q_item = pool_q.get(timeout=10.0)
            except queue.Empty:
                q_item = last_pool_pair
            if q_item is None:
                q_item = last_pool_pair
            if q_item is None:
                break
            qv1, qsc = q_item
            _capture_calib(qv1)
            hit_q = _train_on_pool(
                m10_qat, qv1, qsc, device, opt_qat,
                batch_size=args.batch_size, freeze_trunk=False,
                pol_weight=1.0, val_weight=1.0, mlh_weight=0.05,
                ema_list=[ema_qat], step_state=st_qat, lr_fn=_lr_c2,
                max_positions=qat_pos, s3_ckpt_fn=_save_s2_ckpt,
                stop_deadline_ts=s2_deadline,
            )
            last_pool_pair = (qv1, qsc)
            if hit_q:
                print("[Job C Stage 2] Hit Stage 2 graceful deadline; exporting final QAT FP16 + Middle-2..7 INT8!", flush=True)
                break

        m10_qat.load_state_dict(ema_qat.shadow)
        ev_qat = evaluate_unified_150k(m10_qat, eval_files, device, ref_baseline_pol=ref_pol)
        eval_logs["stage2_qat_mid2_7"] = ev_qat
        print(f"[Job C Stage 2 qat_mid2_7 150K Eval] mean_mse={_mean_150k_val_mse(ev_qat):.5f} | {json.dumps(ev_qat)}", flush=True)

        out_qat = model_dir / "qat_mid2_7"
        exp_qat = export_phase4_to_folded_npz_and_ov(
            m10_qat, out_qat, "10x192_mastery_qat_mid2_7",
            calib_inputs=calib_inputs, int8_blocks=(2, 3, 4, 5, 6, 7),
        )
        torch.save({
            "architecture": "10x192_folded_mid2_7_qat_hl64_mlh",
            "best_stage1_ema": best_s1_lbl,
            "stage1_seen": step_state["seen"],
            "stage2_qat_seen": st_qat["seen"],
            "model_state_dict": m10_qat.state_dict(),
            "ema_09995_state_dict": ema_qat.shadow,
            "eval_150k": ev_qat,
            "export": exp_qat,
        }, out_qat / "10x192_mastery_qat_mid2_7.pt")

    stop_prod.set()
    last_pool_pair = None
    gc.collect()

    rep_c = {
        "mode": "job_c_10x192_mastery",
        "step0_head_choice": step0_choice,
        "best_stage1_ema": best_s1_lbl,
        "stage1_seen": step_state["seen"],
        "stage2_qat_target": qat_pos,
        "elapsed_sec": round(time.time() - t0_job, 1),
        "eval_150k": eval_logs,
    }
    (model_dir / "phase4_job_c_report.json").write_text(json.dumps(rep_c, indent=2), encoding="utf-8")

    if s3_client and args.bucket:
        for p in model_dir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(model_dir).as_posix()
                s3_client.upload_file(str(p), args.bucket, f"checkpoints/phase4_job_c/{rel}")
                print(f"  [S3 Upload] checkpoints/phase4_job_c/{rel}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["job_a", "job_b", "job_c_10x192_mastery", "harvest_shards"], required=True)
    parser.add_argument("--bucket", type=str, default=DEFAULT_BUCKET)
    parser.add_argument("--region", type=str, default=DEFAULT_REGION)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--total_positions", type=int, default=40_000_000)
    parser.add_argument("--max_runtime_sec", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=2_000_000)
    parser.add_argument("--s3_shard_prefix", type=str, default="data/phase4_mix_shards")
    parser.add_argument("--sf_ratio", type=float, default=0.50)
    parser.add_argument("--t80_skip", type=int, default=40)
    parser.add_argument("--sf_line_offset", type=int, default=40_000_000)
    parser.add_argument("--a2_soup75_s3_key", type=str, default="checkpoints/phase4_job_c/init/A2_soup75_10x192.pt")
    parser.add_argument("--job_b_ema95_s3_key", type=str, default="checkpoints/phase4_job_c/init/12x192_mid2_9_hl64_ema_09995.pt")
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
    s3_v2_names = ("shard_000.bin", "shard_001.bin")
    if args.mode == "harvest_shards":
        my_v2 = [s3_v2_names[i] for i in range(len(s3_v2_names)) if (i % max(1, int(args.num_workers))) == int(args.worker_id)]
    elif args.mode == "job_c_10x192_mastery":
        my_v2 = []
    else:
        my_v2 = list(s3_v2_names)
    for sfn in my_v2:
        v2_shards.append(_download_or_locate(search_dirs, sfn, s3_client, args.bucket, f"data/shards/{sfn}", tmp_dir / f"v2_{sfn}"))

    dirs = {
        "data": data_dir, "model": model_dir, "ckpt": ckpt_dir, "tmp": tmp_dir,
        "search": search_dirs, "eval_files": eval_files, "excl_bin": excl_bin,
        "excl_hashes": excl_hashes, "v2_shards": v2_shards,
    }

    if args.mode == "harvest_shards":
        run_harvest_shards(args, s3_client, dirs)
    elif args.mode == "job_a":
        run_job_a(args, device, s3_client, dirs)
    elif args.mode == "job_c_10x192_mastery":
        run_job_c_10x192_mastery(args, device, s3_client, dirs)
    else:
        run_job_b(args, device, s3_client, dirs)


if __name__ == "__main__":
    main()
