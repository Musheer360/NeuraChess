"""
NeuraChess Model Weight Downloader (`neurachess.scripts.download_weights`).

Downloads official NeuraChess OpenVINO IR (v1.1.0 Mid-2..7 QAT INT8 & v1.0.x FP16) and folded NumPy
weights from GitHub Releases and verifies integrity via SHA-256 checksums before placement in `weights/`.
"""

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

from neurachess._paths import find_repo_root

ROOT_DIR = find_repo_root()
DEFAULT_WEIGHTS_DIR = ROOT_DIR / "weights"

MODEL_ASSETS = {
    "10x192": [
        {
            "filename": "neurachess-10x192-qat-int8.xml",
            "aliases": ["neurachess-10x192-qat-int8.xml"],
            "sha256": "86fce43ef5a89580f548cf56fa79079b055afdf1da8eb0903ff6537874df5bcb",
        },
        {
            "filename": "neurachess-10x192-qat-int8.bin",
            "aliases": ["neurachess-10x192-qat-int8.bin"],
            "sha256": "9db3c4d6e1f45a07a4cc0cd6cfc5898edc1c3ffdc8171137a754de0b7f3db260",
        },
        {
            "filename": "neurachess-10x192-qat-int8.npz",
            "aliases": ["neurachess-10x192-qat-int8.npz"],
            "sha256": "99d2689b76cec74927f425d121cddc6a9e2b6360704a721ba0b0f67cc30c5df2",
        },
    ],
    "12x192": [
        {
            "filename": "neurachess-12x192-qat-int8.xml",
            "aliases": ["neurachess-12x192-qat-int8.xml"],
            "sha256": "975589f19b1fb721b4c5eca966eec997c589df2685bbcd3f17e6ac809fc1e52b",
        },
        {
            "filename": "neurachess-12x192-qat-int8.bin",
            "aliases": ["neurachess-12x192-qat-int8.bin"],
            "sha256": "eceadf14d505efeb81d94d0bfa99cefbc6aa702ac2b16839993ab07ff0bdf1fb",
        },
        {
            "filename": "neurachess-12x192-qat-int8.npz",
            "aliases": ["neurachess-12x192-qat-int8.npz"],
            "sha256": "149f17785aac782e697e8c5744372765e62c559af2f7d9ec485bff323ef82d3a",
        },
    ],
    "10x192-fp16": [
        {
            "filename": "neurachess-10x192-fp16.xml",
            "aliases": ["neurachess-10x192-fp16.xml", "346M_10x192_V0_FP16.xml"],
            "sha256": "f70fbd9923c678af0436f2272d4ad962e66b543ee844d620813846c5f7c3b698",
        },
        {
            "filename": "neurachess-10x192-fp16.bin",
            "aliases": ["neurachess-10x192-fp16.bin", "346M_10x192_V0_FP16.bin"],
            "sha256": "1ef15c58aedddbe2562a59373c37c5933a02b9371530f0c725b023b17dc7f02e",
        },
        {
            "filename": "neurachess-10x192-fp16.npz",
            "aliases": ["neurachess-10x192-fp16.npz", "full_346m_10x192_ema09999_folded.npz"],
            "sha256": "52805a44b0670aae020bee26c0bc05e9ee6cb9835b374759ff1cd5f0ddb8f049",
        },
    ],
    "10x128": [
        {
            "filename": "neurachess-10x128-fp16.xml",
            "aliases": ["neurachess-10x128-fp16.xml", "346M_10x128_V0_FP16.xml"],
            "sha256": "2075a2dbaf51147d48c68190658a277368388d17766d601196f91a40846d60b5",
        },
        {
            "filename": "neurachess-10x128-fp16.bin",
            "aliases": ["neurachess-10x128-fp16.bin", "346M_10x128_V0_FP16.bin"],
            "sha256": "a5a5b302b091acb7cc67ff81d03e074ff4d11ec0143942547bd8a30a22119c66",
        },
        {
            "filename": "neurachess-10x128-fp16.npz",
            "aliases": ["neurachess-10x128-fp16.npz", "full_346m_10x128_ema09999_folded.npz"],
            "sha256": "2f993da939572cc02c7962e6b69761706229b2331434d44e887369804cf65137",
        },
    ],
}

MODEL_ALIASES = {
    "10x192": "10x192",
    "10x192-qat-int8": "10x192",
    "12x192": "12x192",
    "12x192-qat-int8": "12x192",
    "10x192-fp16": "10x192-fp16",
    "10x128": "10x128",
    "10x128-fp16": "10x128",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_or_download_asset(repo: str, tag: str, out_dir: Path, spec: dict, force: bool = False) -> Path:
    dest = out_dir / spec["filename"]
    expected = spec["sha256"]

    if dest.exists() and not force:
        actual = sha256_file(dest)
        if actual == expected:
            print(f"[OK] {dest.name} (SHA-256 verified)")
            return dest
        print(f"[WARN] Checksum mismatch on existing {dest.name}; re-downloading...")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None

    for alias in spec["aliases"]:
        url = f"https://github.com/{repo}/releases/download/{tag}/{alias}"
        print(f"[Download] {url} -> {dest.name}")
        try:
            urllib.request.urlretrieve(url, tmp_path)
            actual = sha256_file(tmp_path)
            if actual != expected:
                tmp_path.unlink(missing_ok=True)
                raise ValueError(f"SHA-256 mismatch for {alias}: got {actual}, expected {expected}")
            tmp_path.replace(dest)
            print(f"[Verified] {dest.name} ({expected[:12]}...)")
            return dest
        except (urllib.error.URLError, ValueError) as exc:
            last_err = exc
            tmp_path.unlink(missing_ok=True)

    raise RuntimeError(f"Failed to download verified {dest.name} from {repo}@{tag}: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and SHA-256 verify NeuraChess weights")
    parser.add_argument(
        "--model",
        choices=["10x192", "10x192-qat-int8", "12x192", "12x192-qat-int8", "10x192-fp16", "10x128", "all"],
        default="10x192",
    )
    parser.add_argument("--repo", type=str, default="Musheer360/NeuraChess")
    parser.add_argument("--tag", type=str, default="v1.1.0")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    keys = ["10x192", "12x192", "10x192-fp16", "10x128"] if args.model == "all" else [MODEL_ALIASES[args.model]]
    for k in keys:
        for asset in MODEL_ASSETS[k]:
            verify_or_download_asset(args.repo, args.tag, args.out_dir, asset, force=args.force)


if __name__ == "__main__":
    main()
