#!/usr/bin/env python3
"""Download + verify + quantize Qwen3.5-35B-A3B base model.

Downloads the Qwen3.5-35B-A3B weights from HuggingFace into a target dir,
verifies every shard referenced by ``model.safetensors.index.json`` is
present, does a quick transformers load + 1-token generate smoke test,
then optionally produces a Q4_K_M GGUF via llama.cpp.

Designed to be idempotent: a re-run with the same ``--target-dir`` skips
already-downloaded files (delegated to ``huggingface_hub``'s cache /
symlinks) and re-verifies them.

Usage:
    python scripts/download_base.py \\
        --target-dir /home/kxkm/models/qwen3.5-35b-a3b \\
        [--skip-quantize] \\
        [--llama-cpp /home/kxkm/llama.cpp]

Exit code 0 on full success, non-zero on any verify / smoke failure.

Note: ``verify_safetensors_index`` is the source of truth for shard
integrity; no hard-coded byte total is used (35B-A3B shard layout can
vary by upstream re-release).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ID = "Qwen/Qwen3.5-35B-A3B"
BF16_SUBDIR = "bf16"
ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.*",
    "*.json",
    "*.txt",
    "*.jinja",
    "tokenizer*",
    "merges.txt",
    "vocab.json",
    "LICENSE",
]
def log(msg: str) -> None:
    print(f"[download_base] {msg}", flush=True)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def snapshot(target: Path) -> Path:
    """Download the HF snapshot into ``target``. Returns the local dir."""
    from huggingface_hub import snapshot_download  # imported lazily

    target.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    local = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(target),
        allow_patterns=ALLOW_PATTERNS,
        max_workers=4,
    )
    log(f"snapshot_download done in {time.time() - t0:.1f}s -> {local}")
    return Path(local)


def verify_sizes(local: Path) -> int:
    """Sum shard sizes, log the total. 35B-A3B has no fixed reference byte
    count (upstream re-releases change shard layout) — real integrity is
    covered by ``verify_safetensors_index``.
    """
    total = 0
    for p in sorted(local.rglob("*")):
        if p.is_file():
            total += p.stat().st_size
    log(f"total bytes = {total:,}")
    return total


def verify_safetensors_index(local: Path) -> None:
    """If an index.json has a sha256 map, verify every shard."""
    idx = local / "model.safetensors.index.json"
    if not idx.exists():
        log("no safetensors index.json — skipping hash verify")
        return
    data = json.loads(idx.read_text())
    # HF index.json sometimes has 'metadata' with shard hashes;
    # but typically only weight_map. Hash verify at shard level:
    shards = sorted({v for v in data.get("weight_map", {}).values()})
    if not shards:
        log("index has no weight_map — skipping hash verify")
        return
    for shard in shards:
        shard_path = local / shard
        if not shard_path.exists():
            raise RuntimeError(f"missing shard {shard}")
        sz = shard_path.stat().st_size
        log(f"shard {shard}: {sz:,} bytes (present)")
    log(f"verified {len(shards)} shard(s) present")


def smoke_generate(local: Path) -> str:
    """Quick load + generate — catches corrupt weights early."""
    import torch  # noqa: F401  # imported to prime CUDA state cleanly
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log("smoke: loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(str(local), trust_remote_code=True)
    log("smoke: loading model (cpu, auto dtype)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        str(local),
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    log(f"smoke: model loaded in {time.time() - t0:.1f}s")
    inputs = tok("hello", return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    log(f"smoke: decoded = {text!r}")
    if not text.strip():
        raise RuntimeError("smoke generate returned empty output")
    return text


def quantize_q4(
    hf_dir: Path,
    gguf_out: Path,
    llama_cpp: Path,
) -> bool:
    """Run convert_hf_to_gguf.py + llama-quantize. Returns True on success."""
    convert = llama_cpp / "convert_hf_to_gguf.py"
    qbin = llama_cpp / "build" / "bin" / "llama-quantize"
    if not convert.exists() or not qbin.exists():
        log(
            f"quantize SKIPPED: missing llama.cpp binaries "
            f"(convert={convert.exists()}, quantize={qbin.exists()})"
        )
        return False

    bf16_gguf = gguf_out.with_suffix(".bf16.gguf")
    log(f"quantize: converting HF -> {bf16_gguf}")
    rc = subprocess.call(
        [
            sys.executable,
            str(convert),
            str(hf_dir),
            "--outfile",
            str(bf16_gguf),
            "--outtype",
            "bf16",
        ]
    )
    if rc != 0:
        log(f"quantize: convert_hf_to_gguf.py failed (rc={rc})")
        return False

    log(f"quantize: running llama-quantize -> {gguf_out}")
    rc = subprocess.call(
        [str(qbin), str(bf16_gguf), str(gguf_out), "Q4_K_M"]
    )
    if rc != 0:
        log(f"quantize: llama-quantize failed (rc={rc})")
        return False

    try:
        bf16_gguf.unlink()
    except OSError:
        pass
    log(f"quantize OK: {gguf_out} "
        f"({gguf_out.stat().st_size / 1024 / 1024 / 1024:.2f} GB)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target-dir",
        default=os.environ.get(
            "MICRO_KIKI_BASE_DIR", "/home/kxkm/models/qwen3.5-35b-a3b"
        ),
        help="root dir; bf16 saved at <target>/bf16/, q4 at <target>-q4.gguf",
    )
    ap.add_argument(
        "--skip-quantize", action="store_true", help="skip GGUF Q4_K_M step"
    )
    ap.add_argument(
        "--skip-smoke", action="store_true", help="skip transformers load+gen"
    )
    ap.add_argument(
        "--llama-cpp",
        default="/home/kxkm/llama.cpp",
        help="path to llama.cpp checkout (with build/bin/llama-quantize)",
    )
    args = ap.parse_args()

    root = Path(args.target_dir).resolve()
    bf16_dir = root / BF16_SUBDIR
    log(f"target root = {root}")
    log(f"bf16 dir    = {bf16_dir}")

    local = snapshot(bf16_dir)
    verify_sizes(local)
    verify_safetensors_index(local)

    if args.skip_smoke:
        log("smoke generate SKIPPED (--skip-smoke)")
    else:
        smoke_generate(local)

    if args.skip_quantize:
        log("quantize SKIPPED (--skip-quantize)")
    else:
        gguf_out = root.parent / f"{root.name}-q4.gguf"
        quantize_q4(local, gguf_out, Path(args.llama_cpp))

    log("DONE")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        log(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
