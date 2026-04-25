#!/usr/bin/env python3
"""
sync_hf.py — Update README/model cards on micro-kiki HuggingFace repos.

Repos managed:
  clemsail/micro-kiki-router-v4   — MiniLM router weights
  clemsail/micro-kiki-v4-sota     — 35 LoRA adapters
  clemsail/micro-kiki-v35b        — legacy model (no-op for now)

Usage:
  python scripts/sync_hf.py --dry-run   # preview what would change (default)
  python scripts/sync_hf.py --execute   # actually push to HF
"""

import argparse
import difflib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUTER_REPO = "clemsail/micro-kiki-router-v4"
SOTA_REPO = "clemsail/micro-kiki-v4-sota"
LEGACY_REPO = "clemsail/micro-kiki-v35b"

SOTA_README_SRC = Path(
    "/Users/clems/KIKI-Mac_tunner/output/micro-kiki/lora-qwen36-35b-v4-sota/README.md"
)

ROUTER_README = """\
---
language:
  - en
  - fr
tags:
  - micro-kiki
  - router
  - sentence-transformers
  - mlp
  - domain-routing
  - apple-silicon
license: apache-2.0
---

# Micro-Kiki Router V4

Lightweight domain router used by the **micro-kiki** project to dispatch
incoming prompts to the right LoRA adapter from a pool of 31 fine-tuned
domains.

## Architecture

```
sentence-transformers/all-mpnet-base-v2  (768-d embeddings)
    ↓
Linear(768 → 512) + ReLU
    ↓
Linear(512 → 31) + sigmoid
    ↓
31 domain scores  (multi-label, thresholded at 0.5)
```

The backbone (`all-mpnet-base-v2`) is **frozen**; only the two linear
layers are trained.  Total trainable parameters: ~434 K.

## Accuracy (V4, April 2026)

| Metric | Value |
|--------|-------|
| Top-1  | **70.7 %** |
| Top-3  | **89.5 %** |

Evaluated on a held-out split of the router training dataset
(~2 k labelled prompts across 31 domains).

## Domains (31 after merges)

`chat-fr`, `components`, `cpp`, `devops`, `docker`, `dsp`, `electronics`,
`embedded`, `emc`, `freecad`, `html-css`, `iot`, `kicad-dsl`, `kicad-pcb`,
`llm-ops`, `llm-orch`, `lua-upy`, `math`, `ml-training`, `music-audio`,
`platformio`, `python`, `reasoning`, `rust`, `security`, `shell`, `spice`,
`sql`, `stm32`, `typescript`, `yaml-json`

*(Domains `power`, `spice-sim`, `web-backend`, `web-frontend` were merged
into sibling domains before the V4 training run.)*

## Usage

```python
from huggingface_hub import hf_hub_download
import torch
import json
from sentence_transformers import SentenceTransformer

# Load backbone + head weights
backbone = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
weights_path = hf_hub_download("clemsail/micro-kiki-router-v4", "router_head.pt")
label_path   = hf_hub_download("clemsail/micro-kiki-router-v4", "labels.json")

head    = torch.load(weights_path, map_location="cpu")
labels  = json.loads(Path(label_path).read_text())

def route(prompt: str, threshold: float = 0.5) -> list[str]:
    emb    = torch.tensor(backbone.encode(prompt))
    logits = head(emb)
    scores = torch.sigmoid(logits)
    return [labels[i] for i, s in enumerate(scores) if s > threshold]

print(route("How do I debounce a GPIO in ESP-IDF?"))
# → ['embedded', 'iot']
```

## Training

Router was trained with `train_router_v4.py` inside the
[micro-kiki](https://github.com/electron-rare/micro-kiki) repo using a
curriculum of ~55 k domain-labelled prompts.

## Citation

```bibtex
@software{microkiki_router_v4_2026,
  author  = {Saillant, Clément},
  title   = {Micro-Kiki Router V4},
  year    = {2026},
  month   = {4},
  url     = {https://huggingface.co/clemsail/micro-kiki-router-v4}
}
```

## Links

- GitHub: <https://github.com/electron-rare/micro-kiki>
- LoRA adapters: <https://huggingface.co/clemsail/micro-kiki-v4-sota>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sota_readme() -> str:
    if not SOTA_README_SRC.exists():
        raise FileNotFoundError(
            f"V4-SOTA README not found at {SOTA_README_SRC}\n"
            "Make sure KIKI-Mac_tunner output is available."
        )
    return SOTA_README_SRC.read_text(encoding="utf-8")


def _fetch_current_readme(api, repo_id: str) -> str | None:
    """Return the current README.md content from HF, or None on error."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename="README.md", repo_type="model")
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return None


def _diff_summary(label: str, old: str | None, new: str) -> None:
    if old is None:
        print(f"  [{label}]  no existing README — would create ({len(new)} chars)")
        return
    if old == new:
        print(f"  [{label}]  no changes")
        return
    diff = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile="current",
            tofile="proposed",
            n=3,
        )
    )
    added   = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    print(f"  [{label}]  +{added} / -{removed} lines")
    # Show a condensed diff (first 30 lines)
    for line in diff[:30]:
        print("   ", line, end="")
    if len(diff) > 30:
        print(f"\n   ... ({len(diff) - 30} more diff lines)")


def _push_readme(api, repo_id: str, content: str, dry_run: bool) -> None:
    if dry_run:
        return
    api.upload_file(
        path_or_fileobj=content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: sync README via sync_hf.py",
    )
    print(f"  -> pushed to {repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync HuggingFace model card READMEs.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview changes without pushing (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually push changes to HuggingFace",
    )
    args = parser.parse_args()

    dry_run = not args.execute

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
        return 1

    api = HfApi()

    mode_label = "DRY RUN — no changes will be pushed" if dry_run else "EXECUTE — pushing to HuggingFace"
    print(f"\n{'=' * 60}")
    print(f"  sync_hf.py  |  {mode_label}")
    print(f"{'=' * 60}\n")

    errors: list[str] = []

    # ------------------------------------------------------------------
    # 1. Router README
    # ------------------------------------------------------------------
    print(f"[1/3] {ROUTER_REPO}")
    current_router = _fetch_current_readme(api, ROUTER_REPO)
    _diff_summary("router", current_router, ROUTER_README)
    try:
        _push_readme(api, ROUTER_REPO, ROUTER_README, dry_run)
    except Exception as exc:
        errors.append(f"{ROUTER_REPO}: {exc}")
        print(f"  ERROR: {exc}")

    print()

    # ------------------------------------------------------------------
    # 2. V4-SOTA LoRA README
    # ------------------------------------------------------------------
    print(f"[2/3] {SOTA_REPO}")
    try:
        sota_readme = _load_sota_readme()
    except FileNotFoundError as exc:
        errors.append(str(exc))
        print(f"  SKIP: {exc}")
        sota_readme = None

    if sota_readme is not None:
        current_sota = _fetch_current_readme(api, SOTA_REPO)
        _diff_summary("v4-sota", current_sota, sota_readme)
        try:
            _push_readme(api, SOTA_REPO, sota_readme, dry_run)
        except Exception as exc:
            errors.append(f"{SOTA_REPO}: {exc}")
            print(f"  ERROR: {exc}")

    print()

    # ------------------------------------------------------------------
    # 3. Legacy repo — informational only
    # ------------------------------------------------------------------
    print(f"[3/3] {LEGACY_REPO}")
    print(f"  [legacy]  no update scheduled (older model, kept as-is)")

    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"{'=' * 60}")
    if errors:
        print(f"  COMPLETED WITH {len(errors)} ERROR(S):")
        for e in errors:
            print(f"    - {e}")
        return 1

    if dry_run:
        print("  Dry run complete. Use --execute to push changes.")
    else:
        print("  All repos updated successfully.")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
