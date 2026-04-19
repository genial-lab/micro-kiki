#!/usr/bin/env python3
"""Phase C3 end-to-end: ingest logs -> sanitize -> cluster -> augment -> write corpus.

Usage (mascarade-datasets sources):
    uv run python scripts/build_corpus_real.py \\
        --sources ~/Documents/Projets_Techniques/GitHub/electron-rare/mascarade-datasets/*_chat.jsonl \\
        --target-per-domain 500 \\
        --dry-run-augment \\
        --output-dir data/corpus-real \\
        --stats-output results/c3-corpus-stats.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.augmenter import augment_domain_via_teacher
from src.data.corpus_validator import (
    cluster_embeddings_hdbscan,
    match_clusters_to_domains,
)
from src.data.sanitization import sanitize

logger = logging.getLogger(__name__)


DOMAINS = [
    "dsp", "electronics", "emc", "embedded", "freecad",
    "kicad-dsl", "platformio", "power", "spice", "stm32",
]


def _ingest_source(path: Path) -> list[str]:
    """Extract user-authored text from a JSONL file.

    Handles three schemas:
    - {"role": "user", "content": "..."}                                 (OpenAI chat format)
    - {"question": "...", ...}                                           (SFT instruction format)
    - {"conversations": [{"from": "human", "value": "..."}, ...]}        (ShareGPT/mascarade format)
    """
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Branch 1: OpenAI chat
            if "content" in obj and obj.get("role") == "user":
                texts.append(obj["content"])
            # Branch 2: SFT instruction
            elif "question" in obj:
                texts.append(obj["question"])
            # Branch 3: ShareGPT/mascarade multi-turn
            elif "conversations" in obj and isinstance(obj["conversations"], list):
                for turn in obj["conversations"]:
                    if isinstance(turn, dict) and turn.get("from") == "human":
                        val = turn.get("value")
                        if isinstance(val, str):
                            texts.append(val)
    return texts


def _domain_from_path(path: Path) -> str | None:
    """Infer domain from the filename. mascarade-datasets uses `<domain>_chat.jsonl`.

    Maps 'kicad' -> 'kicad-dsl' and 'iot' -> 'electronics' per inventory mapping.
    """
    stem = path.stem.replace("_chat", "").replace("_dataset", "")
    if stem == "kicad":
        return "kicad-dsl"
    if stem == "iot":
        return "electronics"
    if stem in DOMAINS:
        return stem
    return None


def _embed(texts: list[str], backbone: str = "models/niche-embeddings",
           seq_len: int = 32) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(backbone, device="cpu")
    tok = st.tokenizer
    m = st[0].auto_model.to("cpu")
    out = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=seq_len, padding="max_length")
        with torch.no_grad():
            h = m(**enc).last_hidden_state
        out.append(h.squeeze(0).mean(dim=0).cpu().numpy())
    return np.stack(out).astype(np.float64)


def _teacher_call(url: str, model: str, prompt: str) -> str:
    resp = requests.post(
        f"{url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 200, "temperature": 0.7},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", nargs="+", required=True,
                   help="JSONL files with user questions")
    p.add_argument("--target-per-domain", type=int, default=500)
    p.add_argument("--teacher-url", default="http://studio:18000")
    p.add_argument("--teacher-model", default="qwen3-coder-480b-mxfp4")
    p.add_argument("--existing-data-dir", type=Path, default=Path("data/final"),
                   help="Used to label clusters against the known 10-domain taxonomy")
    p.add_argument("--output-dir", type=Path, default=Path("data/corpus-real"))
    p.add_argument("--stats-output", type=Path, default=Path("results/c3-corpus-stats.json"))
    p.add_argument("--dry-run-augment", action="store_true",
                   help="Skip real teacher calls; use stub augmentation")
    p.add_argument("--max-per-source", type=int, default=0,
                   help="If >0, cap per-source ingestion (useful for dry-run)")
    p.add_argument("--use-filename-domain", action="store_true",
                   help="Trust filename-derived domain (mascarade-datasets style) "
                        "instead of nearest-centroid labelling")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # STAGE 1 - Ingest + sanitize, tracking per-file domain if requested
    raw_texts: list[str] = []
    source_domain: list[int | None] = []
    for src in args.sources:
        path = Path(src)
        t = _ingest_source(path)
        if args.max_per_source > 0:
            t = t[: args.max_per_source]
        logger.info("ingested %d texts from %s", len(t), path.name)
        raw_texts.extend(t)
        if args.use_filename_domain:
            dom_name = _domain_from_path(path)
            dom_idx = DOMAINS.index(dom_name) if dom_name in DOMAINS else None
            source_domain.extend([dom_idx] * len(t))

    if not raw_texts:
        logger.error("no texts ingested - check --sources paths")
        return 2

    logger.info("sanitizing %d texts (regex + NER)...", len(raw_texts))
    clean_texts = [sanitize(t) for t in raw_texts]

    # STAGE 2 - Embed + cluster (for validation / reporting even when using filename domains)
    logger.info("embedding %d texts...", len(clean_texts))
    embs = _embed(clean_texts)

    logger.info("clustering with HDBSCAN...")
    cluster_ids = cluster_embeddings_hdbscan(embs, min_cluster_size=30)
    logger.info("  found %d clusters (excluding noise)", len(set(cluster_ids) - {-1}))

    # Match clusters to existing taxonomy
    if args.use_filename_domain and any(d is not None for d in source_domain):
        assigned_domain = np.array([d if d is not None else -1 for d in source_domain], dtype=np.int64)
        logger.info("using filename-derived domain labels")
    else:
        from src.routing.text_jepa.dataset import load_domain_corpus
        existing_samples = load_domain_corpus(args.existing_data_dir, domains=DOMAINS, max_per_domain=50)
        existing_texts = [s.text for s in existing_samples]
        existing_labels = np.array(
            [DOMAINS.index(s.domain) for s in existing_samples], dtype=np.int64
        )
        existing_embs = _embed(existing_texts)
        existing_centroids = np.stack(
            [existing_embs[existing_labels == d].mean(axis=0) for d in range(len(DOMAINS))]
        )
        assigned_domain = np.full(len(embs), -1, dtype=np.int64)
        for i in range(len(embs)):
            if cluster_ids[i] == -1:
                continue
            dists = np.linalg.norm(existing_centroids - embs[i], axis=1)
            assigned_domain[i] = int(np.argmin(dists))

    match = match_clusters_to_domains(assigned_domain, cluster_ids, n_domains=len(DOMAINS))
    logger.info("  cluster-taxonomy mean_overlap = %.3f (chance ~ 0.1)", match["mean_overlap"])

    # STAGE 3 - Count per domain, augment under-represented
    per_domain_texts: dict[int, list[str]] = defaultdict(list)
    for i, d in enumerate(assigned_domain):
        if d >= 0:
            per_domain_texts[int(d)].append(clean_texts[i])

    stats = {
        "stage1_raw": len(raw_texts),
        "stage1_sanitized": len(clean_texts),
        "stage2_noise_points": int((assigned_domain == -1).sum()),
        "stage2_cluster_taxonomy_overlap": match["mean_overlap"],
        "stage3_per_domain_real": {DOMAINS[d]: len(v) for d, v in per_domain_texts.items()},
        "stage3_per_domain_augmented": {},
        "stage3_per_domain_final": {},
    }

    def teacher(prompt: str) -> str:
        if args.dry_run_augment:
            return "SYNTHETIC: placeholder question about the domain."
        return _teacher_call(args.teacher_url, args.teacher_model, prompt)

    for d, name in enumerate(DOMAINS):
        have = len(per_domain_texts[d])
        # If we already have more than target, truncate (for stable downstream)
        if have >= args.target_per_domain:
            per_domain_texts[d] = per_domain_texts[d][: args.target_per_domain]
            stats["stage3_per_domain_augmented"][name] = 0
            continue
        need = args.target_per_domain - have
        seeds = per_domain_texts[d] if have > 0 else [f"Example question for {name}"]
        logger.info("augmenting %s: %d real + %d synthetic -> %d target",
                    name, have, need, args.target_per_domain)
        new_items = augment_domain_via_teacher(
            domain=name, seeds=seeds, n_to_generate=need, teacher_fn=teacher,
        )
        per_domain_texts[d].extend(new_items)
        stats["stage3_per_domain_augmented"][name] = len(new_items)

    for d, name in enumerate(DOMAINS):
        stats["stage3_per_domain_final"][name] = len(per_domain_texts[d])

    # STAGE 4 - Write output jsonl per domain
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for d, name in enumerate(DOMAINS):
        out_path = args.output_dir / name / "train.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for text in per_domain_texts[d]:
                f.write(json.dumps({"text": text, "domain": name}) + "\n")
        logger.info("  wrote %s (%d lines)", out_path, len(per_domain_texts[d]))

    # STAGE 5 - Stats JSON
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2))
    logger.info("wrote %s", args.stats_output)

    print("\n=== C3 Corpus Stats ===")
    print(f"Raw texts:          {stats['stage1_raw']}")
    print(f"Sanitized:          {stats['stage1_sanitized']}")
    print(f"Cluster overlap:    {stats['stage2_cluster_taxonomy_overlap']:.3f}")
    print(f"{'Domain':<12} {'real':>6} {'aug':>6} {'final':>6}")
    for name in DOMAINS:
        print(f"{name:<12} "
              f"{stats['stage3_per_domain_real'].get(name, 0):>6d} "
              f"{stats['stage3_per_domain_augmented'].get(name, 0):>6d} "
              f"{stats['stage3_per_domain_final'].get(name, 0):>6d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
