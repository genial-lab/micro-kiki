#!/usr/bin/env python3
"""Classifier parallèle — utilise tous les CPU cores."""
import json, re, yaml, time, sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

CONFIG_PATH = "configs/micro_kiki/domains.yaml"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
domains = config["domains"]

domain_data = {}
for name, d in domains.items():
    domain_data[name] = {
        "keywords": d.get("keywords", []),
        "patterns": d.get("patterns", []),
        "phase": d["phase"],
    }

def classify_one(text):
    text_lower = text.lower()
    best_domain = None
    best_score = 0
    for name, d in domain_data.items():
        score = 0
        for kw in d["keywords"]:
            if kw.lower() in text_lower:
                score += 1
                if score >= 5: break
        for pat in d["patterns"]:
            if re.search(pat, text, re.IGNORECASE):
                score += 3
                if score >= 15: break
        if score > best_score:
            best_score = score
            best_domain = name
    return best_domain if best_score >= 1 else None

def normalize(raw):
    if "messages" in raw:
        msgs = raw["messages"]
        if len(msgs) >= 2:
            user = next((m.get("content","") for m in msgs if m.get("role") == "user"), None)
            asst = next((m.get("content","") for m in msgs if m.get("role") == "assistant"), None)
            if user and asst:
                asst = re.sub(r'<think>', '<thinking>', asst)
                asst = re.sub(r'</think>', '</thinking>', asst)
                return user, asst
    for uk, ak in [("instruction","output"), ("prompt","response"), ("question","answer")]:
        if uk in raw and ak in raw:
            return str(raw[uk]), str(raw[ak])
    return None, None

def process_batch(batch):
    results = []
    for raw in batch:
        user, asst = normalize(raw)
        if not user or not asst: continue
        text = user + " " + asst
        domain = classify_one(text)
        if domain:
            results.append((domain, {"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": asst}
            ]}))
    return results

if __name__ == "__main__":
    print("Chargement...")
    t0 = time.time()
    all_examples = []

    for d in sorted(Path("data/raw").iterdir()):
        if not d.is_dir(): continue
        count = 0
        for f in d.glob("*.jsonl"):
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        all_examples.append(json.loads(line))
                        count += 1
        for f in d.glob("*.parquet"):
            import pyarrow.parquet as pq
            for row in pq.read_table(str(f)).to_pylist():
                all_examples.append(row)
                count += 1
        print(f"  {d.name}: {count}")

    print(f"Total: {len(all_examples)} en {time.time()-t0:.1f}s")

    n_workers = cpu_count()
    batch_size = len(all_examples) // (n_workers * 2) + 1
    batches = [all_examples[i:i+batch_size] for i in range(0, len(all_examples), batch_size)]

    print(f"Classification: {n_workers} workers, {len(batches)} batches...")
    t0 = time.time()

    domain_examples = {name: [] for name in domains}
    max_per = 3000

    with Pool(n_workers) as pool:
        for i, results in enumerate(pool.imap_unordered(process_batch, batches)):
            for domain, ex in results:
                if len(domain_examples[domain]) < max_per:
                    domain_examples[domain].append(ex)
            done = (i+1) * batch_size
            pct = min(done*100//len(all_examples), 100)
            print(f"  {min(done, len(all_examples))}/{len(all_examples)} ({pct}%)", flush=True)

    elapsed = time.time() - t0
    print(f"Classifié en {elapsed:.1f}s ({len(all_examples)/elapsed:.0f} ex/s)")

    out = Path("data/micro-kiki/classified")
    out.mkdir(parents=True, exist_ok=True)
    for name, exs in domain_examples.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for ex in exs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n=== Résultat ===")
    total = 0
    for name in sorted(domain_examples, key=lambda n: domains[n]["phase"]):
        c = len(domain_examples[name])
        total += c
        status = "OK" if c >= 1000 else "SPARSE" if c > 0 else "VIDE"
        print(f"  Phase {domains[name]['phase']} {name:<16} {c:>5}/2000 [{status}]")
    print(f"\nTotal classifié: {total}")
