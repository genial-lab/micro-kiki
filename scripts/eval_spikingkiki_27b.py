#!/usr/bin/env python3
"""Compare SpikingKiki-27B vs Qwen3.5-27B baseline.

Story 20: Validate LAS conversion is lossless by comparing outputs
of SpikingKiki-27B (standard forward pass, ignoring LIF metadata)
against the original Qwen3.5-27B base model.

Expected: >=95% output similarity (LAS rescales weights but should
produce identical logits through standard transformers inference).
"""
import json
import sys
import time
from pathlib import Path
from difflib import SequenceMatcher

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SPIKING = "/Users/clems/models/spikingkiki-27b"
BASELINE = "/Users/clems/models/qwen3.5-27b"
RESULTS_DIR = Path("/Users/clems/results")

# 50 diverse prompts: EN + FR, math + code + reasoning + general knowledge
PROMPTS = [
    # General knowledge (EN)
    "Explain quantum computing in 3 sentences.",
    "What causes the northern lights?",
    "Describe the water cycle in simple terms.",
    "What is the difference between mitosis and meiosis?",
    "How does a transistor work?",
    # General knowledge (FR)
    "Qu'est-ce que la photosynthèse ?",
    "Expliquez le fonctionnement d'un moteur électrique.",
    "Quelles sont les causes du réchauffement climatique ?",
    "Décrivez le système solaire en 3 phrases.",
    "Comment fonctionne un réseau de neurones artificiels ?",
    # Code
    "Write a Python function to reverse a string.",
    "Write a Python function to check if a number is prime.",
    "Implement binary search in Python.",
    "Write a JavaScript function to flatten a nested array.",
    "Write a C function to compute the factorial of n.",
    # Math / Logic
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "Solve for x: 2x + 5 = 17",
    "What is the integral of sin(x) dx?",
    "If a train travels at 60 km/h for 2.5 hours, how far does it go?",
    "What is 17 * 23?",
    # Reasoning
    "If all cats are animals, and some animals are pets, can we conclude all cats are pets?",
    "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?",
    "Three boxes contain apples, oranges, or both. All labels are wrong. How to fix them with one pick?",
    "Explain the trolley problem and its ethical implications.",
    "What logical fallacy is: 'Everyone believes it, so it must be true'?",
    # Creative
    "Write a haiku about artificial intelligence.",
    "Suggest three names for a coffee shop in Paris.",
    "Describe a sunset over the ocean in two sentences.",
    "Write a one-paragraph story about a robot discovering music.",
    "Invent a new word and define it.",
    # Technical / Engineering
    "What is the difference between TCP and UDP?",
    "Explain how HTTPS encryption works.",
    "What is a PID controller and where is it used?",
    "Describe the CAP theorem in distributed systems.",
    "What is the difference between SRAM and DRAM?",
    # Science
    "What is Heisenberg's uncertainty principle?",
    "Explain how CRISPR gene editing works.",
    "What is dark matter?",
    "How do vaccines work?",
    "What is entropy in thermodynamics?",
    # Multilingual reasoning (FR)
    "Résolvez: Si Pierre a 3 pommes de plus que Marie, et qu'ensemble ils en ont 11, combien en a Marie ?",
    "Quel est le principe de conservation de l'énergie ?",
    "Expliquez la différence entre courant alternatif et courant continu.",
    "Qu'est-ce qu'un algorithme de tri par fusion (merge sort) ?",
    "Décrivez le fonctionnement d'une diode LED.",
    # Edge cases
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    "What is 0 divided by 0?",
    "List the first 10 Fibonacci numbers.",
    "What is the capital of Australia?",
    "Explain recursion using a simple example.",
]

# Use first 10 prompts (MPS on M3 Ultra should give ~5-15 tok/s for 27B)
NUM_PROMPTS = 10


def run_eval():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    total_start = time.time()

    for model_path, label in [(BASELINE, "baseline"), (SPIKING, "spikingkiki")]:
        print(f"\n{'='*60}")
        print(f"Loading {label} from {model_path}...")
        model_start = time.time()

        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Load to CPU first, then move to MPS (avoids placeholder storage error)
        use_mps = torch.backends.mps.is_available()
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if use_mps:
            print("  Moving model to MPS...")
            model = model.to("mps")
            print("  Model on MPS")
        load_time = time.time() - model_start
        print(f"  Loaded in {load_time:.1f}s")

        for i, prompt in enumerate(PROMPTS[:NUM_PROMPTS]):
            gen_start = time.time()
            inputs = tok(prompt, return_tensors="pt")
            if use_mps:
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,  # greedy for reproducibility
                )
            text = tok.decode(output[0], skip_special_tokens=True)
            gen_time = time.time() - gen_start
            new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
            tok_per_sec = new_tokens / gen_time if gen_time > 0 else 0

            results.append({
                "model": label,
                "prompt_idx": i,
                "prompt": prompt,
                "output": text,
                "gen_time_s": round(gen_time, 2),
                "new_tokens": int(new_tokens),
                "tok_per_sec": round(tok_per_sec, 2),
            })

            if (i + 1) % 5 == 0 or i == 0:
                print(f"  [{label}] {i+1}/{NUM_PROMPTS} done "
                      f"({gen_time:.1f}s, {tok_per_sec:.1f} tok/s)")
                sys.stdout.flush()

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  {label} total inference: {time.time() - model_start - load_time:.1f}s")

    # Save raw results
    results_file = RESULTS_DIR / "spikingkiki-27b-eval.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {results_file}")

    # Compare outputs
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")

    baseline_outputs = {
        r["prompt_idx"]: r["output"] for r in results if r["model"] == "baseline"
    }
    spiking_outputs = {
        r["prompt_idx"]: r["output"] for r in results if r["model"] == "spikingkiki"
    }

    similarities = []
    exact_matches = 0
    for idx in sorted(baseline_outputs.keys()):
        if idx not in spiking_outputs:
            continue
        b_out = baseline_outputs[idx]
        s_out = spiking_outputs[idx]
        sim = SequenceMatcher(None, b_out, s_out).ratio()
        similarities.append(sim)

        if b_out == s_out:
            exact_matches += 1

        # Print first 3 and any divergent comparisons
        if idx < 3 or sim < 0.90:
            prompt_short = PROMPTS[idx][:50]
            print(f"\n--- Prompt {idx}: \"{prompt_short}...\"")
            print(f"  Similarity: {sim:.2%}")
            # Show just the generated part (strip prompt from output)
            b_gen = b_out[len(PROMPTS[idx]):].strip()[:100]
            s_gen = s_out[len(PROMPTS[idx]):].strip()[:100]
            print(f"  Baseline:    {b_gen}")
            print(f"  SpikingKiki: {s_gen}")

    avg_sim = sum(similarities) / len(similarities) if similarities else 0
    min_sim = min(similarities) if similarities else 0
    max_sim = max(similarities) if similarities else 0

    total_time = time.time() - total_start

    summary = {
        "num_prompts": NUM_PROMPTS,
        "avg_similarity": round(avg_sim, 4),
        "min_similarity": round(min_sim, 4),
        "max_similarity": round(max_sim, 4),
        "exact_matches": exact_matches,
        "exact_match_rate": round(exact_matches / len(similarities), 4) if similarities else 0,
        "total_time_s": round(total_time, 1),
        "lossless_verified": avg_sim >= 0.95,
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Prompts evaluated: {NUM_PROMPTS}")
    print(f"  Average similarity: {avg_sim:.2%}")
    print(f"  Min similarity:     {min_sim:.2%}")
    print(f"  Max similarity:     {max_sim:.2%}")
    print(f"  Exact matches:      {exact_matches}/{len(similarities)}")
    print(f"  Total time:         {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  LAS lossless:       {'YES' if summary['lossless_verified'] else 'NO'}")

    summary_file = RESULTS_DIR / "spikingkiki-27b-eval-summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    return summary


if __name__ == "__main__":
    summary = run_eval()
    sys.exit(0 if summary["lossless_verified"] else 1)
