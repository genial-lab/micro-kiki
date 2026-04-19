# Security policy — micro-kiki

micro-kiki is a **research / deployable MoE-LoRA artifact** within
the dreamOfkiki research program. Correctness of the cognitive
layer and reproducibility of the inference pipeline are our top
priorities ; security is handled on a best-effort basis.

## Scope

This policy applies to :

- The code in this repository (`genial-lab/micro-kiki`)
- The LoRA adapters, router weights, and quantized model checkpoints
  released from this repository
- The documentation in `docs/` and the inference / deployment scripts

It does *not* apply to the upstream Qwen3.5 base model ;
please report upstream.

## Reporting a vulnerability

If you discover a security issue that could compromise :

- the determinism of inference under fixed seeds (R1-equivalent
  contract for the cognitive layer)
- the integrity of released LoRA / Q4_K_M quantized weights
  (mismatched SHA-256, tampered tensors)
- prompt-injection paths that bypass the router or the cognitive
  layer in unexpected ways (beyond standard LLM vulnerabilities)
- data-extraction risks specific to the domain experts' training
  corpora

please report it **privately** via one of these channels :

1. Email : `clement@saillant.cc` — subject starting with `[SECURITY]`
2. GitHub Private Vulnerability Reporting :
   https://github.com/genial-lab/micro-kiki/security/advisories/new

Please include :

- a description of the issue
- reproduction steps (with seed, commit SHA, quantization variant)
- affected artifact (repo commit or HuggingFace model card)
- suggested mitigation if available

We aim to acknowledge reports within **5 business days** and publish
a fix within **30 days** for critical issues.

## Out of scope

- Generic LLM jailbreaks that affect any Qwen3.5 derivative —
  upstream responsibility.
- Hardware-specific timing attacks on the MLX / CUDA inference path.
- Feature requests or quality-of-answer complaints — use GitHub issues.

## Threat model

We defend against *inadvertent artifact corruption and
reproducibility regressions* and against *undisclosed
bypasses of the cognitive-layer filters*. Malicious contributors
with write access are outside the threat model ; review happens
via PR discipline and CI.
