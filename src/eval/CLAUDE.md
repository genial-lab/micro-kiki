# Evaluation & Forgetting Checks

Benchmarks, forgetting detection, and bias monitoring.

## Forgetting Framework

- Run after EACH new stack is trained
- Metrics: weight angle (base vs adapted), win-rate on held-out set
- Rollback trigger: angle < 30° AND win-rate drop > 0.03
- Stacks 02-03 get retroactive baseline check when framework activates (step 14)

## Benchmarks

- mt-bench-french (`bofenghuang/mt-bench-french`)
- FrenchBench collection (`manu`)
- Per-domain accuracy on classified test splits

## Adaptive Judge

- First pass: Qwen3.5-35B (fast, cheap) on kxkm-ai
- Escalate to Mistral-Large-Opus (Studio) only if confidence < 0.5
- Never skip the cheap pass

## Bias Monitoring

- KnowBias neuron debiasing: before + after stacks
- RBD runtime detector on every response
- If bias flagged → re-generate via DeFrame

## Anti-Patterns

- Don't evaluate with the same data used for training
- Don't skip bias check — it runs on EVERY response
- Don't use expensive judge when cheap judge is confident
- Don't compare stacks trained at different quantization levels
