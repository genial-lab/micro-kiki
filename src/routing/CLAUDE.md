# Routing (Meta-Router + Dispatcher)

Sigmoid meta-router selects which stacks to activate per query.

## Meta-Router

- 32 sigmoid outputs (one per domain), NOT softmax
- Threshold: 0.12 general, 0.20 floor for chat-mode
- Max 4 active stacks (VRAM constraint)
- Trained on domain-classified examples from pipeline

## Dispatcher

- Training-free YAML mapping: router output → 7 meta-intents
- Lives in `configs/`
- Maps combinations of active stacks to high-level intent categories

## Testing

Retest meta-router after every 4 stacks added.
Validate that threshold tuning doesn't regress existing domains.

## Anti-Patterns

- Don't use softmax — domains are not mutually exclusive
- Don't exceed 4 simultaneous active stacks
- Don't hardcode thresholds — load from config
- Don't train router and stacks simultaneously
