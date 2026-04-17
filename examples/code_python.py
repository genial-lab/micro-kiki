"""Example: Python code generation routing."""
from src.routing.dispatcher import dispatch, load_intent_mapping

mapping = load_intent_mapping("configs/meta_intents.yaml")

logits = [0.05] * 35
logits[2] = 0.88  # python domain

result = dispatch(logits, mapping)
print(f"Intent: {result.intent.value}")  # coding
print(f"Active domains: {result.active_domains}")
