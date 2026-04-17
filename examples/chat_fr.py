"""Example: Chat in French with micro-kiki."""
from src.routing.dispatcher import dispatch, load_intent_mapping

mapping = load_intent_mapping("configs/meta_intents.yaml")

# Simulate router output where chat-fr (idx 0) is dominant
logits = [0.05] * 35
logits[0] = 0.92

result = dispatch(logits, mapping)
print(f"Intent: {result.intent.value}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Active domains: {result.active_domains}")
