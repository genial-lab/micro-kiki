"""Example: Aeon Memory Palace write + recall."""
import hashlib

import numpy as np

from src.memory.aeon import AeonPalace


def _hash_embed(text: str) -> np.ndarray:
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
    vec = rng.randn(384).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


aeon = AeonPalace(dim=384, embed_fn=_hash_embed)

e1 = aeon.write("ESP32-S3 I2C requires external pull-up resistors", domain="embedded")
e2 = aeon.write("I2C bus speed: 100kHz standard, 400kHz fast mode", domain="embedded", links=[e1])

results = aeon.recall("I2C pull-up resistors", top_k=2)
for ep in results:
    print(f"[{ep.domain}] {ep.content}")

print(f"Memory stats: {aeon.stats}")
