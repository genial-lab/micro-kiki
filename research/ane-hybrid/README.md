# ANE Hybrid Pipeline — Qwen3.5-35B-A3B sur Apple Neural Engine

Recherche pour faire tourner un MoE hybride DeltaNet sur ANE.
Voir le plan complet : `docs/plans/2026-04-14-ane-hybrid-pipeline.md`

## Fichiers

| Fichier | Phase | Role |
|---------|-------|------|
| `deltanet_reference.py` | 1.1 | Reference PyTorch recurrent + chunkwise |
| `deltanet_conv2d.py` | 1.2 | Version Conv2d pour ANEMLL (a creer) |
| `convert_deltanet.py` | 1.3 | Conversion CoreML (a creer) |
| `test_deltanet_ane.py` | 1.4 | Test execution ANE (a creer) |

## Usage

```bash
source .venv/bin/activate
cd research/ane-hybrid
python deltanet_reference.py  # Verifie equivalence recurrent vs chunkwise
```
