# Recherche ANE Hybrid

Pipeline hybride ANE+Metal pour Qwen3.5-35B-A3B sur Apple Neural Engine.

## Phases

| Phase | Fichier | Resultat |
|-------|---------|----------|
| 1.1 Reference PyTorch | `deltanet_reference.py` | diff=0.0 |
| 1.2 Conv2d ANEMLL | `deltanet_conv2d.py` | diff=6.64e-15 |
| 1.3 CoreML conversion | `convert_deltanet.py` | .mlpackage generes |
| 1.4 Vrais poids ANE | `deltanet_real.py` | 474 tok/s/couche |
| 2 Stack 40 couches | `phase2_full_stack.py` | 14.4 tok/s ANE |
| 3 Hybrid ANE+CPU | `phase3_moe_hybrid.py` | 9.9 tok/s |
| 3b GPU experts | `phase3b_gpu_experts.py` | 5.7 tok/s pipeline |
| MLX pur | `mlx_pure_full_model.py` | 14.2 tok/s complet |

## Verdict

MLX pur (14.2 tok/s) > ANE hybrid (5.7-9.9 tok/s) sur M3 Ultra.
mlx-vlm natif : 45-89 tok/s (meilleur de loin).
ANE utile seulement si GPU occupe (14 tok/s ANE pendant training).

## DeltaNet Architecture

- 30 couches Gated DeltaNet (attention lineaire recurrente)
- 10 couches Full Attention (GQA 16Q/2KV, head_dim=256)
- State: [1, 32, 128, 128] par couche DeltaNet
- Conv1d depthwise kernel=4

## Anti-Patterns

- CoreML dispatch overhead (~2ms/couche) domine le pipeline hybride
- numpy↔MLX marshalling annule le gain du pipeline parallele
- Qwen3.5 DeltaNet produit du garbage via ANEMLL (architecture non supportee)
