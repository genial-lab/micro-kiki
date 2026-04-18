# Devis — Déploiement Edge AI (GenioBoard + VENTUNO Q)

**Devis N°** : 2026-04-19-HW-EDGE-01
**Date** : 2026-04-19
**Émetteur** : L'Electron Rare (micro-kiki / Factory 4 Life R&D)
**Destinataire** : _(à compléter selon présentation — partenaire, investisseur, comité interne)_
**Validité** : 30 jours
**Contexte** : scouting edge deployment pour Aeon memory predictor + micro-kiki routing stack, réf. `docs/research/edge-deployment-genio-ventunoq.md`.

---

## 1. Synthèse en 1 ligne

Deux scénarios chiffrés : **A (Minimum Viable)** = 1 dev kit + port basique, **B (Proof of Concept)** = 2 dev kits + benchmark complet + résultats Paper A.

---

## 2. Devis A — Minimum Viable (1 board, GenioBoard)

Objectif : valider sur UNE plateforme que Aeon + VQC tournent à latence acceptable ; 2 semaines d'engineering.

### 2.1 Matériel

| Ligne | Désignation | Fournisseur | Qté | PU HT | Total HT |
|-------|-------------|-------------|-----|-------|----------|
| 1 | Grinn GenioBoard 700 (GGC0.700.4G.16G.E) 4GB/16GB | DigiKey | 1 | 320 € | 320 € |
| 2 | Alim USB-C PD 45 W + câble USB-C 2 m | Mouser / DigiKey | 1 | 25 € | 25 € |
| 3 | Câble Ethernet Cat6 2 m | générique | 1 | 8 € | 8 € |
| 4 | MicroSD 128 GB Sandisk Extreme Pro | Amazon / Mouser | 1 | 22 € | 22 € |
| 5 | Câble HDMI 2 m | générique | 1 | 10 € | 10 € |
| 6 | Boîtier impression 3D custom (optionnel) | interne | 1 | 15 € | 15 € |
| — | **Sous-total matériel HT** | | | | **400 €** |
| — | Frais de port DigiKey (US→FR, DHL) | DigiKey | — | — | 60 € |
| — | TVA import 20% (France) sur ligne 1 + port | — | — | — | 76 € |
| — | **TOTAL matériel TTC** | | | | **536 €** |

### 2.2 Engineering

| Ligne | Désignation | Qté | Taux | Total HT |
|-------|-------------|-----|------|----------|
| E1 | Setup OS (Yocto/Ubuntu), Python 3.13 venv, deps port | 8 h | 100 €/h | 800 € |
| E2 | Port Aeon predictor + tests unitaires sur board | 8 h | 100 €/h | 800 € |
| E3 | Port VQC router PennyLane + vérif | 6 h | 100 €/h | 600 € |
| E4 | Latency benchmarks (predict_next, recall, VQC) | 8 h | 100 €/h | 800 € |
| E5 | Rapport technique (2-3 pages, numbers réels vs projetés) | 4 h | 100 €/h | 400 € |
| — | **Sous-total engineering HT** | **34 h** | | **3 400 €** |
| — | **TVA 20% sur engineering** | | | 680 € |
| — | **TOTAL engineering TTC** | | | **4 080 €** |

### 2.3 Grand Total Devis A

| Rubrique | Montant |
|----------|---------|
| Matériel TTC | 536 € |
| Engineering TTC | 4 080 € |
| **TOTAL Devis A TTC** | **4 616 €** |

**Délai** : 2 semaines (1 semaine réception matériel + 1 semaine port/bench).

---

## 3. Devis B — Proof of Concept complet (2 boards, comparatif)

Objectif : valider Aeon sur **les 2 plateformes** (GenioBoard + VENTUNO Q), benchmark comparatif, chiffres intégrables en Paper A Appendix B, 4 semaines.

### 3.1 Matériel

| Ligne | Désignation | Fournisseur | Qté | PU HT | Total HT |
|-------|-------------|-------------|-----|-------|----------|
| 1 | Grinn GenioBoard 700 (GGC0.700.4G.16G.E) | DigiKey | 1 | 320 € | 320 € |
| 2 | Arduino VENTUNO Q (Dragonwing IQ8 + STM32H5) | Arduino Store / DigiKey / Farnell | 1 | 270 € | 270 € |
| 3 | Alim USB-C PD 45 W ×2 + câbles | Mouser | 2 | 25 € | 50 € |
| 4 | Câble Ethernet Cat6 2 m ×2 | générique | 2 | 8 € | 16 € |
| 5 | MicroSD 128 GB Sandisk Extreme Pro ×2 | Amazon | 2 | 22 € | 44 € |
| 6 | Câble HDMI 2 m ×2 | générique | 2 | 10 € | 20 € |
| 7 | Sonde USB JTAG (STM32, VENTUNO MCU debug) | ST-LINK/V3 Mouser | 1 | 35 € | 35 € |
| 8 | Boîtiers impression 3D custom ×2 | interne | 2 | 15 € | 30 € |
| 9 | Wattmètre USB-C (mesure conso edge) | Amazon (YZXstudio ZY1271) | 1 | 55 € | 55 € |
| — | **Sous-total matériel HT** | | | | **840 €** |
| — | Frais de port DigiKey (US→FR) | DigiKey | — | — | 80 € |
| — | Frais de port Arduino Store (IT→FR) | Arduino | — | — | 20 € |
| — | TVA import 20% (France) | — | — | — | 188 € |
| — | **TOTAL matériel TTC** | | | | **1 128 €** |

### 3.2 Engineering

| Ligne | Désignation | Qté | Taux | Total HT |
|-------|-------------|-----|------|----------|
| E1 | Setup OS × 2 boards | 12 h | 100 €/h | 1 200 € |
| E2 | Port Aeon + VQC + tests sur les 2 boards | 20 h | 100 €/h | 2 000 € |
| E3 | Port micro-LLM (Qwen3-3B Q4 via QNN / MediaTek NNAPI) | 20 h | 100 €/h | 2 000 € |
| E4 | Benchmark latence + conso (3 charges : Aeon only, VQC, LLM) | 16 h | 100 €/h | 1 600 € |
| E5 | Dual-brain split VENTUNO Q (MPU/MCU RPC bridge demo) | 12 h | 100 €/h | 1 200 € |
| E6 | Rapport technique complet + Paper A Appendix B update (chiffres réels) | 8 h | 100 €/h | 800 € |
| — | **Sous-total engineering HT** | **88 h** | | **8 800 €** |
| — | **TVA 20% sur engineering** | | | 1 760 € |
| — | **TOTAL engineering TTC** | | | **10 560 €** |

### 3.3 Grand Total Devis B

| Rubrique | Montant |
|----------|---------|
| Matériel TTC | 1 128 € |
| Engineering TTC | 10 560 € |
| **TOTAL Devis B TTC** | **11 688 €** |

**Délai** : 4 semaines à partir du lancement (2 semaines VENTUNO Q Q2 2026 → attendre dispo effective, 2 semaines port + bench).

---

## 4. Comparatif rapide Devis A vs B

| Critère | Devis A | Devis B |
|---------|---------|---------|
| Plateformes testées | 1 (GenioBoard) | 2 (GenioBoard + VENTUNO Q) |
| Durée | 2 semaines | 4 semaines |
| Livrables | Latency bench 1 plateforme | Bench comparatif + LLM local + dual-brain demo |
| Risque dispo | Faible (DigiKey stock) | Moyen (VENTUNO Q dépend Q2 2026) |
| Paper A Appendix B | Partiel | Complet, chiffres réels |
| **TTC** | **4 616 €** | **11 688 €** |

---

## 5. Notes et hypothèses

1. **Prix matériel** : basés sur recherches publiques 2026-04-19. GenioBoard 700 4GB/16GB à $338.12 chez DigiKey (~320 € HT après change). VENTUNO Q annoncé sub-$300 (~270 € HT). Prix ferme à valider à la commande.
2. **Taux engineering** : 100 €/h HT = freelance ingé Python embarqué sénior, région FR. Ajuster selon profil interne / externe.
3. **TVA** : France 20%. Si présentation HORS France, remplacer par le taux local.
4. **Shipping DigiKey** : ~60-80 € en DHL express 3-5 jours. Droits de douane potentiels selon valeur déclarée (généralement inclus via TVA import).
5. **Validité prix** : les prix matériel peuvent fluctuer ±10% selon stock / change EUR/USD. Re-devis à J-7 avant commande si commande décalée.
6. **VENTUNO Q Q2 2026** : disponibilité officielle Arduino Store Q2 2026 (avril-juin). Si décision avant dispo, passer commande en waitlist pour être dans la première vague. Fallback : reporter le volet VENTUNO Q du Devis B si retard.
7. **Hors périmètre** : infrastructure cloud (CI/CD, logs), formation utilisateurs, support post-livraison. Devisable séparément.
8. **Propriété intellectuelle** : code Aeon/micro-kiki sous Apache 2.0 (livrable), rapport technique sous CC-BY 4.0 (cite-able publiquement).

---

## 6. Modalités

- **Devis valide 30 jours** à compter du 2026-04-19, expire 2026-05-19.
- **Acompte 30%** à la commande, solde à la livraison du rapport final.
- **Mode de règlement** : virement bancaire, RIB sur demande.
- **Lieu d'exécution** : GrosMac (M5) pour dev + benchmark, livraison dématérialisée (rapport + repo git).

---

## 7. Sources prix

- [Grinn GenioBoard 700 on DigiKey (GENIO-700-EVK)](https://www.digikey.com/en/products/detail/mediatek/GENIO-700-EVK/24633568)
- [Grinn GenioBoard official store](https://www.genioboard.com/)
- [Arduino VENTUNO Q pricing "under $300" announcement (Arduino Blog)](https://blog.arduino.cc/2026/03/09/introducing-arduino-ventuno-q-your-new-ai-robotics-and-actuation-platform/)
- [VENTUNO Q engadget coverage](https://www.engadget.com/ai/qualcomms-new-arduino-ventuno-q-is-an-ai-focused-computer-designed-for-robotics-113047697.html)
- [TheOutpost VENTUNO Q pricing note "under $300"](https://theoutpost.ai/news-story/qualcomm-s-arduino-ventuno-q-brings-ai-and-robotics-together-on-a-single-board-under-300-24400/)
- [OLogic Pumpkin Genio 700 alternative vendor](https://ologic.store/products/pumpkin-genio-700)
- [Youyeetoo X8390ABV4 MT8370/MT8390 alternative](https://www.youyeetoo.com/products/x8390abv4-mediatek-mt8370-mt8390-edge-ai-iot-board)

---

## 8. Ce que ce devis N'inclut PAS (à chiffrer séparément si besoin)

- **Certification CRA / CE** : GenioBoard est annoncé CRA-ready mais la certification produit end-user est à part
- **Industrialisation** (boîtier IP65, DFM, EMC testing) : +15-30 k€ selon volume
- **Volume / OEM** : remise typique -10 à -25 % sur 10+ unités chez DigiKey / Mouser
- **Support post-livraison** : contrat SLA à devisager séparément
- **Licences tierces** : MediaTek SDK, Qualcomm QNN — tous gratuits pour dev, à revérifier en prod

---

**Signataire** : L'Electron Rare
**Contact** : contact@saillant.cc
**Repo** : `micro-kiki` (privé, accès partenaire sur demande)
