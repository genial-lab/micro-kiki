# VQC-Signal-as-Conditioning for Aeon Latent Predictor — Design Doc

**Date**: 2026-04-19
**Status**: design sketch (no implementation yet)
**Branch ownership**: reserved for future `feat/vqc-conditioned-aeon`
**Relates to**: `paper-a-reframe-aeon-ami.md` (§6 future work), PoC B v2 (`docs/poc/2026-04-17-aeon-predictor-poc-alpha.md`), `scripts/benchmark_quantum_router.py`

---

## 1. Motivation

Le PoC B v2 a empiriquement démontré deux choses sur l'`AeonPredictor` conditionné par `stack_id` :

1. **Le conditionnement aide sur flux non-centré** : condition A (baseline, pas de centrage, stack-cond on) atteint `win_stack = 23 %` sur flux stack-structuré.
2. **Il s'effondre dès qu'on active le centrage DinoV3** : condition D (centrage + stack-cond) descend à `win_stack = 0 %`, condition E à `1 %`. Le signal one-hot amplifié par `sqrt(dim/n_stacks)` est lavé par la soustraction de `running_mean`.

L'hypothèse opérationnelle : le signal de conditionnement sparse (1 coordonnée active sur `n_stacks=16`) est **trop pauvre** pour coexister avec un mécanisme de centrage agressif. De plus, un `stack_id` discret n'a aucune structure de similarité — le predictor ne peut pas exploiter le fait que `stack-stm32`, `stack-embedded`, et `stack-platformio` partagent une dynamique latente commune.

**Direction D** propose de remplacer le one-hot par la **distribution soft** (probabilités post-softmax) du VQC router. Ce signal est dense (toutes les coordonnées sont non-nulles), structuré (les domaines sémantiquement proches reçoivent des probabilités proches), et aligné avec le module Configurator de l'architecture AMI de LeCun (arXiv:2206.15331, §3.4). Cela ouvre deux gains :

- **Robustesse au centrage** : un vecteur dense perd moins de signal qu'un one-hot quand on lui soustrait une moyenne courante.
- **Transfert inter-domaines** : le predictor apprend une dynamique par *cluster* de domaines, pas par identité unique.

Cela transforme le VQC de "routeur LoRA discret" en **Configurator multi-consommateur** : sa sortie conditionne désormais à la fois l'activation d'adapters ET la mémoire de travail (Aeon, Module 7). C'est un changement de framing important pour Paper A.

## 2. Current conditioning (baseline)

L'implémentation actuelle vit dans `src/memory/aeon_predictor.py` :

- `_PairSample` (ligne 196-200) stocke `stack_id: int` (valeur `-1` pour "unknown").
- `pairs_for_training()` (ligne 282-287) produit des triples `(h_t, h_{t+1}, stack_id)`.
- `_stack_onehot()` (ligne 365-376) convertit une liste de `stack_id` en matrice `(batch, n_stacks)`, avec `n_stacks=16` par défaut (`PredictorConfig.n_stacks`, ligne 40).
- La matrice est **scalée** par `sqrt(dim/n_stacks)` (ligne 375) pour contrecarrer le washing-out face à `dim=384`.
- `LatentMLP.forward()` (ligne 106-130) concatène `x` (dim=384) avec `stack_onehot` (dim=16) → entrée MLP de taille `dim + n_stacks = 400`.

Limitations du one-hot :

- **Sparsité** : 1 coordonnée active sur 16 → la majorité des neurones de la première couche voient zéro sur l'axe conditionnement.
- **Pas de similarité** : `stack_id=3` et `stack_id=4` sont aussi différents que `stack_id=3` et `stack_id=15` (distance de Hamming identique).
- **Dimensionnalité arbitraire** : `n_stacks=16` est un paramètre libre non corrélé au nombre réel de LoRA stacks (le projet cible 34 niches + base).

## 3. Proposed conditioning (VQC-signal)

Le VQC router actuel (`src/routing/quantum_router.py`) produit, en vérité-terrain et après lecture du code :

- **`n_qubits = 6`**, **`n_layers = 6`**, **`n_classes = 35`** (34 niches triés alphabétiquement + `"base"`, lignes 40-41, 62-64).
- `circuit()` (ligne 126-137) retourne 6 expectations `PauliZ` ∈ [-1, 1].
- `route()` (ligne 143-170) applique une tête linéaire (6 → 35) puis **softmax** (ligne 154, helper `_softmax` ligne 271-275), extrait `argmax` pour la décision, et renvoie un `RouteDecision`.

**Note importante** : la spec de la tâche mentionnait "11 classes, sigmoid". La réalité du code est **35 classes, softmax**. Le routeur historique 11-classes était un prototype antérieur (cf `docs/superpowers/plans/2026-04-17-text-jepa-vqc-router.md` ligne 5). Cette design doc s'aligne sur la version présente à `master`, soit 35 classes softmax. Cette divergence est notée en §8 comme point à clarifier avec l'auteur.

Le design proposé remplace le one-hot par `p ∈ R^{n_router_classes}` (dim=35) issu du softmax du VQC :

```python
# PredictorConfig (nouveau champ)
@dataclass(frozen=True)
class PredictorConfig:
    dim: int
    hidden: int = 256
    horizon: int = 1
    n_stacks: int = 16                # deprecated when use_vqc_cond=True
    n_router_classes: int = 35        # matches QuantumRouterConfig.n_classes
    use_vqc_cond: bool = False        # feature flag, backward-compat
    cold_start_threshold: int = 500
    seed: int = 0
    use_centering: bool = False
    centering_momentum: float = 0.9

# Current
def predict_next(self, h_t, horizon=1, stack_id: int | None = None):
    stack = self._stack_onehot([stack_id or -1])

# Proposed
def predict_next(
    self,
    h_t,
    horizon=1,
    router_probs: np.ndarray | None = None,
):
    if router_probs is None:
        # cold-start / null conditioning: uniform prior
        router_probs = np.full(
            self.config.n_router_classes,
            1.0 / self.config.n_router_classes,
            dtype=np.float32,
        )
    cond = router_probs.reshape(1, -1)
    # ... reste identique, MLP.forward(x, cond) au lieu de (x, stack)
```

L'entrée du MLP devient `dim + n_router_classes = 384 + 35 = 419` (vs 400 pour l'actuel). Le scaling `sqrt(dim/n_router_classes)` reste pertinent pour équilibrer les magnitudes, mais il agit cette fois sur un vecteur dense ; l'amplification peut être plus modeste (probabilités déjà bien réparties).

Gain sémantique attendu : si le VQC met `p[stack-stm32] = 0.45`, `p[stack-embedded] = 0.30`, `p[stack-c-programming] = 0.15`, le predictor voit un signal structuré qui dit "contexte embarqué" au lieu de "classe 17". Sur flux de conversations multi-domaines, deux turns techniquement proches auront des conditionnements proches — même objectif que les embeddings de mots contre les one-hots de vocabulaire.

## 4. Training data alignment

La question pratique : pendant l'entraînement, il faut triples `(h_t, h_{t+1}, router_probs_at_t)`. Trois sources possibles :

**A. Runtime ingestion** — À chaque `ingest_latent()`, on fait tourner le VQC sur `h` et on stocke `router_probs` dans `_PairSample.router_probs`. Coût : une forward VQC (~quelques ms sur simulator classique PennyLane). Négligeable comparé à l'embedding LLM (~50 ms) qui précède l'ingestion.

**B. Offline batch pre-compute** — On rejoue le buffer existant, on recalcule tous les `router_probs` une fois. Pratique pour bootstrapper sur données historiques, mais nécessite que le VQC soit entraîné avant l'ingestion.

**C. Fake-probs depuis stack_id** — Vecteur `0.9` au `stack_id` + `0.1 / (n-1)` ailleurs. Pratique pour tests unitaires, mais **dégénère vers le one-hot pondéré** et ne valide pas l'hypothèse.

**Recommandation : A.** Le coût est absorbable, et cela garantit que `router_probs_at_t` reflète l'état réel du VQC au moment de l'ingestion (pas un VQC différent à l'entraînement vs à l'inférence). Ajout à `_PairSample` :

```python
@dataclass
class _PairSample:
    turn_id: str
    h: np.ndarray
    ts: datetime
    stack_id: int                            # kept for backward-compat eval
    router_probs: np.ndarray | None = None   # (n_router_classes,), new
```

Pour les sessions de test synthétiques qui n'ont pas encore de VQC entraîné, on injectera des `router_probs` **synthétiques** tirées d'un Dirichlet centré sur le "vrai" domaine — mime le comportement d'un VQC imparfait sans exiger qu'il soit entraîné.

## 5. Integration with AMI Configurator narrative

Dans `paper-a-reframe-aeon-ami.md` §3 (ligne 29), le VQC router est mappé au **Module 1 (Configurator)** d'AMI avec une force "Partial — existe, 86.8 % val_acc unbalanced". Dans la v1 du papier, il est mentionné une seule fois §6.5 comme "candidate Configurator for future integration".

Direction D transforme cette note en contribution concrète :

- **Avant D** : Configurator = sélecteur discret de LoRA stack (routing pur).
- **Après D** : Configurator = source d'un **signal de conditionnement continu** consommé par (a) le dispatcher LoRA (existant) ET (b) la mémoire de travail Aeon (nouveau).

C'est une implémentation *multi-consumer* du Configurator, conforme à l'intuition de LeCun (2022, §3.4) selon laquelle le Configurator "modules the parameters of other modules based on task context". Le signal soft est précisément ce que décrit LeCun — pas un gate discret, mais une configuration continue.

**Implications pour Paper A** :

- Workshop (NeurIPS World Models) : garder le scope actuel (Module 7 seul) ; mentionner D comme *future work* dans §6. La section `5.4 Stack-conditioning ablation (honest failure)` devient "§5.4 Stack-conditioning : one-hot fragile, soft VQC probs as recovery path (sketched in §6)".
- Main track (ICLR 2027 / TMLR) : si D est implémenté et bat le baseline, il devient **co-contribution** avec centrage+rollback. Le framing passe de "Module 7 seul" à "Module 1 × Module 7 coupling — a working-memory architecture conditioned by a quantum Configurator".

Paper A spinoff E (documenté ailleurs) présente l'échec de stack-cond comme cautionary tale ; D est le *fix* correspondant. E et D partagent les données empiriques mais argumentent des choses différentes.

## 6. Centering compatibility

C'est la question critique : le centrage DinoV3 qui a tué stack-cond en PoC B v2 va-t-il aussi tuer VQC-cond ?

**Prédiction : oui mais moins gravement.**

Raisons :

- Le centrage s'applique à `out` (post-MLP, `LatentMLP.forward()` ligne 121-127), **pas à l'entrée conditionnement**. L'`inp` concaténé entre bien dans le MLP. Mais la dérivée via backprop « oublie » la direction moyenne du centrage, et cela désaligne gradiant et signal quand le signal de conditionnement est sparse.
- Un vecteur dense (35 coordonnées actives) résiste mieux à cette soustraction de moyenne que 1 coordonnée active sur 16.
- L'argmax du VQC varie d'une turn à l'autre (contrairement à `stack_id` qui est "collant" sur des runs stack-structurés) → le signal est *naturellement* moins aligné avec la moyenne courante, donc moins érodé par sa soustraction.

**Test empirique requis**. C'est une **question ouverte** jusqu'à ce qu'on obtienne :

- Résultats de Direction F (per-stack centering : une `running_mean` par stack_id) — si F marche, D peut s'empiler additivement (per-class centering avec 35 moyennes indexées par `argmax(router_probs)`).
- Résultats d'un A/B condition D-VQC vs D-stack-onehot sur le même flux — on garde centrage actif, on varie uniquement la source de conditionnement.

Si F échoue mais D réussit seul, c'est que le problème n'était pas la math du centrage mais la *sparsité* du signal. Si les deux échouent, la compatibilité centrage-conditionnement exige une refonte plus profonde (par ex. normalisation de `inp` avant centrage de `out`).

## 7. Implementation plan (3 phases)

**Phase 1 — Refactor API (1 semaine)**

- Ajouter `n_router_classes`, `use_vqc_cond` à `PredictorConfig`.
- Ajouter `router_probs: np.ndarray | None` à `_PairSample`.
- Étendre `ingest_latent()` pour accepter `router_probs=None`.
- Introduire `LatentMLP.__init__(cond_dim=...)` (générique) au lieu de `n_stacks` codé en dur.
- Ajouter `_cond_vector(probs_or_stack_ids)` qui choisit entre one-hot et probs soft selon `use_vqc_cond`.
- Conserver le chemin `stack_id` avec `DeprecationWarning` pendant un cycle.
- Tests : unit tests pour les deux chemins, property test "probs uniformes ≡ null condition".

**Phase 2 — PoC avec router_probs synthétiques (1 semaine)**

- Script d'eval `scripts/eval_vqc_conditioned_predictor.py` qui échantillonne des `router_probs` depuis un Dirichlet centré sur le "vrai" domaine (simule un VQC imparfait).
- Refaire les conditions A–E de PoC B v2 avec (a) stack-cond one-hot, (b) VQC-cond synthétique, même flux, même seeds.
- Critère de succès : sur flux stack-structuré, VQC-cond synthétique non-dégénéré (≥ 10 % `win_stack` en condition D-VQC) ET MRR non dégradé vs baseline sans conditionnement.

**Phase 3 — Intégration réelle avec VQC entraîné (1-2 semaines)**

- Dépend de : VQC router entraîné sur le corpus 10-domaines (PoC A Task 14, cf memory `project_microkiki_session_20260416.md`).
- `hybrid_pipeline.py` : câbler la sortie softmax du `QuantumRouter.route()` directement dans `AeonPredictor.ingest_latent(..., router_probs=...)`.
- Eval end-to-end sur conversations réelles (données PoC A Text-JEPA).
- Reporting : un tableau A–E × {stack-cond, VQC-cond} = 10 lignes.

## 8. Risks & open questions

- **Dimension mismatch** : VQC sort 35-d, embedding 384-d. Même risque de washing-out que stack-cond — il faudra probablement garder le scaling `sqrt(dim/n_router_classes)`. À calibrer expérimentalement.
- **Biais inductif continu vs discret** : conditioning continu suppose que le MLP peut exploiter les gradients fins sur les probs. Peut exiger LR plus faible, plus d'epochs, ou un init différent pour la couche 1.
- **Qualité du VQC** : si accuracy ≈ 70 %, `router_probs` est bruité. Deux sous-risques : (a) le predictor apprend à ignorer le signal (effondrement vers baseline), (b) le bruit agit comme régularisation déguisée et aide la généralisation. Incertain sans expérience.
- **Divergence 11 vs 35 classes** : la spec initiale de cette design doc mentionnait "11-class". Le code actuel est à 35. Soit on garde 35 (code-first, cohérent avec `configs/`), soit on aligne sur 11 si l'auteur confirme un VQC réduit pour cette expérience. **Default: 35**, à confirmer avec l'auteur avant de coder.
- **Couplage 35 classes vs 32 LoRA stacks** : la granularité du conditionnement (35 domaines VQC) diffère de la granularité réelle des stacks (32 LoRAs). Cela peut être une *feature* (conditionnement plus sémantique que "quel stack je vais activer") ou un *bug* (le predictor et le dispatcher voient des mondes différents). À creuser.
- **Cold-start** : tant que le VQC n'est pas entraîné, `router_probs` est du bruit. Prévoir un chemin "predictor en cold-start si router en cold-start".

## 9. Success criteria

Barres minimales, héritées de PoC B v2 et éclaircies :

- **Non-régression MRR** : `mrr(D-VQC) ≥ mrr(baseline, condition A)` sur flux random-walk.
- **Win sur flux structuré** : `win_pred(D-VQC) ≥ 60 %` et `win_vs_null(D-VQC) ≥ 20 %` sur flux stack-structuré.
- **Bar D-specific** : `win_stack(D-VQC) > win_stack(D-stack-onehot)` strictement — sinon D ne vaut pas l'effort.
- **Robustesse au centrage** : `win_stack(D-VQC)` ne chute pas de > 50 % entre conditions A (sans centrage) et D (avec centrage). Compare avec la chute observée pour stack-onehot (23 % → 0 % = chute de 100 %).

## 10. Relationship to F (per-stack centering) and E (spinoff paper)

**F (per-stack centering)** et **D (VQC-cond)** sont orthogonaux et additifs :

- F fixe la math du centrage : au lieu d'une `running_mean` globale, on tient `n_stacks` moyennes indexées par stack_id. Cela préserve le signal one-hot en ne soustrayant que la composante intra-classe.
- D change la source du signal : probs soft denses au lieu de one-hot sparse.

Les deux peuvent se combiner : **per-class centering** où "class" = `argmax(router_probs)`. On tient 35 moyennes, on soustrait celle correspondant à la classe dominante du turn courant. C'est l'extension naturelle si F marche.

**E (spinoff paper / cautionary note)** documente l'échec de stack-cond one-hot sous centrage. D propose le *fix*. Ils partagent les mêmes données empiriques (PoC B v2 table A–E) mais argumentent :

- E : "Sparse conditioning breaks under aggressive distribution shifts. Here's why, with ablations."
- D : "Dense, semantically structured conditioning survives. Here's a VQC-based instantiation."

**Stratégie de publication** :

- Workshop Paper A : cite E en note, mentionne D en §6 future work.
- Main-track Paper A (ICLR 2027) : si D est implémenté et battant, D devient **co-contribution centrale**. E reste une sous-section dans §5 (ablations) pour montrer *pourquoi* on a pivoté.
- Paper B (spinoff Configurator) : D devient contribution principale, avec une étude approfondie de la structure du VQC comme source de conditionnement (sparsity, temperature, top-k truncation).

---

**Recommandation sprint courant vs prochain** : **prochain sprint**. Phase 1 + Phase 2 (synthétique) est un effort 2 semaines pas directement sur le chemin critique de la soumission workshop (4 semaines). Si l'auteur vise le workshop NeurIPS 2026 World Models en priorité, mentionner D en §6 future work et garder ce design doc comme ancrage. Si l'on pivote vers main-track ICLR 2027, alors D devient critique et devrait démarrer dès que F aura livré ses premiers résultats.
