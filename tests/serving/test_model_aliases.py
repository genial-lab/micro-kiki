from src.serving.model_aliases import (
    ModelAlias, build_aliases, lookup, META_PREFIX, NICHE_PREFIX,
)


def test_prefixes():
    assert META_PREFIX == "kiki-meta-"
    assert NICHE_PREFIX == "kiki-niche-"


def test_build_aliases_shape():
    aliases = build_aliases()
    meta = [a for a in aliases if a.mode == "meta"]
    niche = [a for a in aliases if a.mode == "niche"]
    # Authoritative counts come from the repo itself.
    # Adjust these assertions ONLY if the source-of-truth data has changed.
    assert len(meta) == 7, f"expected 7 meta intents, got {len(meta)}"
    assert len(niche) == 34, f"expected 34 niches, got {len(niche)}"
    assert len(aliases) == 41


def test_build_aliases_ids_well_formed():
    aliases = build_aliases()
    for a in aliases:
        assert isinstance(a, ModelAlias)
        assert a.model_id.startswith(META_PREFIX) or a.model_id.startswith(NICHE_PREFIX)
        assert a.mode in {"meta", "niche"}
        assert a.target and " " not in a.target


def test_build_aliases_unique():
    aliases = build_aliases()
    ids = [a.model_id for a in aliases]
    assert len(ids) == len(set(ids)), "alias IDs must be unique"


def test_lookup_by_model_id():
    a = lookup("kiki-niche-stm32")
    assert a is not None
    assert a.mode == "niche"
    assert a.target == "stm32"
    assert lookup("not-a-kiki-model") is None
    # We don't hardcode a meta lookup here because the exact meta intent names
    # depend on configs/meta_intents.yaml — test the structure via shape test.
