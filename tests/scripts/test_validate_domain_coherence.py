"""validate_domain_coherence exits 0 when NICHE_DOMAINS, router weights,
and the local adapters directory are aligned - and non-zero with a
precise diagnostic when any pair drifts."""
from __future__ import annotations

import json
import struct
from pathlib import Path


def _write_fake_router(tmp_path: Path, out_dim: int) -> Path:
    out_dir = tmp_path / "output" / "router-v4"
    out_dir.mkdir(parents=True)
    weights_path = out_dir / "router.safetensors"
    header = {
        "2.weight": {
            "dtype": "F32",
            "shape": [out_dim, 512],
            "data_offsets": [0, out_dim * 512 * 4],
        }
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with weights_path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(b"\x00" * (out_dim * 512 * 4))

    (out_dir / "meta.json").write_text(json.dumps({"num_domains": out_dim}) + "\n")
    (out_dir / "label_map.json").write_text(
        json.dumps({f"d{i}": i for i in range(out_dim)}) + "\n"
    )
    return out_dir


def _write_fake_adapters(tmp_path: Path, domains: list[str]) -> Path:
    adapters_root = tmp_path / "adapters"
    for d in domains:
        dom_dir = adapters_root / d
        dom_dir.mkdir(parents=True)
        (dom_dir / "adapters.safetensors").write_bytes(b"dummy")
        (dom_dir / "adapter_config.json").write_text("{}\n")
    return adapters_root


def test_passes_when_aligned(tmp_path, monkeypatch):
    from scripts import validate_domain_coherence as v

    domains = [f"d{i}" for i in range(3)]
    monkeypatch.setattr(v, "NICHE_DOMAINS", frozenset(domains))
    router = _write_fake_router(tmp_path, out_dim=3)
    adapters = _write_fake_adapters(tmp_path, domains)

    rc = v.validate(
        router_path=router / "router.safetensors",
        meta_path=router / "meta.json",
        adapters_root=adapters,
    )
    assert rc == 0


def test_fails_on_router_shape_mismatch(tmp_path, monkeypatch, capsys):
    from scripts import validate_domain_coherence as v

    domains = [f"d{i}" for i in range(3)]
    monkeypatch.setattr(v, "NICHE_DOMAINS", frozenset(domains))
    router = _write_fake_router(tmp_path, out_dim=2)
    adapters = _write_fake_adapters(tmp_path, domains)

    rc = v.validate(
        router_path=router / "router.safetensors",
        meta_path=router / "meta.json",
        adapters_root=adapters,
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert (
        "router output" in captured.out.lower()
        or "router output" in captured.err.lower()
    )


def _write_fake_multi_layer_router(tmp_path: Path, out_dim: int) -> Path:
    """Router with 0.weight (hidden, shape (512, 384)) + 2.weight (head,
    shape (out_dim, 512)) - mirrors the real router-v4 Sequential layout."""
    out_dir = tmp_path / "output" / "router-v4"
    out_dir.mkdir(parents=True)
    weights_path = out_dir / "router.safetensors"
    hidden_bytes = 512 * 384 * 4
    head_bytes = out_dim * 512 * 4
    header = {
        "0.weight": {
            "dtype": "F32",
            "shape": [512, 384],
            "data_offsets": [0, hidden_bytes],
        },
        "2.weight": {
            "dtype": "F32",
            "shape": [out_dim, 512],
            "data_offsets": [hidden_bytes, hidden_bytes + head_bytes],
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with weights_path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(b"\x00" * (hidden_bytes + head_bytes))

    (out_dir / "meta.json").write_text(json.dumps({"num_domains": out_dim}) + "\n")
    return out_dir


def test_picks_last_layer_weight_not_first(tmp_path, monkeypatch, capsys):
    """Router Sequential has 0.weight (hidden) + 2.weight (head). The
    validator must read 2.weight's shape[0], not 0.weight's."""
    from scripts import validate_domain_coherence as v

    domains = [f"d{i}" for i in range(3)]
    monkeypatch.setattr(v, "NICHE_DOMAINS", frozenset(domains))
    router = _write_fake_multi_layer_router(tmp_path, out_dim=3)
    adapters = _write_fake_adapters(tmp_path, domains)

    rc = v.validate(
        router_path=router / "router.safetensors",
        meta_path=router / "meta.json",
        adapters_root=adapters,
    )
    assert rc == 0, capsys.readouterr().out


def test_fails_on_missing_adapter_dir(tmp_path, monkeypatch, capsys):
    from scripts import validate_domain_coherence as v

    domains = [f"d{i}" for i in range(3)]
    monkeypatch.setattr(v, "NICHE_DOMAINS", frozenset(domains))
    router = _write_fake_router(tmp_path, out_dim=3)
    adapters = _write_fake_adapters(tmp_path, domains[:2])

    rc = v.validate(
        router_path=router / "router.safetensors",
        meta_path=router / "meta.json",
        adapters_root=adapters,
    )
    captured = capsys.readouterr()
    assert rc != 0
    assert "d2" in captured.out or "d2" in captured.err
