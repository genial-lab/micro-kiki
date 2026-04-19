#!/usr/bin/env python3
"""Push all niche adapters to HuggingFace Hub with per-adapter model cards.

Dry-run by default — use --execute to actually push.

Discovers adapters at outputs/stacks/stack-<domain>/adapters.safetensors
and pushes each one to the HF repo under adapters/<domain>/.

Usage:
    uv run python scripts/release_hf.py --repo electron-rare/micro-kiki
    uv run python scripts/release_hf.py --repo electron-rare/micro-kiki --execute
    uv run python scripts/release_hf.py --repo electron-rare/micro-kiki --domain kicad-dsl
    uv run python scripts/release_hf.py --help
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTERS_DIR = PROJECT_ROOT / "outputs" / "stacks"
DEFAULT_MODEL_CARD = PROJECT_ROOT / "MODEL_CARD.md"

# The 10 niche adapter domains (must match train_dpo_niches.py)
NICHE_DOMAINS: list[str] = [
    "kicad-dsl",
    "spice",
    "emc",
    "stm32",
    "embedded",
    "freecad",
    "platformio",
    "power",
    "dsp",
    "electronics",
]


@dataclass(frozen=True)
class ReleaseConfig:
    repo: str
    adapters_dir: str = str(DEFAULT_ADAPTERS_DIR)
    model_card: str = str(DEFAULT_MODEL_CARD)
    execute: bool = False
    private: bool = False
    domains: list[str] = field(default_factory=lambda: list(NICHE_DOMAINS))


def discover_adapters(adapters_dir: Path, domains: list[str]) -> dict[str, Path]:
    """Discover which adapter stacks exist on disk.

    Returns a mapping of domain -> adapter directory for stacks that
    contain an adapters.safetensors file.
    """
    found: dict[str, Path] = {}
    for domain in domains:
        stack_dir = adapters_dir / f"stack-{domain}"
        adapter_file = stack_dir / "adapters.safetensors"
        if adapter_file.exists():
            found[domain] = stack_dir
    return found


def collect_files(stack_dir: Path) -> list[Path]:
    """Collect all files to upload from a single adapter stack directory."""
    if not stack_dir.exists():
        return []
    files: list[Path] = []
    for path in sorted(stack_dir.rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            files.append(path)
    return files


def generate_adapter_model_card(domain: str, repo: str) -> str:
    """Generate a per-adapter model card for a niche domain."""
    domain_descriptions: dict[str, str] = {
        "kicad-dsl": "KiCad DSL generation (schematics, footprints, PCB layout S-expressions)",
        "spice": "SPICE netlist generation and circuit simulation",
        "emc": "EMC/EMI analysis, compliance, and mitigation strategies",
        "stm32": "STM32 HAL firmware development and peripheral configuration",
        "embedded": "Embedded systems programming (FreeRTOS, bare-metal, drivers)",
        "freecad": "FreeCAD/OpenSCAD parametric 3D modeling scripts",
        "platformio": "PlatformIO project configuration and multi-platform firmware",
        "power": "Power electronics design (converters, regulators, thermal)",
        "dsp": "Digital signal processing algorithms and implementations",
        "electronics": "General electronics design, component selection, and analysis",
    }
    desc = domain_descriptions.get(domain, f"Domain-specific adapter for {domain}")

    return dedent(f"""\
    ---
    license: apache-2.0
    language:
      - fr
      - en
    tags:
      - lora
      - adapter
      - {domain}
      - micro-kiki
      - embedded-systems
    base_model: Qwen/Qwen3.6-35B-A3B
    pipeline_tag: text-generation
    ---

    # micro-kiki / {domain} adapter

    LoRA adapter for **{domain}**: {desc}

    Part of the [micro-kiki]({f"https://huggingface.co/{repo}"}) multi-domain expert model.

    ## Usage

    Load this adapter on top of the Qwen3.6-35B-A3B base model:

    ```python
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.6-35B-A3B")
    model = PeftModel.from_pretrained(base, "{repo}", subfolder="adapters/{domain}")
    ```

    ## Training

    - **Method**: SFT via MLX LoRA on Mac Studio M3 Ultra 512 GB
    - **Base model**: Qwen3.6-35B-A3B (MoE, 256 experts, 3B active)
    - **Quantization**: BF16 training, Q4_K_M inference
    - **Forgetting gate**: Verified angle > 30 degrees vs other stacks

    ## License

    Apache 2.0
    """)


def validate_config(config: ReleaseConfig) -> list[str]:
    """Validate release configuration, return list of issues."""
    issues: list[str] = []

    if "/" not in config.repo:
        issues.append(f"Repo must be in 'org/name' format, got: {config.repo}")

    adapters_path = Path(config.adapters_dir)
    if not adapters_path.exists():
        issues.append(f"Adapters directory not found: {config.adapters_dir}")

    for domain in config.domains:
        if domain not in NICHE_DOMAINS:
            issues.append(f"Unknown domain: {domain}")

    return issues


def dry_run(config: ReleaseConfig) -> dict:
    """List what would be uploaded without actually pushing."""
    adapters_path = Path(config.adapters_dir)
    found = discover_adapters(adapters_path, config.domains)
    missing = [d for d in config.domains if d not in found]

    model_card = Path(config.model_card)
    has_model_card = model_card.exists()

    logger.info("DRY RUN — would upload to %s:", config.repo)
    logger.info("  Adapters dir: %s", adapters_path)
    logger.info("  Requested domains: %d", len(config.domains))
    logger.info("  Found adapters: %d", len(found))
    if missing:
        logger.warning("  Missing adapters: %s", ", ".join(missing))

    total_files = 0
    total_size = 0
    domain_reports: list[dict] = []

    for domain, stack_dir in sorted(found.items()):
        files = collect_files(stack_dir)
        size = sum(f.stat().st_size for f in files)
        total_files += len(files)
        total_size += size
        size_mb = size / (1024 * 1024)

        logger.info(
            "  [%s] %d files (%.2f MB) -> adapters/%s/",
            domain, len(files), size_mb, domain,
        )
        for f in files:
            size_kb = f.stat().st_size / 1024
            logger.info("      %s (%.1f KB)", f.name, size_kb)

        logger.info("    + per-adapter model card (generated)")

        domain_reports.append({
            "domain": domain,
            "stack_dir": str(stack_dir),
            "file_count": len(files),
            "size_mb": round(size_mb, 2),
        })

    total_size_mb = total_size / (1024 * 1024)
    logger.info("  Total: %d files, %.2f MB across %d adapters", total_files, total_size_mb, len(found))

    if has_model_card:
        logger.info("  Root model card: %s", model_card)
    else:
        logger.warning("  Root model card NOT found — will skip")

    return {
        "repo": config.repo,
        "adapters_dir": str(adapters_path),
        "domains_found": list(found.keys()),
        "domains_missing": missing,
        "domain_reports": domain_reports,
        "total_files": total_files,
        "total_size_mb": round(total_size_mb, 2),
        "model_card": str(model_card) if has_model_card else None,
        "private": config.private,
        "mode": "dry_run",
    }


def execute_release(config: ReleaseConfig) -> dict:
    """Actually push all adapters to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: uv add huggingface_hub")
        return {"status": "error", "reason": "huggingface_hub not installed"}

    api = HfApi()
    adapters_path = Path(config.adapters_dir)
    model_card = Path(config.model_card)

    found = discover_adapters(adapters_path, config.domains)
    if not found:
        logger.error("No adapters found to upload.")
        return {"status": "error", "reason": "no adapters found"}

    logger.info("Creating/verifying repo: %s", config.repo)
    api.create_repo(config.repo, exist_ok=True, private=config.private)

    # Upload root model card
    if model_card.exists():
        logger.info("Uploading root model card: %s", model_card)
        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=config.repo,
        )

    # Upload each adapter with its own model card
    uploaded: list[str] = []
    for domain, stack_dir in sorted(found.items()):
        logger.info("Uploading adapter: %s from %s", domain, stack_dir)

        # Upload adapter files
        api.upload_folder(
            folder_path=str(stack_dir),
            repo_id=config.repo,
            path_in_repo=f"adapters/{domain}",
        )

        # Generate and upload per-adapter model card
        card_content = generate_adapter_model_card(domain, config.repo)
        api.upload_file(
            path_or_fileobj=card_content.encode("utf-8"),
            path_in_repo=f"adapters/{domain}/README.md",
            repo_id=config.repo,
        )
        uploaded.append(domain)
        logger.info("  Uploaded %s (%d files + model card)", domain, len(collect_files(stack_dir)))

    logger.info(
        "Release complete: %d adapters pushed to https://huggingface.co/%s",
        len(uploaded), config.repo,
    )
    return {
        "status": "success",
        "repo": config.repo,
        "url": f"https://huggingface.co/{config.repo}",
        "uploaded_domains": uploaded,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Push all niche adapters to HuggingFace Hub with per-adapter model cards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available domains: {', '.join(NICHE_DOMAINS)}",
    )
    parser.add_argument(
        "--repo", required=True,
        help="HuggingFace repo ID (e.g. electron-rare/micro-kiki)",
    )
    parser.add_argument(
        "--adapters-dir", default=str(DEFAULT_ADAPTERS_DIR),
        help=f"Path to stacks directory (default: {DEFAULT_ADAPTERS_DIR})",
    )
    parser.add_argument(
        "--model-card", default=str(DEFAULT_MODEL_CARD),
        help=f"Path to root model card (default: {DEFAULT_MODEL_CARD})",
    )
    parser.add_argument(
        "--domain", metavar="NAME", action="append",
        help="Push a specific domain only (can be repeated). Default: all 10.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually push to HuggingFace (default: dry-run only)",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create repo as private",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    domains = args.domain if args.domain else list(NICHE_DOMAINS)

    config = ReleaseConfig(
        repo=args.repo,
        adapters_dir=args.adapters_dir,
        model_card=args.model_card,
        execute=args.execute,
        private=args.private,
        domains=domains,
    )

    issues = validate_config(config)
    if issues:
        for issue in issues:
            logger.warning("Validation: %s", issue)

    if config.execute:
        logger.info("EXECUTE mode — pushing to HuggingFace")
        execute_release(config)
    else:
        dry_run(config)
