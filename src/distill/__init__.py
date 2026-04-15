"""Distillation pipeline: teacher client, dataset generator, dedup."""

from src.distill.generator import (
    GeneratorConfig,
    TeacherProtocol,
    generate_examples,
    hash_record,
    load_existing_hashes,
)

__all__ = [
    "GeneratorConfig",
    "TeacherProtocol",
    "generate_examples",
    "hash_record",
    "load_existing_hashes",
]
