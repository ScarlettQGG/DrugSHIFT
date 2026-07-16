"""Packaging / public-surface smoke tests."""
import importlib

import pytest


def test_public_surface_imports():
    import two_stage
    for name in ("Stage1", "make_model", "NeighborhoodAdapter", "Stage1Cache"):
        assert hasattr(two_stage, name), f"two_stage.{name} missing from public API"


def test_joint_embed_module_importable():
    mod = importlib.import_module("joint_embed")
    assert hasattr(mod, "main"), "joint_embed.main entrypoint missing"
    assert hasattr(mod, "build_arg_parser")


def test_joint_embedcmd_wrapper_delegates():
    # The console script is a thin wrapper around joint_embed.main.
    cmd = importlib.import_module("joint_embedcmd")
    import joint_embed
    assert cmd.main is joint_embed.main


def test_arg_parser_builds_and_has_help():
    import joint_embed
    parser = joint_embed.build_arg_parser()
    with pytest.raises(SystemExit):          # --help exits 0
        parser.parse_args(["--help"])
