"""Packaging, public surface and the "docstrings on every public contract" rule."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from importlib import metadata

import pytest

import stable


def iter_modules():
    """Yield every module in the ``stable`` package."""
    yield stable
    for info in pkgutil.walk_packages(stable.__path__, prefix="stable."):
        yield importlib.import_module(info.name)


ALL_MODULES = list(iter_modules())


def public_names(module) -> list[str]:
    """Public API of a module: its ``__all__`` if it has one, else its exports."""
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    return [name for name in vars(module) if not name.startswith("_")]


def iter_public_contracts():
    """Yield ``(label, object)`` for every public class, function and method."""
    seen: set[tuple[str, str]] = set()
    for module in ALL_MODULES:
        for name in public_names(module):
            obj = getattr(module, name, None)
            origin = getattr(obj, "__module__", None)
            if obj is None or not (origin or "").startswith("stable"):
                continue
            if not (inspect.isclass(obj) or inspect.isroutine(obj)):
                continue
            key = (origin, getattr(obj, "__qualname__", name))
            if key in seen:
                continue
            seen.add(key)
            yield f"{origin}.{getattr(obj, '__qualname__', name)}", obj
            if inspect.isclass(obj):
                for attr_name in list(vars(obj)):
                    if attr_name.startswith("_"):
                        continue
                    member = inspect.getattr_static(obj, attr_name)
                    if isinstance(member, staticmethod | classmethod):
                        member = member.__func__
                    if inspect.isroutine(member) or isinstance(member, property):
                        yield f"{origin}.{obj.__qualname__}.{attr_name}", member


def test_the_package_is_installed():
    """`pip install -e .` works: distribution metadata and console script exist."""
    assert metadata.version("stable-tracker")
    scripts = {ep.name for ep in metadata.entry_points(group="console_scripts")}
    assert "stable" in scripts
    assert stable.__version__


@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda m: m.__name__)
def test_every_module_has_a_docstring(module):
    assert (module.__doc__ or "").strip(), f"{module.__name__} has no module docstring"


@pytest.mark.parametrize(
    ("label", "obj"),
    list(iter_public_contracts()),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_public_contract_has_a_docstring(label, obj):
    doc = inspect.getdoc(obj) if not isinstance(obj, property) else (obj.__doc__ or "")
    assert (doc or "").strip(), f"{label} has no docstring"


@pytest.mark.parametrize("module", ALL_MODULES, ids=lambda m: m.__name__)
def test_declared_exports_exist(module):
    for name in getattr(module, "__all__", []):
        assert hasattr(module, name), f"{module.__name__}.__all__ names missing {name!r}"


def test_top_level_reexports_the_phase_0_contracts():
    expected = {
        "Query",
        "TrackResult",
        "PointTrack",
        "SimilarityTransform",
        "Correction",
        "apply_corrections",
        "Project",
        "TrackerPlugin",
        "registry",
        "get_tracker",
        "plan_anchor_runs",
        "merge_anchor_runs",
        "MergeConfig",
        "select_device",
        "StableError",
    }
    assert expected <= set(stable.__all__)
    for name in stable.__all__:
        assert hasattr(stable, name)


def test_architecture_packages_all_exist():
    """The repo layout matches the Architecture section of PLAN.md."""
    names = {module.__name__ for module in ALL_MODULES}
    for package in ("core", "video", "trackers", "solver", "smoothing", "render", "server", "cli"):
        assert f"stable.{package}" in names


def test_cli_entry_point_runs():
    from stable.cli.main import build_parser, main

    assert build_parser().prog == "stable"
    assert main(["trackers"]) == 0
    assert main(["trackers", "--json"]) == 0
    assert main(["devices"]) == 0
    assert main(["devices", "--json"]) == 0
    assert main([]) == 0
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
