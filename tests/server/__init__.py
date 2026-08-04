"""Making this a real package gives its ``conftest.py`` a dotted module name.

Without this file, pytest imports both ``tests/conftest.py`` and
``tests/server/conftest.py`` under the same bare ``sys.modules["conftest"]``
key (neither directory has an ``__init__.py``, so both look like "rootless"
modules to Python's import system) — whichever loads second silently shadows
the first, breaking every top-level test that does ``from conftest import
...``. With this ``__init__.py`` present, ``tests/`` (still package-less) is
the nearest non-package ancestor, so this directory imports as the distinct
top-level package ``server`` and its conftest becomes ``server.conftest`` —
no collision, and no change needed to ``tests/conftest.py``.
"""

from __future__ import annotations
