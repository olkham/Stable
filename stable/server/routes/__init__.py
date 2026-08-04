"""FastAPI routers, one module per resource area.

:mod:`stable.server.app` mounts all of them under ``/api`` (plus the raw
WebSocket route, which is not under the REST prefix).
"""

from __future__ import annotations
