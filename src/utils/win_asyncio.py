"""
Windows Python asyncio ProactorEventLoop socket reset guard.

Prevents noisy `ConnectionResetError: [WinError 10054]` tracebacks inside
`_ProactorBasePipeTransport._call_connection_lost` when browsers close keep-alive
connections or abort HTTP 206 Range probes on media files.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


def _finish_proactor_cleanup(transport: Any) -> None:
    """Ensure socket close and server detach still run if shutdown(SHUT_RDWR) raised WinError 10054."""
    sock = getattr(transport, "_sock", None)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass
        transport._sock = None
    server = getattr(transport, "_server", None)
    if server is not None:
        try:
            server._detach()
        except Exception:
            pass
        transport._server = None
    transport._called_connection_lost = True


def install_windows_proactor_guard() -> None:
    if sys.platform != "win32":
        return

    try:
        from asyncio import proactor_events

        orig_call_lost = proactor_events._ProactorBasePipeTransport._call_connection_lost

        if not getattr(orig_call_lost, "_pocketlens_guarded", False):
            def _safe_call_connection_lost(self: Any, exc: Any) -> None:
                try:
                    orig_call_lost(self, exc)
                except ConnectionResetError:
                    _finish_proactor_cleanup(self)
                except OSError as oserr:
                    if getattr(oserr, "winerror", None) in {10054, 10053, 10038}:
                        _finish_proactor_cleanup(self)
                    else:
                        raise

            _safe_call_connection_lost._pocketlens_guarded = True  # type: ignore[attr-defined]
            proactor_events._ProactorBasePipeTransport._call_connection_lost = _safe_call_connection_lost  # type: ignore[method-assign]
    except Exception:
        pass

    try:
        loop = asyncio.get_running_loop()
        orig_handler = loop.get_exception_handler()

        def _filtered_handler(lp: asyncio.AbstractEventLoop, ctx: dict[str, Any]) -> None:
            exc = ctx.get("exception")
            if isinstance(exc, ConnectionResetError):
                return
            if isinstance(exc, OSError) and getattr(exc, "winerror", None) in {10054, 10053, 10038}:
                return
            if orig_handler is not None:
                orig_handler(lp, ctx)
            else:
                lp.default_exception_handler(ctx)

        loop.set_exception_handler(_filtered_handler)
    except RuntimeError:
        pass


def resolve_playable_video(media_path: str, base_dir: Path) -> tuple[str, str]:
    """
    Resolve a browser-playable video relative path and MIME type without altering
    verified raw files in data/manifest_verified.json.
    - .ogv (Ogg Theora, dropped in Chrome 120+ / iOS Safari) -> H.264 *_web.mp4 proxy if present
    - .webm -> video/webm
    - .mp4 -> video/mp4
    """
    norm = media_path.replace("\\", "/")
    if norm.lower().endswith(".ogv"):
        web_mp4 = norm[:-4] + "_web.mp4"
        if (base_dir / web_mp4).exists():
            return web_mp4, "video/mp4"
        return norm, "video/ogg"
    if norm.lower().endswith(".webm"):
        return norm, "video/webm"
    return norm, "video/mp4"
