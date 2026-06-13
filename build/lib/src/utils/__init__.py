"""Shared utilities for the JEPA training pipeline."""

import traceback
import datetime
import os
from typing import Optional

_SESSION_ERROR_LOG_PATH: Optional[str] = None


def set_error_log_path(path: str) -> None:
    """Set the session-specific error log path for the current training run."""
    global _SESSION_ERROR_LOG_PATH
    _SESSION_ERROR_LOG_PATH = path


def log_error(tag: str, exc: Optional[Exception] = None) -> None:
    """Log an error with full traceback to the session error log and stdout.

    Args:
        tag: Short description of where the error occurred (e.g. 'effective_rank').
        exc: The exception, if any. If None, the current exception is captured.
    """
    global _SESSION_ERROR_LOG_PATH
    msg = _format_error(tag, exc)
    print(msg, end="")
    if _SESSION_ERROR_LOG_PATH is not None:
        try:
            os.makedirs(os.path.dirname(_SESSION_ERROR_LOG_PATH), exist_ok=True)
            with open(_SESSION_ERROR_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass  # can't log if the log itself fails


def _format_error(tag: str, exc: Optional[Exception] = None) -> str:
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    if exc is None:
        tb = traceback.format_exc().strip()
    else:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
    return f"[{ts}] [{tag}]\n{tb}\n\n"
