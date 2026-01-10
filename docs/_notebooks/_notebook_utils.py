"""Jupyter magics for IDTrack documentation notebooks.

Usage in notebooks:
    # In the first cell, load the magics:
    %load_ext _notebook_utils

    # Then use the collapse magic on any cell:
    %%collapse Click to show download logs
    df = dm.create_database_content(just_download=False)

    # Optional flags:
    #   --no-live or --static: don't update the block while running (live is default)
"""

from __future__ import annotations

import html
import io
import logging
import shlex
import time
import traceback
import uuid
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Callable

from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.display import HTML, display

# ---------------------------------------------------------------------------
# Warnings capture
# ---------------------------------------------------------------------------


@contextmanager
def _capture_warnings(buffer: io.StringIO):
    """Temporarily redirect `warnings.showwarning` to write to `buffer`.

    This ensures warnings are captured even if the `warnings` module cached a reference to `sys.stderr` before
    `redirect_stderr` was applied.

    Args:
        buffer: Buffer that receives formatted warning messages.

    Yields:
        None: Control is yielded to the caller while redirection is active.
    """
    original_showwarning = warnings.showwarning

    def custom_showwarning(message, category, filename, lineno, _file=None, line=None):
        # Format the warning message similar to the default format
        formatted = warnings.formatwarning(message, category, filename, lineno, line)
        buffer.write(formatted)

    warnings.showwarning = custom_showwarning
    try:
        yield
    finally:
        warnings.showwarning = original_showwarning


# ---------------------------------------------------------------------------
# Logging capture
# ---------------------------------------------------------------------------


@contextmanager
def _capture_logging(buffer: io.StringIO):
    """Temporarily redirect `logging.StreamHandler` outputs to `buffer`.

    This patches existing handlers in-place because many loggers cache their handler stream references at creation
    time. Simply redirecting `sys.stderr` won't affect them.

    Note:
        Handlers created during cell execution are not captured. New loggers without explicit handlers will
        propagate to the root logger (which is patched), so most logging output is still captured.

    Args:
        buffer: Buffer that receives log output.

    Yields:
        None: Control is yielded to the caller while redirection is active.
    """
    patched: list[tuple[logging.StreamHandler, object]] = []
    seen: set[int] = set()

    def _patch_handler(handler: logging.Handler) -> None:
        # Only patch StreamHandlers (but not FileHandlers)
        if not isinstance(handler, logging.StreamHandler):
            return
        if isinstance(handler, logging.FileHandler):
            return
        hid = id(handler)
        if hid in seen:
            return
        seen.add(hid)

        original = handler.stream
        try:
            handler.setStream(buffer)
            patched.append((handler, original))
        except Exception:  # noqa: S110
            pass  # Some handlers don't support setStream

    def _patch_logger(logger: logging.Logger) -> None:
        for h in getattr(logger, "handlers", []):
            _patch_handler(h)

    # Patch root logger
    _patch_logger(logging.getLogger())

    # Patch all named loggers
    manager = getattr(logging.root, "manager", None)
    if manager is not None:
        for logger in getattr(manager, "loggerDict", {}).values():
            if isinstance(logger, logging.Logger):
                _patch_logger(logger)

    # Patch lastResort handler (Python 3.2+)
    last_resort = getattr(logging, "lastResort", None)
    if last_resort is not None:
        _patch_handler(last_resort)

    try:
        yield
    finally:
        # Restore all patched handlers
        for handler, original_stream in patched:
            try:
                handler.setStream(original_stream)
            except Exception:  # noqa: S110
                pass


# ---------------------------------------------------------------------------
# Live buffer with throttled callbacks
# ---------------------------------------------------------------------------


class _ThrottledBuffer(io.StringIO):
    """StringIO that triggers a callback on writes, throttled to avoid UI spam."""

    def __init__(self, callback: Callable[[], None], interval_s: float = 0.3):
        super().__init__()
        self._callback = callback
        self._interval_s = interval_s
        self._last_call = 0.0

    def _maybe_trigger_callback(self) -> None:
        """Trigger callback if enough time has passed since last call."""
        now = time.monotonic()
        if now - self._last_call >= self._interval_s:
            self._last_call = now
            try:
                self._callback()
            except Exception:  # noqa: S110
                pass  # Never let UI updates break execution

    def write(self, s: str) -> int:
        n = super().write(s)
        self._maybe_trigger_callback()
        return n

    def writelines(self, lines) -> None:
        """Override writelines to also trigger callback."""
        super().writelines(lines)
        self._maybe_trigger_callback()


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def _escape(text: str) -> str:
    """Escape text for safe HTML embedding."""
    return html.escape(text) if text else ""


def _render_collapsed_block(summary: str, content: str, *, is_open: bool = False) -> str:
    """Render a `<details>` HTML block.

    When `is_open` is False, the block is collapsed by default. The HTML5 `<details>` element is collapsed unless
    the `open` attribute is present.

    Args:
        summary: Text shown in the `<summary>` element.
        content: HTML content to place inside the `<pre>` block.
        is_open: If True, include the `open` attribute.

    Returns:
        The rendered HTML string.
    """
    open_attr = " open" if is_open else ""
    safe_summary = _escape(summary)
    return f"""<details class="idtrack-collapse"{open_attr}>
<summary>{safe_summary}</summary>
<pre class="idtrack-collapse-content">{content}</pre>
</details>"""


def _tail(text: str, max_lines: int, max_line_length: int = 500) -> str:
    """Return the last max_lines of text, with long lines truncated (for live preview)."""
    if not text:
        return text
    lines = text.splitlines()

    # Truncate to last N lines
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]

    # Truncate very long lines to avoid performance issues
    if max_line_length > 0:
        truncated = []
        for line in lines:
            if len(line) > max_line_length:
                truncated.append(line[:max_line_length] + "...")
            else:
                truncated.append(line)
        lines = truncated

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The collapse magic
# ---------------------------------------------------------------------------


@magics_class
class CollapseMagics(Magics):
    """Jupyter magics for collapsible output blocks in Sphinx docs."""

    @cell_magic
    def collapse(self, line: str, cell: str):
        """Execute cell code and wrap all output in a collapsible `<details>` block.

        Usage:
            %%collapse [--no-live] Summary text here
            <your code>

        Options:
            --no-live, --static: Disable live preview updates during execution.

        The block is collapsed by default when execution finishes, which is useful for hiding verbose logs in
        documentation while keeping them accessible.

        Args:
            line: The line after `%%collapse` (flags and summary text).
            cell: The cell body to execute.

        Raises:
            err: Re-raises exceptions raised during cell execution.
        """
        # Parse arguments
        try:
            tokens = shlex.split(line)
        except ValueError:
            tokens = line.split()

        live_mode = True
        summary_parts: list[str] = []

        for tok in tokens:
            if tok in ("--no-live", "--static"):
                live_mode = False
            else:
                summary_parts.append(tok)

        summary = " ".join(summary_parts).strip() or "Show output"

        # Buffers for capturing output
        log_buf = io.StringIO()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        error_tb: str = ""

        # Display handle for in-place updates
        display_id: str | None = str(uuid.uuid4()) if live_mode else None

        def _build_content(*, preview: bool) -> str:
            """Build the HTML content for inside the <pre> block."""
            parts: list[str] = []

            log_text = log_buf.getvalue()
            stdout_text = stdout_buf.getvalue()
            stderr_text = stderr_buf.getvalue()

            # In preview mode, truncate to last N lines
            if preview:
                log_text = _tail(log_text, 150)
                stdout_text = _tail(stdout_text, 150)
                stderr_text = _tail(stderr_text, 50)

            if log_text:
                parts.append(_escape(log_text.rstrip()))
            if stdout_text:
                parts.append(_escape(stdout_text.rstrip()))
            if stderr_text:
                parts.append(f'<span class="stderr">{_escape(stderr_text.rstrip())}</span>')

            # Include error traceback in final output only
            if error_tb and not preview:
                parts.append(f'<span class="stderr">{_escape(error_tb)}</span>')

            if not parts:
                return "(running...)" if preview else "(no output)"

            return "\n".join(parts)

        def _refresh_display() -> None:
            """Update the live preview (called periodically during execution)."""
            if not live_mode or display_id is None:
                return
            content = _build_content(preview=True)
            html_str = _render_collapsed_block(summary, content, is_open=True)
            try:
                display(HTML(html_str), display_id=display_id, update=True)
            except Exception:  # noqa: S110
                pass  # Ignore display errors

        # Create throttled buffers for live mode
        if live_mode:
            log_buf = _ThrottledBuffer(_refresh_display, interval_s=0.3)
            stdout_buf = _ThrottledBuffer(_refresh_display, interval_s=0.3)
            stderr_buf = _ThrottledBuffer(_refresh_display, interval_s=0.3)

            # Show initial placeholder
            try:
                initial_html = _render_collapsed_block(summary, "(running...)", is_open=True)
                display(HTML(initial_html), display_id=display_id)
            except Exception:
                live_mode = False  # Fall back to non-live mode
                display_id = None

        # Execute the cell with all output captured
        exec_result = None
        try:
            with _capture_warnings(stderr_buf):
                with _capture_logging(log_buf):
                    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                        exec_result = self.shell.run_cell(cell)
        except Exception:
            error_tb = traceback.format_exc()

        # Check for execution errors
        if exec_result is not None:
            err = getattr(exec_result, "error_before_exec", None) or getattr(exec_result, "error_in_exec", None)
            if err is not None:
                error_tb = "".join(traceback.format_exception(type(err), err, err.__traceback__)).rstrip()

        # Render final output (COLLAPSED by default - no 'open' attribute)
        final_content = _build_content(preview=False)
        final_html = _render_collapsed_block(summary, final_content, is_open=False)

        if live_mode and display_id is not None:
            # Clear the live preview (browsers preserve DOM state on updates,
            # so we can't just update with collapsed HTML - it would stay open).
            # Instead, clear it and display fresh output.
            try:
                display(HTML(""), display_id=display_id, update=True)
            except Exception:  # noqa: S110
                pass
        # Always display fresh final output (collapsed by default)
        display(HTML(final_html))

        # Show the cell's return value (if any) outside the collapsed block
        if exec_result is not None and getattr(exec_result, "result", None) is not None:
            display(exec_result.result)

        # Re-raise exceptions so the notebook shows the error state
        if exec_result is not None:
            err = getattr(exec_result, "error_before_exec", None) or getattr(exec_result, "error_in_exec", None)
            if err is not None:
                raise err


def load_ipython_extension(ipython):
    """Called by %load_ext to register the magics."""
    ipython.register_magics(CollapseMagics)
