"""Shared comment-history rendering (Streamlit Layer 2).

Used by both the Feature & Grouping Workbench and the GLM Distillation
Workbench — each variable/term keeps a `comment_history` (see
`core.schemas.CommentEntry`), rendered identically in both places: newest
first, labeled "👤 Actuary" / "Claude" (with the logo at `icons/claude-ai.svg`
if present, inlined next to the text for tight baseline alignment; else an
orange text fallback via Streamlit's :orange[] markdown directive).
"""

import base64
import functools
import html
from pathlib import Path

import streamlit as st

from core.schemas import CommentEntry

# icons/ is meant to hold one logo per model provider as more get added (openai,
# mistral, ...) — Claude is the only agent in this app today, so only one is
# wired up. See icons/README.md for where each file was sourced from.
_CLAUDE_LOGO = Path(__file__).parent.parent / "icons" / "claude-ai.svg"


@functools.lru_cache(maxsize=1)
def claude_logo_data_uri() -> str | None:
    """Base64 data URI for the Claude logo, or None if the file isn't there —
    read once and cached, not re-read on every card render."""
    if not _CLAUDE_LOGO.exists():
        return None
    b64 = base64.b64encode(_CLAUDE_LOGO.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def render_comment_history(comment_history: list[CommentEntry]) -> None:
    """Render one variable/term's comment history, newest first."""
    for entry in sorted(comment_history, key=lambda e: e.ts, reverse=True):
        logo_uri = claude_logo_data_uri() if entry.author == "agent" else None
        if logo_uri:
            # Inline <img> via st.caption's own unsafe_allow_html — keeps the
            # same muted/small caption styling as the plain-text branch below,
            # just with the icon actually inline (st.columns puts icon and
            # text in separate block containers, which can't be tuned for
            # tight baseline alignment/spacing the way inline CSS can).
            # entry.text is actuary/LLM-authored free text, so it's escaped
            # before going into raw HTML.
            st.caption(
                f'<img src="{logo_uri}" style="height:1em;vertical-align:-0.15em;'
                f'margin-right:0.3em;"><b>Claude:</b> {html.escape(entry.text)}',
                unsafe_allow_html=True,
            )
        else:
            # :orange[...] is Streamlit's colored-text markdown directive —
            # used as a fallback if the logo file is missing (e.g. a fresh
            # checkout without icons/claude-ai.svg) so this never breaks.
            label = ":orange[**Claude:**]" if entry.author == "agent" else "**👤 Actuary:**"
            st.caption(f"{label} {entry.text}")
