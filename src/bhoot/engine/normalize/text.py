"""Text normalisation for *matching* quotes against page text. Never used to change values.

``normalize_with_index_map`` also returns, for every character of the normalised text, the
index of the original character it came from, so a match found in normalised text can be
mapped back to an exact span (and bounding box) on the page.
"""

from __future__ import annotations

import unicodedata

_DASHES = dict.fromkeys(
    map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"), "-"
)
_QUOTES = {ord(c): "'" for c in "\u2018\u2019\u201a\u201b\u2032`\u00b4"} | {
    ord(c): '"' for c in "\u201c\u201d\u201e\u201f\u2033"
}
_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")
_TRANSLATE = {**_DASHES, **_QUOTES}


def normalize_with_index_map(text: str) -> tuple[str, list[int]]:
    out: list[str] = []
    back: list[int] = []
    pending_space = -1  # index of the first whitespace in a run, -1 if none
    for i, ch in enumerate(text):
        if ch in _ZERO_WIDTH:
            continue
        piece = unicodedata.normalize("NFKC", ch).translate(_TRANSLATE).casefold()
        for c in piece:
            if c.isspace():
                if pending_space < 0:
                    pending_space = i
                continue
            if pending_space >= 0:
                if out:  # collapse whitespace runs to one space; drop leading whitespace
                    out.append(" ")
                    back.append(pending_space)
                pending_space = -1
            out.append(c)
            back.append(i)
    return "".join(out), back


def normalize_for_match(text: str) -> str:
    return normalize_with_index_map(text)[0]


def strip_currency(text: str) -> str:
    """Remove currency markers from already-normalised (casefolded) text, so 'rs. 500' and
    '500' match."""
    out = text
    for token in ("₹", "rs.", "rs ", "inr"):
        out = out.replace(token, "")
    return " ".join(out.split())
