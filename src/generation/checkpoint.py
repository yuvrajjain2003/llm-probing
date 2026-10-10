#!/usr/bin/env python3
"""Split a base CoT into sentences and take prefixes at fixed fractions.

Goal 1 step 2.  The unit we split is the *reasoning span only* -- the text the
model emitted before ``</think>``.  This matters: the R1-Distill models often
write a plain-language summary between ``</think>`` and the final JSON, and that
summary frequently states the answer outright ("the depth is found to be 80
meters").  Splitting ``run_eval``'s ``reasoning`` field instead would smuggle that
sentence into late prefixes and manufacture the very surface-form leakage the
Goal 1 robustness analysis is meant to measure.  See ``extract_cot``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

THINK_CLOSE = "</think>"

# Fractions of the CoT (by sentence count) at which we cut prefixes.
DEFAULT_FRACTIONS: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)


# ── Sentence splitting ────────────────────────────────────────────────────────
#
# Naive `.`-splitting breaks on decimals, equations and abbreviations, all of
# which are dense in math CoTs.  Rather than enumerate every hazard we mask the
# risky spans, split the masked text, then unmask -- offsets are preserved
# because every mask is the same length as what it replaces.

_ABBREVIATIONS = [
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "cf.", "resp.", "al.",
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St.", "Fig.", "Eq.", "No.", "no.",
]

# A period that is NOT a sentence end:
_DECIMAL = re.compile(r"\d\.\d")                 # 0.15, 38.25
_INITIAL = re.compile(r"(?<![A-Za-z])[A-Za-z]\.")  # "A." — choice labels, variables
_ELLIPSIS = re.compile(r"\.\.\.")
# Inline math: $...$, \(...\), \[...\]
_MATH = re.compile(r"\$[^$\n]*\$|\\\([^\n]*?\\\)|\\\[[^\n]*?\\\]")

_SENT_END = re.compile(r"[.!?]['\")\]]*(?=\s)")
_HARD_BREAK = re.compile(r"\n\s*\n")


def _mask(text: str) -> str:
    """Replace periods that must not end a sentence with a placeholder."""
    masked = text

    def blank(m: re.Match) -> str:
        return "\x00" * len(m.group(0))

    # Order matters: mask whole math spans first, then the narrower patterns.
    masked = _MATH.sub(blank, masked)
    masked = _ELLIPSIS.sub(blank, masked)
    for abbr in _ABBREVIATIONS:
        masked = masked.replace(abbr, abbr.replace(".", "\x00"))
    masked = _DECIMAL.sub(lambda m: m.group(0).replace(".", "\x00"), masked)
    masked = _INITIAL.sub(blank, masked)
    return masked


def split_sentences(text: str) -> list[str]:
    """Split CoT text into sentences, math-aware.

    Blank-line breaks are always boundaries (R1 CoTs lean on them heavily);
    within a paragraph we split on unmasked sentence-final punctuation.
    """
    if not text or not text.strip():
        return []

    masked = _mask(text)
    cuts: set[int] = set()

    for m in _HARD_BREAK.finditer(masked):
        cuts.add(m.end())
    for m in _SENT_END.finditer(masked):
        cuts.add(m.end())

    bounds = sorted(cuts | {0, len(text)})
    out: list[str] = []
    for a, b in zip(bounds, bounds[1:]):
        seg = text[a:b]
        if not seg:
            continue
        # Whitespace-only runs are glued onto the previous sentence rather than
        # dropped: ''.join(split_sentences(t)) must reproduce t exactly, or the
        # reconstructed prefix would no longer match the model's own tokens.
        if not seg.strip() and out:
            out[-1] += seg
        else:
            out.append(seg)
    return out


# ── Checkpointing ─────────────────────────────────────────────────────────────

def extract_cot(raw_completion: str) -> str | None:
    """The reasoning span of a completion, or None if the CoT never terminated.

    A completion with no ``</think>`` hit the token cap mid-thought: it has no
    well-defined end, so 'remaining length' and 'final answer' labels are
    undefined for it.  Callers exclude these and log the rate (a Goal 1 metric).
    """
    if not raw_completion or THINK_CLOSE not in raw_completion:
        return None
    return raw_completion.split(THINK_CLOSE, 1)[0]


@dataclass
class Checkpoint:
    fraction: float
    sentence_idx: int      # number of sentences kept
    n_sentences: int       # total sentences in the CoT
    char_offset: int       # len(prefix_text)
    token_offset: int      # tokens in prefix_text (-1 if no tokenizer given)
    prefix_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_checkpoints(
    cot_text: str,
    tokenizer: Any | None = None,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
) -> list[Checkpoint]:
    """Prefixes of ``cot_text`` at each fraction, by sentence count.

    The prefix is the exact original substring (whitespace included) so that
    ``prompt + prefix_text`` reproduces the model's own text byte-for-byte --
    required for continuation sampling, which appends to the raw prompt with no
    chat-template re-application.
    """
    sents = split_sentences(cot_text)
    n = len(sents)
    if n == 0:
        return []

    out: list[Checkpoint] = []
    seen: set[int] = set()
    for f in fractions:
        k = 0 if f == 0 else max(1, min(n, round(f * n)))  # f=0: empty prefix
        if k in seen:          # short CoTs can collapse fractions onto one cut
            continue
        seen.add(k)
        prefix = "".join(sents[:k])
        tok = (len(tokenizer.encode(prefix, add_special_tokens=False))
               if tokenizer is not None else -1)
        out.append(Checkpoint(
            fraction=f, sentence_idx=k, n_sentences=n,
            char_offset=len(prefix), token_offset=tok, prefix_text=prefix,
        ))
    return out
