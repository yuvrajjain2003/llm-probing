"""Local stand-in for the third-party ``pyairports`` package.

WHY.  outlines==0.0.46 (a dependency of vllm==0.5.5, imported when vllm is)
does ``from pyairports.airports import AIRPORT_LIST`` at import time, only to build
an enum of airport codes (``outlines.types.airports``) for structured generation.
The only ``pyairports`` on PyPI today is a single release, 0.0.1 (2025-09-25, author
"John Doe"), which is a stock setuptools sample project: it contains no airports
data, so it cannot satisfy that import, and it registers a console script called
``byted-wandb``. Installing it is worse than useless, so it is replaced by this.

EFFECT.  With an empty list, ``outlines.types.airports.IATA`` is an empty enum. This
repository never uses guided/structured decoding (run_eval / continuations sample
free text), so nothing observable changes. If you ever need outlines' airport types,
this shim is the thing to replace.
"""
AIRPORT_LIST: list = []
