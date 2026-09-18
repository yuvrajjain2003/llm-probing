"""Goal 1 steps 1-3: everything that talks to the model, plus the raw-record
maintenance that operates on its output.

``run_eval`` is also the shared helper library -- prompt rendering, ``parse_output``,
answer normalization, entropy and JSONL resume logic all live there and are reused
by the CPU-only modules in ``analysis`` and by ``probing``.
"""
