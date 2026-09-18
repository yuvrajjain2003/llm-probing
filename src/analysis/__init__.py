"""Pure CPU reductions over the records ``generation`` wrote.

No GPU and no model here: every module reads ``continuations.jsonl`` /
``checkpoints.jsonl`` and emits tables.  ``labels`` computes the Goal 1 behavioral
labels; the rest are the robustness analyses that decide whether those labels are
stable enough to be worth predicting.
"""
