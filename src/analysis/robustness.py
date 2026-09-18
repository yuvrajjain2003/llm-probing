#!/usr/bin/env python3
"""Goal 1 deliverable: robustness analyses on the continuation labels.

These decide whether the behavioral labels are stable enough to be worth
predicting from activations in Goal 2.  Three analyses, each over one run dir
(a dataset x model folder holding continuations.jsonl + checkpoints.jsonl):

  1. K-sufficiency        Are 10 continuations enough?  For each checkpoint we
                          subsample K'=2..10 (bootstrap) and watch how the entropy
                          and remaining-length estimates settle.  Reports the
                          bootstrap SE at each K' so we can see where it plateaus.

  2. Parser-failure catalog  Parse-success rate per dataset/model split into base-CoT
                          truncation vs continuation parse failure, with the actual
                          error strings tallied -- failure *patterns*, not just a rate.

  3. Surface-form leakage  How often the prefix text already states the answer a
                          predictor is supposed to forecast.  For MC datasets we look
                          for the answer letter being *declared* in the prefix (e.g.
                          "the answer is B", "option B"), not merely mentioned, plus
                          whether the choice's own text appears verbatim.  Freeform/
                          math datasets (no letters or choices) get a parallel check:
                          whether the answer's own normalized text appears verbatim
                          in the prefix.

Pure reductions over the JSONL -- no GPU, no model.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Running this file as a script puts src/<package>/ on sys.path, not src/, so the
# cross-package imports below would not resolve.  Put src/ on the path explicitly;
# this also lets the module be imported as `<package>.<module>`.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from generation.run_eval import _entropy, _load_jsonl

# Default sweep for K=100 runs. The pilot's (2,4,6,8,10) could only show SE falling
# up to 10 -- it could never show whether 10 was ENOUGH, which is the whole question.
KVALUES = (2, 5, 10, 20, 30, 50, 75, 100)
BOOT = 300
MIN_VALID_FRAC = 0.9   # keep groups with >=90% of K parseable


# ── 1. K-sufficiency ──────────────────────────────────────────────────────────

def k_sufficiency(cont: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Bootstrap SE of the entropy and remaining-length estimates vs K'.

    For each checkpoint group with all K continuations, draw size-K' samples with
    replacement and recompute the label; the spread across draws is how much the
    estimate would wobble if we had only sampled K' continuations.  Averaged over
    all checkpoints, a falling-then-flat SE means K' is enough.
    """
    keys = ["example_id", "cot_id", "fraction"]
    rows = []
    kept = dropped = 0
    # Requiring len(valid) == max(KVALUES) exactly would drop any group with even one
    # parse failure: at a 0.9% failure rate, ~60% of K=100 groups have at least one,
    # and the survivors would be a biased "clean" subset. MIN_VALID keeps groups that
    # are nearly complete; the bootstrap draws with replacement either way.
    min_valid = max(2, int(MIN_VALID_FRAC * max(KVALUES)))
    for _, grp in cont.groupby(keys):
        valid = [a for a in grp["final_answer_norm"].dropna().tolist() if str(a).strip()]
        rem = grp.loc[grp["terminated"] == True, "remaining_reasoning_tokens"].to_numpy(float)  # noqa: E712
        if len(valid) < min_valid:
            dropped += 1
            continue
        kept += 1
        va = np.asarray(valid, dtype=object)
        for k in KVALUES:
            # Vectorised: one (BOOT, k) draw beats BOOT separate rng.choice calls.
            draws = rng.choice(va, size=(BOOT, k), replace=True)
            ent = [_entropy(list(d)) for d in draws]
            row = {"fraction": grp["fraction"].iloc[0], "k": k,
                   "entropy_se": float(np.std(ent)), "n_valid": len(valid)}
            if rem.size >= min_valid:
                rlen = rng.choice(rem, size=(BOOT, k), replace=True).mean(axis=1)
                row["rem_len_se"] = float(np.std(rlen))
            rows.append(row)
    print(f"  k-sufficiency: {kept} checkpoint groups used, {dropped} dropped "
          f"(needed >= {min_valid} parseable answers)")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.groupby(["fraction", "k"]).mean().reset_index()
            .sort_values(["fraction", "k"]))


# ── 2. Parser-failure catalog ───────────────────────────────────────────────

def parser_catalog(cont: pd.DataFrame, base: pd.DataFrame) -> dict:
    """Failure rates + patterns for base CoTs and continuations."""
    base_trunc = base[~base["raw_completion"].fillna("").str.contains("</think>")]
    cont_fail = cont[~cont["parse_success"]]
    # Not a failure (parse_success=True) but still an instruction-following miss worth
    # tracking: the model emitted \boxed{...} instead of the requested JSON schema.
    # Common on math-competition datasets (MATH-500, AIME) trained heavily on \boxed{}.
    cont_boxed = cont[cont["parse_error"] == "used_boxed_fallback"]
    # Also not a failure: the answer was LaTeX, so it was taken verbatim rather than
    # through json.loads (which mangles LaTeX both loudly and silently). Tracked
    # because it says how often a math answer took the non-standard recovery path.
    cont_latex = cont[cont["parse_error"] == "latex_literal_recovered"]

    def _norm_err(e: str) -> str:
        e = str(e or "")
        if "No JSON found" in e:
            return "no_json_found"
        if "Expecting" in e or "delimiter" in e or "value" in e:
            return "malformed_json"
        return e[:40] if e else "ok"

    return dict(
        base_cots=len(base),
        base_truncated=int(len(base_trunc)),
        base_truncation_rate=float(len(base_trunc) / len(base)) if len(base) else float("nan"),
        continuations=len(cont),
        cont_parse_fail=int(len(cont_fail)),
        cont_parse_fail_rate=float(len(cont_fail) / len(cont)) if len(cont) else float("nan"),
        cont_boxed_fallback=int(len(cont_boxed)),
        cont_boxed_fallback_rate=float(len(cont_boxed) / len(cont)) if len(cont) else float("nan"),
        cont_latex_literal=int(len(cont_latex)),
        cont_latex_literal_rate=float(len(cont_latex) / len(cont)) if len(cont) else float("nan"),
        cont_nonterminated=int((cont["terminated"] == False).sum()),  # noqa: E712
        cont_fail_patterns=dict(Counter(cont_fail["parse_error"].map(_norm_err))),
    )


# ── 3. Surface-form leakage (multiple-choice) ────────────────────────────────
#
# Two distinct ways a prefix can hand the answer to a predictor, reported
# separately because they mean different things:
#
#   letter_declared    the answer LETTER stated in an explicit answer context
#                      ("the answer is B", "option B is correct", "choose B").
#                      A commitment signal.  The trigger must be the word
#                      "answer"/"option"/"choice"/"correct" or an explicit
#                      choose/select verb -- NOT a bare capital letter, because
#                      these datasets use A/B/C/D as point and variable names
#                      (arc AD, point B) and the earlier loose regex counted those.
#
#   choice_text_present  the answer choice's TEXT appears verbatim ("used more
#                      times", "12*sqrt(2)", "130").  A cheatable surface feature
#                      even when the model has not yet declared a choice.  Short
#                      numeric choices make this noisy, so it is reported apart
#                      from the letter signal, never merged into one number.

_DECLARE = re.compile(
    r"(?:"
    r"answer\s*(?:is|:|=|should\s+be|would\s+be|must\s+be)?\s*\(?([A-J])\)?\b"
    r"|(?:option|choice)\s*\(?([A-J])\)?\s*(?:is\s*(?:correct|the\s+answer)|correct)"
    r"|(?:choose|select|go\s+with|pick)\s*(?:option\s+|choice\s+)?\(?([A-J])\)?\b"
    r"|(?:which\s+is|that'?s|so\s+it'?s)\s+(?:option|choice)\s+\(?([A-J])\)?\b"
    r")",
    re.IGNORECASE,
)


def _declared_letters(text: str) -> set[str]:
    out: set[str] = set()
    for m in _DECLARE.finditer(text or ""):
        g = next((x for x in m.groups() if x), None)
        if g:
            out.add(g.upper())
    return out


def _choice_map(choices_json) -> dict[str, str]:
    """{'A': 'transformed', ...} from a stored choices list like ['A) transformed']."""
    if not isinstance(choices_json, str) or not choices_json.strip():
        return {}
    try:
        items = json.loads(choices_json)
    except Exception:
        return {}
    out = {}
    for it in items:
        m = re.match(r"\s*([A-J])\)?\s*(.*)", str(it), re.DOTALL)
        if m:
            out[m.group(1).upper()] = re.sub(r"\s+", " ", m.group(2)).strip().lower()
    return out


def surface_leakage(ckpt: pd.DataFrame, cont: pd.DataFrame,
                    examples: pd.DataFrame) -> pd.DataFrame:
    """Per checkpoint: is the eventual answer already present in the prefix?

    'Eventual answer' = the modal final answer of the prefix's continuations, and
    separately the gold answer -- a predictor could cheat off either.
    """
    mode = (cont.groupby(["example_id", "cot_id", "fraction"])["final_answer_norm"]
            .agg(lambda s: Counter([x for x in s.dropna() if str(x).strip()]).most_common(1)[0][0]
                 if any(str(x).strip() for x in s.dropna()) else None)
            .rename("mode_answer").reset_index())
    gold = cont.groupby(["example_id", "cot_id", "fraction"])["gold_answer_norm"].first().reset_index()

    choices = examples.set_index("example_id")["choices"].map(_choice_map).to_dict()

    m = ckpt.merge(mode, on=["example_id", "cot_id", "fraction"]) \
            .merge(gold, on=["example_id", "cot_id", "fraction"])

    mode_decl, gold_decl, mode_txt, gold_txt, mode_ans_txt, gold_ans_txt = [], [], [], [], [], []
    for r in m.itertuples():
        prefix = r.prefix_text or ""
        prefix_l = prefix.lower()
        decl = _declared_letters(prefix)
        cmap = choices.get(r.example_id, {})
        for ans, decl_out, txt_out, ans_txt_out in (
                (r.mode_answer, mode_decl, mode_txt, mode_ans_txt),
                (r.gold_answer_norm, gold_decl, gold_txt, gold_ans_txt)):
            letter = str(ans).upper() if ans else None
            decl_out.append(bool(letter and letter in decl))
            ctext = cmap.get(letter, "") if letter else ""
            # require >=2 chars of choice text to avoid matching a lone digit
            txt_out.append(bool(ctext) and len(ctext) >= 2 and ctext in prefix_l)
            # Dataset-agnostic leakage check: does the answer's own normalized text
            # (not a choice map) appear verbatim in the prefix? MC datasets already
            # have letter/choice-text signals above; this is the only leakage signal
            # available for freeform/math datasets (math500, aime2025), which have no
            # choices column. Same >=2-char guard against matching a lone digit.
            ans_s = str(ans).lower() if ans is not None else ""
            ans_txt_out.append(bool(ans_s) and len(ans_s) >= 2 and ans_s in prefix_l)

    m["mode_letter_declared"] = mode_decl
    m["gold_letter_declared"] = gold_decl
    m["mode_choice_text_present"] = mode_txt
    m["gold_choice_text_present"] = gold_txt
    m["mode_answer_text_present"] = mode_ans_txt
    m["gold_answer_text_present"] = gold_ans_txt

    return (m.groupby("fraction")
            .agg(n=("mode_letter_declared", "size"),
                 mode_letter_declared=("mode_letter_declared", "mean"),
                 gold_letter_declared=("gold_letter_declared", "mean"),
                 mode_choice_text_present=("mode_choice_text_present", "mean"),
                 gold_choice_text_present=("gold_choice_text_present", "mean"),
                 mode_answer_text_present=("mode_answer_text_present", "mean"),
                 gold_answer_text_present=("gold_answer_text_present", "mean"))
            .reset_index())


# ── driver ───────────────────────────────────────────────────────────────────

def run(run_dir: Path, base_records: Path, out_dir: Path,
        examples_csv: Path, seed: int = 0) -> None:
    cont = _load_jsonl(run_dir / "raw" / "continuations.jsonl")
    ckpt = _load_jsonl(run_dir / "raw" / "checkpoints.jsonl")
    base = _load_jsonl(base_records)
    examples = pd.read_csv(examples_csv)
    if cont.empty:
        raise ValueError(f"no continuations in {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    ksuff = k_sufficiency(cont, rng)
    ksuff.to_csv(out_dir / "k_sufficiency.csv", index=False)

    cat = parser_catalog(cont, base)
    (out_dir / "parser_catalog.json").write_text(json.dumps(cat, indent=2))

    leak = surface_leakage(ckpt, cont, examples)
    leak.to_csv(out_dir / "surface_leakage.csv", index=False)

    print(f"\n{'─'*60}\n  {run_dir.parent.name}/{run_dir.name}")
    print(f"  parser: base trunc {cat['base_truncation_rate']:.1%}, "
          f"cont parse-fail {cat['cont_parse_fail_rate']:.1%}, "
          f"patterns={cat['cont_fail_patterns']}")
    if not ksuff.empty:
        piv = ksuff.pivot(index="k", columns="fraction", values="entropy_se")
        print("  entropy bootstrap-SE by K (rows) x fraction (cols):")
        print(piv.round(3).to_string().replace("\n", "\n    "))
    print("  surface leakage by fraction  [letter declared | choice-text | answer-text], mode/gold:")
    for _, r in leak.iterrows():
        print(f"    {r['fraction']:.1f}: letter {r['mode_letter_declared']:.1%}/{r['gold_letter_declared']:.1%}"
              f"   choice-text {r['mode_choice_text_present']:.1%}/{r['gold_choice_text_present']:.1%}"
              f"   answer-text {r['mode_answer_text_present']:.1%}/{r['gold_answer_text_present']:.1%}")
    print(f"  written to {out_dir}\n{'─'*60}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True,
                   help="dir with continuations.jsonl + checkpoints.jsonl")
    p.add_argument("--base-records", default=None,
                   help="sample_records.jsonl of the base CoTs "
                        "(defaults to --run-dir/raw/sample_records.jsonl)")
    p.add_argument("--out-dir", default=None,
                   help="defaults to --run-dir/robustness")
    p.add_argument("--examples-csv", default=None,
                   help="examples.csv with the choices column "
                        "(defaults to --run-dir/raw/examples.csv)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    base_records = Path(args.base_records) if args.base_records else run_dir / "raw" / "sample_records.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "robustness"
    examples_csv = Path(args.examples_csv) if args.examples_csv else run_dir / "raw" / "examples.csv"
    run(run_dir, base_records, out_dir, examples_csv, args.seed)


if __name__ == "__main__":
    main()
