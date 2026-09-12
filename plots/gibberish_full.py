#!/usr/bin/env python3
"""Gibberish rate on ALL 1,273 questions, not the 573-question test split.

The published table reports 2,292 prompts = 573 questions x 4 options, but the
generation pipeline's own pairs are 2 per question (one correct claim, one
sampled distractor), so its test split is 573 x 2 = 1,146. Neither current code
path produces 2,292; the table predates a change. This script therefore scores
BOTH constructions over ALL 1,273 questions:

    --pairs  balanced 2 per question   -> 2,546 prompts (the pipeline's own set)
    --quad   all four options          -> 5,092 prompts (the evaluation set, and
                                          what the published table describes)

Nothing about a gibberish rate needs a held-out split: no probe is fitted, so
there is no leakage to protect against and no reason to discard 55% of the
questions.

(Cross-validation would be the wrong instrument for the same reason. Averaging
the rate over five disjoint test folds is arithmetically the rate over all
1,273; the fold structure adds only an SD, and a question-clustered bootstrap
gives a better interval for a proportion.)

Everything that could change the number is taken from the pipeline unchanged --
`make_tf_prompt` via the same CSV, `batch_generate`, MAX_NEW_TOKENS=64, greedy
decoding, left padding, bfloat16 -- so the only difference from the published
run is which questions are scored. As a check, the script also reports the rate
restricted to the original test split, which must reproduce the published value.

GPU, one model at a time: plots/slurm_gibberish.sh runs it as an 9-task array.

Usage: python plots/gibberish_full.py METHOD [--quad]
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))

import hidden_knowledge_after_unlearning as hk  # noqa: E402
from readout_edit_all import HF, REPOS          # noqa: E402

OUT = ROOT / "plots" / "activation_vectors"
BATCH = int(__import__("os").environ.get("GEN_BATCH", "16"))


def rate(answers, pairs):
    """Fraction of outputs containing neither True nor False (extract_tf None)."""
    n = len(answers)
    bad = [i for i, a in enumerate(answers) if hk.extract_tf(a) is None]
    return dict(n=n, gibberish=len(bad),
                rate=round(len(bad) / n, 6) if n else 0.0)


def quad_pairs():
    """All four options of every question -- the evaluation construction, and
    what the published gibberish table describes (573 x 4 for its split)."""
    import csv as _csv
    import json as _json
    out = []
    with open(hk.WMDP_CSV_PATH, newline="", encoding="utf-8") as f:
        seen = set()
        for row in _csv.DictReader(f):
            qid = int(row["original_id"])
            if qid in seen:
                continue                       # the CSV holds 2 rows per question
            seen.add(qid)
            choices = _json.loads(row["choices_json"])
            cor = int(row["correct_idx"])
            for i, ch in enumerate(choices):
                out.append({"prompt": hk.make_tf_prompt(row["question"], ch),
                            "expected": "True" if i == cor else "False",
                            "question": row["question"], "answer": ch,
                            "pair_type": "pos" if i == cor else "neg",
                            "original_id": qid, "split": row["split"]})
    return out


def main(method, quad=False):
    if quad:
        pairs = quad_pairs()
        test_p = [p for p in pairs if p["split"] == "test"]
        n_test = len(test_p)
    else:
        loaded = hk.load_tf_pairs_from_csv()
        assert loaded, f"missing {hk.WMDP_CSV_PATH}"
        train_p, val_p, test_p = loaded
        pairs = train_p + val_p + test_p      # all 1,273 questions, 2 claims each
        n_test = len(test_p)
    print(f"{method}: {len(pairs)} prompts over "
          f"{len({p['original_id'] for p in pairs})} questions "
          f"({n_test} of them the original test split)", flush=True)

    repo = REPOS[method]
    tok = hk.AutoTokenizer.from_pretrained(repo, cache_dir=str(HF / "hub"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = hk.AutoModelForCausalLM.from_pretrained(
        repo, cache_dir=str(HF / "hub"), torch_dtype=torch.bfloat16,
        device_map="auto")
    model.eval()

    answers = hk.batch_generate(model, tok, pairs, BATCH, f"gibberish/{method}")

    # the split the paper currently reports, recomputed here as a gate
    if quad:
        idx = [i for i, p in enumerate(pairs) if p["split"] == "test"]
        split_answers = [answers[i] for i in idx]
    else:
        split_answers = answers[len(pairs) - n_test:]
    split_pairs = test_p
    res = dict(method=method,
               full=rate(answers, pairs),
               test_split_only=rate(split_answers, split_pairs),
               construction="quad" if quad else "pairs",
               max_new_tokens=hk.MAX_NEW_TOKENS, batch=BATCH, repo=repo)
    OUT.mkdir(exist_ok=True)
    tag = f"{method}_quad" if quad else method
    with open(OUT / f"gibberish_full_{tag}.json", "w") as f:
        json.dump({**res, "answers": answers}, f)
    print(f"  FULL  n={res['full']['n']:>5}  gibberish={res['full']['gibberish']:>5}"
          f"  rate={res['full']['rate']:.4f}")
    print(f"  TEST  n={res['test_split_only']['n']:>5}  "
          f"gibberish={res['test_split_only']['gibberish']:>5}"
          f"  rate={res['test_split_only']['rate']:.4f}   <- must match the paper")
    print(f"wrote {OUT / f'gibberish_full_{tag}.json'}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], quad="--quad" in sys.argv)
