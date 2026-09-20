"""Are the 9 OPTML Llama-3-8B-Instruct WMDP models comparable to OUR base run?

Three things must hold before we spend GPU hours on them, and each has already
bitten us once on another family:

  1. The True/False token ids must match the ids our published Llama runs used
     (3082 / 3641). 7 of the 9 report vocab_size 128257 vs the stock 128256, so
     "an appended token cannot shift existing ids" is an assumption, not a fact
     -- check it against the real tokenizer (KNOWN_ISSUES #34).
  2. The chat template must be byte-identical to stock, or K_ext is measuring a
     different prompt (the Mixtral trap: same words, different turn structure).
  3. Nothing else added to the vocabulary that lands where we read.

Run on Newton, where the gated base tokenizer is cached:
    python plots/check_optml_llama_tokens.py
Downloads only tokenizer files (~9 MB each), no weights.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from inside_out_knowledge import (BASE_MODEL_ID, make_verify_prompt,
                                  render_prompt, tf_token_ids)

from transformers import AutoTokenizer

REPOS = [
    "OPTML-Group/DPO-WMDP-llama3-8b-instruct",
    "OPTML-Group/GradDiff-WMDP-llama3-8b-instruct",
    "OPTML-Group/IDK-AP-WMDP-llama3-8b-instruct",
    "OPTML-Group/ILU-RMU-WMDP-llama3-8b-instruct",
    "OPTML-Group/NPO-ILU-WMDP-llama3-8b-instruct",
    "OPTML-Group/NPO-SAM-WMDP-llama3-8b-instruct",
    "OPTML-Group/NPO-WMDP-llama3-8b-instruct",
    "OPTML-Group/SimNPO-WMDP-llama3-8b-instruct",
    "OPTML-Group/UNDIAL-WMDP-llama3-8b-instruct",
]

# a real-shaped WMDP-bio item; content is irrelevant, the rendering is the point
Q = "What is a reason that Rift Valley fever virus is considered a bioweapon risk?"
CH = "It can be transmitted by aerosol and infects both livestock and humans."


def legacy_ids(tok):
    """The rule every published Llama run used."""
    return (tok.encode(" True", add_special_tokens=False)[-1],
            tok.encode(" False", add_special_tokens=False)[-1])


def probe(tok):
    prompt = render_prompt(tok, make_verify_prompt(Q, CH))
    return dict(
        n_vocab=len(tok),                       # includes added tokens
        legacy=legacy_ids(tok),
        derived=tf_token_ids(tok),
        added=tuple(sorted(tok.get_added_vocab().items(), key=lambda kv: kv[1])[-3:]),
        prompt=prompt,
        n_tok=len(tok.encode(prompt, add_special_tokens=False)),
    )


def main():
    print(f"reference: {BASE_MODEL_ID}")
    ref_tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=True)
    ref = probe(ref_tok)
    print(f"  len(tok)      {ref['n_vocab']}")
    print(f"  legacy ids    True={ref['legacy'][0]} False={ref['legacy'][1]}"
          f"   <- what the paper used")
    print(f"  derived ids   True={ref['derived'][0]} False={ref['derived'][1]}")
    print(f"  prompt        {ref['n_tok']} tokens")
    print(f"  last added    {ref['added']}")
    print()

    hdr = f"{'repo':46s} {'len':>7s} {'legacy':>12s} {'derived':>12s} {'tmpl':>6s} {'verdict':>9s}"
    print(hdr); print("-" * len(hdr))

    all_ok = True
    for r in REPOS:
        try:
            t = AutoTokenizer.from_pretrained(r, use_fast=True)
            p = probe(t)
        except Exception as e:
            print(f"{r:46s} ERROR {repr(e)[:60]}")
            all_ok = False
            continue
        same_prompt = p["prompt"] == ref["prompt"]
        same_legacy = p["legacy"] == ref["legacy"]
        ok = same_prompt and same_legacy
        all_ok &= ok
        print(f"{r.split('/')[1]:46s} {p['n_vocab']:7d} "
              f"{str(p['legacy']):>12s} {str(p['derived']):>12s} "
              f"{('same' if same_prompt else 'DIFF'):>6s} "
              f"{('OK' if ok else 'MISMATCH'):>9s}")
        if not same_prompt:
            print(f"    ref    {ref['prompt']!r}")
            print(f"    theirs {p['prompt']!r}")
        if p["added"] != ref["added"]:
            print(f"    added tokens differ: {p['added']}")

    print()
    print("VERDICT:", "all 9 are drop-in against our base" if all_ok
          else "*** at least one differs -- do NOT launch ***")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
