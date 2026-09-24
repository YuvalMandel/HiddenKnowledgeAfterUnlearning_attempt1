#!/usr/bin/env python3
"""Appendix C: the longest rendered prompt, per model family and domain.

Renders every verification prompt through each family's chat template exactly
as knowledge_lens.py does and prints the longest in tokens, which must stay
below MAX_PROMPT_TOKENS (4,096) so that no prompt is ever truncated.
Downloads tokenizers only.

Usage: python plots/prompt_lengths.py                  # all four families
       python plots/prompt_lengths.py zephyr_base      # a subset
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import knowledge_lens as kl                    # noqa: E402

FAMILIES = ["base", "zephyr_base", "mixtral_base", "yi_base"]


def main():
    from transformers import AutoTokenizer
    data = {d: kl.load_wmdp(d) for d in kl.WMDP_CONFIGS}
    longest = {d: 0 for d in data}
    for mid in sys.argv[1:] or FAMILIES:
        tok = AutoTokenizer.from_pretrained(kl.model_path(mid), use_fast=True)
        for domain, items in data.items():
            n = max(len(tok(kl.render_prompt(tok, kl.make_verify_prompt(
                    it["question"], ch))).input_ids)
                    for it in items for ch in it["choices"])
            longest[domain] = max(longest[domain], n)
            print(f"{kl.model_path(mid):<42} {domain:<6} longest {n:>5} tokens")
    for domain, n in longest.items():
        print(f"longest {domain} prompt over all families: {n} "
              f"(cap {kl.MAX_PROMPT_TOKENS})")


if __name__ == "__main__":
    main()
