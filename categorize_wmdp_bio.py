# -*- coding: utf-8 -*-
"""
Categorize WMDP-bio questions using the base model (Llama-3-8B-Instruct).

Greedy-generates 1 token per question. Tokens A-F map to a category;
anything else is recorded as gibberish (actual token string saved).

Prompt: few-shot MCQ style with one demo per letter so no category is primed.

Output: data/wmdp_bio_categories.csv
"""
import os
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID   = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 32
REPO       = Path(__file__).parent
DATA_DIR   = REPO / "data"
OUT_CSV    = DATA_DIR / "wmdp_bio_categories.csv"

CATEGORIES = {
    "A": "Dual-use Virology",
    "B": "Bioweapons & Bioterrorism",
    "C": "Reverse Genetics & Easy Editing",
    "D": "Bio Enhanced Potential Pandemic Pathogens",
    "E": "Viral Vector Research",
    "F": "Expanding Access",
}
LETTERS = list(CATEGORIES.keys())

# One demo per letter, interleaved order — no positional bias toward any category.
FEW_SHOT_HEADER = """Classify each biosecurity question into exactly one of the six categories below.
Reply with only the letter (A-F). All six categories are equally common.

A) Dual-use Virology
B) Bioweapons & Bioterrorism
C) Reverse Genetics & Easy Editing
D) Bio Enhanced Potential Pandemic Pathogens
E) Viral Vector Research
F) Expanding Access

Question: Which herpesviruses are most commonly associated with encephalitis in immunocompromised patients?
Answer: A

Question: What regulatory frameworks govern the export of select biological agents across international borders?
Answer: F

Question: How can reverse genetics systems be used to recover infectious virus from cloned cDNA?
Answer: C

Question: What properties of a pathogen are assessed when determining pandemic potential under the NSABB framework?
Answer: D

Question: Which adeno-associated virus serotypes show highest tropism for neurons in vivo?
Answer: E

Question: What historical programs weaponized anthrax spores for large-scale aerosol dispersal?
Answer: B

"""


def make_prompt(question: str) -> str:
    return FEW_SHOT_HEADER + f"Question: {question}\nAnswer:"


def get_valid_letter_ids(tokenizer):
    """Return set of token IDs that decode to a bare letter A-F."""
    valid = {}
    for letter in LETTERS:
        for candidate in [letter, " " + letter, letter.lower(), " " + letter.lower()]:
            toks = tokenizer.encode(candidate, add_special_tokens=False)
            if len(toks) == 1:
                valid[toks[0]] = letter.upper()
    return valid   # tok_id -> canonical letter


@torch.inference_mode()
def generate_batch(prompts, model, tokenizer):
    """Greedy-generate exactly 1 token per prompt. Return list of decoded strings."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=2048
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[:, inputs["input_ids"].shape[1]:]
    return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in new_tokens]


def main():
    tf = pd.read_csv(DATA_DIR / "wmdp_tf_pairs.csv")[
        ["original_id", "question"]
    ].drop_duplicates("original_id").reset_index(drop=True)
    print(f"Unique WMDP-bio questions: {len(tf)}")

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    questions = tf["question"].tolist()
    orig_ids  = tf["original_id"].tolist()
    n         = len(questions)
    raw_tokens = []

    for start in range(0, n, BATCH_SIZE):
        batch_q = questions[start : start + BATCH_SIZE]
        prompts = [make_prompt(q) for q in batch_q]
        tokens  = generate_batch(prompts, model, tokenizer)
        raw_tokens.extend(tokens)
        print(f"  {min(start + BATCH_SIZE, n)}/{n}", end="\r", flush=True)

    print(f"\nDone. Building output CSV ...")

    rows = []
    gibberish_count = 0
    for orig_id, question, tok in zip(orig_ids, questions, raw_tokens):
        upper = tok.upper()
        if upper in CATEGORIES:
            pred_letter   = upper
            pred_category = CATEGORIES[upper]
            is_gibberish  = False
        else:
            pred_letter   = "?"
            pred_category = "gibberish"
            is_gibberish  = True
            gibberish_count += 1
        rows.append({
            "original_id":        orig_id,
            "question":           question,
            "raw_token":          tok,
            "predicted_letter":   pred_letter,
            "predicted_category": pred_category,
            "is_gibberish":       is_gibberish,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(f"\nGibberish: {gibberish_count}/{n} ({100*gibberish_count/n:.1f}%)")
    print(f"\nCategory distribution:")
    print(df_out["predicted_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
