# -*- coding: utf-8 -*-
"""
Categorize WMDP-bio questions using the base model (Llama-3-8B-Instruct).

Logit scoring (no generation): forward pass, take argmax over logits of A-F
tokens at the final position. Categories are shuffled per question (fixed seed)
so letter-position bias (e.g. model preference for "A") averages out.

Output: data/wmdp_bio_categories.csv
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID   = "meta-llama/Meta-Llama-3-8B-Instruct"
BATCH_SIZE = 16
SEED       = 42
REPO       = Path(__file__).parent
DATA_DIR   = REPO / "data"
OUT_CSV    = DATA_DIR / "wmdp_bio_categories.csv"

CATEGORIES = [
    "Dual-use Virology",
    "Bioweapons & Bioterrorism",
    "Reverse Genetics & Easy Editing",
    "Bio Enhanced Potential Pandemic Pathogens",
    "Viral Vector Research",
    "Expanding Access",
]
LETTERS = ["A", "B", "C", "D", "E", "F"]


def get_letter_tok_ids(tokenizer):
    """For each letter A-F, collect all single-token IDs (with/without space prefix)."""
    result = {L: [] for L in LETTERS}
    for letter in LETTERS:
        for candidate in [letter, " " + letter, letter.lower(), " " + letter.lower()]:
            toks = tokenizer.encode(candidate, add_special_tokens=False)
            if len(toks) == 1:
                result[letter].append(toks[0])
    return result  # letter -> list[int]


def make_prompt(tokenizer, question: str, perm: np.ndarray) -> str:
    shuffled = [CATEGORIES[i] for i in perm]
    lines = "\n".join(f"{L}) {c}" for L, c in zip(LETTERS, shuffled))
    msg = (
        "Classify this biosecurity question into exactly one of the six categories "
        "below. Reply with only the letter (A–F).\n\n"
        f"{lines}\n\nQuestion: {question}"
    )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": msg}],
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.inference_mode()
def score_batch(prompts, model, tokenizer, letter_tok_ids):
    """Return predicted letter index (0-5) for each prompt via logit scoring."""
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=2048,
    ).to(model.device)
    logits_last = model(**inputs).logits[:, -1, :]  # [B, vocab]

    # For each letter, take the max logit across all its valid token representations
    B = logits_last.shape[0]
    letter_scores = torch.full((B, len(LETTERS)), float("-inf"), device=logits_last.device)
    for j, letter in enumerate(LETTERS):
        tok_ids = letter_tok_ids[letter]
        if tok_ids:
            letter_scores[:, j] = logits_last[:, tok_ids].max(dim=1).values

    return letter_scores.argmax(dim=1).cpu().numpy()  # [B] index into LETTERS


def main():
    tf = pd.read_csv(DATA_DIR / "wmdp_tf_pairs.csv")[
        ["original_id", "question"]
    ].drop_duplicates("original_id").reset_index(drop=True)
    n = len(tf)
    print(f"Unique WMDP-bio questions: {n}")

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(len(CATEGORIES)) for _ in range(n)]

    print(f"Loading {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    letter_tok_ids = get_letter_tok_ids(tokenizer)
    print("Letter token IDs:", {L: ids for L, ids in letter_tok_ids.items()})

    questions = tf["question"].tolist()
    orig_ids  = tf["original_id"].tolist()
    winner_indices = []  # index into LETTERS for each question

    for start in range(0, n, BATCH_SIZE):
        batch_q    = questions[start : start + BATCH_SIZE]
        batch_perm = perms[start : start + BATCH_SIZE]
        prompts    = [make_prompt(tokenizer, q, p) for q, p in zip(batch_q, batch_perm)]
        winners    = score_batch(prompts, model, tokenizer, letter_tok_ids)
        winner_indices.extend(winners.tolist())
        print(f"  {min(start + BATCH_SIZE, n)}/{n}", end="\r", flush=True)

    print(f"\nBuilding output CSV ...")
    rows = []
    for orig_id, question, perm, winner_j in zip(orig_ids, questions, perms, winner_indices):
        pred_category = CATEGORIES[perm[winner_j]]
        rows.append({
            "original_id":        orig_id,
            "question":           question,
            "predicted_letter":   LETTERS[winner_j],
            "predicted_category": pred_category,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(f"\nCategory distribution:")
    print(df_out["predicted_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
