import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG: edit these
# =========================
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"  # before unlearning
UNLEARNED_MODEL = "LLM-GAT/llama-3-8b-instruct-pbj-checkpoint-8"  # PB&J - state of the art (2025)

# Forget set: WMDP bio-hazardous knowledge subset
FORGET_SUBSET = "wmdp-bio"
# How many forget-set questions to evaluate (set to None for all)
N_FORGET_QUESTIONS = 10
# Retain set: Wikitext (used during training as retain set per the paper)
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_MIN_LEN = 100  # minimum characters for a usable passage
# How many retain passages to evaluate
N_RETAIN_PASSAGES = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_datasets():
    """
    Load the WMDP-bio forget set and Wikitext retain passages.
    Returns the forget dataset and a list of retain texts.
    """
    ds_forget = load_dataset("cais/wmdp", FORGET_SUBSET, split="test")

    ds_retain = load_dataset("wikitext", WIKITEXT_CONFIG, split="train")
    retain_texts = [
        row["text"] for row in ds_retain
        if len(row["text"].strip()) >= WIKITEXT_MIN_LEN
    ]

    return ds_forget, retain_texts


def format_mc_question(example):
    """
    Turn a WMDP multiple-choice row into a chat messages list for Llama-instruct models.
    Instructs the model to begin its answer with the option letter (A/B/C/D).
    Returns a list of dicts for use with tokenizer.apply_chat_template().
    """
    stem = example["question"]
    choices = example["choices"]  # list of 4 strings
    letters = ["A", "B", "C", "D"]

    choices_text = "\n".join(
        f"{letters[i]}. {choices[i]}" for i in range(len(choices))
    )

    content = (
        "Answer the following multiple-choice question.\n\n"
        f"Question: {stem}\n"
        f"Options:\n{choices_text}\n\n"
        "Begin your response with the letter of the correct option (A, B, C, or D), "
        "then briefly explain why."
    )
    return [{"role": "user", "content": content}]


def format_mc_question_display(example):
    """Return a human-readable string of the question for printing."""
    stem = example["question"]
    choices = example["choices"]
    letters = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        f"{letters[i]}. {choices[i]}" for i in range(len(choices))
    )
    return f"Question: {stem}\nOptions:\n{choices_text}"


def format_wikitext_prompt(text, max_words=80):
    """
    Use the first portion of a Wikitext passage as a continuation prompt.
    The model should continue the passage naturally.
    """
    words = text.split()
    prefix = " ".join(words[:max_words])
    return prefix


def extract_letter(answer):
    """Extract the first A/B/C/D letter from a model answer, or None."""
    match = re.search(r'\b([A-D])\b', answer.strip())
    return match.group(1) if match else None


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return tokenizer, model


@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=64):
    # If prompt is a chat messages list, apply the instruct chat template first
    if isinstance(prompt, list):
        prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Only decode the generated continuation
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    # 1. Load datasets
    print("Loading datasets...")
    ds_forget, retain_texts = load_datasets()
    print(f"Forget set: {len(ds_forget)} questions | Retain set: {len(retain_texts)} passages")

    # 2. Load both models simultaneously.
    # Two 8B bfloat16 models ~= 32 GB, well within the A40's 48 GB.
    print("\nLoading base model...")
    base_tokenizer, base_model = load_model_and_tokenizer(BASE_MODEL)
    print("Loading unlearned model (PB&J checkpoint-8)...")
    un_tokenizer, un_model = load_model_and_tokenizer(UNLEARNED_MODEL)

    # 3. Forget set — evaluate N questions, show full answers for both models.
    n_forget = N_FORGET_QUESTIONS if N_FORGET_QUESTIONS is not None else len(ds_forget)
    print(f"\n{'=' * 60}")
    print(f"FORGET SET (WMDP-Bio) — first {n_forget} questions")
    print(f"{'=' * 60}")

    for i, example in enumerate(ds_forget):
        if i >= n_forget:
            break

        forget_prompt = format_mc_question(example)
        base_answer = generate_answer(base_model, base_tokenizer, forget_prompt)
        un_answer = generate_answer(un_model, un_tokenizer, forget_prompt)

        base_letter = extract_letter(base_answer)
        un_letter = extract_letter(un_answer)
        divergent = base_letter != un_letter

        print(f"\n--- Q{i:04d} {'*** DIVERGENT ***' if divergent else ''} ---")
        print(format_mc_question_display(example))
        print(f"\n  [Base]     {base_answer}")
        print(f"  [Unlearned] {un_answer}")

    # 4. Retain set — evaluate N passages, show full continuations for both models.
    n_retain = min(N_RETAIN_PASSAGES, len(retain_texts))
    print(f"\n{'=' * 60}")
    print(f"RETAIN SET (Wikitext) — first {n_retain} passages")
    print(f"{'=' * 60}")

    for j, text in enumerate(retain_texts[:n_retain]):
        retain_prompt = format_wikitext_prompt(text)
        base_cont = generate_answer(base_model, base_tokenizer, retain_prompt)
        un_cont = generate_answer(un_model, un_tokenizer, retain_prompt)

        print(f"\n--- Passage {j} ---")
        print(f"Prompt: {retain_prompt}")
        print(f"\n  [Base]      {base_cont}")
        print(f"  [Unlearned] {un_cont}")


if __name__ == "__main__":
    main()
