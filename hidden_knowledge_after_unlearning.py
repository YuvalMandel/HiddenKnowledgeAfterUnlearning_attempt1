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
# Retain set: Wikitext (used during training as retain set per the paper)
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_MIN_LEN = 100  # minimum characters for a usable passage

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_datasets():
    """
    Load the full WMDP-bio forget set and one Wikitext retain passage.
    Returns the full forget dataset (for iteration) and one retain text.
    """
    ds_forget = load_dataset("cais/wmdp", FORGET_SUBSET, split="test")

    ds_retain = load_dataset("wikitext", WIKITEXT_CONFIG, split="train")
    retain_texts = [
        row["text"] for row in ds_retain
        if len(row["text"].strip()) >= WIKITEXT_MIN_LEN
    ]
    ex_retain = retain_texts[0]

    return ds_forget, ex_retain


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
    ds_forget, retain_text = load_datasets()
    retain_prompt = format_wikitext_prompt(retain_text)
    print(f"Forget set: {len(ds_forget)} questions")

    # 2. Load both models simultaneously.
    # Two 8B bfloat16 models ~= 32 GB, well within the A40's 48 GB.
    print("\nLoading base model...")
    base_tokenizer, base_model = load_model_and_tokenizer(BASE_MODEL)
    print("Loading unlearned model (PB&J checkpoint-8)...")
    un_tokenizer, un_model = load_model_and_tokenizer(UNLEARNED_MODEL)

    # 3. Iterate over forget questions until the two models give different answers.
    print("\nSearching for a question where base and unlearned models disagree...\n")
    divergent_idx = None
    divergent_example = None
    divergent_base_answer = None
    divergent_un_answer = None

    for i, example in enumerate(ds_forget):
        forget_prompt = format_mc_question(example)
        base_answer = generate_answer(base_model, base_tokenizer, forget_prompt)
        un_answer = generate_answer(un_model, un_tokenizer, forget_prompt)

        base_letter = extract_letter(base_answer)
        un_letter = extract_letter(un_answer)

        marker = " <-- DIVERGENT" if base_letter != un_letter else ""
        print(f"[Q{i:04d}] Base={base_letter or '?'}  Unlearned={un_letter or '?'}{marker}")

        if base_letter != un_letter:
            divergent_idx = i
            divergent_example = example
            divergent_base_answer = base_answer
            divergent_un_answer = un_answer
            break

    # 4. Print full details for the divergent question
    print("\n" + "=" * 60)
    if divergent_idx is not None:
        print(f"DIVERGENT ANSWER FOUND at question {divergent_idx}")
    else:
        print("No divergent answer found across all questions.")
    print("=" * 60)

    if divergent_example is not None:
        print("\n[FORGET - WMDP-Bio] question:")
        print(format_mc_question_display(divergent_example))
        print(f"\n[FORGET] base answer:\n{divergent_base_answer}")
        print(f"\n[FORGET] unlearned answer ({UNLEARNED_MODEL}):\n{divergent_un_answer}")

    # 5. Retain set — one passage, both models
    print("\n" + "=" * 60)
    print("RETAIN SET (Wikitext)")
    print("=" * 60)
    print(f"\nPrompt prefix:\n{retain_prompt}")
    print(f"\nBase continuation:\n{generate_answer(base_model, base_tokenizer, retain_prompt)}")
    print(f"\nUnlearned continuation:\n{generate_answer(un_model, un_tokenizer, retain_prompt)}")


if __name__ == "__main__":
    main()
