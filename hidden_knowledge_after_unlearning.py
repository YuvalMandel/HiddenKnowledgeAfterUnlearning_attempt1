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


def load_examples():
    """
    Load one question from the WMDP-bio forget subset and one passage from the Wikitext retain set.
    Per the paper: forget set = bio-remove-split of WMDP, retain set = Wikitext.
    """
    # Forget set: WMDP-bio multiple-choice questions
    ds_forget = load_dataset("cais/wmdp", FORGET_SUBSET, split="test")
    ex_forget = ds_forget[0]

    # Retain set: Wikitext-103 passages (filter out empty/header lines)
    ds_retain = load_dataset("wikitext", WIKITEXT_CONFIG, split="train")
    retain_texts = [
        row["text"] for row in ds_retain
        if len(row["text"].strip()) >= WIKITEXT_MIN_LEN
    ]
    ex_retain = retain_texts[0]

    return ex_forget, ex_retain


def format_mc_question(example):
    """
    Turn a WMDP multiple-choice row into a chat messages list for Llama-instruct models.
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
        "Answer with the letter of the correct option."
    )
    return [{"role": "user", "content": content}]


def format_wikitext_prompt(text, max_words=80):
    """
    Use the first portion of a Wikitext passage as a continuation prompt.
    The model should continue the passage naturally.
    """
    words = text.split()
    prefix = " ".join(words[:max_words])
    return prefix


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
    # 1. Load forget (WMDP-bio) and retain (Wikitext) examples
    forget_example, retain_text = load_examples()
    forget_prompt = format_mc_question(forget_example)
    retain_prompt = format_wikitext_prompt(retain_text)

    # 2. Load models
    print("Loading base model...")
    base_tokenizer, base_model = load_model_and_tokenizer(BASE_MODEL)
    print("Loading unlearned model (PB&J checkpoint-8)...")
    un_tokenizer, un_model = load_model_and_tokenizer(UNLEARNED_MODEL)

    # 3. Run base model
    print("\n=== BASE MODEL RESPONSES (before unlearning) ===")
    print("\n[FORGET - WMDP-Bio] question:")
    print(forget_prompt)
    print("\n[FORGET] base answer:")
    print(generate_answer(base_model, base_tokenizer, forget_prompt))

    print("\n[RETAIN - Wikitext] prompt prefix:")
    print(retain_prompt)
    print("\n[RETAIN] base continuation:")
    print(generate_answer(base_model, base_tokenizer, retain_prompt))

    # 4. Run unlearned model
    print("\n=== UNLEARNED MODEL RESPONSES (after unlearning) ===")
    print(f"\n[FORGET] unlearned answer ({UNLEARNED_MODEL}):")
    print(generate_answer(un_model, un_tokenizer, forget_prompt))

    print("\n[RETAIN] unlearned continuation:")
    print(generate_answer(un_model, un_tokenizer, retain_prompt))


if __name__ == "__main__":
    main()
