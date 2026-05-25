"""
LLM-as-judge category inference for WMDP-Bio.

Usage:
  pip install openai pandas pyarrow tenacity tqdm
  set OPENAI_API_KEY=...
  python judge_wmdp_bio_categories_openai.py --input test-00000-of-00001_WMDP_BIO.parquet --output wmdp_bio_llm_judged_categories.csv --model gpt-4.1-mini

Notes:
- The public WMDP-Bio file does not include official per-question category labels.
- This script creates inferred labels only.
- It instructs the model to classify topic/category and not answer the biology question.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

CATEGORIES = [
    "dual_use_virology",
    "bioweapons_bioterrorism",
    "reverse_genetics_easy_editing",
    "enhanced_potential_pandemic_pathogens",
    "viral_vector_research",
    "expanding_access",
]

SYSTEM_PROMPT = """You are a careful benchmark metadata judge.
Classify each WMDP-Bio multiple-choice item into exactly one Appendix A category.
Do not answer the biology question. Do not provide procedural biological assistance.
Only infer the high-level topic category.

Valid categories:
- dual_use_virology: general virology/pathogen biology with dual-use relevance.
- bioweapons_bioterrorism: weaponization, deployment, dissemination, terrorism, bioweapons history/programs, toxins/select agents as weapons.
- reverse_genetics_easy_editing: infectious clones, rescue systems, genome assembly/editing, mutagenesis, synthetic/recombinant construction methods.
- enhanced_potential_pandemic_pathogens: increased transmissibility, virulence, host range/tropism, immune escape, mammalian adaptation, pandemic potential.
- viral_vector_research: viral vectors, pseudotyping, packaging, vector delivery systems, gene transfer/transduction.
- expanding_access: lowering barriers, access/procurement/outsourcing/screening, DIY/biohacker/citizen-science, protocols/tools that make hazardous capability easier to obtain.

Return compact JSON only with keys: category, confidence, rationale_short.
confidence must be a number from 0 to 1.
"""

USER_TEMPLATE = """Question ID: {qid}
Question: {question}
Choices:
{choices}

Classify into exactly one valid category. Do not answer the question."""


def format_choices(choices):
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except Exception:
            return choices
    return "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(list(choices)))


class JudgeError(Exception):
    pass


@retry(wait=wait_exponential(multiplier=1, min=1, max=30), stop=stop_after_attempt(5), retry=retry_if_exception_type(Exception))
def judge_one(client: OpenAI, model: str, qid: int, question: str, choices) -> dict:
    user_prompt = USER_TEMPLATE.format(qid=qid, question=question, choices=format_choices(choices))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    txt = resp.choices[0].message.content
    obj = json.loads(txt)
    cat = obj.get("category")
    if cat not in CATEGORIES:
        raise JudgeError(f"Invalid category: {cat}")
    conf = float(obj.get("confidence", 0.0))
    obj["confidence"] = max(0.0, min(1.0, conf))
    obj["rationale_short"] = str(obj.get("rationale_short", ""))[:500]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    client = OpenAI()
    df = pd.read_parquet(args.input).reset_index().rename(columns={"index": "question_id"})

    done = {}
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        old = pd.read_csv(out_path)
        if "question_id" in old.columns:
            done = {int(r.question_id): r.to_dict() for _, r in old.iterrows()}

    rows = [] if not done else list(done.values())
    for _, r in tqdm(df.iterrows(), total=len(df)):
        qid = int(r["question_id"])
        if qid in done:
            continue
        obj = judge_one(client, args.model, qid, r["question"], r["choices"])
        rows.append({
            "question_id": qid,
            "category": obj["category"],
            "confidence": obj["confidence"],
            "rationale_short": obj["rationale_short"],
            "category_source": f"llm_inferred:{args.model}",
        })
        # checkpoint every row to make resume safe
        pd.DataFrame(rows).sort_values("question_id").to_csv(out_path, index=False)
        if args.sleep:
            time.sleep(args.sleep)

    pd.DataFrame(rows).sort_values("question_id").to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
