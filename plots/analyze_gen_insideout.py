"""
Gibberish analysis for inside-out pipeline generation outputs.

Reads inside_out_out/{model_id}/bio_gen_test.json for every model_id
(base + 8 methods × 8 checkpoints = 65 total) and produces:
  - Per-method per-checkpoint gibberish rate table (printed + CSV)
  - Top-10 verbatim gibberish strings per method (at ck8)
  - Structural category breakdown per method (at ck8)

Run from repo root on Newton:
    python plots/analyze_gen_insideout.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

OUT_DIR = Path("inside_out_out")
PLOT_DIR = Path("plots")

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_CHECKPOINTS = 8

# HF slug map (matches SWEEP_SLUGS in inside_out_knowledge.py)
SWEEP_SLUGS = {
    "GradDiff": "graddiff",
    "RMU":      "rmu",
    "RMU-LAT":  "rmu-lat",
    "RepNoise": "repnoise",
    "ELM":      "elm",
    "RR":       "rr",
    "TAR":      "tar",
    "PB_J":     "pbj",
}

DISPLAY_NAMES = {
    "GradDiff": "GradDiff",
    "RMU":      "RMU",
    "RMU-LAT":  "RMU-LAT",
    "RepNoise": "RepNoise",
    "ELM":      "ELM",
    "RR":       "RR",
    "TAR":      "TAR",
    "PB_J":     "PB&J",
}


def model_id_for(method: str, ck: int) -> str:
    return f"{method}_ck{ck}"


def extract_tf(text: str):
    m = re.search(r"\b(True|False)\b", text.strip(), re.IGNORECASE)
    return m.group(1).capitalize() if m else None


def classify_gibberish(text: str) -> str:
    s = text.strip()
    if s == "":
        return "<empty>"
    if re.fullmatch(r"\s*", s):
        return "<whitespace>"
    for tok in re.findall(r"(\S+)\s*", s):
        if len(tok) >= 2 and s.count(tok) >= 5:
            return f"<repetition: '{tok[:20]}'>"
    if " " not in s and len(s) > 40:
        return "<long-no-spaces>"
    if re.fullmatch(r"[^a-zA-Z0-9\s]{3,}", s):
        return "<symbols>"
    if re.match(r"^\d", s):
        return "<starts-with-number>"
    if len(s) < 15:
        return "<short>"
    return s[:60].strip()


def load_gen(model_id: str, domain: str = "bio") -> list | None:
    path = OUT_DIR / model_id / f"{domain}_gen_test.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def gibberish_rate(records: list) -> tuple[int, int, float]:
    """Returns (n_total, n_gib, rate)."""
    n_total = len(records)
    n_gib = sum(1 for r in records if extract_tf(r["text"]) is None)
    return n_total, n_gib, n_gib / n_total if n_total else 0.0


def analyze_ck8(method: str) -> dict:
    """Detailed analysis at checkpoint 8."""
    mid = model_id_for(method, 8)
    records = load_gen(mid)
    if records is None:
        return {"method": method, "missing": True}
    gibberish_records = [r for r in records if extract_tf(r["text"]) is None]
    n_total = len(records)
    n_gib = len(gibberish_records)
    return {
        "method":    method,
        "missing":   False,
        "n_total":   n_total,
        "n_gib":     n_gib,
        "rate":      n_gib / n_total if n_total else 0.0,
        "verbatim":  Counter(r["text"].strip() for r in gibberish_records),
        "structural": Counter(classify_gibberish(r["text"]) for r in gibberish_records),
    }


def print_rate_table(rate_table: dict):
    """rate_table[method][ck] = (n_total, n_gib, rate)"""
    methods_ordered = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
    header = f"{'Method':<12}" + "".join(f"  ck{c}" for c in range(1, 9))
    print("\n" + "=" * 70)
    print("GIBBERISH RATE PER METHOD × CHECKPOINT (bio, inside-out gen)")
    print("=" * 70)
    print(header)
    print("-" * 70)
    for method in methods_ordered:
        row = f"{DISPLAY_NAMES[method]:<12}"
        for ck in range(1, 9):
            entry = rate_table.get(method, {}).get(ck)
            if entry is None:
                row += "    —  "
            else:
                _, _, rate = entry
                row += f"  {rate:5.1%}"
        print(row)


def save_rate_csv(rate_table: dict, out_path: Path):
    rows = ["method," + ",".join(f"ck{c}" for c in range(1, 9))]
    for method in ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]:
        vals = []
        for ck in range(1, 9):
            entry = rate_table.get(method, {}).get(ck)
            vals.append("" if entry is None else f"{entry[2]:.4f}")
        rows.append(f"{DISPLAY_NAMES[method]}," + ",".join(vals))
    out_path.write_text("\n".join(rows))
    print(f"\nRate CSV saved → {out_path}")


def print_verbatim(ck8_results: list, top_n: int = 10, trunc: int = 100):
    print("\n" + "=" * 70)
    print(f"TOP-{top_n} VERBATIM GIBBERISH STRINGS AT CK8 (bio)")
    print("=" * 70)
    for r in ck8_results:
        if r.get("missing") or r["n_gib"] == 0:
            continue
        print(f"\n── {DISPLAY_NAMES[r['method']]}  ({r['n_gib']} / {r['n_total']}) ──")
        for string, cnt in r["verbatim"].most_common(top_n):
            display = repr(string[:trunc]) + ("…" if len(string) > trunc else "")
            print(f"  {cnt:6d}x  {display}")


def print_structural(ck8_results: list, top_n: int = 8):
    print("\n" + "=" * 70)
    print("STRUCTURAL CATEGORIES AT CK8 (bio)")
    print("=" * 70)
    for r in ck8_results:
        if r.get("missing") or r["n_gib"] == 0:
            continue
        print(f"\n── {DISPLAY_NAMES[r['method']]} ──")
        for cat, cnt in r["structural"].most_common(top_n):
            pct = cnt / r["n_gib"] * 100
            print(f"  {cnt:6d}  ({pct:5.1f}%)  {cat}")


def check_base():
    records = load_gen("base")
    if records is None:
        print("[base] bio_gen_test.json not found")
        return
    n_total, n_gib, rate = gibberish_rate(records)
    print(f"[base] {n_total} records, {n_gib} gibberish ({rate:.1%})")


def main():
    check_base()

    # Build per-method × per-checkpoint rate table
    rate_table: dict[str, dict[int, tuple]] = {}
    for method in ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]:
        rate_table[method] = {}
        for ck in range(1, N_CHECKPOINTS + 1):
            mid = model_id_for(method, ck)
            records = load_gen(mid)
            if records is None:
                print(f"  [missing] {mid}")
                continue
            rate_table[method][ck] = gibberish_rate(records)

    print_rate_table(rate_table)
    save_rate_csv(rate_table, PLOT_DIR / "insideout_gibberish_rates.csv")

    # Detailed ck8 analysis
    ck8_results = [analyze_ck8(m) for m in
                   ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]]
    print_verbatim(ck8_results)
    print_structural(ck8_results)


if __name__ == "__main__":
    main()
