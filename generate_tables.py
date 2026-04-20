"""
Reads CSV files from data/ and writes LaTeX table body files to latex/tables/.
Run from the repo root: python generate_tables.py
Each output file contains only the \midrule-separated rows (no \begin{tabular} wrapper),
so the LaTeX section files can \input{tables/<name>} inside their own tabular environment.
"""

import csv
import os
from pathlib import Path

DATA = Path("data")
OUT  = Path("latex/tables")
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "Base":       r"Base (8B-Instruct)",
    "GradDiff":   r"GradDiff",
    "RMU":        r"RMU",
    "RMU-LAT":    r"RMU-LAT",
    "RepNoise":   r"RepNoise",
    "ELM":        r"ELM",
    "RR":         r"RR",
    "TAR":        r"TAR",
    "PB&J":       r"PB\&J",
    "Llama3-8B":  r"Llama-3-8B (raw)",
    "Llama3-70B": r"Llama-3-70B-Instruct",
}

METHOD_ORDER_FULL = ["Base", "Llama3-8B", "GradDiff", "RMU", "RMU-LAT",
                     "RepNoise", "ELM", "RR", "TAR", "PB&J"]

METHOD_ORDER_METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise",
                        "ELM", "RR", "TAR", "PB&J"]

def pct(v, decimals=1):
    """Float 0-1 → 'XX.X\\%'"""
    if v is None or v == "":
        return "--"
    try:
        return f"{float(v)*100:.{decimals}f}\\%"
    except ValueError:
        return "--"

def fmt(v, decimals=3):
    """Float → 'X.XXX'"""
    if v is None or v == "":
        return "--"
    try:
        return f"{float(v):.{decimals}f}"
    except ValueError:
        return "--"

def label(method):
    return METHOD_LABELS.get(method, method)

def load_csv(name):
    path = DATA / name
    if not path.exists():
        print(f"  [WARN] {path} not found")
        return {}
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["method"]] = row
    return rows

def write(name, lines):
    path = OUT / name
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Wrote {path}")

# ── Table 1: Gen + Logit accuracy (bio) ──────────────────────────────────────

def make_table1():
    d = load_csv("summary_table1_gen_logit.csv")
    lines = []
    ref_rows  = ["Base", "Llama3-8B"]
    meth_rows = METHOD_ORDER_METHODS
    for group_idx, group in enumerate([ref_rows, meth_rows]):
        if group_idx > 0:
            lines.append(r"\midrule")
        for m in group:
            if m not in d:
                continue
            r = d[m]
            gib = pct(r.get("gibberish",""))
            row = (
                f"{label(m)} & "
                f"{pct(r.get('gen_acc',''))} & "
                f"{pct(r.get('gen_true',''))} & "
                f"{pct(r.get('gen_false',''))} & "
                f"{gib} & "
                f"{pct(r.get('logit_acc',''))} & "
                f"{pct(r.get('logit_true',''))} & "
                f"{pct(r.get('logit_false',''))} & "
                f"{fmt(r.get('logit_auc',''))} \\\\"
            )
            lines.append(row)
    write("table1_gen_logit.tex", lines)

# ── Table 2 (tab:probes-bio): Method probes on own states ────────────────────

def make_table2():
    d = load_csv("summary_table3_method_probes.csv")
    # Also load base probe on itself from table2_base_probes
    d2 = load_csv("summary_table2_base_probes.csv")
    lines = []

    # Base row from table2 (base probe → base states)
    if "Base" in d2:
        r = d2["Base"]
        lines.append(
            f"{label('Base')} & "
            f"{pct(r.get('pl_lr_acc',''))} & "
            f"{pct(r.get('pl_rf_acc',''))} & "
            f"{pct(r.get('pl_adaboost_acc',''))} & "
            f"{pct(r.get('mb_lr_acc',''))} & "
            f"{pct(r.get('mb_rf_acc',''))} & "
            f"{pct(r.get('mb_adaboost_acc',''))} \\\\"
        )
    lines.append(r"\midrule")

    for m in METHOD_ORDER_METHODS:
        if m not in d:
            continue
        r = d[m]
        lines.append(
            f"{label(m)} & "
            f"{pct(r.get('pl_lr_acc',''))} & "
            f"{pct(r.get('pl_rf_acc',''))} & "
            f"{pct(r.get('pl_adaboost_acc',''))} & "
            f"{pct(r.get('mb_lr_acc',''))} & "
            f"{pct(r.get('mb_rf_acc',''))} & "
            f"{pct(r.get('mb_adaboost_acc',''))} \\\\"
        )
    write("table2_probes_bio.tex", lines)

# ── Table 3 (tab:cross-probe): Cross-probe transfer ──────────────────────────

def make_table3():
    base2meth = load_csv("summary_table2_base_probes.csv")   # base probe → method states
    meth2base = load_csv("summary_table5_cross_probes.csv")  # method probe → base states
    lines = []
    for m in METHOD_ORDER_METHODS:
        b2m = base2meth.get(m, {})
        m2b = meth2base.get(m, {})
        lines.append(
            f"{label(m)} & "
            f"{pct(b2m.get('pl_lr_acc',''))} & "
            f"{pct(b2m.get('mb_lr_acc',''))} & "
            f"{pct(m2b.get('pl_lr_acc',''))} & "
            f"{pct(m2b.get('mb_lr_acc',''))} \\\\"
        )
    write("table3_cross_probe.tex", lines)

# ── Table 4 (tab:cyber): Cyber gen + probe accuracy ──────────────────────────

def make_table4():
    gen  = load_csv("summary_table4_cyber_gen_logit.csv")
    meth = load_csv("summary_table4c_cyber_method_probes.csv")
    lines = []
    ref_rows  = ["Base", "Llama3-8B"]
    meth_rows = METHOD_ORDER_METHODS
    for group_idx, group in enumerate([ref_rows, meth_rows]):
        if group_idx > 0:
            lines.append(r"\midrule")
        for m in group:
            g = gen.get(m, {})
            p = meth.get(m, {})
            lines.append(
                f"{label(m)} & "
                f"{pct(g.get('cyber_acc',''))} & "
                f"{fmt(g.get('cyber_logit_auc',''))} & "
                f"{pct(p.get('pl_lr_acc',''))} & "
                f"{pct(p.get('pl_rf_acc',''))} & "
                f"{pct(p.get('mb_lr_acc',''))} \\\\"
            )
    write("table4_cyber.tex", lines)

# ── Table A1 (appendix): Base probes → all models ────────────────────────────

def make_tableA1():
    d = load_csv("summary_table2_base_probes.csv")
    lines = []
    for m in ["Base"] + METHOD_ORDER_METHODS + ["Llama3-8B"]:
        if m not in d:
            continue
        r = d[m]
        sep = r"\midrule" if m in ("Base", "GradDiff") else None
        if sep and m == "GradDiff":
            lines.append(r"\midrule")
        lines.append(
            f"{label(m)} & "
            f"{pct(r.get('pl_lr_acc',''))} & "
            f"{pct(r.get('pl_lr_true',''))} & "
            f"{pct(r.get('pl_lr_fals',''))} & "
            f"{pct(r.get('mb_lr_acc',''))} & "
            f"{pct(r.get('mb_lr_true',''))} & "
            f"{pct(r.get('mb_lr_fals',''))} \\\\"
        )
    write("tableA1_base_probes.tex", lines)

# ── Table A2 (appendix): Cyber method probes ─────────────────────────────────

def make_tableA2():
    d = load_csv("summary_table4c_cyber_method_probes.csv")
    lines = []
    for m in ["Base"] + METHOD_ORDER_METHODS + ["Llama3-8B"]:
        if m not in d:
            continue
        r = d[m]
        if m == "GradDiff":
            lines.append(r"\midrule")
        lines.append(
            f"{label(m)} & "
            f"{pct(r.get('pl_lr_acc',''))} & "
            f"{pct(r.get('pl_lr_true',''))} & "
            f"{pct(r.get('pl_lr_fals',''))} & "
            f"{pct(r.get('mb_lr_acc',''))} & "
            f"{pct(r.get('mb_lr_true',''))} & "
            f"{pct(r.get('mb_lr_fals',''))} \\\\"
        )
    write("tableA2_cyber_probes.tex", lines)

# ── Table A3 (appendix): MCQ accuracy ────────────────────────────────────────

def make_tableA3():
    d = load_csv("summary_table6_mcq.csv")
    lines = []
    ref_rows  = ["Base", "Llama3-8B"]
    meth_rows = METHOD_ORDER_METHODS
    for group_idx, group in enumerate([ref_rows, meth_rows]):
        if group_idx > 0:
            lines.append(r"\midrule")
        for m in group:
            if m not in d:
                continue
            r = d[m]
            lines.append(
                f"{label(m)} & "
                f"{pct(r.get('bio_mcq_acc',''))} & "
                f"{pct(r.get('bio_gibberish',''))} & "
                f"{pct(r.get('cyber_mcq_acc',''))} & "
                f"{pct(r.get('cyber_gibberish',''))} \\\\"
            )
    write("tableA3_mcq.tex", lines)

# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating LaTeX table bodies...")
    make_table1()
    make_table2()
    make_table3()
    make_table4()
    make_tableA1()
    make_tableA2()
    make_tableA3()
    print("Done. Files in latex/tables/")
