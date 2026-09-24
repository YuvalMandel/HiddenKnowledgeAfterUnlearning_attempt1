"""Shared paths, loaders and style for the figure and number scripts."""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl                  # noqa: E402
import pandas as pd                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("KNOWLEDGE_LENS_OUT", ROOT / "outputs"))
FIG_DIR = ROOT / "figures"
DATA_DIR = ROOT / "data"

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_CHECKPOINTS = 8


def label(method: str) -> str:
    """Display name of a method or model id's method part."""
    return method.replace("PB_J", "PB&J")


# Colours shared by every figure
INT_C, EXT_C, WMDP_C = "#2166ac", "#d6604d", "#4d4d4d"
WMDP_TEXT_C = "#5a5a5a"

K_CHANCE, WMDP_CHANCE = 0.5, 0.25   # pairwise K vs four-way WMDP accuracy


# ── Scores ─────────────────────────────────────────────────────────────────────

def load_k(model_id: str, domain: str = "bio") -> pd.DataFrame:
    """Per-question K_int and K_ext for one model, as written by knowledge_lens.py."""
    name = "k_scores.parquet" if domain == "bio" else f"k_scores_{domain}.parquet"
    df = pd.read_parquet(OUT_DIR / model_id / name)
    # The probe configuration every reported number uses.
    return df[(df.split_type == "cv") & (df.clf == "LR") & (df.probe_type == "own")
              & (df.layer_config == "best_layer") & (df.domain == domain)]


def k_summary(model_id: str, domain: str = "bio") -> dict:
    """Mean K_int and K_ext, and their standard deviation across the five folds."""
    f = load_k(model_id, domain).groupby("fold").agg(
        ki=("k_internal", "mean"), ke=("k_external", "mean"))
    return dict(k_int=f.ki.mean(), k_ext=f.ke.mean(),
                k_int_sd=f.ki.std(ddof=1), k_ext_sd=f.ke.std(ddof=1))


def retention(value: float, chance: float, base: float) -> float:
    """% of the base model's above-chance signal retained: 100 (v-c)/(b-c)."""
    return 100.0 * (value - chance) / (base - chance)


def published_accuracy(suite: str, domain: str = "bio") -> dict:
    """Published four-way WMDP accuracies, {model_id: accuracy}."""
    df = pd.read_csv(DATA_DIR / "published_wmdp_accuracy.csv")
    df = df[(df.suite == suite) & (df.domain == domain)]
    return dict(zip(df.model_id, df.accuracy))


def state_shares(model_id: str, domain: str = "bio") -> dict:
    """% of questions in each state (Section 4.4), thresholding both K at 1/2."""
    d = load_k(model_id, domain)
    ki, ke = d.k_internal > K_CHANCE, d.k_external > K_CHANCE
    return {"retained": 100 * (ki & ke).mean(), "suppressed": 100 * (ki & ~ke).mean(),
            "forgotten": 100 * (~ki & ~ke).mean(), "lucky": 100 * (~ki & ke).mean()}


# ── Figures ────────────────────────────────────────────────────────────────────

TEXTWIDTH_IN = 5.5   # ICLR text width; figures are authored at print size


def figsize(aspect: float, width_frac: float = 1.0) -> tuple[float, float]:
    """(width, height) in inches at the paper's print width; aspect = h / w."""
    w = TEXTWIDTH_IN * width_frac
    return (w, w * aspect)


def use_style() -> None:
    mpl.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.titlesize": 10,
        # STIXGeneral ships with matplotlib and matches Times, so machines
        # without Times New Roman still reproduce the paper's layout.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.1,
        "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


def save(fig, stem: str) -> None:
    """Write figures/<stem>.pdf (as in the paper) and a .png preview."""
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", **({"dpi": 200} if ext == "png" else {}))
        print("wrote", p)
