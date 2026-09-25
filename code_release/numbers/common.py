"""Shared paths, loaders and published accuracies for the number scripts."""
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("KNOWLEDGE_LENS_OUT", ROOT / "outputs"))

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
N_CHECKPOINTS = 8

K_CHANCE, WMDP_CHANCE = 0.5, 0.25   # pairwise K vs four-way WMDP accuracy


def label(method: str) -> str:
    """Display name of a method or model id's method part."""
    return method.replace("PB_J", "PB&J")


# ── Published four-way WMDP accuracies (not computed here) ─────────────────────
# {(suite, domain): {model_id: accuracy}}, each with its source.
PUBLISHED_ACCURACY = {
    # LLM-GAT (Che et al., 2025), Table 2: each method's best checkpoint by their
    # unlearning score, compared with our final checkpoint.
    ("llm-gat", "bio"): {
        "base": 0.70, "GradDiff_ck8": 0.25, "RMU_ck8": 0.26, "RMU-LAT_ck8": 0.32,
        "RepNoise_ck8": 0.29, "ELM_ck8": 0.24, "RR_ck8": 0.26, "TAR_ck8": 0.28,
        "PB_J_ck8": 0.31,
    },
    # WMDP (Li et al., 2024), Table 1.
    ("wmdp", "bio"): {
        "zephyr_base": 0.637, "zephyr_rmu": 0.312, "mixtral_base": 0.748,
        "mixtral_rmu": 0.340, "yi_base": 0.753, "yi_rmu": 0.307,
    },
    ("wmdp", "cyber"): {
        "zephyr_base": 0.440, "zephyr_rmu": 0.282, "mixtral_base": 0.520,
        "mixtral_rmu": 0.308, "yi_base": 0.497, "yi_rmu": 0.290,
    },
    # OPTML suite: 1 - UE from Fan et al. (ICML 2025), Table 2 for the NPO
    # variants and the appendix table of NPO, GradDiff and RMU with and without
    # SAM for GradDiff; SimNPO and the base model from the SimNPO model card
    # (1 - Acc_Bio).
    ("optml", "bio"): {
        "zephyr_base": 0.648, "zephyr_simnpo": 0.416, "zephyr_npo": 0.26,
        "zephyr_npo_sam": 0.26, "zephyr_npo_gp": 0.27, "zephyr_npo_cr": 0.25,
        "zephyr_npo_rs": 0.26, "zephyr_npo_wa": 0.26, "zephyr_graddiff": 0.27,
        "zephyr_graddiff_sam": 0.28,
    },
}


def published_accuracy(suite: str, domain: str = "bio") -> dict:
    return PUBLISHED_ACCURACY[suite, domain]


# ── Scores written by knowledge_lens.py ────────────────────────────────────────

def load_k(model_id: str, domain: str = "bio") -> pd.DataFrame:
    """Per-question K_int and K_ext for one model."""
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


def state_shares(model_id: str, domain: str = "bio") -> dict:
    """% of questions in each state (Section 4.4), thresholding both K at 1/2."""
    d = load_k(model_id, domain)
    ki, ke = d.k_internal > K_CHANCE, d.k_external > K_CHANCE
    return {"retained": 100 * (ki & ke).mean(), "suppressed": 100 * (ki & ~ke).mean(),
            "forgotten": 100 * (~ki & ~ke).mean(), "lucky": 100 * (~ki & ke).mean()}
