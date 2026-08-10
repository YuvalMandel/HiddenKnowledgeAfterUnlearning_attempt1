#!/usr/bin/env python3
from pathlib import Path
import matplotlib as mpl
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
EXT_DIR = REPO / "inside_out_ext"
PARQUET_PATH = REPO / "plots" / "all_k_scores.parquet"
SEED = 42
TRAIN_SIZE = 500
VAL_SIZE = 200
N_QUESTIONS = 1273
N_OPTIONS = 4
KNOWS_THRESHOLD = 0.5
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]
SUBSET_ORDER = ["retained", "suppressed", "forgotten", "lucky"]

# ICLR 2027 text rectangle (from the template's own instructions: 5.5in x 9in).
# A figure authored wider than this is scaled down by \includegraphics
# [width=\linewidth], and its fonts shrink by the same factor -- authoring at
# the true print width keeps point sizes honest.
ICLR_TEXTWIDTH_IN = 5.5
ICLR_TEXTHEIGHT_IN = 9.0


def iclr_figsize(aspect=0.62, width_frac=1.0):
    """Figure size at ICLR's real print width. aspect = height/width."""
    w = ICLR_TEXTWIDTH_IN * width_frac
    return (w, w * aspect)


def use_iclr_style():
    """Point sizes that render as-specified once the figure is placed at 5.5in.

    Must be paired with iclr_figsize() -- an explicit figsize= passed to
    plt.subplots() overrides rcParams["figure.figsize"], so setting the style
    alone silently does nothing.
    """
    mpl.rcParams.update({
        "figure.figsize": iclr_figsize(),
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.1,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,   # embed TrueType so reviewers' viewers match ours
    })


def _selfcheck_iclr_style():
    """Round-trip check: author with the style, then read back embedded font pt.

    Guards the actual failure mode -- a figure authored wider than 5.5in gets
    scaled down by LaTeX and its fonts shrink with it.
    """
    import tempfile

    import fitz
    import matplotlib.pyplot as plt

    use_iclr_style()
    fig, ax = plt.subplots(figsize=iclr_figsize())
    ax.plot([0, 1], [0, 1], label="probe")
    ax.set_xlabel("layer")
    ax.set_ylabel("AUC")
    ax.legend()

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.pdf"
        fig.savefig(p)
        plt.close(fig)
        doc = fitz.open(p)
        page = doc[0]
        w_in = page.rect.width / 72.0
        sizes = [
            s["size"]
            for b in page.get_text("dict")["blocks"]
            for line in b.get("lines", [])
            for s in line.get("spans", [])
            if s["text"].strip()
        ]
        doc.close()

    assert sizes, "no selectable text found -- fonts got rasterised to paths"
    # bbox="tight" trims margins, so width lands at or under the text column
    assert w_in <= ICLR_TEXTWIDTH_IN + 0.01, f"authored {w_in:.2f}in > column"
    assert min(sizes) >= 6.5, f"smallest font {min(sizes):.1f}pt is too small"
    print(f"OK  width={w_in:.2f}in (<= {ICLR_TEXTWIDTH_IN}in), "
          f"min font={min(sizes):.1f}pt, scale=1.0 -> renders as authored")


def method_fname(method_name: str) -> str:
    return method_name.replace("/", "_").replace("&", "_")


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def load_correct_idx():
    tf = pd.read_csv(REPO / "data" / "wmdp_tf_pairs.csv")
    q = tf[["original_id", "correct_idx"]].drop_duplicates("original_id")
    return q.sort_values("original_id")["correct_idx"].astype(int).to_numpy()


def load_split_indices():
    te = np.random.default_rng(SEED).permutation(N_QUESTIONS)[TRAIN_SIZE + VAL_SIZE :]
    orig_to_pos = {int(qi): i for i, qi in enumerate(te)}
    return te, orig_to_pos


def load_scores_df():
    return pd.read_parquet(PARQUET_PATH)


def compute_prefilter_mask(te, orig_to_pos, df):
    base_single = df[
        (df["model_id"] == "base")
        & (df["split_type"] == "single")
        & (df["layer_config"] == "full")
        & (df["clf"] == "LR")
        & (df["domain"] == "bio")
    ][["question_idx", "k_internal", "k_external"]]

    filt_mask = np.zeros(len(te), dtype=bool)
    for _, row in base_single.iterrows():
        pos = orig_to_pos.get(int(row["question_idx"]))
        if pos is not None and float(row["k_internal"]) == 1.0 and float(row["k_external"]) == 1.0:
            filt_mask[pos] = True
    return filt_mask


def compute_k_ext(ext_arr, correct_idx, qidx):
    k = np.zeros(len(qidx), np.float32)
    for i, qi in enumerate(qidx):
        c = int(correct_idx[qi])
        ws = [ext_arr[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(ext_arr[qi, c] > w) for w in ws) / len(ws)
    return k


def compute_k_int_from_proba(proba_arr, correct_idx, qidx):
    k = np.zeros(len(qidx), np.float32)
    for i, qi in enumerate(qidx):
        c = int(correct_idx[qi])
        ws = [proba_arr[qi, j] for j in range(N_OPTIONS) if j != c]
        k[i] = sum(float(proba_arr[qi, c] > w) for w in ws) / len(ws)
    return k


def get_parquet_full_single(df):
    return df[
        (df["domain"] == "bio")
        & (df["clf"] == "LR")
        & (df["layer_config"] == "full")
        & (df["split_type"] == "single")
    ][["model_id", "question_idx", "k_internal", "k_external"]]


def get_parquet_multi_single(df):
    """Kept for backward compatibility — prefer get_parquet_full_single."""
    return get_parquet_full_single(df)


def get_cv_full_df(df):
    """CV folds for the full-layer (all-33-layer) probe — primary K_int source."""
    return df[
        (df["split_type"] == "cv")
        & (df["domain"] == "bio")
        & (df["clf"] == "LR")
        & (df["probe_type"] == "own")
        & (df["layer_config"] == "full")
    ].copy()


def get_cv_layer_df(df):
    """Per-layer CV data (kept for reference; use get_cv_full_df for K_int)."""
    sub = df[
        (df["split_type"] == "cv")
        & (df["domain"] == "bio")
        & (df["clf"] == "LR")
        & (df["probe_type"] == "own")
        & (df["layer_config"].astype(str).str.match(r"^layer_\d+$"))
    ].copy()
    sub["layer_idx"] = sub["layer_config"].str.extract(r"layer_(\d+)").astype(int)
    return sub


def get_method_ck8_labels(method, correct_idx, filt_te, filt_orig_to_pos, parquet_full_single):
    fname = method_fname(method)
    ext_path = EXT_DIR / f"{fname}_ck8_bio_ext.npy"
    if not ext_path.exists():
        return None, None, None
    mk_ext = compute_k_ext(np.load(ext_path), correct_idx, filt_te)

    model_id = f"{fname}_ck8"
    mk_int_rows = parquet_full_single[parquet_full_single["model_id"] == model_id][
        ["question_idx", "k_internal"]
    ]
    if not mk_int_rows.empty:
        mk_int = np.full(len(filt_te), np.nan, dtype=np.float32)
        for _, row in mk_int_rows.iterrows():
            pos = filt_orig_to_pos.get(int(row["question_idx"]))
            if pos is not None:
                mk_int[pos] = float(row["k_internal"])
        k_int_source = "full-LR"
    else:
        proba_path = EXT_DIR / f"{fname}_ck8_bio_int_proba.npy"
        if not proba_path.exists():
            return None, None, None
        mk_int = compute_k_int_from_proba(np.load(proba_path), correct_idx, filt_te)
        k_int_source = "layer_26_fallback"

    mk_i = mk_int > KNOWS_THRESHOLD
    mk_e = mk_ext > KNOWS_THRESHOLD
    labels = np.full(len(filt_te), "forgotten", dtype=object)
    labels[mk_i & mk_e] = "retained"
    labels[mk_i & ~mk_e] = "suppressed"
    labels[~mk_i & mk_e] = "lucky"
    return labels, mk_int, k_int_source

if __name__ == "__main__":
    _selfcheck_iclr_style()
