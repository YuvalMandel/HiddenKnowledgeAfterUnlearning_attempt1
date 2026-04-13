#!/usr/bin/env python3
"""
quadrant_counts.py

For each model × probe_type × classifier, counts how many bio (WMDP) test QA
pairs fall into each of the 4 knowledge quadrants:

  Remains:     base probe correct  AND  method probe correct
  Unlearned:   base probe correct  AND  method probe incorrect
  Emerged:     base probe incorrect AND  method probe correct
  Never Known: base probe incorrect AND  method probe incorrect

Two "method knows" variants:
  frozen_base  — base probe weights frozen, applied to method hidden states
                 (tests whether method HS still decodable by base probe)
  method_probe — method's own probe applied to method hidden states
                 (each model judged by its own best probe)

Probe types covered (all × LR, RF, AdaBoost):
  per_layer     — best validation layer per classifier
  mid_band      — layers 10-22 concat → PCA-256
  full_layer    — all layers concat → PCA-256
  init_band     — layers 1-9 concat → PCA-256
  init_band_emb — layers 0-9 concat → PCA-256  (includes embedding)
  end_band      — layers 23-32 concat → PCA-256
  ib_no_pca     — layers 1-6 concat, no PCA

Also covers sweep checkpoints (ck1-ck8) if the checkpoint subdirectories exist.

NO LLM inference required: all hidden states and probes are loaded from
pre-computed checkpoint files produced by hidden_knowledge_after_unlearning.py.

Usage:
  python quadrant_counts.py
  python quadrant_counts.py --checkpoint_dir checkpoints --out quadrant_counts.csv
  python quadrant_counts.py --no_sweep   # skip per-training-step checkpoints
"""

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

# ── Constants (must match hidden_knowledge_after_unlearning.py) ───────────────

METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB&J"]
CLF_NAMES = ["LR", "RF", "AdaBoost"]
N_SWEEP_CHECKPOINTS = 8

# All multi-layer band keys present in a probe_set dict
BAND_KEYS = [
    "per_layer",       # uses best_layers[clf] to pick the single best layer
    "mid_band",        # layers 10-22
    "full_layer",      # all layers
    "init_band",       # layers 1-9
    "init_band_emb",   # layers 0-9 (includes embedding)
    "end_band",        # layers 23-32
    "ib_no_pca",       # layers 1-6, no PCA
]

DEFAULT_BAND_RANGES = {
    "mid_band":       (10, 22),
    "full_layer":     (0, 32),   # overridden from probe_set["full_layer_range"]
    "init_band":      (1, 9),
    "init_band_emb":  (0, 9),
    "end_band":       (23, 32),  # overridden from probe_set["end_band_range"]
    "ib_no_pca":      (1, 6),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(name: str) -> str:
    return (name.replace(" ", "_")
               .replace("/", "_")
               .replace("&", "_")
               .replace("-", "_"))


def _load_npy(path: Path):
    if path.exists():
        return np.load(str(path))
    return None


def _load_pkl(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def load_y_test(ck_dir: Path) -> np.ndarray:
    """
    Load gold labels for the test split.  Two strategies tried in order:

    1. data/wmdp_tf_pairs.csv  — authoritative, uses the 'label' and 'split'
       columns.  Path is resolved relative to the script's working directory.
    2. base_results.json / test_answers — falls back to the list of TF pair
       dicts stored inside the checkpoint (each has an 'expected' key).
    """
    csv_path = Path("data") / "wmdp_tf_pairs.csv"
    if csv_path.exists():
        labels = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == "test":
                    labels.append(1 if row["label"] == "True" else 0)
        if labels:
            print(f"  y_test loaded from {csv_path}  ({len(labels)} rows)", flush=True)
            return np.array(labels, dtype=np.int32)

    # Fallback: base_results.json
    json_path = ck_dir / "base_results.json"
    with open(json_path) as f:
        r = json.load(f)
    answers = r.get("test_answers", [])
    if not answers:
        raise ValueError(
            "Could not load y_test: data/wmdp_tf_pairs.csv not found and "
            "'test_answers' is empty in base_results.json."
        )
    labels = [1 if a.get("expected", a.get("label", "")) == "True" else 0
              for a in answers]
    print(f"  y_test loaded from base_results.json  ({len(labels)} rows)", flush=True)
    return np.array(labels, dtype=np.int32)


# ── Feature extraction ────────────────────────────────────────────────────────

def _band_range(probe_set: dict, band: str, n_layers: int) -> tuple[int, int]:
    """Return (start, end) inclusive layer indices for a multi-layer band."""
    range_key = f"{band}_range"
    if range_key in probe_set:
        return tuple(probe_set[range_key])
    # mid_band has a legacy alias
    if band == "mid_band" and "multi_layer_range" in probe_set:
        return tuple(probe_set["multi_layer_range"])
    lo, hi = DEFAULT_BAND_RANGES.get(band, (0, n_layers - 1))
    return lo, min(hi, n_layers - 1)


def get_predictions(probe_set: dict, hs: np.ndarray,
                    band: str, clf_name: str) -> np.ndarray | None:
    """
    Return per-sample integer predictions (0 or 1) for one band × classifier.
    Returns None if the probe or hidden states are unavailable.
    """
    n, n_layers, _ = hs.shape

    if band == "per_layer":
        best_layers = probe_set.get("best_layers", {})
        l = best_layers.get(clf_name)
        if l is None:
            return None
        pipe = probe_set.get("per_layer", {}).get(clf_name, {}).get(l)
        if pipe is None:
            return None
        return pipe.predict(hs[:, l, :]).astype(int)

    # All other bands: slice + reshape → predict
    band_dict = probe_set.get(band)
    if not band_dict or clf_name not in band_dict:
        return None

    lo, hi = _band_range(probe_set, band, n_layers)
    hi = min(hi, n_layers - 1)
    if lo > hi:
        return None

    X = hs[:, lo:hi + 1, :].reshape(n, -1)
    try:
        return band_dict[clf_name].predict(X).astype(int)
    except Exception as e:
        print(f"    [warn] predict failed ({band}/{clf_name}): {e}", flush=True)
        return None


# ── Quadrant counting ─────────────────────────────────────────────────────────

def count_quadrants(base_correct: np.ndarray,
                    method_correct: np.ndarray) -> dict:
    bc = base_correct.astype(bool)
    mc = method_correct.astype(bool)
    return {
        "remains":     int(( bc &  mc).sum()),
        "unlearned":   int(( bc & ~mc).sum()),
        "emerged":     int((~bc &  mc).sum()),
        "never_known": int((~bc & ~mc).sum()),
        "total":       len(bc),
    }


# ── Per-model processing ──────────────────────────────────────────────────────

def process_pair(
    model_name: str,
    checkpoint: str,
    base_probe_set: dict,
    base_hs: np.ndarray,
    method_probe_set,          # may be None
    method_hs: np.ndarray,
    y_test: np.ndarray,
) -> list[dict]:
    """
    Return one CSV row per (band, clf, probe_variant) for this model/checkpoint.
    """
    rows = []

    if len(method_hs) != len(y_test):
        print(f"  [warn] {model_name}/{checkpoint}: method_hs length "
              f"{len(method_hs)} != y_test length {len(y_test)} — skipping.",
              flush=True)
        return rows

    for band in BAND_KEYS:
        for clf_name in CLF_NAMES:

            # Base predictions: base probe on base HS (the same for every method)
            base_preds = get_predictions(base_probe_set, base_hs, band, clf_name)
            if base_preds is None:
                continue
            base_correct = base_preds == y_test

            # ── Variant A: frozen base probe on method HS ─────────────────
            frozen_preds = get_predictions(base_probe_set, method_hs, band, clf_name)
            if frozen_preds is not None:
                rows.append({
                    "model":         model_name,
                    "checkpoint":    checkpoint,
                    "probe_type":    band,
                    "classifier":    clf_name,
                    "probe_variant": "frozen_base",
                    **count_quadrants(base_correct, frozen_preds == y_test),
                })

            # ── Variant B: method's own probe on method HS ────────────────
            if method_probe_set is not None:
                method_preds = get_predictions(method_probe_set, method_hs,
                                               band, clf_name)
                if method_preds is not None:
                    rows.append({
                        "model":         model_name,
                        "checkpoint":    checkpoint,
                        "probe_type":    band,
                        "classifier":    clf_name,
                        "probe_variant": "method_probe",
                        **count_quadrants(base_correct, method_preds == y_test),
                    })

    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint_dir", default="checkpoints",
                    help="Checkpoint directory (default: checkpoints)")
    ap.add_argument("--out", default="quadrant_counts.csv",
                    help="Output CSV path (default: quadrant_counts.csv)")
    ap.add_argument("--no_sweep", action="store_true",
                    help="Skip per-training-step sweep checkpoints (ck1–ck8 subdirs)")
    args = ap.parse_args()

    ck_dir = Path(args.checkpoint_dir)

    # ── Load base checkpoint ──────────────────────────────────────────────────
    print("Loading base checkpoint ...", flush=True)
    for required in ("base_probes.pkl", "base_hs_test.npy", "base_results.json"):
        if not (ck_dir / required).exists():
            raise FileNotFoundError(
                f"{ck_dir / required} not found. "
                "Run --stage base of hidden_knowledge_after_unlearning.py first."
            )

    base_probe_set = _load_pkl(ck_dir / "base_probes.pkl")
    base_hs_test   = _load_npy(ck_dir / "base_hs_test.npy")
    y_test         = load_y_test(ck_dir)

    print(f"  base_hs_test : {base_hs_test.shape}  "
          f"(n_pairs={base_hs_test.shape[0]}, "
          f"n_layers={base_hs_test.shape[1]}, "
          f"hidden_dim={base_hs_test.shape[2]})",
          flush=True)
    print(f"  y_test       : {len(y_test)} labels  "
          f"({y_test.sum()} True / {(y_test == 0).sum()} False)",
          flush=True)

    all_rows: list[dict] = []

    # ── Final checkpoint (ck8) for each method ────────────────────────────────
    print("\n── Final checkpoints ─────────────────────────────────────────────",
          flush=True)
    for method_name in METHODS:
        sn = safe_name(method_name)
        hs_path    = ck_dir / f"{sn}_hs_test.npy"
        probe_path = ck_dir / f"{sn}_probes.pkl"

        if not hs_path.exists():
            print(f"  {method_name}: {hs_path.name} not found — skipping.",
                  flush=True)
            continue

        print(f"  {method_name} (ck8) ...", end="  ", flush=True)
        method_hs    = _load_npy(hs_path)
        method_probe = _load_pkl(probe_path)   # None if file absent

        if method_probe is not None and "per_layer" not in method_probe:
            print("[old probe format, using frozen_base only]", end="  ", flush=True)
            method_probe = None

        rows = process_pair(
            model_name=method_name,
            checkpoint="ck8",
            base_probe_set=base_probe_set,
            base_hs=base_hs_test,
            method_probe_set=method_probe,
            method_hs=method_hs,
            y_test=y_test,
        )
        all_rows.extend(rows)
        print(f"{len(rows)} rows", flush=True)

    # ── Sweep checkpoints (ck1–ck8 subdirs) ──────────────────────────────────
    if not args.no_sweep:
        print("\n── Sweep checkpoints ─────────────────────────────────────────────",
              flush=True)
        for method_name in METHODS:
            sn = safe_name(method_name)
            for ck_num in range(1, N_SWEEP_CHECKPOINTS + 1):
                ck_subdir = ck_dir / f"{sn}_ck{ck_num}"
                if not ck_subdir.exists():
                    continue

                hs_path    = ck_subdir / "hs_test.npy"
                probe_path = ck_subdir / "probes.pkl"

                if not hs_path.exists():
                    print(f"  {method_name}/ck{ck_num}: hs_test.npy not found — skipping.",
                          flush=True)
                    continue

                print(f"  {method_name} ck{ck_num} ...", end="  ", flush=True)
                method_hs    = _load_npy(hs_path)
                method_probe = _load_pkl(probe_path)

                if method_probe is not None and "per_layer" not in method_probe:
                    method_probe = None

                rows = process_pair(
                    model_name=method_name,
                    checkpoint=f"ck{ck_num}",
                    base_probe_set=base_probe_set,
                    base_hs=base_hs_test,
                    method_probe_set=method_probe,
                    method_hs=method_hs,
                    y_test=y_test,
                )
                all_rows.extend(rows)
                print(f"{len(rows)} rows", flush=True)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo data written — no checkpoint files found.", flush=True)
        return

    fieldnames = [
        "model", "checkpoint", "probe_type", "classifier", "probe_variant",
        "remains", "unlearned", "emerged", "never_known", "total",
    ]
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows → {out_path}", flush=True)

    # ── Quick sanity print ────────────────────────────────────────────────────
    print("\nSample (frozen_base, full_layer, LR):", flush=True)
    header = f"  {'model':<12} {'ck':<5} {'remains':>8} {'unlearned':>9} "
    header += f"{'emerged':>8} {'never_known':>12} {'total':>7}"
    print(header, flush=True)
    for row in all_rows:
        if (row["probe_type"] == "full_layer"
                and row["classifier"] == "LR"
                and row["probe_variant"] == "frozen_base"):
            print(f"  {row['model']:<12} {row['checkpoint']:<5} "
                  f"{row['remains']:>8} {row['unlearned']:>9} "
                  f"{row['emerged']:>8} {row['never_known']:>12} "
                  f"{row['total']:>7}",
                  flush=True)


if __name__ == "__main__":
    main()
