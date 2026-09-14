#!/usr/bin/env python3
"""K_int per probe configuration, to replace the accuracy figures in app:probes.

app:probes and app:cross-probe label their numbers "$K_\\text{int}$ accuracy".
They are not K_int. They are the probe's binary classification accuracy on the
balanced True/False items (54.9--58.2% for PL-LR), whereas K_int is the pairwise
knowledge score of eq:kscore (0.676--0.756 at ck8, section 5.1). The two are
different quantities and, as the printout shows, they do not even rank the eight
methods the same way.

What this script can rebuild, from inside_out_out/<model>/k_scores*.parquet:

    PL-LR  (best layer)      -> layer_config="best_layer"   AVAILABLE
    all-layer PCA-256        -> layer_config="full"         AVAILABLE
    every single layer       -> layer_config="layer_<l>"    AVAILABLE

What it cannot, because only LR/own probes were scored per question:

    ML-LR at layers 12--22   -> "multi" was never written to the parquets
    PL/ML-RF, PL/ML-Ada      -> no per-question scores for RF or AdaBoost
    app:cross-probe          -> probe_type is "own" everywhere

Those need the hidden states, which are not in this checkout, so they need a
cluster run to convert.

Usage: python plots/probe_k_table.py
"""
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "inside_out_out"
AV = ROOT / "plots" / "activation_vectors"
METHODS = ["GradDiff", "RMU", "RMU-LAT", "RepNoise", "ELM", "RR", "TAR", "PB_J"]

# what app:probes currently prints, for the comparison below
PUBLISHED_PL = dict(base=64.4, GradDiff=57.4, RMU=57.4, **{"RMU-LAT": 56.6},
                    RepNoise=57.8, ELM=54.9, RR=58.2, TAR=58.0, PB_J=55.3)


def k_int(model_id, cfg):
    f = OUT / model_id / "k_scores.parquet"
    d = pd.read_parquet(f)
    cv = d[(d.split_type == "cv") & (d.domain == "bio") & (d.clf == "LR")
           & (d.probe_type == "own") & (d.layer_config == cfg)]
    if cv.empty:
        return None
    return cv.groupby("question_idx").k_internal.mean().mean()


def main():
    rows = []
    for name, mid in [("base", "base")] + [(m, f"{m}_ck8") for m in METHODS]:
        rows.append(dict(model=name,
                         k_pl=k_int(mid, "best_layer"),
                         k_all=k_int(mid, "full"),
                         published_acc_pl=PUBLISHED_PL[name]))
    df = pd.DataFrame(rows)
    df.to_csv(AV / "probe_k_table.csv", index=False)

    print(f"{'model':<12}{'K_int PL':>10}{'K_int all':>11}{'| app:probes acc':>18}")
    print("-" * 51)
    for _, r in df.iterrows():
        print(f"{r.model:<12}{r.k_pl:>10.3f}{r.k_all:>11.3f}"
              f"{r.published_acc_pl:>16.1f}%")

    u = df[df.model != "base"]
    rho, p = spearmanr(u.k_pl, u.published_acc_pl)
    print(f"\nSpearman(K_int PL, published accuracy) over the eight methods: "
          f"rho={rho:+.2f}, p={p:.2f}")
    print("The accuracy table does not rank methods as K_int does, so relabelling"
          "\nalone would leave a table that says something different from 5.1.")
    print(f"\nwrote {AV / 'probe_k_table.csv'}")


if __name__ == "__main__":
    main()
