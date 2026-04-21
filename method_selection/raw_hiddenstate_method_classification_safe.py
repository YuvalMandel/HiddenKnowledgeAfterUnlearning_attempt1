#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Could not load ndarray from: {path}")
    return arr

def load_split_data(data_dir: Path, prefix: str) -> Dict[str, np.ndarray]:
    return {"train": load_npy(data_dir / f"{prefix}_train.npy"),
            "val": load_npy(data_dir / f"{prefix}_val.npy"),
            "test": load_npy(data_dir / f"{prefix}_test.npy")}

def parse_layer_spec(spec: str | None, num_layers: int, band_name: str) -> List[int]:
    if spec is None or str(spec).strip() == "":
        one_third = num_layers // 3
        two_third = 2 * one_third
        if band_name == "early": return list(range(0, one_third))
        if band_name == "mid": return list(range(one_third, two_third))
        if band_name == "late": return list(range(two_third, num_layers))
        if band_name == "all": return list(range(0, num_layers))
        raise ValueError(f"Unknown band_name: {band_name}")
    layers = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part: continue
        if ":" in part:
            a, b = part.split(":", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    uniq = sorted(set(layers))
    bad = [x for x in uniq if x < 0 or x >= num_layers]
    if bad: raise ValueError(f"Layer indices out of bounds: {bad}")
    return uniq

def load_labels_df(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    if "split" not in df.columns:
        raise ValueError("labels CSV must contain a 'split' column")
    return df

def get_test_indices_from_labels(labels_csv: Path, eval_split: str, expected_n: int) -> Optional[np.ndarray]:
    s = eval_split.lower()
    if s in {"val", "test"}: return None
    if s.startswith("test") and s[4:].isdigit():
        df = load_labels_df(labels_csv).copy()
        split_str = df["split"].astype(str).str.lower()
        test_df = df.loc[split_str.eq("test") | split_str.str.match(r"^test\d+$", na=False)].copy().reset_index(drop=True)
        idx = np.where(test_df["split"].astype(str).str.lower().to_numpy() == s)[0]
        if len(idx) == 0: raise ValueError(f"No rows found for eval_split={eval_split}")
        return idx.astype(int)
    if s.startswith("testw"):
        k = int(s.replace("testw", ""))
        if not (1 <= k <= 7): raise ValueError("test window must be testw1..testw7")
        start = (k - 1) * 100
        size = 520
        end = start + size
        if end > expected_n: raise ValueError(f"Window {eval_split} exceeds expected_n={expected_n}")
        return np.arange(start, end, dtype=int)
    raise ValueError(f"Unsupported eval_split: {eval_split}")

def extract_question_features(hs: np.ndarray, layers: List[int]) -> np.ndarray:
    x = hs[:, layers, :]
    return x.reshape(x.shape[0], -1)

def downsample_rows(X: np.ndarray, y: np.ndarray, row_ids: np.ndarray, methods: np.ndarray,
                    max_rows_per_method: int, seed: int):
    if max_rows_per_method <= 0:
        return X, y, row_ids, methods
    rng = np.random.default_rng(seed)
    keep_idx = []
    for m in np.unique(y):
        idx = np.where(y == m)[0]
        if len(idx) <= max_rows_per_method:
            keep_idx.extend(idx.tolist())
        else:
            keep_idx.extend(rng.choice(idx, size=max_rows_per_method, replace=False).tolist())
    keep_idx = np.array(sorted(keep_idx), dtype=int)
    return X[keep_idx], y[keep_idx], row_ids[keep_idx], methods[keep_idx]

def maybe_apply_pca(X_train: np.ndarray, X_test: np.ndarray, pca_components: int):
    if pca_components <= 0:
        return X_train, X_test, None
    n_comp = min(pca_components, X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0)
    return pca.fit_transform(X_train), pca.transform(X_test), pca

def build_matrix_for_split(data_dir: Path, labels_csv: Path, methods: List[str], method_suffix: str,
                           split_name: str, layers: List[int]):
    Xs, ys, row_ids, method_names = [], [], [], []
    for method in methods:
        hs = load_split_data(data_dir, f"{method}{method_suffix}")
        if split_name == "val":
            arr = hs["val"]; idx = None
        else:
            arr = hs["test"]; idx = get_test_indices_from_labels(labels_csv, split_name, expected_n=arr.shape[0])
        if idx is not None:
            arr = arr[idx]; qids = idx
        else:
            qids = np.arange(arr.shape[0], dtype=int)
        X = extract_question_features(arr, layers)
        Xs.append(X)
        ys.append(np.array([method] * X.shape[0], dtype=object))
        row_ids.append(qids)
        method_names.append(np.array([method] * X.shape[0], dtype=object))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(row_ids), np.concatenate(method_names)

def make_models(seed: int, model_names: List[str]):
    all_models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed)),
        "nn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    }
    bad = [m for m in model_names if m not in all_models]
    if bad: raise ValueError(f"Unsupported model(s): {bad}. Choose from: {list(all_models.keys())}")
    return {k: all_models[k] for k in model_names}

def evaluate_models(X_train, y_train, X_test, y_test, meta_eval_split, meta_method, meta_row_id, seed, model_names):
    results, preds = [], []
    for model_name, model in make_models(seed, model_names).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append({"model": model_name, "n_train_rows": int(X_train.shape[0]), "n_test_rows": int(X_test.shape[0]),
                        "n_features": int(X_train.shape[1]), "accuracy": float(accuracy_score(y_test, y_pred))})
        for i in range(len(y_test)):
            preds.append({"model": model_name, "eval_split": str(meta_eval_split[i]), "method": str(meta_method[i]),
                          "question_row_id": int(meta_row_id[i]), "y_true": str(y_test[i]), "y_pred": str(y_pred[i]),
                          "correct": int(y_test[i] == y_pred[i])})
    return pd.DataFrame(results), pd.DataFrame(preds)

def ci95_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0: return (float("nan"), float("nan"), float("nan"))
    m = float(np.mean(x))
    if len(x) == 1: return (m, m, m)
    s = float(np.std(x, ddof=1)); se = s / math.sqrt(len(x)); half = 1.96 * se
    return (m, m - half, m + half)

def parse_args():
    p = argparse.ArgumentParser(description="Safe raw hidden-state method classification baseline.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--method_suffix", default="_hs")
    p.add_argument("--band", default="late", choices=["early","mid","late","all"])
    p.add_argument("--early_layers", default="")
    p.add_argument("--mid_layers", default="")
    p.add_argument("--late_layers", default="")
    p.add_argument("--all_layers", default="")
    p.add_argument("--train_split", default="val", choices=["val"])
    p.add_argument("--test_splits", default="test,test1,test2,test3,test4,test5,testw1,testw2,testw3,testw4,testw5,testw6,testw7")
    p.add_argument("--max_rows_per_method", type=int, default=100)
    p.add_argument("--pca_components", type=int, default=0)
    p.add_argument("--models", default="rf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="out_raw_hiddenstate_method_classification_safe")
    return p.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir); labels_csv = Path(args.labels_csv); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    test_splits = [s.strip() for s in args.test_splits.split(",") if s.strip()]
    hs0 = load_split_data(data_dir, f"{methods[0]}{args.method_suffix}")["val"]
    num_layers = hs0.shape[1]
    if args.band == "early": layers = parse_layer_spec(args.early_layers, num_layers, "early")
    elif args.band == "mid": layers = parse_layer_spec(args.mid_layers, num_layers, "mid")
    elif args.band == "late": layers = parse_layer_spec(args.late_layers, num_layers, "late")
    else: layers = parse_layer_spec(args.all_layers, num_layers, "all")
    X_train, y_train, row_train, meth_train = build_matrix_for_split(data_dir, labels_csv, methods, args.method_suffix, args.train_split, layers)
    X_train, y_train, row_train, meth_train = downsample_rows(X_train, y_train, row_train, meth_train, args.max_rows_per_method, args.seed)
    X_test_parts=[]; y_test_parts=[]; row_test_parts=[]; meth_test_parts=[]; split_meta_parts=[]
    for split in test_splits:
        Xs, ys, rids, mnames = build_matrix_for_split(data_dir, labels_csv, methods, args.method_suffix, split, layers)
        X_test_parts.append(Xs); y_test_parts.append(ys); row_test_parts.append(rids); meth_test_parts.append(mnames)
        split_meta_parts.append(np.array([split]*len(ys), dtype=object))
    X_test = np.vstack(X_test_parts); y_test = np.concatenate(y_test_parts); row_test = np.concatenate(row_test_parts)
    meth_test = np.concatenate(meth_test_parts); split_meta = np.concatenate(split_meta_parts)
    X_train_use, X_test_use, pca_obj = maybe_apply_pca(X_train, X_test, args.pca_components)
    overall_results, overall_preds = evaluate_models(X_train_use, y_train, X_test_use, y_test, split_meta, meth_test, row_test, args.seed, model_names)
    overall_results.to_csv(out_dir / "method_classification_results_overall.csv", index=False)
    overall_preds.to_csv(out_dir / "method_classification_predictions_overall.csv", index=False)
    split_results=[]; split_preds=[]; start=0
    for split, Xs, ys, rids, mnames in zip(test_splits, X_test_parts, y_test_parts, row_test_parts, meth_test_parts):
        n=len(ys); Xs_use = X_test_use[start:start+n]; start += n
        meta_split = np.array([split]*n, dtype=object)
        r,p = evaluate_models(X_train_use, y_train, Xs_use, ys, meta_split, mnames, rids, args.seed, model_names)
        r["test_split"] = split; split_results.append(r); p["test_split"] = split; split_preds.append(p)
    by_split = pd.concat(split_results, ignore_index=True)
    by_split.to_csv(out_dir / "method_classification_results_by_split.csv", index=False)
    pd.concat(split_preds, ignore_index=True).to_csv(out_dir / "method_classification_predictions_by_split.csv", index=False)
    summary_rows=[]
    for model, sub in by_split.groupby("model"):
        arr = sub["accuracy"].to_numpy(dtype=float); mean, lo, hi = ci95_mean(arr)
        std = float(np.std(arr, ddof=1)) if len(arr)>1 else 0.0
        summary_rows.append({"model": model, "band": args.band, "n_splits": len(sub), "accuracy_mean": mean,
                             "accuracy_std": std, "accuracy_ci95_low": lo, "accuracy_ci95_high": hi,
                             "best_split": sub.sort_values("accuracy", ascending=False).iloc[0]["test_split"],
                             "best_accuracy": float(sub["accuracy"].max()),
                             "worst_split": sub.sort_values("accuracy", ascending=True).iloc[0]["test_split"],
                             "worst_accuracy": float(sub["accuracy"].min()),
                             "n_features": int(sub.iloc[0]["n_features"]), "n_train_rows": int(sub.iloc[0]["n_train_rows"])})
    pd.DataFrame(summary_rows).sort_values("accuracy_mean", ascending=False).to_csv(out_dir / "method_classification_summary.csv", index=False)
    meta = {"data_dir": str(data_dir), "labels_csv": str(labels_csv), "methods": methods, "method_suffix": args.method_suffix,
            "band": args.band, "layers": layers, "train_split": args.train_split, "test_splits": test_splits,
            "max_rows_per_method": args.max_rows_per_method, "pca_components": args.pca_components,
            "models": model_names, "seed": args.seed, "n_train_rows": int(X_train_use.shape[0]),
            "n_test_rows": int(X_test_use.shape[0]), "n_features_after_prep": int(X_train_use.shape[1]),
            "pca_applied": bool(pca_obj is not None)}
    with open(out_dir / "run_metadata.json","w",encoding="utf-8") as f: json.dump(meta, f, indent=2)
    pd.DataFrame([{"band": args.band, "n_layers_selected": len(layers), "hidden_size": int(hs0.shape[2]),
                   "raw_feature_dim": int(len(layers)*hs0.shape[2]), "train_rows_after_cap": int(X_train.shape[0]),
                   "n_train_rows_final": int(X_train_use.shape[0]), "n_test_rows_final": int(X_test_use.shape[0]),
                   "n_features_final": int(X_train_use.shape[1])}]).to_csv(out_dir / "feature_dim_report.csv", index=False)
    print(f"Saved summary to: {out_dir / 'method_classification_summary.csv'}")

if __name__ == "__main__":
    main()
