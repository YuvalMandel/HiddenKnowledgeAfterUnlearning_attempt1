
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

LABEL_CANDIDATES = ['best_method_label', 'best_method', 'label']
GROUP_CANDIDATES = ['question_group_id']
REGIME_CANDIDATES = ['train_regime']
NON_FEATURE_COLS = set(['question_group_id', 'train_regime', 'official_split', 'best_method_name'])


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def find_dataset_files(datasets_dir: Path):
    return sorted([p for p in datasets_dir.glob('phase22_dataset_*.csv')])


def detect_label_col(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f'No label column found. Available columns: {list(df.columns)}')


def detect_feature_cols(df: pd.DataFrame, label_col: str):
    feats = []
    for c in df.columns:
        if c == label_col or c in NON_FEATURE_COLS or c.startswith('best_method'):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            feats.append(c)
    if not feats:
        raise ValueError('No numeric feature columns detected.')
    return feats


def detect_regime(df: pd.DataFrame, path: Path) -> str:
    if 'train_regime' in df.columns and df['train_regime'].nunique() == 1:
        return str(df['train_regime'].iloc[0])
    return path.stem.replace('phase22_dataset_', '')


def build_model(kind: str, feature_cols):
    if kind == 'logreg':
        clf = LogisticRegression(
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced'
        )
    elif kind == 'rf':
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=0,
            class_weight='balanced_subsample',
            n_jobs=-1,
        )
    elif kind == 'gb':
        clf = GradientBoostingClassifier(random_state=0)
    else:
        raise ValueError(f'Unsupported classifier: {kind}')

    pre = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]), feature_cols)
        ],
        remainder='drop'
    )
    return Pipeline([('pre', pre), ('clf', clf)])


def run_one_dataset(ds_path: Path, classifier: str, out_root: Path):
    df = pd.read_csv(ds_path)
    label_col = detect_label_col(df)
    feature_cols = detect_feature_cols(df, label_col)
    regime = detect_regime(df, ds_path)

    X = df[feature_cols].copy()
    y = df[label_col].astype(int).copy()
    groups = df['question_group_id'].astype(str).copy()

    model = build_model(classifier, feature_cols)
    gkf = GroupKFold(n_splits=5)

    reports_dir = out_root / 'reports' / classifier
    scored_dir = out_root / 'scored' / classifier
    models_dir = out_root / 'models' / classifier
    for d in [reports_dir, scored_dir, models_dir]:
        ensure_dir(d)

    fold_rows = []
    pred_rows = []

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), start=1):
        Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        gte = groups.iloc[te_idx]

        model.fit(Xtr, ytr)
        yhat = model.predict(Xte)

        acc = accuracy_score(yte, yhat)
        bacc = balanced_accuracy_score(yte, yhat)
        mf1 = f1_score(yte, yhat, average='macro', zero_division=0)

        top2 = np.nan
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Xte)
            try:
                top2 = top_k_accuracy_score(yte, proba, k=2, labels=model.classes_)
            except Exception:
                top2 = np.nan

        fold_rows.append({
            'classifier': classifier,
            'regime': regime,
            'fold': fold_idx,
            'n_train_rows': int(len(tr_idx)),
            'n_test_rows': int(len(te_idx)),
            'n_train_groups': int(groups.iloc[tr_idx].nunique()),
            'n_test_groups': int(gte.nunique()),
            'accuracy': float(acc),
            'balanced_accuracy': float(bacc),
            'macro_f1': float(mf1),
            'top2_accuracy': None if pd.isna(top2) else float(top2),
        })

        pred_rows.append(pd.DataFrame({
            'classifier': classifier,
            'regime': regime,
            'fold': fold_idx,
            'question_group_id': gte.values,
            'y_true': yte.values,
            'y_pred': yhat,
        }))

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)

    fold_df.to_csv(reports_dir / f'fold_results_{regime}.csv', index=False)
    pred_df.to_csv(scored_dir / f'predictions_{regime}.csv', index=False)

    summary = pd.DataFrame([{
        'classifier': classifier,
        'regime': regime,
        'mean_accuracy': fold_df['accuracy'].mean(),
        'std_accuracy': fold_df['accuracy'].std(ddof=0),
        'mean_balanced_accuracy': fold_df['balanced_accuracy'].mean(),
        'mean_macro_f1': fold_df['macro_f1'].mean(),
        'mean_top2_accuracy': fold_df['top2_accuracy'].mean(skipna=True),
        'n_rows': int(len(df)),
        'n_groups': int(groups.nunique()),
        'n_classes': int(y.nunique()),
        'n_features': int(len(feature_cols)),
    }])
    summary.to_csv(reports_dir / f'regime_summary_{regime}.csv', index=False)
    dump_json(models_dir / f'metadata_{regime}.json', {
        'classifier': classifier,
        'regime': regime,
        'dataset': str(ds_path),
        'feature_cols': feature_cols,
        'label_col': label_col,
        'n_rows': int(len(df)),
        'n_groups': int(groups.nunique()),
        'n_classes': int(y.nunique()),
    })
    return summary, fold_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classifier', required=True, choices=['logreg', 'rf', 'gb'])
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    out_root = Path(args.out_dir)
    files = find_dataset_files(out_root / 'datasets')
    if not files:
        raise FileNotFoundError(f'No phase22 dataset files found under {out_root / "datasets"}')

    print(f'[train_eval] classifier={args.classifier}', flush=True)
    print(f'[train_eval] found {len(files)} regime datasets', flush=True)

    all_summaries = []
    all_folds = []
    for ds_path in files:
        print(f'[train_eval] using dataset: {ds_path}', flush=True)
        s, f = run_one_dataset(ds_path, args.classifier, out_root)
        all_summaries.append(s)
        all_folds.append(f)

    reports_dir = out_root / 'reports' / args.classifier
    ensure_dir(reports_dir)
    overall_summary = pd.concat(all_summaries, ignore_index=True)
    overall_summary.to_csv(reports_dir / 'overall_summary_by_regime.csv', index=False)
    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(reports_dir / 'fold_results_all_regimes.csv', index=False)
    overall = pd.DataFrame([{
        'classifier': args.classifier,
        'mean_accuracy': overall_summary['mean_accuracy'].mean(),
        'std_accuracy': overall_summary['mean_accuracy'].std(ddof=0),
        'mean_balanced_accuracy': overall_summary['mean_balanced_accuracy'].mean(),
        'mean_macro_f1': overall_summary['mean_macro_f1'].mean(),
        'mean_top2_accuracy': overall_summary['mean_top2_accuracy'].mean(skipna=True),
        'n_regimes': int(len(overall_summary)),
    }])
    overall.to_csv(reports_dir / 'overall_summary.csv', index=False)
    print(f'[train_eval] completed classifier={args.classifier}', flush=True)


if __name__ == '__main__':
    main()
