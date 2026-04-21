
import argparse
from pathlib import Path
import pandas as pd
from phase22_lib import (
    build_pair_and_logit_features,
    ensure_dir,
    dump_json,
    feature_columns_for_mode,
    resolve_default_paths,
    LABEL_CANDIDATES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs_csv', default=None)
    ap.add_argument('--logits_dir', default=None)
    ap.add_argument('--labels_csv', default=None)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--feature_mode', required=True, choices=['2.2', '2.4', 'joint', 'pair', 'logits'])
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    defaults = resolve_default_paths(script_dir)
    pairs_csv = Path(args.pairs_csv) if args.pairs_csv else defaults['pairs_csv']
    logits_dir = Path(args.logits_dir) if args.logits_dir else defaults['data_dir']
    out_dir = Path(args.out_dir)
    datasets_dir = ensure_dir(out_dir / 'datasets')

    labels_csv = Path(args.labels_csv) if args.labels_csv else (datasets_dir / 'best_method_labels_by_question_regime.csv')
    if not labels_csv.exists():
        raise FileNotFoundError(f'Missing labels file: {labels_csv}')

    print(f'[features] building features mode={args.feature_mode} from {pairs_csv} and {logits_dir}', flush=True)
    feat_df = build_pair_and_logit_features(pairs_csv, logits_dir)
    feat_df.to_csv(datasets_dir / 'pair_and_logit_features.csv', index=False)
    dump_json(datasets_dir / 'pair_and_logit_debug.json', getattr(feat_df, 'attrs', {}).get('debug', {}))

    labels = pd.read_csv(labels_csv)
    label_col = next((c for c in LABEL_CANDIDATES if c in labels.columns), None)
    if not label_col:
        raise KeyError(f'No label column found in {labels_csv}. Columns: {list(labels.columns)}')
    if 'train_regime' not in labels.columns:
        raise KeyError(f'Expected train_regime in {labels_csv}. Columns: {list(labels.columns)}')
    if 'question_group_id' not in labels.columns:
        raise KeyError(f'Expected question_group_id in {labels_csv}. Columns: {list(labels.columns)}')

    keep_cols = ['question_group_id', 'train_regime', label_col]
    extra_name_cols = [c for c in labels.columns if c.startswith('best_method') and c != label_col]
    keep_cols.extend(extra_name_cols)
    labels = labels[keep_cols].copy()
    labels['question_group_id'] = labels['question_group_id'].astype(str)

    feature_cols = feature_columns_for_mode(args.feature_mode)
    base = feat_df[['question_group_id', 'official_split'] + feature_cols].copy()
    base['question_group_id'] = base['question_group_id'].astype(str)

    regimes = sorted(labels['train_regime'].astype(str).unique().tolist())
    written = []
    for regime in regimes:
        sub = labels[labels['train_regime'].astype(str) == regime].copy()
        merged = base.merge(sub, on='question_group_id', how='inner')
        out_path = datasets_dir / f'phase22_dataset_{regime}.csv'
        merged.to_csv(out_path, index=False)
        written.append(out_path.name)

    dump_json(datasets_dir / 'feature_mode_metadata.json', {
        'feature_mode': args.feature_mode,
        'feature_columns': feature_cols,
        'labels_csv': str(labels_csv),
        'pairs_csv': str(pairs_csv),
        'logits_dir': str(logits_dir),
        'regimes': regimes,
        'written_files': written,
    })
    print(f'[features] wrote {len(written)} regime datasets', flush=True)


if __name__ == '__main__':
    main()
