
import argparse
from pathlib import Path
from phase22_lib import dump_json, ensure_dir, build_scientific_labels, resolve_default_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--measurements_csv', default=None)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    defaults = resolve_default_paths(script_dir)
    measurements_csv = Path(args.measurements_csv) if args.measurements_csv else defaults['measurements_csv']
    out_dir = Path(args.out_dir)
    datasets_dir = ensure_dir(out_dir / 'datasets')

    print(f'[labels] reading measurements from {measurements_csv}', flush=True)
    scores, labels = build_scientific_labels(measurements_csv)

    scores_path = datasets_dir / 'scientific_scores_by_question_method_regime.csv'
    labels_path = datasets_dir / 'best_method_labels_by_question_regime.csv'
    scores.to_csv(scores_path, index=False)
    labels.to_csv(labels_path, index=False)

    dump_json(datasets_dir / 'labels_metadata.json', {
        'measurements_csv': str(measurements_csv),
        'n_scores_rows': int(len(scores)),
        'n_labels_rows': int(len(labels)),
        'regimes': sorted(labels['train_regime'].astype(str).unique().tolist()),
    })
    print(f'[labels] wrote {scores_path}', flush=True)
    print(f'[labels] wrote {labels_path}', flush=True)
    print('[labels] completed', flush=True)


if __name__ == '__main__':
    main()
