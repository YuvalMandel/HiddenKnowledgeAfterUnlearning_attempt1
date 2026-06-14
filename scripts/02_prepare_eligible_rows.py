from __future__ import annotations

import argparse
from pathlib import Path

from threat_model_analysis_v2_lib import DEFAULT_LABELS_INPUT, DEFAULT_PAIR_DATASET, DEFAULT_TOPIC_INPUT, V2Config, prepare_eligible_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged-input", default=str(DEFAULT_PAIR_DATASET))
    parser.add_argument("--labels-input", default=str(DEFAULT_LABELS_INPUT))
    parser.add_argument("--topic-input", default=str(DEFAULT_TOPIC_INPUT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--cv-repeats", type=int, default=20)
    parser.add_argument("--min-cell-n", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = V2Config(Path(args.merged_input), Path(args.labels_input), Path(args.topic_input), Path(args.output_dir), args.bootstrap_repetitions, args.cv_repeats, args.min_cell_n, args.seed)
    eligible = prepare_eligible_outputs(config)
    print(config.output_dir / "data" / "eligible_question_method_rows.csv")
    print(f"rows={len(eligible)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
