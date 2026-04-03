from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_pipeline.common.paths import project_path
from dl_pipeline.common.registry import append_registry_row
from dl_pipeline.inference.submission import build_submission_dataframe
from dl_pipeline.inference.submission import save_submission_dataframe


def _load_submission(path_value: str | Path) -> pd.DataFrame:
    path = Path(path_value)
    if not path.is_absolute():
        path = project_path(path)
    frame = pd.read_csv(path)
    required_columns = {"id", "class"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"submission 缺少必须列 {required_columns}: {path}")
    return frame.sort_values("id", kind="mergesort").reset_index(drop=True)


def _ensure_same_ids(frames: list[pd.DataFrame]) -> None:
    if not frames:
        raise ValueError("至少需要一个 submission。")
    base_ids = frames[0]["id"].tolist()
    for index, frame in enumerate(frames[1:], start=1):
        if frame["id"].tolist() != base_ids:
            raise ValueError(f"第 {index + 1} 个 submission 的 id 顺序与第一个不一致。")


def _majority_vote(row: pd.Series, columns: list[str], primary_column: str) -> int:
    votes = [int(row[column]) for column in columns]
    counts = Counter(votes)
    top_count = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == top_count)
    if len(winners) == 1:
        return winners[0]
    primary_vote = int(row[primary_column])
    if primary_vote in winners:
        return primary_vote
    return winners[0]


def _conservative_vote(row: pd.Series, columns: list[str], primary_column: str, other_label: int) -> int:
    votes = [int(row[column]) for column in columns]
    target_votes = [vote for vote in votes if vote != other_label]
    if len(target_votes) < 2:
        return other_label

    counts = Counter(target_votes)
    top_count = max(counts.values())
    winners = sorted(label for label, count in counts.items() if count == top_count)
    if len(winners) == 1:
        return winners[0]

    primary_vote = int(row[primary_column])
    if primary_vote in winners:
        return primary_vote
    return other_label


def _diverse_vote(
    row: pd.Series,
    primary_column: str,
    secondary_column: str,
    tertiary_column: str,
    other_label: int,
) -> int:
    primary_vote = int(row[primary_column])
    secondary_vote = int(row[secondary_column])
    tertiary_vote = int(row[tertiary_column])

    if secondary_vote == tertiary_vote:
        return secondary_vote
    if primary_vote == secondary_vote:
        return primary_vote
    if primary_vote == tertiary_vote:
        return primary_vote
    if primary_vote != other_label:
        return primary_vote
    if secondary_vote != other_label and tertiary_vote == other_label:
        return secondary_vote
    if tertiary_vote != other_label and secondary_vote == other_label:
        return tertiary_vote
    return other_label


def _prediction_distribution(frame: pd.DataFrame) -> dict[str, int]:
    counts = Counter(frame["class"].astype(int).tolist())
    return {str(label): int(count) for label, count in sorted(counts.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description="对多个 submission 做可复现的标签级投票融合。")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--primary-name", default="exp_061")
    parser.add_argument("--secondary-name", default="exp_067")
    parser.add_argument("--tertiary-name", default="exp_068")
    parser.add_argument("--primary-submission", required=True)
    parser.add_argument("--secondary-submission", required=True)
    parser.add_argument("--tertiary-submission", required=True)
    parser.add_argument("--other-label", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["majority_vote_3way", "conservative_vote", "diverse_ensemble"],
        required=True,
    )
    args = parser.parse_args()

    primary = _load_submission(args.primary_submission)
    secondary = _load_submission(args.secondary_submission)
    tertiary = _load_submission(args.tertiary_submission)
    _ensure_same_ids([primary, secondary, tertiary])

    merged = primary.rename(columns={"class": args.primary_name}).merge(
        secondary.rename(columns={"class": args.secondary_name}),
        on="id",
    ).merge(
        tertiary.rename(columns={"class": args.tertiary_name}),
        on="id",
    )

    vote_columns = [args.primary_name, args.secondary_name, args.tertiary_name]
    if args.mode == "majority_vote_3way":
        merged["class"] = merged.apply(
            _majority_vote,
            axis=1,
            columns=vote_columns,
            primary_column=args.primary_name,
        )
    elif args.mode == "conservative_vote":
        merged["class"] = merged.apply(
            _conservative_vote,
            axis=1,
            columns=vote_columns,
            primary_column=args.primary_name,
            other_label=args.other_label,
        )
    else:
        merged["class"] = merged.apply(
            _diverse_vote,
            axis=1,
            primary_column=args.primary_name,
            secondary_column=args.secondary_name,
            tertiary_column=args.tertiary_name,
            other_label=args.other_label,
        )

    ordered_predictions = merged["class"].astype(int).tolist()
    submission = build_submission_dataframe(merged[["id"]].copy(), ordered_predictions)

    output_root = project_path("outputs", args.exp_name)
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = project_path(
        "data",
        "submissions",
        f"{timestamp}_{args.exp_name}_submission.csv",
    )
    save_submission_dataframe(submission, submission_path)

    metrics = {
        "mode": args.mode,
        "primary_name": args.primary_name,
        "secondary_name": args.secondary_name,
        "tertiary_name": args.tertiary_name,
        "primary_submission": str(Path(args.primary_submission)),
        "secondary_submission": str(Path(args.secondary_submission)),
        "tertiary_submission": str(Path(args.tertiary_submission)),
        "prediction_distribution": _prediction_distribution(submission.reset_index()),
        "diff_vs_primary": int((submission["class"].reset_index(drop=True) != primary["class"]).sum()),
        "diff_vs_secondary": int((submission["class"].reset_index(drop=True) != secondary["class"]).sum()),
        "diff_vs_tertiary": int((submission["class"].reset_index(drop=True) != tertiary["class"]).sum()),
    }
    (output_root / "prototype_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    append_registry_row(
        project_path("reports", "experiments", "registry.csv"),
        {
            "experiment_name": args.exp_name,
            "stage": "predict",
            "submission_path": str(submission_path),
            "checkpoint_path": json.dumps(
                {
                    "primary_submission": args.primary_submission,
                    "secondary_submission": args.secondary_submission,
                    "tertiary_submission": args.tertiary_submission,
                },
                ensure_ascii=False,
            ),
            "config_path": f"scripts/predict_submission_vote.py::{args.mode}",
        },
    )

    print(f"submission 已生成: {submission_path}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
