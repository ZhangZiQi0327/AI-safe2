from __future__ import annotations

import argparse
import csv
from pathlib import Path

from eval_transfer_attacks import evaluate_transfer, write_rows


MODEL_NAMES = ("resnet18", "resnet34", "vgg16", "densenet121")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search transfer-attack parameters for B-part tuning.")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--models-root", default="models_public")
    parser.add_argument("--attack-set", choices=("all", "mi_fgsm", "ensemble"), default="all")
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--records-output", default="b_search_records.csv")
    parser.add_argument("--summary-output", default="b_search_summary.csv")
    return parser.parse_args()


def checkpoint_map(models_root: str | Path) -> dict[str, str]:
    root = Path(models_root)
    return {name: str(root / f"{name}_best.pt") for name in MODEL_NAMES}


def mi_fgsm_search_configs() -> list[dict[str, object]]:
    return [
        {
            "config_id": "mi_e8_a2_s10_m1.0",
            "attack": "mi_fgsm",
            "epsilon": 8 / 255,
            "alpha": 2 / 255,
            "steps": 10,
            "momentum": 1.0,
            "diversity_prob": 0.0,
            "resize_rate": 0.9,
        },
        {
            "config_id": "mi_e8_a1_s12_m1.0",
            "attack": "mi_fgsm",
            "epsilon": 8 / 255,
            "alpha": 1 / 255,
            "steps": 12,
            "momentum": 1.0,
            "diversity_prob": 0.0,
            "resize_rate": 0.9,
        },
        {
            "config_id": "mi_e10_a2_s12_m1.0",
            "attack": "mi_fgsm",
            "epsilon": 10 / 255,
            "alpha": 2 / 255,
            "steps": 12,
            "momentum": 1.0,
            "diversity_prob": 0.0,
            "resize_rate": 0.9,
        },
        {
            "config_id": "mi_e10_a2_s12_m0.75_d0.3",
            "attack": "mi_fgsm",
            "epsilon": 10 / 255,
            "alpha": 2 / 255,
            "steps": 12,
            "momentum": 0.75,
            "diversity_prob": 0.3,
            "resize_rate": 0.85,
        },
    ]


def ensemble_search_configs() -> list[dict[str, object]]:
    return [
        {
            "config_id": "ens_e8_a2_s10_m1.0",
            "attack": "ensemble",
            "epsilon": 8 / 255,
            "alpha": 2 / 255,
            "steps": 10,
            "momentum": 1.0,
            "diversity_prob": 0.0,
            "resize_rate": 0.9,
        },
        {
            "config_id": "ens_e8_a1_s12_m1.0",
            "attack": "ensemble",
            "epsilon": 8 / 255,
            "alpha": 1 / 255,
            "steps": 12,
            "momentum": 1.0,
            "diversity_prob": 0.0,
            "resize_rate": 0.9,
        },
        {
            "config_id": "ens_e10_a2_s12_m1.0",
            "attack": "ensemble",
            "epsilon": 10 / 255,
            "alpha": 2 / 255,
            "steps": 12,
            "momentum": 1.0,
            "diversity_prob": 0.0,
            "resize_rate": 0.9,
        },
        {
            "config_id": "ens_e10_a1_s15_m1.0",
            "attack": "ensemble",
            "epsilon": 10 / 255,
            "alpha": 1 / 255,
            "steps": 15,
            "momentum": 1.0,
            "diversity_prob": 0.0,
            "resize_rate": 0.9,
        },
        {
            "config_id": "ens_e10_a2_s12_m1.0_d0.3",
            "attack": "ensemble",
            "epsilon": 10 / 255,
            "alpha": 2 / 255,
            "steps": 12,
            "momentum": 1.0,
            "diversity_prob": 0.3,
            "resize_rate": 0.85,
        },
        {
            "config_id": "ens_e12_a2_s12_m1.0",
            "attack": "ensemble",
            "epsilon": 12 / 255,
            "alpha": 2 / 255,
            "steps": 12,
            "momentum": 1.0,
            "diversity_prob": 0.0,
            "resize_rate": 0.9,
        },
    ]


def build_namespace(
    *,
    dataset: str,
    source_checkpoints: list[str],
    target_checkpoints: list[str],
    attack: str,
    epsilon: float,
    alpha: float,
    steps: int,
    momentum: float,
    diversity_prob: float,
    resize_rate: float,
    batch_size: int,
    num_workers: int,
    max_samples: int,
    seed: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        source_checkpoints=source_checkpoints,
        target_checkpoints=target_checkpoints,
        attack=attack,
        epsilon=epsilon,
        alpha=alpha,
        steps=steps,
        momentum=momentum,
        random_start=False,
        diversity_prob=diversity_prob,
        resize_rate=resize_rate,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
        seed=seed,
        output="",
    )


def add_metadata(row: dict[str, object], **metadata: object) -> dict[str, object]:
    merged = dict(row)
    merged.update(metadata)
    return merged


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["family"]),
            str(row["config_id"]),
            str(row.get("source_protocol", "")),
            str(row.get("source_anchor", "")),
        )
        summary = summaries.setdefault(
            key,
            {
                "family": row["family"],
                "config_id": row["config_id"],
                "attack": row["attack"],
                "source_protocol": row["source_protocol"],
                "source_anchor": row.get("source_anchor", ""),
                "targets_evaluated": 0,
                "mean_transfer_score_target_clean": 0.0,
                "mean_transfer_score_joint_clean": 0.0,
                "mean_transfer_asr_target_clean": 0.0,
                "mean_transfer_asr_joint_clean": 0.0,
                "mean_ssim_target_clean": 0.0,
                "mean_ssim_joint_clean": 0.0,
            },
        )
        summary["targets_evaluated"] += 1
        summary["mean_transfer_score_target_clean"] += float(row["transfer_score_target_clean"])
        summary["mean_transfer_score_joint_clean"] += float(row["transfer_score_joint_clean"])
        summary["mean_transfer_asr_target_clean"] += float(row["transfer_asr_target_clean"])
        summary["mean_transfer_asr_joint_clean"] += float(row["transfer_asr_joint_clean"])
        summary["mean_ssim_target_clean"] += float(row["mean_ssim_target_clean"])
        summary["mean_ssim_joint_clean"] += float(row["mean_ssim_joint_clean"])

    aggregated: list[dict[str, object]] = []
    for summary in summaries.values():
        count = max(int(summary["targets_evaluated"]), 1)
        for field in (
            "mean_transfer_score_target_clean",
            "mean_transfer_score_joint_clean",
            "mean_transfer_asr_target_clean",
            "mean_transfer_asr_joint_clean",
            "mean_ssim_target_clean",
            "mean_ssim_joint_clean",
        ):
            summary[field] = float(summary[field]) / count
        aggregated.append(summary)

    aggregated.sort(key=lambda row: (row["family"], -float(row["mean_transfer_score_target_clean"])))
    return aggregated


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output_path = Path(path)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_mi_fgsm_search(args: argparse.Namespace, checkpoints: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source_model in MODEL_NAMES:
        source_path = checkpoints[source_model]
        target_models = [name for name in MODEL_NAMES if name != source_model]
        for config in mi_fgsm_search_configs():
            namespace = build_namespace(
                dataset=args.dataset,
                source_checkpoints=[source_path],
                target_checkpoints=[checkpoints[name] for name in target_models],
                attack="mi_fgsm",
                epsilon=float(config["epsilon"]),
                alpha=float(config["alpha"]),
                steps=int(config["steps"]),
                momentum=float(config["momentum"]),
                diversity_prob=float(config["diversity_prob"]),
                resize_rate=float(config["resize_rate"]),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            result_rows = evaluate_transfer(namespace)
            rows.extend(
                add_metadata(
                    row,
                    family="mi_fgsm",
                    config_id=config["config_id"],
                    source_protocol="single_source_to_others",
                    source_anchor=source_model,
                )
                for row in result_rows
            )
    return rows


def run_ensemble_search(args: argparse.Namespace, checkpoints: dict[str, str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for config in ensemble_search_configs():
        for target_model in MODEL_NAMES:
            source_models = [name for name in MODEL_NAMES if name != target_model]
            namespace = build_namespace(
                dataset=args.dataset,
                source_checkpoints=[checkpoints[name] for name in source_models],
                target_checkpoints=[checkpoints[target_model]],
                attack="ensemble",
                epsilon=float(config["epsilon"]),
                alpha=float(config["alpha"]),
                steps=int(config["steps"]),
                momentum=float(config["momentum"]),
                diversity_prob=float(config["diversity_prob"]),
                resize_rate=float(config["resize_rate"]),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                max_samples=args.max_samples,
                seed=args.seed,
            )
            result_rows = evaluate_transfer(namespace)
            rows.extend(
                add_metadata(
                    row,
                    family="ensemble",
                    config_id=config["config_id"],
                    source_protocol="leave_one_out_ensemble",
                    source_anchor="",
                )
                for row in result_rows
            )
    return rows


def main() -> None:
    args = parse_args()
    checkpoints = checkpoint_map(args.models_root)

    rows: list[dict[str, object]] = []
    if args.attack_set in {"all", "mi_fgsm"}:
        rows.extend(run_mi_fgsm_search(args, checkpoints))
    if args.attack_set in {"all", "ensemble"}:
        rows.extend(run_ensemble_search(args, checkpoints))

    write_csv(args.records_output, rows)
    summary_rows = aggregate_rows(rows)
    write_csv(args.summary_output, summary_rows)

    print(f"wrote {len(rows)} search rows to {args.records_output}")
    print(f"wrote {len(summary_rows)} summary rows to {args.summary_output}")


if __name__ == "__main__":
    main()
