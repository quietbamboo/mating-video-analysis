#!/usr/bin/env python3
"""Reproduce the identity-attribution metrics reported in the manuscript.

The script reads event-level CSV files produced by the identity-attribution
pipeline and calculates:

* rooster, hen, and pair accuracy on evaluable events;
* successful attribution yield over all confirmed events;
* 95% Wilson confidence intervals;
* pair-accuracy differences from the deployed configuration;
* fallback trigger rate and fallback/main-path stratified metrics; and
* an exact paired McNemar comparison with the nearest-center baseline; and
* corrected/broken transitions, exact McNemar tests, and family-wise Holm
  corrections for every parameter-sensitivity setting.

Every input CSV is a final 499-event file after the cross-track duplicate
audit.  This script never excludes rows.  For consistency with the original
reported analysis, a trailing ``*`` on ``predicted_hen`` identifies the
fallback group; all remaining records are assigned to the main group.

Example
-------
python scripts/analyze_results.py
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


REQUIRED_FIELDS = {
    "clip_name",
    "expected_rooster",
    "predicted_rooster",
    "rooster_correct",
    "expected_hen",
    "predicted_hen",
    "hen_correct",
    "status",
}

OUTPUT_FIELDS = [
    "experiment",
    "family",
    "parameter",
    "setting",
    "is_primary",
    "total_events",
    "successful_events",
    "rooster_evaluable",
    "rooster_correct",
    "rooster_accuracy",
    "rooster_ci95_low",
    "rooster_ci95_high",
    "rooster_overall_yield",
    "hen_evaluable",
    "hen_correct",
    "hen_accuracy",
    "hen_ci95_low",
    "hen_ci95_high",
    "hen_overall_yield",
    "pair_evaluable",
    "pair_correct",
    "pair_accuracy",
    "pair_ci95_low",
    "pair_ci95_high",
    "pair_overall_yield",
    "pair_delta_from_primary_pp",
    "fallback_triggered",
    "fallback_hen_evaluable",
    "fallback_hen_correct",
    "fallback_hen_accuracy",
    "fallback_hen_ci95_low",
    "fallback_hen_ci95_high",
    "fallback_pair_evaluable",
    "fallback_pair_correct",
    "fallback_pair_accuracy",
    "fallback_pair_ci95_low",
    "fallback_pair_ci95_high",
    "main_path_events",
    "main_path_hen_evaluable",
    "main_path_hen_correct",
    "main_path_hen_accuracy",
    "main_path_hen_ci95_low",
    "main_path_hen_ci95_high",
    "main_path_pair_evaluable",
    "main_path_pair_correct",
    "main_path_pair_accuracy",
    "main_path_pair_ci95_low",
    "main_path_pair_ci95_high",
]

EXPECTED_EVENT_COUNT = 499
SENSITIVITY_FIELDS = [
    "experiment", "family", "parameter", "setting", "is_primary_setting",
    "compared_pair_evaluable", "pair_evaluable", "pair_correct", "pair_accuracy",
    "pair_delta_from_primary_pp", "both_correct", "corrected", "broken",
    "neither_correct", "exact_mcnemar_p", "holm_adjusted_p",
    "fallback_triggered", "main_path_pair_correct", "main_path_pair_evaluable",
    "main_path_pair_accuracy", "fallback_pair_correct",
    "fallback_pair_evaluable", "fallback_pair_accuracy",
]

MANUSCRIPT_VALUE_FIELDS = [
    "section",
    "metric",
    "numerator",
    "denominator",
    "value",
    "ci95_low",
    "ci95_high",
    "notes",
]


@dataclass(frozen=True)
class Experiment:
    family: str
    parameter: str
    setting: str
    numeric_setting: float | None


@dataclass(frozen=True)
class BinaryMetric:
    evaluable: int
    correct: int
    accuracy: float | None
    ci95_low: float | None
    ci95_high: float | None
    overall_yield: float | None


def parse_decimal(token: str) -> float:
    return float(token.replace("_", "."))


def classify_experiment(path: Path) -> Experiment:
    stem = path.stem
    if stem == "nearest-baseline-results":
        return Experiment("nearest-baseline", "method", "nearest-center", None)

    match = re.fullmatch(r"tau-gap-results-(\d+(?:_\d+)?)", stem)
    if match:
        value = parse_decimal(match.group(1))
        return Experiment("tau-gap", "tau_gap_seconds", f"{value:g}", value)

    match = re.fullmatch(r"nw-results-(\d+(?:_\d+)?)", stem)
    if match:
        value = parse_decimal(match.group(1))
        return Experiment(
            "observation-window", "window_fps_multiplier", f"{value:g}", value
        )

    match = re.fullmatch(r"roi-results-(\d+)(?:_(\d+))?", stem)
    if match:
        numerator = int(match.group(1))
        denominator = int(match.group(2) or 1)
        if denominator == 0:
            raise ValueError(f"Invalid zero ROI denominator in {path.name}")
        setting = str(numerator) if denominator == 1 else f"{numerator}/{denominator}"
        return Experiment("roi", "roi_area_fraction", setting, numerator / denominator)

    if stem == "outlier-ablation-results-none":
        return Experiment("outlier-ablation", "outlier_filter", "disabled", None)

    match = re.fullmatch(r"outlier-mean-std-results-(\d+(?:_\d+)?)", stem)
    if match:
        value = parse_decimal(match.group(1))
        return Experiment("outlier-mean-std", "sigma_multiplier", f"{value:g}", value)

    return Experiment("other", "", "", None)


def normalize_label(value: str | None) -> str:
    normalized = (value or "").strip()
    if normalized.endswith("*"):
        normalized = normalized[:-1]
    if normalized.endswith(","):
        normalized = normalized[:-1]
    return normalized


def expected_correctness(row: dict[str, str], role: str) -> str:
    expected = normalize_label(row.get(f"expected_{role}"))
    if not expected:
        return ""
    return "1" if normalize_label(row.get(f"predicted_{role}")) == expected else "0"


def read_rows(path: Path, expected_events: int) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED_FIELDS.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path.name} is missing required columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    if len(rows) != expected_events:
        raise ValueError(
            f"{path.name} contains {len(rows)} rows; expected {expected_events} "
            "final unique events"
        )

    seen: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        clip_name = (row.get("clip_name") or "").strip()
        if not clip_name:
            raise ValueError(f"{path.name}:{row_number}: empty clip_name")
        key = clip_name.casefold()
        if key in seen:
            raise ValueError(f"{path.name}: duplicate clip_name: {clip_name}")
        seen.add(key)

        for role in ("rooster", "hen"):
            calculated = expected_correctness(row, role)
            supplied = (row.get(f"{role}_correct") or "").strip()
            if supplied and supplied not in {"0", "1"}:
                raise ValueError(
                    f"{path.name}:{row_number}: invalid {role}_correct={supplied!r}"
                )
            if supplied and supplied != calculated:
                raise ValueError(
                    f"{path.name}:{row_number}: {role}_correct={supplied} "
                    f"disagrees with labels (calculated {calculated!r})"
                )
            row[f"{role}_correct"] = calculated
    return rows


def wilson_interval(correct: int, total: int) -> tuple[float | None, float | None]:
    if total == 0:
        return None, None
    z = 1.959963984540054
    proportion = correct / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def metric(
    rows: Sequence[dict[str, str]],
    total_events: int,
    correctness_fields: Sequence[str],
) -> BinaryMetric:
    evaluable_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
        and all(row.get(field) in {"0", "1"} for field in correctness_fields)
    ]
    correct = sum(
        all(row[field] == "1" for field in correctness_fields)
        for row in evaluable_rows
    )
    evaluable = len(evaluable_rows)
    accuracy = correct / evaluable if evaluable else None
    low, high = wilson_interval(correct, evaluable)
    overall_yield = correct / total_events if total_events else None
    return BinaryMetric(evaluable, correct, accuracy, low, high, overall_yield)


def summarize(
    path: Path, primary_name: str
) -> tuple[dict[str, object], dict[str, dict[str, str]]]:
    rows = read_rows(path, EXPECTED_EVENT_COUNT)
    experiment = classify_experiment(path)
    total = len(rows)
    successful = sum(row.get("status") == "ok" for row in rows)
    rooster = metric(rows, total, ("rooster_correct",))
    hen = metric(rows, total, ("hen_correct",))
    pair = metric(rows, total, ("rooster_correct", "hen_correct"))

    has_fallback_marker = lambda row: (
        (row.get("predicted_hen") or "").strip().endswith("*")
    )
    fallback_rows = [row for row in rows if has_fallback_marker(row)]
    main_rows = [row for row in rows if not has_fallback_marker(row)]
    fallback_hen = metric(fallback_rows, len(fallback_rows), ("hen_correct",))
    fallback_pair = metric(
        fallback_rows, len(fallback_rows), ("rooster_correct", "hen_correct")
    )
    main_hen = metric(main_rows, len(main_rows), ("hen_correct",))
    main_pair = metric(main_rows, len(main_rows), ("rooster_correct", "hen_correct"))

    result: dict[str, object] = {
        "experiment": path.name,
        "family": experiment.family,
        "parameter": experiment.parameter,
        "setting": experiment.setting,
        "numeric_setting": experiment.numeric_setting,
        "is_primary": path.name.casefold() == primary_name.casefold(),
        "total_events": total,
        "successful_events": successful,
        "rooster_evaluable": rooster.evaluable,
        "rooster_correct": rooster.correct,
        "rooster_accuracy": rooster.accuracy,
        "rooster_ci95_low": rooster.ci95_low,
        "rooster_ci95_high": rooster.ci95_high,
        "rooster_overall_yield": rooster.overall_yield,
        "hen_evaluable": hen.evaluable,
        "hen_correct": hen.correct,
        "hen_accuracy": hen.accuracy,
        "hen_ci95_low": hen.ci95_low,
        "hen_ci95_high": hen.ci95_high,
        "hen_overall_yield": hen.overall_yield,
        "pair_evaluable": pair.evaluable,
        "pair_correct": pair.correct,
        "pair_accuracy": pair.accuracy,
        "pair_ci95_low": pair.ci95_low,
        "pair_ci95_high": pair.ci95_high,
        "pair_overall_yield": pair.overall_yield,
        "pair_delta_from_primary_pp": None,
        "fallback_triggered": len(fallback_rows),
        "fallback_hen_evaluable": fallback_hen.evaluable,
        "fallback_hen_correct": fallback_hen.correct,
        "fallback_hen_accuracy": fallback_hen.accuracy,
        "fallback_hen_ci95_low": fallback_hen.ci95_low,
        "fallback_hen_ci95_high": fallback_hen.ci95_high,
        "fallback_pair_evaluable": fallback_pair.evaluable,
        "fallback_pair_correct": fallback_pair.correct,
        "fallback_pair_accuracy": fallback_pair.accuracy,
        "fallback_pair_ci95_low": fallback_pair.ci95_low,
        "fallback_pair_ci95_high": fallback_pair.ci95_high,
        "main_path_events": len(main_rows),
        "main_path_hen_evaluable": main_hen.evaluable,
        "main_path_hen_correct": main_hen.correct,
        "main_path_hen_accuracy": main_hen.accuracy,
        "main_path_hen_ci95_low": main_hen.ci95_low,
        "main_path_hen_ci95_high": main_hen.ci95_high,
        "main_path_pair_evaluable": main_pair.evaluable,
        "main_path_pair_correct": main_pair.correct,
        "main_path_pair_accuracy": main_pair.accuracy,
        "main_path_pair_ci95_low": main_pair.ci95_low,
        "main_path_pair_ci95_high": main_pair.ci95_high,
    }
    rows_by_clip = {(row["clip_name"]).casefold(): row for row in rows}
    return result, rows_by_clip


def verify_comparable_inputs(
    rows_by_experiment: dict[str, dict[str, dict[str, str]]]
) -> None:
    names = list(rows_by_experiment)
    reference_name = names[0]
    reference = rows_by_experiment[reference_name]
    for name in names[1:]:
        candidate = rows_by_experiment[name]
        if set(candidate) != set(reference):
            raise ValueError(
                f"Clip set differs between {reference_name} and {name}; "
                "the experiments are not directly comparable"
            )
        mismatches = [
            clip
            for clip in reference
            if (
                reference[clip].get("expected_rooster"),
                reference[clip].get("expected_hen"),
            )
            != (
                candidate[clip].get("expected_rooster"),
                candidate[clip].get("expected_hen"),
            )
        ]
        if mismatches:
            raise ValueError(
                f"Ground truth differs between {reference_name} and {name} "
                f"for {len(mismatches)} clips"
            )


def exact_mcnemar_p(proposed_only: int, baseline_only: int) -> float:
    discordant = proposed_only + baseline_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, k) for k in range(min(proposed_only, baseline_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def paired_comparison(
    primary_rows: dict[str, dict[str, str]],
    baseline_rows: dict[str, dict[str, str]],
) -> dict[str, int | float]:
    common = sorted(set(primary_rows).intersection(baseline_rows))
    counts = {
        "both_correct": 0,
        "primary_only": 0,
        "baseline_only": 0,
        "neither_correct": 0,
    }
    compared = 0
    for clip in common:
        primary = primary_rows[clip]
        baseline = baseline_rows[clip]
        if primary.get("status") != "ok" or baseline.get("status") != "ok":
            continue
        fields = ("rooster_correct", "hen_correct")
        if not all(
            primary.get(field) in {"0", "1"}
            and baseline.get(field) in {"0", "1"}
            for field in fields
        ):
            continue
        compared += 1
        primary_correct = all(primary[field] == "1" for field in fields)
        baseline_correct = all(baseline[field] == "1" for field in fields)
        if primary_correct and baseline_correct:
            counts["both_correct"] += 1
        elif primary_correct:
            counts["primary_only"] += 1
        elif baseline_correct:
            counts["baseline_only"] += 1
        else:
            counts["neither_correct"] += 1
    return {
        "compared_pair_evaluable": compared,
        **counts,
        "exact_mcnemar_p": exact_mcnemar_p(
            counts["primary_only"], counts["baseline_only"]
        ),
    }


def is_deployed_setting(result: dict[str, object]) -> bool:
    deployed = {
        "tau-gap": 1.0,
        "observation-window": 1.0,
        "roi": 1.0 / 9.0,
        "outlier-mean-std": 2.0,
    }
    family = str(result["family"])
    numeric = result.get("numeric_setting")
    return (
        family in deployed
        and numeric is not None
        and math.isclose(float(numeric), deployed[family])
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: (p_values[key], key))
    adjusted: dict[str, float] = {}
    previous = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (total - index) * p_values[key])
        value = max(previous, value)
        adjusted[key] = value
        previous = value
    return adjusted


def sensitivity_statistics(
    results: Sequence[dict[str, object]],
    rows_by_experiment: dict[str, dict[str, dict[str, str]]],
    primary_rows: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    sensitivity_families = {
        "tau-gap", "observation-window", "roi",
        "outlier-mean-std", "outlier-ablation",
    }
    records: list[dict[str, object]] = []
    for result in results:
        family = str(result["family"])
        if family not in sensitivity_families:
            continue
        comparison = paired_comparison(
            primary_rows, rows_by_experiment[str(result["experiment"])]
        )
        records.append(
            {
                "experiment": result["experiment"],
                "family": family,
                "parameter": result["parameter"],
                "setting": result["setting"],
                "numeric_setting": result.get("numeric_setting"),
                "is_primary_setting": is_deployed_setting(result),
                "compared_pair_evaluable": comparison["compared_pair_evaluable"],
                "pair_evaluable": result["pair_evaluable"],
                "pair_correct": result["pair_correct"],
                "pair_accuracy": result["pair_accuracy"],
                "pair_delta_from_primary_pp": result["pair_delta_from_primary_pp"],
                "both_correct": comparison["both_correct"],
                "corrected": comparison["baseline_only"],
                "broken": comparison["primary_only"],
                "neither_correct": comparison["neither_correct"],
                "exact_mcnemar_p": comparison["exact_mcnemar_p"],
                "holm_adjusted_p": None,
                "fallback_triggered": result["fallback_triggered"],
                "main_path_pair_correct": result["main_path_pair_correct"],
                "main_path_pair_evaluable": result["main_path_pair_evaluable"],
                "main_path_pair_accuracy": result["main_path_pair_accuracy"],
                "fallback_pair_correct": result["fallback_pair_correct"],
                "fallback_pair_evaluable": result["fallback_pair_evaluable"],
                "fallback_pair_accuracy": result["fallback_pair_accuracy"],
            }
        )

    correction_families = {
        "tau-gap": "tau-gap",
        "observation-window": "observation-window",
        "roi": "roi",
        "outlier-mean-std": "trajectory-outlier",
        "outlier-ablation": "trajectory-outlier",
    }
    for correction_family in sorted(set(correction_families.values())):
        alternatives = [
            record
            for record in records
            if correction_families[str(record["family"])] == correction_family
            and not record["is_primary_setting"]
        ]
        adjusted = holm_adjust(
            {
                str(record["experiment"]): float(record["exact_mcnemar_p"])
                for record in alternatives
            }
        )
        for record in alternatives:
            record["holm_adjusted_p"] = adjusted[str(record["experiment"])]
    return sorted(records, key=family_sort_key)


def family_sort_key(result: dict[str, object]) -> tuple[int, float, str]:
    order = {
        "nearest-baseline": 0,
        "tau-gap": 1,
        "observation-window": 2,
        "roi": 3,
        "outlier-ablation": 4,
        "outlier-mean-std": 5,
        "other": 6,
    }
    numeric = result.get("numeric_setting")
    return (
        order.get(str(result["family"]), 99),
        float(numeric) if numeric is not None else -1.0,
        str(result["experiment"]),
    )


def csv_value(value: object) -> object:
    if isinstance(value, float):
        if value != 0.0 and abs(value) < 1e-9:
            return f"{value:.10e}"
        return f"{value:.10f}"
    return "" if value is None else value


def write_metrics(path: Path, results: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: csv_value(result.get(field)) for field in OUTPUT_FIELDS})


def write_sensitivity_statistics(
    path: Path, records: Iterable[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SENSITIVITY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field: csv_value(record.get(field)) for field in SENSITIVITY_FIELDS}
            )


def metric_row(
    section: str,
    name: str,
    numerator: int,
    denominator: int,
    notes: str = "",
    with_ci: bool = True,
) -> dict[str, object]:
    value = numerator / denominator if denominator else None
    low, high = wilson_interval(numerator, denominator) if with_ci else (None, None)
    return {
        "section": section,
        "metric": name,
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "ci95_low": low,
        "ci95_high": high,
        "notes": notes,
    }


def error_breakdown(rows: dict[str, dict[str, str]]) -> dict[str, int]:
    evaluable = [
        row
        for row in rows.values()
        if row.get("status") == "ok"
        and row.get("rooster_correct") in {"0", "1"}
        and row.get("hen_correct") in {"0", "1"}
    ]
    incorrect = [
        row
        for row in evaluable
        if not (row["rooster_correct"] == "1" and row["hen_correct"] == "1")
    ]
    return {
        "pair_evaluable": len(evaluable),
        "pair_incorrect": len(incorrect),
        "hen_only_incorrect": sum(
            row["rooster_correct"] == "1" and row["hen_correct"] == "0"
            for row in incorrect
        ),
        "rooster_only_incorrect": sum(
            row["rooster_correct"] == "0" and row["hen_correct"] == "1"
            for row in incorrect
        ),
        "both_incorrect": sum(
            row["rooster_correct"] == "0" and row["hen_correct"] == "0"
            for row in incorrect
        ),
    }


def write_manuscript_values(
    path: Path,
    primary: dict[str, object],
    baseline: dict[str, object] | None,
    comparison: dict[str, int | float] | None,
    primary_rows: dict[str, dict[str, str]],
) -> None:
    total = int(primary["total_events"])
    rows: list[dict[str, object]] = []

    for label in ("rooster", "hen", "pair"):
        correct = int(primary[f"{label}_correct"])
        evaluable = int(primary[f"{label}_evaluable"])
        rows.append(
            metric_row(
                "identity_accuracy",
                f"{label}_conditional_accuracy",
                correct,
                evaluable,
                "Accuracy among identity-evaluable unique events.",
            )
        )
        rows.append(
            metric_row(
                "identity_accuracy",
                f"{label}_overall_yield",
                correct,
                total,
                "Correct unique events divided by all 499 unique confirmed events.",
            )
        )

    pair_evaluable = int(primary["pair_evaluable"])
    pair_correct = int(primary["pair_correct"])
    rows.extend(
        [
            metric_row(
                "identity_accuracy",
                "pair_evaluable_fraction",
                pair_evaluable,
                total,
                "Fraction of unique events with verifiable identities for both birds.",
                with_ci=False,
            ),
            metric_row(
                "identity_accuracy",
                "unsuccessful_or_unresolved",
                total - pair_correct,
                total,
                "Incorrect pair assignments plus pair-unevaluable events.",
                with_ci=False,
            ),
            metric_row(
                "identity_accuracy",
                "incorrect_pair_assignments",
                pair_evaluable - pair_correct,
                total,
                "Incorrect among evaluable events, expressed over all unique events.",
                with_ci=False,
            ),
            metric_row(
                "identity_accuracy",
                "pair_unevaluable",
                total - pair_evaluable,
                total,
                "Events for which at least one ground-truth identity was not verifiable.",
                with_ci=False,
            ),
        ]
    )

    rows.extend(
        [
            metric_row(
                "fallback",
                "fallback_trigger_rate",
                int(primary["fallback_triggered"]),
                total,
                "Fallback is identified by a trailing * on predicted_hen.",
                with_ci=False,
            ),
            metric_row(
                "fallback",
                "fallback_hen_accuracy",
                int(primary["fallback_hen_correct"]),
                int(primary["fallback_hen_evaluable"]),
            ),
            metric_row(
                "fallback",
                "fallback_pair_accuracy",
                int(primary["fallback_pair_correct"]),
                int(primary["fallback_pair_evaluable"]),
            ),
            metric_row(
                "main_path",
                "main_path_hen_accuracy",
                int(primary["main_path_hen_correct"]),
                int(primary["main_path_hen_evaluable"]),
            ),
            metric_row(
                "main_path",
                "main_path_pair_accuracy",
                int(primary["main_path_pair_correct"]),
                int(primary["main_path_pair_evaluable"]),
            ),
        ]
    )

    if baseline is not None:
        rows.extend(
            [
                metric_row(
                    "nearest_center_baseline",
                    "baseline_hen_accuracy",
                    int(baseline["hen_correct"]),
                    int(baseline["hen_evaluable"]),
                ),
                metric_row(
                    "nearest_center_baseline",
                    "baseline_pair_accuracy",
                    int(baseline["pair_correct"]),
                    int(baseline["pair_evaluable"]),
                ),
            ]
        )

    if comparison is not None:
        compared = int(comparison["compared_pair_evaluable"])
        for key in ("both_correct", "primary_only", "baseline_only", "neither_correct"):
            rows.append(
                metric_row(
                    "paired_comparison",
                    key,
                    int(comparison[key]),
                    compared,
                    "Paired classification outcome on pair-evaluable events.",
                    with_ci=False,
                )
            )
        rows.append(
            {
                "section": "paired_comparison",
                "metric": "exact_mcnemar_p",
                "numerator": "",
                "denominator": compared,
                "value": comparison["exact_mcnemar_p"],
                "ci95_low": "",
                "ci95_high": "",
                "notes": "Two-sided exact McNemar test from discordant pairs.",
            }
        )

    errors = error_breakdown(primary_rows)
    for key in (
        "pair_incorrect",
        "hen_only_incorrect",
        "rooster_only_incorrect",
        "both_incorrect",
    ):
        rows.append(
            metric_row(
                "error_breakdown",
                key,
                errors[key],
                errors["pair_evaluable"],
                "Count and fraction among pair-evaluable unique events.",
                with_ci=False,
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANUSCRIPT_VALUE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: csv_value(row.get(field)) for field in MANUSCRIPT_VALUE_FIELDS}
            )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    path: Path,
    input_paths: Sequence[Path],
    primary: dict[str, object],
    package_root: Path,
) -> None:
    manifest = {
        "analysis_unit": "unique confirmed mating event after cross-track audit",
        "input_state": "final 499-event files; no exclusions are applied",
        "analyzed_unique_events": int(primary["total_events"]),
        "primary_configuration": str(primary["experiment"]),
        "input_files": [],
    }
    for item in input_paths:
        try:
            display_path = item.relative_to(package_root).as_posix()
        except ValueError:
            display_path = f"<external>/{item.name}"
        manifest["input_files"].append(
            {"relative_path": display_path, "sha256": sha256_file(item)}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def percent(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.1f}"


def delta_pp(value: object) -> str:
    if value is None:
        return "N/A"
    number = float(value)
    if abs(number) < 0.05:
        return "0.0"
    return f"{number:+.1f}"


def format_ci(metric_result: dict[str, object], prefix: str) -> str:
    return (
        f"{metric_result[prefix + '_correct']}/{metric_result[prefix + '_evaluable']} "
        f"({percent(metric_result[prefix + '_accuracy'])}%; 95% CI "
        f"{percent(metric_result[prefix + '_ci95_low'])}--"
        f"{percent(metric_result[prefix + '_ci95_high'])}%)"
    )


def write_complete_sensitivity_latex(
    path: Path, sensitivity: Sequence[dict[str, object]]
) -> None:
    """Write the complete corrected/broken and pathway table used in the paper."""
    lines = ["% Generated by run_analysis.py; do not edit manually."]
    groups = [
        ({"tau-gap"}, r"$\tau_{\mathrm{gap}}$ (s)"),
        ({"observation-window"}, r"Observation window $N_w/F$"),
        ({"roi"}, "ROI area fraction"),
        ({"outlier-mean-std", "outlier-ablation"},
         r"Trajectory outlier threshold $k$"),
    ]
    for families, label in groups:
        members = sorted(
            (row for row in sensitivity if str(row["family"]) in families),
            key=lambda row: (
                float(row["numeric_setting"])
                if row.get("numeric_setting") is not None
                else math.inf,
                str(row["experiment"]),
            ),
        )
        for index, row in enumerate(members):
            setting = str(row["setting"])
            if row["family"] in {"tau-gap", "observation-window"}:
                setting = f"{float(row['numeric_setting']):.2f}"
            values = [
                f"{row['pair_correct']}/{row['pair_evaluable']} "
                f"({percent(row['pair_accuracy'])})",
                delta_pp(row["pair_delta_from_primary_pp"]),
                str(row["corrected"]),
                str(row["broken"]),
                str(row["fallback_triggered"]),
                f"{row['main_path_pair_correct']}/{row['main_path_pair_evaluable']} "
                f"({percent(row['main_path_pair_accuracy'])}) / "
                f"{row['fallback_pair_correct']}/{row['fallback_pair_evaluable']} "
                f"({percent(row['fallback_pair_accuracy'])})",
            ]
            if row["is_primary_setting"]:
                setting = rf"\textbf{{{setting}}}"
                values = [rf"\textbf{{{value}}}" for value in values]
            parameter = label if index == 0 else ""
            lines.append(
                f"{parameter} & {setting} & " + " & ".join(values) + r" \\"
            )
        if members:
            lines.append(r"\hline")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_stratified_ci(metric_result: dict[str, object], prefix: str) -> str:
    return (
        f"{metric_result[prefix + '_correct']}/{metric_result[prefix + '_evaluable']} "
        f"({percent(metric_result[prefix + '_accuracy'])}%; 95% CI "
        f"{percent(metric_result[prefix + '_ci95_low'])}--"
        f"{percent(metric_result[prefix + '_ci95_high'])}%)"
    )


def print_report(
    results: Sequence[dict[str, object]],
    primary: dict[str, object],
    comparison: dict[str, int | float] | None,
    sensitivity: Sequence[dict[str, object]],
) -> None:
    print(f"Primary configuration: {primary['experiment']}")
    print(f"  Rooster: {format_ci(primary, 'rooster')}")
    print(f"  Hen:     {format_ci(primary, 'hen')}")
    print(f"  Pair:    {format_ci(primary, 'pair')}")
    print(
        f"  Pair overall yield: {primary['pair_correct']}/{primary['total_events']} "
        f"({percent(primary['pair_overall_yield'])}%)"
    )
    print(
        f"  Fallback: {primary['fallback_triggered']}/{primary['total_events']} triggered; "
        f"hen {format_stratified_ci(primary, 'fallback_hen')}; "
        f"pair {format_stratified_ci(primary, 'fallback_pair')}"
    )
    print(
        f"  Main path: hen {format_stratified_ci(primary, 'main_path_hen')}; "
        f"pair {format_stratified_ci(primary, 'main_path_pair')}"
    )

    if comparison is not None:
        p_value = float(comparison["exact_mcnemar_p"])
        print("Nearest-center paired comparison:")
        print(
            "  both correct={both_correct}, primary only={primary_only}, "
            "baseline only={baseline_only}, neither={neither_correct}".format(**comparison)
        )
        print(f"  Exact McNemar p={p_value:.6g}")

    print("\nSensitivity metrics (accuracy on evaluable events):")
    print("experiment, rooster %, hen %, pair %, delta pair pp")
    for result in results:
        print(
            f"{result['experiment']}, {percent(result['rooster_accuracy'])}, "
            f"{percent(result['hen_accuracy'])}, {percent(result['pair_accuracy'])}, "
            f"{delta_pp(result['pair_delta_from_primary_pp'])}"
        )

    print("\nComplete paired sensitivity statistics:")
    print("experiment, corrected, broken, exact McNemar p, Holm-adjusted p")
    for record in sensitivity:
        adjusted = record["holm_adjusted_p"]
        adjusted_text = "deployed" if adjusted is None else f"{float(adjusted):.6g}"
        print(
            f"{record['experiment']}, {record['corrected']}, {record['broken']}, "
            f"{float(record['exact_mcnemar_p']):.6g}, {adjusted_text}"
        )


def parse_args() -> argparse.Namespace:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        default=package_root / "results",
        help="Directory containing final 499-event CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--primary",
        default="roi-results-1_9.csv",
        help="CSV representing the deployed configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--baseline",
        default="nearest-baseline-results.csv",
        help="Nearest-center baseline CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=package_root / "output" / "analysis",
        help="Directory for all generated results (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results directory does not exist: {results_dir}")
    paths = sorted(
        path
        for path in results_dir.rglob("*.csv")
        if "generated" not in path.relative_to(results_dir).parts
    )
    if not paths:
        raise SystemExit(f"No CSV files found in {results_dir}")
    names: dict[str, Path] = {}
    for path in paths:
        key = path.name.casefold()
        if key in names:
            raise SystemExit(
                f"Duplicate result filename: {names[key]} and {path}"
            )
        names[key] = path
    if args.primary.casefold() not in names:
        raise SystemExit(f"Primary CSV not found: {args.primary}")

    summaries: list[dict[str, object]] = []
    rows_by_experiment: dict[str, dict[str, dict[str, str]]] = {}
    try:
        for path in paths:
            summary, rows = summarize(path, args.primary)
            summaries.append(summary)
            rows_by_experiment[path.name] = rows
        verify_comparable_inputs(rows_by_experiment)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Input validation failed: {exc}") from None
    summaries.sort(key=family_sort_key)

    primary = next(result for result in summaries if result["is_primary"])
    primary_accuracy = float(primary["pair_accuracy"])
    for result in summaries:
        result["pair_delta_from_primary_pp"] = (
            100.0 * (float(result["pair_accuracy"]) - primary_accuracy)
            if result["pair_accuracy"] is not None
            else None
        )

    comparison = None
    baseline_summary = None
    baseline_path = names.get(args.baseline.casefold())
    if baseline_path is not None:
        baseline_summary = next(
            result
            for result in summaries
            if str(result["experiment"]).casefold() == args.baseline.casefold()
        )
        comparison = paired_comparison(
            rows_by_experiment[names[args.primary.casefold()].name],
            rows_by_experiment[baseline_path.name],
        )

    primary_rows = rows_by_experiment[names[args.primary.casefold()].name]
    sensitivity = sensitivity_statistics(summaries, rows_by_experiment, primary_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "identity_metrics_all_experiments.csv"
    sensitivity_path = output_dir / "sensitivity_statistics.csv"
    manuscript_path = output_dir / "manuscript_values.csv"
    latex_path = output_dir / "identity_sensitivity_rows.tex"
    report_path = output_dir / "summary_report.txt"
    manifest_path = output_dir / "analysis_manifest.json"

    write_metrics(metrics_path, summaries)
    write_sensitivity_statistics(sensitivity_path, sensitivity)
    write_manuscript_values(
        manuscript_path,
        primary,
        baseline_summary,
        comparison,
        primary_rows,
    )
    write_complete_sensitivity_latex(latex_path, sensitivity)
    write_manifest(
        manifest_path,
        paths,
        primary,
        Path(__file__).resolve().parents[1],
    )

    report_buffer = io.StringIO()
    with contextlib.redirect_stdout(report_buffer):
        print_report(summaries, primary, comparison, sensitivity)
    report = report_buffer.getvalue()
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"\nOutputs written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
