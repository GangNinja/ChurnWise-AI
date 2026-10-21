"""Human-readable Phase 7 benchmark report generated from measured artifacts."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from churnwise.evaluation.contracts import (
    FAMILY_LABELS,
    LOGISTIC_FAMILY,
    MLP_FAMILY,
    RANDOM_FOREST_FAMILY,
    SUPPORTED_FAMILIES,
    BenchmarkResult,
)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.{digits}f}%"


def _config_summary(block: Mapping[str, Any]) -> str:
    family = block.get("family")
    model = block.get("model") or {}
    config = model.get("config") or {}
    if family == MLP_FAMILY:
        architecture = config.get("hidden_units") or []
        return (
            f"MLP {'/'.join(str(units) for units in architecture)} ReLU, "
            f"dropout {config.get('dropout')}, L2 {config.get('l2')}, "
            f"Adam {config.get('learning_rate')}, batch {config.get('batch_size')}, "
            f"epochs <= {config.get('epochs')}"
        )
    if family == LOGISTIC_FAMILY:
        return (
            f"LogisticRegression(C={config.get('C')}, solver={config.get('solver')}, "
            f"penalty={config.get('penalty')}, max_iter={config.get('max_iter')})"
        )
    if family == RANDOM_FOREST_FAMILY:
        return (
            f"RandomForestClassifier(n_estimators={config.get('n_estimators')}, "
            f"max_depth={config.get('max_depth')}, "
            f"min_samples_leaf={config.get('min_samples_leaf')}, "
            f"max_features={config.get('max_features')})"
        )
    return "unknown configuration"


def _completed(metrics: Mapping[str, Any]) -> list[str]:
    families = metrics.get("families") or {}
    return [
        name
        for name in SUPPORTED_FAMILIES
        if (families.get(name) or {}).get("status") == "completed"
    ]


def _strengths_and_weaknesses(metrics: Mapping[str, Any]) -> list[str]:
    families = metrics.get("families") or {}
    completed = _completed(metrics)
    if not completed:
        return ["No model family completed, so no strengths can be reported."]
    test = {
        family: (families[family].get("test_metrics") or {}) for family in completed
    }
    lines: list[str] = []
    for family in completed:
        label = FAMILY_LABELS[family]
        block = families[family]
        strengths: list[str] = []
        weaknesses: list[str] = []
        for metric, label_text in (
            ("recall", "recall"),
            ("precision", "precision"),
            ("f1", "F1"),
            ("roc_auc", "ROC-AUC"),
        ):
            values = {
                name: test[name].get(metric)
                for name in completed
                if test[name].get(metric) is not None
            }
            if len(values) < 2 or test[family].get(metric) is None:
                continue
            best = max(values.items(), key=lambda item: item[1])
            worst = min(values.items(), key=lambda item: item[1])
            if best[0] == family:
                strengths.append(
                    f"highest test {label_text} ({_fmt(best[1])})"
                )
            if worst[0] == family:
                weaknesses.append(f"lowest test {label_text} ({_fmt(worst[1])})")
        briers = {
            name: test[name].get("brier")
            for name in completed
            if test[name].get("brier") is not None
        }
        if len(briers) >= 2 and test[family].get("brier") is not None:
            if min(briers.items(), key=lambda item: item[1])[0] == family:
                strengths.append(f"best probability calibration by Brier score ({_fmt(briers[family])})")
            if max(briers.items(), key=lambda item: item[1])[0] == family:
                weaknesses.append(f"weakest Brier score ({_fmt(briers[family])})")
        threshold = (block.get("threshold") or {}).get("selected")
        strengths_text = "; ".join(strengths) if strengths else "no leading test metric"
        weaknesses_text = "; ".join(weaknesses) if weaknesses else "no trailing test metric"
        lines.append(
            f"- **{label}** (threshold {_fmt(threshold, 3)}): observed strengths — "
            f"{strengths_text}. Observed weaknesses — {weaknesses_text}."
        )
    lines.append(
        "- These relative statements describe this held-out split only; a different "
        "customer base or period may reorder them."
    )
    return lines


def _family_config_table(metrics: Mapping[str, Any]) -> list[str]:
    families = metrics.get("families") or {}
    lines = [
        "| Family | Source | Configuration | Train rows | Fit time (s) | Threshold |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for family in SUPPORTED_FAMILIES:
        block = families.get(family) or {}
        status = block.get("status")
        if status != "completed":
            error = block.get("error") or {}
            lines.append(
                f"| {FAMILY_LABELS[family]} | {block.get('source', 'n/a')} | "
                f"**failed** — {error.get('message', 'see metrics.json')} | — | — | — |"
            )
            continue
        model = block.get("model") or {}
        threshold = (block.get("threshold") or {}).get("selected")
        lines.append(
            f"| {FAMILY_LABELS[family]} | {block.get('source')} | "
            f"{_config_summary(block)} | {model.get('train_rows', 'n/a')} | "
            f"{_fmt(model.get('fit_seconds'), 2)} | {_fmt(threshold, 3)} |"
        )
    return lines


def _main_table(metrics: Mapping[str, Any]) -> list[str]:
    rows = metrics.get("comparison_table") or []
    lines = [
        "| Family | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | TN | FP | FN | TP |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {threshold} | {accuracy} | {precision} | {recall} | {f1} | "
            "{roc} | {pr} | {brier} | {tn} | {fp} | {fn} | {tp} |".format(
                label=row.get("label"),
                threshold=_fmt(row.get("threshold"), 3),
                accuracy=_fmt(row.get("accuracy")),
                precision=_fmt(row.get("precision")),
                recall=_fmt(row.get("recall")),
                f1=_fmt(row.get("f1")),
                roc=_fmt(row.get("roc_auc")),
                pr=_fmt(row.get("pr_auc")),
                brier=_fmt(row.get("brier")),
                tn=row.get("tn", "n/a"),
                fp=row.get("fp", "n/a"),
                fn=row.get("fn", "n/a"),
                tp=row.get("tp", "n/a"),
            )
        )
    if not rows:
        lines.append("| no completed family | — | — | — | — | — | — | — | — | — | — | — | — |")
    return lines


def _default_threshold_table(metrics: Mapping[str, Any]) -> list[str]:
    families = metrics.get("families") or {}
    lines = [
        "| Family | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    drawn = False
    for family in _completed(metrics):
        block = families[family].get("test_metrics_at_0_5") or {}
        lines.append(
            f"| {FAMILY_LABELS[family]} | {_fmt(block.get('accuracy'))} | "
            f"{_fmt(block.get('precision'))} | {_fmt(block.get('recall'))} | "
            f"{_fmt(block.get('f1'))} | {_fmt(block.get('roc_auc'))} |"
        )
        drawn = True
    if not drawn:
        lines.append("| no completed family | — | — | — | — | — |")
    return lines


def _cv_table(metrics: Mapping[str, Any]) -> list[str]:
    families = metrics.get("families") or {}
    lines = [
        "| Family | Folds | Accuracy | Precision | Recall | F1 | ROC-AUC |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    drawn = False
    for family in _completed(metrics):
        cv = (families[family].get("cv") or {})
        if not cv:
            continue
        values = cv.get("metrics") or {}
        cells = []
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            block = values.get(key) or {}
            if block.get("mean") is None:
                cells.append("n/a")
            else:
                cells.append(f"{_fmt(block['mean'])} ± {_fmt(block.get('std'))}")
        lines.append(
            f"| {FAMILY_LABELS[family]} | {cv.get('folds_completed')}/{cv.get('folds_requested')} | "
            + " | ".join(cells)
            + " |"
        )
        drawn = True
    if not drawn:
        lines.append("| not run | 0 | — | — | — | — | — |")
    return lines


def _cv_preprocessing_sentence(metrics: Mapping[str, Any]) -> str:
    block = metrics.get("cross_validation") or {}
    preprocessing = block.get("preprocessing") or {}
    if not block.get("enabled"):
        return "Cross-validation was disabled for this run."
    if preprocessing.get("fold_safe"):
        return (
            "Each fold refits the Phase 5 column plan, imputers, scaler and encoder "
            "on its own training rows; validation rows are transform-only, so no "
            "fold sees its own validation statistics."
        )
    return (
        "Fold-safe preprocessing was unavailable for this run; see "
        "`cross_validation.preprocessing` in `metrics.json` for the reason."
    )


def render_benchmark_markdown(result: BenchmarkResult) -> str:
    metrics = result.metrics
    preprocessing = metrics.get("preprocessing") or {}
    dataset = metrics.get("dataset") or {}
    selection = metrics.get("selection") or {}
    config = metrics.get("evaluation_config") or {}
    families = metrics.get("families") or {}
    completed = _completed(metrics)
    lines: list[str] = [
        f"# ChurnWise AI — Phase 7 benchmark `{result.benchmark_run_id}`",
        "",
        "All metrics below were measured on the held-out test partition created in "
        "Phase 5 and scored once, after thresholds were frozen on validation data. "
        "No number is illustrative.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Benchmark run | `{result.benchmark_run_id}` |",
        f"| Preprocessing run | `{preprocessing.get('run_id', '')}` |",
        f"| Dataset | `{preprocessing.get('dataset_id', '')}` |",
        f"| Dataset SHA-256 | `{preprocessing.get('dataset_sha256', '')}` |",
        f"| Split fingerprint | `{preprocessing.get('split_fingerprint', '')}` |",
        f"| Preprocessor hash | `{preprocessing.get('preprocessor_hash', '')}` |",
        f"| Transformed features | {dataset.get('feature_count')} |",
        f"| Train / validation / test rows | {dataset.get('n_train')} / {dataset.get('n_val')} / {dataset.get('n_test')} |",
        f"| Test churn rows | {dataset.get('test_positive_count')} "
        f"({_pct(dataset.get('test_positive_rate'))}) |",
        f"| Generated | {metrics.get('generated_at', '')} |",
        f"| Code version | `{metrics.get('code_version', '')}` |",
        f"| Complete (all three families) | {'yes' if metrics.get('complete') else 'no'} |",
        "",
        "## 1. Methodology",
        "",
        "- **Shared data:** Logistic Regression, Random Forest and the Phase 6 MLP all "
        "consume the same Phase 5 transformed matrices and the same split fingerprint "
        "recorded above.",
        "- **Fitting scope:** baselines are fitted on the training partition only. The "
        "MLP artifact was trained on that same partition in Phase 6 and is loaded, not "
        "retrained.",
        "- **Imbalance treatment:** all families receive the Phase 5 training class "
        f"weights `{dataset.get('class_weights')}`.",
        "- **Threshold policy:** each family's operating threshold is selected on "
        f"validation predictions (`{config.get('threshold_strategy')}`) and frozen "
        "before the test partition is scored. The 0.5 default is also reported but "
        "never used for selection.",
        f"- **Selection criterion:** `{config.get('selection_metric')}` measured on "
        "validation, with the optional recall floor "
        f"`{config.get('selection_min_recall')}`. Test metrics are not an input.",
        "- **Test isolation:** the test matrix was evaluated once "
        f"(`test_evaluation_calls = {(metrics.get('validity') or {}).get('test_evaluation_calls')}`); "
        "no threshold, hyperparameter or model choice used test labels.",
        f"- **Cross-validation:** {metrics.get('cross_validation', {}).get('status')} "
        f"({metrics.get('cross_validation', {}).get('folds')} folds on train+validation"
        + (
            ", preprocessing refitted per fold)."
            if (metrics.get("cross_validation", {}).get("preprocessing") or {}).get(
                "fold_safe"
            )
            else ")."
        ),
        "",
        "## 2. Model configurations",
        "",
        *_family_config_table(metrics),
        "",
        "Hyperparameters were fixed in advance from the architecture document and were "
        "not searched on the test set. The MLP configuration is the one recorded in its "
        "Phase 6 metadata.",
        "",
        "## 3. Metric definitions and positive class",
        "",
        "The positive class is **1 = churned** (Phase 4/5 target mapping; source label "
        f"`{(dataset.get('target') or {}).get('positive_label')}`).",
        "",
        "- `accuracy = (TP + TN) / N`",
        "- `precision = TP / (TP + FP)` — of customers flagged for retention, the share "
        "who would actually churn",
        "- `recall = TP / (TP + FN)` — of customers who churn, the share flagged",
        "- `F1 = 2 · precision · recall / (precision + recall)`",
        "- `ROC-AUC` and `PR-AUC` are threshold-free ranking metrics; PR-AUC here is "
        "average precision for the positive (churn) class",
        "- `Brier = mean((p − y)²)` — lower is better; a calibration-oriented summary",
        "",
        "Precision, recall, F1 and AUC are reported as `n/a` when their denominator is "
        "undefined (for example no predicted positives, or a single-class partition) "
        "rather than being replaced with a misleading zero. Macro precision/recall/F1 "
        "are also stored in `metrics.json` for both classes.",
        "",
        "## 4. Threshold selection",
        "",
        "Thresholds are chosen from validation predictions only. ROC-AUC ranks across "
        "all thresholds; the operating threshold is a separate decision. A statistically "
        "optimal threshold is not automatically the best business decision — contact "
        "costs and capacity belong to later phases.",
        "",
        "| Family | Strategy | Objective | Threshold | Validation precision | Validation recall | Validation F1 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for family in completed:
        block = families[family]
        threshold = block.get("threshold") or {}
        validation = block.get("validation_metrics") or {}
        lines.append(
            f"| {FAMILY_LABELS[family]} | {threshold.get('strategy')} | "
            f"{threshold.get('objective')} | {_fmt(threshold.get('selected'), 3)} | "
            f"{_fmt(validation.get('precision'))} | {_fmt(validation.get('recall'))} | "
            f"{_fmt(validation.get('f1'))} |"
        )
    if not completed:
        lines.append("| — | — | — | — | — | — | — |")
    lines.append("")
    for family in completed:
        for note in (families[family].get("threshold") or {}).get("notes") or []:
            lines.append(f"- {FAMILY_LABELS[family]}: {note}")
    lines.extend(
        [
            "",
            "Threshold sweeps and the frozen choice are stored in "
            "[threshold_analysis.json](threshold_analysis.json) and plotted in "
            "`threshold_analysis.png`.",
            "",
            "## 5. Held-out test results",
            "",
            "Positive-class metrics at each family's frozen threshold:",
            "",
            *_main_table(metrics),
            "",
            "Confusion-matrix counts reconcile to the number of test rows "
            f"({dataset.get('n_test')}).",
            "",
            "Secondary view at the default 0.5 threshold (not used for selection):",
            "",
            *_default_threshold_table(metrics),
            "",
            "### Cross-validation stability",
            "",
            *_cv_table(metrics),
            "",
            "CV values are means ± standard deviations across folds drawn from "
            "train+validation rows at the 0.5 reference threshold. They are a stability "
            "estimate, not the primary evidence.",
            "",
            _cv_preprocessing_sentence(metrics),
            "",
            "## 6. Figures",
            "",
            "| Figure | File |",
            "| --- | --- |",
            "| Confusion matrix per family | "
            + ", ".join(
                f"[{FAMILY_LABELS[family]}](confusion_matrix_{family}.png)"
                for family in completed
            )
            + " |",
            "| ROC curves | [roc_curves.png](roc_curves.png) |",
            "| Precision-recall curves | [precision_recall_curves.png](precision_recall_curves.png) |",
            "| Metric comparison | [metric_comparison.png](metric_comparison.png) |",
            "| Threshold analysis | [threshold_analysis.png](threshold_analysis.png) |",
            "| Probability distributions | [probability_distributions.png](probability_distributions.png) |",
            "",
            "## 7. Model selection",
            "",
            f"**Selected family:** {selection.get('selected_label') or 'none'} "
            f"(via validation `{selection.get('selection_metric')}`, "
            f"threshold {_fmt(selection.get('threshold'), 3)}).",
            "",
            selection.get("rationale", ""),
            "",
            *[f"- {note}" for note in selection.get("notes") or []],
            "",
            "| Rank | Family | Validation metric | Validation recall | Validation ROC-AUC | Test metric |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in selection.get("ranking") or []:
        lines.append(
            f"| {row.get('rank', '—')} | {row.get('label')} | {_fmt(row.get('validation_value'))} | "
            f"{_fmt(row.get('validation_recall'))} | {_fmt(row.get('validation_roc_auc'))} | "
            f"{_fmt(row.get('test_value'))} |"
        )
    lines.extend(
        [
            "",
            "The `Test metric` column is shown for transparency only; it did not "
            "participate in the choice.",
            "",
            "## 8. Error analysis and practical interpretation",
            "",
            (metrics.get("error_analysis") or {}).get("interpretation", ""),
            "",
            "| Family | False positives | False negatives | FP rate of negatives | FN rate of positives | Mean P(FP) | Mean P(FN) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    error_families = (metrics.get("error_analysis") or {}).get("families") or {}
    for family in completed:
        row = error_families.get(family) or {}
        lines.append(
            f"| {FAMILY_LABELS[family]} | {row.get('false_positives', 'n/a')} | "
            f"{row.get('false_negatives', 'n/a')} | "
            f"{_pct(row.get('false_positive_rate_of_negatives'))} | "
            f"{_pct(row.get('false_negative_rate_of_positives'))} | "
            f"{_fmt(row.get('mean_probability_false_positives'))} | "
            f"{_fmt(row.get('mean_probability_false_negatives'))} |"
        )
    lines.extend(
        [
            "",
            "No customer identifiers or raw customer records are included in this "
            "report; the analysis is aggregate.",
            "",
            "### Observed strengths and weaknesses",
            "",
            *_strengths_and_weaknesses(metrics),
            "",
            "### Overfitting signals",
            "",
            "| Family | Train F1 | Validation F1 | Test F1 | Train−val ROC-AUC |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for family in completed:
        block = families[family]
        train = block.get("train_metrics") or {}
        validation = block.get("validation_metrics") or {}
        test = block.get("test_metrics") or {}
        gap = (block.get("generalisation") or {}).get(
            "train_roc_auc_minus_validation_roc_auc"
        )
        lines.append(
            f"| {FAMILY_LABELS[family]} | {_fmt(train.get('f1'))} | "
            f"{_fmt(validation.get('f1'))} | {_fmt(test.get('f1'))} | {_fmt(gap)} |"
        )
    lines.extend(
        [
            "",
            "Train scores are in-sample and are shown only to surface overfitting; they "
            "are not performance evidence.",
            "",
            "## 9. Limitations and warnings",
            "",
            "".join(f"- {item}\n" for item in metrics.get("limitations") or []).rstrip(),
            "",
        ]
    )
    warnings = result.warnings or metrics.get("warnings") or []
    if warnings:
        lines.append("Run warnings:")
        lines.append("")
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(
        [
            "## 10. Reproducibility and artifacts",
            "",
            "```bash",
            "python -m churnwise.cli evaluate \\",
            f"  --preprocessing-run-id {preprocessing.get('run_id', '<run>')} \\",
            f"  --mlp-run-id {(families.get(MLP_FAMILY, {}).get('model') or {}).get('model_run_id', '<mlp-run>')}",
            "```",
            "",
            "| Artifact | Path |",
            "| --- | --- |",
            f"| Structured metrics | `{(metrics.get('artifacts') or {}).get('metrics', 'metrics.json')}` |",
            f"| Curve data | `{(metrics.get('artifacts') or {}).get('curves', 'curves.json')}` |",
            f"| Threshold analysis | `{(metrics.get('artifacts') or {}).get('threshold_analysis', 'threshold_analysis.json')}` |",
            f"| Saved predictions | `{(metrics.get('artifacts') or {}).get('predictions', 'predictions.npz')}` |",
            f"| Run manifest | `{(metrics.get('artifacts') or {}).get('run_manifest', 'run_manifest.json')}` |",
            "",
            "`python -m churnwise.cli evaluate --regenerate "
            f"{result.benchmark_run_id}` recomputes every metric and figure from the "
            "saved probabilities without refitting a model or rescoring test rows.",
            "",
            "Environment: "
            + ", ".join(
                f"{key} {value}"
                for key, value in (metrics.get("environment") or {}).items()
                if key in {"python", "numpy", "pandas", "scikit_learn", "tensorflow", "keras"}
            ),
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def render_summary(result: BenchmarkResult) -> str:
    """Short console summary; the markdown report carries the full evidence."""
    metrics = result.metrics
    dataset = metrics.get("dataset") or {}
    selection = metrics.get("selection") or {}
    lines = [
        "ChurnWise AI - Phase 7 benchmark",
        "",
        f"Benchmark run: {result.benchmark_run_id}",
        f"Preprocessing run: {(metrics.get('preprocessing') or {}).get('run_id', '')}",
        f"Dataset: {dataset.get('dataset_id', '')} "
        f"(test rows {dataset.get('n_test')}, churn {_pct(dataset.get('test_positive_rate'))})",
        f"Families completed: {', '.join(_completed(metrics)) or 'none'}",
        "",
        "Held-out metrics at the frozen threshold:",
    ]
    for row in metrics.get("comparison_table") or []:
        lines.append(
            f"  {row.get('label'):<24} threshold {_fmt(row.get('threshold'), 3)}  "
            f"acc {_fmt(row.get('accuracy'), 3)}  precision {_fmt(row.get('precision'), 3)}  "
            f"recall {_fmt(row.get('recall'), 3)}  F1 {_fmt(row.get('f1'), 3)}  "
            f"ROC-AUC {_fmt(row.get('roc_auc'), 3)}"
        )
    lines.extend(
        [
            "",
            f"Selected model: {selection.get('selected_label') or 'none'} "
            f"(validation {selection.get('selection_metric')}"
            + (
                f" {_fmt(selection.get('selected_validation_value'), 4)})"
                if selection.get("selected_validation_value") is not None
                else ")"
            ),
            f"Test evaluation calls: {(metrics.get('validity') or {}).get('test_evaluation_calls')}",
        ]
    )
    if not metrics.get("complete"):
        lines.append(
            "Warning: incomplete benchmark; missing families: "
            + ", ".join(metrics.get("missing_families") or [])
        )
    for warning in result.warnings:
        lines.append(f"Warning: {warning}")
    if result.paths.get("markdown"):
        lines.append(f"Report: {result.paths['markdown']}")
    return "\n".join(lines)


__all__ = ["render_benchmark_markdown", "render_summary"]
