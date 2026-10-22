"""Command line entry point for the ChurnWise pipeline.

Usage:
    python -m churnwise.cli discover
    python -m churnwise.cli acquire --offline
    python -m churnwise.cli validate --offline --json
    python -m churnwise.cli prepare
    python -m churnwise.cli train
    python -m churnwise.cli predict --input customers.csv
    python -m churnwise.cli evaluate --cv-folds 5
    python -m churnwise.cli evaluate --regenerate <benchmark-run-id>
    python -m churnwise.cli score --population val
Exit codes: 0 = success (a candidate was selected / an artifact was acquired /
validation passed), 2 = structured failure or rejection, 3 = invalid
configuration or report write failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from churnwise.acquisition.coordinator import AcquisitionCoordinator
from churnwise.acquisition.inputs import load_candidates
from churnwise.acquisition.reporting import render_summary as render_acquisition_summary
from churnwise.acquisition.reporting import write_attempts
from churnwise.config import Settings
from churnwise.contracts import SearchQuery
from churnwise.discovery import build_adapters, discover
from churnwise.discovery.coordinator import CODE_INVALID_CONFIG
from churnwise.discovery.reporting import render_summary, write_reports
from churnwise.logging_utils import configure_logging
from churnwise.preprocessing import (
    PreprocessConfig,
    PreprocessingCoordinator,
    PreprocessingError,
)
from churnwise.preprocessing.reporting import render_summary as render_preprocessing_summary
from churnwise.validation import ValidationCoordinator, ValidationResult, ValidationStatus
from churnwise.validation.reporting import render_summary as render_validation_summary

DEFAULTS = SearchQuery()


def _add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    """Discovery and validation options shared by validate and prepare."""
    parser.add_argument("--term", action="append", dest="terms", metavar="TERM")
    parser.add_argument("--min-rows", type=int, default=DEFAULTS.min_rows)
    parser.add_argument("--limit", type=int, default=DEFAULTS.max_results_per_source)
    parser.add_argument("--enrich-limit", type=int, default=DEFAULTS.enrich_limit)
    parser.add_argument("--timeout", type=float, default=DEFAULTS.timeout_s)
    parser.add_argument("--retries", type=int, default=DEFAULTS.max_retries)
    parser.add_argument(
        "--max-requests", type=int, default=DEFAULTS.max_requests_per_source
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=3,
        help="maximum candidates to try before giving up (default: 3)",
    )
    parser.add_argument(
        "--refresh-discovery",
        action="store_true",
        help="run Phase 2 discovery even when a candidates report already exists",
    )
    parser.add_argument(
        "--allow-unverified-license",
        action="store_true",
        help="allow acquisition from candidates whose licence is unverified",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use only verified cached artifacts (no network)",
    )
    parser.add_argument(
        "--data-dir", metavar="DIR", help="override the managed dataset directory"
    )
    parser.add_argument(
        "--report-dir", metavar="DIR", help="override the report directory (default: reports)"
    )
    parser.add_argument(
        "--json", action="store_true", dest="as_json", help="print the full result"
    )
    parser.add_argument("--verbose", action="store_true", help="debug-level structured logs")
    parser.add_argument("--log-file", metavar="PATH", help="also write structured JSONL logs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="churnwise",
        description="ChurnWise AI — autonomous customer churn dataset discovery",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser(
        "discover",
        help="search configured public sources for a customer churn dataset",
        description="Search configured public sources and select a recommended candidate.",
    )
    discover_parser.add_argument(
        "--term",
        action="append",
        dest="terms",
        metavar="TERM",
        help="search term; repeatable (default: churn, customer churn, attrition)",
    )
    discover_parser.add_argument("--min-rows", type=int, default=DEFAULTS.min_rows)
    discover_parser.add_argument(
        "--max-rows", type=int, default=DEFAULTS.max_rows, help="optional upper size bound"
    )
    discover_parser.add_argument(
        "--min-predictors",
        type=int,
        default=DEFAULTS.min_predictors,
        help="minimum predictor columns required (default: 5)",
    )
    discover_parser.add_argument(
        "--require-numeric",
        action="store_true",
        help="require explicit evidence of a numeric feature",
    )
    discover_parser.add_argument(
        "--require-categorical",
        action="store_true",
        help="require explicit evidence of a categorical feature",
    )
    discover_parser.add_argument(
        "--require-customer-features",
        action="store_true",
        help="require evidence of a customer-related feature column",
    )
    discover_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULTS.max_results_per_source,
        help="maximum results requested per search term",
    )
    discover_parser.add_argument(
        "--enrich-limit", type=int, default=DEFAULTS.enrich_limit, help="maximum datasets enriched per source"
    )
    discover_parser.add_argument("--timeout", type=float, default=DEFAULTS.timeout_s)
    discover_parser.add_argument("--retries", type=int, default=DEFAULTS.max_retries)
    discover_parser.add_argument(
        "--max-requests", type=int, default=DEFAULTS.max_requests_per_source
    )
    discover_parser.add_argument(
        "--allow-unverified-license",
        action="store_true",
        help="accept candidates whose licence metadata is missing or unverified",
    )
    discover_parser.add_argument(
        "--offline", action="store_true", help="force offline mode (discovery then fails fast)"
    )
    discover_parser.add_argument(
        "--report-dir", metavar="DIR", help="override the report directory (default: reports)"
    )
    discover_parser.add_argument("--json", action="store_true", dest="as_json", help="print the full JSON report")
    discover_parser.add_argument("--verbose", action="store_true", help="debug-level structured logs")
    discover_parser.add_argument("--log-file", metavar="PATH", help="also write structured JSONL logs")

    acquire_parser = subparsers.add_parser(
        "acquire",
        help="acquire and verify the highest-ranked eligible dataset",
        description="Acquire a verified dataset and provenance manifest from Phase 2 candidates.",
    )
    acquire_parser.add_argument("--term", action="append", dest="terms", metavar="TERM")
    acquire_parser.add_argument("--min-rows", type=int, default=DEFAULTS.min_rows)
    acquire_parser.add_argument(
        "--limit", type=int, default=DEFAULTS.max_results_per_source
    )
    acquire_parser.add_argument("--enrich-limit", type=int, default=DEFAULTS.enrich_limit)
    acquire_parser.add_argument("--timeout", type=float, default=DEFAULTS.timeout_s)
    acquire_parser.add_argument("--retries", type=int, default=DEFAULTS.max_retries)
    acquire_parser.add_argument(
        "--max-requests", type=int, default=DEFAULTS.max_requests_per_source
    )
    acquire_parser.add_argument(
        "--download-timeout",
        type=float,
        help="per-request download timeout in seconds (default: CHURNWISE_DOWNLOAD_TIMEOUT_S or 60)",
    )
    acquire_parser.add_argument(
        "--max-download-bytes",
        type=int,
        help="maximum artifact bytes (default: CHURNWISE_MAX_DOWNLOAD_BYTES or 200 MB)",
    )
    acquire_parser.add_argument(
        "--max-extracted-bytes",
        type=int,
        help="maximum extracted bytes (default: CHURNWISE_MAX_EXTRACTED_BYTES or 300 MB)",
    )
    acquire_parser.add_argument(
        "--max-archive-members",
        type=int,
        help="maximum archive members (default: CHURNWISE_MAX_ARCHIVE_MEMBERS or 200)",
    )
    acquire_parser.add_argument(
        "--refresh-discovery",
        action="store_true",
        help="run Phase 2 discovery even when a candidates report already exists",
    )
    acquire_parser.add_argument(
        "--allow-unverified-license",
        action="store_true",
        help="allow acquisition from candidates whose licence is unverified",
    )
    acquire_parser.add_argument(
        "--offline", action="store_true", help="use only a verified cached artifact"
    )
    acquire_parser.add_argument(
        "--data-dir", metavar="DIR", help="override the managed dataset directory"
    )
    acquire_parser.add_argument(
        "--report-dir", metavar="DIR", help="override the report directory (default: reports)"
    )
    acquire_parser.add_argument("--json", action="store_true", dest="as_json", help="print the full acquisition result")
    acquire_parser.add_argument("--verbose", action="store_true", help="debug-level structured logs")
    acquire_parser.add_argument("--log-file", metavar="PATH", help="also write structured JSONL logs")

    validate_parser = subparsers.add_parser(
        "validate",
        help="validate the acquired dataset and generate EDA intelligence",
        description=(
            "Verify, load, profile, gate and report the acquired dataset. "
            "Falls back to the next eligible candidate when a dataset is rejected."
        ),
    )
    _add_validation_arguments(validate_parser)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="split, fit and persist the leakage-safe preprocessing pipeline",
        description=(
            "Validate the acquired dataset when needed, then create reproducible "
            "train/validation/test splits, fit imputers/encoders/scalers on train "
            "only and persist the preprocessor, schema and matrices for Phase 6."
        ),
    )
    _add_validation_arguments(prepare_parser)
    prepare_parser.add_argument(
        "--artifact-dir", metavar="DIR", help="override the artifact directory (default: artifacts)"
    )
    prepare_parser.add_argument(
        "--seed", type=int, default=None, help="seed (default: CHURNWISE_SEED or 42)"
    )
    prepare_parser.add_argument("--train-fraction", type=float, default=0.60)
    prepare_parser.add_argument("--val-fraction", type=float, default=0.20)
    prepare_parser.add_argument("--test-fraction", type=float, default=0.20)
    prepare_parser.add_argument(
        "--imbalance",
        choices=("class_weight", "oversample", "none"),
        default="class_weight",
        help="imbalance strategy; class weights are the default",
    )
    prepare_parser.add_argument(
        "--oversample-ratio",
        type=float,
        default=1.0,
        help="target minority/majority ratio when --imbalance oversample is used",
    )
    prepare_parser.add_argument("--max-categorical-levels", type=int, default=30)
    prepare_parser.add_argument(
        "--min-frequency",
        type=float,
        default=0.0,
        help="merge rarer categories below this training share (0 disables)",
    )
    prepare_parser.add_argument(
        "--no-derived",
        action="store_true",
        help="disable capability-guarded derived features",
    )
    prepare_parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="do not drop exact duplicate rows before splitting",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="train the Phase 6 MLP on a persisted preprocessed run",
        description=(
            "Fit a small Keras MLP on the Phase 5 training split, early-stop on "
            "validation loss, and persist the model, metadata, history and curves."
        ),
    )
    train_parser.add_argument(
        "--preprocessing-run-id",
        metavar="RUN",
        help="Phase 5 run id (default: latest prepared run)",
    )
    train_parser.add_argument("--model-run-id", metavar="RUN", help="optional model run id")
    train_parser.add_argument("--epochs", type=int, default=150)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument(
        "--hidden",
        type=int,
        action="append",
        dest="hidden",
        metavar="UNITS",
        help="repeat for each hidden layer (default: 64 32)",
    )
    train_parser.add_argument(
        "--dropout",
        type=float,
        action="append",
        dest="dropout",
        metavar="RATE",
        help="one rate per hidden layer, or one rate for all layers",
    )
    train_parser.add_argument("--l2", type=float, default=1e-4)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--early-stopping-patience", type=int, default=12)
    train_parser.add_argument("--reduce-lr-patience", type=int, default=5)
    train_parser.add_argument("--reduce-lr-factor", type=float, default=0.5)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.add_argument("--seed", type=int, default=None)
    train_parser.add_argument(
        "--deterministic",
        action="store_true",
        help="enable TensorFlow op determinism (slower, more reproducible)",
    )
    train_parser.add_argument("--artifact-dir", metavar="DIR")
    train_parser.add_argument("--report-dir", metavar="DIR")
    train_parser.add_argument("--json", action="store_true", dest="as_json")
    train_parser.add_argument("--verbose", action="store_true", help="debug-level logs")
    train_parser.add_argument("--log-file", metavar="PATH")

    predict_parser = subparsers.add_parser(
        "predict",
        help="predict churn probabilities for raw customer records",
        description=(
            "Load a verified Phase 6 model, transform raw CSV records with the "
            "fitted Phase 5 pipeline, and write probabilities and classes."
        ),
    )
    predict_parser.add_argument(
        "--model-run-id", metavar="RUN", help="model run id (default: latest model)"
    )
    predict_parser.add_argument("--input", metavar="CSV", required=True)
    predict_parser.add_argument("--output", metavar="CSV", help="optional predictions CSV")
    predict_parser.add_argument("--threshold", type=float, default=None)
    predict_parser.add_argument("--artifact-dir", metavar="DIR")
    predict_parser.add_argument("--report-dir", metavar="DIR")
    predict_parser.add_argument("--json", action="store_true", dest="as_json")
    predict_parser.add_argument("--verbose", action="store_true", help="debug-level logs")
    predict_parser.add_argument("--log-file", metavar="PATH")

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="benchmark LR, RF and the Phase 6 MLP on the held-out test split",
        description=(
            "Fit the documented scikit-learn baselines, load the Phase 6 MLP, "
            "select thresholds on validation predictions and evaluate every "
            "family once on the untouched test partition."
        ),
    )
    evaluate_parser.add_argument(
        "--preprocessing-run-id",
        metavar="RUN",
        help="Phase 5 run id (default: latest prepared run)",
    )
    evaluate_parser.add_argument(
        "--mlp-run-id",
        metavar="RUN",
        help="Phase 6 model run id (default: latest MLP trained on this run)",
    )
    evaluate_parser.add_argument(
        "--benchmark-run-id", metavar="RUN", help="optional evaluation run id"
    )
    evaluate_parser.add_argument(
        "--regenerate",
        metavar="RUN",
        help="recompute metrics and figures from a saved benchmark's predictions",
    )
    evaluate_parser.add_argument(
        "--threshold-strategy",
        choices=("fixed", "max_f1", "max_youden", "recall_floor"),
        default="max_f1",
        help="validation-only threshold policy (default: max_f1)",
    )
    evaluate_parser.add_argument("--fixed-threshold", type=float, default=0.5)
    evaluate_parser.add_argument(
        "--recall-floor",
        type=float,
        default=None,
        help="minimum validation recall for --threshold-strategy recall_floor",
    )
    evaluate_parser.add_argument(
        "--selection-metric",
        choices=("f1", "roc_auc", "pr_auc", "recall", "balanced_accuracy", "accuracy"),
        default="f1",
        help="validation metric used to choose the shipped model (default: f1)",
    )
    evaluate_parser.add_argument(
        "--selection-min-recall",
        type=float,
        default=None,
        help="optional validation recall floor applied during model selection",
    )
    evaluate_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="stratified CV folds on train+validation (0 disables; default: 5)",
    )
    evaluate_parser.add_argument(
        "--no-cv", action="store_true", help="skip cross-validation"
    )
    evaluate_parser.add_argument("--seed", type=int, default=None)
    evaluate_parser.add_argument("--artifact-dir", metavar="DIR")
    evaluate_parser.add_argument("--report-dir", metavar="DIR")
    evaluate_parser.add_argument("--json", action="store_true", dest="as_json")
    evaluate_parser.add_argument("--verbose", action="store_true", help="debug-level logs")
    evaluate_parser.add_argument("--log-file", metavar="PATH")

    explain_parser = subparsers.add_parser(
        "explain",
        help="explain churn predictions with SHAP (Phase 8)",
        description=(
            "Load a verified Phase 6/7 model, compute global and customer-level "
            "SHAP explanations on validation rows, and persist JSON plus figures "
            "with full provenance. The test partition is never used."
        ),
    )
    explain_parser.add_argument(
        "--family",
        choices=("logistic_regression", "random_forest", "mlp"),
        default=None,
        help="model family to explain (default: the Phase 7 selected family)",
    )
    explain_parser.add_argument(
        "--benchmark-run-id",
        metavar="RUN",
        help="Phase 7 benchmark run (default: latest with metrics.json)",
    )
    explain_parser.add_argument(
        "--model-run-id", metavar="RUN", help="Phase 6 MLP run id (mlp only)"
    )
    explain_parser.add_argument(
        "--explain-run-id", metavar="RUN", help="optional explanation run id"
    )
    explain_parser.add_argument(
        "--input",
        metavar="CSV",
        help="explain raw customer rows from a CSV instead of a validation sample",
    )
    explain_parser.add_argument("--sample-size", type=int, default=None)
    explain_parser.add_argument("--background-size", type=int, default=None)
    explain_parser.add_argument("--mlp-sample-size", type=int, default=None)
    explain_parser.add_argument("--mlp-background-size", type=int, default=None)
    explain_parser.add_argument("--local-examples", type=int, default=None)
    explain_parser.add_argument("--top-k", type=int, default=None)
    explain_parser.add_argument("--seed", type=int, default=None)
    explain_parser.add_argument("--artifact-dir", metavar="DIR")
    explain_parser.add_argument("--report-dir", metavar="DIR")
    explain_parser.add_argument("--json", action="store_true", dest="as_json")
    explain_parser.add_argument("--verbose", action="store_true", help="debug-level logs")
    explain_parser.add_argument("--log-file", metavar="PATH")

    return parser


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    adapters=None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        resolved = settings or Settings.load()
    except ValueError as exc:
        print(f"error: invalid settings: {exc}", file=sys.stderr)
        return 3

    if args.command == "acquire":
        configure_logging(
            verbose=args.verbose,
            log_file=Path(args.log_file) if args.log_file else None,
        )
        return _run_acquire(args, resolved)

    if args.command == "validate":
        configure_logging(
            verbose=args.verbose,
            log_file=Path(args.log_file) if args.log_file else None,
        )
        return _run_validate(args, resolved)

    if args.command == "prepare":
        configure_logging(
            verbose=args.verbose,
            log_file=Path(args.log_file) if args.log_file else None,
        )
        return _run_prepare(args, resolved)

    if args.command == "train":
        configure_logging(
            verbose=args.verbose,
            log_file=Path(args.log_file) if args.log_file else None,
        )
        return _run_train(args, resolved)

    if args.command == "predict":
        configure_logging(
            verbose=args.verbose,
            log_file=Path(args.log_file) if args.log_file else None,
        )
        return _run_predict(args, resolved)

    if args.command == "evaluate":
        configure_logging(
            verbose=args.verbose,
            log_file=Path(args.log_file) if args.log_file else None,
        )
        return _run_evaluate(args, resolved)

    if args.command == "explain":
        configure_logging(
            verbose=args.verbose,
            log_file=Path(args.log_file) if args.log_file else None,
        )
        return _run_explain(args, resolved)


    overrides: dict[str, object] = {}
    if args.report_dir:
        overrides["report_dir"] = Path(args.report_dir)
    if args.offline:
        overrides["offline"] = True
    if overrides:
        resolved = resolved.with_overrides(**overrides)

    query = SearchQuery(
        terms=tuple(args.terms) if args.terms else DEFAULTS.terms,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_predictors=args.min_predictors,
        require_numeric_features=args.require_numeric,
        require_categorical_features=args.require_categorical,
        require_customer_features=args.require_customer_features,
        max_results_per_source=args.limit,
        enrich_limit=args.enrich_limit,
        timeout_s=args.timeout,
        max_retries=args.retries,
        max_requests_per_source=args.max_requests,
        allow_unverified_license=bool(
            args.allow_unverified_license or resolved.allow_unverified_license
        ),
    )

    configure_logging(
        verbose=args.verbose,
        log_file=Path(args.log_file) if args.log_file else None,
    )

    if adapters is None:
        adapters = build_adapters(resolved, query)

    report = discover(adapters, query, resolved)

    try:
        paths = write_reports(report, resolved.report_dir)
    except OSError as exc:
        print(f"error: could not write discovery reports: {exc}", file=sys.stderr)
        return 3

    if args.as_json:
        # ASCII-escaped on purpose: stdout may be a cp1252 console or redirected file.
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(render_summary(report, paths))

    if report.selected is not None:
        return 0
    if report.failure is not None and report.failure.code == CODE_INVALID_CONFIG:
        return 3
    return 2


def _run_acquire(args: argparse.Namespace, settings: Settings) -> int:
    overrides: dict[str, object] = {}
    if args.report_dir:
        overrides["report_dir"] = Path(args.report_dir)
    if args.data_dir:
        overrides["data_dir"] = Path(args.data_dir)
    if args.offline:
        overrides["offline"] = True
    if args.allow_unverified_license:
        overrides["allow_unverified_license"] = True
    if args.download_timeout is not None:
        overrides["download_timeout_s"] = args.download_timeout
    if args.max_download_bytes is not None:
        overrides["max_download_bytes"] = args.max_download_bytes
    if args.max_extracted_bytes is not None:
        overrides["max_extracted_bytes"] = args.max_extracted_bytes
    if args.max_archive_members is not None:
        overrides["max_archive_members"] = args.max_archive_members
    if overrides:
        settings = settings.with_overrides(**overrides)

    problems = settings.validate()
    if problems:
        print("error: invalid acquisition settings: " + "; ".join(problems), file=sys.stderr)
        return 3

    report_path = settings.report_dir / "acquisition" / "candidates.json"
    candidates = []
    if report_path.is_file() and not args.refresh_discovery:
        try:
            candidates = load_candidates(report_path)
        except (OSError, ValueError) as exc:
            print(f"error: could not read discovery report: {exc}", file=sys.stderr)
            return 3

    if not candidates and not settings.offline:
        query = SearchQuery(
            terms=tuple(args.terms) if args.terms else DEFAULTS.terms,
            min_rows=args.min_rows,
            max_results_per_source=args.limit,
            enrich_limit=args.enrich_limit,
            timeout_s=args.timeout,
            max_retries=args.retries,
            max_requests_per_source=args.max_requests,
            allow_unverified_license=settings.allow_unverified_license,
        )
        report = discover(build_adapters(settings, query), query, settings)
        try:
            write_reports(report, settings.report_dir)
        except OSError as exc:
            print(f"error: could not write discovery reports: {exc}", file=sys.stderr)
            return 3
        if report.selected is None:
            failure = report.failure
            print("discovery produced no acquirable candidate", file=sys.stderr)
            if failure:
                print(f"  {failure.code}: {failure.message}", file=sys.stderr)
                for reason in failure.reasons[:6]:
                    print(f"  - {reason}", file=sys.stderr)
            return 2
        candidates = load_candidates(report_path)

    coordinator = AcquisitionCoordinator(settings)
    result = coordinator.acquire(candidates, offline=settings.offline, origin="phase2_ranking")
    attempts_path = None
    try:
        attempts_path = write_attempts(result, settings.report_dir, offline=settings.offline)
    except OSError as exc:
        print(f"error: could not write acquisition attempts: {exc}", file=sys.stderr)
        return 3

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(render_acquisition_summary(result, attempts_path))
    return 0 if result.succeeded else 2


def _apply_common_overrides(args: argparse.Namespace, settings: Settings) -> Settings:
    overrides: dict[str, object] = {}
    if getattr(args, "report_dir", None):
        overrides["report_dir"] = Path(args.report_dir)
    if getattr(args, "data_dir", None):
        overrides["data_dir"] = Path(args.data_dir)
    if getattr(args, "artifact_dir", None):
        overrides["artifact_dir"] = Path(args.artifact_dir)
    if getattr(args, "offline", False):
        overrides["offline"] = True
    if getattr(args, "allow_unverified_license", False):
        overrides["allow_unverified_license"] = True
    return settings.with_overrides(**overrides) if overrides else settings


def _resolve_validation(
    args: argparse.Namespace, settings: Settings
) -> tuple[int, ValidationResult | None]:
    """Load candidates, discover when needed, and validate with fallback."""
    report_path = settings.report_dir / "acquisition" / "candidates.json"
    candidates = []
    if report_path.is_file() and not args.refresh_discovery:
        try:
            candidates = load_candidates(report_path)
        except (OSError, ValueError) as exc:
            print(f"error: could not read discovery report: {exc}", file=sys.stderr)
            return 3, None

    if not candidates and not settings.offline:
        query = SearchQuery(
            terms=tuple(args.terms) if args.terms else DEFAULTS.terms,
            min_rows=args.min_rows,
            max_results_per_source=args.limit,
            enrich_limit=args.enrich_limit,
            timeout_s=args.timeout,
            max_retries=args.retries,
            max_requests_per_source=args.max_requests,
            allow_unverified_license=settings.allow_unverified_license,
        )
        report = discover(build_adapters(settings, query), query, settings)
        try:
            write_reports(report, settings.report_dir)
        except OSError as exc:
            print(f"error: could not write discovery reports: {exc}", file=sys.stderr)
            return 3, None
        if report.selected is None:
            failure = report.failure
            print("discovery produced no validatable candidate", file=sys.stderr)
            if failure:
                print(f"  {failure.code}: {failure.message}", file=sys.stderr)
                for reason in failure.reasons[:6]:
                    print(f"  - {reason}", file=sys.stderr)
            return 2, None
        candidates = load_candidates(report_path)

    coordinator = ValidationCoordinator(settings)
    result = coordinator.validate_candidates(
        candidates,
        offline=settings.offline,
        max_candidates=args.max_candidates,
    )
    return 0, result


def _run_validate(args: argparse.Namespace, settings: Settings) -> int:
    settings = _apply_common_overrides(args, settings)
    problems = settings.validate()
    if problems:
        print("error: invalid validation settings: " + "; ".join(problems), file=sys.stderr)
        return 3
    if args.max_candidates < 1:
        print("error: --max-candidates must be at least 1", file=sys.stderr)
        return 3

    code, result = _resolve_validation(args, settings)
    if result is None:
        return code

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(render_validation_summary(result))

    if result.status is ValidationStatus.REJECT:
        return 2
    return 0


def _run_prepare(args: argparse.Namespace, settings: Settings) -> int:
    settings = _apply_common_overrides(args, settings)
    problems = settings.validate()
    if problems:
        print("error: invalid preprocessing settings: " + "; ".join(problems), file=sys.stderr)
        return 3
    if args.max_candidates < 1:
        print("error: --max-candidates must be at least 1", file=sys.stderr)
        return 3

    config = PreprocessConfig(
        seed=settings.seed if args.seed is None else args.seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        imbalance=args.imbalance,
        oversample_ratio=args.oversample_ratio,
        max_categorical_levels=args.max_categorical_levels,
        min_frequency=args.min_frequency,
        derived_features=not args.no_derived,
        drop_duplicate_rows=not args.keep_duplicates,
    )
    config_problems = config.validate()
    if config_problems:
        print(
            "error: invalid preprocessing configuration: " + "; ".join(config_problems),
            file=sys.stderr,
        )
        return 3

    code, validation = _resolve_validation(args, settings)
    if validation is None:
        return code
    if not validation.can_preprocess:
        if args.as_json:
            print(json.dumps(validation.to_dict(), indent=2))
        else:
            print(render_validation_summary(validation))
            print("error: Phase 4 did not approve this dataset for preprocessing", file=sys.stderr)
        return 2

    try:
        dataset = PreprocessingCoordinator(settings).prepare(validation, config=config)
    except PreprocessingError as exc:
        if args.as_json:
            print(
                json.dumps(
                    {"status": "error", "error": exc.to_dict()},
                    indent=2,
                )
            )
        else:
            print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
            if exc.remediation:
                print(f"  next: {exc.remediation}", file=sys.stderr)
        return 2

    if args.as_json:
        print(json.dumps(dataset.to_dict(), indent=2))
    else:
        print(render_preprocessing_summary(dataset))
    return 0


def _model_config(args: argparse.Namespace, settings: Settings):
    from churnwise.models.contracts import MLPConfig, coerce_dropout

    defaults = MLPConfig()
    hidden = tuple(args.hidden) if args.hidden else defaults.hidden_units
    if args.dropout:
        dropout = coerce_dropout(args.dropout, hidden)
    elif hidden == defaults.hidden_units:
        dropout = defaults.dropout
    else:
        dropout = tuple(0.2 for _ in hidden)
    return MLPConfig(
        seed=settings.seed if args.seed is None else args.seed,
        hidden_units=hidden,
        dropout=dropout,
        l2=args.l2,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        reduce_lr_patience=args.reduce_lr_patience,
        reduce_lr_factor=args.reduce_lr_factor,
        min_learning_rate=args.min_learning_rate,
        threshold=args.threshold,
        deterministic=bool(args.deterministic),
    )


def _latest_prepared_run(settings: Settings) -> str | None:
    root = Path(settings.artifact_dir) / "prepared"
    if not root.is_dir():
        return None
    candidates = sorted(
        path.name for path in root.iterdir() if (path / "summary.json").is_file()
    )
    return candidates[-1] if candidates else None


def _run_train(args: argparse.Namespace, settings: Settings) -> int:
    settings = _apply_common_overrides(args, settings)
    problems = settings.validate()
    if problems:
        print("error: invalid modeling settings: " + "; ".join(problems), file=sys.stderr)
        return 3

    from churnwise.models import MLPTrainer, ModelingError
    from churnwise.models.reporting import render_summary as render_training_summary

    try:
        config = _model_config(args, settings)
        config_problems = config.validate()
        if config_problems:
            print("error: invalid MLP configuration: " + "; ".join(config_problems), file=sys.stderr)
            return 3
    except ValueError as exc:
        print(f"error: invalid MLP configuration: {exc}", file=sys.stderr)
        return 3

    preprocessing_run_id = args.preprocessing_run_id or _latest_prepared_run(settings)
    if not preprocessing_run_id:
        print(
            "error: no prepared Phase 5 run was found; run the prepare command first",
            file=sys.stderr,
        )
        return 2

    try:
        dataset = PreprocessingCoordinator(settings).load(preprocessing_run_id)
        result = MLPTrainer(settings, config).train(
            dataset, model_run_id=args.model_run_id, persist=True
        )
    except (PreprocessingError, ModelingError) as exc:
        if args.as_json:
            print(json.dumps({"status": "error", "error": exc.to_dict()}, indent=2))
        else:
            print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
            if exc.remediation:
                print(f"  next: {exc.remediation}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not write model artifacts: {exc}", file=sys.stderr)
        return 3

    if args.as_json:
        print(
            json.dumps(
                {
                    "metadata": result.metadata,
                    "history": result.history,
                    "diagnostics": result.diagnostics,
                    "paths": result.paths,
                },
                indent=2,
            )
        )
    else:
        print(render_training_summary(result))
    return 0


def _run_predict(args: argparse.Namespace, settings: Settings) -> int:
    settings = _apply_common_overrides(args, settings)
    problems = settings.validate()
    if problems:
        print("error: invalid modeling settings: " + "; ".join(problems), file=sys.stderr)
        return 3
    if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
        print("error: --threshold must be in [0, 1]", file=sys.stderr)
        return 3

    import pandas as pd

    from churnwise.models import ChurnPredictor, ModelRegistry, ModelingError

    try:
        model_run_id = args.model_run_id or ModelRegistry(settings).latest_model_run_id()
        if not model_run_id:
            print("error: no trained model was found; run the train command first", file=sys.stderr)
            return 2
        predictor = ChurnPredictor.load(
            settings,
            model_run_id,
            threshold=args.threshold,
            with_preprocessing=True,
        )
    except ModelingError as exc:
        print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
        if exc.remediation:
            print(f"  next: {exc.remediation}", file=sys.stderr)
        return 2

    try:
        frame = pd.read_csv(args.input)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"error: input CSV could not be read: {exc}", file=sys.stderr)
        return 2
    if frame.empty:
        print("error: input CSV contains no rows", file=sys.stderr)
        return 2

    try:
        batch = predictor.predict_frame(frame)
    except ModelingError as exc:
        print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
        if exc.remediation:
            print(f"  next: {exc.remediation}", file=sys.stderr)
        return 2

    output = pd.DataFrame(
        {
            "row_index": batch.row_positions,
            "churn_probability": batch.probabilities,
            "churn_prediction": batch.classes,
        }
    )
    if args.output:
        try:
            target = Path(args.output)
            target.parent.mkdir(parents=True, exist_ok=True)
            output.to_csv(target, index=False)
        except OSError as exc:
            print(f"error: predictions could not be written: {exc}", file=sys.stderr)
            return 3

    if args.as_json:
        print(json.dumps(batch.to_dict(), indent=2))
    else:
        positives = int(batch.classes.sum())
        print(
            f"model {model_run_id}: {batch.n_rows} rows, threshold {batch.threshold:.3f}, "
            f"predicted churn {positives} ({positives / max(batch.n_rows, 1):.1%})"
        )
        if args.output:
            print(f"predictions: {args.output}")
    return 0


def _run_evaluate(args: argparse.Namespace, settings: Settings) -> int:
    settings = _apply_common_overrides(args, settings)
    problems = settings.validate()
    if problems:
        print("error: invalid evaluation settings: " + "; ".join(problems), file=sys.stderr)
        return 3

    from churnwise.evaluation import (
        BenchmarkCoordinator,
        EvaluationConfig,
        EvaluationError,
    )
    from churnwise.evaluation.reporting import render_summary as render_benchmark_summary

    if args.regenerate:
        try:
            regenerated = BenchmarkCoordinator(settings, None).regenerate(
                args.regenerate
            )
        except EvaluationError as exc:
            print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
            if exc.remediation:
                print(f"  next: {exc.remediation}", file=sys.stderr)
            return 2
        if args.as_json:
            print(json.dumps(regenerated.to_dict(), indent=2))
        else:
            print(regenerated.metrics.get("summary") or render_benchmark_summary(regenerated))
            print(f"regenerated from stored predictions: {regenerated.paths.get('metrics')}")
        return 0

    config = EvaluationConfig(
        seed=settings.seed if args.seed is None else args.seed,
        threshold_strategy=args.threshold_strategy,
        fixed_threshold=args.fixed_threshold,
        recall_floor=args.recall_floor,
        selection_metric=args.selection_metric,
        selection_min_recall=args.selection_min_recall,
        cv_folds=0 if args.no_cv else args.cv_folds,
    )
    config_problems = config.validate()
    if config_problems:
        print(
            "error: invalid evaluation configuration: " + "; ".join(config_problems),
            file=sys.stderr,
        )
        return 3

    preprocessing_run_id = args.preprocessing_run_id or _latest_prepared_run(settings)
    if not preprocessing_run_id:
        print(
            "error: no prepared Phase 5 run was found; run the prepare command first",
            file=sys.stderr,
        )
        return 2

    try:
        dataset = PreprocessingCoordinator(settings).load(preprocessing_run_id)
        result = BenchmarkCoordinator(settings, config).run(
            dataset,
            mlp_run_id=args.mlp_run_id,
            benchmark_run_id=args.benchmark_run_id,
            persist=True,
        )
    except (PreprocessingError, EvaluationError) as exc:
        if args.as_json:
            print(json.dumps({"status": "error", "error": exc.to_dict()}, indent=2))
        else:
            print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
            if exc.remediation:
                print(f"  next: {exc.remediation}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: could not write evaluation artifacts: {exc}", file=sys.stderr)
        return 3

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.metrics.get("summary") or render_benchmark_summary(result))
    return 0 if result.complete else 2


def _run_explain(args: argparse.Namespace, settings: Settings) -> int:
    settings = _apply_common_overrides(args, settings)
    problems = settings.validate()
    if problems:
        print("error: invalid explanation settings: " + "; ".join(problems), file=sys.stderr)
        return 3

    from churnwise.evaluation import EvaluationError
    from churnwise.explain import (
        CODE_INVALID_CONFIG,
        ExplainConfig,
        ExplainCoordinator,
        ExplainError,
    )
    from churnwise.explain.reporting import render_summary as render_explain_summary
    from churnwise.models import ModelingError
    from churnwise.preprocessing import PreprocessingError

    defaults = ExplainConfig(seed=settings.seed if args.seed is None else args.seed)
    config = ExplainConfig(
        sample_size=defaults.sample_size if args.sample_size is None else args.sample_size,
        background_size=(
            defaults.background_size if args.background_size is None else args.background_size
        ),
        mlp_sample_size=(
            defaults.mlp_sample_size if args.mlp_sample_size is None else args.mlp_sample_size
        ),
        mlp_background_size=(
            defaults.mlp_background_size
            if args.mlp_background_size is None
            else args.mlp_background_size
        ),
        local_examples=(
            defaults.local_examples if args.local_examples is None else args.local_examples
        ),
        top_k=defaults.top_k if args.top_k is None else args.top_k,
        seed=defaults.seed,
        tolerance=defaults.tolerance,
        chunk_size=defaults.chunk_size,
        max_local_batch=defaults.max_local_batch,
        permutation_max_evals=defaults.permutation_max_evals,
    )
    try:
        coordinator = ExplainCoordinator(settings, config)
        if args.input:
            import pandas as pd

            try:
                frame = pd.read_csv(args.input)
            except (OSError, ValueError) as exc:
                print(f"error: could not read {args.input}: {exc}", file=sys.stderr)
                return 3
            result = coordinator.explain_records(
                frame,
                family=args.family,
                benchmark_run_id=args.benchmark_run_id,
                model_run_id=args.model_run_id,
                explain_run_id=args.explain_run_id,
            )
        else:
            progress = (
                None
                if args.as_json
                else lambda message: print(f"  {message}", file=sys.stderr)
            )
            result = coordinator.run(
                args.family,
                benchmark_run_id=args.benchmark_run_id,
                model_run_id=args.model_run_id,
                explain_run_id=args.explain_run_id,
                progress=progress,
            )
    except (ExplainError, ModelingError, PreprocessingError, EvaluationError) as exc:
        if args.as_json:
            payload = exc.to_dict() if hasattr(exc, "to_dict") else {"message": str(exc)}
            print(json.dumps({"status": "error", "error": payload}, indent=2))
        else:
            print(f"error: {exc.code}: {exc.message}", file=sys.stderr)
            if getattr(exc, "remediation", ""):
                print(f"  next: {exc.remediation}", file=sys.stderr)
        return 3 if exc.code == CODE_INVALID_CONFIG else 2
    except OSError as exc:
        print(f"error: could not write explanation artifacts: {exc}", file=sys.stderr)
        return 3

    if args.as_json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(render_explain_summary(result))
    return 0 if result.complete else 2


def _risk_config(args: argparse.Namespace, settings: Settings):
    from churnwise.intelligence import RiskConfig, RiskThresholds

    boundaries = (
        args.risk_tiers
        if getattr(args, "risk_tiers", None)
        else settings.risk_tiers
    )
    config = RiskConfig(
        thresholds=RiskThresholds.from_value(boundaries),
        classification_threshold=getattr(args, "classification_threshold", None),
        batch_size=getattr(args, "batch_size", None) or 200,
        top_k=getattr(args, "top_k", None) or 5,
        max_driver_rows=(
            getattr(args, "max_driver_rows", None)
            if getattr(args, "max_driver_rows", None) is not None
            else 1000
        ),
        population_split="val",
        population_explain="all",
        value_column=getattr(args, "value_column", None),
        reference_column=getattr(args, "reference_column", None),
        segment_fields=tuple(getattr(args, "segment_fields", None) or ()),
    )
    config.validate_or_raise()
    return config


if __name__ == "__main__":
    raise SystemExit(main())
