"""Role-driven column planning and the shared ColumnTransformer.

The plan is computed from the training partition only. It records numeric,
boolean and categorical columns, numeric-looking string coercions, manual
missing indicators and any column dropped for high cardinality. The fitted
transformer is serialisable and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnwise.preprocessing.contracts import PreprocessConfig
from churnwise.preprocessing.errors import CODE_TRANSFORM_FAILED, PreprocessingError

MISSING_CATEGORY = "__missing__"
NUMERIC_NAME_PATTERN = re.compile(
    r"(charge|amount|price|cost|total|balance|revenue|spend|salary|income|minutes|"
    r"usage|calls|tenure|age|quantity|qty|count|score|rate|value)",
    re.IGNORECASE,
)
NOMINAL_NAME_PATTERN = re.compile(
    r"(^|_)(code|zip|postal|phone|area|state|region|country|city|id)($|_)",
    re.IGNORECASE,
)

BOOL_VALUES = {
    "true": 1.0,
    "false": 0.0,
    "yes": 1.0,
    "no": 0.0,
    "1": 1.0,
    "0": 0.0,
    "t": 1.0,
    "f": 0.0,
}


@dataclass
class TypePlan:
    numeric: list[str] = field(default_factory=list)
    boolean: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    numeric_coerced: list[str] = field(default_factory=list)
    missing_indicators: list[str] = field(default_factory=list)
    dropped: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def indicator_columns(self) -> list[str]:
        return [f"{name}__missing" for name in self.missing_indicators]

    @property
    def feature_columns(self) -> list[str]:
        return [
            *self.numeric,
            *self.indicator_columns,
            *self.boolean,
            *self.categorical,
        ]


def _parse_numeric_text(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"^[\$€£]\s*", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _is_boolean(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    labels = series.dropna().astype(str).str.strip().str.lower()
    return bool(len(labels)) and labels.isin({key for key in BOOL_VALUES if not key.isdigit()}).all()


def _should_coerce(series: pd.Series, name: str, parsed: pd.Series, threshold: float) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    observed = series.notna().sum()
    if observed == 0:
        return False
    if float(parsed.notna().sum()) / float(observed) < threshold:
        return False
    if parsed.dropna().nunique() < 2:
        return False
    if NOMINAL_NAME_PATTERN.search(name):
        return False
    raw = series.dropna().astype(str)
    decimal = bool(raw.str.contains(r"\d\.\d", regex=True).any())
    return bool(NUMERIC_NAME_PATTERN.search(name)) or decimal


def plan_features(
    train_frame: pd.DataFrame,
    profile: Any,
    config: PreprocessConfig,
    derived_specs: list[dict[str, Any]],
) -> TypePlan:
    """Derive the column plan from the training partition only."""
    plan = TypePlan()
    roles = profile.roles
    target = profile.target.column
    id_column = roles.id
    dropped_names = {str(item["column"]) for item in roles.dropped}
    base_numeric = [str(name) for name in roles.numeric]
    base_categorical = [str(name) for name in roles.categorical]
    derived_numeric = [spec["name"] for spec in derived_specs if spec["dtype"] == "numeric"]
    derived_categorical = [spec["name"] for spec in derived_specs if spec["dtype"] == "categorical"]

    for name in [*base_numeric, *derived_numeric]:
        if name not in train_frame.columns:
            plan.dropped.append({"column": name, "reason": "column is absent from the training frame"})
            continue
        if name == target or name == id_column:
            continue
        if name in dropped_names:
            plan.dropped.append({"column": name, "reason": "rejected by Phase 4 validation"})
            continue
        if train_frame[name].notna().sum() == 0:
            plan.dropped.append({"column": name, "reason": "no observed values in the training split"})
            continue
        plan.numeric.append(name)

    for name in [*base_categorical, *derived_categorical]:
        if name not in train_frame.columns:
            plan.dropped.append({"column": name, "reason": "column is absent from the training frame"})
            continue
        if name == target or name == id_column:
            continue
        if name in dropped_names:
            plan.dropped.append({"column": name, "reason": "rejected by Phase 4 validation"})
            continue
        observed = train_frame[name].notna().sum()
        if observed == 0:
            plan.dropped.append({"column": name, "reason": "no observed values in the training split"})
            continue
        series = train_frame[name]
        if _is_boolean(series):
            plan.boolean.append(name)
            continue
        if _should_coerce(series, name, _parse_numeric_text(series), config.numeric_coerce_threshold):
            plan.numeric.append(name)
            plan.numeric_coerced.append(name)
            plan.warnings.append(
                f"column {name!r} was reliable numeric text on the training split and was coerced to numeric"
            )
            continue
        levels = int(series.nunique(dropna=True))
        if levels > config.max_categorical_levels:
            plan.dropped.append(
                {
                    "column": name,
                    "reason": (
                        f"{levels} training categories exceeds the configured "
                        f"{config.max_categorical_levels}-level limit"
                    ),
                }
            )
            plan.warnings.append(f"column {name!r} was excluded as high-cardinality ({levels} levels)")
            continue
        plan.categorical.append(name)

    if not plan.numeric and not plan.categorical and not plan.boolean:
        raise PreprocessingError(
            CODE_TRANSFORM_FAILED,
            "no usable feature columns remain after applying the Phase 4 roles",
            "Inspect reports/validation and the dataset profile before preprocessing.",
        )
    return plan


def clean_frame(frame: pd.DataFrame, plan: TypePlan) -> pd.DataFrame:
    """Apply the plan's conversions and return columns in a stable order."""
    required = [
        *plan.numeric,
        *plan.boolean,
        *plan.categorical,
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise PreprocessingError(
            CODE_TRANSFORM_FAILED,
            f"required columns are missing: {', '.join(missing)}",
            "Re-run inference with every column recorded in the preprocessing schema.",
        )

    output = pd.DataFrame(index=frame.index)
    for name in plan.numeric:
        if pd.api.types.is_numeric_dtype(frame[name]):
            values = pd.to_numeric(frame[name], errors="coerce")
        else:
            values = _parse_numeric_text(frame[name])
        output[name] = values.astype("float64").replace([np.inf, -np.inf], np.nan)

    for name in plan.boolean:
        labels = frame[name]
        if pd.api.types.is_bool_dtype(labels):
            output[name] = labels.astype("float64")
        else:
            output[name] = labels.astype("string").str.strip().str.lower().map(BOOL_VALUES)
        output[name] = output[name].replace([np.inf, -np.inf], np.nan)

    for name in plan.categorical:
        values = frame[name].astype("object")
        values = values.map(lambda value: value.strip() if isinstance(value, str) else value)
        output[name] = values.where(~pd.isna(values), np.nan)

    for source in plan.missing_indicators:
        # The indicator must describe the *cleaned* value: infinities and
        # unparseable numeric text become missing here, not in the raw frame.
        output[f"{source}__missing"] = output[source].isna().astype("float64")

    order = plan.feature_columns
    return output.loc[:, order]


def drop_unusable_columns(plan: TypePlan, cleaned: pd.DataFrame) -> list[str]:
    """Remove columns with no usable value after cleaning (for example all-inf)."""
    removed: list[str] = []
    for group in (plan.numeric, plan.boolean, plan.categorical):
        for name in list(group):
            if name in cleaned.columns and cleaned[name].notna().sum() == 0:
                group.remove(name)
                removed.append(name)
                if name in plan.numeric_coerced:
                    plan.numeric_coerced.remove(name)
                plan.dropped.append(
                    {
                        "column": name,
                        "reason": "no usable values remain after cleaning the training split",
                    }
                )
    if removed and not (plan.numeric or plan.boolean or plan.categorical):
        raise PreprocessingError(
            CODE_TRANSFORM_FAILED,
            "no usable feature columns remain after cleaning the training split",
            "Inspect the validation report and the offending columns; acquire a better dataset.",
        )
    return removed


def numeric_quality(
    raw: pd.DataFrame, cleaned: pd.DataFrame, numeric: list[str]
) -> dict[str, dict[str, Any]]:
    """Report non-null numeric values that cleaning had to turn into missing."""
    report: dict[str, dict[str, Any]] = {}
    for name in numeric:
        if name not in raw.columns or name not in cleaned.columns:
            continue
        source = raw[name]
        observed = int(source.notna().sum())
        if observed == 0:
            continue
        parsed = (
            pd.to_numeric(source, errors="coerce")
            if pd.api.types.is_numeric_dtype(source)
            else _parse_numeric_text(source)
        )
        unparseable = int(observed - parsed.notna().sum())
        infinite = int(np.isinf(parsed).sum())
        invalid = int(observed - cleaned[name].notna().sum())
        if invalid <= 0:
            continue
        report[name] = {
            "invalid": invalid,
            "pct": invalid / float(observed),
            "unparseable": unparseable,
            "infinite": infinite,
            "treatment": "converted to missing and imputed from the training split",
        }
    return report


def plan_missing_indicators(
    cleaned_train: pd.DataFrame,
    plan: TypePlan,
    config: PreprocessConfig,
    *,
    derived_names: set[str] | None = None,
) -> list[str]:
    """Add indicators for columns with more than the configured missing share."""
    derived = derived_names or set()
    indicators = [
        name
        for name in [*plan.numeric, *plan.boolean]
        if name not in derived
        and float(cleaned_train[name].isna().mean()) > config.missing_indicator_threshold
    ]
    plan.missing_indicators = indicators
    return indicators


def _numeric_steps() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def _boolean_steps() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )


def _categorical_steps(config: PreprocessConfig) -> Pipeline:
    if config.min_frequency > 0:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=config.min_frequency,
            sparse_output=False,
            dtype=np.float64,
        )
    else:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float64,
        )
    if config.categorical_imputation == "most_frequent":
        imputer = SimpleImputer(strategy="most_frequent", missing_values=np.nan)
    else:
        imputer = SimpleImputer(
            strategy="constant",
            fill_value=MISSING_CATEGORY,
            missing_values=np.nan,
        )
    return Pipeline([("imputer", imputer), ("encoder", encoder)])


def build_preprocessor(plan: TypePlan, config: PreprocessConfig) -> ColumnTransformer:
    """Assemble the shared ColumnTransformer from the training-only plan."""
    transformers: list[tuple[str, Any, list[str]]] = []
    numeric_columns = [*plan.numeric, *plan.indicator_columns]
    if numeric_columns:
        transformers.append(("numeric", _numeric_steps(), numeric_columns))
    if plan.boolean:
        transformers.append(("boolean", _boolean_steps(), plan.boolean))
    if plan.categorical:
        transformers.append(("categorical", _categorical_steps(config), plan.categorical))
    return ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=True,
    )


def transform_cleaned(
    preprocessor: ColumnTransformer,
    cleaned: pd.DataFrame,
    *,
    expected_width: int | None = None,
) -> np.ndarray:
    """Transform an already-cleaned frame; validates shape and finiteness."""
    matrix = np.asarray(preprocessor.transform(cleaned), dtype=np.float64)
    if matrix.ndim != 2:
        raise PreprocessingError(
            CODE_TRANSFORM_FAILED,
            f"transform produced a {matrix.ndim}-dimensional array",
            "Inspect the fitted ColumnTransformer; it must produce a 2-D matrix.",
        )
    if expected_width is not None and matrix.shape[1] != expected_width:
        raise PreprocessingError(
            CODE_TRANSFORM_FAILED,
            f"transform produced {matrix.shape[1]} columns, schema expects {expected_width}",
            "Reload the preprocessor and schema from the same run_id.",
        )
    if not np.isfinite(matrix).all():
        raise PreprocessingError(
            CODE_TRANSFORM_FAILED,
            "transform produced non-finite values after imputation",
            "Inspect the offending columns; all-missing training columns must be excluded.",
        )
    return matrix


def transform_frame(
    preprocessor: ColumnTransformer,
    frame: pd.DataFrame,
    plan: TypePlan,
    *,
    expected_width: int | None = None,
) -> np.ndarray:
    """Clean then transform a raw frame (used by tests and callers)."""
    return transform_cleaned(
        preprocessor, clean_frame(frame, plan), expected_width=expected_width
    )


def plan_from_schema(schema: dict[str, Any]) -> TypePlan:
    """Rebuild the inference plan recorded in a persisted schema."""
    block = schema.get("preprocessing", {}) or {}
    return TypePlan(
        numeric=list(block.get("numeric") or []),
        boolean=list(block.get("boolean") or []),
        categorical=list(block.get("categorical") or []),
        numeric_coerced=list(block.get("numeric_coerced") or []),
        missing_indicators=list(block.get("missing_indicators") or []),
    )


def describe_outputs(names: list[str], plan: TypePlan) -> list[dict[str, Any]]:
    """Map every transformed column back to its source column."""
    outputs: list[dict[str, Any]] = []
    for name in names:
        prefix, _, rest = name.partition("__")
        source = rest
        value: str | None = None
        role = "unknown"
        if prefix in {"numeric", "boolean"}:
            if rest.endswith("__missing"):
                source = rest[: -len("__missing")]
                role = "missing_indicator"
            else:
                source = rest
                role = "numeric" if prefix == "numeric" else "boolean"
        elif prefix == "categorical":
            matches = [
                column
                for column in plan.categorical
                if rest == column or rest.startswith(f"{column}_")
            ]
            source = max(matches, key=len) if matches else rest
            value = rest[len(source) + 1 :] if rest.startswith(f"{source}_") else None
            role = "categorical"
        outputs.append(
            {
                "name": name,
                "source": source,
                "role": role,
                "value": value,
                "dtype": "float64",
            }
        )
    return outputs
