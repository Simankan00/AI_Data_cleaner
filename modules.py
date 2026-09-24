from __future__ import annotations

import logging
import math
import re
import time
import unicodedata
import warnings
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Optional third-party imports.
# Imported defensively so a broken/missing wheel degrades a single feature
# instead of preventing the whole application from starting.
# -----------------------------------------------------------------------------
try:  # Fast C++ string similarity - preferred fuzzy matching backend.
    from rapidfuzz import fuzz as _rapidfuzz_fuzz

    _HAS_RAPIDFUZZ = True
except Exception:  # pragma: no cover - environment dependent
    from difflib import SequenceMatcher  # stdlib fallback (slower, pure Python)

    _rapidfuzz_fuzz = None
    _HAS_RAPIDFUZZ = False

try:  # PyOD - primary ML outlier detector.
    from pyod.models.iforest import IForest as _PyODIForest

    _HAS_PYOD = True
except Exception:  # pragma: no cover
    _PyODIForest = None
    _HAS_PYOD = False

try:  # scikit-learn is a hard dependency but we still guard the import.
    from sklearn.ensemble import IsolationForest as _SklearnIForest
    from sklearn.impute import KNNImputer, SimpleImputer
    from sklearn.preprocessing import StandardScaler

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _SklearnIForest = KNNImputer = SimpleImputer = StandardScaler = None
    _HAS_SKLEARN = False

# `great_expectations` and `dedupe` are imported LAZILY (inside the functions
# that need them) because both are slow to import - GX alone can add several
# seconds to Streamlit cold start.

__all__ = [
    # Exceptions
    "DataCleaningError",
    "CleaningStepError",
    "DependencyUnavailableError",
    # Infrastructure
    "configure_logging",
    "make_json_safe",
    "cleaning_step",
    # Profiling
    "profile_dataframe",
    "summarize_profile_for_llm",
    # Missing values
    "handle_missing_values",
    "normalize_missing_tokens",
    # Standardization
    "standardize_column_names",
    "standardize_text",
    "standardize_types",
    # Deduplication
    "drop_exact_duplicates",
    "fuzzy_deduplicate",
    # Outliers
    "detect_outliers_iqr",
    "detect_outliers_isolation_forest",
    "treat_outliers",
    # Validation
    "build_expectation_suite_from_profile",
    "validate_dataframe",
    "summarize_validation_for_llm",
    # Registry
    "CLEANING_FUNCTION_REGISTRY",
]

# -----------------------------------------------------------------------------
# Logging. Library best practice: attach a NullHandler and let the *application*
# (Step 3, the Streamlit entrypoint) decide on formatting and level.
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def configure_logging(level: int = logging.INFO) -> None:
    """Convenience helper for the app layer / notebooks.

    Parameters
    ----------
    level : int
        Standard ``logging`` level constant.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# SECTION 0 - Exceptions, constants, and low-level helpers
# =============================================================================


class DataCleaningError(Exception):
    """Base class for every error raised by this module."""


class CleaningStepError(DataCleaningError):
    """A cleaning step failed. Wraps the original exception via ``__cause__``."""


class DependencyUnavailableError(DataCleaningError):
    """A required optional dependency is not installed or is incompatible."""


# Tokens that mean "missing" in real-world exports but are not recognised by
# pandas. Compared case-insensitively after stripping whitespace.
NULL_SENTINELS: frozenset = frozenset(
    {
        "", "na", "n/a", "n.a.", "#n/a", "#na", "null", "none", "nil", "nan",
        "not available", "not applicable", "missing", "unknown", "undefined",
        "-", "--", "---", "?", "??", ".", "\\n", "<na>", "<null>", "(blank)",
    }
)

# Boolean-ish string encodings. Keys are lowercase + stripped.
TRUE_TOKENS: frozenset = frozenset({"true", "t", "yes", "y", "1", "1.0", "on", "si", "sim"})
FALSE_TOKENS: frozenset = frozenset({"false", "f", "no", "n", "0", "0.0", "off", "nao", "não"})

# Currency and thousands-separator noise stripped before numeric coercion.
_CURRENCY_SYMBOLS = "$€£¥₹₽¢₩₪₺R\\$"
_NUMERIC_NOISE_RE = re.compile(rf"[{_CURRENCY_SYMBOLS}\s,_]")
_ACCOUNTING_NEGATIVE_RE = re.compile(r"^\((.+)\)$")
_LEADING_ZERO_RE = re.compile(r"^0\d+$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_WS_RE = re.compile(r"\s+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z]+")

# Lightweight semantic-type detectors used by the profiler.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)
_PHONE_RE = re.compile(r"^\+?\(?\d[\d\s().-]{6,19}$")
# Dates match the phone pattern (digits + separators), so they are excluded
# explicitly. Without this, "2023-01-15" is classified as a phone number.
_DATE_LIKE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}$")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


def _safe_isna(value: Any) -> bool:
    """``pd.isna`` that never raises.

    ``pd.isna`` returns an *array* for list-likes, which blows up in a boolean
    context. Object columns holding lists/dicts are common in JSON-derived data,
    so every scalar null check in this module routes through here.
    """
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (np.ndarray, pd.Series)):
        return False
    return bool(result)


def _safe_nunique(series: pd.Series, dropna: bool = True) -> int:
    """``nunique`` that survives unhashable cell values (lists, dicts, sets).

    JSON-derived frames routinely contain list-valued cells. ``nunique`` hashes
    every value, so it raises ``TypeError`` on those columns. Returns ``-1`` to
    signal "cardinality unknown" rather than aborting an entire profile run.
    """
    try:
        return int(series.nunique(dropna=dropna))
    except TypeError:
        return -1


def _to_text(series: pd.Series) -> pd.Series:
    """Coerce any Series to pandas ``StringDtype`` while preserving nulls.

    ``astype(str)`` is deliberately avoided: it turns ``NaN`` into the literal
    string ``"nan"``, which silently corrupts null accounting downstream.
    """
    def _convert(value: Any) -> Any:
        if _safe_isna(value):
            return pd.NA
        if isinstance(value, str):
            return value
        return str(value)

    return series.map(_convert).astype("string")


def _ensure_dataframe(df: Any, step_name: str) -> None:
    """Guard clause: reject anything that is not a usable DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{step_name}: expected a pandas DataFrame, got {type(df).__name__}."
        )
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].unique().tolist()
        raise DataCleaningError(
            f"{step_name}: DataFrame has duplicate column names {dupes}. "
            "Run standardize_column_names() first - duplicate labels make "
            "column-wise selection ambiguous."
        )


def _resolve_columns(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]],
    default: Optional[Sequence[str]] = None,
) -> List[str]:
    """Validate a user-supplied column subset, or fall back to a default."""
    if columns is None:
        return list(default) if default is not None else list(df.columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not present in DataFrame: {missing}")
    return list(columns)


def make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars into JSON-serializable natives.

    Reports are persisted to disk and pushed into Chroma metadata, both of which
    reject ``np.int64``/``np.float32``/``Timestamp``. NaN and Inf become ``None``
    because ``json.dumps`` emits non-standard ``NaN`` literals otherwise.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)) and not isinstance(obj, np.generic):
        return obj
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, np.generic):
        return make_json_safe(obj.item())
    if isinstance(obj, (np.ndarray, pd.Index)):
        return [make_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {make_json_safe(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [make_json_safe(v) for v in obj]
    if obj is pd.NaT or obj is pd.NA:
        return None
    return str(obj)


def _pct(numerator: float, denominator: float, digits: int = 2) -> float:
    """Percentage helper that returns 0.0 instead of dividing by zero."""
    if not denominator:
        return 0.0
    return round(100.0 * numerator / denominator, digits)


def _set_nan(df: pd.DataFrame, mask: pd.Series, columns: Sequence[str]) -> None:
    """Write NaN into masked cells, widening integer/boolean columns first.

    pandas <3.0 silently upcast int64 -> float64 when you assigned NaN. pandas
    3.0 raises instead, so the upcast has to be explicit. Modifies ``df`` in
    place; callers always pass a copy they own.
    """
    for col in columns:
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            df[col] = df[col].astype("float64")
        df.loc[mask, col] = np.nan


# =============================================================================
# SECTION 1 - The @cleaning_step decorator (cross-cutting concerns)
# =============================================================================


def cleaning_step(step_name: str) -> Callable:
    """Wrap a cleaning function with timing, validation, logging and error policy.

    Keeping these concerns in one decorator means each function body contains
    only its actual transformation logic, and every report in the system shares
    the same envelope keys: ``step``, ``status``, ``timestamp``,
    ``duration_seconds``, ``input_shape``, ``output_shape``.

    The wrapped function must return ``(DataFrame, dict)``.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
            error_mode = kwargs.get("errors", "raise")
            if error_mode not in {"raise", "coerce"}:
                raise ValueError("`errors` must be either 'raise' or 'coerce'.")

            started = time.perf_counter()
            envelope: Dict[str, Any] = {
                "step": step_name,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }

            try:
                _ensure_dataframe(df, step_name)
                result = func(df, *args, **kwargs)

                # Defensive contract check - catches refactoring mistakes early.
                if not (isinstance(result, tuple) and len(result) == 2):
                    raise TypeError(
                        f"{step_name}: expected a (DataFrame, dict) tuple, "
                        f"got {type(result).__name__}."
                    )
                out_df, report = result
                if not isinstance(report, dict):
                    raise TypeError(f"{step_name}: report must be a dict.")

            except Exception as exc:
                elapsed = time.perf_counter() - started
                logger.error(
                    "Step '%s' failed after %.3fs: %s", step_name, elapsed, exc, exc_info=True
                )
                if error_mode == "raise":
                    raise CleaningStepError(f"Step '{step_name}' failed: {exc}") from exc

                # errors="coerce": return the input untouched so the pipeline
                # can continue, but make the failure loud in the report.
                envelope.update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "duration_seconds": round(elapsed, 4),
                        "input_shape": list(df.shape) if isinstance(df, pd.DataFrame) else None,
                        "output_shape": list(df.shape) if isinstance(df, pd.DataFrame) else None,
                    }
                )
                return df, envelope

            elapsed = time.perf_counter() - started
            merged: Dict[str, Any] = dict(envelope)
            merged["status"] = "success"
            merged.update(report)  # Body may override status (e.g. "skipped").
            merged["duration_seconds"] = round(elapsed, 4)
            merged["input_shape"] = list(df.shape)
            merged["output_shape"] = list(out_df.shape) if isinstance(out_df, pd.DataFrame) else None

            logger.info(
                "Step '%s' -> %s in %.3fs | %s -> %s",
                step_name, merged["status"], elapsed, merged["input_shape"], merged["output_shape"],
            )
            return out_df, make_json_safe(merged)

        return wrapper

    return decorator


# =============================================================================
# SECTION 2 - Data profiling
# =============================================================================


def _infer_semantic_type(series: pd.Series, sample_limit: int = 2000) -> str:
    """Guess what a column *means*, not just how pandas stored it.

    This is the signal the agent reasons over: knowing a column is
    ``numeric_as_text`` rather than ``object`` is what triggers a type-coercion
    rule instead of a mode-imputation rule.
    """
    non_null = series.dropna()
    if non_null.empty:
        return "empty"

    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "timedelta"
    if pd.api.types.is_integer_dtype(dtype):
        # An integer column where every value is unique is almost always a key.
        return "identifier" if non_null.is_unique and len(non_null) > 20 else "integer"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical"

    # ---- object / string columns: probe a sample for latent structure -------
    sample = non_null.sample(min(sample_limit, len(non_null)), random_state=0)
    text = _to_text(sample).str.strip()
    total = len(text)

    lowered = text.str.lower()
    if lowered.isin(TRUE_TOKENS | FALSE_TOKENS).mean() >= 0.95:
        return "boolean_as_text"

    _, numeric_ratio = _try_numeric(sample)
    if numeric_ratio >= 0.90:
        return "numeric_as_text"

    _, datetime_ratio = _try_datetime(sample)
    if datetime_ratio >= 0.90:
        return "datetime_as_text"

    if (text.str.match(_EMAIL_RE).fillna(False)).mean() >= 0.80:
        return "email"
    if (text.str.match(_URL_RE).fillna(False)).mean() >= 0.80:
        return "url"
    if (text.str.match(_UUID_RE).fillna(False)).mean() >= 0.80:
        return "uuid"
    phone_like = text.str.match(_PHONE_RE).fillna(False) & ~text.str.match(_DATE_LIKE_RE).fillna(False)
    if phone_like.mean() >= 0.80:
        return "phone"

    n_distinct = _safe_nunique(non_null)
    unique_ratio = (n_distinct / max(len(non_null), 1)) if n_distinct >= 0 else 0.0
    if unique_ratio > 0.95 and total > 20:
        return "identifier"
    mean_len = float(text.str.len().mean() or 0)
    if unique_ratio < 0.5 and mean_len < 50:
        return "categorical"
    return "text"


def _column_issues(series: pd.Series, stats: Dict[str, Any], iqr_multiplier: float) -> List[str]:
    """Flag actionable problems. These strings drive the agent's rule retrieval."""
    issues: List[str] = []
    if stats["missing_pct"] >= 50.0:
        issues.append("high_missing_rate")
    elif stats["missing_pct"] > 0:
        issues.append("has_missing_values")
    if stats["n_unique"] <= 1 and stats["n_non_null"] > 0:
        issues.append("constant_column")
    if stats["n_non_null"] == 0:
        issues.append("empty_column")
    if stats["semantic_type"] in {"numeric_as_text", "datetime_as_text", "boolean_as_text"}:
        issues.append("type_mismatch")
    if stats.get("has_whitespace_issues"):
        issues.append("untrimmed_whitespace")
    if stats.get("has_mixed_case_duplicates"):
        issues.append("inconsistent_casing")
    if stats.get("mixed_python_types"):
        issues.append("mixed_types")
    if stats.get("n_outliers_iqr", 0) > 0:
        issues.append("contains_outliers")
    if stats.get("n_infinite", 0) > 0:
        issues.append("contains_infinite_values")
    if stats["semantic_type"] == "categorical" and stats["n_unique"] > 100:
        issues.append("high_cardinality_categorical")
    return issues


@cleaning_step("profile_dataframe")
def profile_dataframe(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    sample_size: Optional[int] = None,
    top_n: int = 5,
    iqr_multiplier: float = 1.5,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a structural + statistical profile of a DataFrame.

    NOTE ON CONTRACT: the DataFrame is returned **unchanged**; the profile is
    delivered as the report. This keeps profiling composable inside the same
    pipeline machinery as the transformations.

    Parameters
    ----------
    columns : sequence of str, optional
        Restrict profiling to these columns. Defaults to all.
    sample_size : int, optional
        Profile a random sample of this many rows. Use on wide/long frames where
        an exhaustive scan is too slow for an interactive Streamlit session.
        Row counts in ``overview`` always reflect the FULL frame; distributional
        stats reflect the sample and ``sampled=True`` is recorded.
    top_n : int
        How many most-frequent values to capture per column.
    iqr_multiplier : float
        Tukey fence multiplier used for the outlier pre-count.

    Returns
    -------
    (df, profile) where profile has keys: ``overview``, ``columns``,
    ``dataset_issues``.
    """
    target_cols = _resolve_columns(df, columns)
    working = df[target_cols]
    sampled = False
    if sample_size is not None and sample_size > 0 and len(working) > sample_size:
        working = working.sample(sample_size, random_state=42)
        sampled = True

    n_rows_full = int(len(df))
    n_rows = int(len(working))
    total_cells = n_rows * max(len(target_cols), 1)
    missing_cells = int(working.isna().sum().sum())

    # ---- Dataset-level overview --------------------------------------------
    try:
        duplicate_rows = int(df.duplicated().sum())
    except TypeError:
        # Unhashable cell contents (lists/dicts) break duplicated(). Not fatal.
        logger.warning("Duplicate detection skipped: unhashable values present.")
        duplicate_rows = -1

    overview: Dict[str, Any] = {
        "n_rows": n_rows_full,
        "n_columns": int(len(df.columns)),
        "n_profiled_columns": len(target_cols),
        "sampled": sampled,
        "sample_rows": n_rows if sampled else n_rows_full,
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_pct": _pct(duplicate_rows, n_rows_full) if duplicate_rows >= 0 else None,
        "total_cells": total_cells,
        "missing_cells": missing_cells,
        "missing_cells_pct": _pct(missing_cells, total_cells),
    }

    # ---- Per-column analysis -----------------------------------------------
    column_profiles: Dict[str, Dict[str, Any]] = {}
    for col in target_cols:
        series = working[col]
        non_null = series.dropna()
        n_non_null = int(len(non_null))
        n_missing = int(n_rows - n_non_null)

        n_unique = _safe_nunique(non_null)  # -1 when values are unhashable.

        stats: Dict[str, Any] = {
            "dtype": str(series.dtype),
            "semantic_type": _infer_semantic_type(series),
            "n_non_null": n_non_null,
            "n_missing": n_missing,
            "missing_pct": _pct(n_missing, n_rows),
            "n_unique": n_unique,
            "unique_pct": _pct(n_unique, n_non_null) if n_unique >= 0 else None,
            "is_constant": n_unique == 1,
            "sample_values": [make_json_safe(v) for v in non_null.head(3).tolist()],
        }

        # Most frequent values - drives categorical standardization rules.
        if n_unique not in (-1, 0) and n_unique <= max(1000, top_n * 50):
            try:
                counts = non_null.value_counts().head(top_n)
                # Keys are stringified because JSON object keys must be strings.
                stats["top_values"] = {
                    str(make_json_safe(k)): int(v) for k, v in counts.items()
                }
                # ...which is why the closed vocabulary is ALSO stored as a
                # natively-typed list. A value set built from the stringified
                # keys would compare a boolean column against "True"/"False"
                # and fail under great_expectations (JSON arrays, unlike keys,
                # preserve bools and numbers).
                if 0 < n_unique <= 50:
                    stats["distinct_values"] = [
                        make_json_safe(v) for v in non_null.unique().tolist()
                    ]
            except TypeError:
                stats["top_values"] = {}

        # ---- Numeric statistics --------------------------------------------
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(non_null, errors="coerce")
            # Infinities are a data-quality defect in their own right (usually a
            # division by zero upstream). They poison every downstream statistic
            # - mean, std and quantiles all become inf/NaN - so they are counted,
            # reported, and then excluded from the distribution summary.
            n_infinite = int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())
            numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            stats["n_infinite"] = n_infinite
            if not numeric.empty:
                q1, q3 = float(numeric.quantile(0.25)), float(numeric.quantile(0.75))
                iqr = q3 - q1
                lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")  # skew/kurt warn on tiny samples
                    skew = float(numeric.skew()) if len(numeric) > 2 else 0.0
                    kurt = float(numeric.kurtosis()) if len(numeric) > 3 else 0.0
                stats.update(
                    {
                        "min": float(numeric.min()),
                        "max": float(numeric.max()),
                        "mean": float(numeric.mean()),
                        "median": float(numeric.median()),
                        "std": float(numeric.std()) if len(numeric) > 1 else 0.0,
                        "q1": q1,
                        "q3": q3,
                        "iqr": float(iqr),
                        "skew": 0.0 if math.isnan(skew) else round(skew, 4),
                        "kurtosis": 0.0 if math.isnan(kurt) else round(kurt, 4),
                        "n_zeros": int((numeric == 0).sum()),
                        "n_negative": int((numeric < 0).sum()),
                        "iqr_lower_bound": float(lower),
                        "iqr_upper_bound": float(upper),
                        "n_outliers_iqr": int(((numeric < lower) | (numeric > upper)).sum()),
                    }
                )

        # ---- Datetime statistics -------------------------------------------
        elif pd.api.types.is_datetime64_any_dtype(series) and n_non_null:
            stats.update(
                {
                    "min": make_json_safe(non_null.min()),
                    "max": make_json_safe(non_null.max()),
                    "range_days": int((non_null.max() - non_null.min()).days),
                    "n_future_dates": int((non_null > pd.Timestamp.now(tz=non_null.dt.tz)).sum()),
                }
            )

        # ---- String statistics ---------------------------------------------
        elif n_non_null and (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            text = _to_text(non_null)
            lengths = text.str.len()
            stripped = text.str.strip()
            stats.update(
                {
                    "min_length": int(lengths.min()),
                    "max_length": int(lengths.max()),
                    "mean_length": round(float(lengths.mean()), 2),
                    "has_whitespace_issues": bool((stripped != text).any()),
                    "n_empty_strings": int((stripped == "").sum()),
                    # Same value differing only by case => needs normalization.
                    "has_mixed_case_duplicates": bool(
                        _safe_nunique(stripped.str.lower()) < _safe_nunique(stripped)
                    ),
                    "n_null_sentinels": int(stripped.str.lower().isin(NULL_SENTINELS).sum()),
                    "mixed_python_types": len({type(v).__name__ for v in non_null.head(500)}) > 1,
                }
            )

        stats["issues"] = _column_issues(series, stats, iqr_multiplier)
        column_profiles[col] = stats

    # ---- Roll issues up to the dataset level -------------------------------
    dataset_issues: Dict[str, List[str]] = {}
    for col, stats in column_profiles.items():
        for issue in stats["issues"]:
            dataset_issues.setdefault(issue, []).append(col)
    if duplicate_rows > 0:
        dataset_issues.setdefault("duplicate_rows", []).append(f"{duplicate_rows} rows")

    profile = {
        "overview": overview,
        "columns": column_profiles,
        "dataset_issues": dataset_issues,
    }
    return df, profile


def summarize_profile_for_llm(profile: Dict[str, Any], max_columns: int = 40) -> str:
    """Render a profile as compact markdown for LLM prompts / vector embedding.

    Raw profile JSON is far too token-heavy to paste into a prompt. This keeps
    only what a cleaning decision actually depends on: type, null rate,
    cardinality and detected issues.
    """
    if not profile or "columns" not in profile:
        return "No profile available."

    ov = profile.get("overview", {})
    lines = [
        "## Dataset profile",
        f"- Rows: {ov.get('n_rows', '?'):,} | Columns: {ov.get('n_columns', '?')}",
        f"- Missing cells: {ov.get('missing_cells_pct', 0)}% | Duplicate rows: {ov.get('duplicate_rows', '?')}",
        "",
        "## Columns",
    ]

    for i, (col, stats) in enumerate(profile["columns"].items()):
        if i >= max_columns:
            lines.append(f"... and {len(profile['columns']) - max_columns} more columns.")
            break
        parts = [
            f"- **{col}** ({stats.get('semantic_type')}, stored as {stats.get('dtype')})",
            f"missing={stats.get('missing_pct', 0)}%",
            f"unique={stats.get('n_unique', '?')}",
        ]
        if "min" in stats and "max" in stats:
            parts.append(f"range=[{stats['min']}, {stats['max']}]")
        if stats.get("issues"):
            parts.append(f"issues={','.join(stats['issues'])}")
        lines.append(" | ".join(parts))

    if profile.get("dataset_issues"):
        lines += ["", "## Detected issues"]
        for issue, cols in profile["dataset_issues"].items():
            preview = ", ".join(map(str, cols[:8]))
            suffix = f" (+{len(cols) - 8} more)" if len(cols) > 8 else ""
            lines.append(f"- {issue}: {preview}{suffix}")

    return "\n".join(lines)


# =============================================================================
# SECTION 3 - Missing values
# =============================================================================


@cleaning_step("normalize_missing_tokens")
def normalize_missing_tokens(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    extra_tokens: Optional[Iterable[str]] = None,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convert disguised nulls ("N/A", "-", "unknown", "") into real ``NaN``.

    Run this FIRST. Every downstream missing-value statistic is wrong if
    ``"N/A"`` is still being counted as a valid category.
    """
    out = df.copy()
    tokens = set(NULL_SENTINELS)
    if extra_tokens:
        tokens |= {str(t).strip().lower() for t in extra_tokens}

    text_cols = [
        c for c in _resolve_columns(df, columns)
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c])
    ]

    replaced: Dict[str, int] = {}
    for col in text_cols:
        original_na = int(out[col].isna().sum())
        normalized = _to_text(out[col]).str.strip().str.lower()
        mask = normalized.isin(tokens).fillna(False)
        if mask.any():
            out.loc[mask, col] = np.nan
            replaced[col] = int(out[col].isna().sum() - original_na)

    return out, {
        "columns_scanned": len(text_cols),
        "columns_modified": list(replaced.keys()),
        "values_converted_to_null": replaced,
        "total_converted": int(sum(replaced.values())),
    }


def _auto_missing_strategy(series: pd.Series, missing_pct: float, drop_threshold: float) -> str:
    """Heuristic strategy selection for a single column.

    Deliberately conservative - the LLM agent (Step 2) can override any of these
    using business context retrieved from the vector store. This is the safe
    default when no rule is retrieved.
    """
    if missing_pct >= drop_threshold * 100:
        return "drop_columns"  # Too sparse to impute credibly.
    if series.isna().all():
        return "drop_columns"
    if pd.api.types.is_bool_dtype(series):
        return "mode"
    if pd.api.types.is_numeric_dtype(series):
        # Infinities make every quantile and moment meaningless; drop them for
        # the purpose of choosing a strategy (the values themselves are left
        # alone - treat_outliers or an explicit rule handles them).
        numeric = series.replace([np.inf, -np.inf], np.nan).dropna()
        if numeric.empty:
            return "drop_columns"
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            skew = numeric.skew()
            q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
            iqr = q3 - q1
            has_outliers = bool(
                iqr and ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).any()
            )
        # The mean is only trustworthy on a symmetric, outlier-free column. On
        # dirty data a single extreme value drags every imputed cell with it, so
        # median is the default and mean must be earned.
        if not has_outliers and not pd.isna(skew) and abs(float(skew)) <= 0.5:
            return "mean"
        return "median"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "ffill"  # Assumes time-ordered rows; documented limitation.
    # Categorical / text: mode only when it is genuinely a category.
    n_unique = _safe_nunique(series)
    if n_unique > 0 and n_unique <= max(20, 0.05 * len(series)):
        return "mode"
    return "constant"


@cleaning_step("handle_missing_values")
def handle_missing_values(
    df: pd.DataFrame,
    *,
    strategy: str = "auto",
    column_strategies: Optional[Dict[str, str]] = None,
    columns: Optional[Sequence[str]] = None,
    fill_value: Any = "Unknown",
    numeric_fill_value: Any = 0,
    drop_threshold: float = 0.60,
    row_missing_threshold: Optional[float] = None,
    add_indicator: bool = False,
    knn_neighbors: int = 5,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Impute or remove missing values, per column.

    Parameters
    ----------
    strategy : str
        Global default. One of: ``auto``, ``drop_rows``, ``drop_columns``,
        ``mean``, ``median``, ``mode``, ``constant``, ``ffill``, ``bfill``,
        ``interpolate``, ``knn``, ``iterative``, ``leave``.
    column_strategies : dict, optional
        Per-column overrides, e.g. ``{"revenue": "median", "notes": "leave"}``.
        This is the hook the RAG agent writes its retrieved rules into.
    drop_threshold : float
        Under ``strategy="auto"``, columns missing more than this FRACTION are
        dropped rather than imputed.
    row_missing_threshold : float, optional
        Drop any row whose missing fraction exceeds this value. Applied before
        column imputation.
    add_indicator : bool
        Append ``<col>__was_missing`` boolean columns before imputing. Preserves
        missingness as a feature for downstream models - missing is often
        informative (MNAR), and imputing it away destroys signal.
    knn_neighbors : int
        ``n_neighbors`` for ``strategy="knn"``.
    """
    valid = {
        "auto", "drop_rows", "drop_columns", "mean", "median", "mode", "constant",
        "ffill", "bfill", "interpolate", "knn", "iterative", "leave",
    }
    if strategy not in valid:
        raise ValueError(f"Unknown strategy '{strategy}'. Valid: {sorted(valid)}")

    out = df.copy()
    target_cols = _resolve_columns(df, columns)
    column_strategies = dict(column_strategies or {})
    unknown_overrides = set(column_strategies.values()) - valid
    if unknown_overrides:
        raise ValueError(f"Unknown per-column strategies: {sorted(unknown_overrides)}")

    report: Dict[str, Any] = {
        "global_strategy": strategy,
        "missing_before": int(out[target_cols].isna().sum().sum()),
        "rows_dropped": 0,
        "columns_dropped": [],
        "indicators_added": [],
        "per_column": {},
        "warnings": [],
    }

    # ---- Step A: drop mostly-empty ROWS ------------------------------------
    if row_missing_threshold is not None:
        if not 0 < row_missing_threshold <= 1:
            raise ValueError("row_missing_threshold must be in (0, 1].")
        row_missing_frac = out[target_cols].isna().mean(axis=1)
        drop_mask = row_missing_frac > row_missing_threshold
        if drop_mask.any():
            out = out.loc[~drop_mask].copy()
            report["rows_dropped"] = int(drop_mask.sum())

    # ---- Step B: resolve a concrete strategy per column --------------------
    n_rows = max(len(out), 1)
    plan: Dict[str, str] = {}
    for col in target_cols:
        if col not in out.columns:
            continue
        n_missing = int(out[col].isna().sum())
        if n_missing == 0 and strategy != "drop_columns":
            continue  # Nothing to do; avoid pointless work and noisy reports.
        missing_pct = 100.0 * n_missing / n_rows
        if col in column_strategies:
            plan[col] = column_strategies[col]
        elif strategy == "auto":
            plan[col] = _auto_missing_strategy(out[col], missing_pct, drop_threshold)
        else:
            plan[col] = strategy

    # ---- Step C: missingness indicators (before values are overwritten) ----
    if add_indicator:
        for col in plan:
            if out[col].isna().any() and plan[col] not in {"drop_columns", "leave"}:
                indicator = f"{col}__was_missing"
                out[indicator] = out[col].isna()
                report["indicators_added"].append(indicator)

    # ---- Step D: apply column-wise strategies ------------------------------
    knn_columns, iterative_columns = [], []

    for col, col_strategy in plan.items():
        n_missing = int(out[col].isna().sum())
        entry = {"strategy": col_strategy, "missing_before": n_missing}
        try:
            if col_strategy == "leave":
                entry["note"] = "explicitly skipped"

            elif col_strategy == "drop_columns":
                out = out.drop(columns=[col])
                report["columns_dropped"].append(col)
                entry["note"] = "column dropped"

            elif col_strategy == "drop_rows":
                before = len(out)
                out = out.loc[out[col].notna()].copy()
                removed = before - len(out)
                report["rows_dropped"] += removed
                entry["rows_removed"] = removed

            elif col_strategy in {"mean", "median"}:
                if not pd.api.types.is_numeric_dtype(out[col]):
                    raise TypeError(f"'{col_strategy}' requires a numeric column.")
                value = out[col].mean() if col_strategy == "mean" else out[col].median()
                if pd.isna(value):
                    raise ValueError("column is entirely null - no statistic available")
                out[col] = out[col].fillna(value)
                entry["fill_value"] = make_json_safe(value)

            elif col_strategy == "mode":
                modes = out[col].mode(dropna=True)
                if modes.empty:
                    raise ValueError("column is entirely null - no mode available")
                value = modes.iloc[0]
                # Categorical dtype rejects unseen fill values; register first.
                if isinstance(out[col].dtype, pd.CategoricalDtype) and value not in out[col].cat.categories:
                    out[col] = out[col].cat.add_categories([value])
                out[col] = out[col].fillna(value)
                entry["fill_value"] = make_json_safe(value)

            elif col_strategy == "constant":
                value = numeric_fill_value if pd.api.types.is_numeric_dtype(out[col]) else fill_value
                if isinstance(out[col].dtype, pd.CategoricalDtype) and value not in out[col].cat.categories:
                    out[col] = out[col].cat.add_categories([value])
                out[col] = out[col].fillna(value)
                entry["fill_value"] = make_json_safe(value)

            elif col_strategy in {"ffill", "bfill"}:
                out[col] = out[col].ffill() if col_strategy == "ffill" else out[col].bfill()
                # Leading (ffill) / trailing (bfill) nulls survive - close the gap.
                if out[col].isna().any():
                    out[col] = out[col].bfill() if col_strategy == "ffill" else out[col].ffill()
                    entry["note"] = "opposite-direction fill applied to edge nulls"

            elif col_strategy == "interpolate":
                if not pd.api.types.is_numeric_dtype(out[col]):
                    raise TypeError("'interpolate' requires a numeric column.")
                out[col] = out[col].interpolate(method="linear", limit_direction="both")

            elif col_strategy == "knn":
                knn_columns.append(col)
                entry["note"] = "deferred to batched KNN imputation"

            elif col_strategy == "iterative":
                iterative_columns.append(col)
                entry["note"] = "deferred to batched iterative imputation"

            if col in out.columns:
                entry["missing_after"] = int(out[col].isna().sum())
            report["per_column"][col] = entry

        except Exception as exc:
            # One column failing must not abort the other 200 columns.
            msg = f"Column '{col}' ({col_strategy}) failed: {exc}"
            logger.warning(msg)
            report["warnings"].append(msg)
            entry["error"] = str(exc)
            report["per_column"][col] = entry

    # ---- Step E: multivariate imputers (batched across columns) ------------
    for cols, kind in ((knn_columns, "knn"), (iterative_columns, "iterative")):
        if not cols:
            continue
        if not _HAS_SKLEARN:
            report["warnings"].append(f"scikit-learn unavailable - '{kind}' skipped.")
            continue
        try:
            # Multivariate imputers borrow strength from ALL numeric columns,
            # not only the ones being imputed.
            numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = sorted(set(numeric_cols) | set(cols))
            feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(out[c])]
            if not feature_cols:
                raise ValueError("no numeric columns available")

            if kind == "knn":
                imputer = KNNImputer(n_neighbors=min(knn_neighbors, max(len(out) - 1, 1)))
            else:
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer

                imputer = IterativeImputer(random_state=42, max_iter=10)

            imputed = imputer.fit_transform(out[feature_cols])
            imputed_df = pd.DataFrame(imputed, columns=feature_cols, index=out.index)
            for col in cols:  # Only write back the columns we were asked to fix.
                out[col] = imputed_df[col]
                report["per_column"].setdefault(col, {})["missing_after"] = int(out[col].isna().sum())
        except Exception as exc:
            msg = f"{kind} imputation failed for {cols}: {exc}"
            logger.warning(msg)
            report["warnings"].append(msg)

    remaining_cols = [c for c in target_cols if c in out.columns]
    report["missing_after"] = int(out[remaining_cols].isna().sum().sum()) if remaining_cols else 0
    report["values_imputed"] = max(report["missing_before"] - report["missing_after"], 0)
    return out, report


# =============================================================================
# SECTION 4 - Type and format standardization
# =============================================================================


def _try_numeric(
    series: pd.Series, *, preserve_leading_zeros: bool = True, percent_to_fraction: bool = False
) -> Tuple[pd.Series, float]:
    """Attempt numeric coercion, returning ``(converted, success_ratio)``.

    Handles: currency symbols, thousands separators, accounting negatives
    ``(1,234)`` -> ``-1234``, and trailing percent signs.

    ``preserve_leading_zeros`` protects ZIP codes, product SKUs and account
    numbers - converting "007" to 7 is unrecoverable data loss.
    """
    non_null = series.dropna()
    if non_null.empty:
        return series, 0.0

    text = _to_text(non_null).str.strip()

    if preserve_leading_zeros and text.str.match(_LEADING_ZERO_RE).fillna(False).any():
        return series, 0.0  # Signal "do not convert".

    is_percent = text.str.endswith("%").fillna(False)
    cleaned = text.str.rstrip("%")
    cleaned = cleaned.str.replace(_ACCOUNTING_NEGATIVE_RE, r"-\1", regex=True)
    cleaned = cleaned.str.replace(_NUMERIC_NOISE_RE, "", regex=True)

    # `to_numeric` on a StringDtype input returns the NULLABLE Float64 extension
    # dtype. pandas 3.x refuses to write that into a numpy float64 series, so we
    # normalize to plain float64 immediately.
    converted = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    ratio = float(converted.notna().mean())

    if percent_to_fraction and is_percent.any():
        converted = converted.where(~is_percent, converted / 100.0)

    # `reindex` re-aligns to the original index and fills dropped nulls with NaN
    # without any dtype-coercing setitem.
    return converted.reindex(series.index), ratio


def _try_datetime(
    series: pd.Series, *, fmt: Optional[str] = None, dayfirst: bool = False
) -> Tuple[pd.Series, float]:
    """Attempt datetime coercion, returning ``(converted, success_ratio)``."""
    non_null = series.dropna()
    if non_null.empty:
        return series, 0.0

    text = _to_text(non_null).str.strip()

    # Guard: bare integers like 2020 or 12345 parse as dates but almost never
    # mean one. Require at least one separator or alphabetic month name.
    if fmt is None and text.str.match(r"^\d+$").fillna(False).mean() > 0.5:
        return series, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress "Could not infer format" noise.
        try:
            converted = pd.to_datetime(text, format=fmt, errors="coerce", dayfirst=dayfirst)
        except Exception:
            converted = None

        # pandas infers ONE format for the whole column. Real exports routinely
        # mix "2023-01-15" with "15/01/2023", which leaves most rows NaT. Retry
        # with format="mixed" (per-element parsing) and keep whichever wins.
        if fmt is None:
            best_ratio = 0.0 if converted is None else float(converted.notna().mean())
            if best_ratio < 0.90:
                try:
                    mixed = pd.to_datetime(
                        text, format="mixed", errors="coerce", dayfirst=dayfirst
                    )
                    if float(mixed.notna().mean()) > best_ratio:
                        converted = mixed
                except Exception:
                    pass

    if converted is None:
        return series, 0.0

    ratio = float(converted.notna().mean())
    # reindex fills absent labels with NaT and preserves any tz-awareness,
    # avoiding the naive-vs-aware setitem conflict entirely.
    return converted.reindex(series.index), ratio


def _try_boolean(series: pd.Series) -> Tuple[pd.Series, float]:
    """Attempt boolean coercion, returning ``(converted, success_ratio)``."""
    non_null = series.dropna()
    if non_null.empty:
        return series, 0.0

    lowered = _to_text(non_null).str.strip().str.lower()
    ratio = float((lowered.isin(TRUE_TOKENS) | lowered.isin(FALSE_TOKENS)).mean())

    # Build by mapping rather than masked assignment: nullable-boolean setitem
    # semantics differ across pandas versions, mapping does not.
    mapped = lowered.map(
        lambda v: True if v in TRUE_TOKENS else (False if v in FALSE_TOKENS else pd.NA)
    )
    converted = mapped.astype("boolean").reindex(series.index)
    return converted, ratio


@cleaning_step("standardize_column_names")
def standardize_column_names(
    df: pd.DataFrame,
    *,
    case: str = "snake",
    max_length: int = 64,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize column labels to a predictable, query-safe form.

    ``"  Customer ID#  "`` -> ``"customer_id"``; ``"totalRevenue"`` -> ``"total_revenue"``.

    The returned report contains the full ``{old: new}`` mapping, which the
    pipeline must persist - downstream business rules and the RAG store both
    reference original column names.
    """
    if case not in {"snake", "lower", "upper", "none"}:
        raise ValueError("case must be one of: snake, lower, upper, none")

    mapping: Dict[str, str] = {}
    seen: Dict[str, int] = {}

    for original in df.columns:
        name = str(original).strip()
        name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

        if case == "snake":
            name = _CAMEL_BOUNDARY_RE.sub("_", name)   # camelCase -> camel_Case
            name = _NON_ALNUM_RE.sub("_", name)        # punctuation -> underscore
            name = re.sub(r"_+", "_", name).strip("_").lower()
        elif case == "lower":
            name = name.lower()
        elif case == "upper":
            name = name.upper()

        if not name:
            name = "unnamed_column"
        if name[0].isdigit():
            name = f"col_{name}"  # Keep labels valid Python identifiers.
        name = name[:max_length]

        # Resolve collisions deterministically: name, name_2, name_3, ...
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        mapping[original] = name

    out = df.rename(columns=mapping)
    changed = {str(k): v for k, v in mapping.items() if str(k) != v}
    return out, {
        "column_mapping": {str(k): v for k, v in mapping.items()},
        "columns_renamed": len(changed),
        "renamed_columns": changed,
    }


@cleaning_step("standardize_text")
def standardize_text(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    strip: bool = True,
    collapse_whitespace: bool = True,
    case: Optional[str] = None,
    unicode_form: Optional[str] = "NFKC",
    remove_control_chars: bool = True,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Normalize the *content* of text columns.

    ``case=None`` leaves casing untouched, which is the right default for names
    and free text. Use ``case="lower"`` on join keys and categorical codes.
    ``unicode_form="NFKC"`` folds curly quotes, non-breaking spaces and
    full-width characters - a frequent cause of "identical" values not matching.
    """
    if case not in {None, "lower", "upper", "title"}:
        raise ValueError("case must be one of: None, lower, upper, title")

    out = df.copy()
    text_cols = [
        c for c in _resolve_columns(df, columns)
        if pd.api.types.is_object_dtype(out[c]) or pd.api.types.is_string_dtype(out[c])
    ]

    modified: Dict[str, int] = {}
    for col in text_cols:
        original = out[col]
        series = _to_text(original)

        if unicode_form:
            series = series.map(
                lambda v: unicodedata.normalize(unicode_form, v) if isinstance(v, str) else v
            )
        if remove_control_chars:
            series = series.str.replace(_CONTROL_CHARS_RE, "", regex=True)
        if collapse_whitespace:
            series = series.str.replace(_MULTI_WS_RE, " ", regex=True)
        if strip:
            series = series.str.strip()
        if case == "lower":
            series = series.str.lower()
        elif case == "upper":
            series = series.str.upper()
        elif case == "title":
            series = series.str.title()

        # Empty strings created by stripping are missing values, not categories.
        series = series.replace("", pd.NA)

        # Back to object dtype so the frame's dtype profile stays consistent.
        new_values = series.astype(object).where(series.notna(), np.nan)
        n_changed = int((new_values.fillna("\x00") != original.fillna("\x00")).sum())
        if n_changed:
            modified[col] = n_changed
        out[col] = new_values

    return out, {
        "columns_scanned": len(text_cols),
        "columns_modified": list(modified.keys()),
        "values_changed_per_column": modified,
        "total_values_changed": int(sum(modified.values())),
        "options": {
            "strip": strip, "collapse_whitespace": collapse_whitespace,
            "case": case, "unicode_form": unicode_form,
        },
    }


@cleaning_step("standardize_types")
def standardize_types(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    convert_numeric: bool = True,
    convert_datetime: bool = True,
    convert_boolean: bool = True,
    datetime_format: Optional[str] = None,
    dayfirst: bool = False,
    min_match_ratio: float = 0.90,
    inference_sample_size: Optional[int] = 10_000,
    category_threshold: Optional[float] = 0.05,
    downcast: bool = True,
    preserve_leading_zeros: bool = True,
    percent_to_fraction: bool = False,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Infer and apply correct dtypes for object columns.

    Conversion order is boolean -> numeric -> datetime, because the checks get
    progressively more permissive (``pd.to_datetime`` will happily swallow
    things that are really numbers).

    Parameters
    ----------
    min_match_ratio : float
        A conversion is only applied if at least this fraction of non-null
        values parse successfully. Guards against destroying a text column
        because 3 of 10,000 rows looked like dates.
    inference_sample_size : int, optional
        Probe candidate types on at most this many rows, then apply the winning
        conversion to the FULL column. Inference tries three parsers per column;
        the conversion itself runs once. Sampling the expensive half is a large
        win on wide frames. ``values_coerced_to_null`` is still measured against
        the full column, so a sample that flattered the data is still visible in
        the report. Set to ``None`` to probe every row.
    category_threshold : float, optional
        Convert low-cardinality object columns to ``category`` dtype when
        ``n_unique / n_rows`` falls below this. Large memory win on wide frames.
        Set to ``None`` to disable.
    downcast : bool
        Shrink int64/float64 to the smallest safe subtype.
    """
    if not 0 < min_match_ratio <= 1:
        raise ValueError("min_match_ratio must be in (0, 1].")

    out = df.copy()
    target_cols = _resolve_columns(df, columns)
    conversions: Dict[str, Dict[str, Any]] = {}
    warnings_list: List[str] = []

    for col in target_cols:
        series = out[col]
        original_dtype = str(series.dtype)

        if series.dropna().empty:
            continue

        # --- object -> typed -------------------------------------------------
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Probe on a sample, convert on the whole column (see docstring).
            if inference_sample_size and len(series) > inference_sample_size:
                probe = series.sample(inference_sample_size, random_state=42)
            else:
                probe = series

            candidates: List[Tuple[str, float]] = []
            try:
                if convert_boolean:
                    _, ratio = _try_boolean(probe)
                    if ratio >= min_match_ratio:
                        candidates.append(("boolean", ratio))
                if convert_numeric:
                    _, ratio = _try_numeric(
                        probe,
                        preserve_leading_zeros=preserve_leading_zeros,
                        percent_to_fraction=percent_to_fraction,
                    )
                    if ratio >= min_match_ratio:
                        candidates.append(("numeric", ratio))
                if convert_datetime:
                    _, ratio = _try_datetime(probe, fmt=datetime_format, dayfirst=dayfirst)
                    if ratio >= min_match_ratio:
                        candidates.append(("datetime", ratio))
            except Exception as exc:
                warnings_list.append(f"Type inference failed for '{col}': {exc}")
                continue

            if candidates:
                # First match wins - the checks are ordered from strictest to
                # most permissive, so boolean beats numeric beats datetime.
                kind, ratio = candidates[0]
                if kind == "boolean":
                    converted, _ = _try_boolean(series)
                elif kind == "numeric":
                    converted, _ = _try_numeric(
                        series,
                        preserve_leading_zeros=preserve_leading_zeros,
                        percent_to_fraction=percent_to_fraction,
                    )
                else:
                    converted, _ = _try_datetime(series, fmt=datetime_format, dayfirst=dayfirst)

                n_lost = int(converted.isna().sum() - series.isna().sum())
                out[col] = converted
                conversions[col] = {
                    "from": original_dtype,
                    "to": str(out[col].dtype),
                    "inferred_as": kind,
                    "match_ratio": round(ratio, 4),
                    "inferred_on_sample": probe is not series,
                    "values_coerced_to_null": max(n_lost, 0),
                }
                if n_lost > 0:
                    warnings_list.append(
                        f"'{col}': {n_lost} unparseable value(s) became null during "
                        f"{kind} conversion."
                    )
                continue

            # --- object -> category ------------------------------------------
            if category_threshold is not None and len(out) > 0:
                n_unique = _safe_nunique(series)
                if n_unique > 0 and n_unique / len(out) <= category_threshold:
                    out[col] = series.astype("category")
                    conversions[col] = {
                        "from": original_dtype,
                        "to": "category",
                        "inferred_as": "categorical",
                        "n_categories": int(n_unique),
                    }
                    continue

        # --- numeric downcasting --------------------------------------------
        if downcast and pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            try:
                kind = "integer" if pd.api.types.is_integer_dtype(series) else "float"
                shrunk = pd.to_numeric(series, downcast=kind)
                if str(shrunk.dtype) != original_dtype:
                    out[col] = shrunk
                    conversions[col] = {
                        "from": original_dtype, "to": str(shrunk.dtype), "inferred_as": "downcast",
                    }
            except Exception as exc:
                warnings_list.append(f"Downcast failed for '{col}': {exc}")

    return out, {
        "columns_scanned": len(target_cols),
        "columns_converted": len(conversions),
        "conversions": conversions,
        "warnings": warnings_list,
        "dtypes_after": {c: str(t) for c, t in out.dtypes.items()},
    }


# =============================================================================
# SECTION 5 - Deduplication (exact + fuzzy)
# =============================================================================


class _UnionFind:
    """Minimal union-find (disjoint set) for grouping transitive fuzzy matches.

    Needed because similarity is not transitive by default: if A~B and B~C we
    must place A, B and C in one cluster even when A and C fall below threshold.
    """

    def __init__(self) -> None:
        self._parent: Dict[int, int] = {}

    def find(self, item: int) -> int:
        self._parent.setdefault(item, item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # Path compression.
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: int, b: int) -> None:
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def clusters(self) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for item in self._parent:
            groups.setdefault(self.find(item), []).append(item)
        return {root: members for root, members in groups.items() if len(members) > 1}


def _similarity(a: str, b: str) -> float:
    """String similarity in [0, 1]. Token-sort ignores word order differences."""
    if not a or not b:
        return 0.0
    if _HAS_RAPIDFUZZ:
        return float(_rapidfuzz_fuzz.token_sort_ratio(a, b)) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def _normalize_for_matching(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Build one lowercase, punctuation-free comparison key per row."""
    parts = []
    for col in columns:
        text = _to_text(df[col]).fillna("")
        text = text.str.lower().str.replace(r"[^\w\s]", " ", regex=True)
        text = text.str.replace(_MULTI_WS_RE, " ", regex=True).str.strip()
        parts.append(text)
    combined = parts[0]
    for extra in parts[1:]:
        combined = combined.str.cat(extra, sep=" ")
    return combined.str.strip()


def _pick_survivor(df: pd.DataFrame, positions: List[int], keep: str) -> int:
    """Choose which record in a duplicate cluster to retain."""
    if keep == "first":
        return min(positions)
    if keep == "last":
        return max(positions)
    # "most_complete": the row with the fewest nulls wins; ties break to first.
    subset = df.iloc[positions]
    completeness = subset.notna().sum(axis=1).to_numpy()
    return positions[int(np.argmax(completeness))]


@cleaning_step("drop_exact_duplicates")
def drop_exact_duplicates(
    df: pd.DataFrame,
    *,
    subset: Optional[Sequence[str]] = None,
    keep: str = "first",
    ignore_case: bool = True,
    ignore_whitespace: bool = True,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove exactly duplicated rows.

    ``ignore_case`` / ``ignore_whitespace`` compare a normalized *copy* of the
    data, so "  ACME Corp " and "acme corp" are treated as duplicates while the
    surviving row keeps its original formatting.
    """
    if keep not in {"first", "last", False}:
        raise ValueError("keep must be 'first', 'last', or False.")

    out = df.copy()
    subset_cols = _resolve_columns(df, subset)

    comparison = out[subset_cols].copy()
    if ignore_case or ignore_whitespace:
        for col in subset_cols:
            if pd.api.types.is_object_dtype(comparison[col]) or pd.api.types.is_string_dtype(comparison[col]):
                text = _to_text(comparison[col])
                if ignore_whitespace:
                    text = text.str.replace(_MULTI_WS_RE, " ", regex=True).str.strip()
                if ignore_case:
                    text = text.str.lower()
                comparison[col] = text

    try:
        dup_mask = comparison.duplicated(keep=keep)
    except TypeError as exc:
        # Unhashable cell values (lists/dicts) - stringify and retry once.
        logger.warning("Falling back to stringified duplicate detection: %s", exc)
        comparison = comparison.astype(str)
        dup_mask = comparison.duplicated(keep=keep)

    n_removed = int(dup_mask.sum())
    result = out.loc[~dup_mask].copy()

    # Capture a few examples so a human (and the audit log) can verify the call.
    examples: List[Dict[str, Any]] = []
    if n_removed:
        all_dupes = comparison.duplicated(keep=False)
        if all_dupes.any():
            keys = comparison.loc[all_dupes].astype(str).apply(lambda row: "|".join(row), axis=1)
            for _, labels in list(keys.groupby(keys).groups.items())[:3]:
                examples.append(
                    {
                        "n_copies": int(len(labels)),
                        "record": make_json_safe(out.loc[labels[0], subset_cols].to_dict()),
                    }
                )

    return result, {
        "subset_columns": subset_cols,
        "keep": keep,
        "duplicates_removed": n_removed,
        "duplicates_removed_pct": _pct(n_removed, len(out)),
        "rows_remaining": int(len(result)),
        "examples": examples,
    }


def _fuzzy_dedupe_with_dedupe_lib(
    df: pd.DataFrame, columns: Sequence[str], settings_file: str, threshold: float
) -> Dict[int, List[int]]:
    """Cluster records using a PRE-TRAINED ``dedupe`` model.

    Important operational note: ``dedupe.Dedupe`` requires interactive active
    learning (``console_label``) to produce a model, which cannot run inside an
    automated pipeline or a Streamlit request. The supported production pattern
    is: train once offline, write ``settings_file``, then load it here with
    ``StaticDedupe``. Without that file the caller must use the
    blocking + rapidfuzz backend.
    """
    try:
        import dedupe  # Lazy import: several seconds and a C extension.
    except ImportError as exc:
        raise DependencyUnavailableError(
            "The 'dedupe' package is not installed. Install it or use backend='fuzzy'."
        ) from exc

    with open(settings_file, "rb") as handle:
        deduper = dedupe.StaticDedupe(handle)

    # dedupe requires a {hashable_id: {field: str_or_None}} mapping.
    records = {
        position: {
            col: (None if _safe_isna(value) else str(value).strip() or None)
            for col, value in row.items()
        }
        for position, (_, row) in enumerate(df[list(columns)].iterrows())
    }

    clusters: Dict[int, List[int]] = {}
    for cluster_id, (record_ids, _scores) in enumerate(deduper.partition(records, threshold)):
        members = [int(r) for r in record_ids]
        if len(members) > 1:
            clusters[cluster_id] = members
    return clusters


@cleaning_step("fuzzy_deduplicate")
def fuzzy_deduplicate(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    threshold: float = 0.88,
    blocking_keys: Optional[Sequence[str]] = None,
    block_prefix_length: int = 4,
    keep: str = "most_complete",
    backend: str = "auto",
    dedupe_settings_file: Optional[str] = None,
    max_block_size: int = 2000,
    max_examples: int = 5,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Remove near-duplicate records ("Acme Corp." vs "ACME Corporation").

    Backends
    --------
    ``"dedupe"``  Uses a pre-trained ``dedupe`` model (requires
                  ``dedupe_settings_file``). Best quality; needs offline training.
    ``"fuzzy"``   Blocking + rapidfuzz token-sort similarity + union-find
                  clustering. No training required.
    ``"auto"``    ``dedupe`` when a settings file is supplied, else ``fuzzy``.

    Complexity warning
    ------------------
    Pairwise comparison is O(n^2) *within a block*. Blocking is what makes this
    tractable: records only compete against others sharing a block key. Blocks
    larger than ``max_block_size`` are skipped and reported rather than allowed
    to hang the UI.

    Parameters
    ----------
    columns : sequence of str
        Fields that identify a record (e.g. ``["name", "email", "city"]``).
    threshold : float
        Similarity in [0, 1] above which two records are considered the same.
        0.85-0.90 is a sensible starting band; tune on labelled samples.
    blocking_keys : sequence of str, optional
        Columns whose exact value must match for two records to be compared
        (e.g. ``["country"]``). Far more precise than prefix blocking when a
        reliable key exists.
    keep : {"most_complete", "first", "last"}
        Which record in a cluster survives.
    """
    if not columns:
        raise ValueError("`columns` is required for fuzzy deduplication.")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")
    if keep not in {"most_complete", "first", "last"}:
        raise ValueError("keep must be one of: most_complete, first, last")
    if backend not in {"auto", "fuzzy", "dedupe"}:
        raise ValueError("backend must be one of: auto, fuzzy, dedupe")

    match_cols = _resolve_columns(df, columns)
    block_cols = _resolve_columns(df, blocking_keys) if blocking_keys else []

    # Work on a 0..n-1 positional index so blocking, union-find and iloc all
    # speak the same coordinate system. The original labels are restored at the
    # end from `df.index`, which avoids reset_index() colliding with a real
    # column already named "index".
    out = df.copy().reset_index(drop=True)

    resolved_backend = backend
    if backend == "auto":
        resolved_backend = "dedupe" if (dedupe_settings_file and _has_dedupe()) else "fuzzy"

    report: Dict[str, Any] = {
        "backend": resolved_backend,
        "match_columns": match_cols,
        "blocking_keys": block_cols,
        "threshold": threshold,
        "similarity_engine": "rapidfuzz" if _HAS_RAPIDFUZZ else "difflib",
        "warnings": [],
    }

    clusters: Dict[int, List[int]] = {}

    # ---- Backend A: trained dedupe model -----------------------------------
    if resolved_backend == "dedupe":
        try:
            if not dedupe_settings_file:
                raise DependencyUnavailableError(
                    "backend='dedupe' requires dedupe_settings_file (a trained model)."
                )
            clusters = _fuzzy_dedupe_with_dedupe_lib(
                out, match_cols, dedupe_settings_file, threshold
            )
        except Exception as exc:
            msg = f"dedupe backend failed ({exc}); falling back to blocking+fuzzy matching."
            logger.warning(msg)
            report["warnings"].append(msg)
            report["backend"] = resolved_backend = "fuzzy"

    # ---- Backend B: blocking + similarity + union-find ---------------------
    if resolved_backend == "fuzzy":
        if not _HAS_RAPIDFUZZ:
            report["warnings"].append(
                "rapidfuzz is not installed; falling back to difflib, which penalises "
                "length differences much harder than token_sort_ratio "
                "('acme corp' vs 'acme corporation' scores ~0.72 instead of ~1.00). "
                "Lower `threshold` by roughly 0.15, or install rapidfuzz."
            )
        keys = _normalize_for_matching(out, match_cols)

        # Build blocks. Explicit blocking keys are preferred; otherwise fall
        # back to "first N characters of the normalized key".
        if block_cols:
            block_ids = out[block_cols].astype(str).apply(lambda row: "|".join(row), axis=1)
        else:
            block_ids = keys.str[:block_prefix_length]

        blocks: Dict[str, List[int]] = {}
        for position, block_id in enumerate(block_ids):
            if keys.iloc[position]:  # Skip rows with no comparable content.
                blocks.setdefault(str(block_id), []).append(position)

        union = _UnionFind()
        pair_scores: Dict[Tuple[int, int], float] = {}
        comparisons = skipped_blocks = 0

        for block_id, positions in blocks.items():
            if len(positions) < 2:
                continue
            if len(positions) > max_block_size:
                skipped_blocks += 1
                report["warnings"].append(
                    f"Block '{block_id}' has {len(positions)} records (> max_block_size="
                    f"{max_block_size}); skipped. Add a blocking key to partition it."
                )
                continue
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    left, right = positions[i], positions[j]
                    comparisons += 1
                    score = _similarity(keys.iloc[left], keys.iloc[right])
                    if score >= threshold:
                        union.union(left, right)
                        pair_scores[(left, right)] = score

        clusters = union.clusters()
        report.update(
            {
                "n_blocks": len(blocks),
                "blocks_skipped": skipped_blocks,
                "pairwise_comparisons": comparisons,
            }
        )
        # A skipped block means records were never compared. Reporting
        # "0 duplicates removed" with status=success would read as "the data is
        # clean" when the truth is "we did not look", so the status is demoted
        # and the pipeline/UI can surface it.
        if skipped_blocks:
            report["status"] = "partial"
            report["status_reason"] = (
                f"{skipped_blocks} block(s) exceeded max_block_size and were not "
                f"compared - results are incomplete."
            )

        # Example clusters make the decision auditable in the UI and in the
        # cleaning log the RAG store indexes.
        examples: List[Dict[str, Any]] = []
        for members in list(clusters.values())[:max_examples]:
            score = next(
                (s for (a, b), s in pair_scores.items() if a in members and b in members), None
            )
            examples.append(
                {
                    "size": len(members),
                    "similarity": round(score, 4) if score else None,
                    "records": [
                        make_json_safe(out.iloc[p][match_cols].to_dict()) for p in members[:3]
                    ],
                }
            )
        report["example_clusters"] = examples

    # ---- Resolve clusters to survivors -------------------------------------
    drop_positions: List[int] = []
    for members in clusters.values():
        survivor = _pick_survivor(out, members, keep)
        drop_positions.extend([p for p in members if p != survivor])

    result = out.drop(index=drop_positions) if drop_positions else out
    # Map surviving positions back to the caller's original index labels.
    result.index = df.index[result.index]

    report.update(
        {
            "clusters_found": len(clusters),
            "records_removed": len(drop_positions),
            "records_removed_pct": _pct(len(drop_positions), len(df)),
            "rows_remaining": int(len(result)),
            "keep_policy": keep,
        }
    )
    return result, report


def _has_dedupe() -> bool:
    """Check ``dedupe`` availability without paying the import cost twice."""
    try:
        import dedupe  # noqa: F401

        return True
    except Exception:
        return False


# =============================================================================
# SECTION 6 - Outlier detection and treatment
# =============================================================================


@cleaning_step("detect_outliers_iqr")
def detect_outliers_iqr(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    multiplier: float = 1.5,
    min_samples: int = 8,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Flag univariate outliers using Tukey's IQR fences.

    CONTRACT NOTE: returns a boolean **mask DataFrame** (same index, one column
    per analysed numeric column), not cleaned data. Pair with ``treat_outliers``.

    ``multiplier=1.5`` marks "outliers"; ``3.0`` marks "extreme outliers".
    This method is distribution-free but univariate - it cannot see a record
    that is only anomalous in combination (age=8, salary=200000).

    Parameters
    ----------
    min_samples : int
        Columns with fewer non-null values than this are skipped. Quartiles
        estimated from a handful of points produce fences wide enough to contain
        anything, which silently reports "no outliers" on data that is obviously
        anomalous. Skipping and saying so is more honest than a null result.
    """
    if multiplier <= 0:
        raise ValueError("multiplier must be positive.")

    numeric_default = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = _resolve_columns(df, columns, default=numeric_default)
    target_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])
                   and not pd.api.types.is_bool_dtype(df[c])]

    mask = pd.DataFrame(False, index=df.index, columns=target_cols)
    bounds: Dict[str, Dict[str, Any]] = {}

    for col in target_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        n_valid = int(series.notna().sum())
        if n_valid < min_samples:
            bounds[col] = {
                "skipped": True,
                "reason": f"only {n_valid} non-null values (min_samples={min_samples})",
            }
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            # Zero IQR (constant or heavily tied column) makes fences degenerate
            # and would flag every distinct value. Skip rather than mislead.
            bounds[col] = {"skipped": True, "reason": "zero or undefined IQR"}
            continue
        lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
        col_mask = ((series < lower) | (series > upper)) & series.notna()
        mask[col] = col_mask
        bounds[col] = {
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "lower_bound": float(lower), "upper_bound": float(upper),
            "n_outliers": int(col_mask.sum()),
            "outlier_pct": _pct(int(col_mask.sum()), len(df)),
            "n_below": int(((series < lower) & series.notna()).sum()),
            "n_above": int(((series > upper) & series.notna()).sum()),
        }

    return mask, {
        "method": "iqr",
        "multiplier": multiplier,
        "columns_analyzed": target_cols,
        "bounds": bounds,
        "total_outlier_cells": int(mask.to_numpy().sum()),
        "rows_with_any_outlier": int(mask.any(axis=1).sum()),
    }


@cleaning_step("detect_outliers_isolation_forest")
def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    contamination: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 42,
    scale: bool = True,
    impute_missing: bool = True,
    return_scores: bool = False,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Flag MULTIVARIATE anomalies with Isolation Forest (PyOD, sklearn fallback).

    CONTRACT NOTE: returns a single-column boolean mask DataFrame
    (``__is_outlier__``) indexed like the input - anomaly is a property of the
    whole row here, not of an individual cell.

    Why this and not just IQR: Isolation Forest finds records that are odd in
    *combination*, which is exactly what rule-based fences miss. It is also
    robust in higher dimensions and scales near-linearly.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies. This sets the decision threshold and
        is effectively a budget, not a discovery - treat it as a tuning knob.
    scale : bool
        Standardize features first. Isolation Forest is fairly scale-robust but
        standardizing keeps split quality consistent across mixed units.
    impute_missing : bool
        Median-impute NaNs for fitting only. The estimator rejects NaN; original
        data is untouched. Rows still fully null after imputation are marked
        non-outlier and reported.
    """
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be in (0, 0.5).")
    if not (_HAS_PYOD or _HAS_SKLEARN):
        raise DependencyUnavailableError(
            "Isolation Forest needs either 'pyod' or 'scikit-learn' installed."
        )

    numeric_default = df.select_dtypes(include=[np.number]).columns.tolist()
    target_cols = _resolve_columns(df, columns, default=numeric_default)
    target_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])
                   and not pd.api.types.is_bool_dtype(df[c])]

    mask = pd.DataFrame(False, index=df.index, columns=["__is_outlier__"])
    if not target_cols:
        return mask, {
            "method": "isolation_forest",
            "status": "skipped",
            "reason": "no numeric columns available",
        }

    features = df[target_cols].apply(pd.to_numeric, errors="coerce")

    # Rows that are entirely null carry no signal - exclude from fitting.
    usable = features.notna().any(axis=1)
    if usable.sum() < 10:
        return mask, {
            "method": "isolation_forest",
            "status": "skipped",
            "reason": f"only {int(usable.sum())} usable rows (minimum 10)",
        }

    matrix = features.loc[usable].copy()
    if impute_missing and matrix.isna().any().any():
        matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    else:
        matrix = matrix.dropna()
        usable = usable & features.index.isin(matrix.index)

    # Infinities are not NaN but still break the estimator.
    matrix = matrix.replace([np.inf, -np.inf], np.nan).fillna(matrix.median(numeric_only=True)).fillna(0.0)

    if scale and _HAS_SKLEARN:
        matrix = pd.DataFrame(
            StandardScaler().fit_transform(matrix), index=matrix.index, columns=matrix.columns
        )

    backend = "pyod.IForest" if _HAS_PYOD else "sklearn.IsolationForest"
    try:
        if _HAS_PYOD:
            model = _PyODIForest(
                contamination=contamination, n_estimators=n_estimators, random_state=random_state
            )
            model.fit(matrix.to_numpy())
            labels = model.labels_.astype(bool)          # PyOD: 1 = outlier
            scores = np.asarray(model.decision_scores_)  # Higher = more anomalous
            threshold = float(getattr(model, "threshold_", np.nan))
        else:
            model = _SklearnIForest(
                contamination=contamination, n_estimators=n_estimators,
                random_state=random_state, n_jobs=-1,
            )
            predictions = model.fit_predict(matrix.to_numpy())
            labels = predictions == -1                    # sklearn: -1 = outlier
            scores = -np.asarray(model.score_samples(matrix.to_numpy()))
            threshold = float(np.quantile(scores, 1 - contamination))
    except Exception as exc:
        raise CleaningStepError(f"Isolation Forest fitting failed: {exc}") from exc

    mask.loc[matrix.index, "__is_outlier__"] = labels

    report: Dict[str, Any] = {
        "method": "isolation_forest",
        "backend": backend,
        "columns_analyzed": target_cols,
        "contamination": contamination,
        "n_estimators": n_estimators,
        "rows_scored": int(len(matrix)),
        "rows_excluded": int(len(df) - len(matrix)),
        "n_outliers": int(labels.sum()),
        "outlier_pct": _pct(int(labels.sum()), len(df)),
        "score_threshold": None if math.isnan(threshold) else round(threshold, 6),
        "score_stats": {
            "min": round(float(scores.min()), 6),
            "max": round(float(scores.max()), 6),
            "mean": round(float(scores.mean()), 6),
        },
    }
    if return_scores:
        # Index-aligned so the UI can sort records by anomaly score.
        report["scores"] = {
            str(idx): round(float(score), 6) for idx, score in zip(matrix.index, scores)
        }
    return mask, report


@cleaning_step("treat_outliers")
def treat_outliers(
    df: pd.DataFrame,
    *,
    method: str = "iqr",
    action: str = "clip",
    columns: Optional[Sequence[str]] = None,
    multiplier: float = 1.5,
    contamination: float = 0.05,
    flag_column: str = "is_outlier",
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Detect and then act on outliers in one call.

    Parameters
    ----------
    method : {"iqr", "isolation_forest", "both"}
        ``"both"`` takes the UNION of the two detectors - conservative, use when
        recall matters more than precision.
    action : {"flag", "clip", "nan", "remove"}
        ``flag``   add a boolean column; change nothing else (safest default for
                   analytics - preserves the data, surfaces the finding).
        ``clip``   winsorize to the IQR fences. IQR only; ignored for the
                   multivariate detector, which has no per-column bound.
        ``nan``    null the offending cells so ``handle_missing_values`` can
                   impute them. Often the best pipeline composition.
        ``remove`` drop the whole row. Destructive - only when you are certain
                   the records are invalid, not merely extreme.

    Domain caution: a genuine extreme value (a real high-value transaction) is
    not an error. Prefer ``flag`` unless a retrieved business rule says otherwise.
    """
    if method not in {"iqr", "isolation_forest", "both"}:
        raise ValueError("method must be one of: iqr, isolation_forest, both")
    if action not in {"flag", "clip", "nan", "remove"}:
        raise ValueError("action must be one of: flag, clip, nan, remove")

    out = df.copy()
    report: Dict[str, Any] = {"method": method, "action": action, "detection": {}, "warnings": []}

    cell_mask: Optional[pd.DataFrame] = None
    row_mask = pd.Series(False, index=out.index)

    # ---- Detection ----------------------------------------------------------
    if method in {"iqr", "both"}:
        cell_mask, iqr_report = detect_outliers_iqr(
            out, columns=columns, multiplier=multiplier, errors="raise"
        )
        report["detection"]["iqr"] = iqr_report
        row_mask |= cell_mask.any(axis=1)

    if method in {"isolation_forest", "both"}:
        if_mask, if_report = detect_outliers_isolation_forest(
            out, columns=columns, contamination=contamination, errors="raise"
        )
        report["detection"]["isolation_forest"] = if_report
        row_mask |= if_mask["__is_outlier__"]

    report["rows_flagged"] = int(row_mask.sum())
    report["rows_flagged_pct"] = _pct(int(row_mask.sum()), len(out))

    # ---- Treatment ----------------------------------------------------------
    if action == "flag":
        out[flag_column] = row_mask
        report["flag_column"] = flag_column

    elif action == "clip":
        if cell_mask is None:
            report["warnings"].append(
                "action='clip' needs IQR bounds; Isolation Forest is multivariate. "
                "Falling back to 'flag'."
            )
            out[flag_column] = row_mask
            report["flag_column"] = flag_column
            report["action"] = "flag"
        else:
            clipped: Dict[str, int] = {}
            for col, bound in report["detection"]["iqr"]["bounds"].items():
                if bound.get("skipped"):
                    continue
                before = out[col].copy()
                out[col] = out[col].clip(lower=bound["lower_bound"], upper=bound["upper_bound"])
                n_changed = int((before != out[col]).sum())
                if n_changed:
                    clipped[col] = n_changed
            report["values_clipped"] = clipped
            report["total_values_clipped"] = int(sum(clipped.values()))

    elif action == "nan":
        if cell_mask is not None:
            # Cell-level precision: only the offending value is nulled.
            nulled: Dict[str, int] = {}
            for col in cell_mask.columns:
                col_mask = cell_mask[col]
                if col_mask.any():
                    _set_nan(out, col_mask, [col])
                    nulled[col] = int(col_mask.sum())
            report["values_nulled"] = nulled
            report["total_values_nulled"] = int(sum(nulled.values()))
        else:
            # Row-level detector: null the analysed numeric columns for that row.
            analysed = report["detection"]["isolation_forest"].get("columns_analyzed", [])
            _set_nan(out, row_mask, analysed)
            report["total_values_nulled"] = int(row_mask.sum() * len(analysed))
        report["note"] = "Run handle_missing_values() next to impute the nulled cells."

    elif action == "remove":
        out = out.loc[~row_mask].copy()
        report["rows_removed"] = int(row_mask.sum())
        report["rows_remaining"] = int(len(out))

    return out, report


# =============================================================================
# SECTION 7 - Data quality validation (great_expectations)
# =============================================================================


def build_expectation_suite_from_profile(
    profile: Dict[str, Any],
    *,
    suite_name: str = "auto_generated_suite",
    missing_tolerance: float = 0.05,
    range_padding: float = 0.10,
    include_ranges: bool = True,
    include_sets: bool = True,
    max_set_size: int = 25,
    unique_ratio_threshold: float = 0.99,
) -> Dict[str, Any]:
    """Derive an expectation suite from a profile of a KNOWN-GOOD dataset.

    The output is a plain JSON-serializable dict (not a GX object), so it can be
    versioned in git, stored in Chroma, edited by a human, or generated by the
    LLM agent - and executed by any of the validation backends below.

    Parameters
    ----------
    missing_tolerance : float
        Columns whose observed null rate is at or below this get a
        ``not_be_null`` expectation with ``mostly = 1 - observed_rate``.
    range_padding : float
        Widen observed min/max by this fraction of the range so the suite does
        not fail on the next batch's legitimate variation.
    """
    if not profile or "columns" not in profile:
        raise ValueError("A profile produced by profile_dataframe() is required.")

    overview = profile.get("overview", {})
    columns = profile["columns"]
    expectations: List[Dict[str, Any]] = []

    # ---- Table-level expectations ------------------------------------------
    n_rows = int(overview.get("n_rows", 0) or 0)
    if n_rows:
        expectations.append(
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": max(1, int(n_rows * 0.5)), "max_value": int(n_rows * 2)},
                "meta": {"rationale": "Row count within 50%-200% of the reference batch."},
            }
        )
    expectations.append(
        {
            "expectation_type": "expect_table_columns_to_match_set",
            "kwargs": {"column_set": list(columns.keys()), "exact_match": False},
            "meta": {"rationale": "Schema drift guard: required columns must be present."},
        }
    )

    # ---- Column-level expectations -----------------------------------------
    for col, stats in columns.items():
        expectations.append(
            {"expectation_type": "expect_column_to_exist", "kwargs": {"column": col}}
        )

        missing_rate = float(stats.get("missing_pct", 0)) / 100.0
        if missing_rate <= missing_tolerance:
            mostly = round(max(0.0, 1.0 - max(missing_rate, 0.0)), 4)
            expectations.append(
                {
                    "expectation_type": "expect_column_values_to_not_be_null",
                    "kwargs": {"column": col, "mostly": mostly},
                    "meta": {"rationale": f"Observed null rate {stats.get('missing_pct', 0)}%."},
                }
            )

        # Uniqueness - only for columns that look like keys.
        unique_pct = stats.get("unique_pct")
        if unique_pct is not None and unique_pct / 100.0 >= unique_ratio_threshold and stats.get("n_non_null", 0) > 20:
            expectations.append(
                {
                    "expectation_type": "expect_column_values_to_be_unique",
                    "kwargs": {"column": col},
                    "meta": {"rationale": "Column is (near-)unique; treated as an identifier."},
                }
            )

        semantic = stats.get("semantic_type")

        # Numeric ranges.
        if (
            include_ranges
            and semantic in {"integer", "float"}
            and stats.get("min") is not None
            and stats.get("max") is not None
        ):
            low, high = float(stats["min"]), float(stats["max"])
            spread = (high - low) or abs(high) or 1.0
            pad = spread * range_padding
            # Never let padding invent negative values for a non-negative column.
            min_value = low - pad if low < 0 else max(0.0, low - pad)
            expectations.append(
                {
                    "expectation_type": "expect_column_values_to_be_between",
                    "kwargs": {
                        "column": col,
                        "min_value": round(min_value, 6),
                        "max_value": round(high + pad, 6),
                        "mostly": 0.99,
                    },
                    "meta": {"rationale": f"Observed range [{low}, {high}] padded by {range_padding:.0%}."},
                }
            )

        # Categorical value sets.
        if include_sets and semantic in {"categorical", "boolean", "boolean_as_text"}:
            distinct = stats.get("distinct_values") or []
            n_unique = stats.get("n_unique", 0) or 0
            if distinct and 0 < n_unique <= max_set_size and len(distinct) >= n_unique:
                expectations.append(
                    {
                        "expectation_type": "expect_column_values_to_be_in_set",
                        "kwargs": {"column": col, "value_set": distinct},
                        "meta": {"rationale": "Closed vocabulary observed in the reference batch."},
                    }
                )

        # Text length bounds.
        if semantic in {"text", "categorical"} and stats.get("min_length") is not None:
            expectations.append(
                {
                    "expectation_type": "expect_column_value_lengths_to_be_between",
                    "kwargs": {
                        "column": col,
                        "min_value": max(0, int(stats["min_length"]) - 2),
                        "max_value": int(stats["max_length"]) * 2,
                        "mostly": 0.99,
                    },
                }
            )

        # Format regexes.
        if semantic == "email":
            expectations.append(
                {
                    "expectation_type": "expect_column_values_to_match_regex",
                    "kwargs": {"column": col, "regex": _EMAIL_RE.pattern, "mostly": 0.95},
                }
            )

    return {
        "suite_name": suite_name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_expectations": len(expectations),
        "expectations": expectations,
    }


# --- Pandas fallback validators ----------------------------------------------
# One callable per expectation type, returning (success, details). These make
# the module usable with zero great_expectations install and give us a stable
# behavioural reference when GX changes its API.


def _elementwise(
    series: pd.Series, predicate: pd.Series, mostly: Optional[float]
) -> Tuple[bool, Dict[str, Any]]:
    """Shared ``mostly`` semantics: fraction of NON-NULL values satisfying the rule."""
    non_null = series.notna()
    n_evaluated = int(non_null.sum())
    if n_evaluated == 0:
        return True, {"note": "no non-null values to evaluate", "element_count": 0}
    satisfied = int((predicate & non_null).sum())
    fraction = satisfied / n_evaluated
    required = 1.0 if mostly is None else float(mostly)
    return fraction >= required, {
        "element_count": n_evaluated,
        "unexpected_count": n_evaluated - satisfied,
        "unexpected_percent": round(100.0 * (1 - fraction), 4),
        "mostly_required": required,
    }


def _validate_with_pandas(df: pd.DataFrame, suite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute an expectation suite using pandas only."""
    results: List[Dict[str, Any]] = []

    for spec in suite.get("expectations", []):
        etype = spec.get("expectation_type")
        kwargs = dict(spec.get("kwargs", {}))
        column = kwargs.get("column")
        mostly = kwargs.get("mostly")
        entry: Dict[str, Any] = {"expectation_type": etype, "column": column, "kwargs": kwargs}

        try:
            # --- Table-level ------------------------------------------------
            if etype == "expect_table_row_count_to_be_between":
                n = len(df)
                lo, hi = kwargs.get("min_value"), kwargs.get("max_value")
                success = (lo is None or n >= lo) and (hi is None or n <= hi)
                entry.update({"success": success, "observed_value": n})

            elif etype == "expect_table_columns_to_match_set":
                expected = set(kwargs.get("column_set", []))
                actual = set(df.columns)
                success = (expected == actual) if kwargs.get("exact_match") else expected.issubset(actual)
                entry.update(
                    {
                        "success": success,
                        "missing_columns": sorted(expected - actual),
                        "unexpected_columns": sorted(actual - expected),
                    }
                )

            elif etype == "expect_column_to_exist":
                entry.update({"success": column in df.columns})

            # --- Column-level: guard existence once -------------------------
            elif column is not None and column not in df.columns:
                entry.update({"success": False, "error": f"column '{column}' not found"})

            elif etype == "expect_column_values_to_not_be_null":
                series = df[column]
                n = len(series)
                non_null = int(series.notna().sum())
                fraction = non_null / n if n else 1.0
                required = 1.0 if mostly is None else float(mostly)
                entry.update(
                    {
                        "success": fraction >= required,
                        "observed_non_null_fraction": round(fraction, 4),
                        "unexpected_count": n - non_null,
                    }
                )

            elif etype == "expect_column_values_to_be_unique":
                series = df[column].dropna()
                n_dupes = int(series.duplicated().sum())
                entry.update({"success": n_dupes == 0, "unexpected_count": n_dupes})

            elif etype == "expect_column_values_to_be_between":
                series = pd.to_numeric(df[column], errors="coerce")
                lo, hi = kwargs.get("min_value"), kwargs.get("max_value")
                predicate = pd.Series(True, index=series.index)
                if lo is not None:
                    predicate &= series >= lo
                if hi is not None:
                    predicate &= series <= hi
                success, details = _elementwise(series, predicate.fillna(False), mostly)
                entry.update({"success": success, **details})

            elif etype == "expect_column_values_to_be_in_set":
                series = df[column]
                value_set = list(kwargs.get("value_set", []))
                # Type-STRICT membership, deliberately. Stringifying both sides
                # would be more forgiving but would then disagree with the GX
                # backend on the same suite, and a validator whose verdict
                # depends on which engine ran it is worse than a strict one.
                predicate = series.isin(value_set)
                success, details = _elementwise(series, predicate, mostly)
                unexpected = series[~predicate & series.notna()].unique().tolist()
                entry.update(
                    {
                        "success": success,
                        **details,
                        "unexpected_values": [make_json_safe(v) for v in unexpected[:10]],
                    }
                )

            elif etype == "expect_column_value_lengths_to_be_between":
                series = df[column]
                lengths = _to_text(series).str.len()
                lo, hi = kwargs.get("min_value"), kwargs.get("max_value")
                predicate = pd.Series(True, index=series.index)
                if lo is not None:
                    predicate &= lengths >= lo
                if hi is not None:
                    predicate &= lengths <= hi
                success, details = _elementwise(series, predicate.fillna(False), mostly)
                entry.update({"success": success, **details})

            elif etype == "expect_column_values_to_match_regex":
                series = df[column]
                pattern = kwargs.get("regex", "")
                predicate = _to_text(series).str.match(pattern, na=False).fillna(False)
                success, details = _elementwise(series, predicate, mostly)
                entry.update({"success": success, **details})

            elif etype in {"expect_column_values_to_be_of_type", "expect_column_values_to_be_in_type_list"}:
                actual = str(df[column].dtype)
                expected = kwargs.get("type_") or kwargs.get("type_list") or []
                expected_list = [expected] if isinstance(expected, str) else list(expected)
                success = any(str(t).lower() in actual.lower() for t in expected_list)
                entry.update({"success": success, "observed_value": actual})

            else:
                entry.update(
                    {"success": None, "error": f"unsupported expectation type '{etype}' in fallback"}
                )

        except Exception as exc:
            logger.warning("Expectation %s failed to execute: %s", etype, exc)
            entry.update({"success": False, "error": str(exc)})

        results.append(entry)

    return results


def _validate_with_gx(df: pd.DataFrame, suite: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    """Execute a suite through great_expectations, handling 1.x and 0.18.x.

    GX is imported lazily and every API call is defensive: the 1.0 release
    replaced the entire ``DataContext``/``Batch`` surface, and a version bump
    silently breaking validation is a production incident we would rather
    downgrade to a logged fallback.
    """
    # GX renders a tqdm "Calculating Metrics" bar straight to the terminal,
    # which floods Streamlit logs and CLI output. tqdm honours this env var, and
    # it must be set BEFORE great_expectations is imported.
    import os

    os.environ.setdefault("TQDM_DISABLE", "1")

    import great_expectations as gx  # Lazy: expensive import.

    version = str(getattr(gx, "__version__", "0"))
    major = int(version.split(".")[0]) if version[0].isdigit() else 0

    # ---- GX 1.x ------------------------------------------------------------
    if major >= 1:
        context = gx.get_context(mode="ephemeral")
        data_source = context.data_sources.add_pandas(name="cleaning_agent_pandas")
        asset = data_source.add_dataframe_asset(name="runtime_asset")
        batch_definition = asset.add_batch_definition_whole_dataframe("runtime_batch")

        suite_obj = context.suites.add(gx.ExpectationSuite(name=suite.get("suite_name", "suite")))
        for spec in suite.get("expectations", []):
            etype = spec["expectation_type"]
            # expect_column_values_to_not_be_null -> ExpectColumnValuesToNotBeNull
            class_name = "".join(part.title() for part in etype.split("_"))
            expectation_cls = getattr(gx.expectations, class_name, None)
            if expectation_cls is None:
                logger.warning("GX 1.x has no expectation class '%s'; skipping.", class_name)
                continue
            try:
                suite_obj.add_expectation(expectation_cls(**spec.get("kwargs", {})))
            except Exception as exc:
                logger.warning("Could not add expectation %s: %s", etype, exc)

        validation_definition = context.validation_definitions.add(
            gx.ValidationDefinition(
                data=batch_definition, suite=suite_obj, name="cleaning_agent_validation"
            )
        )
        raw = validation_definition.run(batch_parameters={"dataframe": df})
        raw_dict = raw.to_json_dict() if hasattr(raw, "to_json_dict") else dict(raw)
        backend = f"great_expectations {version}"

    # ---- GX 0.18.x ---------------------------------------------------------
    else:
        gx_df = gx.from_pandas(df)
        for spec in suite.get("expectations", []):
            method = getattr(gx_df, spec["expectation_type"], None)
            if method is None:
                logger.warning("GX 0.x has no method '%s'; skipping.", spec["expectation_type"])
                continue
            try:
                method(**spec.get("kwargs", {}))
            except Exception as exc:
                logger.warning("Expectation %s failed: %s", spec["expectation_type"], exc)
        raw = gx_df.validate(result_format="SUMMARY")
        raw_dict = raw.to_json_dict() if hasattr(raw, "to_json_dict") else dict(raw)
        backend = f"great_expectations {version}"

    # ---- Normalize GX output to our flat result schema ---------------------
    normalized: List[Dict[str, Any]] = []
    for item in raw_dict.get("results", []):
        config = item.get("expectation_config", {}) or {}
        kwargs = config.get("kwargs", {}) or {}
        etype = config.get("type") or config.get("expectation_type")
        normalized.append(
            {
                "expectation_type": etype,
                "column": kwargs.get("column"),
                "kwargs": {k: v for k, v in kwargs.items() if k != "batch_id"},
                "success": item.get("success"),
                "element_count": (item.get("result") or {}).get("element_count"),
                "unexpected_count": (item.get("result") or {}).get("unexpected_count"),
                "unexpected_percent": (item.get("result") or {}).get("unexpected_percent"),
                "observed_value": (item.get("result") or {}).get("observed_value"),
                "partial_unexpected_list": (item.get("result") or {}).get("partial_unexpected_list"),
            }
        )
    return normalized, backend


@cleaning_step("validate_dataframe")
def validate_dataframe(
    df: pd.DataFrame,
    *,
    suite: Optional[Dict[str, Any]] = None,
    profile: Optional[Dict[str, Any]] = None,
    backend: str = "auto",
    fail_fast: bool = False,
    errors: str = "raise",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run data quality expectations against a DataFrame.

    CONTRACT NOTE: the DataFrame is returned unchanged; validation results are
    the report. This makes validation a drop-in pipeline stage (typically run
    both before and after cleaning to prove the cleaning improved quality).

    Parameters
    ----------
    suite : dict, optional
        A suite from ``build_expectation_suite_from_profile``, loaded from YAML,
        or generated by the LLM agent. If omitted, one is auto-derived from
        ``profile`` (or from a fresh profile of ``df``).
    backend : {"auto", "great_expectations", "pandas"}
        ``"auto"`` prefers GX and silently falls back to the pandas engine if GX
        is absent or raises. The backend actually used is always reported.
    fail_fast : bool
        Raise ``CleaningStepError`` when any expectation fails. Use in CI or a
        scheduled job; leave ``False`` in the interactive app.
    """
    if backend not in {"auto", "great_expectations", "pandas"}:
        raise ValueError("backend must be one of: auto, great_expectations, pandas")

    # ---- Resolve the suite --------------------------------------------------
    if suite is None:
        if profile is None:
            _, profile = profile_dataframe(df, errors="raise")
        suite = build_expectation_suite_from_profile(profile)
        suite_source = "auto_generated"
    else:
        suite_source = "provided"

    if not suite.get("expectations"):
        return df, {"status": "skipped", "reason": "suite contains no expectations"}

    # ---- Execute ------------------------------------------------------------
    used_backend = "pandas"
    fallback_reason: Optional[str] = None

    if backend in {"auto", "great_expectations"}:
        try:
            results, used_backend = _validate_with_gx(df, suite)
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            if backend == "great_expectations":
                raise CleaningStepError(
                    f"great_expectations backend failed: {exc}. "
                    "Retry with backend='pandas'."
                ) from exc
            logger.warning("GX unavailable/incompatible (%s); using pandas validator.", exc)
            results = _validate_with_pandas(df, suite)
            used_backend = "pandas"
    else:
        results = _validate_with_pandas(df, suite)

    # ---- Aggregate ----------------------------------------------------------
    evaluated = [r for r in results if r.get("success") is not None]
    successes = [r for r in evaluated if r["success"]]
    failures = [r for r in evaluated if not r["success"]]

    report: Dict[str, Any] = {
        "suite_name": suite.get("suite_name"),
        "suite_source": suite_source,
        "backend": used_backend,
        "backend_fallback_reason": fallback_reason,
        "success": len(failures) == 0,
        "statistics": {
            "expectations_evaluated": len(evaluated),
            "expectations_skipped": len(results) - len(evaluated),
            "successful": len(successes),
            "failed": len(failures),
            "success_percent": _pct(len(successes), len(evaluated)),
        },
        "failed_expectations": failures[:50],  # Cap: suites can be large.
        "results": results,
    }

    if failures and fail_fast:
        summary = ", ".join(
            f"{f['expectation_type']}({f.get('column')})" for f in failures[:5]
        )
        raise CleaningStepError(f"{len(failures)} expectation(s) failed: {summary}")

    return df, report


def summarize_validation_for_llm(results: Dict[str, Any], max_failures: int = 15) -> str:
    """Compact markdown summary of a validation run for prompts / RAG documents."""
    if not results:
        return "No validation results available."

    stats = results.get("statistics", {})
    status = "PASSED" if results.get("success") else "FAILED"
    lines = [
        f"## Validation {status} (suite: {results.get('suite_name', 'n/a')}, "
        f"engine: {results.get('backend', 'n/a')})",
        f"- {stats.get('successful', 0)}/{stats.get('expectations_evaluated', 0)} "
        f"expectations passed ({stats.get('success_percent', 0)}%)",
    ]

    failures = results.get("failed_expectations", [])
    if failures:
        lines += ["", "## Failures"]
        for failure in failures[:max_failures]:
            detail = []
            if failure.get("unexpected_count") is not None:
                detail.append(f"{failure['unexpected_count']} unexpected")
            if failure.get("observed_value") is not None:
                detail.append(f"observed={failure['observed_value']}")
            if failure.get("unexpected_values"):
                detail.append(f"examples={failure['unexpected_values'][:5]}")
            suffix = f" ({'; '.join(detail)})" if detail else ""
            lines.append(
                f"- `{failure.get('expectation_type')}` on "
                f"**{failure.get('column', 'table')}**{suffix}"
            )
        if len(failures) > max_failures:
            lines.append(f"- ... and {len(failures) - max_failures} more failures.")

    return "\n".join(lines)


# =============================================================================
# SECTION 8 - Registry
# =============================================================================
# Name -> callable map consumed by Step 2. The agent selects a tool by name from
# an LLM response, so a single explicit registry is safer than getattr() on the
# module namespace (which would expose private helpers to model-chosen calls).

CLEANING_FUNCTION_REGISTRY: Dict[str, Callable] = {
    "profile_dataframe": profile_dataframe,
    "normalize_missing_tokens": normalize_missing_tokens,
    "handle_missing_values": handle_missing_values,
    "standardize_column_names": standardize_column_names,
    "standardize_text": standardize_text,
    "standardize_types": standardize_types,
    "drop_exact_duplicates": drop_exact_duplicates,
    "fuzzy_deduplicate": fuzzy_deduplicate,
    "detect_outliers_iqr": detect_outliers_iqr,
    "detect_outliers_isolation_forest": detect_outliers_isolation_forest,
    "treat_outliers": treat_outliers,
    "validate_dataframe": validate_dataframe,
}


def _dependency_status() -> Dict[str, bool]:
    """Runtime capability report - surface this in the Streamlit sidebar."""
    return {
        "rapidfuzz": _HAS_RAPIDFUZZ,
        "pyod": _HAS_PYOD,
        "scikit_learn": _HAS_SKLEARN,
        "dedupe": _has_dedupe(),
        "great_expectations": _gx_available(),
    }


def _gx_available() -> bool:
    """Check GX availability without importing it eagerly at module load."""
    import importlib.util

    return importlib.util.find_spec("great_expectations") is not None


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    configure_logging()
    logger.info("Optional dependency status: %s", _dependency_status())

    demo = pd.DataFrame(
        {
            "Customer ID": [1, 2, 3, 4, 5, 5],
            " Full Name ": ["Acme Corp.", "ACME Corporation", "Globex", "Initech", "Hooli", "Hooli"],
            "revenue": ["$1,200.50", "1,200.50", "N/A", "(300)", "99999999", "99999999"],
            "signup_date": ["2023-01-15", "15/01/2023", "2023-03-01", None, "2023-05-20", "2023-05-20"],
            "active": ["Yes", "yes", "NO", "true", "0", "0"],
        }
    )

    # RECOMMENDED STAGE ORDER (Step 2 encodes this as the default plan):
    #   1. profile            - understand before touching anything
    #   2. column names       - stable identifiers for every later rule
    #   3. missing tokens     - "N/A" must become NaN before any null statistic
    #   4. text normalization - so dedup and categorical matching can work
    #   5. type coercion      - numeric/datetime stats need real dtypes
    #   6. deduplication      - shrink the frame before expensive row-wise work
    #   7. outliers           - BEFORE imputation, or imputed values contaminate
    #                           the distribution the fences are computed from
    #   8. missing values     - impute last, on clean, deduplicated data
    #   9. validation         - prove the result meets expectations
    frame, prof = profile_dataframe(demo)
    print(summarize_profile_for_llm(prof))

    frame, _ = standardize_column_names(frame)
    frame, _ = normalize_missing_tokens(frame)
    frame, _ = standardize_text(frame)
    frame, type_report = standardize_types(frame)
    frame, dup_report = drop_exact_duplicates(frame)
    # threshold=0.70 because this environment has no rapidfuzz (see report warnings).
    frame, fuzzy_report = fuzzy_deduplicate(frame, columns=["full_name"], threshold=0.70)
    frame, outlier_report = treat_outliers(frame, method="iqr", action="flag")
    frame, missing_report = handle_missing_values(frame, strategy="auto")
    frame, validation = validate_dataframe(frame)

    print("\nType conversions:", type_report["conversions"])
    print("Exact duplicates removed:", dup_report["duplicates_removed"])
    print("Fuzzy clusters:", fuzzy_report["clusters_found"],
          "| removed:", fuzzy_report["records_removed"])
    print("Outlier rows flagged:", outlier_report["rows_flagged"])
    print("Imputation plan:", {c: v.get("strategy") for c, v in missing_report["per_column"].items()})
    print("\n" + summarize_validation_for_llm(validation))
    print("\nFinal frame:\n", frame)
