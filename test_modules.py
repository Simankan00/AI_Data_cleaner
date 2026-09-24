import logging
import numpy as np
import pandas as pd

from modules import (
    CleaningStepError, profile_dataframe, normalize_missing_tokens,
    handle_missing_values, standardize_column_names, standardize_text,
    standardize_types, drop_exact_duplicates, fuzzy_deduplicate,
    detect_outliers_iqr, detect_outliers_isolation_forest, treat_outliers,
    build_expectation_suite_from_profile, validate_dataframe,
    summarize_validation_for_llm, make_json_safe,
)

logging.disable(logging.CRITICAL)  # Keep the output readable.
results = []


def check(name, fn):
    try:
        fn()
        results.append((True, name, ""))
    except Exception as exc:
        results.append((False, name, f"{type(exc).__name__}: {exc}"))


# 1. Empty DataFrame
def t_empty():
    df = pd.DataFrame()
    _, p = profile_dataframe(df)
    assert p["overview"]["n_rows"] == 0
    _, _ = standardize_types(df)
    _, _ = handle_missing_values(df)
    _, _ = drop_exact_duplicates(df)
check("empty dataframe", t_empty)


# 2. All-null column
def t_all_null():
    df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
    _, p = profile_dataframe(df)
    assert p["columns"]["a"]["semantic_type"] == "empty"
    assert "empty_column" in p["columns"]["a"]["issues"]
    out, rep = handle_missing_values(df, strategy="auto")
    assert "a" in rep["columns_dropped"], rep
check("all-null column", t_all_null)


# 3. Single row
def t_single_row():
    df = pd.DataFrame({"x": [5], "y": ["hello"]})
    _, _ = profile_dataframe(df)
    _, r = detect_outliers_iqr(df)
    assert r["bounds"]["x"]["skipped"] is True
    _, r2 = detect_outliers_isolation_forest(df)
    assert r2["status"] == "skipped"
check("single row", t_single_row)


# 4. Unhashable cell values (JSON-derived data)
def t_unhashable():
    df = pd.DataFrame({"tags": [["a", "b"], ["c"], ["a", "b"]], "n": [1, 2, 3]})
    _, p = profile_dataframe(df)
    assert p["overview"]["duplicate_rows"] == -1
    _, _ = drop_exact_duplicates(df)
check("unhashable cells", t_unhashable)


# 5. Duplicate column names must raise a clear error
def t_dupe_cols():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    try:
        profile_dataframe(df)
        raise AssertionError("should have raised")
    except CleaningStepError as exc:
        assert "duplicate column names" in str(exc)
check("duplicate column names rejected", t_dupe_cols)


# 6. errors='coerce' returns the frame untouched with a failure report
def t_coerce():
    df = pd.DataFrame({"a": [1, 2, 3]})
    out, rep = handle_missing_values(df, strategy="mean", column_strategies=None,
                                     columns=["does_not_exist"], errors="coerce")
    assert rep["status"] == "failed", rep
    assert out.equals(df)
check("errors='coerce' failure path", t_coerce)


# 7. errors='raise' actually raises
def t_raise():
    df = pd.DataFrame({"a": [1, 2, 3]})
    try:
        handle_missing_values(df, columns=["nope"], errors="raise")
        raise AssertionError("should have raised")
    except CleaningStepError:
        pass
check("errors='raise' failure path", t_raise)


# 8. Leading zeros must survive type standardization
def t_leading_zeros():
    df = pd.DataFrame({"zip": ["01234", "05678", "09999"], "n": ["1", "2", "3"]})
    out, rep = standardize_types(df)
    assert out["zip"].iloc[0] == "01234", out["zip"].tolist()
    assert pd.api.types.is_numeric_dtype(out["n"])
check("leading zeros preserved", t_leading_zeros)


# 9. Text column must NOT be destroyed by a few date-looking values
def t_no_false_conversion():
    vals = ["note A", "note B", "2023-01-01"] + [f"note {i}" for i in range(20)]
    df = pd.DataFrame({"notes": vals})
    out, _ = standardize_types(df)
    assert not pd.api.types.is_datetime64_any_dtype(out["notes"])
check("partial matches do not trigger conversion", t_no_false_conversion)


# 10. Currency / accounting negatives / percentages
def t_currency():
    df = pd.DataFrame({"amt": ["$1,234.56", "(789.00)", "€ 42", "1 000,5".replace(",", ".")]})
    out, rep = standardize_types(df)
    assert pd.api.types.is_numeric_dtype(out["amt"]), rep
    assert out["amt"].iloc[1] == -789.0, out["amt"].tolist()
    assert out["amt"].iloc[0] == 1234.56
check("currency and accounting negatives", t_currency)


# 11. Categorical dtype + mode imputation (needs add_categories)
def t_categorical():
    s = pd.Series(["a", "b", "a", None], dtype="category")
    df = pd.DataFrame({"c": s})
    out, rep = handle_missing_values(df, strategy="mode")
    assert out["c"].isna().sum() == 0, rep
check("categorical mode imputation", t_categorical)


# 12. Constant column -> zero IQR must be skipped, not flag everything
def t_constant():
    df = pd.DataFrame({"k": [7] * 20})
    _, rep = detect_outliers_iqr(df)
    assert rep["bounds"]["k"]["skipped"] is True
check("zero-IQR column skipped", t_constant)


# 13. Isolation Forest on a real multivariate anomaly
def t_iforest():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "age": np.concatenate([rng.normal(40, 5, 200), [8]]),
        "salary": np.concatenate([rng.normal(60000, 8000, 200), [500000]]),
    })
    mask, rep = detect_outliers_isolation_forest(df, contamination=0.02)
    assert mask["__is_outlier__"].iloc[-1], rep   # the odd combination is caught
    assert rep["backend"] in {"pyod.IForest", "sklearn.IsolationForest"}, rep["backend"]
check("isolation forest catches multivariate anomaly", t_iforest)


# 14. NaN treatment on an integer column (pandas 3.x upcast strictness)
def t_nan_on_int():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100000]})
    out, rep = treat_outliers(df, method="iqr", action="nan")
    assert out["v"].isna().sum() == 1, rep
    assert pd.api.types.is_float_dtype(out["v"])
check("nan action upcasts int column", t_nan_on_int)


# 15. Clip action
def t_clip():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100000]})
    out, rep = treat_outliers(df, method="iqr", action="clip")
    assert out["v"].max() < 100000, rep
check("clip action winsorizes", t_clip)


# 16. Remove action
def t_remove():
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100000]})
    out, rep = treat_outliers(df, method="iqr", action="remove")
    assert len(out) == 9, rep
check("remove action drops rows", t_remove)


# 17. Fuzzy dedup preserves a non-default index
def t_fuzzy_index():
    df = pd.DataFrame(
        {"name": ["Acme Corporation", "Acme Corporatlon", "Globex", "Initech"]},
        index=["w", "x", "y", "z"],
    )
    out, rep = fuzzy_deduplicate(df, columns=["name"], threshold=0.85)
    assert rep["records_removed"] == 1, rep
    assert list(out.index) == ["w", "y", "z"], list(out.index)
check("fuzzy dedup preserves index labels", t_fuzzy_index)


# 18. Fuzzy dedup keeps the most complete record
def t_keep_complete():
    df = pd.DataFrame({
        "name": ["Acme Corporation", "Acme Corporatlon"],
        "email": [None, "hi@acme.com"],
        "phone": [None, "555-1234"],
    })
    out, rep = fuzzy_deduplicate(df, columns=["name"], threshold=0.85, keep="most_complete")
    assert len(out) == 1 and out["email"].iloc[0] == "hi@acme.com", out
check("keep='most_complete' picks richest row", t_keep_complete)


# 19. Fuzzy dedup with a blocking key must not merge across blocks
def t_blocking():
    df = pd.DataFrame({
        "name": ["Acme Corp", "Acme Corp", "Acme Corp"],
        "country": ["US", "UK", "US"],
    })
    out, rep = fuzzy_deduplicate(df, columns=["name"], blocking_keys=["country"], threshold=0.9)
    assert len(out) == 2, (len(out), rep)
check("blocking key prevents cross-block merges", t_blocking)


# 20. Exact dedup with case/whitespace normalization
def t_exact_norm():
    df = pd.DataFrame({"n": ["  ACME  ", "acme", "globex"]})
    out, rep = drop_exact_duplicates(df)
    assert rep["duplicates_removed"] == 1
    assert out["n"].iloc[0] == "  ACME  "  # survivor keeps original formatting
check("exact dedup normalizes but preserves survivor", t_exact_norm)


# 21. Null sentinel normalization
def t_sentinels():
    df = pd.DataFrame({"a": ["N/A", "-", "real", "", "NULL", "unknown"]})
    out, rep = normalize_missing_tokens(df)
    assert out["a"].isna().sum() == 5, out["a"].tolist()
    assert rep["total_converted"] == 5
check("null sentinels converted", t_sentinels)


# 22. Unicode / whitespace normalization
def t_unicode():
    df = pd.DataFrame({"a": ["Caf\u00e9\u00a0 ", "  multiple   spaces ", "\u201cquoted\u201d"]})
    out, rep = standardize_text(df)
    assert out["a"].iloc[1] == "multiple spaces", repr(out["a"].iloc[1])
    assert not out["a"].iloc[0].endswith(" ")
check("unicode and whitespace normalization", t_unicode)


# 23. Column name standardization incl. collisions
def t_colnames():
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["Total Revenue", "totalRevenue", "2024", " x# "])
    out, rep = standardize_column_names(df)
    cols = list(out.columns)
    assert cols[0] == "total_revenue" and cols[1] == "total_revenue_2", cols
    assert cols[2] == "col_2024", cols
    assert len(set(cols)) == 4
check("column name standardization + collisions", t_colnames)


# 24. Suite generation + validation round-trip, and a deliberate failure
def t_validation():
    good = pd.DataFrame({
        "id": range(1, 101),
        "status": ["active", "churned"] * 50,
        "amount": np.linspace(10, 500, 100),
    })
    _, prof = profile_dataframe(good)
    suite = build_expectation_suite_from_profile(prof)
    _, res = validate_dataframe(good, suite=suite)
    assert res["success"], res["failed_expectations"]

    bad = good.copy()
    bad.loc[0:10, "status"] = "UNKNOWN_STATUS"     # breaks the value set
    bad.loc[0:10, "id"] = 1                        # breaks uniqueness
    _, res2 = validate_dataframe(bad, suite=suite)
    assert not res2["success"]
    types = {f["expectation_type"] for f in res2["failed_expectations"]}
    assert "expect_column_values_to_be_in_set" in types, types
    assert "expect_column_values_to_be_unique" in types, types
    assert "## Validation FAILED" in summarize_validation_for_llm(res2)
check("expectation suite round-trip + failure detection", t_validation)


# 25. fail_fast raises
def t_fail_fast():
    df = pd.DataFrame({"a": [1, 1, 1]})
    suite = {"suite_name": "s", "expectations": [
        {"expectation_type": "expect_column_values_to_be_unique", "kwargs": {"column": "a"}}]}
    try:
        validate_dataframe(df, suite=suite, fail_fast=True)
        raise AssertionError("should have raised")
    except CleaningStepError:
        pass
check("fail_fast raises on failure", t_fail_fast)


# 26. Reports must be JSON-serializable (Chroma metadata requirement)
def t_json():
    import json
    df = pd.DataFrame({"a": [1, 2, np.nan, np.inf], "b": pd.date_range("2024-01-01", periods=4)})
    _, prof = profile_dataframe(df)
    json.dumps(prof)                                   # must not raise
    _, rep = handle_missing_values(df, strategy="auto")
    json.dumps(rep)
    _, res = validate_dataframe(df)
    json.dumps(res)
    assert make_json_safe(np.int64(5)) == 5
    assert make_json_safe(float("nan")) is None
check("all reports JSON-serializable", t_json)


# 27. KNN imputation
def t_knn():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 50), "b": rng.normal(5, 2, 50)})
    df.loc[0:5, "a"] = np.nan
    out, rep = handle_missing_values(df, strategy="knn", knn_neighbors=3)
    assert out["a"].isna().sum() == 0, rep
check("KNN imputation", t_knn)


# 28. Missingness indicators
def t_indicator():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})
    out, rep = handle_missing_values(df, strategy="median", add_indicator=True)
    assert "a__was_missing" in out.columns
    assert out["a__was_missing"].sum() == 2
check("missingness indicator columns", t_indicator)


# 29. Row-level drop threshold
def t_row_threshold():
    df = pd.DataFrame({"a": [1, np.nan, 3], "b": [1, np.nan, 3], "c": [1, np.nan, 3]})
    out, rep = handle_missing_values(df, strategy="leave", row_missing_threshold=0.5)
    assert len(out) == 2, rep
check("row_missing_threshold drops sparse rows", t_row_threshold)


# 30. Mixed python types in one object column
def t_mixed_types():
    df = pd.DataFrame({"m": [1, "two", 3.0, None, True]})
    _, p = profile_dataframe(df)
    out, _ = standardize_text(df)
    assert len(out) == 5
check("mixed python types survive profiling", t_mixed_types)


# -------------------------------------------------------------------------
print("\n" + "=" * 72)
passed = sum(1 for ok, _, _ in results if ok)
for ok, name, err in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if err:
        print(f"         -> {err}")
print("=" * 72)
print(f"{passed}/{len(results)} passed")
