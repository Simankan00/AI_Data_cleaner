from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import modules as M
from rag_feedback_manager import RAGFeedbackManager

logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    """End-to-end pipeline coordinator for automated tabular data cleaning."""

    def __init__(
        self,
        *,
        missing_strategy: str = "auto",
        outlier_method: str = "iqr",
        outlier_action: str = "clip",
        fuzzy_dedup: bool = False,
        fuzzy_columns: Optional[List[str]] = None,
        fuzzy_threshold: float = 0.88,
        text_case: Optional[str] = None,
        use_rag: bool = True,
        rag_manager: Optional[RAGFeedbackManager] = None,
        errors: str = "coerce",
    ) -> None:
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_action = outlier_action
        self.fuzzy_dedup = fuzzy_dedup
        self.fuzzy_columns = fuzzy_columns
        self.fuzzy_threshold = fuzzy_threshold
        self.text_case = text_case
        self.use_rag = use_rag
        self.errors = errors
        self.rag = rag_manager or RAGFeedbackManager()

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run the full cleaning sequence on a dataset."""
        M._ensure_dataframe(df, "DataCleaningPipeline")
        start_time = time.perf_counter()
        audit_log: List[Dict[str, Any]] = []

        current_df = df.copy()

        # Step 1: Initial Profiling
        _, initial_profile = M.profile_dataframe(current_df, errors=self.errors)
        audit_log.append(initial_profile)

        detected_issues = initial_profile.get("dataset_issues", {})
        raw_columns = list(current_df.columns)

        # Step 2: RAG Retrieval
        impute_overrides: Dict[str, str] = {}
        extra_tokens: List[str] = []
        rag_outlier_rules: Dict[str, Any] = {}
        retrieved_rules_count = 0

        if self.use_rag:
            impute_overrides, extra_tokens, rag_outlier_rules = (
                self.rag.resolve_pipeline_overrides(raw_columns, detected_issues)
            )
            retrieved_rules_count = len(impute_overrides) + len(extra_tokens) + len(rag_outlier_rules)

        # Step 3: Column Name Standardization
        current_df, col_report = M.standardize_column_names(current_df, case="snake", errors=self.errors)
        audit_log.append(col_report)
        col_map = col_report.get("column_mapping", {})

        # Remap RAG overrides to standardized names
        standardized_impute_overrides = {
            col_map.get(k, k): v for k, v in impute_overrides.items()
        }

        # Step 4: Missing Token Normalization
        current_df, token_report = M.normalize_missing_tokens(
            current_df, extra_tokens=extra_tokens, errors=self.errors
        )
        audit_log.append(token_report)

        # Step 5: Text Standardization
        current_df, text_report = M.standardize_text(
            current_df, case=self.text_case, strip=True, collapse_whitespace=True, errors=self.errors
        )
        audit_log.append(text_report)

        # Step 6: Data Type Inference & Standardization
        current_df, type_report = M.standardize_types(current_df, errors=self.errors)
        audit_log.append(type_report)

        # Step 7: Exact Deduplication
        current_df, exact_dup_report = M.drop_exact_duplicates(current_df, errors=self.errors)
        audit_log.append(exact_dup_report)

        # Step 8: Optional Fuzzy Deduplication
        if self.fuzzy_dedup:
            target_cols = [col_map.get(c, c) for c in (self.fuzzy_columns or [])]
            target_cols = [c for c in target_cols if c in current_df.columns]

            if not target_cols:
                # Fall back to text/categorical columns
                target_cols = [
                    c for c in current_df.columns
                    if pd.api.types.is_object_dtype(current_df[c]) or pd.api.types.is_string_dtype(current_df[c])
                ][:2]

            if target_cols:
                current_df, fuzzy_report = M.fuzzy_deduplicate(
                    current_df, columns=target_cols, threshold=self.fuzzy_threshold, errors=self.errors
                )
                audit_log.append(fuzzy_report)

        # Step 9: Outlier Treatment
        current_df, outlier_report = M.treat_outliers(
            current_df, method=self.outlier_method, action=self.outlier_action, errors=self.errors
        )
        audit_log.append(outlier_report)

        # Step 10: Missing Value Handling
        current_df, missing_report = M.handle_missing_values(
            current_df,
            strategy=self.missing_strategy,
            column_strategies=standardized_impute_overrides,
            errors=self.errors,
        )
        audit_log.append(missing_report)

        # Step 11: Final Validation against derived rules
        suite = M.build_expectation_suite_from_profile(initial_profile)
        # Adapt suite expectations to standardized column names
        for exp in suite.get("expectations", []):
            if "column" in exp.get("kwargs", {}):
                orig = exp["kwargs"]["column"]
                exp["kwargs"]["column"] = col_map.get(orig, orig)
            if "column_set" in exp.get("kwargs", {}):
                exp["kwargs"]["column_set"] = [col_map.get(c, c) for c in exp["kwargs"]["column_set"]]

        current_df, validation_report = M.validate_dataframe(
            current_df, suite=suite, backend="auto", errors=self.errors
        )
        audit_log.append(validation_report)

        # Step 12: Final Profiling
        _, final_profile = M.profile_dataframe(current_df, errors=self.errors)

        total_duration = time.perf_counter() - start_time
        init_ov = initial_profile.get("overview", {})
        final_ov = final_profile.get("overview", {})

        metrics_diff = {
            "initial_rows": init_ov.get("n_rows", 0),
            "final_rows": final_ov.get("n_rows", 0),
            "rows_removed": init_ov.get("n_rows", 0) - final_ov.get("n_rows", 0),
            "initial_missing_cells": init_ov.get("missing_cells", 0),
            "final_missing_cells": final_ov.get("missing_cells", 0),
            "missing_cells_imputed": max(0, init_ov.get("missing_cells", 0) - final_ov.get("missing_cells", 0)),
            "initial_duplicates": init_ov.get("duplicate_rows", 0),
            "final_duplicates": final_ov.get("duplicate_rows", 0),
            "validation_passed": validation_report.get("success", False),
            "validation_score_pct": validation_report.get("statistics", {}).get("success_percent", 0.0),
            "retrieved_rag_rules": retrieved_rules_count,
            "total_duration_seconds": round(total_duration, 4),
        }

        pipeline_summary = {
            "status": "success",
            "metrics_diff": metrics_diff,
            "audit_log": audit_log,
            "initial_profile": initial_profile,
            "final_profile": final_profile,
        }

        return current_df, M.make_json_safe(pipeline_summary)