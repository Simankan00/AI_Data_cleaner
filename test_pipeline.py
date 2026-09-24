import numpy as np
import pandas as pd
import pytest

from pipeline import DataCleaningPipeline
from rag_feedback_manager import RAGFeedbackManager


@pytest.fixture
def clean_rag_store(tmp_path):
    """Provide an isolated temporary directory for RAG memory testing."""
    manager = RAGFeedbackManager(storage_dir=str(tmp_path / ".rag_test"))
    return manager


@pytest.fixture
def messy_dataframe():
    """Construct an intentional dirty dataset."""
    return pd.DataFrame({
        "Full Name": ["Alice", "Bob", "Alice", "Charlie", "David"],
        "age": [25, np.nan, 25, 150, 40],
        "salary": ["$50,000", "$60,000", "$50,000", "N/A", "(70,000)"],
        "signup_date": ["2023-01-01", "2023-02-01", "2023-01-01", "invalid", "2023-05-01"],
    })


def test_pipeline_end_to_end(clean_rag_store, messy_dataframe):
    pipeline = DataCleaningPipeline(
        missing_strategy="median",
        outlier_method="iqr",
        outlier_action="clip",
        use_rag=False,
        rag_manager=clean_rag_store,
    )
    cleaned_df, summary = pipeline.run(messy_dataframe)

    assert summary["status"] == "success"
    assert "full_name" in cleaned_df.columns  # Column standardized
    assert cleaned_df["salary"].isna().sum() == 0  # Missing filled
    assert len(cleaned_df) == 4  # Duplicate row dropped
    assert cleaned_df["age"].max() <= 150  # Outlier addressed


def test_rag_rule_injection(clean_rag_store, messy_dataframe):
    # Register an explicit business rule: impute salary with zero
    clean_rag_store.add_rule(
        column_pattern="salary",
        issue_type="has_missing_values",
        action_type="imputation",
        action_value="constant",
        rationale="Unreported salary defaults to 0.",
    )

    pipeline = DataCleaningPipeline(
        missing_strategy="median",
        use_rag=True,
        rag_manager=clean_rag_store,
    )
    cleaned_df, summary = pipeline.run(messy_dataframe)

    assert summary["metrics_diff"]["retrieved_rag_rules"] > 0
    assert cleaned_df["salary"].notna().all()


def test_empty_dataframe_resilience(clean_rag_store):
    empty_df = pd.DataFrame()
    pipeline = DataCleaningPipeline(use_rag=False, rag_manager=clean_rag_store, errors="coerce")
    out_df, summary = pipeline.run(empty_df)
    assert len(out_df) == 0
    assert summary["status"] == "success"