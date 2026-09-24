from __future__ import annotations

import io
import json
import numpy as np
import pandas as pd
import streamlit as st

import modules as M
from pipeline import DataCleaningPipeline
from rag_feedback_manager import RAGFeedbackManager

st.set_page_config(
    page_title="AI Data Cleaning Agent",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_rag_manager() -> RAGFeedbackManager:
    return RAGFeedbackManager()


def create_demo_data() -> pd.DataFrame:
    """Generate a realistic messy dataset for immediate verification."""
    return pd.DataFrame({
        "Customer ID": [101, 102, 103, 104, 105, 105, 106, 107],
        " Full Name ": [
            "Acme Corp.", "ACME Corporation", "Globex LLC", "Initech",
            "Hooli", "Hooli", "  Umbrella Corp  ", "Wayne Enterprises"
        ],
        "revenue": ["$1,200.50", "1,200.50", "N/A", "(450.00)", "99999999", "99999999", "5000", "None"],
        "signup_date": ["2023-01-15", "15/01/2023", "2023-03-01", "2023-04-10", "2023-05-20", "2023-05-20", "invalid_date", "2023-07-01"],
        "age": [34, 34, np.nan, 29, 45, 45, 140, 52],
        "status": ["Active", "active", "churned", "Active", "suspended", "suspended", "UNKNOWN", "active"],
    })


def main():
    rag = get_rag_manager()

    st.title("🧹 AI Agent for Automated Data Cleaning")
    st.markdown("Automated tabular profiling, ML-driven transformations, and RAG-powered feedback loops.")

    # ---- Sidebar Controls ----
    st.sidebar.header("⚙️ Pipeline Configuration")

    missing_strategy = st.sidebar.selectbox(
        "Missing Value Strategy",
        ["auto", "median", "mean", "mode", "constant", "knn", "drop_rows", "drop_columns"],
        index=0,
        help="Strategy to impute or drop missing cells.",
    )

    outlier_method = st.sidebar.selectbox("Outlier Detection", ["iqr", "isolation_forest", "both"], index=0)
    outlier_action = st.sidebar.selectbox("Outlier Action", ["clip", "flag", "nan", "remove"], index=0)

    st.sidebar.subheader("Entity Resolution")
    enable_fuzzy = st.sidebar.checkbox("Enable Fuzzy Deduplication", value=False)
    fuzzy_threshold = st.sidebar.slider("Fuzzy Match Threshold", 0.70, 1.0, 0.88, 0.02) if enable_fuzzy else 0.88

    st.sidebar.subheader("Text & Types")
    text_case = st.sidebar.selectbox("Text Standardization Case", [None, "title", "lower", "upper"], index=0)

    use_rag = st.sidebar.checkbox("Enable RAG Memory & Rules", value=True)

    # Status indicator
    dep_status = M._dependency_status()
    with st.sidebar.expander("🛠️ Engine Capabilities"):
        for k, v in dep_status.items():
            st.write(f"- **{k}**: {'✅ Active' if v else '⚠️ Fallback'}")
        st.write(f"- **Vector Store**: {'✅ ChromaDB' if rag.is_vector_enabled else '⚠️ JSON Store'}")

    # ---- Input Data Section ----
    st.subheader("1. Ingest Data")
    col_up1, col_up2 = st.columns([3, 1])

    uploaded_file = col_up1.file_uploader("Upload tabular data", type=["csv", "xlsx", "json"])
    load_demo = col_up2.button("Load Sample Messy Data", use_container_width=True)

    if "raw_df" not in st.session_state:
        st.session_state["raw_df"] = None

    if load_demo:
        st.session_state["raw_df"] = create_demo_data()
        st.success("Sample dirty dataset loaded!")
    elif uploaded_file is not None:
        try:
            name = uploaded_file.name.lower()
            if name.endswith(".csv"):
                st.session_state["raw_df"] = pd.read_csv(uploaded_file)
            elif name.endswith(".xlsx"):
                st.session_state["raw_df"] = pd.read_excel(uploaded_file)
            elif name.endswith(".json"):
                st.session_state["raw_df"] = pd.read_json(uploaded_file)
            st.success(f"Loaded '{uploaded_file.name}' successfully.")
        except Exception as exc:
            st.error(f"Error loading file: {exc}")

    df_raw = st.session_state["raw_df"]

    if df_raw is None:
        st.info("Upload a file or click 'Load Sample Messy Data' to proceed.")
        return

    # Raw Data Preview
    st.write("### Raw Data Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

    # ---- Pipeline Execution ----
    st.subheader("2. Execute Autonomous Cleaning")
    if st.button("🚀 Run AI Cleaning Pipeline", type="primary"):
        with st.spinner("Profiling, cleaning, and validating dataset..."):
            pipeline = DataCleaningPipeline(
                missing_strategy=missing_strategy,
                outlier_method=outlier_method,
                outlier_action=outlier_action,
                fuzzy_dedup=enable_fuzzy,
                fuzzy_threshold=fuzzy_threshold,
                text_case=text_case,
                use_rag=use_rag,
                rag_manager=rag,
                errors="coerce",
            )
            cleaned_df, summary = pipeline.run(df_raw)
            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["cleaning_summary"] = summary

    # ---- Results Presentation ----
    if "cleaned_df" in st.session_state:
        cleaned_df: pd.DataFrame = st.session_state["cleaned_df"]
        summary = st.session_state["cleaning_summary"]
        diff = summary["metrics_diff"]

        st.subheader("3. Cleaning Results & Impact")

        # Metric Badges
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{diff['final_rows']}", f"-{diff['rows_removed']} removed" if diff['rows_removed'] else "No change")
        m2.metric("Missing Cells", f"{diff['final_missing_cells']}", f"-{diff['missing_cells_imputed']} fixed")
        m3.metric("Duplicates", f"{diff['final_duplicates']}", f"-{diff['initial_duplicates'] - diff['final_duplicates']} removed")
        m4.metric("Quality Score", f"{diff['validation_score_pct']}%", "Validated")
        m5.metric("RAG Rules Applied", f"{diff['retrieved_rag_rules']}")

        # Tabs for detailed inspection
        tab1, tab2, tab3, tab4 = st.tabs(["Cleaned Data", "Audit Trail", "Profile Comparison", "Downloads"])

        with tab1:
            st.dataframe(cleaned_df, use_container_width=True)

        with tab2:
            st.write("#### Step-by-Step Action Log")
            for log in summary.get("audit_log", []):
                step = log.get("step", "Step")
                status = log.get("status", "success")
                duration = log.get("duration_seconds", 0)
                with st.expander(f"🔹 {step} ({status}) - {duration}s"):
                    st.json(log)

        with tab3:
            st.write("#### Initial vs. Final Dataset Issues")
            c1, c2 = st.columns(2)
            c1.markdown("**Before Cleaning:**")
            c1.json(summary["initial_profile"].get("dataset_issues", {}))
            c2.markdown("**After Cleaning:**")
            c2.json(summary["final_profile"].get("dataset_issues", {}))

        with tab4:
            st.write("#### Export Cleaned Dataset")
            c_csv, c_xlsx = st.columns(2)

            csv_buffer = io.StringIO()
            cleaned_df.to_csv(csv_buffer, index=False)
            c_csv.download_button(
                "📥 Download as CSV",
                data=csv_buffer.getvalue(),
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                cleaned_df.to_excel(writer, index=False, sheet_name="CleanedData")
            c_xlsx.download_button(
                "📥 Download as Excel (.xlsx)",
                data=excel_buffer.getvalue(),
                file_name="cleaned_dataset.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        # ---- RAG Feedback Section ----
        st.subheader("4. Human-in-the-Loop Feedback & Learning")
        with st.form("feedback_form"):
            st.markdown("Teach the AI Agent how to treat specific data attributes for future runs:")
            f_col1, f_col2, f_col3 = st.columns(3)
            col_target = f_col1.selectbox("Target Column", list(cleaned_df.columns))
            action_type = f_col2.selectbox("Action Category", ["imputation", "outlier_treatment", "add_sentinel"])
            action_value = f_col3.text_input("Preferred Action / Value", placeholder="e.g., median, clip, or N/A")
            rationale = st.text_input("Rationale", placeholder="e.g., 'Revenue outliers should always be clipped'")

            submit_feedback = st.form_submit_button("💾 Save Rule to RAG Memory")

            if submit_feedback and action_value:
                rule = rag.add_rule(
                    column_pattern=col_target,
                    issue_type="user_defined",
                    action_type=action_type,
                    action_value=action_value,
                    rationale=rationale or "User feedback rule",
                    source="ui_feedback",
                )
                st.success(f"Rule registered into vector store: {rule['action_type']} -> {rule['action_value']} on '{col_target}'")


if __name__ == "__main__":
    main()