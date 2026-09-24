# 🧹 AI Agent for Automated Data Cleaning

An autonomous, end-to-end data preparation pipeline that transforms messy tabular data (CSV, Excel, JSON) into high-quality, analysis-ready datasets. 

Unlike static cleaning scripts, this agent features a **Human-in-the-Loop RAG (Retrieval-Augmented Generation) Memory**. It learns from your custom business rules and past corrections, storing them in a local vector database to automatically apply them to future datasets.

---
Streamlit Demo :- 

## 🚀 Overview

Data scientists spend up to 80% of their time cleaning data. This tool automates the heavy lifting by profiling datasets, detecting anomalies, imputing missing values, and validating quality, all orchestrated through a user-friendly Streamlit web dashboard.

## ✨ Key Features

* **Autonomous Pipeline:** Automatically handles type coercion, null sentinel normalization (e.g., converting "N/A" to `NaN`), and text standardization.
* **Smart Imputation:** Fills missing values using statistical strategies (mean, median, mode) or advanced Machine Learning (KNN, Iterative Imputation).
* **Entity Resolution:** Drops exact duplicates and uses ML-driven fuzzy string matching (`dedupe` / `rapidfuzz`) to merge near-duplicates.
* **Multivariate Outlier Detection:** Employs statistical fences (IQR) and Machine Learning (Isolation Forests via `PyOD`) to detect, flag, or clip anomalous records.
* **Data Quality Validation:** Auto-generates expectation suites and validates the final dataset using `great_expectations`.
* **🧠 RAG Feedback Memory:** Uses `LangChain` and `ChromaDB` to store your specific cleaning rules (e.g., "always clip revenue outliers") and retrieves them semantically for future runs.

---

## 🏗️ Pipeline Architecture

The agent processes data in a strict, deterministic sequence to ensure data integrity:
1. **Initial Profiling:** Scans the dataset for missing rates, semantic types, and issues.
2. **Context Retrieval (RAG):** Queries ChromaDB for historical user rules.
3. **Standardization:** Normalizes column names (snake_case) and text casing.
4. **Type Coercion:** Safely casts strings to numerics, booleans, and datetimes.
5. **Deduplication:** Removes exact matches and fuzzy near-duplicates.
6. **Outlier Treatment:** Flags or clips univariate and multivariate anomalies.
7. **Missing Value Handling:** Imputes remaining nulls based on user rules or heuristics.
8. **Validation:** Asserts data quality and generates a final report.

---

## 📂 Repository Structure

```text
ai-data-cleaning-agent/
├── .cleaning_rag_store/         # Persistent ChromaDB vector store for RAG rules
├── data/                        # Folder for your raw datasets
├── modules.py                   # Core stateless cleaning primitives
├── rag_feedback_manager.py      # Vector-based RAG store & feedback engine
├── pipeline.py                  # Pipeline orchestrator chaining modules + RAG
├── app.py                       # Streamlit interactive web dashboard
├── test_modules.py              # Pytest edge-case stress tests
├── test_pipeline.py             # Pytest end-to-end integration tests
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container build recipe
└── docker-compose.yml           # One-command Docker orchestration

```

## Usage Guide

Ingest Data: Upload your .csv, .xlsx, or .json file via the sidebar, or click "Load Sample Messy Data" to test the system.

Configure Pipeline: Use the sidebar to set your default fallback rules for missing values, outlier detection, and entity resolution.

Execute Cleaning: Click Run AI Cleaning Pipeline. The agent will profile the data, query the RAG memory for your past preferences, and clean the dataset.

Review Results: Explore the dashboard to see exactly what changed:

Cleaned Data: Preview the transformed table.

Audit Trail: View the exact actions taken during every step of the pipeline.

Profile Comparison: Compare the dataset's issues before and after execution.

Teach the Agent: Scroll to the "Human-in-the-Loop Feedback" section. If the agent missed something, create a rule (e.g., Target Column: salary, Action: imputation, Value: median). Click Save Rule to RAG Memory. The agent will automatically apply this rule to similar data next time!

Export: Download your cleaned data as a CSV or Excel file.
