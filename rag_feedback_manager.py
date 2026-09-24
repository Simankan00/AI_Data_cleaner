"""
rag_feedback_manager.py
=======================
Vector-based Retrieval-Augmented Generation (RAG) and persistent feedback memory
for the AI Automated Data Cleaning Agent.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Defensive imports for LangChain and ChromaDB
_HAS_RAG = False
try:
    from chromadb.config import Settings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_community.embeddings import FakeEmbeddings

    try:
        # Preferred local embedding model (no API key required)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _EMBEDDER = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception:
        # Lightweight zero-dependency fallback embedder
        _EMBEDDER = FakeEmbeddings(size=384)

    _HAS_RAG = True
except Exception:
    _HAS_RAG = False


class RAGFeedbackManager:
    """Manages cleaning rules, past user corrections, and semantic context retrieval."""

    def __init__(self, storage_dir: str = ".cleaning_rag_store") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.fallback_file = self.storage_dir / "rules_fallback.json"

        self.vector_store: Optional[Any] = None
        self.is_vector_enabled = False

        self._init_backend()
        self._init_seed_rules_if_empty()

    def _init_backend(self) -> None:
        """Initialize ChromaDB vector store with fallback to JSON storage."""
        if _HAS_RAG:
            try:
                chroma_path = str(self.storage_dir / "chroma_db")
                self.vector_store = Chroma(
                    collection_name="data_cleaning_memory",
                    embedding_function=_EMBEDDER,
                    persist_directory=chroma_path,
                )
                self.is_vector_enabled = True
                logger.info("ChromaDB vector store initialized successfully.")
                return
            except Exception as exc:
                logger.warning("Failed to initialize ChromaDB (%s). Using JSON storage.", exc)

        self.is_vector_enabled = False
        if not self.fallback_file.exists():
            self._save_fallback([])

    def _load_fallback(self) -> List[Dict[str, Any]]:
        if not self.fallback_file.exists():
            return []
        try:
            with open(self.fallback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_fallback(self, rules: List[Dict[str, Any]]) -> None:
        with open(self.fallback_file, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, default=str)

    def _init_seed_rules_if_empty(self) -> None:
        """Bootstrap default domain rules if memory is fresh."""
        existing = self.get_all_rules()
        if existing:
            return

        seed_rules = [
            {
                "column_pattern": "age",
                "issue_type": "has_missing_values",
                "action_type": "imputation",
                "action_value": "median",
                "rationale": "Age is typically right-skewed; median imputation prevents bias.",
            },
            {
                "column_pattern": "salary|income|revenue|price",
                "issue_type": "contains_outliers",
                "action_type": "outlier_treatment",
                "action_value": "clip",
                "rationale": "Financial figures have high natural variance; clip rather than remove.",
            },
            {
                "column_pattern": ".*",
                "issue_type": "null_sentinel",
                "action_type": "add_sentinel",
                "action_value": "N/A,null,unknown,none,-",
                "rationale": "Standard business export placeholders indicating missingness.",
            },
        ]

        for rule in seed_rules:
            self.add_rule(
                column_pattern=rule["column_pattern"],
                issue_type=rule["issue_type"],
                action_type=rule["action_type"],
                action_value=rule["action_value"],
                rationale=rule["rationale"],
                source="system_seed",
            )

    def add_rule(
        self,
        column_pattern: str,
        issue_type: str,
        action_type: str,
        action_value: str,
        rationale: str,
        source: str = "user_feedback",
    ) -> Dict[str, Any]:
        """Record a cleaning decision or feedback into memory."""
        rule_record = {
            "id": f"rule_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            "column_pattern": column_pattern.strip().lower(),
            "issue_type": issue_type.strip().lower(),
            "action_type": action_type.strip().lower(),
            "action_value": action_value.strip(),
            "rationale": rationale.strip(),
            "source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Save to Chroma vector store if available
        if self.is_vector_enabled and self.vector_store is not None:
            try:
                content = (
                    f"Column: {rule_record['column_pattern']}\n"
                    f"Issue: {rule_record['issue_type']}\n"
                    f"Action: {rule_record['action_type']} -> {rule_record['action_value']}\n"
                    f"Rationale: {rule_record['rationale']}"
                )
                doc = Document(page_content=content, metadata=rule_record)
                self.vector_store.add_documents([doc])
            except Exception as exc:
                logger.warning("Could not add document to vector store: %s", exc)

        # Always maintain the JSON fallback ledger
        rules = self._load_fallback()
        rules.append(rule_record)
        self._save_fallback(rules)
        return rule_record

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """Return all persisted cleaning rules."""
        return self._load_fallback()

    def query_relevant_rules(
        self,
        column_names: List[str],
        detected_issues: Dict[str, List[str]],
        query_text: Optional[str] = None,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve matching historical rules and corrections."""
        matched_rules: List[Dict[str, Any]] = []
        all_rules = self.get_all_rules()

        # Keyword / pattern matching over local rules
        lowered_cols = [c.lower() for c in column_names]
        for rule in all_rules:
            pattern = rule.get("column_pattern", "")
            if pattern == ".*" or any(pattern in col for col in lowered_cols):
                matched_rules.append(rule)
            elif rule.get("issue_type") in detected_issues:
                matched_rules.append(rule)

        # Vector semantic retrieval if specific query is provided
        if self.is_vector_enabled and self.vector_store is not None and query_text:
            try:
                docs = self.vector_store.similarity_search(query_text, k=k)
                for doc in docs:
                    meta = getattr(doc, "metadata", {})
                    if meta and meta not in matched_rules:
                        matched_rules.append(meta)
            except Exception as exc:
                logger.warning("Vector search query failed (%s); returning rule matches.", exc)

        # Deduplicate results by ID
        unique_matches: Dict[str, Dict[str, Any]] = {}
        for r in matched_rules:
            unique_matches[r["id"]] = r

        return list(unique_matches.values())

    def resolve_pipeline_overrides(
        self,
        columns: List[str],
        detected_issues: Dict[str, List[str]],
    ) -> Tuple[Dict[str, str], List[str], Dict[str, Any]]:
        """Convert stored context into direct parameters for the cleaning modules."""
        rules = self.query_relevant_rules(columns, detected_issues)
        column_imputation_strategies: Dict[str, str] = {}
        extra_null_tokens: List[str] = []
        outlier_actions: Dict[str, Any] = {}

        for rule in rules:
            pattern = rule["column_pattern"]
            action = rule["action_type"]
            val = rule["action_value"]

            if action == "add_sentinel":
                tokens = [t.strip() for t in val.split(",") if t.strip()]
                extra_null_tokens.extend(tokens)

            for col in columns:
                col_lower = col.lower()
                if pattern != ".*" and pattern not in col_lower:
                    continue

                if action == "imputation":
                    column_imputation_strategies[col] = val
                elif action == "outlier_treatment":
                    outlier_actions[col] = val

        return column_imputation_strategies, list(set(extra_null_tokens)), outlier_actions

    def reset_store(self) -> None:
        """Clear the memory store (useful for clean testing)."""
        if self.storage_dir.exists():
            shutil.rmtree(self.storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._init_backend()