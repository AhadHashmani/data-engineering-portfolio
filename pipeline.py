"""
Modular ETL Pipeline Framework
================================
A production-style ETL pipeline with pluggable extractors, transformers,
and loaders. Demonstrates chunked processing, schema validation, logging,
retry logic, and basic data quality checks.

Author: Ahad Hashmani
"""

import logging
import time
import sqlite3
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("etl_pipeline")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    chunk_size: int = 10_000
    max_retries: int = 3
    retry_delay: float = 1.0
    null_threshold: float = 0.20    # flag columns with >20% nulls
    output_db: str = "pipeline_output.db"
    required_columns: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Data Quality Report
# ─────────────────────────────────────────────
@dataclass
class QualityReport:
    total_rows: int = 0
    rows_after_clean: int = 0
    null_flags: Dict[str, float] = field(default_factory=dict)
    schema_errors: List[str] = field(default_factory=list)
    duplicate_rows_dropped: int = 0
    started_at: str = ""
    finished_at: str = ""

    def summary(self) -> str:
        lines = [
            "─" * 50,
            "  DATA QUALITY REPORT",
            "─" * 50,
            f"  Started     : {self.started_at}",
            f"  Finished    : {self.finished_at}",
            f"  Total rows  : {self.total_rows:,}",
            f"  Clean rows  : {self.rows_after_clean:,}",
            f"  Duplicates  : {self.duplicate_rows_dropped:,}",
        ]
        if self.null_flags:
            lines.append("  High-null columns:")
            for col, pct in self.null_flags.items():
                lines.append(f"    {col}: {pct:.1%} null")
        if self.schema_errors:
            lines.append("  Schema errors:")
            for err in self.schema_errors:
                lines.append(f"    {err}")
        lines.append("─" * 50)
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Abstract Base Classes
# ─────────────────────────────────────────────
class Extractor(ABC):
    @abstractmethod
    def extract(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        """Yield DataFrames in chunks."""


class Transformer(ABC):
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations and return cleaned DataFrame."""


class Loader(ABC):
    @abstractmethod
    def load(self, df: pd.DataFrame, table_name: str) -> None:
        """Persist a DataFrame to the target store."""


# ─────────────────────────────────────────────
# Concrete Implementations
# ─────────────────────────────────────────────
class CSVExtractor(Extractor):
    """Extract data from a CSV file in chunks."""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Source file not found: {filepath}")

    def extract(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        log.info("Extracting from %s (chunk_size=%d)", self.filepath, chunk_size)
        for chunk in pd.read_csv(self.filepath, chunksize=chunk_size):
            yield chunk


class SyntheticExtractor(Extractor):
    """Generate synthetic financial data for testing."""

    def __init__(self, n_rows: int = 50_000):
        self.n_rows = n_rows

    def extract(self, chunk_size: int) -> Generator[pd.DataFrame, None, None]:
        log.info("Generating %d synthetic rows (chunk_size=%d)", self.n_rows, chunk_size)
        rng = np.random.default_rng(42)
        start = 0

        while start < self.n_rows:
            end = min(start + chunk_size, self.n_rows)
            size = end - start
            chunk = pd.DataFrame({
                "transaction_id": [f"TXN_{i:08d}" for i in range(start, end)],
                "timestamp": pd.date_range("2024-01-01", periods=size, freq="min"),
                "amount": rng.lognormal(mean=4.5, sigma=1.2, size=size).round(2),
                "category": rng.choice(["retail", "food", "travel", "utilities", None],
                                        size=size, p=[0.35, 0.30, 0.20, 0.10, 0.05]),
                "user_id": rng.integers(1000, 9999, size=size),
                "is_flagged": rng.choice([True, False], size=size, p=[0.03, 0.97]),
            })
            start = end
            yield chunk


class DataTransformer(Transformer):
    """Clean, validate, and enrich a DataFrame."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        original_len = len(df)

        # Drop full duplicates
        df = df.drop_duplicates()

        # Standardise string columns
        str_cols = df.select_dtypes(include="object").columns
        for col in str_cols:
            df[col] = df[col].str.strip().str.lower()

        # Drop rows where required columns are null
        if self.config.required_columns:
            df = df.dropna(subset=self.config.required_columns)

        # Clip negative amounts if present
        if "amount" in df.columns:
            df["amount"] = df["amount"].clip(lower=0)

        # Feature engineering: log-amount bucket
        if "amount" in df.columns:
            df["amount_log"] = np.log1p(df["amount"]).round(4)
            df["amount_bucket"] = pd.cut(
                df["amount"],
                bins=[0, 50, 200, 1000, np.inf],
                labels=["micro", "small", "medium", "large"],
            )

        # Parse timestamps
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.day_name()

        log.debug("Transform: %d → %d rows", original_len, len(df))
        return df


class SQLiteLoader(Loader):
    """Load DataFrames into a local SQLite database."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def load(self, df: pd.DataFrame, table_name: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            df.to_sql(table_name, conn, if_exists="append", index=False)
        log.debug("Loaded %d rows → table '%s'", len(df), table_name)


# ─────────────────────────────────────────────
# Pipeline Orchestrator
# ─────────────────────────────────────────────
class Pipeline:
    """
    Orchestrates extract → transform → load with retry logic,
    chunked processing, and a data quality report.
    """

    def __init__(
        self,
        extractor: Extractor,
        transformer: Transformer,
        loader: Loader,
        config: PipelineConfig,
        table_name: str = "processed_data",
    ):
        self.extractor = extractor
        self.transformer = transformer
        self.loader = loader
        self.config = config
        self.table_name = table_name
        self.report = QualityReport()

    def _with_retry(self, func, *args, **kwargs) -> Any:
        for attempt in range(1, self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                log.warning("Attempt %d/%d failed: %s", attempt, self.config.max_retries, exc)
                if attempt == self.config.max_retries:
                    raise
                time.sleep(self.config.retry_delay * attempt)

    def _quality_check(self, df: pd.DataFrame) -> None:
        for col in df.columns:
            null_pct = df[col].isna().mean()
            if null_pct > self.config.null_threshold:
                self.report.null_flags[col] = null_pct
                log.warning("Column '%s' has %.1f%% null values", col, null_pct * 100)

        for col in self.config.required_columns:
            if col not in df.columns:
                self.report.schema_errors.append(f"Missing required column: '{col}'")

    def run(self) -> QualityReport:
        self.report.started_at = datetime.now().isoformat(timespec="seconds")
        log.info("Pipeline started")

        total_in, total_out = 0, 0

        for chunk in self.extractor.extract(self.config.chunk_size):
            total_in += len(chunk)
            self._quality_check(chunk)

            cleaned = self._with_retry(self.transformer.transform, chunk)
            total_out += len(cleaned)

            self._with_retry(self.loader.load, cleaned, self.table_name)

        self.report.total_rows = total_in
        self.report.rows_after_clean = total_out
        self.report.finished_at = datetime.now().isoformat(timespec="seconds")

        log.info("Pipeline complete: %d → %d rows", total_in, total_out)
        print(self.report.summary())
        return self.report


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main() -> None:
    config = PipelineConfig(
        chunk_size=10_000,
        max_retries=3,
        retry_delay=0.5,
        null_threshold=0.10,
        output_db="pipeline_output.db",
        required_columns=["transaction_id", "amount"],
    )

    extractor = SyntheticExtractor(n_rows=50_000)
    transformer = DataTransformer(config)
    loader = SQLiteLoader(config.output_db)

    pipeline = Pipeline(
        extractor=extractor,
        transformer=transformer,
        loader=loader,
        config=config,
        table_name="transactions",
    )
    pipeline.run()


if __name__ == "__main__":
    main()
