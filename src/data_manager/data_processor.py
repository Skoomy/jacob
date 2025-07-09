"""
Data processing and cleaning utilities for Jacob analysis platform.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes and cleans collected data for analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.industry_config = config.get("industry", {})
        self.processing_summary = []

    def process_all_data(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Process all collected data."""
        logger.info("Processing all collected data")

        processed_data = {}

        # Process Google Trends data
        if "google_trends" in data:
            processed_data["google_trends"] = self._process_google_trends(
                data["google_trends"]
            )

        # # Process market data
        # if "market_data" in data:
        #     processed_data["market_data"] = self._process_market_data(data["market_data"])

        # Process UK vehicle data
        if "uk_vehicle" in data:
            processed_data["uk_vehicle"] = self._process_uk_vehicle_data(
                data["uk_vehicle"]
            )

        logger.info(f"Processed {len(processed_data)} datasets")
        return processed_data

    def _process_google_trends(self, trends_data: pd.DataFrame) -> pd.DataFrame:
        """Process Google Trends data."""
        logger.info("Processing Google Trends data")

        df = trends_data.copy()
        # remove first two rows

        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
        df.set_index("week", inplace=True)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Remove any non-numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_columns]

        # Handle missing values
        df = df.fillna(0)

        # Normalize to 0-1 range if needed
        if df.max().max() > 1:
            df = df / 100.0

        # Apply smoothing if configured
        if self.config.get("data_processing", {}).get("smooth_trends", False):
            df = df.rolling(window=4, center=True).mean().fillna(df)

        self.processing_summary.append(
            {
                "dataset": "google_trends",
                "original_shape": trends_data.shape,
                "processed_shape": df.shape,
                "processing_steps": [
                    "datetime_index",
                    "numeric_only",
                    "fill_na",
                    "normalize",
                ],
            }
        )

        logger.info(f"Processed Google Trends data: {df.shape}")
        return df

    def _process_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Process market share data."""
        logger.info("Processing market data")

        df = market_data.copy()

        # Ensure datetime column
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        # Calculate market share if we have total sales
        if "total_sales" in df.columns and "market_category" in df.columns:
            # Group by time period and calculate market share
            df["market_share"] = df.groupby(df.index)["total_sales"].transform(
                lambda x: x / x.sum()
            )

        # Handle missing values
        df = df.fillna(method="ffill").fillna(method="bfill")

        self.processing_summary.append(
            {
                "dataset": "market_data",
                "original_shape": market_data.shape,
                "processed_shape": df.shape,
                "processing_steps": ["datetime_index", "market_share_calc", "fill_na"],
            }
        )

        logger.info(f"Processed market data: {df.shape}")
        return df

    def _process_uk_vehicle_data(self, vehicle_data: pd.DataFrame) -> pd.DataFrame:
        """Process UK vehicle registration data."""
        logger.info("Processing UK vehicle data")

        df = vehicle_data.copy()

        # Clean column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        # Extract relevant columns for automotive analysis
        relevant_columns = []
        for col in df.columns:
            if any(
                keyword in col.lower()
                for keyword in [
                    "electric",
                    "hybrid",
                    "petrol",
                    "diesel",
                    "registration",
                ]
            ):
                relevant_columns.append(col)

        if relevant_columns:
            df = df[relevant_columns]

        # Convert to numeric where possible
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Handle missing values
        df = df.fillna(0)

        self.processing_summary.append(
            {
                "dataset": "uk_vehicle",
                "original_shape": vehicle_data.shape,
                "processed_shape": df.shape,
                "processing_steps": [
                    "clean_columns",
                    "select_relevant",
                    "convert_numeric",
                    "fill_na",
                ],
            }
        )

        logger.info(f"Processed UK vehicle data: {df.shape}")
        return df

    def align_time_series(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Align time series data to common time periods."""
        logger.info("Aligning time series data")

        # Find common time range
        date_ranges = []
        for dataset_name, df in data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                date_ranges.append((df.index.min(), df.index.max()))

        if not date_ranges:
            logger.warning("No datetime indexed datasets found")
            return data

        # Find overlap
        common_start = max(start for start, end in date_ranges)
        common_end = min(end for start, end in date_ranges)

        logger.info(f"Common time range: {common_start} to {common_end}")

        # Align datasets
        aligned_data = {}
        for dataset_name, df in data.items():
            if isinstance(df.index, pd.DatetimeIndex):
                aligned_data[dataset_name] = df.loc[common_start:common_end]
            else:
                aligned_data[dataset_name] = df

        return aligned_data

    def calculate_correlation_matrix(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate correlation matrix between datasets."""
        logger.info("Calculating correlation matrix")

        # Combine all numeric data
        combined_data = pd.DataFrame()

        for dataset_name, df in data.items():
            if isinstance(df, pd.DataFrame):
                numeric_df = df.select_dtypes(include=[np.number])
                if not numeric_df.empty:
                    # Add dataset prefix to column names
                    numeric_df.columns = [
                        f"{dataset_name}_{col}" for col in numeric_df.columns
                    ]
                    combined_data = pd.concat([combined_data, numeric_df], axis=1)

        if combined_data.empty:
            logger.warning("No numeric data found for correlation calculation")
            return pd.DataFrame()

        correlation_matrix = combined_data.corr()

        logger.info(f"Calculated correlation matrix: {correlation_matrix.shape}")
        return correlation_matrix

    def get_processing_summary(self) -> List[Dict[str, Any]]:
        """Get summary of processing steps performed."""
        return self.processing_summary

    def create_analysis_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a combined dataset for analysis."""
        logger.info("Creating analysis dataset")

        # Start with Google Trends data as base
        if "google_trends" in data:
            base_df = data["google_trends"].copy()
        else:
            logger.error("Google Trends data required for analysis")
            return pd.DataFrame()

        # Add market data if available
        if "market_data" in data:
            market_df = data["market_data"]
            if isinstance(market_df.index, pd.DatetimeIndex):
                # Resample to match base frequency
                market_resampled = market_df.resample("M").mean()
                base_df = pd.concat([base_df, market_resampled], axis=1)

        # Add other datasets
        for dataset_name, df in data.items():
            if dataset_name not in ["google_trends", "market_data"]:
                if isinstance(df.index, pd.DatetimeIndex):
                    df_resampled = df.resample("M").mean()
                    base_df = pd.concat([base_df, df_resampled], axis=1)

        # Clean the final dataset
        analysis_df = base_df.dropna(how="all")

        logger.info(f"Created analysis dataset: {analysis_df.shape}")
        return analysis_df
