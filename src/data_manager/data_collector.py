"""
Data collection system for Jacob analysis platform.
Collects Google Trends and UK Government statistics data.
"""

import logging
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class GoogleTrendsCollector:
    """Collects Google Trends data for share of voice analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trends_config = config.get("data_sources", {}).get("google_trends", {})
        self.pytrends = TrendReq(hl="en-GB", tz=0, retries=3, backoff_factor=0.5)

    def collect_trends_data(
        self, keywords: List[str], timeframe: str = "today 5-y"
    ) -> pd.DataFrame:
        """Collect Google Trends data for specified keywords."""
        logger.info(f"Collecting Google Trends data for keywords: {keywords}")

        try:
            # Build payload
            self.pytrends.build_payload(
                keywords,
                cat=self.trends_config.get("category", 0),
                timeframe=timeframe,
                geo=self.trends_config.get("geo", "GB"),
                gprop="",
            )

            # Get interest over time
            interest_over_time = self.pytrends.interest_over_time()

            if not interest_over_time.empty:
                # Remove 'isPartial' column if present
                if "isPartial" in interest_over_time.columns:
                    interest_over_time = interest_over_time.drop("isPartial", axis=1)

                logger.info(
                    f"Successfully collected trends data for {len(keywords)} keywords"
                )
                return interest_over_time
            else:
                logger.warning("No trends data returned")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting Google Trends data: {e}")
            raise

    def collect_related_queries(self, keyword: str) -> Dict[str, pd.DataFrame]:
        """Collect related queries for a keyword."""
        logger.info(f"Collecting related queries for: {keyword}")

        try:
            self.pytrends.build_payload(
                [keyword],
                cat=self.trends_config.get("category", 0),
                timeframe="today 5-y",
                geo=self.trends_config.get("geo", "GB"),
                gprop="",
            )

            related_queries = self.pytrends.related_queries()

            # Add sleep to avoid rate limiting
            sleep_time = self.trends_config.get("sleep_between_requests", 1)
            time.sleep(sleep_time)

            return related_queries

        except Exception as e:
            logger.error(f"Error collecting related queries for {keyword}: {e}")
            return {}


class UKGovDataCollector:
    """Collects UK Government statistics data for market share analysis."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gov_config = config.get("data_sources", {}).get("uk_gov_stats", {})
        self.base_url = self.gov_config.get(
            "base_url", "https://www.gov.uk/government/statistics"
        )
        self.session = requests.Session()

    def collect_vehicle_registration_data(self) -> pd.DataFrame:
        """Collect UK vehicle registration data."""
        logger.info("Collecting UK vehicle registration data")

        try:
            # For now, use existing data file
            data_file = Path("data/df_VEH0120_GB.csv")
            if data_file.exists():
                df = pd.read_csv(data_file)
                logger.info(f"Loaded existing vehicle registration data: {df.shape}")
                return df
            else:
                logger.warning("Vehicle registration data file not found")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error collecting vehicle registration data: {e}")
            raise

    def collect_pharmaceutical_data(self) -> pd.DataFrame:
        """Collect UK pharmaceutical market data."""
        logger.info("Collecting UK pharmaceutical market data")

        # This would be implemented with actual API calls
        # For now, return placeholder
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", "2024-12-31", freq="M"),
                "total_sales": [100 + i for i in range(60)],
                "market_category": ["pharma"] * 60,
            }
        )

    def collect_alcohol_sales_data(self) -> pd.DataFrame:
        """Collect UK alcohol sales data."""
        logger.info("Collecting UK alcohol sales data")

        # This would be implemented with actual API calls
        # For now, return placeholder
        return pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", "2024-12-31", freq="M"),
                "total_sales": [80 + i for i in range(60)],
                "market_category": ["alcohol"] * 60,
            }
        )


class DataCollector:
    """Main data collector orchestrating all data collection."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.google_collector = GoogleTrendsCollector(config)
        self.gov_collector = UKGovDataCollector(config)
        self.output_dir = Path(config.get("output_dir", "data"))
        self.output_dir.mkdir(exist_ok=True)

    def collect_all_data(self) -> Dict[str, pd.DataFrame]:
        """Collect all data for the configured industry."""
        logger.info("Starting comprehensive data collection")

        industry_config = self.config.get("industry", {})
        industry_name = industry_config.get("name", "unknown")

        collected_data = {}

        # # Collect Google Trends data
        # try:
        #     search_terms = industry_config.get("search_terms", [])
        #     if search_terms:
        #         trends_data = self.google_collector.collect_trends_data(search_terms)
        #         if not trends_data.empty:
        #             collected_data["google_trends"] = trends_data
        #             self._save_data(trends_data, f"{industry_name}_google_trends.csv")
        # except Exception as e:
        #     logger.error(f"Failed to collect Google Trends data: {e}")

        # Collect market data based on industry
        try:
            market_data_source = industry_config.get("market_data_source", "")

            if market_data_source == "uk_gov_vehicle_registrations":
                market_data = self.gov_collector.collect_vehicle_registration_data()
            elif market_data_source == "uk_gov_pharmaceutical_sales":
                market_data = self.gov_collector.collect_pharmaceutical_data()
            elif market_data_source == "uk_gov_alcohol_sales":
                market_data = self.gov_collector.collect_alcohol_sales_data()
            else:
                logger.warning(f"Unknown market data source: {market_data_source}")
                market_data = pd.DataFrame()

            if not market_data.empty:
                collected_data["market_data"] = market_data
                self._save_data(market_data, f"{industry_name}_market_data.xlsx")

        except Exception as e:
            logger.error(f"Failed to collect market data: {e}")

        # # Collect related queries for each search term
        # try:
        #     search_terms = industry_config.get("search_terms", [])
        #     for term in search_terms[:3]:  # Limit to first 3 to avoid rate limiting
        #         related_queries = self.google_collector.collect_related_queries(term)
        #         if related_queries:
        #             collected_data[f"related_queries_{term}"] = related_queries
        #             self._save_related_queries(
        #                 related_queries, f"{industry_name}_related_queries_{term}.json"
        #             )
        # except Exception as e:
        #     logger.error(f"Failed to collect related queries: {e}")
        # Collect Google Trends data source
        google_trend_data_source = industry_config.get("google_trend_data_source", "")

        if google_trend_data_source:
            logger.info(
                f"Using pre-collected Google Trends data from {google_trend_data_source}"
            )
            try:
                collected_data["google_trends"] = pd.read_csv(google_trend_data_source)
                self._save_data(
                    collected_data["google_trends"],
                    f"{industry_name}_google_trends.xlsx",
                )

            except FileNotFoundError:
                logger.warning(
                    f"Google Trends data file not found: {google_trend_data_source}"
                )
            except Exception as e:
                logger.error(f"Error loading Google Trends data: {e}")
        else:
            logger.warning("No Google Trends data source configured")

        logger.info(
            f"Data collection completed. Collected {len(collected_data)} datasets"
        )

        return collected_data

    def _save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to Excel file."""
        try:
            filepath = self.output_dir / filename
            data.to_excel(filepath, index=True)
            logger.info(f"Saved data to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save data to {filename}: {e}")

    def _save_related_queries(self, data: Dict, filename: str) -> None:
        """Save related queries to JSON file."""
        try:
            filepath = self.output_dir / filename
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved related queries to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save related queries to {filename}: {e}")

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data."""
        summary = {
            "collection_timestamp": datetime.now().isoformat(),
            "industry": self.config.get("industry", {}).get("name", "unknown"),
            "data_sources": list(self.config.get("data_sources", {}).keys()),
            "output_directory": str(self.output_dir),
            "files_created": [],
        }

        # List files in output directory
        if self.output_dir.exists():
            summary["files_created"] = [f.name for f in self.output_dir.glob("*")]

        return summary
