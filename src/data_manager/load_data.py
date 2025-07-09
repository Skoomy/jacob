"""
Data loading utilities for Jacob analysis platform.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


def load_uk_vehicle_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load UK vehicle registration data."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded UK vehicle data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load UK vehicle data: {e}")
        raise


def load_google_trends_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """Load Google Trends data."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded Google Trends data from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load Google Trends data: {e}")
        raise


def load_data(config: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """Main data loading function."""
    if config is None:
        config = {}
    
    data = {}
    
    # Load existing data files
    data_dir = Path("data")
    
    # Load UK vehicle data if available
    uk_vehicle_file = data_dir / "df_VEH0120_GB.csv"
    if uk_vehicle_file.exists():
        data["uk_vehicle"] = load_uk_vehicle_data(uk_vehicle_file)
    
    logger.info(f"Loaded {len(data)} datasets")
    return data