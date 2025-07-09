"""
Configuration management system for Jacob analysis platform.
Enables config-driven analysis across different industries.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""

    google_trends: Dict[str, Any]
    uk_gov_stats: Dict[str, Any]
    update_frequency: str = "daily"


@dataclass
class ModelConfig:
    """Configuration for Bayesian modeling."""

    prior_distributions: Dict[str, Any]
    mcmc_samples: int = 2000
    tune_samples: int = 1000
    chains: int = 4
    target_accept: float = 0.95


@dataclass
class VisualizationConfig:
    """Configuration for Plotly visualizations."""

    theme: str = "plotly_white"
    width: int = 1200
    height: int = 800
    export_formats: list = None

    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ["html", "png", "svg"]


@dataclass
class ReportConfig:
    """Configuration for reporting system."""

    output_format: str = "txt"
    include_sections: list = None
    auto_save: bool = True

    def __post_init__(self):
        if self.include_sections is None:
            self.include_sections = [
                "executive_summary",
                "methodology",
                "results",
                "model_diagnostics",
                "conclusions",
            ]


@dataclass
class IndustryConfig:
    """Configuration for industry-specific analysis."""

    name: str
    brands: list
    search_terms: list
    market_data_source: str
    google_trend_data_source: str = (None,)
    analysis_period: Dict[str, str]
    seasonality_adjustments: bool = True


class ConfigManager:
    """Manages configuration loading and validation."""

    INDUSTRY_TEMPLATES = {
        "automotive": {
            "name": "automotive",
            "brands": ["Tesla", "BMW", "Mercedes", "Audi", "Jaguar", "Volvo"],
            "search_terms": ["electric car", "EV", "tesla model", "bmw electric"],
            "market_data_source": "uk_gov_vehicle_registrations",
            "google_trend_data_source": "data/google_trend_automotive.csv",
            "analysis_period": {"start": "2020-01-01", "end": "2024-12-31"},
        },
        "pharma": {
            "name": "pharma",
            "brands": ["Pfizer", "GSK", "AstraZeneca", "Novartis", "Roche"],
            "search_terms": ["covid vaccine", "flu shot", "medication"],
            "market_data_source": "uk_gov_pharmaceutical_sales",
            "analysis_period": {"start": "2020-01-01", "end": "2024-12-31"},
        },
        "alcohol": {
            "name": "alcohol",
            "brands": ["Guinness", "Stella Artois", "Budweiser", "Heineken"],
            "search_terms": ["beer", "lager", "guinness", "stella artois"],
            "market_data_source": "uk_gov_alcohol_sales",
            "analysis_period": {"start": "2020-01-01", "end": "2024-12-31"},
        },
    }

    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            logger.info(f"Configuration loaded from {config_path}")
            return cls._validate_config(config)

        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config file: {e}")
            raise

    @classmethod
    def _validate_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration structure."""
        required_sections = ["data_sources", "modeling", "visualization", "reporting"]

        for section in required_sections:
            if section not in config:
                logger.warning(f"Missing config section: {section}, using defaults")
                config[section] = cls._get_default_section(section)

        return config

    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "data_sources": asdict(
                DataSourceConfig(
                    google_trends={
                        "geo": "GB",
                        "timeframe": "today 5-y",
                        "category": 0,
                    },
                    uk_gov_stats={
                        "base_url": "https://www.gov.uk/government/statistics",
                        "cache_duration": 3600,
                    },
                )
            ),
            "modeling": asdict(ModelConfig()),
            "visualization": asdict(VisualizationConfig()),
            "reporting": asdict(ReportConfig()),
            "output_dir": "output",
            "log_level": "INFO",
        }

    @classmethod
    def _get_default_section(cls, section: str) -> Dict[str, Any]:
        """Get default configuration for a specific section."""
        defaults = cls._get_default_config()
        return defaults.get(section, {})

    @classmethod
    def create_config_template(cls, industry: str, output_path: str) -> None:
        """Create a configuration template for specified industry."""
        if industry not in cls.INDUSTRY_TEMPLATES:
            raise ValueError(
                f"Industry '{industry}' not supported. Choose from: {list(cls.INDUSTRY_TEMPLATES.keys())}"
            )

        config = cls._get_default_config()
        config["industry"] = cls.INDUSTRY_TEMPLATES[industry]
        config["created_at"] = datetime.now().isoformat()

        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        logger.info(
            f"Configuration template created for {industry} industry at {output_path}"
        )

    @classmethod
    def get_industry_config(cls, industry: str) -> IndustryConfig:
        """Get industry-specific configuration."""
        if industry not in cls.INDUSTRY_TEMPLATES:
            raise ValueError(f"Industry '{industry}' not supported")

        template = cls.INDUSTRY_TEMPLATES[industry]
        return IndustryConfig(**template)
