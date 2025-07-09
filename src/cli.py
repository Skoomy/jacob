"""
Jacob: Share of Voice vs Market Share Analysis Platform
CLI interface for the application
"""

import click
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .utils.logger import get_logger, setup_logging
from .config.config_manager import ConfigManager
from .pipeline.analysis_pipeline import AnalysisPipeline
from .reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", type=click.Path(), help="Log file path")
@click.option("--config", type=click.Path(exists=True), help="Configuration file")
@click.pass_context
def cli(ctx, log_level, log_file, config):
    """Jacob: Share of Voice vs Market Share Analysis Platform.

    Analysis platform for testing Les Binet's hypothesis
    that share of voice can predict market share across different industries.
    """
    ctx.ensure_object(dict)

    # Set up logging
    setup_logging(
        level=log_level, log_file=Path(log_file) if log_file else None, json_logs=False
    )

    # Load configuration
    config_path = config or "config/default.yaml"
    ctx.obj["config"] = ConfigManager.load_config(config_path)


@cli.command()
@click.option(
    "--industry",
    type=str,
    required=True,
    help="Industry to analyze (automotive, pharma, alcohol)",
)
@click.option(
    "--output-dir", type=click.Path(), default="output", help="Output directory"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Configuration file (overrides global config)",
)
@click.pass_context
def analyze(ctx, industry: str, output_dir: str, config: Optional[str]):
    """Run the complete Share of Voice vs Market Share analysis pipeline."""
    logger.info(f"Starting analysis for {industry} industry")

    # Use command-level config if provided, otherwise use global config
    if config:
        analysis_config = ConfigManager.load_config(config)
    else:
        analysis_config = ctx.obj["config"].copy()

    # Get industry-specific configuration
    try:
        industry_config = ConfigManager.get_industry_config(industry)
        analysis_config["industry"] = {
            "name": industry_config.name,
            "brands": industry_config.brands,
            "search_terms": industry_config.search_terms,
            "market_data_source": industry_config.market_data_source,
            "google_trend_data_source": industry_config.google_trend_data_source,
            "analysis_period": industry_config.analysis_period,
            "seasonality_adjustments": getattr(
                industry_config, "seasonality_adjustments", True
            ),
        }
    except ValueError:
        logger.warning(
            f"No predefined configuration for {industry} industry, using basic config"
        )
        analysis_config["industry"] = {"name": industry}

    analysis_config["output_dir"] = Path(output_dir)

    # Initialize pipeline
    pipeline = AnalysisPipeline(analysis_config)

    # Run analysis
    results = pipeline.run()

    # Generate reports
    report_generator = ReportGenerator(analysis_config)
    report_generator.generate_final_report(results)

    logger.info("Analysis completed successfully")
    return results


@cli.command()
@click.option(
    "--config-template", type=str, default="automotive", help="Template to use"
)
@click.option(
    "--output",
    type=click.Path(),
    default="config/custom.yaml",
    help="Output config file",
)
def init_config(config_template: str, output: str):
    """Initialize a new configuration file for analysis."""
    logger.info(f"Creating config template for {config_template}")
    ConfigManager.create_config_template(config_template, output)
    logger.info(f"Configuration template created at {output}")


@cli.command()
@click.option(
    "--industry", type=str, required=True, help="Industry to collect data for"
)
@click.pass_context
def collect_data(ctx, industry: str):
    """Collect data for the specified industry."""
    logger.info(f"Collecting data for {industry} industry")

    config = ctx.obj["config"]
    config["industry"] = industry

    from .data_manager.data_collector import DataCollector

    collector = DataCollector(config)
    collector.collect_all_data()

    logger.info("Data collection completed")


if __name__ == "__main__":
    cli()
