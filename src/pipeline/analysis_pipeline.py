"""
Main analysis pipeline for Jacob platform.
Orchestrates data collection, modeling, visualization, and reporting.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
from datetime import datetime
import click
from ..data_manager.data_collector import DataCollector
from ..data_manager.load_data import load_data
from ..modeling.bayesian_model import BayesianMarketShareModel
from ..visualization.plotly_visualizer import PlotlyVisualizer
from ..reporting.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class AnalysisPipeline:
    """Main pipeline orchestrating the complete analysis workflow."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.industry = config.get("industry", "unknown")
        self.output_dir = Path(config.get("output_dir", "output"))
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.data_collector = DataCollector(config)
        self.model = BayesianMarketShareModel(config)
        self.visualizer = PlotlyVisualizer(config)
        self.reporter = ReportGenerator(config)

        # Pipeline state
        self.pipeline_state = {
            "start_time": None,
            "end_time": None,
            "steps_completed": [],
            "current_step": None,
            "errors": [],
            "data_summary": {},
            "model_results": {},
            "visualizations": {},
            "reports": {},
        }

    def run(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline."""
        logger.info(f"Starting analysis pipeline for {self.industry} industry")

        self.pipeline_state["start_time"] = datetime.now()

        try:
            # Step 1: Data Collection
            self._run_step("data_collection", self._collect_data)

            # Step 2: Data Preprocessing
            self._run_step("data_preprocessing", self._preprocess_data)
            import sys

            click.secho("END DEBUG", fg="red")
            sys.exit(0)

            # Step 3: Bayesian Modeling
            self._run_step("bayesian_modeling", self._run_bayesian_model)

            # Step 4: Model Diagnostics
            self._run_step("model_diagnostics", self._run_model_diagnostics)

            # Step 5: Visualization
            self._run_step("visualization", self._create_visualizations)

            # Step 6: Reporting
            self._run_step("reporting", self._generate_reports)

            self.pipeline_state["end_time"] = datetime.now()

            logger.info("Analysis pipeline completed successfully")
            return self.pipeline_state

        except Exception as e:
            self.pipeline_state["end_time"] = datetime.now()
            self.pipeline_state["errors"].append(f"Pipeline failed: {str(e)}")
            logger.error(f"Pipeline failed: {e}")
            raise

    def _run_step(self, step_name: str, step_function):
        """Run a pipeline step with error handling and logging."""
        logger.info(f"Running pipeline step: {step_name}")
        self.pipeline_state["current_step"] = step_name

        try:
            step_function()
            self.pipeline_state["steps_completed"].append(step_name)
            logger.info(f"Completed pipeline step: {step_name}")

        except Exception as e:
            error_msg = f"Step {step_name} failed: {str(e)}"
            self.pipeline_state["errors"].append(error_msg)
            logger.error(error_msg)
            raise

    def _collect_data(self):
        """Step 1: Collect data from various sources."""
        logger.info("Collecting data from all sources")

        # Check if data already exists
        existing_data = load_data(self.config)

        # Collect fresh data
        collected_data = self.data_collector.collect_all_data()

        # Combine existing and new data
        self.pipeline_state["data_summary"] = {
            "existing_datasets": list(existing_data.keys()),
            "collected_datasets": list(collected_data.keys()),
            "collection_summary": self.data_collector.get_data_summary(),
        }

        # Store data for next steps
        self.collected_data = {**existing_data, **collected_data}

        logger.info(
            f"Data collection completed. Total datasets: {len(self.collected_data)}"
        )

    def _preprocess_data(self):
        """Step 2: Preprocess and clean data."""
        logger.info("Preprocessing collected data")

        from ..data_manager.data_processor import DataProcessor

        processor = DataProcessor(self.config)

        self.processed_data = processor.process_all_data(self.collected_data)

        # Store preprocessing summary
        self.pipeline_state["data_summary"]["preprocessing"] = {
            "processed_datasets": list(self.processed_data.keys()),
            "preprocessing_steps": processor.get_processing_summary(),
        }

        logger.info("Data preprocessing completed")

    def _run_bayesian_model(self):
        """Step 3: Run Bayesian modeling."""
        logger.info("Running Bayesian market share model")

        # Prepare model data
        model_data = self._prepare_model_data()

        # Fit the model
        self.model_results = self.model.fit(model_data)

        # Store model results
        self.pipeline_state["model_results"] = {
            "model_type": "bayesian_market_share",
            "convergence_diagnostics": self.model.get_convergence_diagnostics(),
            "parameter_estimates": self.model.get_parameter_estimates(),
            "model_summary": self.model.get_model_summary(),
        }

        logger.info("Bayesian modeling completed")

    def _run_model_diagnostics(self):
        """Step 4: Run model diagnostics."""
        logger.info("Running model diagnostics")

        diagnostics = self.model.run_diagnostics()

        self.pipeline_state["model_results"]["diagnostics"] = diagnostics

        logger.info("Model diagnostics completed")

    def _create_visualizations(self):
        """Step 5: Create visualizations."""
        logger.info("Creating visualizations")

        # Create various plots
        visualizations = {}

        # Time series plots
        if "google_trends" in self.processed_data:
            visualizations["trends_timeseries"] = (
                self.visualizer.create_trends_timeseries(
                    self.processed_data["google_trends"]
                )
            )

        # Market share plots
        if "market_data" in self.processed_data:
            visualizations["market_share_plot"] = (
                self.visualizer.create_market_share_plot(
                    self.processed_data["market_data"]
                )
            )

        # Model results plots
        if hasattr(self, "model_results"):
            visualizations["model_results"] = (
                self.visualizer.create_model_results_plots(self.model_results)
            )

        # Correlation analysis
        visualizations["correlation_analysis"] = (
            self.visualizer.create_correlation_plot(self.processed_data)
        )

        self.pipeline_state["visualizations"] = visualizations

        logger.info(f"Created {len(visualizations)} visualizations")

    def _generate_reports(self):
        """Step 6: Generate reports."""
        logger.info("Generating reports")

        # Generate comprehensive report
        report_data = {
            "pipeline_state": self.pipeline_state,
            "data_summary": self.pipeline_state["data_summary"],
            "model_results": self.pipeline_state.get("model_results", {}),
            "visualizations": self.pipeline_state.get("visualizations", {}),
            "config": self.config,
        }

        reports = self.reporter.generate_all_reports(report_data)

        self.pipeline_state["reports"] = reports

        logger.info("Report generation completed")

    def _prepare_model_data(self) -> Dict[str, Any]:
        """Prepare data for Bayesian modeling."""
        logger.info("Preparing data for Bayesian modeling")

        model_data = {}

        # Prepare share of voice data (from Google Trends)
        if "google_trends" in self.processed_data:
            trends_data = self.processed_data["google_trends"]
            # Normalize to get share of voice
            model_data["share_of_voice"] = trends_data.div(
                trends_data.sum(axis=1), axis=0
            ).fillna(0)

        # Prepare market share data
        if "market_data" in self.processed_data:
            market_data = self.processed_data["market_data"]
            # Process market data to get market share
            model_data["market_share"] = market_data

        # Add time index
        if "google_trends" in self.processed_data:
            model_data["time_index"] = self.processed_data["google_trends"].index

        return model_data

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution."""
        return {
            "industry": self.industry,
            "start_time": self.pipeline_state["start_time"],
            "end_time": self.pipeline_state["end_time"],
            "duration": (
                self.pipeline_state["end_time"] - self.pipeline_state["start_time"]
                if self.pipeline_state["end_time"] and self.pipeline_state["start_time"]
                else None
            ),
            "steps_completed": self.pipeline_state["steps_completed"],
            "errors": self.pipeline_state["errors"],
            "success": len(self.pipeline_state["errors"]) == 0,
        }
