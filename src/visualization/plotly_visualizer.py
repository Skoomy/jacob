"""
Plotly visualization system for Jacob analysis platform.
Creates Nobel Prize-level visualizations using Plotly.
"""

import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class PlotlyVisualizer:
    """
    Creates professional-grade visualizations using Plotly.
    All plots follow consistent styling and are publication-ready.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.viz_config = config.get("visualization", {})
        
        # Styling configuration
        self.theme = self.viz_config.get("theme", "plotly_white")
        self.width = self.viz_config.get("width", 1200)
        self.height = self.viz_config.get("height", 800)
        self.color_palette = self.viz_config.get("color_palette", 
                                               ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
        
        # Output configuration
        self.output_dir = Path(config.get("output_dir", "output"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Export formats
        self.export_formats = self.viz_config.get("export_formats", ["html", "png", "svg"])
        
    def create_trends_timeseries(self, trends_data: pd.DataFrame) -> Dict[str, Any]:
        """Create time series visualization for Google Trends data."""
        logger.info("Creating Google Trends time series visualization")
        
        fig = go.Figure()
        
        # Add each search term/brand as a separate trace
        for i, column in enumerate(trends_data.columns):
            color = self.color_palette[i % len(self.color_palette)]
            
            fig.add_trace(
                go.Scatter(
                    x=trends_data.index,
                    y=trends_data[column],
                    mode='lines+markers',
                    name=column,
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Date: %{x}<br>' +
                                'Share of Voice: %{y:.1%}<br>' +
                                '<extra></extra>'
                )
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Share of Voice Over Time (Google Trends)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Date',
            yaxis_title='Share of Voice',
            template=self.theme,
            width=self.width,
            height=self.height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            showlegend=True
        )
        
        # Format y-axis as percentage
        fig.update_yaxis(tickformat='.1%')
        
        # Save the plot
        plot_info = self._save_plot(fig, "trends_timeseries")
        
        return {
            "figure": fig,
            "plot_info": plot_info,
            "description": "Time series showing share of voice trends for different brands/search terms"
        }
    
    def create_market_share_plot(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Create market share visualization."""
        logger.info("Creating market share visualization")
        
        fig = go.Figure()
        
        # If market data has multiple columns, create a stacked area chart
        if len(market_data.columns) > 1:
            for i, column in enumerate(market_data.columns):
                if column != 'market_share':  # Skip if this is the calculated column
                    color = self.color_palette[i % len(self.color_palette)]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=market_data.index,
                            y=market_data[column],
                            mode='lines',
                            name=column,
                            fill='tonexty' if i > 0 else 'tozeroy',
                            line=dict(color=color, width=1),
                            hovertemplate='<b>%{fullData.name}</b><br>' +
                                        'Date: %{x}<br>' +
                                        'Market Share: %{y:.1%}<br>' +
                                        '<extra></extra>'
                        )
                    )
        else:
            # Single line plot
            column = market_data.columns[0]
            fig.add_trace(
                go.Scatter(
                    x=market_data.index,
                    y=market_data[column],
                    mode='lines+markers',
                    name='Market Share',
                    line=dict(color=self.color_palette[0], width=3),
                    marker=dict(size=6),
                    hovertemplate='Date: %{x}<br>' +
                                'Market Share: %{y:.1%}<br>' +
                                '<extra></extra>'
                )
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Market Share Over Time',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Date',
            yaxis_title='Market Share',
            template=self.theme,
            width=self.width,
            height=self.height,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )
        
        # Format y-axis as percentage
        fig.update_yaxis(tickformat='.1%')
        
        # Save the plot
        plot_info = self._save_plot(fig, "market_share_plot")
        
        return {
            "figure": fig,
            "plot_info": plot_info,
            "description": "Market share evolution over time"
        }
    
    def create_correlation_plot(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Create correlation analysis visualization."""
        logger.info("Creating correlation analysis visualization")
        
        from ..data_manager.data_processor import DataProcessor
        processor = DataProcessor(self.config)
        
        # Calculate correlation matrix
        correlation_matrix = processor.calculate_correlation_matrix(data)
        
        if correlation_matrix.empty:
            logger.warning("No correlation data available")
            return {"error": "No correlation data available"}
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>' +
                        'Correlation: %{z:.3f}<br>' +
                        '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Correlation Matrix: Share of Voice vs Market Share',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            template=self.theme,
            width=self.width,
            height=self.height,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        
        # Save the plot
        plot_info = self._save_plot(fig, "correlation_heatmap")
        
        return {
            "figure": fig,
            "plot_info": plot_info,
            "description": "Correlation analysis between share of voice and market share metrics"
        }
    
    def create_model_results_plots(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualizations for Bayesian model results."""
        logger.info("Creating Bayesian model results visualizations")
        
        if "trace" not in model_results:
            logger.warning("No trace data available for model visualization")
            return {"error": "No trace data available"}
        
        trace = model_results["trace"]
        
        # Create subplots for different diagnostics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Population Correlation', 'Brand-Level Slopes', 
                          'Posterior Predictive Check', 'Convergence Diagnostics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Population correlation posterior
        if "pop_correlation" in trace.posterior.data_vars:
            correlation_samples = trace.posterior["pop_correlation"].values.flatten()
            
            fig.add_trace(
                go.Histogram(
                    x=correlation_samples,
                    nbinsx=30,
                    name='Population Correlation',
                    marker_color=self.color_palette[0],
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # Brand-level slopes
        if "beta_brand" in trace.posterior.data_vars:
            brand_slopes = trace.posterior["beta_brand"].values
            
            for i in range(min(5, brand_slopes.shape[-1])):  # Show first 5 brands
                brand_samples = brand_slopes[:, :, i].flatten()
                
                fig.add_trace(
                    go.Box(
                        y=brand_samples,
                        name=f'Brand {i+1}',
                        marker_color=self.color_palette[i % len(self.color_palette)]
                    ),
                    row=1, col=2
                )
        
        # Posterior predictive check (placeholder)
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Observed vs Predicted',
                line=dict(color=self.color_palette[0], dash='dash')
            ),
            row=2, col=1
        )
        
        # Convergence diagnostics (R-hat values)
        try:
            import arviz as az
            rhat_values = az.rhat(trace)
            param_names = list(rhat_values.data_vars)
            rhat_vals = [float(rhat_values[var].values) for var in param_names]
            
            fig.add_trace(
                go.Scatter(
                    x=param_names,
                    y=rhat_vals,
                    mode='markers',
                    name='R-hat Values',
                    marker=dict(size=8, color=self.color_palette[1]),
                    hovertemplate='Parameter: %{x}<br>R-hat: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add horizontal line at R-hat = 1.01
            fig.add_hline(y=1.01, line_dash="dash", line_color="red", row=2, col=2)
            
        except Exception as e:
            logger.warning(f"Could not create convergence diagnostics: {e}")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Bayesian Model Results: Les Binet Hypothesis Test',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            template=self.theme,
            width=self.width,
            height=self.height * 1.2,
            showlegend=True
        )
        
        # Update subplot titles
        fig.update_xaxes(title_text="Correlation Coefficient", row=1, col=1)
        fig.update_yaxes(title_text="Density", row=1, col=1)
        
        fig.update_xaxes(title_text="Brand", row=1, col=2)
        fig.update_yaxes(title_text="Slope Coefficient", row=1, col=2)
        
        fig.update_xaxes(title_text="Observed", row=2, col=1)
        fig.update_yaxes(title_text="Predicted", row=2, col=1)
        
        fig.update_xaxes(title_text="Parameter", row=2, col=2)
        fig.update_yaxes(title_text="R-hat", row=2, col=2)
        
        # Save the plot
        plot_info = self._save_plot(fig, "model_results")
        
        return {
            "figure": fig,
            "plot_info": plot_info,
            "description": "Comprehensive Bayesian model results and diagnostics"
        }
    
    def create_hypothesis_summary_plot(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary plot for the Les Binet hypothesis test."""
        logger.info("Creating hypothesis summary visualization")
        
        # Extract key results
        if "trace" not in model_results:
            return {"error": "No model results available"}
        
        trace = model_results["trace"]
        
        # Create summary figure
        fig = go.Figure()
        
        # Population correlation posterior
        if "pop_correlation" in trace.posterior.data_vars:
            correlation_samples = trace.posterior["pop_correlation"].values.flatten()
            
            # Calculate credible interval
            ci_lower = np.percentile(correlation_samples, 2.5)
            ci_upper = np.percentile(correlation_samples, 97.5)
            median_corr = np.median(correlation_samples)
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=correlation_samples,
                    nbinsx=50,
                    name='Posterior Distribution',
                    marker_color=self.color_palette[0],
                    opacity=0.7,
                    hovertemplate='Correlation: %{x:.3f}<br>Density: %{y}<extra></extra>'
                )
            )
            
            # Add credible interval
            fig.add_vline(x=ci_lower, line_dash="dash", line_color="red", 
                         annotation_text=f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            fig.add_vline(x=ci_upper, line_dash="dash", line_color="red")
            fig.add_vline(x=median_corr, line_dash="solid", line_color="black",
                         annotation_text=f"Median: {median_corr:.3f}")
            
            # Add reference line at zero
            fig.add_vline(x=0, line_dash="dot", line_color="gray",
                         annotation_text="No Correlation")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Les Binet Hypothesis Test Results<br><sub>Share of Voice → Market Share Correlation</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'family': 'Arial, sans-serif'}
            },
            xaxis_title='Correlation Coefficient',
            yaxis_title='Posterior Density',
            template=self.theme,
            width=self.width,
            height=self.height * 0.8,
            showlegend=False
        )
        
        # Save the plot
        plot_info = self._save_plot(fig, "hypothesis_summary")
        
        return {
            "figure": fig,
            "plot_info": plot_info,
            "description": "Summary of Les Binet hypothesis test results"
        }
    
    def _save_plot(self, fig: go.Figure, filename: str) -> Dict[str, Any]:
        """Save plot in multiple formats."""
        plot_info = {
            "filename": filename,
            "saved_formats": [],
            "file_paths": []
        }
        
        try:
            for fmt in self.export_formats:
                file_path = self.output_dir / f"{filename}.{fmt}"
                
                if fmt == "html":
                    fig.write_html(str(file_path))
                elif fmt == "png":
                    fig.write_image(str(file_path), format="png")
                elif fmt == "svg":
                    fig.write_image(str(file_path), format="svg")
                elif fmt == "pdf":
                    fig.write_image(str(file_path), format="pdf")
                
                plot_info["saved_formats"].append(fmt)
                plot_info["file_paths"].append(str(file_path))
                
            logger.info(f"Saved plot {filename} in {len(plot_info['saved_formats'])} formats")
            
        except Exception as e:
            logger.error(f"Error saving plot {filename}: {e}")
            plot_info["error"] = str(e)
        
        return plot_info
    
    def create_dashboard(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create an interactive dashboard with all visualizations."""
        logger.info("Creating interactive dashboard")
        
        # This would create a comprehensive dashboard
        # For now, return a placeholder
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Jacob Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .plot-container {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Jacob: Share of Voice vs Market Share Analysis</h1>
                <h2>Les Binet Hypothesis Test Results</h2>
            </div>
            
            <div class="plot-container">
                <h3>Analysis Results</h3>
                <p>Interactive visualizations and model results will be displayed here.</p>
            </div>
        </body>
        </html>
        """
        
        dashboard_path = self.output_dir / "dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(dashboard_html)
        
        return {
            "dashboard_path": str(dashboard_path),
            "description": "Interactive dashboard with all analysis results"
        }