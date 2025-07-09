"""
Report generation system for Jacob analysis platform.
Creates comprehensive .txt reports with key findings and results.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import numpy as np

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates comprehensive text reports for analysis results.
    Reports are saved as .txt files and accessible throughout the pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.report_config = config.get("reporting", {})
        self.output_dir = Path(config.get("output_dir", "output"))
        self.output_dir.mkdir(exist_ok=True)
        
        # Report configuration
        self.include_sections = self.report_config.get("include_sections", [
            "executive_summary", "methodology", "data_overview", 
            "model_results", "model_diagnostics", "conclusions", "recommendations"
        ])
        self.timestamp_reports = self.report_config.get("timestamp_reports", True)
        
        # Running report for pipeline access
        self.running_report = {
            "key_findings": [],
            "metrics": {},
            "warnings": [],
            "recommendations": []
        }
    
    def generate_all_reports(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all configured reports."""
        logger.info("Generating comprehensive analysis reports")
        
        reports = {}
        
        # Generate main analysis report
        reports["main_report"] = self.generate_main_report(report_data)
        
        # Generate executive summary
        reports["executive_summary"] = self.generate_executive_summary(report_data)
        
        # Generate technical report
        reports["technical_report"] = self.generate_technical_report(report_data)
        
        # Generate dashboard report
        reports["dashboard_report"] = self.generate_dashboard_report(report_data)
        
        # Update running report
        self.update_running_report(report_data)
        
        logger.info(f"Generated {len(reports)} reports")
        return reports
    
    def generate_main_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the main comprehensive analysis report."""
        logger.info("Generating main analysis report")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jacob_analysis_report_{timestamp}.txt" if self.timestamp_reports else "jacob_analysis_report.txt"
        
        report_content = self._build_main_report_content(report_data)
        
        # Save report
        report_path = self.output_dir / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        return {
            "filename": filename,
            "file_path": str(report_path),
            "content_length": len(report_content),
            "sections": self.include_sections
        }
    
    def _build_main_report_content(self, report_data: Dict[str, Any]) -> str:
        """Build the main report content."""
        
        industry = report_data.get("config", {}).get("industry", "Unknown")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = []
        report.append("="*80)
        report.append("JACOB: SHARE OF VOICE VS MARKET SHARE ANALYSIS")
        report.append("Testing Les Binet's Hypothesis Across Industries")
        report.append("="*80)
        report.append(f"")
        report.append(f"Industry: {industry}")
        report.append(f"Analysis Date: {timestamp}")
        report.append(f"")
        
        # Executive Summary
        if "executive_summary" in self.include_sections:
            report.extend(self._build_executive_summary_section(report_data))
        
        # Methodology
        if "methodology" in self.include_sections:
            report.extend(self._build_methodology_section(report_data))
        
        # Data Overview
        if "data_overview" in self.include_sections:
            report.extend(self._build_data_overview_section(report_data))
        
        # Model Results
        if "model_results" in self.include_sections:
            report.extend(self._build_model_results_section(report_data))
        
        # Model Diagnostics
        if "model_diagnostics" in self.include_sections:
            report.extend(self._build_model_diagnostics_section(report_data))
        
        # Conclusions
        if "conclusions" in self.include_sections:
            report.extend(self._build_conclusions_section(report_data))
        
        # Recommendations
        if "recommendations" in self.include_sections:
            report.extend(self._build_recommendations_section(report_data))
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)
    
    def _build_executive_summary_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Build executive summary section."""
        section = []
        section.append("EXECUTIVE SUMMARY")
        section.append("-"*50)
        section.append("")
        
        # Extract key findings
        model_results = report_data.get("model_results", {})
        parameters = model_results.get("parameters", {})
        
        if "population_correlation" in parameters:
            pop_corr = parameters["population_correlation"]
            corr_mean = pop_corr.get("mean", 0)
            corr_low = pop_corr.get("hdi_low", 0)
            corr_high = pop_corr.get("hdi_high", 0)
            
            section.append(f"KEY FINDING: Share of Voice → Market Share Correlation")
            section.append(f"Population-level correlation: {corr_mean:.3f}")
            section.append(f"95% Credible Interval: [{corr_low:.3f}, {corr_high:.3f}]")
            section.append("")
            
            # Interpret the results
            if corr_low > 0:
                section.append("✓ HYPOTHESIS SUPPORTED: Strong evidence for Les Binet's hypothesis.")
                section.append("  Share of voice positively predicts market share.")
            elif corr_high < 0:
                section.append("✗ HYPOTHESIS REJECTED: Evidence against Les Binet's hypothesis.")
                section.append("  Share of voice negatively predicts market share.")
            else:
                section.append("? HYPOTHESIS UNCERTAIN: Mixed evidence for Les Binet's hypothesis.")
                section.append("  Credible interval includes zero.")
        else:
            section.append("No correlation results available in model output.")
        
        section.append("")
        
        # Pipeline summary
        pipeline_state = report_data.get("pipeline_state", {})
        errors = pipeline_state.get("errors", [])
        
        section.append(f"ANALYSIS STATUS:")
        section.append(f"Steps completed: {len(pipeline_state.get('steps_completed', []))}")
        section.append(f"Errors encountered: {len(errors)}")
        
        if errors:
            section.append("ERRORS:")
            for error in errors[:3]:  # Show first 3 errors
                section.append(f"  - {error}")
        
        section.append("")
        return section
    
    def _build_methodology_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Build methodology section."""
        section = []
        section.append("METHODOLOGY")
        section.append("-"*50)
        section.append("")
        
        section.append("This analysis tests Les Binet's hypothesis that 'Share of Voice'")
        section.append("can predict 'Market Share' using a hierarchical Bayesian approach.")
        section.append("")
        
        section.append("DATA SOURCES:")
        section.append("• Share of Voice: Google Trends data (search volume)")
        section.append("• Market Share: UK Government statistics")
        section.append("")
        
        section.append("STATISTICAL MODEL:")
        section.append("• Hierarchical Bayesian regression")
        section.append("• Brand-level random effects")
        section.append("• Time-varying components")
        section.append("• PyMC implementation with NUTS sampler")
        section.append("")
        
        # Model configuration
        config = report_data.get("config", {})
        model_config = config.get("modeling", {})
        
        section.append("MODEL CONFIGURATION:")
        section.append(f"• MCMC Samples: {model_config.get('mcmc_samples', 'N/A')}")
        section.append(f"• Chains: {model_config.get('chains', 'N/A')}")
        section.append(f"• Target Accept: {model_config.get('target_accept', 'N/A')}")
        section.append("")
        
        return section
    
    def _build_data_overview_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Build data overview section."""
        section = []
        section.append("DATA OVERVIEW")
        section.append("-"*50)
        section.append("")
        
        data_summary = report_data.get("data_summary", {})
        
        section.append("DATA COLLECTION SUMMARY:")
        existing_datasets = data_summary.get("existing_datasets", [])
        collected_datasets = data_summary.get("collected_datasets", [])
        
        section.append(f"• Existing datasets: {len(existing_datasets)}")
        for dataset in existing_datasets:
            section.append(f"  - {dataset}")
        
        section.append(f"• Newly collected datasets: {len(collected_datasets)}")
        for dataset in collected_datasets:
            section.append(f"  - {dataset}")
        
        section.append("")
        
        # Data processing summary
        preprocessing = data_summary.get("preprocessing", {})
        if preprocessing:
            section.append("DATA PREPROCESSING:")
            processed_datasets = preprocessing.get("processed_datasets", [])
            for dataset in processed_datasets:
                section.append(f"  - {dataset}")
        
        section.append("")
        return section
    
    def _build_model_results_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Build model results section."""
        section = []
        section.append("MODEL RESULTS")
        section.append("-"*50)
        section.append("")
        
        model_results = report_data.get("model_results", {})
        parameters = model_results.get("parameters", {})
        
        # Population-level results
        if "population_correlation" in parameters:
            pop_corr = parameters["population_correlation"]
            section.append("POPULATION-LEVEL CORRELATION:")
            section.append(f"  Mean: {pop_corr.get('mean', 0):.4f}")
            section.append(f"  95% HDI: [{pop_corr.get('hdi_low', 0):.4f}, {pop_corr.get('hdi_high', 0):.4f}]")
            section.append("")
        
        # Brand-level results
        if "brand_slopes" in parameters:
            brand_slopes = parameters["brand_slopes"]
            section.append("BRAND-LEVEL SLOPES:")
            means = brand_slopes.get("mean", [])
            
            for i, mean in enumerate(means[:5]):  # Show first 5 brands
                section.append(f"  Brand {i+1}: {mean:.4f}")
            
            if len(means) > 5:
                section.append(f"  ... and {len(means) - 5} more brands")
            section.append("")
        
        # Model fit statistics
        if "model_type" in model_results:
            section.append(f"MODEL TYPE: {model_results['model_type']}")
            section.append("")
        
        return section
    
    def _build_model_diagnostics_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Build model diagnostics section."""
        section = []
        section.append("MODEL DIAGNOSTICS")
        section.append("-"*50)
        section.append("")
        
        model_results = report_data.get("model_results", {})
        convergence = model_results.get("convergence_diagnostics", {})
        
        if convergence:
            section.append("CONVERGENCE DIAGNOSTICS:")
            section.append(f"  R-hat (max): {convergence.get('rhat', 'N/A'):.4f}")
            section.append(f"  ESS Bulk (min): {convergence.get('ess_bulk', 'N/A'):.0f}")
            section.append(f"  ESS Tail (min): {convergence.get('ess_tail', 'N/A'):.0f}")
            section.append(f"  Converged: {convergence.get('converged', 'Unknown')}")
            section.append("")
            
            warnings = convergence.get("warnings", [])
            if warnings:
                section.append("CONVERGENCE WARNINGS:")
                for warning in warnings:
                    section.append(f"  ⚠ {warning}")
                section.append("")
        
        # Diagnostics from model
        diagnostics = model_results.get("diagnostics", {})
        if diagnostics:
            section.append("ADDITIONAL DIAGNOSTICS:")
            
            # Posterior predictive checks
            ppc = diagnostics.get("posterior_predictive", {})
            if ppc and "mean_observed" in ppc:
                section.append("  Posterior Predictive Checks:")
                section.append(f"    Mean observed: {ppc.get('mean_observed', 0):.4f}")
                section.append(f"    Mean predicted: {ppc.get('mean_predicted', 0):.4f}")
                section.append("")
            
            # Model comparison metrics
            metrics = diagnostics.get("model_metrics", {})
            if metrics and "waic" in metrics:
                waic = metrics["waic"]
                section.append("  Model Comparison:")
                section.append(f"    WAIC: {waic.get('waic', 'N/A'):.2f}")
                section.append(f"    p_WAIC: {waic.get('p_waic', 'N/A'):.2f}")
                section.append("")
        
        return section
    
    def _build_conclusions_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Build conclusions section."""
        section = []
        section.append("CONCLUSIONS")
        section.append("-"*50)
        section.append("")
        
        model_results = report_data.get("model_results", {})
        parameters = model_results.get("parameters", {})
        
        if "population_correlation" in parameters:
            pop_corr = parameters["population_correlation"]
            corr_mean = pop_corr.get("mean", 0)
            corr_low = pop_corr.get("hdi_low", 0)
            corr_high = pop_corr.get("hdi_high", 0)
            
            section.append("HYPOTHESIS EVALUATION:")
            section.append("")
            
            if corr_low > 0:
                section.append("✓ STRONG SUPPORT for Les Binet's hypothesis")
                section.append("  The 95% credible interval excludes zero, indicating")
                section.append("  a positive relationship between share of voice and market share.")
                section.append("")
                section.append("  INTERPRETATION:")
                section.append("  • Increased share of voice leads to increased market share")
                section.append("  • The relationship is statistically significant")
                section.append("  • Marketing investments in visibility pay off")
                
            elif corr_high < 0:
                section.append("✗ STRONG EVIDENCE AGAINST Les Binet's hypothesis")
                section.append("  The 95% credible interval excludes zero and is negative,")
                section.append("  indicating an inverse relationship.")
                section.append("")
                section.append("  INTERPRETATION:")
                section.append("  • Increased share of voice leads to decreased market share")
                section.append("  • This contradicts the expected relationship")
                section.append("  • May indicate market saturation or other effects")
                
            else:
                section.append("? INCONCLUSIVE EVIDENCE for Les Binet's hypothesis")
                section.append("  The 95% credible interval includes zero, indicating")
                section.append("  uncertainty about the relationship direction.")
                section.append("")
                section.append("  INTERPRETATION:")
                section.append("  • The relationship may be weak or non-linear")
                section.append("  • More data or different modeling approaches needed")
                section.append("  • Industry-specific factors may be important")
            
            section.append("")
            section.append(f"EFFECT SIZE: {corr_mean:.3f}")
            section.append(f"This represents a {abs(corr_mean)*100:.1f}% change in market share")
            section.append(f"for each unit change in share of voice.")
        
        section.append("")
        return section
    
    def _build_recommendations_section(self, report_data: Dict[str, Any]) -> List[str]:
        """Build recommendations section."""
        section = []
        section.append("RECOMMENDATIONS")
        section.append("-"*50)
        section.append("")
        
        model_results = report_data.get("model_results", {})
        parameters = model_results.get("parameters", {})
        
        section.append("STRATEGIC RECOMMENDATIONS:")
        section.append("")
        
        if "population_correlation" in parameters:
            pop_corr = parameters["population_correlation"]
            corr_mean = pop_corr.get("mean", 0)
            corr_low = pop_corr.get("hdi_low", 0)
            
            if corr_low > 0:
                section.append("1. INVEST IN SHARE OF VOICE")
                section.append("   • Increase marketing spend on visibility")
                section.append("   • Focus on search engine optimization")
                section.append("   • Expand brand awareness campaigns")
                section.append("")
                
                section.append("2. MONITOR COMPETITIVE LANDSCAPE")
                section.append("   • Track competitors' share of voice")
                section.append("   • Identify opportunities for differentiation")
                section.append("   • Adjust strategy based on market dynamics")
                section.append("")
                
                section.append("3. MEASURE AND OPTIMIZE")
                section.append("   • Implement continuous monitoring")
                section.append("   • Test different messaging strategies")
                section.append("   • Optimize channel allocation")
                
            else:
                section.append("1. INVESTIGATE UNDERLYING FACTORS")
                section.append("   • Analyze market saturation effects")
                section.append("   • Examine quality vs. quantity trade-offs")
                section.append("   • Consider non-linear relationships")
                section.append("")
                
                section.append("2. REFINE MEASUREMENT APPROACH")
                section.append("   • Improve market share data quality")
                section.append("   • Consider alternative SOV metrics")
                section.append("   • Extend analysis time period")
                
        section.append("")
        
        section.append("TECHNICAL RECOMMENDATIONS:")
        section.append("")
        section.append("1. DATA QUALITY IMPROVEMENTS")
        section.append("   • Expand data collection frequency")
        section.append("   • Include additional market indicators")
        section.append("   • Validate with external data sources")
        section.append("")
        
        section.append("2. MODEL ENHANCEMENTS")
        section.append("   • Test non-linear relationships")
        section.append("   • Include external variables (seasonality, etc.)")
        section.append("   • Cross-validate with other industries")
        section.append("")
        
        section.append("3. OPERATIONAL INTEGRATION")
        section.append("   • Automate regular analysis updates")
        section.append("   • Create real-time dashboards")
        section.append("   • Integrate with marketing systems")
        
        section.append("")
        return section
    
    def generate_executive_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"executive_summary_{timestamp}.txt"
        
        # Build executive summary content
        content = []
        content.append("EXECUTIVE SUMMARY")
        content.append("="*50)
        content.append("")
        content.extend(self._build_executive_summary_section(report_data))
        
        # Save summary
        report_path = self.output_dir / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        return {
            "filename": filename,
            "file_path": str(report_path),
            "content_length": len("\n".join(content))
        }
    
    def generate_technical_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate technical report for researchers."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"technical_report_{timestamp}.txt"
        
        # Build technical content
        content = []
        content.append("TECHNICAL REPORT")
        content.append("="*50)
        content.append("")
        content.extend(self._build_methodology_section(report_data))
        content.extend(self._build_model_results_section(report_data))
        content.extend(self._build_model_diagnostics_section(report_data))
        
        # Save report
        report_path = self.output_dir / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        return {
            "filename": filename,
            "file_path": str(report_path),
            "content_length": len("\n".join(content))
        }
    
    def generate_dashboard_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dashboard summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dashboard_summary_{timestamp}.txt"
        
        # Build dashboard content
        content = []
        content.append("DASHBOARD SUMMARY")
        content.append("="*50)
        content.append("")
        
        visualizations = report_data.get("visualizations", {})
        content.append(f"VISUALIZATIONS CREATED: {len(visualizations)}")
        
        for viz_name, viz_data in visualizations.items():
            content.append(f"• {viz_name}: {viz_data.get('description', 'No description')}")
        
        content.append("")
        
        # Save report
        report_path = self.output_dir / filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        return {
            "filename": filename,
            "file_path": str(report_path),
            "content_length": len("\n".join(content))
        }
    
    def update_running_report(self, report_data: Dict[str, Any]) -> None:
        """Update the running report accessible throughout the pipeline."""
        model_results = report_data.get("model_results", {})
        parameters = model_results.get("parameters", {})
        
        # Update key findings
        if "population_correlation" in parameters:
            pop_corr = parameters["population_correlation"]
            corr_mean = pop_corr.get("mean", 0)
            
            self.running_report["key_findings"].append({
                "finding": "Population correlation between share of voice and market share",
                "value": corr_mean,
                "timestamp": datetime.now().isoformat()
            })
        
        # Update metrics
        convergence = model_results.get("convergence_diagnostics", {})
        if convergence:
            self.running_report["metrics"].update({
                "rhat_max": convergence.get("rhat"),
                "ess_bulk_min": convergence.get("ess_bulk"),
                "converged": convergence.get("converged")
            })
        
        # Update warnings
        warnings = convergence.get("warnings", [])
        self.running_report["warnings"].extend(warnings)
        
        # Save running report
        running_report_path = self.output_dir / "running_report.json"
        with open(running_report_path, "w") as f:
            json.dump(self.running_report, f, indent=2, default=str)
    
    def get_key_findings(self) -> List[Dict[str, Any]]:
        """Get key findings from the running report."""
        return self.running_report.get("key_findings", [])
    
    def get_running_metrics(self) -> Dict[str, Any]:
        """Get current metrics from the running report."""
        return self.running_report.get("metrics", {})
    
    def generate_final_report(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final comprehensive report."""
        logger.info("Generating final comprehensive report")
        
        # Generate all reports
        reports = self.generate_all_reports(pipeline_results)
        
        # Create final summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_report_path = self.output_dir / f"FINAL_REPORT_{timestamp}.txt"
        
        with open(final_report_path, "w", encoding="utf-8") as f:
            f.write("JACOB ANALYSIS PLATFORM - FINAL REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Reports generated: {len(reports)}\n")
            f.write("\nFILES CREATED:\n")
            
            for report_name, report_info in reports.items():
                f.write(f"• {report_name}: {report_info.get('filename', 'N/A')}\n")
            
            f.write("\nKEY FINDINGS:\n")
            for finding in self.get_key_findings():
                f.write(f"• {finding['finding']}: {finding['value']}\n")
        
        return {
            "final_report_path": str(final_report_path),
            "reports_generated": reports,
            "key_findings": self.get_key_findings()
        }