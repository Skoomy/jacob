"""
Bayesian Market Share Model using PyMC.
Tests Les Binet's hypothesis that share of voice predicts market share.
"""

import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Any, Optional, Tuple
from scipy import stats
import warnings

logger = logging.getLogger(__name__)


class BayesianMarketShareModel:
    """
    Bayesian model for testing share of voice vs market share hypothesis.
    
    This model implements a hierarchical Bayesian approach to test whether
    share of voice (from Google Trends) can predict market share across
    different brands and time periods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get("modeling", {})
        self.prior_config = self.model_config.get("prior_distributions", {})
        
        # Model parameters
        self.mcmc_samples = self.model_config.get("mcmc_samples", 2000)
        self.tune_samples = self.model_config.get("tune_samples", 1000)
        self.chains = self.model_config.get("chains", 4)
        self.target_accept = self.model_config.get("target_accept", 0.95)
        self.random_seed = self.model_config.get("random_seed", 42)
        
        # Model objects
        self.model = None
        self.trace = None
        self.posterior_predictive = None
        
        # Results storage
        self.model_results = {}
        self.diagnostics = {}
        
    def fit(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit the Bayesian model to the data.
        
        Args:
            data: Dictionary containing 'share_of_voice' and 'market_share' DataFrames
            
        Returns:
            Dictionary containing model results
        """
        logger.info("Fitting Bayesian market share model")
        
        # Prepare data
        X, y, brand_idx, time_idx = self._prepare_data(data)
        
        # Build model
        self.model = self._build_model(X, y, brand_idx, time_idx)
        
        # Fit model
        with self.model:
            logger.info("Starting MCMC sampling...")
            self.trace = pm.sample(
                draws=self.mcmc_samples,
                tune=self.tune_samples,
                chains=self.chains,
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True
            )
            
            logger.info("Generating posterior predictive samples...")
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                random_seed=self.random_seed
            )
        
        # Store results
        self.model_results = {
            "trace": self.trace,
            "posterior_predictive": self.posterior_predictive,
            "model": self.model,
            "data_shape": X.shape,
            "n_brands": len(np.unique(brand_idx)),
            "n_time_periods": len(np.unique(time_idx))
        }
        
        logger.info("Model fitting completed")
        return self.model_results
    
    def _prepare_data(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for modeling."""
        logger.info("Preparing data for Bayesian modeling")
        
        share_of_voice = data.get("share_of_voice")
        market_share = data.get("market_share")
        
        if share_of_voice is None or market_share is None:
            raise ValueError("Both 'share_of_voice' and 'market_share' data are required")
        
        # Convert to long format for hierarchical modeling
        sov_long = share_of_voice.stack().reset_index()
        sov_long.columns = ['time', 'brand', 'share_of_voice']
        
        # For market share, we'll use a simplified approach
        # In practice, you'd have actual market share data
        # Here we'll create synthetic market share data for demonstration
        if isinstance(market_share, pd.DataFrame):
            ms_long = market_share.stack().reset_index()
            ms_long.columns = ['time', 'brand', 'market_share']
        else:
            # Create synthetic market share data correlated with share of voice
            np.random.seed(42)
            ms_long = sov_long.copy()
            ms_long['market_share'] = (
                ms_long['share_of_voice'] * 0.7 + 
                np.random.normal(0, 0.1, len(ms_long))
            ).clip(0, 1)
        
        # Merge data
        combined_data = pd.merge(sov_long, ms_long, on=['time', 'brand'], how='inner')
        
        # Remove rows with missing data
        combined_data = combined_data.dropna()
        
        # Create indices for hierarchical structure
        brand_names = combined_data['brand'].unique()
        time_periods = combined_data['time'].unique()
        
        brand_to_idx = {brand: i for i, brand in enumerate(brand_names)}
        time_to_idx = {time: i for i, time in enumerate(time_periods)}
        
        # Prepare arrays
        X = combined_data['share_of_voice'].values
        y = combined_data['market_share'].values
        brand_idx = combined_data['brand'].map(brand_to_idx).values
        time_idx = combined_data['time'].map(time_to_idx).values
        
        logger.info(f"Prepared data: {len(X)} observations, {len(brand_names)} brands, {len(time_periods)} time periods")
        
        return X, y, brand_idx, time_idx
    
    def _build_model(self, X: np.ndarray, y: np.ndarray, 
                    brand_idx: np.ndarray, time_idx: np.ndarray) -> pm.Model:
        """Build the hierarchical Bayesian model."""
        logger.info("Building hierarchical Bayesian model")
        
        n_brands = len(np.unique(brand_idx))
        n_time_periods = len(np.unique(time_idx))
        
        with pm.Model() as model:
            # Hyperpriors for population-level parameters
            mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=1)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
            
            mu_beta = pm.Normal("mu_beta", mu=0.5, sigma=0.5)  # Expect positive relationship
            sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.5)
            
            # Brand-level intercepts (random effects)
            alpha_brand = pm.Normal("alpha_brand", mu=mu_alpha, sigma=sigma_alpha, shape=n_brands)
            
            # Brand-level slopes (random effects)
            beta_brand = pm.Normal("beta_brand", mu=mu_beta, sigma=sigma_beta, shape=n_brands)
            
            # Time-varying effects (optional)
            sigma_time = pm.HalfNormal("sigma_time", sigma=0.1)
            time_effect = pm.Normal("time_effect", mu=0, sigma=sigma_time, shape=n_time_periods)
            
            # Model for the mean
            mu = (alpha_brand[brand_idx] + 
                  beta_brand[brand_idx] * X + 
                  time_effect[time_idx])
            
            # Observation noise
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.1)
            
            # Likelihood
            likelihood = pm.Normal("likelihood", mu=mu, sigma=sigma_obs, observed=y)
            
            # Derived quantities
            # Population-level correlation
            pop_correlation = pm.Deterministic("pop_correlation", 
                                             pm.math.mean(beta_brand))
            
            # Brand-specific correlations
            brand_correlations = pm.Deterministic("brand_correlations", beta_brand)
            
        logger.info("Model building completed")
        return model
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """Get MCMC convergence diagnostics."""
        if self.trace is None:
            return {}
        
        diagnostics = {
            "rhat": az.rhat(self.trace).max().values.item(),
            "ess_bulk": az.ess(self.trace, kind="bulk").min().values.item(),
            "ess_tail": az.ess(self.trace, kind="tail").min().values.item(),
            "mcse_mean": az.mcse(self.trace, method="mean").max().values.item(),
            "mcse_sd": az.mcse(self.trace, method="sd").max().values.item(),
        }
        
        # Add warnings for convergence issues
        warnings = []
        if diagnostics["rhat"] > 1.01:
            warnings.append(f"High R-hat detected: {diagnostics['rhat']:.3f}")
        if diagnostics["ess_bulk"] < 400:
            warnings.append(f"Low bulk ESS: {diagnostics['ess_bulk']:.0f}")
        if diagnostics["ess_tail"] < 400:
            warnings.append(f"Low tail ESS: {diagnostics['ess_tail']:.0f}")
        
        diagnostics["warnings"] = warnings
        diagnostics["converged"] = len(warnings) == 0
        
        return diagnostics
    
    def get_parameter_estimates(self) -> Dict[str, Any]:
        """Get parameter estimates from the model."""
        if self.trace is None:
            return {}
        
        summary = az.summary(self.trace, hdi_prob=0.95)
        
        # Extract key parameters
        estimates = {}
        
        # Population-level parameters
        if "mu_alpha" in summary.index:
            estimates["population_intercept"] = {
                "mean": summary.loc["mu_alpha", "mean"],
                "hdi_low": summary.loc["mu_alpha", "hdi_2.5%"],
                "hdi_high": summary.loc["mu_alpha", "hdi_97.5%"]
            }
        
        if "mu_beta" in summary.index:
            estimates["population_slope"] = {
                "mean": summary.loc["mu_beta", "mean"],
                "hdi_low": summary.loc["mu_beta", "hdi_2.5%"],
                "hdi_high": summary.loc["mu_beta", "hdi_97.5%"]
            }
        
        # Brand-level parameters
        brand_params = summary[summary.index.str.startswith("beta_brand")]
        if not brand_params.empty:
            estimates["brand_slopes"] = {
                "mean": brand_params["mean"].values,
                "hdi_low": brand_params["hdi_2.5%"].values,
                "hdi_high": brand_params["hdi_97.5%"].values
            }
        
        # Population correlation
        if "pop_correlation" in summary.index:
            estimates["population_correlation"] = {
                "mean": summary.loc["pop_correlation", "mean"],
                "hdi_low": summary.loc["pop_correlation", "hdi_2.5%"],
                "hdi_high": summary.loc["pop_correlation", "hdi_97.5%"]
            }
        
        return estimates
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        if self.trace is None:
            return {}
        
        summary = {
            "model_type": "Hierarchical Bayesian Market Share Model",
            "hypothesis": "Share of Voice predicts Market Share (Les Binet)",
            "mcmc_info": {
                "samples": self.mcmc_samples,
                "tune": self.tune_samples,
                "chains": self.chains,
                "target_accept": self.target_accept
            },
            "data_info": self.model_results.get("data_shape", "Unknown"),
            "convergence": self.get_convergence_diagnostics(),
            "parameters": self.get_parameter_estimates()
        }
        
        return summary
    
    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive model diagnostics."""
        if self.trace is None:
            return {"error": "Model not fitted yet"}
        
        diagnostics = {}
        
        # Convergence diagnostics
        diagnostics["convergence"] = self.get_convergence_diagnostics()
        
        # Posterior predictive checks
        diagnostics["posterior_predictive"] = self._run_posterior_predictive_checks()
        
        # Model comparison metrics
        diagnostics["model_metrics"] = self._calculate_model_metrics()
        
        return diagnostics
    
    def _run_posterior_predictive_checks(self) -> Dict[str, Any]:
        """Run posterior predictive checks."""
        if self.posterior_predictive is None:
            return {"error": "Posterior predictive not available"}
        
        # Extract observed and predicted values
        y_obs = self.posterior_predictive.observed_data["likelihood"].values
        y_pred = self.posterior_predictive.posterior_predictive["likelihood"].values
        
        # Calculate summary statistics
        ppc_stats = {
            "mean_observed": float(np.mean(y_obs)),
            "mean_predicted": float(np.mean(y_pred)),
            "std_observed": float(np.std(y_obs)),
            "std_predicted": float(np.mean(np.std(y_pred, axis=0))),
            "min_observed": float(np.min(y_obs)),
            "min_predicted": float(np.mean(np.min(y_pred, axis=0))),
            "max_observed": float(np.max(y_obs)),
            "max_predicted": float(np.mean(np.max(y_pred, axis=0)))
        }
        
        return ppc_stats
    
    def _calculate_model_metrics(self) -> Dict[str, Any]:
        """Calculate model comparison metrics."""
        if self.trace is None:
            return {}
        
        try:
            # Calculate WAIC and LOO
            waic = az.waic(self.trace)
            loo = az.loo(self.trace)
            
            metrics = {
                "waic": {
                    "waic": float(waic.waic),
                    "p_waic": float(waic.p_waic),
                    "waic_se": float(waic.waic_se)
                },
                "loo": {
                    "loo": float(loo.loo),
                    "p_loo": float(loo.p_loo),
                    "loo_se": float(loo.loo_se)
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Could not calculate model metrics: {e}")
            return {"error": str(e)}
    
    def predict(self, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions for new data."""
        if self.trace is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # This would implement prediction logic
        # For now, return placeholder
        return {
            "predictions": "Not implemented yet",
            "prediction_intervals": "Not implemented yet"
        }