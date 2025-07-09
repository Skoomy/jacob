# Jacob: Share of Voice vs Market Share Analysis Platform

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![PyMC](https://img.shields.io/badge/PyMC-5.0+-green.svg)
![Plotly](https://img.shields.io/badge/Plotly-5.15+-orange.svg)

A Nobel Prize-level analysis platform for testing **Les Binet's hypothesis** that Share of Voice can predict Market Share across different industries. This project implements a rigorous Bayesian modeling approach using PyMC to analyze the relationship between search volume (Google Trends) and market share data.

## 🎯 Project Overview

Jacob tests the fundamental marketing hypothesis proposed by Les Binet: **"Share of Voice drives Market Share"**. Using modern Bayesian statistical methods, the platform analyzes this relationship across multiple industries:

- **Automotive Industry**: Electric vehicle market analysis
- **Pharmaceutical Industry**: Healthcare market dynamics  
- **Alcoholic Beverage Industry**: Consumer goods market patterns

## 🏗️ Architecture

The platform follows a modular, config-driven architecture designed for scalability and reproducibility:

```
jacob/
├── src/
│   ├── config/           # Configuration management
│   ├── data_manager/     # Data collection & processing
│   ├── modeling/         # Bayesian modeling (PyMC)
│   ├── visualization/    # Plotly visualizations
│   ├── pipeline/         # Analysis orchestration
│   └── reporting/        # Comprehensive reporting
├── config/              # Industry-specific configs
├── docker/              # Containerization
├── docs/                # Documentation
└── output/              # Analysis results
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Make (for convenient commands)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jacob.git
cd jacob

# Run setup script to create all directories
./scripts/setup.sh

# Check system requirements
make check-requirements

# Build Docker images
make build

# Initialize configuration for automotive industry
make init-config INDUSTRY=automotive
```

### Running Analysis

```bash
# Run complete automotive industry analysis
make analyze-automotive

# Or run the full pipeline
make full-analysis INDUSTRY=automotive

# Start development environment
make dev

# Launch Jupyter Lab for exploration
make jupyter
```

### 📁 File Access

All files are directly accessible on your local machine - **no need to enter Docker containers**:

```
jacob/
├── src/           # Source code (edit directly)
├── config/        # Configuration files
├── data/          # Input data
├── output/        # Analysis results
├── docs/          # Documentation
├── tests/         # Test files
├── jupyterlab/    # Jupyter notebooks
└── logs/          # Application logs
```

**Key Benefits:**
- Edit source code with your favorite IDE
- Access results immediately after analysis
- View logs in real-time
- Manage configurations directly
- No Docker exec commands needed

## 📊 Key Features

### 1. **Bayesian Modeling**
- Hierarchical Bayesian regression using PyMC
- Brand-level random effects
- Time-varying components
- Comprehensive convergence diagnostics

### 2. **Data Integration**
- Google Trends API for Share of Voice
- UK Government statistics for Market Share
- Automated data collection and processing
- Cross-industry comparison capabilities

### 3. **Professional Visualizations**
- Publication-ready Plotly charts
- Interactive dashboards
- Multiple export formats (HTML, PNG, SVG)
- Consistent styling and branding

### 4. **Comprehensive Reporting**
- Executive summaries
- Technical reports
- Model diagnostics
- Actionable recommendations
- All reports saved as .txt files

## 🔬 Methodology

### Statistical Model

The platform implements a hierarchical Bayesian model:

```
Market Share ~ Normal(μ, σ)
μ = α_brand + β_brand * Share_of_Voice + γ_time
α_brand ~ Normal(μ_α, σ_α)  # Brand intercepts
β_brand ~ Normal(μ_β, σ_β)  # Brand slopes
γ_time ~ Normal(0, σ_γ)      # Time effects
```

### Data Sources

1. **Share of Voice**: Google Trends search volume data
2. **Market Share**: UK Government official statistics
3. **Temporal Coverage**: 2020-2024 (5-year analysis)
4. **Geographic Scope**: United Kingdom

## 🏭 Industry Coverage

### Automotive Industry
- **Brands**: Tesla, BMW, Mercedes, Audi, Jaguar, Volvo
- **Data Source**: UK vehicle registration statistics
- **Key Metrics**: Electric vehicle adoption, brand performance

### Pharmaceutical Industry
- **Brands**: Pfizer, GSK, AstraZeneca, Novartis, Roche
- **Data Source**: UK pharmaceutical sales data
- **Key Metrics**: Market penetration, brand awareness

### Alcoholic Beverage Industry
- **Brands**: Guinness, Stella Artois, Budweiser, Heineken
- **Data Source**: UK alcohol sales statistics
- **Key Metrics**: Consumer preference, market dynamics

## 📈 Usage Examples

### Command Line Interface

```bash
# Run analysis for specific industry
jacob analyze --industry automotive --config config/automotive.yaml

# Collect fresh data
jacob collect-data --industry automotive

# Generate configuration template
jacob init-config --config-template pharma --output config/pharma.yaml
```

### Docker Commands

```bash
# Development environment
make dev

# Production analysis
make prod

# Run tests
make test

# Code formatting
make format

# Generate documentation
make docs
```

## 🛠️ Development

### Project Structure

```python
# Main analysis pipeline
from src.pipeline.analysis_pipeline import AnalysisPipeline
from src.config.config_manager import ConfigManager

# Load configuration
config = ConfigManager.load_config("config/automotive.yaml")

# Run analysis
pipeline = AnalysisPipeline(config)
results = pipeline.run()
```

### Adding New Industries

1. Create industry configuration in `config/`
2. Add industry template to `ConfigManager.INDUSTRY_TEMPLATES`
3. Implement data collection logic in `UKGovDataCollector`
4. Update documentation and tests

### Testing

```bash
# Run full test suite
make test

# Run specific test categories
docker-compose run --rm jacob pytest tests/test_modeling.py -v
docker-compose run --rm jacob pytest tests/test_data_collection.py -v
```

## 📝 Configuration

### Industry Configuration Example

```yaml
# config/automotive.yaml
industry:
  name: "automotive"
  brands: ["Tesla", "BMW", "Mercedes", "Audi"]
  search_terms: ["electric car", "EV", "tesla model"]
  market_data_source: "uk_gov_vehicle_registrations"
  analysis_period:
    start: "2020-01-01"
    end: "2024-12-31"

modeling:
  mcmc_samples: 2000
  chains: 4
  target_accept: 0.95

visualization:
  theme: "plotly_white"
  export_formats: ["html", "png", "svg"]
```

## 📊 Output Files

After running analysis, the following files are generated:

```
output/
├── automotive_analysis_20240101_120000.txt    # Main report
├── executive_summary_20240101_120000.txt      # Executive summary
├── technical_report_20240101_120000.txt       # Technical details
├── trends_timeseries.html                     # Interactive plots
├── model_results.png                          # Model diagnostics
├── correlation_heatmap.svg                    # Correlation analysis
└── FINAL_REPORT_20240101_120000.txt          # Comprehensive summary
```

## 🤝 Contributing

This is an open-source project designed to showcase advanced analytics capabilities. Contributions are welcome!

### Development Setup

```bash
# Fork the repository
git clone https://github.com/yourusername/jacob.git
cd jacob

# Set up development environment
make setup-dev

# Run tests
make test

# Format code
make format

# Submit pull request
```

### Code Standards

- **Python**: PEP 8 compliance with Black formatting
- **Type Hints**: Full type annotation coverage
- **Documentation**: Comprehensive docstrings
- **Testing**: >90% code coverage
- **Containers**: All development through Docker

## 📚 Documentation

- **[API Documentation](docs/api.md)**: Complete API reference
- **[User Guide](docs/user_guide.md)**: Step-by-step tutorials
- **[Development Guide](docs/development.md)**: Contributing guidelines
- **[Claude.md](CLAUDE.md)**: AI development context

## 🎯 Results & Impact

### Key Findings

The platform generates publication-ready results including:

- **Correlation Coefficients**: Quantified relationship strength
- **Credible Intervals**: Bayesian uncertainty quantification
- **Brand-Level Effects**: Individual brand performance
- **Time-Series Analysis**: Temporal trend identification
- **Cross-Industry Comparisons**: Hypothesis validation across sectors

### Scientific Rigor

- **Bayesian Statistics**: Principled uncertainty quantification
- **Hierarchical Modeling**: Appropriate for grouped data
- **Convergence Diagnostics**: Rigorous model validation
- **Reproducibility**: Containerized, version-controlled analysis
- **Transparency**: Open-source methodology

## 🏆 Why This Matters

This project demonstrates:

1. **Advanced Statistical Modeling**: Modern Bayesian methods
2. **Software Engineering Excellence**: Production-ready code
3. **Data Science Best Practices**: Reproducible research
4. **Business Intelligence**: Actionable insights
5. **Open Science**: Transparent methodology

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Les Binet**: Original hypothesis and marketing science research
- **PyMC Team**: Bayesian modeling framework
- **Plotly Team**: Visualization capabilities
- **UK Government**: Open data initiatives

## 📚 References 

- [Les Binet's Marketing Research](https://www.youtube.com/watch?v=Ty7TqjHKBZo)
- [Bayesian Statistics in Marketing](https://en.wikipedia.org/wiki/Bayesian_statistics)
- [PyMC Documentation](https://docs.pymc.io/)
- [Plotly Documentation](https://plotly.com/python/)

---

**Jacob** - Testing marketing hypotheses with Nobel Prize-level rigor.

*Built with ❤️ for the analytics community*
