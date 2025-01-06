
MlFinlab is a python package which helps portfolio managers and traders who want to leverage the power of machine learning by providing reproducible, interpretable, and easy to use tools. 

## Installation

For users:
```bash
pip install mlfinlab
```

Or with poetry:
```bash
poetry add mlfinlab
```

For developers:
```bash
# Clone the repository
git clone https://github.com/hudson-and-thames/mlfinlab.git
cd mlfinlab

# Install dependencies with poetry
poetry install

# Run tests
poetry run pytest
```

## Project Structure

```
mlfinlab/
├── backtest_statistics/    # Backtesting tools and statistics
├── bet_sizing/            # Position sizing and bet sizing tools
├── clustering/            # Clustering algorithms for financial data
├── codependence/          # Codependence and correlation metrics
├── cross_validation/      # Cross-validation for financial data
├── data_structures/       # Financial data structures
├── datasets/              # Sample datasets and loaders
├── ensemble/              # Ensemble methods
├── feature_importance/    # Feature importance analysis
├── features/             # Feature engineering tools
├── filters/              # Financial data filters
├── labeling/             # Financial data labeling tools
├── microstructural_features/  # Market microstructure features
├── multi_product/        # Multi-product analysis
├── online_portfolio_selection/  # Online portfolio selection
├── portfolio_optimization/  # Portfolio optimization tools
├── sample_weights/       # Sample weight generation
├── sampling/             # Financial data sampling
├── structural_breaks/    # Structural break detection
└── tests/               # Unit tests
```

## Features

We source all of our implementations from the most elite and peer-reviewed journals. Including publications from: 
1. The Journal of Financial Data Science
2. The Journal of Portfolio Management
3. The Journal of Algorithmic Finance
4. Cambridge University Press

The package has its foundations in the two graduate level textbooks: 
1. Advances in Financial Machine Learning
2. Machine Learning for Asset Managers

## Development

### Building Documentation

Documentation can be built in several ways:

1. Using sphinx-build directly (recommended):
```bash
poetry run sphinx-build -b html docs/source docs/build/html
```

2. Using make:
```bash
cd docs && poetry run make html
```

3. Using make.bat on Windows:
```bash
cd docs && poetry run make.bat html
```

The documentation will be generated in `docs/build/html/`.

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=mlfinlab

# Run specific test file
poetry run pytest mlfinlab/tests/test_specific_file.py
```

## Praise for MlFinLab
> "Financial markets are complex systems like no other. Extracting signal from financial data requires specialized tools
> that are distinct from those used in general machine learning. The MlFinLab package compiles important algorithms 
> that every quant should know and use."

Dr. Marcos Lopez de Prado, Co-founder and CIO at True Positive Technologies; Professor of Practice at Cornell University

>"Those who doubt open source libraries just need to look at the impact of Pandas, Scikit-learn, and the like. MIFinLab 
is doing to financial machine learning what Tensorflow and PyTorch are doing to deep learning."

Dr. Ernest Chan, Hedge Fund Manager at QTS & Author

>"For many decades, finance has relied on overly simplistic statistical techniques to identify patterns in data. 
>Machine learning promises to change that by allowing researchers to use modern nonlinear and highly dimensional 
>techniques. Yet, applying those machine learning algorithms to model financial problems is easier said than done: 
>finance is not a plug-and-play subject as it relates to machine learning.
>
>MlFinLab provides access to the latest cutting edges methods. MlFinLab is thus essential for quants who want to be 
>ahead of the technology rather than being replaced by it."

Dr. Thomas Raffinot, Financial Data Scientist at ENGIE Global Markets


## Contributing
We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details on how to get involved.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
