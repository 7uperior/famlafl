# FAMLAFL: FAMLAFL Aren’t Machine Learning And Financial Laboratory


## Installation

For users:
```bash
pip install famlafl
```

Or with poetry:
```bash
poetry add famlafl
```

## Project Structure

```
famlafl/
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

## Development

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=famlafl

# Run specific test file
poetry run pytest famlafl/tests/test_specific_file.py
```


## Contributing
We welcome contributions from the community! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details on how to get involved.

## License
This is a **fork** of [mlfinlab (ArbitrageLab)](https://github.com/hudson-and-thames/mlfinlab), 
developed by Hudson & Thames Quantitative Research.

> **Important**  
> - All mlfinlab-derived code here remains under Hudson & Thames’s 
>   [“all rights reserved” license](https://github.com/hudson-and-thames/mlfinlab#license).
> - Any new or original code that I (Vadim Surin) wrote **from scratch** (and **does not** derive from mlfinlab code) 
>   is released under the [BSD-3-Clause License](./LICENSE). 
>   However, usage in combination with mlfinlab code is still governed by Hudson & Thames’s restrictions.

### Licensing Overview

1. **Hudson & Thames License (All Rights Reserved)**  
   The original mlfinlab portion of this repository is subject to the 
   [Hudson & Thames license](https://github.com/hudson-and-thames/mlfinlab#license) 
   (or see the license text included in this repo’s `LICENSE` file). 
   
   Their license **overrides** any open-source terms with respect to the mlfinlab files.

2. **BSD-3-Clause (for My Independent Code)**  
   Purely original files that do not include or derive from mlfinlab 
   logic can be used under BSD-3-Clause terms. 
   
   > **Note**: If these files are used in conjunction with mlfinlab code, 
   > the combined work is effectively subject to Hudson & Thames’s license 
   > to the extent of mlfinlab’s portion.

### Usage

Feel free to experiment with my additions, but remember mlfinlab’s license 
requires you to comply with Hudson & Thames’s terms for the original 
(and derived) code.
