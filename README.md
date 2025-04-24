# ML Visualizations

A multipage Streamlit application showcasing various visualizations for Machine Learning and Data Science concepts.

## Overview

This project provides interactive visualizations to help understand common ML and data science concepts including:

- Data exploration and preprocessing
- Classification algorithms and decision boundaries
- Regression analysis and curve fitting
- Clustering techniques
- Dimensionality reduction methods
- Model evaluation metrics

## Project Structure

```
ml-visualizations/
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── .pre-commit-config.yaml  # Pre-commit configuration
├── .flake8                  # Flake8 linter configuration
├── setup.cfg                # Project configuration
├── src/                     # Source code directory
│   ├── Home.py              # Main entry point for the app
│   ├── utils/               # Utility functions
│   └── pages/               # App pages
│       ├── 1_📊_Data_Exploration.py
│       ├── 2_🔍_Classification.py
│       ├── 3_📈_Regression.py
│       ├── 4_🔮_Clustering.py
│       └── 5_📉_Dimensionality_Reduction.py
└── tests/                   # Unit tests
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ml-visualizations.git
cd ml-visualizations
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install pre-commit hooks:

```bash
pre-commit install
```

## Usage

Run the Streamlit application:

```bash
streamlit run src/Home.py
```

Navigate to http://localhost:8501 in your web browser to see the application.

## Development

- Follow the code style guidelines enforced by pre-commit hooks
- Run tests with `pytest`
- Add new pages in the `src/pages/` directory following the naming convention
