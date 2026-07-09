# Credit Risk Modeling: PD and LGD Estimation

Statistical and machine learning approaches to Probability of Default (PD) and Loss Given Default (LGD) estimation, consistent with the Basel II/III framework for Expected Credit Loss (Expected Loss = PD × LGD × EAD).

## Overview

This project implements a full credit risk modeling pipeline on loan level data:

- **PD estimation**: logistic regression compared against gradient boosted trees, plus a Markov chain framework for modeling delinquency state transitions over time.
- **LGD estimation**: regression based recovery rate modeling on defaulted loans.
- **Interpretability**: SHAP values.
- **Deliverable**: a reproducible pipeline and an interactive FastAPI and React dashboard for portfolio level risk visualization.

## Setup

**Requirements:** Python 3.11+

```bash
# Clone the repo
git clone https://github.com/yelyzaveta-boiko/credit-risk-modelling.git
cd credit-risk-modeling

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install core dependencies
pip install -e .

# Install with dashboard dependencies
pip install -e ".[dashboard]"

# Install with dev dependencies
pip install -e ".[dev]"

# Copy environment template and fill in values
cp .env.example .env
```

## Dataset

## Development workflow

This project follows a standard feature-branch workflow:

- `main`
- `dev` - integration branch
- `feature/*` - individual feature branches, merged via PR into `dev`

## Running tests
