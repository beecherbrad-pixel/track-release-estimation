# Track Release Timing Estimation
**Structural Econometrics & Reinforcement Learning Pipeline**

This repository implements a two-stage econometric and machine learning pipeline to determine the optimal timing for music track releases. It uses **JAX** for structural demand estimation and **PyTorch** for solving the resulting optimal stopping problem via Deep Q-Learning.

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/track-release-estimation.git](https://github.com/yourusername/track-release-estimation.git)
cd track-release-estimation
```

### 2. Environment Configuration
It is highly recommended to use a virtual environment:
```bash
python -m venv .venv

source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
### 3. Install Dependencies
You can install these by uncommenting the first line of python code in the notebook
```bash
pip install jax jaxlib optax torch pandas numpy statsmodels matplotlib kaggle python-dotenv
```

### 4. Data Access (Kaggle API)
The project automates data acquisition via the Kaggle API. Create a .env file in the root directory and add your credentials:

```Code snippet
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

However, I have also provided a data.zip file for easy access to the data, so there is no need to go through the trouble of obtaining a key.

## Overview
The main analysis is 
* Data Preparation (build_estimation_dataset.py): Cleans raw Spotify charts and interpolates daily streaming flows.
* State Dynamics (estimate_state_process.py): Estimates a stationary VAR(1) process for market and genre heat.
* Structural Estimation (structural_estimation.py): Estimates the artist potential ($\eta$) and decay ($\lambda$) using JAX.
* Policy Training (dqn_trainer.py): Trains a DQN agent to recognize optimal market entry points.
* Evaluation (dqn_evaluation.py): Generates policy maps and calculates the economic "lift" of the timed release.

⚖️ Determinism & Reproducibility
To ensure identical results across executions, the project uses a centralized seeding mechanism. The set_seed(42) function synchronizes RNGs for:
* Python random & os
* NumPy
* PyTorch (including CuDNN determinism)
* JAX PRNG keys

Created as part of ECON 622: Graduate Econometrics at the University of British Columbia.