# NASA CMAPSS Engine Degradation Prediction

This project uses machine learning to predict engine degradation based on the NASA CMAPSS dataset.

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Tasks](#tasks)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Development and Evaluation](#3-model-development-and-evaluation)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the code](#running-the-code)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Dataset Overview

Dataset Source: [NASA CMAPSS Dataset on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasacmaps)

Description: The CMAPSS dataset contains engine degradation simulation data, designed for predictive maintenance modeling. It provides sensor data and operational conditions for turbofan engines.

## Tasks

### 1. Exploratory Data Analysis (EDA)

Objective: Identify meaningful patterns and specific insights from the dataset.

*   Understand the distribution of sensor readings and operational conditions.
*   Analyze the trend of degradation patterns over time.
*   Investigate correlations between sensor data and engine degradation.
*   Detect anomalies or outliers in the dataset.

**Deliverables:**

*   EDA Insights:
    *   Visualizations (e.g., trend charts, correlation matrices, distributions).  Include these in a folder like `reports/figures`.
    *   Key findings (e.g., which sensors are most indicative of engine degradation). Document these in a file like `reports/eda_summary.txt` or within the notebook itself.

### 2. Feature Engineering

Objective: Identify and create features that enhance model performance.

*   Evaluate the importance of existing features (e.g., sensor data, operational conditions).
*   Generate additional features:
    *   Cumulative degradation indicators.
    *   Rolling averages or window-based aggregations for sensor values.
    *   Derived metrics from sensor data and operational conditions.
*   Handle missing data and scale/normalize features appropriately.

**Deliverables:**

*   Engineered Features:
    *   List of selected features. Store this in a file like `data/feature_list.txt` or as a comment in your code.
    *   Description of newly created features and their significance.  Document this in the code or in a separate file.

### 3. Model Development and Evaluation

Objective: Train and evaluate predictive maintenance models.

**Model Selection:**

*   Linear Regression (for baseline).
*   Gradient Boosting (e.g., XGBoost, LightGBM).
*   Neural Networks for complex patterns.

**Evaluation Metrics:**

*   Root Mean Squared Error (RMSE).
*   Mean Absolute Error (MAE).
*   R-squared (for regression models).
*   Precision, Recall, F1-score (for classification models, if applicable).

**Deliverables:**

*   Model Results:
    *   Model training and testing performance (e.g., RMSE, MAE, R-squared).  Include these in a file like `reports/model_performance.txt`.
    *   Comparison of different models.  A table is good for this. Put it in the `model_performance.txt` file or a notebook.
    *   Key metrics for the best-performing model.

## Getting Started

### Prerequisites

List all the required libraries and software:

*   Python 3.x
*   scikit-learn
*   pandas
*   numpy
*   matplotlib
*   seaborn
*   XGBoost (if used)
*   LightGBM (if used)
*   TensorFlow or PyTorch (if using neural networks)
*   Jupyter Notebook (recommended)

### Installation

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the dependencies
pip install -r requirements.txt
