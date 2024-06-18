# Air Quality Analysis Using Linear Classification

This Jupyter notebook is dedicated to analyzing air quality measurements and predicting carbon monoxide levels using a binary linear classifier with gradient descent optimization.

## Dataset and Knowledge Preparation

- **Data Description**: The dataset comprises 3304 hourly averaged measurements from a multi-sensor device. This includes readings from spectrometer analyzers (variables marked by "GT") and solid state metal oxide detectors (variables marked by "PT08.Sx"), along with temperature (T), relative humidity (RH), and absolute humidity (AH) sensors.
- **Objective**: Predict the binary state of CO(GT) levels—whether they indicate good (<=4.5) or bad (>4.5) air quality.
- **Missing Data**: Features with missing values are flagged with `-999`. Imputation is required for handling these entries.

## Prerequisites

```bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import sklearn.model_selection
```

## Usage

### Setup
- Ensure all prerequisites are installed.
- Download the notebook and the dataset (`sensor_data.xlsx`) into the same directory.

### Execution
- Open the notebook in your Jupyter environment.
- Execute the notebook cells sequentially to perform data preprocessing, model training, and evaluation.

## Features

### Data Preprocessing:
- Handling missing data using imputation.
- Feature scaling to standardize the dataset.

### Model Training:
- Implement a binary linear classifier.
- Use gradient descent to minimize the hinge loss with L2 regularization.

### Model Evaluation:
- Evaluate model performance using accuracy and F1 score metrics.
- Visualize training and testing performance metrics over iterations.

## Experimentation

- **Linear Classification via Gradient Descent**: Detailed steps are provided to train the model using custom functions that implement gradient descent without the use of scikit-learn's built-in functions.
- **Learning Rate Analysis**: Explore the impact of the learning rate on model training and performance, demonstrating the effects through visual plots.

## Contributing

We welcome contributions to improve the analysis and predictions. Please fork the repository, make your changes, and submit a pull request.

## Conclusion

This project illustrates the practical application of linear models in environmental science, specifically for air quality assessment using machine learning techniques.
