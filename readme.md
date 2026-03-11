# Statistical_Learning_Probability_of_Default_Prediction
This repository contains the end-to-end predictive pipeline developed for the 2025 Statistical Learning forecasting competition. The project objective was to predict the probability of credit default using the American Express Default Prediction dataset from Kaggle (see link below).

The solution achieved the highest out-of-sample ROC AUC (0.953) by combining automated feature selection, non-linear feature engineering, and a highly optimized gradient boosting architecture.

# Model Architecture
The pipeline follows a modular four-stage process designed for high-dimensional financial data:
- **Data Processing:** Automated multicollinearity filtering using Variance Inflation Factor (VIF) and Alias detection to ensure a lean, interpretable feature set.
- **Feature Selection:** Dimensionality reduction via Random Forest Gini importance to isolate high-impact predictors.
- **Feature Engineering:** Implementation of 3rd-degree polynomial expansions, regularized and validated through an Elastic Net framework.
- **Final Classifier:** An XGBoost model optimized through parallelized grid search and 5-fold cross-validation.

# Repository Structure
- `main`: Repository overview
- `model-functions`: Full R script implementation, including the parallelized training suite and VIF removal logic.
- `visualizations`: Comprehensive diagnostic suite including a Feature Importance plot and a Confusion Matrix heatmap.

# Data Access
The underlying data is proprietary to the American Express - Default Prediction competition hosted on Kaggle.
- **Link to data:** https://www.kaggle.com/competitions/amex-default-prediction/overview
- **Setup:** To reproduce the results, download the training and validation CSVs and place them in a /data directory at the project root.

# Final Notes
Before reproducing the results, it is recommended to call the `session_info.txt` file to ensure the project's prerequisites are met.
