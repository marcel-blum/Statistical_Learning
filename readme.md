## Diagnostic Visualizations
This branch houses the diagnostic suite used to validate the model's predictive power and interpretability. These outputs demonstrate the 1st-place performance and the economic logic behind the feature selection.

## Model Performance
**1. ROC-AUC Curve (Final Score: 0.953)** [02_SL_roc_auc_curve.png](https://github.com/marcel-blum/Statistical_Learning/blob/visualizations/02_SL_roc_auc_curve.png) <br>
The model demonstrates exceptional discrimination power between default and non-default cases. The steepness of the curve indicates a high true positive rate with a minimal false positive trade-off.

**2. Confusion Matrix Heatmap** [03_SL_confusion_matrix.png](https://github.com/marcel-blum/Statistical_Learning/blob/visualizations/03_SL_confusion_matrix.png) <br>
This diagnostic view confirms high precision in identifying non-defaults while capturing the majority of actual defaults, essential for credit risk appetite.

## Model Interpretability
**3. Top 15 Predictive Features (XGBoost)** [01_SL_feature_importance.png](https://github.com/marcel-blum/Statistical_Learning/blob/visualizations/01_SL_feature_importance.png) <br>
Extracted using the XGBoost importance matrix, this plot validates that the model relies on economically meaningful drivers:
- **Delinquency (`D_75`):**`The strongest predictor, indicating past payment behavior.
- **Balance (`B_9`):** A critical measure of utilization and credit stress.
- **Feature Interactions:** Inclusion of polynomial terms (e.g., `B_9_deg3`) validates the non-linear feature engineering approach.
