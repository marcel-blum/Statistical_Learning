# Technical Implementations & Logic
This branch contains the end-to-end R pipeline. The code is designed for high-dimensional financial data, emphasizing feature stability and computational efficiency.

# Core Pipeline Stages
**1. Data Sanitization & Collinearity Filtering**
To manage 80+ variables, the script implements an automated "pruning" process:
- **Log-Transformations:** Applied to numeric predictors to stabilize variance and reduce outlier influence.
- **Iterative VIF Removal:** A custom `remove_high_vif()` function filters features with a Variance Inflation Factor > 10 to ensure model stability.
- **Alias Detection:** Automatically drops perfectly collinear variables that would otherwise cause rank-deficiency.

**2. Nonlinear Feature Engineering**
Instead of a simple linear approach, the solution captures complex relationships:
- **RF Gini Importance:** Random Forest identifies the top 5 global predictors.
- **Polynomial Expansion:** These features are expanded into 3rd-degree polynomials.
- **Elastic Net Validation:** Polynomial interactions are validated via `glmnet` (Elastic Net) with 5-fold cross-validation to prevent overfitting through regularization.

**3. Parallelized XGBoost Optimization**
The final classifier utilizes a high-performance gradient boosting framework:
- **Parallel Grid Search:** Uses `doParallel` to utilize all available CPU cores, significantly accelerating hyperparameter tuning.
- **Early Stopping:** `xgb.cv` uses a 5-round early stopping threshold to find the optimal boosting iteration.
- **Winning Result:** Achieved a final out-of-sample ROC AUC of 0.953.

## Technical Environment
- **R Version:** 4.4.1
- **Full Reproducibility:** See the [session_info.txt](https://github.com/marcel-blum/Statistical_Learning/blob/main/session_info.txt) for exact package versions used.
