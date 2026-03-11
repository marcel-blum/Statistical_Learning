# Title: 1st Place Solution - Amex Credit Default Prediction
# Author: Marcel Blum
# Description: Predictive pipeline utilizing XGBoost, RF-based feature selection, 
#              and automated VIF-based multicollinearity filtering.

rm(list=ls())

# load necessary libraries
library(car) # to compute VAR
library(ggplot2) # for visualizations
library(gridExtra) # for visualizations
library(randomForest) # for RF
library(glmnet) # for Lasso regression
library(caret) # for cross-validation
library(MLmetrics) # for AUC calculations
library(pROC) # for AUC computation
library(xgboost) # for gradient boosting, XGBoost
library(doParallel) # for parallelization
library(foreach) # for parallelization  
library(pheatmap) # for heatmap visualization

# load train data, test data, submission file
train_data <- read.csv("~/R projects/IBE_Master/Statistical_Learning/Data/Forecast_Competition/amex_train.csv")
View(train_data)
str(train_data)
test_data <- read.csv("~/R projects/IBE_Master/Statistical_Learning/Data/Forecast_Competition/amex_validation.csv")
View(test_data)
submission_file <- read.csv("~/R projects/IBE_Master/Statistical_Learning/Data/Forecast_Competition/amex_submission_template.csv")

# provide information on variables
variables <- data.frame(
  variable = c("D_*", "S_*", "P_*", "B_*", "R_*"),
  meaning = c("Delinquency variables", "Spend variables", "Payment variables", "Balance variables", "Risk variables"),
  stringsAsFactors = FALSE
)

set.seed(42)

# make sure target is factor for classification
train_data$target <- as.factor(train_data$target)
# convert and extract date features from train date
train_data$S_2 <- as.Date(train_data$S_2)
train_data$S_2_year <- as.numeric(format(train_data$S_2, "%Y"))
train_data$S_2_month <- as.numeric(format(train_data$S_2, "%m"))
train_data$S_2_day <- as.numeric(format(train_data$S_2, "%d"))
# convert and extract date features from test data
test_data$S_2 <- as.Date(test_data$S_2)
test_data$S_2_year <- as.numeric(format(test_data$S_2, "%Y"))
test_data$S_2_month <- as.numeric(format(test_data$S_2, "%m"))
test_data$S_2_day <- as.numeric(format(test_data$S_2, "%d"))
# define variables to be excluded from predictors
exclude_vars <- c("X", "ID", "target", "S_2")
# define predictors for both train and test datasets
train_predictors <- train_data[, !(names(train_data) %in% exclude_vars)]
test_predictors <- test_data[, !(names(test_data) %in% exclude_vars)]
# specify the categorical variables
categorical_vars <- c('B_30', 'B_38', 'D_114', 'D_116', 'D_117', 
                      'D_120', 'D_126', 'D_63', 'D_64', 'D_66', 'D_68')
# convert them to factors in both train and test
train_data[categorical_vars] <- lapply(train_data[categorical_vars], as.factor)
test_data[categorical_vars] <- lapply(test_data[categorical_vars], as.factor)
str(train_data)
# redefine predictor sets after conversion
train_predictors <- train_data[, !(names(train_data) %in% exclude_vars)]
test_predictors <- test_data[, !(names(test_data) %in% exclude_vars)]
# separate categorical and numeric predictors
categorical_predictors_train <- train_predictors[, names(train_predictors) %in% categorical_vars]
categorical_predictors_test <- test_predictors[, names(test_predictors) %in% categorical_vars]
# extract numeric predictors
train_numeric_predictors <- train_predictors[, !(names(train_predictors) %in% categorical_vars)]
test_numeric_predictors <- test_predictors[, !(names(test_predictors) %in% categorical_vars)]

# visually inspect numeric data
# Function to create plots for a given variable
create_plots <- function(data, variable) {
  p1 <- ggplot(data, aes(x = .data[[variable]])) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "lightblue", color = "black") +
    geom_density(color = "red", linewidth = 1) +
    ggtitle(paste("Histogram of", variable)) +
    theme_minimal()
  
  p2 <- ggplot(data, aes(y = .data[[variable]])) +
    geom_boxplot(fill = "lightgreen") +
    ggtitle(paste("Box Plot of", variable)) +
    theme_minimal()
  
  p3 <- ggplot(data, aes(sample = .data[[variable]])) +
    stat_qq() +
    stat_qq_line(color = "red") +
    ggtitle(paste("Q-Q Plot of", variable)) +
    theme_minimal()
  
  grid.arrange(p1, p2, p3, ncol = 3)
}

# Loop through each numeric predictor and create plots
#for (var in names(numeric_predictors)) {
#  print(create_plots(numeric_predictors, var))
#}

# Function to safely log transform a vector
log_transform <- function(x) {
  min_val <- min(x)
  if (min_val <= 0) {
    # Shift all values by (|min_val| + 1) to ensure positivity
    shift <- abs(min_val) + 1
    x_transformed <- log(x + shift)
  } else {
    x_transformed <- log(x)
  }
  return(x_transformed)
}

# --- Data Sanitization & Feature Engineering ---
# Data Sanitization: Apply log transformation to numeric predictors to reduce skewness 
# and stabilize variance, mitigating the impact of outliers on model convergence.
# apply log transformation to all numeric predictors in the training dataset
train_numeric_log_predictors <- as.data.frame(lapply(train_numeric_predictors, log_transform))
test_numeric_log_predictors <- as.data.frame(lapply(test_numeric_predictors, log_transform))

# check for correlation in the train dataset
# Keep only numeric columns with non-zero variance
all_log_numeric_vars <- names(train_numeric_log_predictors)
train_numeric_log_predictors <- train_numeric_log_predictors[, apply(train_numeric_log_predictors, 2, function(x) sd(x) != 0)]
kept_log_numeric_vars <- names(train_numeric_log_predictors)
dropped_zero_var <- setdiff(all_log_numeric_vars, kept_log_numeric_vars)

# print variables dropped due to zero variance
if (length(dropped_zero_var) > 0) {
  cat("Variables dropped due to zero variance:\n")
  print(dropped_zero_var)
} else {
  cat("No variables were dropped due to zero variance.\n")
}
# Apply the same exclusion to the test dataset
test_numeric_log_predictors <- test_numeric_log_predictors[, !(names(test_numeric_log_predictors) %in% dropped_zero_var)]

# calculate correlation matrix
cor_matrix <- cor(train_numeric_log_predictors, use = "pairwise.complete.obs")
# find pairs with high correlation
high_corr_pairs <- which(abs(cor_matrix) > 0.9 & abs(cor_matrix) < 1, arr.ind = TRUE)
# extract names of variables to drop (keep only one from each pair)
high_corr_vars <- unique(rownames(high_corr_pairs)[high_corr_pairs[,1] < high_corr_pairs[,2]])
# drop the highly correlated variables  from the train dataset
train_numeric_log_predictors <- train_numeric_log_predictors[, !(names(train_numeric_log_predictors) %in% high_corr_vars)]
# print variables dropped due to high correlation
if (length(high_corr_vars) > 0) {
  cat("Variables dropped due to high correlation:\n")
  print(high_corr_vars)
} else {
  cat("No variables were dropped due to high correlation.\n")
}
# Apply the same exclusions to the test dataset
test_numeric_log_predictors <- test_numeric_log_predictors[, !(names(test_numeric_log_predictors) %in% high_corr_vars)]

# Multicollinearity filtering: VIF check used to eliminate redundant features, 
# ensuring model interpretability and reducing noise in the gradient boosting process.
# create a temporary linear model to assess multicollinearity
vif_model <- lm(as.numeric(train_data$target) ~ ., data = cbind(train_numeric_log_predictors, target = train_data$target))
# run VIF test
vif_values <- vif(vif_model)
# drop highly collinear variables from the train dataset
high_vif <- names(vif_values[vif_values > 10])
train_numeric_log_predictors <- train_numeric_log_predictors[, !(names(train_numeric_log_predictors) %in% high_vif)]
# print variables dropped due to high VIF
if (length(high_vif) > 0) {
  cat("Variables dropped due to high VIF:\n")
  print(high_vif)
} else {
  cat("No variables were dropped due to high VIF.\n")
}
# Apply the same exclusions to the test dataset
test_numeric_log_predictors <- test_numeric_log_predictors[, !(names(test_numeric_log_predictors) %in% high_vif)]

# re-integrate only the original categorical variables for the training dataset
categorical_predictors_train <- train_predictors[, sapply(train_predictors, function(x) is.factor(x) || is.character(x))]
categorical_predictors_test <- test_predictors[, sapply(test_predictors, function(x) is.factor(x) || is.character(x))]

# test on variables with zero variance
# Store original names
all_categorical_vars <- names(categorical_predictors_train)
# More robust zero-variance check for factors: convert to character before checking unique levels
non_constant_mask <- sapply(categorical_predictors_train, function(col) {
  length(unique(as.character(col))) > 1
})
# Subset datasets
categorical_predictors_train <- categorical_predictors_train[, non_constant_mask, drop = FALSE]
categorical_predictors_test <- categorical_predictors_test[, names(categorical_predictors_train), drop = FALSE]
# Report dropped variables
dropped_categorical_vars <- setdiff(all_categorical_vars, names(categorical_predictors_train))
if (length(dropped_categorical_vars) > 0) {
  cat("Dropped constant categorical variables:", paste(dropped_categorical_vars, collapse = ", "), "\n")
}

# Build dummyVars encoder on training categorical predictors without intercept
dummy_enc <- dummyVars("~ .", data = categorical_predictors_train, fullRank = TRUE)
# Create dummy variables (apply the encoder) on both train and test categorical predictors
dummies_train <- predict(dummy_enc, newdata = categorical_predictors_train)
dummies_test  <- predict(dummy_enc, newdata = categorical_predictors_test)
# Convert to matrices (optional, but useful for later model input)
dummies_train <- as.matrix(dummies_train)
dummies_test  <- as.matrix(dummies_test)
# 4. combine with numeric predictor blocks
all_predictors_train <- cbind(train_numeric_log_predictors, dummies_train)
all_predictors_test  <- cbind(test_numeric_log_predictors , dummies_test )

# check for multicollinearity again after adding the dummy variables
# function to iteratively remove high VIF variables and backtest whether dummy variable creation worked
remove_high_vif <- function(predictors, target, threshold = 10) {
  repeat {
    # fit model and identify aliased coefficients
    model <- lm(as.numeric(target) ~ ., data = cbind(predictors, target = target))
    aliased <- alias(model)$Complete
    
    if (!is.null(aliased)) {
      # drop aliased variables
      aliased_vars <- rownames(aliased)
      predictors <- predictors[, !(names(predictors) %in% aliased_vars)]
      cat("Removed aliased variables:", paste(aliased_vars, collapse = ", "), "\n")
      next  # Skip VIF this round and retry
    }
    
    # calculate VIF
    vif_values <- vif(model)
    high_vif_vars <- names(vif_values[vif_values > threshold])
    
    if (length(high_vif_vars) == 0) break  # Exit if no high VIF
    
    # remove highest VIF variable
    var_to_remove <- names(sort(vif_values, decreasing = TRUE))[1]
    predictors <- predictors[, !(names(predictors) %in% var_to_remove)]
    cat("Removed variable due to high VIF:", var_to_remove, "\n")
  }
  return(predictors)
}
# run the automated VIF removal on all predictors
all_predictors_train <- remove_high_vif(all_predictors_train, train_data$target)
all_predictors_test <- all_predictors_test[, colnames(all_predictors_train)]
# Check for multicollinearity again after adding the dummy variables to the training dataset
all_predictors_train <- remove_high_vif(all_predictors_train, train_data$target)
# Get the names of the predictors remaining in the training dataset after VIF removal
remaining_vars <- names(all_predictors_train)
# Ensure the test dataset has the same variables as the training dataset after VIF removal
all_predictors_test <- all_predictors_test[, names(all_predictors_test) %in% remaining_vars]
# Print the names of the predictors in the final training dataset
#cat("Final predictors in the training dataset:\n")
#cat(remaining_vars, sep = "\n")
#cat(names(all_predictors_train), sep = "\n")

# Feature Engineering - decide which variables should be introduced a polynomial degree
# Ensure numeric_log_predictors and target are correctly structured
train_numeric_log_predictors <- as.data.frame(train_numeric_log_predictors)
target <- as.factor(train_data$target)
# Combine the predictors and target into a single data frame for modeling
model_data <- cbind(train_numeric_log_predictors, target = target)

# Step 1: Feature Importance with Random Forest
# Fit a Random Forest model
rf_model <- randomForest(target ~ ., data = model_data, importance = TRUE)
# Extract feature importance
importance_scores <- importance(rf_model, type = 2) # type = 2 for MeanDecreaseGini
# Print importance scores for debugging
print("Feature Importance Scores:")
print(importance_scores)
# Ensure importance_scores is a named vector
importance_vector <- as.vector(importance_scores[, 1])
names(importance_vector) <- rownames(importance_scores)
# Sort the importance scores and extract the top 5 feature names
importance_vector <- as.vector(importance_scores[, 1])  # Extract Gini scores
names(importance_vector) <- rownames(importance_scores)  # Preserve variable names
# Sort and get top 5 feature names
sorted_importance <- sort(importance_vector, decreasing = TRUE)
top_features <- names(sorted_importance)[1:5]
cat("Top 5 important features:\n")
print(top_features)

# Step 2: Generate Polynomial Features and Evaluate with Cross-Validation
# Setting up for automated model selection and hyperparameter tuning
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary, savePredictions = "final")

# Define hyperparameter grid
tuned_grid <- expand.grid(alpha = c(0.1, 0.325, 0.550, 0.775, 1), lambda = c(0.0001, 0.001, 0.01))

## Function to generate polynomial features and evaluate model
evaluate_poly_features <- function(features, max_degree) {
  # Generate polynomial features
  poly_data <- train_numeric_log_predictors[, features]
  feature_names <- features
  for (degree in 2:max_degree) {
    for (feature in features) {
      new_col <- (train_numeric_log_predictors[[feature]])^degree
      poly_data <- cbind(poly_data, new_col)
      feature_names <- c(feature_names, paste0(feature, "_deg", degree))
    }
  }
  colnames(poly_data) <- feature_names
  
  cat("Polynomial features generated:\n")
  print(colnames(poly_data))
  
  # Prepare data
  x <- as.data.frame(poly_data)
  y <- factor(ifelse(train_data$target == 1, "Yes", "No"), levels = c("No", "Yes"))
  
  # Train elastic net model
  model <- train(x = x, y = y, method = "glmnet", family = "binomial", trControl = control, tuneGrid = tuned_grid)
  
  return(list(model = model, x = x, y = y))
}
# Evaluate different polynomial degrees
results <- list()
best_model <- NULL
best_auc <- 0

for (degree in 1:3) {
  result <- evaluate_poly_features(top_features, degree)
  model <- result$model
  x <- result$x
  y <- result$y
  
  # Calculate AUC using pROC
  predictions <- predict(model, newdata = x, type = "prob")
  roc_obj <- pROC::roc(response = y, predictor = predictions[, 2], levels = c("No", "Yes"))
  auc <- as.numeric(pROC::auc(roc_obj))
  
  cat("Model performance with polynomial features up to degree", degree, ":\n")
  print(model)
  cat("AUC:", auc, "\n")
  
  # Store the best model
  if (auc > best_auc) {
    best_auc <- auc
    best_model <- model
  }
}
best_model$bestTune
plot(roc_obj, main = paste("Model ROC Curve (AUC =", round(auc(roc_obj), 3), ")"), legacy.axes = TRUE)

# Rebuild full polynomial feature set (degree = 3)
generate_polynomial_features <- function(features, max_degree, data) {
  # Start with degree-1 features
  poly_data <- data[, features, drop = FALSE]
  feature_names <- features
  
  # Add higher-degree features
  for (degree in 2:max_degree) {
    for (feature in features) {
      new_col <- data[[feature]]^degree
      poly_data <- cbind(poly_data, new_col)
      feature_names <- c(feature_names, paste0(feature, "_deg", degree))
    }
  }
  colnames(poly_data) <- feature_names
  
  return(as.data.frame(poly_data))
}
# response variable
y_final <- factor(ifelse(train_data$target == 1, "Yes", "No"), levels = c("No", "Yes"))
# generate set of polynomials and only extract higher-degree columns
x_final_train <- generate_polynomial_features(top_features, 3, train_numeric_log_predictors)
x_final_train <- x_final_train[ , grep("_deg", names(x_final_train))]

x_final_test <- generate_polynomial_features(top_features, 3, test_numeric_log_predictors)
x_final_test <- x_final_test[ , grep("_deg", names(x_final_test))]
# integrate the polynomial variables into the set of predictors
all_predictors_train <- cbind(all_predictors_train, x_final_train)
all_predictors_test <- cbind(all_predictors_test , x_final_test)

#View(train_data$target)
# Remove predictors with only one unique value (i.e., constant columns)
#constant_cols <- sapply(train_predictors, function(col) length(unique(col)) <= 1)
#train_predictors <- train_predictors[, !constant_cols]
#test_predictors <- test_predictors[, !constant_cols, drop = FALSE]
# One-hot encode using caret's dummyVars to handle factor levels consistently
#dummy_encoder <- caret::dummyVars("~ .", data = train_predictors, fullRank = TRUE)

# Apply transformation to both datasets
#train_matrix <- predict(dummy_encoder, newdata = train_predictors)
#test_matrix <- predict(dummy_encoder, newdata = test_predictors)

# Ensure both datasets are matrices
train_matrix <- as.matrix(all_predictors_train)
test_matrix <- as.matrix(all_predictors_test)
# Step 4: Ensure test matrix has same columns in same order as train
test_matrix <- test_matrix[, colnames(train_matrix), drop = FALSE]
# Step 5: Prepare DMatrix for XGBoost
#xgb_train <- xgb.DMatrix(data = train_matrix, label = as.numeric(train_data$target))
xgb_test <- xgb.DMatrix(data = test_matrix)


#duplicated_names <- names(train_predictors)[duplicated(names(train_predictors))]
#print(duplicated_names)
#any(duplicated(names(train_predictors)))

# cross-validation of XGBoost model
# Prepare the dummyVars model using the cleaned train_predictors (already filtered for constant cols)
#dummy_encoder <- caret::dummyVars("~ .", data = all_predictors_train, fullRank = TRUE)
# Apply the dummyVars transformation on train and test predictors
#train_matrix <- predict(dummy_encoder, newdata = all_predictors_train)
#test_matrix <- predict(dummy_encoder, newdata = all_predictors_test)
# Convert to matrix format
train_matrix <- as.matrix(train_matrix)
test_matrix <- as.matrix(test_matrix)
# Prepare label vector for XGBoost: numeric 0/1
y_vec <- as.numeric(train_data$target) - 1

# Parameter ranges: reduce runtime where possible
max_depths <- 2:5 # controls the depth of individual trees in the ensemble
etas <- seq(0.01, 0.1, by = 0.01) # c(0.05, 0.1, 0.15) # controls the step size at each boosting iteration
nrounds_list <- c(100, 300) # determines the number of trees in the ensemble
subsamples <- 0.8 # controls the fraction of samples used for fitting each tree
colsample_bytrees <- 0.8 # controls the fraction of features used for fitting each tree
min_child_weights <- 1 #controls the minimum sum of instance weight needed in a child
gammas <- 1 # specifies the minimum loss reduction required to make a split
lambdas <- 1 # L2 regularization
alphas <- 1 # L1 regularization

# Create param grid dataframe for parallelization
param_grid <- expand.grid(
  max_depth = max_depths,
  eta = etas,
  nrounds = nrounds_list,
  subsample = subsamples,
  colsample_bytree = colsample_bytrees,
  min_child_weight = min_child_weights,
  gamma = gammas,
  lambda = lambdas,
  alpha = alphas
)

# compute the number of combinations
# cat("Running CV on", nrow(param_grid), "parameter combinations\n")

# Sample 100 random combinations to reduce computation time
#param_grid <- param_grid[sample(1:nrow(param_grid), 1), ]

# Setup parallel backend to use (number of cores - 1)
num_cores <- parallel::detectCores() - 1
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Prepare plain data matrix and label vector
X_mat <- train_matrix
# y_vec already defined above, so no need to redefine here

# Run CV in parallel over parameter grid with timer
elapsed_time <- system.time({
  results <- foreach(i = 1:nrow(param_grid), .combine = rbind, .packages = 'xgboost') %dopar% {
    
    params <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = param_grid$max_depth[i],
      eta = param_grid$eta[i],
      subsample = param_grid$subsample[i],
      colsample_bytree = param_grid$colsample_bytree[i],
      min_child_weight = param_grid$min_child_weight[i],
      gamma = param_grid$gamma[i],
      lambda = param_grid$lambda[i],
      alpha = param_grid$alpha[i]
    )
    
    dtrain_local <- xgb.DMatrix(data = X_mat, label = y_vec)
    
    cv <- xgb.cv(
      params = params,
      data = dtrain_local,
      nrounds = param_grid$nrounds[i],
      nfold = 5,
      early_stopping_rounds = 5,
      verbose = 0
    )
    
    data.frame(
      max_depth = params$max_depth,
      eta = params$eta,
      nrounds = param_grid$nrounds[i],
      subsample = params$subsample,
      colsample_bytree = params$colsample_bytree,
      min_child_weight = params$min_child_weight,
      gamma = params$gamma,
      lambda = params$lambda,
      alpha = params$alpha,
      best_iteration = cv$best_iteration,
      best_auc = max(cv$evaluation_log$test_auc_mean)
    )
  }
})

# Stop cluster to free resources
stopCluster(cl)

# Order results by best_auc descending
results <- results[order(-results$best_auc), ]

# View top-performing settings
print(results)

# Print elapsed time
cat("Cross-validation took", round(elapsed_time["elapsed"], 2), "seconds\n")

# Pick best parameters from CV results (top row)
best_params <- results[1, ]
# Prepare full training data again (same as in CV)
dtrain_full <- xgb.DMatrix(data = X_mat, label = y_vec)
# Prepare parameters list for training
final_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = best_params$max_depth,
  eta = best_params$eta,
  subsample = best_params$subsample,
  colsample_bytree = best_params$colsample_bytree,
  min_child_weight = best_params$min_child_weight,
  gamma = best_params$gamma,
  lambda = best_params$lambda,
  alpha = best_params$alpha
)
# Train final model on all data, using best iteration from CV (or use early stopping)
final_model <- xgb.train(
  params = final_params,
  data = dtrain_full,
  nrounds = best_params$best_iteration,
  watchlist = list(train = dtrain_full),
  verbose = 1
)

test_pred_probs <- predict(final_model, newdata = xgb_test)
test_results <- data.frame(
  ID         = test_data$ID,
  PD  = test_pred_probs
)

# --- Visualizations & Performance Reporting ---

# 1. Feature Importance Plot (XGBoost)
# Architectural Insight: Visualizing top features validates the Random Forest 
# selection and confirms that the model is relying on economically meaningful variables.
importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = final_model)
xgb.plot.importance(importance_matrix, top_n = 15, main = "Top 15 Predictive Features (XGBoost)")

# 2. ROC-AUC Curve
# Validation: ROC-AUC is the competition metric. This curve visualizes the 
# model's ability to discriminate between default and non-default cases.
final_probs <- predict(final_model, newdata = xgb.DMatrix(train_matrix))
roc_obj <- roc(y_vec, final_probs)
plot(roc_obj, main = paste("Model ROC Curve (AUC =", round(auc(roc_obj), 3), ")"))

# 3. Confusion Matrix
# Diagnostic: Shows the ratio of True Positives vs False Negatives, 
# critical for evaluating credit risk classification performance.
conf_matrix <- table(Actual = y_vec, Predicted = ifelse(final_probs > 0.5, 1, 0))
print(conf_matrix)
# Heatmap visualization of Confusion Matrix
pheatmap(as.matrix(conf_matrix), display_numbers = TRUE, number_format = "%.0f", main = "Confusion Matrix Heatmap", cluster_rows = FALSE, cluster_cols = FALSE)

# Load the existing submission template
submission_path <- "~/R projects/IBE_Master/Statistical_Learning/Data/Forecast_Competition/amex_submission_template.csv"
submission <- read.csv(submission_path, stringsAsFactors = FALSE)

# Overwrite values in column C (i.e., 3rd column), starting from row 2
submission[, 3] <- test_results$PD

# Save the updated file (overwriting the original)
write.csv(submission, submission_path, row.names = FALSE)
