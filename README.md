# Missing Data Imputation Simulation Study
A simulation study evaluating how different imputation methods (MICE, KNN, Simple Imputer) impact classification model performance under various missing data scenarios.

## Running the Simulation
To run 2 iterations (for testing):
python run_simulations.py --iterations 2

For full study (100 iterations):
python run_simulations.py --iterations 100

## Project Structure

Main Simulation Files:
run_simulations.py - Main file that iterates through all methods to run n number of simulations
setup_directories.py - Creates necessary directories for data and results
data_generation.py - Generates synthetic datasets
apply_missing.py - Induces missingness (MAR/MCAR mechanisms)
run_mice.R - MICE imputation (m=30, maxit=20)
run_knn.R - KNN imputation (k=25)
simple_imputer.py - Simple imputation (mean/mode)
run_classification.py - Trains classifiers with hyperparameter tuning and extracts feature importance

Diagnostic Files
Parameter selection and validation notebooks:
data_generation.ipynb - Detailed walkthrough of data generation process
knn_diagnostics.Rmd - K parameter selection with elbow plots
mice_diagnostics.Rmd - m and maxit parameter selection with convergence diagnostics
classification_diagnostics.ipynb - Comprehensive grid search for optimal classifier parameters

Note on Grid Search: Initial extensive parameter search identified optimal ranges (e.g., SVM C values of 1-3), which were then used in the main simulation to reduce computational time while maintaining model quality.

## Outputs:
Data Files (generated at runtime)
complete - Original datasets without missingness
amputed - Datasets with induced missingness
imputed - Imputed datasets from each method

Results Files
imputation_summary.csv - RMSE and MAE for all methods across iterations
results/classification_results.csv - Accuracy and F1-scores for all classifier-imputation combinations
feature_importance.csv - Feature importance rankings for interpretability analysis

Methodology
Data: 1000 samples, 25 features (continuous, discrete, categorical), binary classification target

## Missing Data:
Mechanisms: MAR (Missing At Random), MCAR (Missing Completely At Random)
Rates: 10% (low), 25% (high)

## Imputation Methods:
MICE (Multivariate Imputation by Chained Equations)
KNN (K-Nearest Neighbors)
Simple Imputer (mean/mode baseline)

## Classifiers:
Support Vector Machine (SVM)
Random Forest
Neural Network 

## Evaluation Metrics:
Imputation quality: RMSE, MAE
Classification performance: Accuracy, F1-Score
Interpretability: Feature importance preservation

## Key Findings
Imputation Quality:
Simple Imputer achieved best imputation quality with lowest test RMSE (0.90) across both missing rates
KNN followed closely (RMSE 0.92), while MICE showed highest errors (RMSE 0.94)
MICE demonstrated greatest sensitivity to missing rates (6.89% RMSE increase from 10% to 25%)
Categorical variables showed better performance under MAR compared to MCAR across all methods

Classification Performance:
MICE-imputed data outperformed other methods for downstream classification despite higher imputation errors
MICE with SVM achieved 93.4% accuracy at low missing rates, surpassing complete data (92.7%)
Simple Imputer consistently outperformed KNN in both missing scenarios
Random Forest showed evidence of overfitting, with significant train-test performance gaps

Feature Importance Preservation:
SVM best preserved feature importance (80-82% accuracy at low missing rates)
MICE-imputed datasets retained feature importance most accurately with SVM (81.92%)
Performance declined substantially at higher missing rates across all combinations
Random Forest struggled most with feature importance preservation (54-58% accuracy)
