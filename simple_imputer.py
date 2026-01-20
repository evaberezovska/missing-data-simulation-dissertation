
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import datetime

def load_datasets(iteration):
    # define paths
    complete_dir = f"data/complete/iteration_{iteration}"
    amputed_dir = f"data/amputed/iteration_{iteration}"
    
    # load complete datasets
    train_complete = pd.read_csv(f"{complete_dir}/train_complete.csv")
    test_complete = pd.read_csv(f"{complete_dir}/test_complete.csv")
    
    # load datasets with missing data
    train_low_missing = pd.read_csv(f"{amputed_dir}/train_low.csv")
    test_low_missing = pd.read_csv(f"{amputed_dir}/test_low.csv")
    train_high_missing = pd.read_csv(f"{amputed_dir}/train_high.csv")
    test_high_missing = pd.read_csv(f"{amputed_dir}/test_high.csv")
    
    return train_complete, test_complete, train_low_missing, test_low_missing, train_high_missing, test_high_missing

def process_iteration(iteration):
    print(f"\n--- Processing iteration {iteration} ---")

    # create directories
    imputed_dir = f"data/imputed/iteration_{iteration}"
    os.makedirs(imputed_dir, exist_ok=True)
    
    # load all datasets
    train_complete, test_complete, train_low_missing, test_low_missing, train_high_missing, test_high_missing = load_datasets(iteration)
    
    # identify features and column types
    features = [col for col in train_low_missing.columns if col != 'target']
    categorical_cols = [col for col in features if col.startswith('cat_')]
    numerical_cols = [col for col in features if col not in categorical_cols]
    
    # ------- LOW MISSING RATE DATA -------
    # create imputers for low missing data
    numerical_imputer_low = SimpleImputer(strategy='mean')
    categorical_imputer_low = SimpleImputer(strategy='most_frequent')
    
    # train  - low missing
    X_train_low = train_low_missing[features].copy()
    train_low_mask = X_train_low.isnull()
    
    # apply on train data
    X_train_low[numerical_cols] = numerical_imputer_low.fit_transform(X_train_low[numerical_cols])
    X_train_low[categorical_cols] = categorical_imputer_low.fit_transform(X_train_low[categorical_cols])
    
    # evaluate train performance
    train_low_mask_np = train_low_mask.to_numpy()
    y_true_low = train_complete[features].to_numpy()[train_low_mask_np]
    y_pred_low = X_train_low.to_numpy()[train_low_mask_np]
    
    train_low_rmse = root_mean_squared_error(y_true_low, y_pred_low)
    train_low_mae = mean_absolute_error(y_true_low, y_pred_low)
    
    # test imputation - low missing
    X_test_low = test_low_missing[features].copy()
    test_low_mask = X_test_low.isnull()
    
    # apply on test data
    X_test_low[numerical_cols] = numerical_imputer_low.transform(X_test_low[numerical_cols])
    X_test_low[categorical_cols] = categorical_imputer_low.transform(X_test_low[categorical_cols])
    
    # evaluate test performance
    test_low_mask_np = test_low_mask.to_numpy()
    y_true_low_test = test_complete[features].to_numpy()[test_low_mask_np]
    y_pred_low_test = X_test_low.to_numpy()[test_low_mask_np]
    
    test_low_rmse = root_mean_squared_error(y_true_low_test, y_pred_low_test)
    test_low_mae = mean_absolute_error(y_true_low_test, y_pred_low_test)
    
    print(f"Low missing - Train RMSE: {train_low_rmse:.4f}, MAE: {train_low_mae:.4f}")
    print(f"Low missing - Test RMSE: {test_low_rmse:.4f}, MAE: {test_low_mae:.4f}")
    
    # ------- HIGH MISSING RATE DATA -------
    # create imputers for high missing data
    numerical_imputer_high = SimpleImputer(strategy='mean')
    categorical_imputer_high = SimpleImputer(strategy='most_frequent')
    
    # train - high missing
    X_train_high = train_high_missing[features].copy()
    train_high_mask = X_train_high.isnull()
    
    # apply train data
    X_train_high[numerical_cols] = numerical_imputer_high.fit_transform(X_train_high[numerical_cols])
    X_train_high[categorical_cols] = categorical_imputer_high.fit_transform(X_train_high[categorical_cols])
    
    # evaluate train performance
    train_high_mask_np = train_high_mask.to_numpy()
    y_true_high = train_complete[features].to_numpy()[train_high_mask_np]
    y_pred_high = X_train_high.to_numpy()[train_high_mask_np]
    
    train_high_rmse = root_mean_squared_error(y_true_high, y_pred_high)
    train_high_mae = mean_absolute_error(y_true_high, y_pred_high)
    
    # test - high missing
    X_test_high = test_high_missing[features].copy()
    test_high_mask = X_test_high.isnull()
    
    # apply test data
    X_test_high[numerical_cols] = numerical_imputer_high.transform(X_test_high[numerical_cols])
    X_test_high[categorical_cols] = categorical_imputer_high.transform(X_test_high[categorical_cols])
    
    # evaluate test performance
    test_high_mask_np = test_high_mask.to_numpy()
    y_true_high_test = test_complete[features].to_numpy()[test_high_mask_np]
    y_pred_high_test = X_test_high.to_numpy()[test_high_mask_np]
    
    test_high_rmse = root_mean_squared_error(y_true_high_test, y_pred_high_test)
    test_high_mae = mean_absolute_error(y_true_high_test, y_pred_high_test)
    
    print(f"High missing - Train RMSE: {train_high_rmse:.4f}, MAE: {train_high_mae:.4f}")
    print(f"High missing - Test RMSE: {test_high_rmse:.4f}, MAE: {test_high_mae:.4f}")
    
    # ------- SAVE RESULTS -------
    # prepare and save imputed datasets with target
    train_low_simple = X_train_low.copy()
    train_low_simple['target'] = train_low_missing['target'].values
    
    test_low_simple = X_test_low.copy()
    test_low_simple['target'] = test_low_missing['target'].values
    
    train_high_simple = X_train_high.copy()
    train_high_simple['target'] = train_high_missing['target'].values
    
    test_high_simple = X_test_high.copy()
    test_high_simple['target'] = test_high_missing['target'].values
    
    # save imputed datasets 
    train_low_simple.to_csv(f"{imputed_dir}/train_low_simple.csv", index=False)
    test_low_simple.to_csv(f"{imputed_dir}/test_low_simple.csv", index=False)
    train_high_simple.to_csv(f"{imputed_dir}/train_high_simple.csv", index=False)
    test_high_simple.to_csv(f"{imputed_dir}/test_high_simple.csv", index=False)
    
    print(f"Saved imputed datasets for iteration {iteration}")
     
    results = [
        {
            'iteration': iteration,
            'method': 'simple',
            'missing_level': 'low',
            'data_set': 'train',
            'rmse': train_low_rmse,
            'mae': train_low_mae
        },
        {
            'iteration': iteration,
            'method': 'simple',
            'missing_level': 'low',
            'data_set': 'test',
            'rmse': test_low_rmse,
            'mae': test_low_mae
        },
        {
            'iteration': iteration,
            'method': 'simple',
            'missing_level': 'high',
            'data_set': 'train',
            'rmse': train_high_rmse,
            'mae': train_high_mae
        },
        {
            'iteration': iteration,
            'method': 'simple',
            'missing_level': 'high',
            'data_set': 'test',
            'rmse': test_high_rmse,
            'mae': test_high_mae
        }
    ]
    
    results_df = pd.DataFrame(results)
    
    # save results to summary file 
    summary_file = "results/imputation_summary.csv"
    if os.path.exists(summary_file):
        existing_results = pd.read_csv(summary_file)
        # remove any existing results 
        existing_results = existing_results[
            ~((existing_results['iteration'] == iteration) & 
              (existing_results['method'] == 'simple'))
        ]
        combined_results = pd.concat([existing_results, results_df], ignore_index=True)
        combined_results.to_csv(summary_file, index=False)
    else:
        results_df.to_csv(summary_file, index=False)
    
    print(f"Completed iteration {iteration}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply simple imputation methods")
    parser.add_argument("--iteration", type=int, required=True,
                        help="Iteration number to process")
    
    args = parser.parse_args()
    
    try:
        start_time = datetime.datetime.now()
        
        print(f"Running simple imputation for iteration {args.iteration}")
        process_iteration(args.iteration)
        
        end_time = datetime.datetime.now()
        elapsed = (end_time - start_time).total_seconds() / 60.0
        
        print(f"\nExecution time: {elapsed:.2f} minutes")
        print(f"Simple imputation complete for iteration {args.iteration}!")
        
    except Exception as e:
        print(f"Error: {e}")