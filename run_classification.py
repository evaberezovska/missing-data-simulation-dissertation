
import os
import argparse
import pandas as pd
import numpy as np
import time
import datetime
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance

os.makedirs("results", exist_ok=True)

def load_datasets(iteration, imputation_method):
    """Load datasets for a specific iteration and imputation method"""
    complete_dir = f"data/complete/iteration_{iteration}"
    imputed_dir = f"data/imputed/iteration_{iteration}"
    
    file_suffix = f"_{imputation_method}.csv"
    
    # verify paths before loading
    train_complete_path = f"{complete_dir}/train_complete.csv"
    test_complete_path = f"{complete_dir}/test_complete.csv"
    train_low_path = f"{imputed_dir}/train_low{file_suffix}"
    test_low_path = f"{imputed_dir}/test_low{file_suffix}"
    train_high_path = f"{imputed_dir}/train_high{file_suffix}"
    test_high_path = f"{imputed_dir}/test_high{file_suffix}"
    
    for path in [train_complete_path, test_complete_path, train_low_path, 
                test_low_path, train_high_path, test_high_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # load complete datasets 
    train_complete = pd.read_csv(train_complete_path)
    test_complete = pd.read_csv(test_complete_path)
    
    # load imputed datasets
    train_low = pd.read_csv(train_low_path)
    test_low = pd.read_csv(test_low_path)
    train_high = pd.read_csv(train_high_path)
    test_high = pd.read_csv(test_high_path)
    
    # check dataset shapes 
    print(f"  Dataset shapes for {imputation_method} imputation:")
    print(f"    Train complete: {train_complete.shape}")
    print(f"    Test complete: {test_complete.shape}")
    print(f"    Train low: {train_low.shape}")
    print(f"    Test low: {test_low.shape}")
    print(f"    Train high: {train_high.shape}")
    print(f"    Test high: {test_high.shape}")
    
    return {
        "train_complete": train_complete,
        "test_complete": test_complete,
        "train_low": train_low,
        "test_low": test_low,
        "train_high": train_high,
        "test_high": test_high
    }

def prepare_data(dataset, target_column="target"):
    """Extract features and target variable"""
    features = [col for col in dataset.columns if col != target_column]
    X = dataset[features]
    y = dataset[target_column]
    return X, y, features

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Calculate accuracy and F1 score for train and test sets"""
    # training metrics
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    # test metrics
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    return {
        "train_accuracy": train_accuracy,
        "train_f1": train_f1,
        "test_accuracy": test_accuracy,
        "test_f1": test_f1
    }

def train_svm(X_train, y_train, dataset_type, random_seed=57):
    """Train SVM classifier with grid search parameters"""
    print(f"  Training SVM on {dataset_type} data...")
    
    # set grid search parameters based on the dataset type
    if dataset_type in ["complete", "none"]:
        param_grid = {
            'C': [10, 20],
            'kernel': ['linear'],
            'gamma': ['scale']
        }
    elif dataset_type == "low":
        param_grid = {
            'C': [10, 15, 20],
            'kernel': ['linear'],
            'gamma': ['scale']
        }
    else:  # high
        param_grid = {
            'C': [1, 2, 3],
            'kernel': ['linear'],
            'gamma': ['scale']
        }
    
    svm = GridSearchCV(
        SVC(probability=True, random_state=random_seed),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    svm.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"    SVM training completed in {end_time - start_time:.2f} seconds")
    print(f"    Best parameters: {svm.best_params_}")
    return svm.best_estimator_

def train_random_forest(X_train, y_train, dataset_type, random_seed=57):
    """Train Random Forest classifier with grid search parameters"""
    print(f"  Training Random Forest on {dataset_type} data...")
    
    # set grid search parameters based on the dataset type
    if dataset_type in ["complete", "none"]:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [2, 3, 4],
            'min_samples_split': [50, 100],
            'min_samples_leaf': [20, 30],
            'max_features': ['sqrt'],
            'bootstrap': [True],                         
            'oob_score': [True]
        }
    elif dataset_type == "low":
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 4, 5],
            'min_samples_split': [50, 100],
            'min_samples_leaf': [20, 30],
            'max_features': ['sqrt'],
            'bootstrap': [True],                         
            'oob_score': [True]
        }
    else:  # high
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [1, 2, 5],
            'min_samples_split': [100, 250],
            'min_samples_leaf': [40, 60],
            'max_features': ['log2'],
            'bootstrap': [True],                         
            'oob_score': [True],
            'class_weight': ['balanced']
        }
    
    rf = GridSearchCV(
        RandomForestClassifier(random_state=random_seed),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"    Random Forest training completed in {end_time - start_time:.2f} seconds")
    print(f"    Best parameters: {rf.best_params_}")
    return rf.best_estimator_

def train_neural_network(X_train, y_train, dataset_type, random_seed=57):
    """Train Neural Network classifier with grid search parameters"""
    print(f"  Training Neural Network on {dataset_type} data...")
    
    # set grid search parameters based on the dataset type
    if dataset_type in ["complete", "none", "low"]:
        param_grid = {
            'hidden_layer_sizes': [(50, 50), (100, 50)],  
            'activation': ['tanh', 'logistic'],                  
            'alpha': [0.0001, 0.001, 0.01],                             
            'learning_rate_init': [0.001, 0.01],                   
            'max_iter': [2000, 3000]                                     
        }
    else:  # high
        param_grid = {
            'hidden_layer_sizes': [(10,), (50,), (100,)],  
            'activation': ['logistic'],                  
            'alpha': [0.0001, 0.01, 2.0],                             
            'learning_rate_init': [0.001, 0.01],                   
            'max_iter': [2000, 3000]                                     
        }
    
    nn = GridSearchCV(
        MLPClassifier(random_state=random_seed),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    nn.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"    Neural Network training completed in {end_time - start_time:.2f} seconds")
    print(f"    Best parameters: {nn.best_params_}")
    return nn.best_estimator_

# feature importance
def get_feature_importance(model, model_type, features, X_test=None, y_test=None, random_seed=57):
    """Extract feature importance from different model types"""
    if model_type == "svm" and hasattr(model, "coef_"):
        # SVM
        feature_coefficients = model.coef_[0]
        feature_importance = np.abs(feature_coefficients)
        
    elif model_type == "random_forest" and hasattr(model, "feature_importances_"):
        # Random Forest
        feature_importance = model.feature_importances_
        
    elif model_type == "neural_network" and X_test is not None and y_test is not None:
        # Neural Network - use permutation importance
        perm_importance = permutation_importance(
            model, X_test, y_test, 
            n_repeats=5, random_state=random_seed, scoring='accuracy'
        )
        feature_importance = perm_importance.importances_mean
        
    else:
        feature_importance = np.zeros(len(features))
    
    # create a dictionary of feature names 
    importance_dict = dict(zip(features, feature_importance))
    return importance_dict


def process_dataset(X_train, y_train, X_test, y_test, imputation, missing_level, iteration, random_seed=57):
    """Process a single dataset with all classifiers"""
    results = []
    feature_importance_results = []

    # Get feature names
    features = X_train.columns.tolist()

    # SVM
    svm_model = train_svm(X_train, y_train, missing_level, random_seed)
    svm_metrics = evaluate_model(svm_model, X_train, y_train, X_test, y_test)
    
    # calculate feature importance
    svm_importance = get_feature_importance(svm_model, "svm", features, X_test, y_test)
    
    # add feature importance results
    for feature, importance in svm_importance.items():
        feature_importance_results.append({
            'iteration': iteration,
            'imputation': imputation,
            'missing_level': missing_level,
            'classifier': 'svm',
            'feature': feature,
            'importance': importance
        })

    results.append({
        'iteration': iteration,
        'imputation': imputation,
        'missing_level': missing_level,
        'classifier': 'svm',
        'train_accuracy': svm_metrics["train_accuracy"],
        'train_f1': svm_metrics["train_f1"],
        'test_accuracy': svm_metrics["test_accuracy"],
        'test_f1': svm_metrics["test_f1"]
    })
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train, missing_level, random_seed)
    rf_metrics = evaluate_model(rf_model, X_train, y_train, X_test, y_test)
    
    # calculate feature importance
    rf_importance = get_feature_importance(rf_model, "random_forest", features)
    
    # add feature importance results
    for feature, importance in rf_importance.items():
        feature_importance_results.append({
            'iteration': iteration,
            'imputation': imputation,
            'missing_level': missing_level,
            'classifier': 'random_forest',
            'feature': feature,
            'importance': importance
        })

    results.append({
        'iteration': iteration,
        'imputation': imputation,
        'missing_level': missing_level,
        'classifier': 'random_forest',
        'train_accuracy': rf_metrics["train_accuracy"],
        'train_f1': rf_metrics["train_f1"],
        'test_accuracy': rf_metrics["test_accuracy"],
        'test_f1': rf_metrics["test_f1"]
    })
    
    # Neural Network
    nn_model = train_neural_network(X_train, y_train, missing_level, random_seed)
    nn_metrics = evaluate_model(nn_model, X_train, y_train, X_test, y_test)
    
    # calculate feature importance
    nn_importance = get_feature_importance(nn_model, "neural_network", features, X_test, y_test, random_seed)
    
    # add feature importance results
    for feature, importance in nn_importance.items():
        feature_importance_results.append({
            'iteration': iteration,
            'imputation': imputation,
            'missing_level': missing_level,
            'classifier': 'neural_network',
            'feature': feature,
            'importance': importance
        })

    results.append({
        'iteration': iteration,
        'imputation': imputation,
        'missing_level': missing_level,
        'classifier': 'neural_network',
        'train_accuracy': nn_metrics["train_accuracy"],
        'train_f1': nn_metrics["train_f1"],
        'test_accuracy': nn_metrics["test_accuracy"],
        'test_f1': nn_metrics["test_f1"]
    })
    
    return results, feature_importance_results

# run_iteration function 
def run_iteration(iteration, imputation_methods=None, random_seed=57):
    """Run classification for one iteration"""
    if imputation_methods is None:
        imputation_methods = ["knn", "mice", "simple"]
    
    iteration_results = []
    all_feature_importance = []
    
    try:
        print(f"\nProcessing complete dataset (iteration {iteration})...")
        datasets = load_datasets(iteration, imputation_methods[0])
        X_train, y_train, features = prepare_data(datasets["train_complete"])
        X_test, y_test, _ = prepare_data(datasets["test_complete"])
        
        complete_results, complete_importance = process_dataset(X_train, y_train, X_test, y_test, "complete", "complete", iteration, random_seed)
        iteration_results.extend(complete_results)
        all_feature_importance.extend(complete_importance)
    except Exception as e:
        print(f"ERROR processing complete dataset: {e}")
    
    # process each imputation method
    for method in imputation_methods:
        print(f"Processing {method.upper()} imputation (iteration {iteration})...")
        
        try:
            datasets = load_datasets(iteration, method)
            
            # low missing data
            X_train, y_train, features = prepare_data(datasets["train_low"])
            X_test, y_test, _ = prepare_data(datasets["test_low"])
            
            low_results, low_importance = process_dataset(X_train, y_train, X_test, y_test, method, "low", iteration, random_seed)
            iteration_results.extend(low_results)
            all_feature_importance.extend(low_importance)
            
            # high missing data
            X_train, y_train, features = prepare_data(datasets["train_high"])
            X_test, y_test, _ = prepare_data(datasets["test_high"])
            
            high_results, high_importance = process_dataset(X_train, y_train, X_test, y_test, method, "high", iteration, random_seed)
            iteration_results.extend(high_results)
            all_feature_importance.extend(high_importance)
            
        except Exception as e:
            print(f"ERROR processing {method} imputation: {e}")
    
    results_df = pd.DataFrame(iteration_results)
    
    # save results to summary file
    summary_file = "results/classification_summary.csv"
    if os.path.exists(summary_file):
        existing_results = pd.read_csv(summary_file)
        # remove any existing results 
        existing_results = existing_results[existing_results['iteration'] != iteration]
        combined_results = pd.concat([existing_results, results_df], ignore_index=True)
        combined_results.to_csv(summary_file, index=False)
    else:
        results_df.to_csv(summary_file, index=False)
    
    # save feature importance 
    importance_file = "results/feature_importance.csv"
    importance_df = pd.DataFrame(all_feature_importance)
    
    if os.path.exists(importance_file):
        existing_importance = pd.read_csv(importance_file)
        # remove any existing results 
        existing_importance = existing_importance[existing_importance['iteration'] != iteration]
        combined_importance = pd.concat([existing_importance, importance_df], ignore_index=True)
        combined_importance.to_csv(importance_file, index=False)
    else:
        importance_df.to_csv(importance_file, index=False)
    
    print(f"Completed classification for iteration {iteration}")
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run classification on imputed datasets")
    parser.add_argument("--iterations", type=int, default=1,
                      help="Number of iterations to process (default: 1)")
    parser.add_argument("--iteration", type=int, default=1,
                      help="Current iteration number (default: 1)")
    parser.add_argument("--methods", nargs="+", default=["knn", "mice", "simple"],
                      help="Imputation methods to include (default: knn mice simple)")
    parser.add_argument("--seed", type=int, default=57,
                      help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    start_time = datetime.datetime.now()
    print(f"Starting classification for iteration {args.iteration} at {start_time}")
    print(f"Using seed: {args.seed}")
    
    try:
        # run current iteration
        iter_results = run_iteration(args.iteration, args.methods, args.seed)
        
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        
        print(f"\nStarted:  {start_time}")
        print(f"Finished: {end_time}")
        print(f"Elapsed:  {elapsed_time}")
        print(f"Classification complete for iteration {args.iteration}!")
        
    except Exception as e:
        print(f"Error in iteration {args.iteration}: {e}")