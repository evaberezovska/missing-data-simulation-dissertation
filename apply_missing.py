import pandas as pd
import numpy as np
import os
from pyampute import MultivariateAmputation
import argparse

def load_complete_datasets(iteration):
    """
    Load complete datasets for a specific iteration
    
    """
    # define paths
    iter_dir = f"data/complete/iteration_{iteration}"
    train_path = f"{iter_dir}/train_complete.csv"
    test_path = f"{iter_dir}/test_complete.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Complete datasets for iteration {iteration} not found in {iter_dir}")
    
    # load datasets
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    print(f"Loaded complete datasets from iteration {iteration}")
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    
    return train_data, test_data

def apply_amputation(df, mcar_features, mar_features, prop, seed=57):
    """
    Apply mixed missing data mechanisms to a dataset
    
    """
    # get feature names excluding target
    features = [col for col in df.columns if col != 'target']
    
    # define amputation patterns
    patterns = [
        {
            "incomplete_vars": mcar_features,
            "mechanism": "MCAR",
            "freq": 0.5
        },
        {
            "incomplete_vars": mar_features,
            "mechanism": "MAR",
            "freq": 0.5,
            "weights": {
                "cont_7": 1.5,
                "cont_9": 1.3,
                "cont_11": 1.1,
                "disc_3": 1.0,
            }
        }
    ]
    
    # create amputer and apply missing values
    amputer = MultivariateAmputation(prop=prop, patterns=patterns, seed=seed)
    features_with_missing = amputer.fit_transform(df[features])
    
    # create new dataframe with missing values
    df_miss = df.copy()
    df_miss[features] = features_with_missing
    
    # calculate missing statistics
    missing_count = df_miss.drop('target', axis=1).isnull().sum().sum()
    total_cells = df_miss.shape[0] * (df_miss.shape[1] - 1)
    missing_rate = (missing_count / total_cells) * 100
    
    # create missing report
    missing_report = {
        'missing_count': missing_count,
        'total_cells': total_cells,
        'missing_rate': missing_rate
    }
    
    return df_miss, missing_report

def create_missing_datasets(train_data, test_data, iteration=1, random_seed=57):
    """
    Create datasets with low and high missing rates
    
    """

    print(f"Creating missing datasets for iteration {iteration} with seed {random_seed}")
    
    # create copies of data
    train_1 = train_data.copy()
    test_1 = test_data.copy()
    train_2 = train_data.copy()
    test_2 = test_data.copy()
    
    # define features for missing data mechanisms
    mcar_features = ['cont_0', 'cont_1', 'cont_2', 'disc_2', 'disc_4', 'cat_0_Yes', 'cat_3_Yes']  
    mar_features = ['cont_3', 'cont_4', 'cont_5', 'disc_0', 'disc_1', 'cat_4_Level_B', 'cat_4_Level_C']
    
    # apply low missing rate (10%)
    print("Applying low missing rate (target: ~10%):")
    train_low, train_low_report = apply_amputation(train_1, mcar_features, mar_features, prop=0.371, seed=random_seed)
    print(f"  Train low missing rate: {train_low_report['missing_rate']:.2f}%")
    
    test_low, test_low_report = apply_amputation(test_1, mcar_features, mar_features, prop=0.393, seed=random_seed)
    print(f"  Test low missing rate: {test_low_report['missing_rate']:.2f}%")
    
    # apply high missing rate (25%)
    print("Applying high missing rate (target: ~25%):")
    train_high, train_high_report = apply_amputation(train_2, mcar_features, mar_features, prop=0.91, seed=random_seed)
    print(f"  Train high missing rate: {train_high_report['missing_rate']:.2f}%")
    
    test_high, test_high_report = apply_amputation(test_2, mcar_features, mar_features, prop=0.91923, seed=random_seed)
    print(f"  Test high missing rate: {test_high_report['missing_rate']:.2f}%")
    
    return train_low, test_low, train_high, test_high

def save_missing_datasets(train_low, test_low, train_high, test_high, iteration):
    """
    Save datasets with missing values
    
    """
    # create iteration directory
    iter_dir = f"data/amputed/iteration_{iteration}"
    os.makedirs(iter_dir, exist_ok=True)
    
    # save files
    train_low.to_csv(f"{iter_dir}/train_low.csv", index=False)
    test_low.to_csv(f"{iter_dir}/test_low.csv", index=False)
    train_high.to_csv(f"{iter_dir}/train_high.csv", index=False)
    test_high.to_csv(f"{iter_dir}/test_high.csv", index=False)
    
    print(f"Saved missing datasets to {iter_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply missing data mechanisms")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration number for file naming (default: 1)")
    parser.add_argument("--seed", type=int, default=57,
                        help="Base random seed (default: 57)")
    
    args = parser.parse_args()
    
    # load complete datasets
    try:
        train_data, test_data = load_complete_datasets(args.iteration)
        
        # create and save datasets with missing values
        train_low, test_low, train_high, test_high = create_missing_datasets(
            train_data, 
            test_data, 
            iteration=args.iteration, 
            random_seed=args.seed
        )
        
        # cave datasets
        save_missing_datasets(
            train_low,
            test_low,
            train_high,
            test_high,
            args.iteration
        )
        
        print(f"Successfully created missing datasets for iteration {args.iteration}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate complete datasets first using data_generation.py")