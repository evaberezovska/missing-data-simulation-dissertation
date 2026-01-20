import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import scipy.special
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import os

def generate_features(n_samples=1000, random_state=57):
    """
    Generates mixed data types: continuous, discrete, and categorical features
    
    """
    # set random seed for reproducibility
    np.random.seed(random_state)

    # generate continuous features
    continuous_features = {}
    for i in range(15):
        continuous_features[f'cont_{i}'] = np.random.normal(0, 1, n_samples)
    
    # generate discrete features
    discrete_features = {}
    lambdas = [2, 3, 4, 5, 6] 
    for i in range(5):
        discrete_features[f'disc_{i}'] = np.random.poisson(lambdas[i], n_samples)

    # generate categorical features
    categorical_features = {}
    for i in range(5):
        if i < 4:
            # 4 binomial features
            categories = ['Yes', 'No']
            prob = np.random.uniform(0.3, 0.7)  
            probs = [prob, 1-prob]
            categorical_features[f'cat_{i}'] = np.random.choice(categories, size=n_samples, p=probs)
        else: 
            # 1 multinomial feature 
            categories = ['Level_A', 'Level_B', 'Level_C']
            probs = [0.4, 0.35, 0.25]  
            categorical_features[f'cat_{i}'] = np.random.choice(categories, size=n_samples, p=probs)
    
    return continuous_features, discrete_features, categorical_features

def standardize_features(cont_features, disc_features, cat_features):
    """
    Standardize features: scale numerical features and one-hot encode categorical
    
    """
    # create initial dataframe
    df = pd.DataFrame(cont_features)
    for col, values in disc_features.items():
        df[col] = values
    for col, values in cat_features.items():
        df[col] = values
    
    # get column lists
    cont_cols = list(cont_features.keys())
    disc_cols = list(disc_features.keys())
    cat_cols = list(cat_features.keys())
    
    # scale continuous and discrete features
    scaler = StandardScaler()
    df_scaled_cont_disc = pd.DataFrame(
        scaler.fit_transform(df[cont_cols + disc_cols]),
        columns=cont_cols + disc_cols
    )
    
    # one-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')  
    df_encoded_cat = pd.DataFrame(
        encoder.fit_transform(df[cat_cols]),
        columns=encoder.get_feature_names_out(cat_cols)
    )
    df_encoded_cat = df_encoded_cat.astype(int)
    
    # combine all features
    df_processed = pd.concat([df_scaled_cont_disc, df_encoded_cat], axis=1)
    
    return df_processed, cont_cols, disc_cols, cat_cols

def assign_feature_importance(df_processed, random_state=57):
    """
    Assign importance weights to features
    
    """
    # set random seed
    np.random.seed(random_state)
    
    all_features = df_processed.columns.to_list()
    
    # define importance categories
    high_importance = ['cont_0', 'cont_3', 'cont_7', 'disc_1', 'disc_3', 'cat_0_Yes', 'cat_2_Yes']
    medium_importance = ['cont_1', 'cont_5', 'cont_9', 'cont_11', 'cont_13', 'disc_0', 'disc_4', 'cat_1_Yes', 'cat_4_Level_C']
    low_importance = [col for col in all_features if col not in high_importance + medium_importance]
    
    # define weight ranges (absolute values)
    high_abs_range = (2.0, 3.0)
    medium_abs_range = (1.0, 2.0)
    low_abs_range = (0.0, 1.0)
    
    # assign weights with random signs
    feature_weights = {}
    for f in high_importance:
        weight = np.random.uniform(*high_abs_range)
        sign = np.random.choice([-1, 1])  
        feature_weights[f] = np.round(sign * weight, 2)
        
    for f in medium_importance:
        weight = np.random.uniform(*medium_abs_range)
        sign = np.random.choice([-1, 1])  
        feature_weights[f] = np.round(sign * weight, 2)
        
    for f in low_importance:
        weight = np.random.uniform(*low_abs_range)
        sign = np.random.choice([-1, 1])  
        feature_weights[f] = np.round(sign * weight, 2)
    
    return feature_weights, high_importance, medium_importance, low_importance

def generate_target(df_processed, feature_weights, random_state=57):
    """
    Generate target variable using logistic function
    
    """
    # set random seed
    np.random.seed(random_state)
    
    all_features = df_processed.columns.to_list()
    X = df_processed[all_features].values
    B = np.array([feature_weights[f] for f in all_features])
    
    # use intercept to balance classes
    my_intercept = 4
    logit_p = X @ B + my_intercept
    p = scipy.special.expit(logit_p)
    y = np.random.binomial(1, p)
    
    # add target to dataframe
    synthetic_data = df_processed.copy()
    synthetic_data['target'] = y
    
    return synthetic_data

def check_data_quality(synthetic_data, show_plots=False):
    """
    Check data quality: types, distributions, collinearity
    """
    # check data types
    dtypes = synthetic_data.dtypes
    
    # check target distribution
    target_balance = synthetic_data['target'].mean()
    
    # check collinearity
    X_vif = synthetic_data.drop(columns=['target'])
    vif_data = pd.DataFrame()
    vif_data['feature'] = X_vif.columns
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    max_vif = vif_data['VIF'].max()
    
    quality_report = {
        'target_balance': target_balance,
        'max_vif': max_vif,
        'num_features': len(synthetic_data.columns) - 1
    }
    
    return quality_report

def split_and_save_data(synthetic_data, iteration, output_dir="data/complete", random_state=57):
    """
    Split data into train and test sets and save to files
    
    """
    # stratified split data
    train_data, test_data = train_test_split(
        synthetic_data, 
        test_size=0.3, 
        random_state=random_state, 
        stratify=synthetic_data['target']
    )
    
    # create iteration directory path
    iter_dir = f"{output_dir}/iteration_{iteration}"
    os.makedirs(iter_dir, exist_ok=True)
    
    # save files
    train_data.to_csv(f"{iter_dir}/train_complete.csv", index=False)
    test_data.to_csv(f"{iter_dir}/test_complete.csv", index=False)
    
    print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    print(f"Train target distribution: {train_data['target'].value_counts(normalize=True)}")
    print(f"Test target distribution: {test_data['target'].value_counts(normalize=True)}")
    
    return train_data, test_data

def generate_synthetic_dataset(iteration=1, random_seed=57, n_samples=1000, show_plots=False):
    """
    Generate a complete synthetic dataset
    
    """
    print(f"Generating synthetic dataset for iteration {iteration} with seed {random_seed}")
    

    cont_features, disc_features, cat_features = generate_features(n_samples, random_state=random_seed)
    df_processed, cont_cols, disc_cols, cat_cols = standardize_features(cont_features, disc_features, cat_features)
    feature_weights, high, medium, low = assign_feature_importance(df_processed, random_state=random_seed)
    synthetic_data = generate_target(df_processed, feature_weights, random_state=random_seed)
    
    quality_report = check_data_quality(synthetic_data)
    print(f"Data quality report: {quality_report}")
    
    train_data, test_data = split_and_save_data(
        synthetic_data, 
        iteration=iteration, 
        output_dir="data/complete", 
        random_state=random_seed
    )
    
    return train_data, test_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic dataset")
    parser.add_argument("--iteration", type=int, default=1,
                        help="Iteration number for file naming (default: 1)")
    parser.add_argument("--seed", type=int, default=57,
                        help="Random seed (default: 57)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of samples to generate (default: 1000)")
    
    args = parser.parse_args()
    
    # generate dataset
    train_data, test_data = generate_synthetic_dataset(
        iteration=args.iteration, 
        random_seed=args.seed,
        n_samples=args.samples
    )
    
    print(f"Successfully generated synthetic data for iteration {args.iteration}")