import os
import subprocess
import argparse
import datetime
import sys
import json
import numpy as np
from setup_directories import create_directory_structure

def generate_seeds(base_seed, num_iterations):
    """Generate algorithm-specific seeds for each iteration"""
    np.random.seed(base_seed)
    
    # create a dictionary to store all seeds
    seeds = {}
    
    # generate seeds for each iteration and each algorithm component
    for i in range(1, num_iterations + 1):
        seeds[f"iteration_{i}"] = {
            "data_generation": np.random.randint(1, 100000),
            "missing_data": np.random.randint(1, 100000),
            "mice_imputation": np.random.randint(1, 100000),
            "knn_imputation": np.random.randint(1, 100000),
            "classification": np.random.randint(1, 100000)
        }
    
    return seeds

def run_simulations(num_iterations=100, base_seed=57):
    """
    Main controller that manages multiple simulation runs with different seeds.
    """
    start_time = datetime.datetime.now()
    print(f"Starting simulation at {start_time}")
    print(f"Running {num_iterations} iterations with base seed {base_seed}")
    
    # generate and save seeds for reproducibility
    seeds = generate_seeds(base_seed, num_iterations)
    os.makedirs("results", exist_ok=True)
    with open("results/simulation_seeds.json", "w") as f:
        json.dump(seeds, f, indent=2)
    
    print(f"Generated unique seeds for each algorithm and iteration")
    print(f"Seeds saved to results/simulation_seeds.json for reproducibility")
    
    # 1. set directory structure
    print("\nSetting up directory structure...")
    create_directory_structure(num_iterations)
    
    # 2 & 3. generate data and apply missing data for each iteration
    for i in range(1, num_iterations + 1):
        iter_seeds = seeds[f"iteration_{i}"]
            
        # 2. generate synthetic data
        print(f"\n[Iteration {i}/{num_iterations}] Generating synthetic data...")
        print(f"Using seed: {iter_seeds['data_generation']}")
        subprocess.run(
            f"python data_generation.py --iteration {i} --seed {iter_seeds['data_generation']} --samples 1000",
            shell=True,
            check=True
        )
            
        # 3. apply missing data mechanisms
        print(f"[Iteration {i}/{num_iterations}] Applying missing data mechanisms...")
        print(f"Using seed: {iter_seeds['missing_data']}")
        subprocess.run(
            f"python apply_missing.py --iteration {i} --seed {iter_seeds['missing_data']}",
            shell=True, 
            check=True
        )
    
    # 4. run MICE imputation across all iterations
    print(f"\nRunning MICE imputation across {num_iterations} iterations...")
    print(f"Using fixed parameters: m=30, maxit=20 (based on diagnostic analysis)")
    for i in range(1, num_iterations + 1):
        print(f"[Iteration {i}/{num_iterations}] Running MICE imputation...")
        print(f"Using seed: {seeds[f'iteration_{i}']['mice_imputation']}")
        subprocess.run(
            f"Rscript run_mice.R {i} {seeds[f'iteration_{i}']['mice_imputation']}",
            shell=True,
            check=True
        )
    
    # 5. run KNN imputation across all iterations
    print(f"\nRunning KNN imputation across {num_iterations} iterations...")
    print(f"Using fixed k=30 for both low and high missing data")
    for i in range(1, num_iterations + 1):
        print(f"[Iteration {i}/{num_iterations}] Running KNN imputation...")
        print(f"Using seed: {seeds[f'iteration_{i}']['knn_imputation']}")
        subprocess.run(
            f"Rscript run_knn.R {i} {seeds[f'iteration_{i}']['knn_imputation']}",
            shell=True,
            check=True
        )
    
    # 6. run Simple imputation across all iterations
    print(f"\nRunning Simple imputation across {num_iterations} iterations...")
    for i in range(1, num_iterations + 1):
        print(f"[Iteration {i}/{num_iterations}] Running Simple imputation...")
        subprocess.run(
            f"python simple_imputer.py --iteration {i}",
            shell=True,
            check=True
        )
    
    # 7. run classification for each iteration 
    print(f"\nRunning classification analysis across {num_iterations} iterations...")
    for i in range(1, num_iterations + 1):
        print(f"[Iteration {i}/{num_iterations}] Running classification...")
        print(f"Using seed: {seeds[f'iteration_{i}']['classification']}")
        subprocess.run(
            f"python run_classification.py --iteration {i} --seed {seeds[f'iteration_{i}']['classification']}",
            shell=True,
            check=True
        )
    
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    
    print(f"\nSimulation started at:  {start_time}")
    print(f"Simulation finished at: {end_time}")
    print(f"Total elapsed time:     {elapsed_time}")
    
    # summary of outputs
    print("\n" + "="*50)
    print("SIMULATION SUMMARY")
    print("="*50)
    print(f"Completed {num_iterations} iterations")
    print(f"MICE parameters: m=30, maxit=20")
    print(f"KNN parameter: k=30")
    
    # summary of created files
    print("\nFiles generated per iteration:")
    print("\n  Complete datasets:")
    print("    - data/complete/iteration_N/train_complete.csv")
    print("    - data/complete/iteration_N/test_complete.csv")
    
    print("\n  Missing data:")
    print("    - data/amputed/iteration_N/train_low.csv")
    print("    - data/amputed/iteration_N/test_low.csv")
    print("    - data/amputed/iteration_N/train_high.csv")
    print("    - data/amputed/iteration_N/test_high.csv")
    
    print("\n  Imputed datasets:")
    print("    - data/imputed/iteration_N/train_low_mice.csv")
    print("    - data/imputed/iteration_N/test_low_mice.csv")  
    print("    - data/imputed/iteration_N/train_high_mice.csv")
    print("    - data/imputed/iteration_N/test_high_mice.csv")
    print("    - data/imputed/iteration_N/train_low_knn.csv")
    print("    - data/imputed/iteration_N/test_low_knn.csv")
    print("    - data/imputed/iteration_N/train_high_knn.csv")
    print("    - data/imputed/iteration_N/test_high_knn.csv")
    print("    - data/imputed/iteration_N/train_low_simple.csv") 
    print("    - data/imputed/iteration_N/test_low_simple.csv")
    print("    - data/imputed/iteration_N/train_high_simple.csv")
    print("    - data/imputed/iteration_N/test_high_simple.csv")
    
    print("\n  Summary results:")
    print("    - results/imputation_summary.csv")
    print("    - results/classification_summary.csv")
    print("    - results/feature_importance.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multiple simulation iterations")
    parser.add_argument("--iterations", "-i", type=int, default=2,
                       help="Number of iterations to run (default: 2)")
    parser.add_argument("--seed", type=int, default=57,
                       help="Base random seed for reproducibility (default: 57)")
    
    args = parser.parse_args()
    
    run_simulations(args.iterations, args.seed)