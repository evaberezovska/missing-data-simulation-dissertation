import os
import argparse

def create_directory_structure(num_iterations=5):
    """
    Create the directory structure for multiple simulation runs
    """
    # base directories 
    base_dirs = [
        "data/complete",
        "data/amputed", 
        "data/imputed",
        "results"
    ]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    # iteration-specific directories for data 
    for i in range(1, num_iterations + 1):
        iteration_dirs = [
            f"data/complete/iteration_{i}",
            f"data/amputed/iteration_{i}",
            f"data/imputed/iteration_{i}"
        ]
        
        for dir_path in iteration_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")
    
    print(f"\nSuccessfully created directory structure for {num_iterations} iterations")
    print("\nDirectory structure:")
    print("├── data/")
    print("│   ├── complete/iteration_1...N/")
    print("│   ├── amputed/iteration_1...N/")
    print("│   └── imputed/iteration_1...N/")
    print("└── results/")
    print("    ├── imputation_summary.csv")
    print("    ├── classification_summary.csv")
    print("    └── simulation_seeds.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create directory structure for simulation runs")
    parser.add_argument("--iterations", type=int, default=5, 
                        help="Number of simulation iterations (default: 5)")
    
    args = parser.parse_args()
    create_directory_structure(args.iterations)