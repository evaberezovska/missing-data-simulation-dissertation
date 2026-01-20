library(VIM)
library(Metrics)

# parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
iteration <- as.numeric(args[1])        
seed_value <- if (length(args) >= 2) as.numeric(args[2]) else 571

set.seed(seed_value)

# identified best k for both low and high
k_value <- 25 

cat("Running KNN imputation with:\n")
cat("- Iteration:", iteration, "\n")
cat("- Seed:", seed_value, "\n")
cat("- k value:", k_value, "\n")

# storage for results
results_summary <- data.frame(
  iteration = integer(),
  method = character(),
  missing_level = character(),
  data_set = character(),
  rmse = numeric(),
  mae = numeric(),
  stringsAsFactors = FALSE
)

# convert binary columns to factors
prepare_data_for_knn <- function(data_df) {
  binary_cols <- sapply(data_df, function(x) {
    all(x %in% c(0, 1, NA), na.rm = TRUE)
  })
  
  for (col in names(data_df)[binary_cols]) {
    if (col != "target") {  
      data_df[[col]] <- factor(data_df[[col]], levels = c(0, 1))
    }
  }
  
  return(data_df)
}

# function to calculate imputation metrics
calculate_metrics <- function(original_data, imputed_data, complete_data, features) {
  true_vals <- c()
  pred_vals <- c()
  
  for(row in 1:nrow(original_data)) {
    for(col in features) {
      if(is.na(original_data[row, col])) {
        pred_val <- if(is.factor(imputed_data[[col]])) {
          as.numeric(as.character(imputed_data[row, col]))
        } else {
          as.numeric(imputed_data[row, col])
        }
        
        truth_val <- if(is.factor(complete_data[[col]])) {
          as.numeric(as.character(complete_data[row, col]))
        } else {
          as.numeric(complete_data[row, col])
        }
        
        pred_vals <- c(pred_vals, pred_val)
        true_vals <- c(true_vals, truth_val)
      }
    }
  }
  
  if(length(true_vals) == 0) {
    return(list(rmse = NA, mae = NA))
  }
  
  return(list(
    rmse = rmse(true_vals, pred_vals),
    mae = mae(true_vals, pred_vals)
  ))
}

# start timing
start_time <- Sys.time()

tryCatch({
  cat("\n--- Processing iteration", iteration, "---\n")
  
  # load data for an iteration
  iteration_dir <- file.path("data/amputed", paste0("iteration_", iteration))
  complete_dir <- file.path("data/complete", paste0("iteration_", iteration))
  
  if (!dir.exists(iteration_dir) || !dir.exists(complete_dir)) {
    stop("Data directories don't exist for iteration ", iteration)
  }
  
  train_complete <- read.csv(file.path(complete_dir, "train_complete.csv"))
  test_complete <- read.csv(file.path(complete_dir, "test_complete.csv"))
  train_low <- read.csv(file.path(iteration_dir, "train_low.csv"))
  test_low <- read.csv(file.path(iteration_dir, "test_low.csv"))
  train_high <- read.csv(file.path(iteration_dir, "train_high.csv"))
  test_high <- read.csv(file.path(iteration_dir, "test_high.csv"))
  
  # prepare features list, excluding target
  features <- setdiff(names(train_low), "target")
  
  # convert binary columns to factors
  train_low <- prepare_data_for_knn(train_low)
  test_low <- prepare_data_for_knn(test_low)
  train_high <- prepare_data_for_knn(train_high)
  test_high <- prepare_data_for_knn(test_high)
  
  # process LOW missing data
  cat("Processing low missing data...\n")

  # find features with missing values
  missing_features_train_low <- features[sapply(train_low[,features], function(x) any(is.na(x)))]
  missing_features_test_low <- features[sapply(test_low[,features], function(x) any(is.na(x)))]
  
  # impute training data - LOW
  set.seed(seed_value + iteration)
  imp_train_low <- kNN(
    data = train_low,
    variable = missing_features_train_low,
    metric = "gower", 
    k = k_value,
    numFun = mean,
    catFun = maxCat,
    impNA = TRUE,
    methodStand = "range",
    imp_var = FALSE,
    trace = FALSE
  )
  
  # impute test data - LOW
  set.seed(seed_value + iteration)
  combined_test_low <- rbind(train_low[,features], test_low[,features])
  imp_test_comb_low <- kNN(
    data = combined_test_low,
    variable = missing_features_test_low,
    metric = "gower", 
    k = k_value,
    numFun = mean,
    catFun = maxCat,
    impNA = TRUE,
    methodStand = "range",
    imp_var = FALSE,
    trace = FALSE
  )
  
  n_train_low <- nrow(train_low)
  imp_test_low <- imp_test_comb_low[(n_train_low+1):nrow(imp_test_comb_low), ]
  imp_test_low$target <- test_low$target
  
  # calculate metrics for LOW missing data
  train_low_metrics <- calculate_metrics(train_low, imp_train_low, train_complete, features)
  test_low_metrics <- calculate_metrics(test_low, imp_test_low, test_complete, features)
  
  cat("Low missing - Train RMSE:", round(train_low_metrics$rmse, 4), "MAE:", round(train_low_metrics$mae, 4), "\n")
  cat("Low missing - Test RMSE:", round(test_low_metrics$rmse, 4), "MAE:", round(test_low_metrics$mae, 4), "\n")
  
  # process HIGH missing data
  cat("Processing high missing data...\n")
  
  # find features with missing values
  missing_features_train_high <- features[sapply(train_high[,features], function(x) any(is.na(x)))]
  missing_features_test_high <- features[sapply(test_high[,features], function(x) any(is.na(x)))]
  
  # impute training data - HIGH
  set.seed(seed_value + iteration + 100)
  imp_train_high <- kNN(
    data = train_high,
    variable = missing_features_train_high,
    metric = "gower", 
    k = k_value,
    numFun = mean,
    catFun = maxCat,
    impNA = TRUE,
    methodStand = "range",
    imp_var = FALSE,
    trace = FALSE
  )

  # impute test data - HIGH 
  set.seed(seed_value + iteration + 100)
  combined_test_high <- rbind(train_high[,features], test_high[,features])
  imp_test_comb_high <- kNN(
    data = combined_test_high,
    variable = missing_features_test_high,
    metric = "gower", 
    k = k_value,
    numFun = mean,
    catFun = maxCat,
    impNA = TRUE,
    methodStand = "range",
    imp_var = FALSE,
    trace = FALSE
  )
  
  n_train_high <- nrow(train_high)
  imp_test_high <- imp_test_comb_high[(n_train_high+1):nrow(imp_test_comb_high), ]
  imp_test_high$target <- test_high$target
  
  # calculate metrics for HIGH missing data
  train_high_metrics <- calculate_metrics(train_high, imp_train_high, train_complete, features)
  test_high_metrics <- calculate_metrics(test_high, imp_test_high, test_complete, features)
  
  cat("High missing - Train RMSE:", round(train_high_metrics$rmse, 4), "MAE:", round(train_high_metrics$mae, 4), "\n")
  cat("High missing - Test RMSE:", round(test_high_metrics$rmse, 4), "MAE:", round(test_high_metrics$mae, 4), "\n")
  
  # save imputed datasets
  imputed_dir <- file.path("data/imputed", paste0("iteration_", iteration))
  dir.create(imputed_dir, recursive = TRUE, showWarnings = FALSE)

  write.csv(imp_train_low, file.path(imputed_dir, "train_low_knn.csv"), row.names = FALSE)
  write.csv(imp_test_low, file.path(imputed_dir, "test_low_knn.csv"), row.names = FALSE)
  write.csv(imp_train_high, file.path(imputed_dir, "train_high_knn.csv"), row.names = FALSE)
  write.csv(imp_test_high, file.path(imputed_dir, "test_high_knn.csv"), row.names = FALSE)
  
  cat("Saved imputed datasets for iteration", iteration, "\n")
  
  # store results
  results_summary <- rbind(
    results_summary,
    data.frame(
      iteration = iteration,
      method = "knn",
      missing_level = "low",
      data_set = "train",
      rmse = train_low_metrics$rmse,
      mae = train_low_metrics$mae,
      stringsAsFactors = FALSE
    ),
    data.frame(
      iteration = iteration,
      method = "knn",
      missing_level = "low",
      data_set = "test",
      rmse = test_low_metrics$rmse,
      mae = test_low_metrics$mae,
      stringsAsFactors = FALSE
    ),
    data.frame(
      iteration = iteration,
      method = "knn",
      missing_level = "high",
      data_set = "train",
      rmse = train_high_metrics$rmse,
      mae = train_high_metrics$mae,
      stringsAsFactors = FALSE
    ),
    data.frame(
      iteration = iteration,
      method = "knn",
      missing_level = "high",
      data_set = "test",
      rmse = test_high_metrics$rmse,
      mae = test_high_metrics$mae,
      stringsAsFactors = FALSE
    )
  )
  
  # save results to summary 
  summary_file <- "results/imputation_summary.csv"
  if (file.exists(summary_file)) {
    existing_results <- read.csv(summary_file)
    # remove any existing results 
    existing_results <- existing_results[!(existing_results$iteration == iteration & existing_results$method == "knn"), ]
    combined_results <- rbind(existing_results, results_summary)
    write.csv(combined_results, summary_file, row.names = FALSE)
  } else {
    write.csv(results_summary, summary_file, row.names = FALSE)
  }
  
  cat("Completed iteration", iteration, "\n")
  
}, error = function(e) {
  cat("ERROR in iteration", iteration, ":", conditionMessage(e), "\n")
})

# end timing
end_time <- Sys.time()
elapsed <- difftime(end_time, start_time, units = "mins")
cat("\nExecution time:", round(elapsed, 2), "minutes\n")

cat("\nKNN imputation complete for iteration", iteration, "!\n")
