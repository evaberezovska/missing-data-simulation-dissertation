library(mice)
library(Metrics)

# parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
iteration <- as.numeric(args[1])
seed_value <- if (length(args) >= 2) as.numeric(args[2]) else 571

set.seed(seed_value)

m_value <- 30     
maxit_value <- 20

cat("Running MICE imputation with:\n")
cat("- Iteration:", iteration, "\n")
cat("- Seed:", seed_value, "\n")
cat("- m =", m_value, "maxit =", maxit_value, "\n")

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

# prepare data converting categorical variables to factors
prepare_data <- function(data) {
  binary_cols <- c("cat_0_Yes", "cat_1_Yes", "cat_2_Yes", "cat_3_Yes", 
                   "cat_4_Level_B", "cat_4_Level_C")
  
  for (col in binary_cols) {
    if (col %in% names(data)) {
      data[[col]] <- factor(data[[col]], levels = c(0, 1))
    }
  }
  
  return(data)
}

# function to create a pooled dataset from multiple imputations
create_pooled_dataset <- function(mice_object) {
  m <- mice_object$m
  data <- mice_object$data  
  
  pooled_data <- data

  for (var in names(data)) {
    miss_idx <- which(is.na(data[[var]]))
    if (length(miss_idx) == 0) next
    
    # for numeric variables
    if (is.numeric(data[[var]])) {
      all_values <- matrix(NA, nrow = length(miss_idx), ncol = m)
      for (i in 1:m) {
        imp_data <- complete(mice_object, i)
        all_values[, i] <- imp_data[miss_idx, var]
      }
      pooled_values <- rowMeans(all_values)
      pooled_data[miss_idx, var] <- pooled_values
    }
    
    # for categorical
    else if (is.factor(data[[var]])) {
      for (j in seq_along(miss_idx)) {
        idx <- miss_idx[j]
        categories <- character(m)
        for (i in 1:m) {
          imp_data <- complete(mice_object, i)
          categories[i] <- as.character(imp_data[idx, var])
        }
        most_common <- names(sort(table(categories), decreasing=TRUE))[1]
        pooled_data[idx, var] <- factor(most_common, levels=levels(data[[var]]))
      }
    }
  }
  
  return(pooled_data)
}

# calculate errors 
calculate_errors <- function(original_data, imputed_data, complete_data) {
  overall_rmse <- 0
  overall_mae <- 0
  total_missing <- 0
  
  for (var in names(original_data)) {
    if (var == "target") next
    
    missing_idx <- which(is.na(original_data[[var]]))
    n_missing <- length(missing_idx)
    
    if (n_missing == 0) next
    
    true_vals <- complete_data[missing_idx, var]
    imp_vals <- imputed_data[missing_idx, var]
    
    if (is.factor(true_vals) || is.factor(imp_vals)) {
      true_vals <- as.numeric(as.character(true_vals))
      imp_vals <- as.numeric(as.character(imp_vals))
    }
    
    overall_rmse <- overall_rmse + sum((true_vals - imp_vals)^2)
    overall_mae <- overall_mae + sum(abs(true_vals - imp_vals))
    total_missing <- total_missing + n_missing
  }
  
  if (total_missing > 0) {
    overall_rmse <- sqrt(overall_rmse / total_missing)
    overall_mae <- overall_mae / total_missing
  } else {
    overall_rmse <- NA
    overall_mae <- NA
  }
  
  return(list(rmse = overall_rmse, mae = overall_mae))
}

# function MICE imputation and evaluation pooled
run_mice <- function(train_data, test_data, train_complete, test_complete, m, maxit, seed_val) {
  # Initialize MICE
  init <- mice(train_data, m = 5, maxit = 0, printFlag = FALSE)
  method <- init$method
  
  # set methods by data type
  continuous_vars <- paste0("cont_", 0:14)
  discrete_vars <- paste0("disc_", 0:4)
  binary_cols <- c("cat_0_Yes", "cat_1_Yes", "cat_2_Yes", "cat_3_Yes", 
                   "cat_4_Level_B", "cat_4_Level_C")
  
  method[continuous_vars] <- "norm"
  method[discrete_vars] <- "pmm"
  method[binary_cols] <- "logreg"
  method["target"] <- ""
  
  # create predictor matrix
  pred <- matrix(1, ncol = ncol(train_data), nrow = ncol(train_data))
  rownames(pred) <- colnames(train_data)
  colnames(pred) <- colnames(train_data)
  diag(pred) <- 0
  pred["target",] <- 0
  
  # run MICE on train data
  set.seed(seed_val)
  imp_train <- mice(train_data, m = m, maxit = maxit, method = method, 
                    predictorMatrix = pred, printFlag = FALSE)
  
  # create pooled train dataset 
  train_pooled <- create_pooled_dataset(imp_train)
  
  # run MICE on test data
  set.seed(seed_val + 1)
  imp_test <- mice(test_data, m = m, maxit = maxit, method = method, 
                   predictorMatrix = pred, printFlag = FALSE)
  
  # create pooled test dataset
  test_pooled <- create_pooled_dataset(imp_test)
  
  # calculate errors on pooled datasets
  train_errors <- calculate_errors(train_data, train_pooled, train_complete)
  test_errors <- calculate_errors(test_data, test_pooled, test_complete)
  
  # results
  return(list(
    train_imputed = train_pooled,   
    test_imputed = test_pooled,     
    train_errors = train_errors,
    test_errors = test_errors
  ))
}

# start timing
start_time <- Sys.time()

tryCatch({
  cat("\n--- Processing iteration", iteration, "---\n")
  
  iteration_dir <- file.path("data/amputed", paste0("iteration_", iteration))
  complete_dir <- file.path("data/complete", paste0("iteration_", iteration))
  
  if (!dir.exists(iteration_dir) || !dir.exists(complete_dir)) {
    stop("Data directories don't exist for iteration ", iteration)
  }
  
  # load data
  cat("Loading data...\n")
  train_low <- prepare_data(read.csv(file.path(iteration_dir, "train_low.csv")))
  test_low <- prepare_data(read.csv(file.path(iteration_dir, "test_low.csv")))
  train_high <- prepare_data(read.csv(file.path(iteration_dir, "train_high.csv")))
  test_high <- prepare_data(read.csv(file.path(iteration_dir, "test_high.csv")))
  train_complete <- read.csv(file.path(complete_dir, "train_complete.csv"))
  test_complete <- read.csv(file.path(complete_dir, "test_complete.csv"))
  
  # process LOW missing data
  cat("Processing LOW missing data...\n")
  low_results <- run_mice(
    train_low, test_low, train_complete, test_complete, 
    m = m_value, maxit = maxit_value, seed_val = seed_value
  )
  
  # process HIGH missing data
  cat("Processing HIGH missing data...\n")
  high_results <- run_mice(
    train_high, test_high, train_complete, test_complete, 
    m = m_value, maxit = maxit_value, seed_val = seed_value + 100
  )
  
  imputed_dir <- file.path("data/imputed", paste0("iteration_", iteration))
  dir.create(imputed_dir, recursive = TRUE, showWarnings = FALSE)
  
  # save imputed data (now pooled datasets)
  write.csv(low_results$train_imputed, file.path(imputed_dir, "train_low_mice.csv"), row.names = FALSE)
  write.csv(low_results$test_imputed, file.path(imputed_dir, "test_low_mice.csv"), row.names = FALSE)
  write.csv(high_results$train_imputed, file.path(imputed_dir, "train_high_mice.csv"), row.names = FALSE)
  write.csv(high_results$test_imputed, file.path(imputed_dir, "test_high_mice.csv"), row.names = FALSE)
  
  cat("Saved imputed datasets for iteration", iteration, "\n")
  
  # print results
  cat("Low missing - Train RMSE:", round(low_results$train_errors$rmse, 4), "MAE:", round(low_results$train_errors$mae, 4), "\n")
  cat("Low missing - Test RMSE:", round(low_results$test_errors$rmse, 4), "MAE:", round(low_results$test_errors$mae, 4), "\n")
  cat("High missing - Train RMSE:", round(high_results$train_errors$rmse, 4), "MAE:", round(high_results$train_errors$mae, 4), "\n")
  cat("High missing - Test RMSE:", round(high_results$test_errors$rmse, 4), "MAE:", round(high_results$test_errors$mae, 4), "\n")
  
  # store results
  results_summary <- rbind(
    results_summary,
    data.frame(
      iteration = iteration,
      method = "mice",
      missing_level = "low",
      data_set = "train",
      rmse = low_results$train_errors$rmse,
      mae = low_results$train_errors$mae,
      stringsAsFactors = FALSE
    ),
    data.frame(
      iteration = iteration,
      method = "mice",
      missing_level = "low",
      data_set = "test",
      rmse = low_results$test_errors$rmse,
      mae = low_results$test_errors$mae,
      stringsAsFactors = FALSE
    ),
    data.frame(
      iteration = iteration,
      method = "mice",
      missing_level = "high",
      data_set = "train",
      rmse = high_results$train_errors$rmse,
      mae = high_results$train_errors$mae,
      stringsAsFactors = FALSE
    ),
    data.frame(
      iteration = iteration,
      method = "mice",
      missing_level = "high",
      data_set = "test",
      rmse = high_results$test_errors$rmse,
      mae = high_results$test_errors$mae,
      stringsAsFactors = FALSE
    )
  )
  
  # save results to summary file
  summary_file <- "results/imputation_summary.csv"
  if (file.exists(summary_file)) {
    existing_results <- read.csv(summary_file)
    # remove any existing results 
    existing_results <- existing_results[!(existing_results$iteration == iteration & existing_results$method == "mice"), ]
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

cat("\nMICE imputation complete for iteration", iteration, "!\n")