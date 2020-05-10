###
# Runs cross validation over hyperparameter settings.
###
source("manifest.R")
library(easiernet)

# Select one of the two example settings below
# Change the data files accordingly
regression_settings <- list(
  data_file="data/iris_regression.csv",
  fit_file="run_easier_net_regression.R",
  loss="eval_mse")
classification_settings <- list(
  data_file="data/iris.csv",
  fit_file="run_easier_net_classification.R",
  loss="eval_sparse_categorical_crossentropy")
settings <- regression_settings

RUNS_DIR <- "lambda_tuning"
FINAL_DIR <- "final_run"
FINAL_MODEL_PATH <- "_output/final_model"

NUM_NETS = 6

# Tunes the hyperparameters -----------------------------------------------
# Currently tune hyperparameters locally.
# For cloud version, follow instructions here:
# https://github.com/rstudio/cloudml/tree/master/inst/examples/tfestimators
# https://tensorflow.rstudio.com/tools/cloudml/getting_started/

runs <- tuning_run(settings$fit_file, runs_dir = RUNS_DIR, flags = list(
  data_path = c(settings$data_file),
  num_layers = c(5),
  num_hidden_nodes = c(100),
  lambda1 = c(0.001),
  lambda2 = c(0.1,0.01),
  epochs_adam = c(200),
  epochs_prox = c(10),
  validation_split = c(0.2),
  data_seed = c(2),
  model_seed = c(2),
  num_nets = c(NUM_NETS/2)
), confirm=FALSE)

# Get best hyperparameter setting -------------------------------------------
ordered_runs <- runs[order(runs[[settings$loss]], decreasing = FALSE),]
print(ordered_runs)
best_run <- ordered_runs[1,]

# Do final fit --------------------------------------------------------------
final_train <- training_run(settings$fit_file, run_dir = FINAL_DIR, flags = list(
  data_path = settings$data_file,
  num_layers = best_run$flag_num_layers,
  num_hidden_nodes = best_run$flag_num_hidden_nodes,
  lambda1 = best_run$flag_lambda1,
  lambda2 = best_run$flag_lambda2,
  epochs_adam = best_run$flag_epochs_adam,
  epochs_prox = best_run$flag_epochs_prox,
  data_seed = 0,
  model_seed = 0,
  num_nets = NUM_NETS,
  validation_split = 0,
  out_path = FINAL_MODEL_PATH
))

# Load the final model---------------------------------------------------------
reloaded_model <- load_model_tf(FINAL_MODEL_PATH)
