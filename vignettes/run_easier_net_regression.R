###
# Fits an easier net for a regression problem
###

source("manifest.R")
library(easiernet)

# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  flag_string("data_path", "data/iris_regression.csv", description = "CSV file with no row names, last column is the observed outcome"),
  flag_numeric("data_seed", 0),
  flag_numeric("model_seed", 0),
  flag_numeric("num_nets", 2),
  flag_numeric("num_layers", 2),
  flag_numeric("num_hidden_nodes", 10),
  flag_numeric("lambda1", 0.4),
  flag_numeric("lambda2", 0.3),
  flag_numeric("epochs_adam", 100),
  flag_numeric("epochs_prox", 10),
  flag_numeric("validation_split", 0),
  flag_string("out_path", "")
)

# Data Preparation ---------------------------------------------------
set.seed(FLAGS$data_seed)
assert(nchar(FLAGS$data_path) > 0)

data_df <- fread(FLAGS$data_path)
# Shuffle data
rows <- sample(nrow(data_df))
data_df <- as.matrix(data_df[rows,])
colnames(data_df) <- NULL
n_train <- floor(nrow(data_df) * (1 - FLAGS$validation_split))
x_train <- data_df[1:n_train, 1:(ncol(data_df) - 1)]
y_train <- data_df[1:n_train, ncol(data_df)]

# Train Model --------------------------------------------------------------

model <- fit_easier_net(
  x_train,
  y_train,
  num_layers = FLAGS$num_layers,
  num_hidden_nodes = FLAGS$num_hidden_nodes,
  num_out=1,
  num_nets = FLAGS$num_nets,
  loss="mse",
  metrics=c("mse"),
  loss_func = loss_mean_squared_error,
  lambda1 = FLAGS$lambda1,
  lambda2 = FLAGS$lambda2,
  epochs_adam = FLAGS$epochs_adam,
  epochs_prox = FLAGS$epochs_prox,
  sample_weight = NULL,
  seed = FLAGS$model_seed,
  validation_split = 0
)

# Evaluate Model ----------------------------------------------------------

if (n_train >= nrow(data_df)) {
  model$ensemble %>% evaluate(x_train, y_train, sample_weight = NULL)
} else {
  x_val <- data_df[(n_train + 1):nrow(data_df), 1:(ncol(data_df) - 1)]
  y_val <- data_df[(n_train + 1): nrow(data_df), ncol(data_df)]
  model$ensemble %>% evaluate(x_val, y_val, sample_weight = NULL)
}

# Save Model --------------------------------------------------------------
if (nchar(FLAGS$out_path) > 1) {
  save_model_tf(object = model$ensemble, filepath = FLAGS$out_path)
}