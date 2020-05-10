###
# Fits an easier net for a classification problem
###

source("manifest.R")
library(easiernet)

# Hyperparameter flags ---------------------------------------------------

FLAGS <- flags(
  flag_string("data_path", "data/iris.csv", description = "CSV file with no row names, last column is the observed outcome (i.e. 0-indexed class)."),
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
rows <- sample(nrow(data_df))
data_df <- as.matrix(data_df[rows,])
colnames(data_df) <- NULL
n_train <- floor(nrow(data_df) * (1 - FLAGS$validation_split))
x_train <- data_df[1:n_train, 1:(ncol(data_df) - 1)]
y_train <- data_df[1:n_train, ncol(data_df)]

# Create sample weights
class_weight <- 1.0/(table(y_train)/nrow(x_train))
class_weight <- class_weight/sum(class_weight)
sample_weight <- k_cast_to_floatx(class_weight[y_train + 1])

# Train Model --------------------------------------------------------------

model <- fit_easier_net(
  x_train,
  y_train,
  num_layers = FLAGS$num_layers,
  num_hidden_nodes = FLAGS$num_hidden_nodes,
  num_out=length(class_weight),
  num_nets = FLAGS$num_nets,
  loss = "sparse_categorical_crossentropy",
  loss_func = loss_sparse_categorical_crossentropy,
  metrics = c("accuracy", "sparse_categorical_crossentropy"),
  lambda1 = FLAGS$lambda1,
  lambda2 = FLAGS$lambda2,
  epochs_adam = FLAGS$epochs_adam,
  epochs_prox = FLAGS$epochs_prox,
  sample_weight = sample_weight,
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
if (nchar(FLAGS$out_path) > 0) {
  save_model_tf(object = model$ensemble, filepath = FLAGS$out_path)
}