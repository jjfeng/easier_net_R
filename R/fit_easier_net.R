# Creates an EASIER-net
easier_net <- function(sier_net_list) {
  keras::keras_model_custom(name = "EASIERnet", function(self) {
    self$sier_net_list <- sier_net_list
    self$is_classification <- sier_net_list[[1]]$is_classification
    self$num_nets <- length(sier_net_list)/1.0

    function(inputs, mask = NULL, training=FALSE) {
      res <- 0
      for (i in seq(0, self$num_nets - 1)) {
        if (self$is_classification) {
          model_probs <- self$sier_net_list[i]$call(inputs)
          res <- res + model_probs
        } else {
          res <- res + self$sier_net_list[i]$call(inputs)
        }
      }
      res/self$num_nets
    }
  })
}

#' Fits an ensemble by averaging sparse-input hierarchical networks
#' Currently does this sequentially, no parallelism
#'
#' @param x_train matrix with rows as observations, columns as covariates
#' @param y_train the observed responses for each observation, should be the number of the class (0-indexed)
#' @param loss the name of the loss function to use ("mse" or "sparse_categorical_crossentropy")
#' @param loss_func the loss function to use (keras::loss_mean_squared_error or keras::loss_sparse_categorical_crossentropy)
#' @param num_layers number of hidden layer
#' @param num_hidden_nodes number of hidden nodes per layer
#' @param num_out number of outputs
#' @param num_nets number of member networks
#' @param lambda1 penalty parameter for input sparsity (lambda1 in the paper)
#' @param lambda2 penalty parameter for network size (lambda2 in the paper)
#' @param epochs_adam number of epochs to run Adam
#' @param epochs_prox number of epochs to run proximal gradient descent
#' @param validation_split proportion of data to split for validation
#' @param batch_size the size of the mini-batch in Adam
#' @param sample_weight how much to weight each observation, optional
#' @param learning_rate the step size/learning rate for the optimization algorithms
#' @param seed random seed for initializing weights
#' @return A fitted sparse-input hierarchical network (a pytorch object)
#' @export
fit_easier_net <- function(
  x_train,
  y_train,
  loss,
  loss_func,
  num_layers,
  num_hidden_nodes,
  num_out,
  lambda1,
  lambda2,
  epochs_adam,
  epochs_prox,
  num_nets = 3,
  validation_split = 0,
  batch_size = 100,
  sample_weight = NULL,
  learning_rate = 0.0001,
  seed = 1,
  metrics = NULL
) {
  sier_net_list <- vector("list", num_nets)
  for (i in seq(1:num_nets)) {
    sier_net_list[[i]] <- fit_sier_net(
      x_train,
      y_train,
      loss,
      loss_func,
      num_layers,
      num_hidden_nodes,
      num_out,
      lambda1,
      lambda2,
      epochs_adam,
      epochs_prox,
      validation_split,
      batch_size,
      sample_weight,
      learning_rate,
      seed = seed + i)
  }
  my_easier_net <- easier_net(sier_net_list)
  my_easier_net %>% compile(
    loss = loss,
    metrics= metrics,
    optimizer = "adam")
  list(ensemble=my_easier_net, members=sier_net_list)
}
