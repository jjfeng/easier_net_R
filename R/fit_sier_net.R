# In Keras, weight regularization is added by passing weight regularizer instances to layers
# l2(0.001) means that every coefficient in the weight matrix of the layer will add 0.001 * weight_coefficient_value to the total loss of the network
# penalty is only added at training time

# Create custom keras layer: input filter layer
InputFilterLayer <- R6::R6Class("InputFilterLayer",
  inherit = keras::KerasLayer,
  public = list(
    kernel = NULL,
    regularizer = NULL,
    initialize = function(regularizer) {
      self$regularizer <- regularizer
    },
    build = function(input_shape) {
      self$kernel <- self$add_weight(
        name = 'kernel',
        shape = as.integer(list(1, input_shape[[2]])),
        dtype = tensorflow::tf$dtypes$float32,
        initializer = initializer_constant(value = 1),
        regularizer = self$regularizer,
        trainable = TRUE
      )
    },
    call = function(x, mask = NULL) {
      x * self$kernel
    },
    compute_output_shape = function(input_shape) {
      input_shape
    }
  )
)

layer_input_filter <- function(object, regularizer, name = NULL, trainable = TRUE) {
  create_layer(InputFilterLayer, object, list(
    name = name,
    regularizer = regularizer,
    trainable = trainable
  ))
}

# Create skip-connection cross-layer
SkipCrossLayer <- R6::R6Class("SkipCrossLayer",
  inherit = keras::KerasLayer,
  public = list(
    kernel = NULL,
    num_layers = NULL,
    initialize = function(num_layers) {
      self$num_layers <- num_layers
    },
    build = function(input_shape){
      self$kernel <- self$add_weight(
        name = 'kernel',
        shape = as.integer(c(1, self$num_layers)),
        dtype= tensorflow::tf$dtypes$float32,
        # Initialize all connections to have the same positive value
        initializer = initializer_constant(value=1),
        trainable = TRUE
      )
    },
    call = function(x, mask=NULL) {
      normalization = tensorflow::tf$reduce_sum(tensorflow::tf$abs(self$kernel))
      k_sum(x * tensorflow::tf$abs(self$kernel)/normalization, axis=3)
    },
    compute_output_shape = function(input_shape) {
      input_shape
    }
  )
)

cross_layer_skips <- function(object, num_layers, name = NULL, trainable = TRUE) {
  create_layer(SkipCrossLayer, object, list(
    name = name,
    num_layers = as.integer(num_layers),
    trainable = trainable
  ))
}


# Creates a sparse input hierarchical network
#' @export
sier_net <- function(num_layers=1, num_hidden_nodes=10, num_out=1, lambda1=0, lambda2=0, seed=1) {
  keras::keras_model_custom(name = "SIERnet", function(self) {
    seed_idx <- seed
    self$num_layers <- num_layers
    self$num_out <- num_out
    self$is_classification <- num_out > 1

    # Add in skip-connection alpha weights
    self$skip_connects <- cross_layer_skips(num_layers=num_layers + 1, name="cross_layer")

    # Input filter layer
    self$input_filter_layer <- layer_input_filter(
      regularizer = regularizer_l1(l = lambda1),
      name="input_filter"
    )

    # Create the hidden layers
    for (layer in seq(num_layers)) {
      hidden_layer_name <- paste("hidden_layer", layer, sep="")
      self[[hidden_layer_name]] <- layer_dense(
        units=num_hidden_nodes,
        activation="relu",
        kernel_initializer = initializer_glorot_normal(seed = seed_idx),
        bias_initializer = initializer_glorot_normal(seed = seed_idx + 1),
        kernel_regularizer = regularizer_l1(l = lambda2),
        bias_regularizer = regularizer_l1(l = ifelse(self$is_classification, lambda2, 0)),
        name=hidden_layer_name)
      seed_idx <- seed_idx + 2
    }

    for (layer in seq(num_layers + 1)) {
      connect_layer_name <- paste("connect_layer", layer, sep="")
      self[[connect_layer_name]] <- layer_dense(
        units=num_out,
        kernel_initializer = initializer_glorot_normal(seed = seed_idx),
        bias_initializer = initializer_glorot_normal(seed = seed_idx + 1),
        kernel_regularizer = regularizer_l1(l = ifelse(layer == 1, lambda1, lambda2)),
        bias_regularizer = regularizer_l1(l = ifelse(self$is_classification, ifelse(layer == 1, lambda1, lambda2), 0)),
        name=connect_layer_name)
      seed_idx <- seed_idx + 2
    }

    self$pen1 <- function() {
      pen1_val <- tensorflow::tf$reduce_sum(tensorflow::tf$abs(self$input_filter_layer$variables[[1]])) +
        tensorflow::tf$reduce_sum(tensorflow::tf$abs(self$connect_layer1$kernel))
      if (self$is_classification) {
        pen1_val + tensorflow::tf$reduce_sum(tensorflow::tf$abs(self$connect_layer1$bias))
      } else {
        pen1_val
      }
    }
    self$pen2 <- function() {
      pen2_val <- 0
      for (layer in seq(num_layers)) {
        hidden_layer_name <- paste("hidden_layer", layer, sep="")
        pen2_val <- pen2_val + tensorflow::tf$reduce_sum(tensorflow::tf$abs(self[[hidden_layer_name]]$kernel))
        if (self$is_classification) {
          pen2_val <- pen2_val + tensorflow::tf$reduce_sum(tensorflow::tf$abs(self[[hidden_layer_name]]$bias))
        }
      }
      for (layer in seq(2, num_layers + 1)) {
        connect_layer_name <- paste("connect_layer", layer, sep="")
        pen2_val <- pen2_val + tensorflow::tf$reduce_sum(tensorflow::tf$abs(self[[connect_layer_name]]$kernel))
        if (self$is_classification) {
          pen2_val <- pen2_val + tensorflow::tf$reduce_sum(tensorflow::tf$abs(self[[connect_layer_name]]$bias))
        }
      }
      pen2_val
    }

    function(inputs, mask = NULL, training=FALSE) {
      scaled_inputs <- inputs %>% self$input_filter_layer()

      in_layer <- scaled_inputs
      hidden_layers <- list()
      skip_layers <- list()
      for (layer in seq(num_layers)) {
        connect_layer_name <- paste("connect_layer", layer, sep="")
        skip_layers[[layer]] <- self[[connect_layer_name]](in_layer)

        hidden_layer_name <- paste("hidden_layer", layer, sep="")
        hidden_layers[[hidden_layer_name]] <- self[[hidden_layer_name]](in_layer)
        in_layer <- hidden_layers[[hidden_layer_name]]
      }
      last_connect_layer_name = paste("connect_layer", num_layers + 1, sep="")
      skip_layers[[num_layers + 1]] <- self[[last_connect_layer_name]](in_layer)

      skip_layer_stack <- tensorflow::tf$stack(skip_layers, axis=2)
      if (self$is_classification) {
        k_softmax(skip_layer_stack %>% self$skip_connects())
      } else {
        skip_layer_stack %>% self$skip_connects()
      }
    }
  })
}

soft_threshold <- function(tf_var, thres) {
  var_val <- tf_var$value()
  soft_thres_update <- tensorflow::tf$sign(var_val) * tensorflow::tf$maximum(tensorflow::tf$abs(var_val) - thres, 0)
  tf_var$assign(soft_thres_update)
}

emp_loss <- function(model, x_train, y_train, sample_weight, loss_func) {
    if (is.null(sample_weight)) {
      tensorflow::tf$reduce_mean(loss_func(y_train, model$call(x_train)))
    } else {
      tensorflow::tf$reduce_sum(loss_func(y_train, model$call(x_train)) * sample_weight)/tensorflow::tf$reduce_sum(sample_weight)
    }
}

do_prox_grad_descent_step <- function(model, x_train, y_train, sample_weight, loss_func, lambda1, lambda2, learning_rate = 0.0001) {
  # Do the batch gradient descent for the smooth part
  with (tensorflow::tf$GradientTape() %as% t, {
    current_loss <- emp_loss(model, x_train, y_train, sample_weight, loss_func)
  })

  grad <- t$gradient(current_loss, model$variables)
  for (i in seq(1: length(model$variables))) {
    v <- model$variables[[i]]
    v$assign_sub(learning_rate * grad[[i]])
  }

  # Do the soft-thresholding steps now
  soft_threshold(model[["connect_layer1"]]$kernel, lambda1 * learning_rate)
  soft_threshold(model$input_filter_layer$variables[[1]], lambda1 * learning_rate)
  if (model$is_classification) {
    soft_threshold(model[["connect_layer1"]]$bias, lambda1 * learning_rate)
  }

  for (layer in seq(model$num_layers)) {
    hidden_layer_name <- paste("hidden_layer", layer, sep="")
    soft_threshold(model[[hidden_layer_name]]$kernel, lambda2 * learning_rate)
    if (model$is_classification) {
      soft_threshold(model[[hidden_layer_name]]$bias, lambda2 * learning_rate)
    }
  }

  for (layer in seq(2, model$num_layers + 1)) {
    connect_layer_name <- paste("connect_layer", layer, sep="")
    soft_threshold(model[[connect_layer_name]]$kernel, lambda2 * learning_rate)
    if (model$is_classification) {
      soft_threshold(model[[connect_layer_name]]$bias, lambda2 * learning_rate)
    }
  }

  # Get current penalized loss
  emp_loss(model, x_train, y_train, sample_weight, loss_func) +
    lambda1 * model$pen1() + lambda2 * model$pen2()
}

#' Fits a sparse-input hierarchical network
#'
#' @param x_train matrix with rows as observations, columns as covariates
#' @param y_train the observed responses for each observation, should be the number of the class (0-indexed)
#' @param loss the name of the loss function to use ("mse" or "sparse_categorical_crossentropy")
#' @param loss_func the loss function to use (keras::loss_mean_squared_error or keras::loss_sparse_categorical_crossentropy)
#' @param num_layers number of hidden layer
#' @param num_hidden_nodes number of hidden nodes per layer
#' @param num_out number of outputs
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
fit_sier_net <- function(
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
  validation_split = 0,
  batch_size = 100,
  sample_weight = NULL,
  learning_rate = 0.0001,
  seed = 1
) {
  my_sier_net <- sier_net(num_layers = num_layers, num_hidden_nodes = num_hidden_nodes, num_out=num_out, lambda1=lambda1, lambda2=lambda2, seed=seed)

  batch_size = min(ceiling(nrow(x_train) / 3), batch_size)
  my_sier_net %>% compile(
    loss = loss,
    metrics= loss,
    optimizer = "adam")

  # Run adam until convergence first
  my_sier_net %>% fit(
      x = x_train,
      y = y_train,
      sample_weight = sample_weight,
      batch_size = batch_size,
      epochs = epochs_adam,
      validation_split = validation_split,
      shuffle = TRUE,
      verbose = 2
  )

  # Run proximal gradient descent
  if (epochs_prox > 0){
    for (epoch_i in seq(1:epochs_prox)){
      current_pen_loss <- do_prox_grad_descent_step(
        my_sier_net,
        x_train,
        y_train,
        sample_weight,
        loss_func=loss_func,
        lambda1=lambda1,
        lambda2=lambda2,
        learning_rate = learning_rate)
      cat(glue::glue("Prox Epoch: {epoch_i}, Loss: {as.numeric(current_pen_loss)}"), "\n")
    }
  }
  my_sier_net
}
