### Optimizers----------------------------------------------------------------
## Stochastic Gradient Descent (vanilla + decay & momentum options) ##
#' @export
optimizer_SGD = list(
  update_params = function(layer_forward, layer_backward, #required
                           learning_rate = 1,
                           lr_decay = 0, iteration = 1,
                           momentum_rate = 0
  ){
    #extract number of neurons from the forward layers output matrix
    n_neurons = ncol(layer_forward$output)
    #current params
    current_weights = layer_forward$weights
    current_biases = layer_forward$biases
    #gradients
    weight_gradients = layer_backward$dweights
    bias_gradients = layer_backward$dbiases

    #learning rate
    currnt_learn_rate = learning_rate * (1 / (1 + lr_decay*iteration))

    #param updates with momentum
    #If momentum == TRUE in parameter initialization, then weights_momentum
    #(and implictly, bias_momentum) will exist
    if (exists("weight_momentums", where = layer_forward)) {
      #current momentums
      weight_momentums = layer_forward$weight_momentums
      bias_momentums = layer_forward$bias_momentums

      #Update weights & biases with momentum:
      #Take prior updates X retainment factor (the "momentum rate"),
      #and update with current gradients
      weight_update =
        (momentum_rate*weight_momentums) - (currnt_learn_rate*weight_gradients)
      bias_update =
        (momentum_rate*bias_momentums) - (currnt_learn_rate*bias_gradients)
      #update params with the calculated updates
      weights = current_weights + weight_update
      biases = current_biases + bias_update
      #also update momentums
      weight_momentums = weight_update
      bias_momentums = bias_update
      #save:
      list("weights" = weights, "biases" = biases, "lr" = currnt_learn_rate,
           "weight_momentums" = weight_momentums,
           "bias_momentums" = bias_momentums,
           "n_neurons" = n_neurons)
    } else {

      #param updates without momentum (vanilla)
      #calculate updates
      weight_update = -currnt_learn_rate*weight_gradients
      bias_update = -currnt_learn_rate*bias_gradients
      #apply updates
      weights = current_weights + weight_update
      biases = current_biases + bias_update
      #save:
      list("weights" = weights, "biases" = biases, "lr" = currnt_learn_rate,
           "n_neurons" = n_neurons)
    }

  })
## AdaGrad ##
#' @export
#NOTE: must initialize params with cache==TRUE
optimizer_AdaGrad = list(
  update_params = function(layer_forward, layer_backward, #required
                           learning_rate = 1,
                           lr_decay = 0, iteration = 1,
                           epsilon = 1e-7
  ){
    #extract number of neurons from the forward layers output matrix
    n_neurons = ncol(layer_forward$output)

    #current params
    current_weights = layer_forward$weights
    current_biases = layer_forward$biases
    #gradients
    weight_gradients = layer_backward$dweights
    bias_gradients = layer_backward$dbiases

    #learning rate
    currnt_learn_rate = learning_rate * (1 / (1 + lr_decay*iteration))

    #cache
    ##update cache with squared current gradients
    weight_cache = layer_forward$weight_cache + weight_gradients^2
    bias_cache = layer_forward$bias_cache + bias_gradients^2

    #SGD param updates with normalization
    #calculate updates
    weight_update = -currnt_learn_rate*weight_gradients /
      (sqrt(weight_cache) + epsilon)
    bias_update = -currnt_learn_rate*bias_gradients /
      (sqrt(bias_cache) + epsilon)
    #apply updates
    weights = current_weights + weight_update
    biases = current_biases + bias_update
    #save:
    list("weights" = weights, "biases" = biases, "lr" = currnt_learn_rate,
         "weight_cache" = weight_cache, "bias_cache" = bias_cache,
         "n_neurons" = n_neurons)
  } ##these are the new params to be passed on to the layers when loop restarts
)

## RMSProp ##
#' @export
#NOTE: must initialize params with cache==TRUE
optimizer_RMSProp = list(
  update_params = function(layer_forward, layer_backward, #required
                           learning_rate = 0.001,
                           lr_decay = 0, iteration = 1,
                           epsilon = 1e-7,
                           rho = 0.9
  ){
    #extract number of neurons from the forward layers output matrix
    n_neurons = ncol(layer_forward$output)

    #current params
    current_weights = layer_forward$weights
    current_biases = layer_forward$biases
    #gradients
    weight_gradients = layer_backward$dweights
    bias_gradients = layer_backward$dbiases

    #learning rate
    currnt_learn_rate = learning_rate * (1 / (1 + lr_decay*iteration))

    #cache
    ##update cache with squared current gradients
    weight_cache = rho * layer_forward$weight_cache +
      (1-rho) * weight_gradients^2
    bias_cache = rho * layer_forward$bias_cache +
      (1-rho) * bias_gradients^2

    #SGD param updates with normalization
    #calculate updates
    weight_update = -currnt_learn_rate*weight_gradients /
      (sqrt(weight_cache) + epsilon)
    bias_update = -currnt_learn_rate*bias_gradients /
      (sqrt(bias_cache) + epsilon)
    #apply updates
    weights = current_weights + weight_update
    biases = current_biases + bias_update
    #save:
    list("weights" = weights, "biases" = biases, "lr" = currnt_learn_rate,
         "weight_cache" = weight_cache, "bias_cache" = bias_cache,
         "n_neurons" = n_neurons)
  } ##these are the new params to be passed on to layers when the loop restarts
)

## Adam ##
#' @export
#NOTE: must intialize paramters with both cache AND momentum
optimizer_Adam = list(
  update_params = function(layer_forward, layer_backward, #required
                           learning_rate = 0.001,
                           lr_decay = 0, iteration = 1,
                           epsilon = 1e-7,
                           beta_1 = 0.9, beta_2 = 0.999
  ){
    #extract number of neurons from the forward layers output matrix
    n_neurons = ncol(layer_forward$output)

    #current params
    current_weights = layer_forward$weights
    current_biases = layer_forward$biases
    #gradients
    weight_gradients = layer_backward$dweights
    bias_gradients = layer_backward$dbiases

    #learning rate
    currnt_learn_rate = learning_rate * (1 / (1 + lr_decay*iteration))

    #momentums
    #update momentums with current gradients and bias correct
    weight_momentums =
      (beta_1*layer_forward$weight_momentums + (1-beta_1)*weight_gradients ) /
      (1 - (beta_1^iteration )) ##bias correction

    bias_momentums =
      (beta_1*layer_forward$bias_momentums + (1-beta_1)*bias_gradients) /
      (1 - (beta_1^iteration))


    #cache
    #update cache with squared gradients and bias correct
    weight_cache =
      (beta_2*layer_forward$weight_cache + (1-beta_2)*weight_gradients^2) /
      (1 - (beta_2^iteration)) ##bias correction

    bias_cache =
      (beta_2*layer_forward$bias_cache + (1-beta_2)*bias_gradients^2) /
      (1 - (beta_2^iteration))

    #calculate param updates (with momentums, and normalize with cache)
    weight_update = -currnt_learn_rate*weight_momentums /
      (sqrt(weight_cache) + epsilon)
    bias_update = -currnt_learn_rate*bias_momentums /
      (sqrt(bias_cache) + epsilon)

    #apply updates
    weights = current_weights + weight_update
    biases = current_biases + bias_update
    #save:
    list("weights" = weights, "biases" = biases, "lr" = currnt_learn_rate,
         "weight_momentums" = weight_momentums,
         "bias_momentums" = bias_momentums,
         "weight_cache" = weight_cache,
         "bias_cache" = bias_cache,
         "n_neurons" = n_neurons)

  }##these are the updated params to be passed back into the layers
)


## Reminder function ##
#' @export
optimizer_REMINDME = function(
  optimizer = "sgd, adagrad, rmsprop, adam, or general"){
  if(optimizer == "sgd"){
    message("
Momentum is optional (set TRUE or FALSE), cache is not available (set FALSE).
If you choose to set 'momentum_rate' in SGD arguments to use momentum and control the exponential decay, set momentum = TRUE in parameter initialization function.")
  }
  if(optimizer == "adagrad"){
    message("
Initialize parameters with cache=TRUE, momentum=FALSE.")
  }
  if(optimizer == "rmsprop"){
    message("
Initialize parameters with cache=TRUE, momentum=FALSE.
Also set the 'rho' hyperparameter - the cache memory decay rate.")
  }
  if(optimizer == "adam"){
    message("
Initialize parameters with cache=TRUE & momentum=TRUE.
Also set the 'beta_1' and 'beta_2' hyperparameters, the exponential decay rates for the momentums and cache respectively.")
  }
  if(optimizer == "general"){
    message("
'learning_rate' (also known as alpha) determines the proportion of the gradient values used to update the weights & biases. This controls the speed the model learns.
'lr_decay' controls the rate at which that proportion decays (gets smaller) over the course of the epochs.

'epsilon' is a small value meant to prevent any division by zero.

'momentum' is an exponentially decaying moving average of past gradients applied in the update equation so that current updates are informed by past updates.

'cache', or an adaptive learning rate, keeps a history of previous parameter updates to normalize them, effectively reducing the learning rate for parameters receiving large gradients and increasing the learning rate for those receiving small or infrequent updates." )
  }

}
