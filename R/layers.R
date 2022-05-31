# Dense Layer -----------------------------------------------------------------

#' @export
layer_dense = list(
  ## FORWARD PASS
  forward = function(
    inputs,
    parameters, ## from initialize_parameters
    weight_L1 = 0, weight_L2 = 0,  ##regularization
    bias_L1 = 0, bias_L2 = 0
  ){

    if(is.matrix(inputs) == FALSE ) message("Convert inputs to matrix first")

    n_inputs = ncol(inputs)
    n_neurons = parameters$n_neurons
    weights = parameters$weights
    biases = parameters$biases
    #Forward Pass
    output = inputs%*%weights + biases[col(inputs%*%weights)]

    #Regularization values:
    regularization = list("weight_L1" = weight_L1, "weight_L2" = weight_L2,
                          "bias_L1" = bias_L1, "bias_L2" = bias_L2)

    #SAVING:
    #then layer saves momentum only
    if (exists(x = "weight_momentums", where = parameters) &
        !exists(x = "weight_cache", where = parameters)){
      list("output" = output, ##for forward pass
           "inputs" = inputs, "weights"= weights, "biases" = biases, ##for backprop
           "weight_momentums"=parameters$weight_momentums, ##for momentum
           "bias_momentums"=parameters$bias_momentums,
           "regularization" = regularization)  ##for regularization
      #if momentum==FALSE & cache==TRUE, saves cache only
    } else if (!exists(x = "weight_momentums", where = parameters) &
               exists(x = "weight_cache", where = parameters)){
      list("output" = output, ##for forward pass
           "inputs" = inputs, "weights"= weights, "biases" = biases, ##for backprop
           "weight_cache"=parameters$weight_cache, ##for cache
           "bias_cache"=parameters$bias_cache,
           "regularization" = regularization)

      #if momentum==TRUE & cache==TRUE, saves both
    } else if (exists(x = "weight_momentums", where = parameters) &
               exists(x = "weight_cache", where = parameters)){
      list("output" = output, ##for forward pass
           "inputs" = inputs, "weights"= weights, "biases" = biases, ##for backprop
           "weight_momentums"=parameters$weight_momentums, ##for momentum
           "bias_momentums"=parameters$bias_momentums,
           "weight_cache"=parameters$weight_cache, ##for cache
           "bias_cache"=parameters$bias_cache,
           "regularization" = regularization)

      #else both==FALSE, ignore momentum & cache
    } else {
      #otherwise, just save
      list("output" = output, ##for forward pass
           "inputs" = inputs, "weights"= weights, "biases" = biases, ##for backprop
           "regularization" = regularization)
    }

  },#end fwd

  # BACKWARD
  backward = function(d_layer="layer object that occurs prior in backward pass",
                      layer="the layer object from the forward pass"){

    dvalues = d_layer$dinputs
    #Gradients on parameters
    dweights = t(layer$inputs)%*%dvalues
    dbiases = colSums(dvalues)

    #Gradients on regularization
    ##regularization hyperparams:
    layer_reg = layer$regularization ##a list of the set lambda values

    ##L1 Weights##
    if (layer_reg$weight_L1 > 0){
      dL1 = matrix(1, nrow = nrow(layer$weights), ncol = ncol(layer$weights))
      #make matrix filled with 1s
      dL1[layer$weight < 0] = -1
      #convert matrix value to -1 where weight is less than zero
      dweights = dweights + layer_reg$weight_L1 * dL1
    }
    ##L2 Weights##
    if (layer_reg$weight_L2 > 0){
      dweights = dweights + 2 * layer_reg$weight_L2 * layer$weights
    }                     #2 * lambda * weights

    ##L1 Biases##
    if (layer_reg$bias_L1 > 0){
      dL1 = matrix(1, nrow = nrow(layer$biases), ncol = ncol(layer$biases))
      #make matrix filled with 1s
      dL1[layer$bias < 0] = -1
      #convert matrix value to -1 where weight is less than zero
      dbiases = dbiases + layer_reg$bias_L1 * dL1
    }
    ##L2 Biases##
    if (layer_reg$bias_L2 > 0){
      dbiases = dbiases + 2 * layer_reg$bias_L2 * layer$biases
    }

    #Gradients on values
    dinputs = dvalues%*%t(layer$weights)

    #saves:
    list("dinputs"=dinputs,
         "dweights"=dweights,
         "dbiases"=dbiases)
  }#end bwd
)



### Dropout Layer -------------------------------------------------------------

#' @export
layer_dropout = list(
  ## FORWARD PASS
  forward = function(
    input_layer, ##layer to apply dropout to
    dropout_rate ##rate of neuron deactivation
  ) {
    inputs = input_layer$output   ##the outputs from the previous layer

    #Dropout mask/filter
    dropout_filter = matrix(data =
                              rbinom(n = nrow(inputs)*ncol(inputs),
                                     size = 1,
                                     p = (1-dropout_rate)),
                            nrow = nrow(inputs),
                            ncol = ncol(inputs)) /
      (1 - dropout_rate)
    ##Creates matrix that is shape of the input layer's output (from nrow, ncol)
    ##and fills it with 1s and 0s from "Bernoulli". The length of the rbinom
    ##output is equal to nrow*ncol so it fills the input layer's shape.
    ##We also apply the scaling step to the filter directly since it makes the
    ##backprop step even simpler.

    ##Apply mask to inputs and scale by (1 - dropout_rate)
    output = inputs * dropout_filter

    list("output" = output, "dropout_filter" = dropout_filter)

  },
  ## BACKWARD PASS
  backward = function(
    d_layer = "layer object that occurs prior in backward pass",
    #derivative values being passed back from next layer
    layer_dropout = "the layer object from the forward pass"
    ##the forward-pass dropout layer object
  ){
    dvalues = d_layer$dinputs ##extract the derivative object from the layer
    dinputs = dvalues * layer_dropout$dropout_filter
    ##Thus if the filter is 0 at the given index, the derivative is 0
    ##And if the filter is 1/(1-dropout_rate) at the given index,
    ##the derivative is dvalues * 1/(1-dropout_rate)
    list("dinputs" = dinputs)
  }

)#end layer_dropout
