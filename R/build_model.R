# Model Building for Higher Level Training
## library(devtools); library(roxygen2)
## (devtools::document() to update documentation)

#' Initialize Model
#'
#' This function initializes an empty model object to which layers can be added.
#' A model must be initialized before any layers are added. Layers must then be
#' added sequentially like so: every dense layer (`nnfR::add_dense()`) is
#' followed by an activation layer (`nnfR::add_activation()`),
#' and the last layer in the model is a loss function (`nnfR::add_loss()`).
#'
#' @param none no arguments/parameters
#' @return An empty list object to which layers can be added.
#'
#' @export
initialize_model = function(){
  # Create an empty list for network objects
  model = list()
  return(model)
}

#' Add Dense Layer
#'
#' Add a dense layer to the model. Specify the correct number of inputs and the
#' desired number of neurons, and additional parameters as desired.
#'
#' @param model The model object to which this layer will be added.
#' @param n_inputs The number of inputs this layer will receive. If the first layer,
#'      this should be equal to the number of samples fed into the neural net. If
#'      a subsequent hidden layer, this should be equal to the number of neurons set
#'      in the last dense layer.
#' @param n_neurons The desire number of neurons or nodes for the layer to contain.
#' @param dropout_rate The rate of dropout. Corresponds to the proportion of neurons
#'     in the layer that will be randomly disabled.
#' @param weight_L1 L1 regularization for the weights in this layer.
#' @param weight_L2 L2 regularization for the weights in this layer.
#' @param bias_L1 L1 regularization for the biases in this layer.
#' @param bias_L2 L2 regularization for the biases in this layer.
#'
#' @export
add_dense = function(model, n_inputs, n_neurons, dropout_rate = 0,
                     weight_L1=0, weight_L2=0, bias_L1=0, bias_L2=0){
  pos = length(model) + 1 ##the layer's position in sequence of layers

  dense_args = list(n_inputs=n_inputs, n_neurons=n_neurons,
                    dropout_rate = dropout_rate,
                    weight_L1=weight_L1, weight_L2=weight_L2,
                    bias_L1=bias_L1, bias_L2=bias_L2,
                    class = "layer_dense", pos = pos)

  #layer_n = length(model) + 1 ##determine the layer's sequence number
  #layer_name = deparse(substitute(layer)) ##extract string with layer name

  # Add layer objects to the model
  model = c(model, "layer" = list(dense_args))
  # Give the layer a unique name corresponding to position in sequence
  names(model)[[pos]] = paste0("layer", pos)
  #names(model)[['new']] = layer_name ##rename the layer object appropriately
  return(model)

}

#' Add Activation Layer
#'
#' Add an activation layer to the model. This layer must directly follow a dense
#' layer, and must be directly followed by either another dense layer, or the loss
#' function.
#'
#' @param model The model object to which this layer will be added.
#' @param activ_fn The name of the activation function which will be applied to
#'     the output of the previous dense layer. One of "linear", "relu", "sigmoid",
#'     "softmax_crossentropy". If a softmax activation function is to be used
#'     with the categorical crossentropy loss function, use softmax_crossentropy.
#'
#' @export
add_activation =  function(model, activ_fn = c("linear", "relu", "sigmoid",
                                               "softmax_crossentropy")){
  class = activ_fn
  pos = length(model) + 1
  ##We want to tie the activation function back to the dense layer
  ##its applied upon. This depends on if dropout was used in between
  base_layer = pos-1
  activ_args = list(class=class, pos=pos, base_layer=base_layer)
  model = c(model, "layer" = list(activ_args))
  names(model)[[pos]] = paste0("layer",pos)

  return(model)
}

#' Add Loss Layer
#'
#' Add a loss function to the model. This should be the last layer specified in
#' the sequential list of layers built on top of `nnfR::initialize_model()`. The
#' previous layer must be an activation layer (`nnfR::add_activation()`).
#'
#' @param model The model object to which this layer will be added.
#' @param loss_fn The name of the loss function which will be used as the objective
#'     function in the training of the neural network. One of "mse" for Mean Squared Error,
#'     "mae" for Mean Absolute Error, "binary_crossentropy", or "categorical_crossentropy".
#'
#' @export
add_loss = function(model,
                    loss_fn=c("mse", "mae",
                              "binary_crossentropy",
                              "categorical_crossentropy")){

  pos = length(model) + 1
  loss_args = list(class=loss_fn, pos=pos)
  model = c(model, "layer" = list(loss_args))
  names(model)[[pos]] = paste0("layer",pos)

  return(model)
}
