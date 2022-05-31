# Model Building for Higher Level Training
## library(devtools); library(roxygen2)
## (devtools::document() to update documentation)

#' Initialize Model
#'
#' This function initializes an empty model object to which layers can be added.
#' A model must be initialized before any layers are added.
#'
#' @param none no arguments/parameters
#' @return An empty list object to which layers can be added.
#' @export
initialize_model = function(){
  # Create an empty list for network objects
  model = list()
  return(model)
}

## Add Dense Layer
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

##Add Activation Layer
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

##Add Loss Layer
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
