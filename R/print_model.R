#' Print model information
#'
#' After specifying the structure of a model, print it in a nice format to
#' visually inspect.
#'
#' @param model A model built on top of nnfR::initialize_model(), which is a list
#'     of layer objects. Not a trained/fitted model.
#'
#' @export
print_model = function(model){
  ##How many unique "base layers" (dense layers)
  baselayers = c()
  for (i in seq_along(model)){
    base_i = model[[i]]$base_layer
    baselayers = c(baselayers, base_i)
  }
  baselayers = unique(baselayers) ##return vector of each baselayer position

  #Extract information from each layer and format
  model_print = data.frame()
  for (b in baselayers){
    layer_b = c(
      inputs = model[[b]]$n_inputs,
      neurons = model[[b]]$n_neurons,
      weights_L1 = model[[b]]$weight_L1,
      weights_L2 = model[[b]]$weight_L2,
      biases_L1 = model[[b]]$bias_L1,
      biases_L2 = model[[b]]$bias_L2,
      dropout = model[[b]]$dropout_rate,
      activation = model[[b+1]]$class
    )
    model_print = rbind(model_print, layer_b)
  }
  model_print = cbind(layer = 1:length(baselayers), model_print)
  colnames(model_print) = c("layer","inputs","neurons","weights_L1",
                            "weights_L2","biases_L1","biases_L2",
                            "dropout","activation")
  #Last layer is a loss layer
  loss = model[[length(model)]]$class

  list("network"=model_print, "loss"=loss)
}
