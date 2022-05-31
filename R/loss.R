### Loss ----------------------------------------------------------------------
## MSE ##
#' @export
loss_MeanSquaredError = list(
  forward = function(y_pred, y_true){
    ##calculate MSE for each sample (row)
    sample_losses = rowMeans( (y_true - y_pred)^2 )
    list("sample_losses" = sample_losses, "y_true" = y_true)
  },
  backward = function(dvalues = "linear activation fn output",
                      loss_layer = "loss layer object from forward pass"){

    y_true = loss_layer$y_true
    #number of samples
    samples = nrow(dvalues)
    #number of output neurons
    n_outputs = ncol(dvalues)

    #Gradient on values (dvalues = y_pred)
    dinputs = -2 * (y_true - dvalues) / n_outputs

    #Normalize
    dinputs = dinputs / samples

    list("dinputs"=dinputs)
  }
)

## MAE ##
#' @export
loss_MeanAbsoluteError = list(
  forward = function(y_pred, y_true){
    ##calculate MAE for each sample (row)
    sample_losses = rowMeans( abs(y_true - y_pred) )
    list("sample_losses" = sample_losses, "y_true" = y_true)
  },
  backward = function(dvalues = "linear activation fn output",
                      loss_layer = "loss layer object from forward pass"){
    y_true = loss_layer$y_true
    #number of samples
    samples = nrow(dvalues)
    #number of output neurons
    n_outputs = ncol(dvalues)

    #calculate gradient (sign returns 1 for values >0, -1 for values <0)
    dinputs = sign(y_true - dvalues) / n_outputs
    ##dvalues = y_pred
    #normalize gradients
    dinputs = dinputs / samples

    list("dinputs"=dinputs)
  }
)

## Binary Crossentropy ##
#' @export
loss_BinaryCrossentropy = list(
  forward = function(y_pred, y_true){
    #clip data to prevent division by zero (both sides to keep mean unbiased)
    y_pred_clipped = ifelse(y_pred >= 1-1e-7, 1-1e-7,
                            ifelse(y_pred <= 1e-7, 1e-7, y_pred))

    #calculate sample-wise losse per neuron
    sample_losses = -(y_true*log(y_pred_clipped) +
                        (1 - y_true)*log(1 - y_pred_clipped)
    )
    #calculate total (mean) loss for each sample
    ## (mean across neurons/vector of outputs, neurons = cols, samples = rows)
    sample_losses = rowMeans(sample_losses)

    #calculate mean loss across the entire batch
    data_loss = mean(sample_losses)

    list("sample_losses" = sample_losses, "data_loss" = data_loss)
  },
  backward = function(dvalues = "sigmoid output", y_true){

    #number of samples
    samples = nrow(dvalues)
    #number of output neurons
    n_outputs = ncol(dvalues)

    #clip data to prevent divide by zero
    clipped_dvalues = ifelse(dvalues >= 1-1e-7, 1-1e-7,
                             ifelse(dvalues <= 1e-7, 1e-7, dvalues))
    #calculate gradients
    dinputs = -(y_true / clipped_dvalues -
                  (1 - y_true) / (1 - clipped_dvalues)
    ) / n_outputs

    #normalize gradient
    dinputs = dinputs/samples
    ##We have to perform this normalization since each output returns its own
    ##derivative, and without normalization, each additional input will increase
    ##the gradients mechanically
    list("dinputs" = dinputs)
  }
)


## Categorical Cross Entropy ##
#' @export
Categorical_CrossEntropy = list(
  #FORWARD PASS
  forward = function(y_pred = "softmax output", y_true = "target labels"){

    #DETECT NUMBER OF SAMPLES
    samples = length(y_true)

    #CLIP SAMPLES TO AVOID -Inf ERROR
    y_pred_clipped = ifelse(y_pred <= 1e-7, 1e-7,
                            ifelse(y_pred >= (1-1e-7), (1-1e-7), y_pred))

    if (nrow(t(y_true)) == 1){ ##if y_true is a vector of labels (sparse)
      if (min(y_true) == 0){ #check if the first label is 0.
        y_true = y_true + 1  #add 1 to use y_true as index
      }
      confidences = y_pred_clipped[cbind(1:samples, y_true)]
    } else if (nrow(y_true) > 1){ #else if y_true is one-hot encoded
      confidences = rowSums(y_pred_clipped * y_true)
    } else warning("error indexing predicted class confidences [cat crossent]")

    #CALC LOSS FOR EACH SAMPLE (ROW)
    sample_losses = -log(confidences)
    return(sample_losses)

  },
  #BACKWARD PASS
  backward = function(y_true, d_layer){
    dvalues = d_layer$dinputs

    #number of samples
    samples = length(dvalues)

    #number of labels
    labels = length(unique(dvalues[1,]))

    #if labels are sparse, turn them into one-hot encoded vector
    y_true = ifelse(#if
      nrow(t(y_true)) ==1,
      #one-hot-encode
      y_true = do.call(rbind,
                       lapply(X = y_true,
                              function(X) as.integer(
                                !is.na(match(unique(
                                  unlist(y_true)
                                ), X)
                                ))
                       )),
      #else
      y_true)

    #calculate gradient
    dinputs = -y_true/dvalues
    #normalize gradient
    dinputs = dinputs/samples
    list("dinputs" = dinputs)
  }
)


## Regularization ##
#' @export
regularization_loss = function(layer){
  #Regularization hyperparams:
  layer_reg = layer$regularization ##a list of the set lambda values

  #L1-regularization: weights
  l1_weight_apply = layer_reg$weight_L1 * sum(abs(layer$weights))
  #L1-regularization: bias
  l1_bias_apply = layer_reg$bias_L1 * sum(abs(layer$biases))

  #L2-regularization: weights
  l2_weight_apply = layer_reg$weight_L2 * sum(layer$weights^2)
  #L2-regularization: biases
  l2_bias_apply = layer_reg$bias_L2 * sum(layer$biases^2)

  #Overall regularization loss
  reg_loss = l1_weight_apply + l1_bias_apply + l2_weight_apply + l2_bias_apply
  #save:
  return(reg_loss)
}
