### Activation Functions ------------------------------------------------------
## Linear ##
#' @export
activation_Linear = list(
  forward = function(input_layer){
    output = input_layer$output
    list("output" = output)
  },

  backward = function(d_layer){
    dinputs = d_layer$dinputs
    list("dinputs" = dinputs)
    #derivative = 1, so 1 * dvalues = dvalues. dvalues correspond to the
    #previously backpropagated layer's dinputs
  }
)


## ReLU ##
#' @export
activation_ReLU = list(
  #FORWARD PASS
  forward = function(input_layer){

    inputs = input_layer$output
    output = matrix(sapply(X = inputs,
                           function(X){max(c(0, X))}
    ),
    nrow = nrow(inputs), ncol = ncol(inputs))
    #ReLU function coerced into a matrix so the shape
    #is maintained (it will be equivalent to that of the input shape)

    #Function saves:
    list("output" = output, "inputs" = inputs)
  },
  #BACKWARD PASS
  backward = function(d_layer, layer){

    inputs = layer$inputs
    dinputs = d_layer$dinputs ##the dinputs from the next layer

    dinputs[inputs <= 0] = 0
    #save:
    list("dinputs"=dinputs)
  }
)

## SoftMax ##
#' @export
activation_Softmax = list(
  forward = function(inputs){
    #scale inputs
    max_value = apply(X = inputs, MARGIN = 2,  FUN = max)
    scaled_inputs = sapply(X = 1:ncol(inputs),
                           FUN = function(X){
                             inputs[,X] - abs(max_value[X])})

    # exponetiate
    exp_values = as.matrix(exp(scaled_inputs))
    # normalize
    norm_base = matrix(rowSums(exp_values),
                       nrow = nrow(inputs), ncol = 1)
    probabilities = sapply(X = 1:nrow(inputs),
                           FUN = function(X){exp_values[X,]/norm_base[X,]})
    return(t(probabilities))
    #(transpose probabilities)
  },
  backward = function(softmax_output, dvalues){
    # Create zero array the shape of the dvalues
    dinputs = matrix(0, nrow=nrow(dvalues), ncol=ncol(dvalues))
    # Enumerate sample-wise (rows) through outputs and gradients
    for (i in 1:nrow(softmax_output)){
      ##ensure each row is now a column vector with matrix()
      single_output = matrix(softmax_output[i,])
      single_dvalues = matrix(dvalues[i,])

      # Calculate jacobian matrix of output
      ##Note: convert single_output to simple row vector with c() for diag to work
      jacobian_matrix = diag(c(single_output)) - (single_output %*% t(single_output))
      # Calculate sample-wise gradient & add to array of gradients (as a row)
      dinputs[i,] = c(jacobian_matrix %*% single_dvalues)
    }
    return(dinputs)
  }
)

## Sigmoid ##
#' @export
activation_Sigmoid = list(
  forward = function(input_layer){
    inputs = input_layer$output
    sigmoid = 1 / (1 + exp(-inputs))
    list("output" = sigmoid)
  },
  backward = function(d_layer = "the layer object being passed back",
                      layer = "the sigmoid layer obj from the forward pass"
  ){
    dvalues = d_layer$dinputs ##the values being passed back
    output = layer$output ##the output from the forward-pass
    dinputs = dvalues * (1 - output) * output
    list("dinputs" = dinputs)
  }
)

## Softmax X Cross Entropy ##
#' @export
activation_loss_SoftmaxCrossEntropy = list(
  #FORWARD PASS
  forward = function(input_layer, y_true){

    inputs = input_layer$output
    #output layer's activation function
    softmax_out = activation_Softmax$forward(inputs)
    #calculate loss (unless this were in a prediction phase)
    if (!is.null(y_true)){
      sample_losses = Categorical_CrossEntropy$forward(softmax_out, y_true)
    } else sample_losses = NULL
    #function saves:
    list("output"=softmax_out, "sample_losses"=sample_losses)
  },
  #BACKWARD PASS
  backward = function(dvalues, y_true){

    #Detect number of samples
    if (is.vector(dvalues)) {      #if one sample
      samples = 1
    } else if (is.array(dvalues)) {  #else if multiple samples
      samples = nrow(dvalues)
    } else message("error checking shape of inputs")

    #Reverse One-Hot Encoding
    #if labels are one-hot encoded, turn them into discrete values
    ##helper function
    anti_ohe = function(y_true){
      unique_classes = ncol(y_true)
      samples = nrow(y_true)
      y_true_vec = as.vector(y_true)

      class_key = rep(1:unique_classes, each = samples)
      y_true = class_key[y_true_vec==1]
      #selects the classes that correspond to 1s in y_true vector
      return(y_true)
    }
    ##check & modify
    y_true = if(is.array(y_true)){ #if one-hot encoded
      #change to sparse
      anti_ohe(y_true)
    } else y_true

    #Calculate gradient
    #Copy so we can modify
    dinputs = dvalues
    #Calculate gradient
    #index the prediction array with the sample number and its
    #true value index, subtracting 1 from these values. Requires discrete,
    #not one-hot encoded, true labels (explaining the need for the above step)
    dinputs[cbind(1:samples, y_true)] = dinputs[cbind(1:samples, y_true)] - 1
    #Normalize gradient
    dinputs = dinputs/samples
    #save desired outputs
    list("dinputs" = dinputs, "samples" = samples, "y_true" = y_true)
  }
)
