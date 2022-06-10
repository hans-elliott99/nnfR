#' Test Model
#'
#' Make training or testing predictions with a fitted model. This function can be
#' used to get predictions from a model after it is trained, or to make final
#' predictions on test data.
#'
#' @param model A model object built ontop of the `nnfR::initialize_model()` function
#' @param trained_model The trained/fitted model object created from the
#'      `nnfR::train_model()` function.
#' @param X_test the X inputs (the features, without the outcome variable/y/labels)
#' @param y_test Optional. The y outputs corresponding to the X inputs. These can be
#'     provided if you want to generate final loss and metrics from training data. If
#'     excluded, this function will simply make predictions from the X inputs.
#' @param metric_list An optional string or list of strings containing the names
#'     of metrics to track and calculate in addition to the chosen loss function.
#'     May include one or more of "mse", "mae", "accuracy", or "regression accuracy".
#'
#' @export
test_model = function(model, trained_model,
                      X_test, y_test=NULL,
                      metric_list=NULL){
  inputs = X_test
  y_true = y_test
  ##How many baselayers were used
  baselayers = c()
  for (i in seq_along(model)){
    base_i = model[[i]]$base_layer
    baselayers = c(baselayers, base_i)
  }
  baselayers = unique(baselayers) ##return vector of each baselayer position

  # Forward Pass----
  layers = list()
  ##First layer
  dense1 = layer_dense$forward(inputs = inputs,
                               parameters = trained_model$parameters[[1]],
                               weight_L1 = model[[1]]$weight_L1,
                               weight_L2 = model[[1]]$weight_L2,
                               bias_L1 = model[[1]]$bias_L1,
                               bias_L2 = model[[1]]$bias_L2
  )
  ###determine activation function
  activ_fn1 = model[[2]]$class ##second layer is the first activation layer

  if (activ_fn1=="linear"){
    activ1 = activation_Linear$forward(input_layer = dense1)
  } else if (activ_fn1=="relu"){
    activ1 = activation_ReLU$forward(input_layer = dense1)
  } else if (activ_fn1=="sigmoid"){
    activ1 = activation_Sigmoid$forward(input_layer = dense1)
  } else if (activ_fn1=="softmax"){
    warning("Pure Softmax is broken Hans")
    activ1 = activation_Softmax$forward(inputs = dense1$output)
  } else if (activ_fn1=="softmax_crossentropy"){
    activ1 = activation_loss_SoftmaxCrossEntropy$forward(
      input_layer = dense1,
      y_true = y_true)
  }
  output1 = activ1

  layer1 = list(dense=dense1, activ=activ1, output=output1)
  layers = c(layers, "layer1" = list(layer1))

  if (length(baselayers) > 1){
    # Next Layers
    for (b in baselayers[-1]){##for all other base layers (- first element/layer)

      ##index of prior baselayer (the baselayer that comes next in seq, hence +1)
      prior_baselayer = baselayers[which(baselayers==b)-1]
      ##which will serve as the input layer
      input_layer = paste0("layer",prior_baselayer)
      current_layer = paste0("layer",b)

      dense_b = layer_dense$forward(inputs = layers[[input_layer]]$output$output,
                                    parameters = trained_model$parameters[[current_layer]],
                                    weight_L1 = model[[current_layer]]$weight_L1,
                                    weight_L2 = model[[current_layer]]$weight_L2,
                                    bias_L1 = model[[current_layer]]$bias_L1,
                                    bias_L2 = model[[current_layer]]$bias_L2
      )

      activ_fn = model[[b+1]]$class
      ##current layer+1 since activation follows dense.

      if (activ_fn=="linear"){
        activ_b = activation_Linear$forward(input_layer = dense_b)
      } else if (activ_fn=="relu"){
        activ_b = activation_ReLU$forward(input_layer = dense_b)
      } else if (activ_fn=="sigmoid"){
        activ_b = activation_Sigmoid$forward(input_layer = dense_b)
      } else if (activ_fn=="softmax"){
        warning("Pure Softmax is broken Hans")
        activ_b = activation_Softmax$forward(inputs = dense_b$output)
      } else if (activ_fn=="softmax_crossentropy"){
        activ_b = activation_loss_SoftmaxCrossEntropy$forward(
          input_layer = dense_b,
          y_true = y_true)
      }
      output_b = activ_b

      layer = list(dense=dense_b,activ=activ_b,output=output_b)
      layers = c(layers, "layer" = list(layer))
      names(layers)[[length(layers)]] = paste0("layer",b)
    }#end loop
  }

  # Calculate Loss Layer if y_true is given
  ##Determine the selected loss function
  loss_fn = model[[length(model)]]$class ##last layer should be a loss layer

  if(!is.null(y_true)){
    if (loss_fn=="mse"){
      loss_layer = loss_MeanSquaredError$forward(
        y_pred = layers[[length(layers)]]$output$output,
        y_true = y_true
      )
    } else if (loss_fn=="mae"){
      loss_layer = loss_MeanAbsoluteError$forward(
        y_pred = layers[[length(layers)]]$output$output,
        y_true = y_true
      )
    } else if (loss_fn=="binary_crossentropy"){
      loss_layer = loss_BinaryCrossentropy$forward(
        y_pred = layers[[length(layers)]]$output$output,
        y_true = y_true
      )
    } else if (loss_fn=="categorical_crossentropy"){
      loss_layer = layers[[length(layers)]]$output
    }

    # Calculate Regularized Loss
    ##If regularization is not used, this code will still run but reg_loss
    ##will equal zero.
    layers_reg_loss = c()
    for (b in baselayers){
      current_layer = paste0("layer",b)
      reg_loss_b = regularization_loss(layers[[current_layer]]$dense)
      layers_reg_loss = c(layers_reg_loss, reg_loss_b)
    }
    reg_loss = sum(layers_reg_loss)

    ## Calculate final loss----
    data_loss = mean(loss_layer$sample_losses)
    loss = data_loss + reg_loss ##loss = data_loss if regularization not used

    ## Calculate other metrics
    metrics = list()
    for (metric in metric_list){
      if (metric=="mse"){
        mse = loss_MeanSquaredError$forward(
          y_pred = layers[[length(layers)]]$output$output,
          y_true = y_true
        )
        metric_i = mean(mse$sample_losses)
      } else if (metric=="mae"){
        mae = loss_MeanAbsoluteError$forward(
          y_pred = layers[[length(layers)]]$output$output,
          y_true = y_true
        )
        metric_i = mean(mae$sample_losses)
      } else if (metric=="accuracy"){ ##depends on if binary or cateogircal task
        if (loss_fn=="binary_crossentropy"){
          pred = (layers[[length(layers)]]$output$output > 0.5) * 1
          ##returns TRUE (1) if true, and FALSE (0) if false
        } else if (loss_fn=="categorical_crossentropy"){
          pred = max.col(layers[[length(layers)]]$output$output,
                         ties.method = "random")
        }
        accuracy = mean(pred==y_true, na.rm = T)
        metric_i = accuracy
      } else if (metric=="regression_accuracy"){
        reg_pred = layers[[length(layers)]]$output$output
        accuracy_precision = sd(y_true)/max(y_true)
        reg_accuracy = mean(abs(reg_pred - y_true) < accuracy_precision)
        metric_i = reg_accuracy
      }

      ##add metric to the metrics list (unless don't want to track them)
      metrics = c(metrics, "m" = list(metric_i))
      names(metrics)[[length(metrics)]] = paste0(metric)

    }#end metrics loop
  } else { ##if y_true is not given, then no loss/metrics calculated
    loss = NULL
    data_loss = NULL
    metrics = NULL
  }

  ## Final Predictions
  raw_predictions = layers[[length(layers)]]$output$output
  if (loss_fn=="categorical_crossentropy"){
    predictions = max.col(raw_predictions, ties.method = "random")
  } else if (loss_fn=="binary_crossentropy"){
    predictions = (raw_predictions > 0.5) * 1
  }
  else {
    if (ncol(raw_predictions) > 1){
      ##if using multiple output neurons for a regression or binary problem...
      ##this is temporary, may want to change
      predictions = rowMeans(raw_predictions)
    } else predictions = raw_predictions
  }


  ##Output:
  if(!is.null(y_true)){
    list(predictions=predictions, raw_predictions = raw_predictions,
         loss=loss, data_loss=data_loss,
         metrics=metrics)
  } else {
    list(predictions=predictions, raw_predictions = raw_predictions)
  }

}
