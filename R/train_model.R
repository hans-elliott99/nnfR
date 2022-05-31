# Train Model

#' @export
train_model = function(model, inputs, y_true, epochs,
                       optimizer = c("adam"),
                       learning_rate=0, lr_decay=0, epsilon=1e-7,...,
                       metric_list=NULL,
                       validation_X=NULL, validation_y=NULL,
                       batch_size=NULL,
                       print_every=100){

  optim_args = list(...)
  if (!is.null(metric_list)){
    ##set metrics list to store metrics
    metrics = data.frame(epoch = 1:epochs)
  }

  # Setup ----
  ##How many unique "base layers" (dense layers)
  baselayers = c()
  for (i in seq_along(model)){
    base_i = model[[i]]$base_layer
    baselayers = c(baselayers, base_i)
  }
  baselayers = unique(baselayers) ##return vector of each baselayer position

  # Set up the layer parameters
  layer_parameters = list()
  for (i in seq_along(baselayers)){
    ##Initialize empty list
    layer_parameters = c(layer_parameters, "layer"=list(NULL))
    names(layer_parameters)[[i]] = paste0("layer",baselayers[i])
  }
  ##the layer params list will be the length of unique baselayers and will
  ##be overwritten with the updated params at the end of the training loop

  ## Derive parameter settings from the selected optimizer
  if (optimizer=="adam"){
    ##requires cache and momentum
    cache_set=TRUE
    momentum_set=TRUE
    ##set default betas if not provided by user in `...`
    if (!exists(x = "beta_1", where = optim_args)){
      warning("Adam utilizes beta_1 and beta_2 hyperparameters which were not provided, and have been set to defaults of 0.9 and 0.999 respectively. Include specific 'beta_1' and 'beta_2' values in the arguments if desired.")
      optim_args$beta_1 = 0.9
      optim_args$beta_2 = 0.999
    }

  } else if (optimizer=="sgd"){
    ##momentum is optional, no cache
    if (exists(x = "momentum_rate", where = optim_args)){
      momentum_set=TRUE
      cache_set=FALSE
    } else {
      momentum_set=FALSE
      cache_set=FALSE
      optim_args$momentum_rate = 0
    }
  } else if (optimizer=="adagrad"){
    ##requires cache, no momentum
    cache_set=TRUE
    momentum_set=FALSE
  } else if (optimizer=="rmsprop"){
    ##requires cache, no momentum
    cache_set=TRUE
    momentum_set=FALSE
    ##requires rho hyperparameter
    if (!exists(x = "rho", where = optim_args)){
      warning("RMSProp utilizes the rho hyperparameter which was not provided, and has been set to a default of 0.9. Include a specific 'rho' value in the arguments if desired.")
      optim_args$rho = 0.9
    }

  } else {
    warning("Please use one of the following strings to select an optimizer: 'sgd', 'adagrad', 'rmsprop', 'adam'. Otherwise, the network will not perform as intended.")
  }

  # Initialize random params for every layer
  for (b in baselayers){
    current_layer = paste0("layer", b)
    n_inputs = model[[current_layer]]$n_inputs

    random_params = init_params(n_inputs = model[[current_layer]]$n_inputs,
                                n_neurons = model[[current_layer]]$n_neurons,
                                momentum = momentum_set,
                                cache = cache_set)

    layer_parameters[[current_layer]] = random_params
    #names(layer_parameters)[[length(layer_parameters)]] = paste0("layer",b)
  }

  # Batch size
  if (!is.null(batch_size)){
    ##determine how many training "steps" will be needed
    train_steps = nrow(inputs) %/% batch_size	 #(integer division)

    ##Since integer division rounds down, there may be some remaining samples
    ##not in a full batch. So we add 1 to include this last mini-batch
    if (train_steps*batch_size < nrow(inputs)){
      train_steps = train_steps + 1
    }
    # Now we repeat this process for the validation data
    if (!is.null(validation_X)){
      validation_steps = nrow(validation_X) %/% batch_size
      if(validation_steps*batch_size < nrow(validation_X)){
        validation_steps = validation_steps + 1
      }
    }
  } else { ##if no batch_size given, train in one batch
    batch_size = nrow(inputs)
    train_steps = 1
    validation_steps = 1
  }

  # Training Loop ----
  for (epoch in 1:epochs){

    step_loss = 0      ##we will accumulate these over the steps
    step_reg_loss = 0
    accumulated_count = 0

    ##set metrics list to store batch metrics (reset each epoch)
    batch_metrics = data.frame(step = 1:train_steps)

    for(step in 1:train_steps){

      #Deal with batches:
      ##if no batch_size given, then train using one step and full dataset
      if (train_steps == 1){
        batch_X = inputs
        batch_y = y_true

        ##otherwise slice a batch
      } else {
        if (step != train_steps){
          batch_X = inputs[((step-1)*batch_size +1):(step*batch_size), ]
          batch_y = y_true[((step-1)*batch_size +1):(step*batch_size) ]
        } else {
          ##for the last step, need to index precisely to the last row
          batch_X = inputs[((step-1)*batch_size +1):(nrow(inputs)), ]
          batch_y = y_true[((step-1)*batch_size +1):(nrow(inputs)) ]
        }
        ## +1 keeps the process from reusing the sample at the end of last batch
      }

      # Forward Pass----
      layers = list() ##container to store layer objects, resets every step/epoch
      ##First layer
      dense1 = layer_dense$forward(inputs = batch_X,
                                   parameters = layer_parameters[[1]], ##the randoms
                                   weight_L1 = model[[1]]$weight_L1,
                                   weight_L2 = model[[1]]$weight_L2,
                                   bias_L1 = model[[1]]$bias_L1,
                                   bias_L2 = model[[1]]$bias_L2
      )
      dropout1 = layer_dropout$forward(input_layer = dense1,
                                       dropout_rate = model[[1]]$dropout_rate)

      ###determine activation function
      activ_fn1 = model[[2]]$class ##second layer is the first activation layer

      if (activ_fn1=="linear"){
        activ1 = activation_Linear$forward(input_layer = dropout1)
      } else if (activ_fn1=="relu"){
        activ1 = activation_ReLU$forward(input_layer = dropout1)
      } else if (activ_fn1=="sigmoid"){
        activ1 = activation_Sigmoid$forward(input_layer = dropout1)
      } else if (activ_fn1=="softmax"){
        warning("Pure Softmax is broken Hans")
        activ1 = activation_Softmax$forward(inputs = dropout1$output)
      } else if (activ_fn1=="softmax_crossentropy"){
        activ1 = activation_loss_SoftmaxCrossEntropy$forward(
          input_layer = dropout1,
          y_true = y_true)
      } else message("Please enter one of the following strings in $add_activation to select an activation function: 'linear', 'relu', 'sigmoid', 'softmax_crossentropy'")

      output1 = activ1

      layer1 = list(dense=dense1,dropout=dropout1, activ=activ1, output=output1)
      layers = c(layers, "layer1" = list(layer1))

      if (length(baselayers) > 1){
        ## Next Layers
        for (b in baselayers[-1]){##for all other base layers (- first element/layer)

          ##index of prior baselayer (the baselayer that comes next in seq, hence +1)
          prior_baselayer = baselayers[which(baselayers==b)-1]
          ##which will serve as the input layer
          input_layer = paste0("layer",prior_baselayer)
          current_layer = paste0("layer",b)

          #prior_baselayer = baselayers[b] ##index of prior baselayer (includ. 1st)
          #input_layer = paste0("layer",prior_baselayer)

          dense_b = layer_dense$forward(inputs = layers[[input_layer]]$output$output,
                                        parameters = layer_parameters[[current_layer]],
                                        weight_L1 = model[[current_layer]]$weight_L1,
                                        weight_L2 = model[[current_layer]]$weight_L2,
                                        bias_L1 = model[[current_layer]]$bias_L1,
                                        bias_L2 = model[[current_layer]]$bias_L2
          )
          dropout_b = layer_dropout$forward(input_layer = dense_b,
                                            dropout_rate = model[[current_layer]]$dropout_rate)

          activ_fn = model[[b+1]]$class
          ##current layer+1 since activation follows dense.

          if (activ_fn=="linear"){
            activ_b = activation_Linear$forward(input_layer = dropout_b)
          } else if (activ_fn=="relu"){
            activ_b = activation_ReLU$forward(input_layer = dropout_b)
          } else if (activ_fn=="sigmoid"){
            activ_b = activation_Sigmoid$forward(input_layer = dropout_b)
          } else if (activ_fn=="softmax"){
            warning("Pure Softmax is broken Hans")
            activ_b = activation_Softmax$forward(inputs = dropout_b$output)
          } else if (activ_fn=="softmax_crossentropy"){
            activ_b = activation_loss_SoftmaxCrossEntropy$forward(
              input_layer = dropout_b,
              y_true = batch_y)
          } else message("Please enter one of the following strings in $add_activation to select an activation function: 'linear', 'relu', 'sigmoid', 'softmax_crossentropy'")

          output_b = activ_b

          layer = list(dense=dense_b,dropout=dropout_b,activ=activ_b,output=output_b)
          layers = c(layers, "layer" = list(layer))
          names(layers)[[length(layers)]] = paste0("layer",b)
        }#end loop
      }

      # Calculate Loss ----
      ##Determine the selected loss function
      loss_fn = model[[length(model)]]$class ##last layer should be a loss layer

      if (loss_fn=="mse"){
        loss_layer = loss_MeanSquaredError$forward(
          y_pred = layers[[length(layers)]]$output$output,
          y_true = batch_y
        )
      } else if (loss_fn=="mae"){
        loss_layer = loss_MeanAbsoluteError$forward(
          y_pred = layers[[length(layers)]]$output$output,
          y_true = batch_y
        )
      } else if (loss_fn=="binary_crossentropy"){
        loss_layer = loss_BinaryCrossentropy$forward(
          y_pred = layers[[length(layers)]]$output$output,
          y_true = batch_y
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
      step_reg_loss = step_reg_loss + reg_loss

      # Calculate any other metrics ----
      ##Determine which metrics to calculate, then calculate and save
      for (metric in metric_list){

        if (metric=="mse"){
          mse = loss_MeanSquaredError$forward(
            y_pred = layers[[length(layers)]]$output$output,
            y_true = batch_y
          )
          metric_i = (mse$sample_losses)
        } else if (metric=="mae"){
          mae = loss_MeanAbsoluteError$forward(
            y_pred = layers[[length(layers)]]$output$output,
            y_true = batch_y
          )
          metric_i = (mae$sample_losses)
        } else if (metric=="accuracy"){ ##depends on if binary or cateogircal task
          if (loss_fn=="binary_crossentropy"){
            pred = (layers[[length(layers)]]$output$output > 0.5) * 1
            ##returns TRUE (1) if true, and FALSE (0) if false
          } else if (loss_fn=="categorical_crossentropy"){
            pred = max.col(layers[[length(layers)]]$output$output,
                           ties.method = "random")
          }
          accuracy = (pred==batch_y)
          metric_i = accuracy
        } else if (metric=="regression_accuracy"){
          reg_pred = layers[[length(layers)]]$output$output
          accuracy_precision = sd(batch_y)/max(batch_y)
          ##count number of "accurate" preictions
          reg_accuracy = (abs(reg_pred - batch_y) < accuracy_precision)
          metric_i = reg_accuracy
        }

        ##add metric to the batch_metrics dataframe (unless not tracking metrics)
        ##accumulate the sums of the sample losses and then calculate epoch-wise
        ##average at the end
        if (!is.null(metric_list)){
          batch_metrics[step, "count"] = length(metric_i)
          batch_metrics[step, paste0(metric)] = sum(metric_i, na.rm = T)
        }
      }#end metrics loop

      # Backward Pass ----
      layers_back = list()

      ## The first backprop layer (the last layer in the training sequence)
      last_layer = paste0("layer",baselayers[length(baselayers)])
      ###First backprop the loss function
      ###Determine which loss function was chosen and backprop it
      loss_fn_last = model[[length(model)]]$class

      if (loss_fn_last=="mse"){
        loss_backprop1 = loss_MeanSquaredError$backward(
          dvalues = layers[[length(layers)]]$output$output,
          loss_layer = loss_layer
        )
      } else if (loss_fn_last=="mae"){
        loss_backprop1 = loss_MeanAbsoluteError$backward(
          dvalues = layers[[length(layers)]]$output$output,
          loss_layer = loss_layer
        )
      } else if (loss_fn_last=="binary_crossentropy"){
        loss_backprop1 = loss_BinaryCrossentropy$backward(
          dvalues = layers[[length(layers)]]$output$output,
          y_true = batch_y
        )
      } else if (loss_fn_last=="categorical_crossentropy"){
        loss_backprop1 = activation_loss_SoftmaxCrossEntropy$backward(
          dvalues = layers[[length(layers)]]$output$output,
          y_true = batch_y
        )
      } else warning("Please select an available loss function when building the model.")

      ###Then backprop the activation function
      activ_fn_last = model[[length(model)-1]]$class ##last layer before loss layer

      if (activ_fn_last=="linear"){
        activ_backprop1 = activation_Linear$backward(d_layer = loss_backprop1)
      } else if (activ_fn_last=="relu"){
        activ_backprop1 = activation_ReLU$backward(d_layer = loss_backprop1,
                                                   layer = layers[[last_layer]]$activ)
      } else if (activ_fn_last=="sigmoid"){
        activ_backprop1 = activation_Sigmoid$backward(d_layer = loss_backprop1,
                                                      layer = layers[[last_layer]]$activ)
      } else if (activ_fn_last=="softmax"){
        warning("pure softmax is incomplete")
      } else if (activ_fn_last=="softmax_crossentropy"){
        activ_backprop1 = loss_backprop1 ##due to combined softmax/crossent fn
      }
      ###Then backprop the dropout layer, IF dropout was used
      if (model[[baselayers[length(baselayers)]]]$dropout_rate == 0){
        #dropout1 = FALSE
        d_layer_todense1=activ_backprop1
      } else {
        #dropout1 = TRUE
        dropout_backprop1 = layer_dropout$backward(
          d_layer = activ_backprop1,
          layer_dropout = layers[[length(layers)]]$dropout
        )
        d_layer_todense1=dropout_backprop1
      }

      ###Finally, backprop the dense layer
      ###Note: the layer_dense backward fn auto-adjusts if regularization is used
      dense_backprop1 = layer_dense$backward(d_layer = d_layer_todense1,
                                             layer = layers[[length(layers)]]$dense)

      layer_b = list(dense_backprop=dense_backprop1)
      layers_back = c(layers_back, "layer" = list(layer_b))
      names(layers_back)[[length(layers_back)]] = ##this is the last baselayer
        paste0("layer",baselayers[length(baselayers)])

      ## Next Layers
      for (b in rev(baselayers[-length(baselayers)])){
        ##for all layers but the last (rev order since moving back through layers)

        ##index of prior baselayer (the baselayer that comes next in seq, hence +1)
        prior_baselayer = baselayers[which(baselayers==b)+1]
        ##which will serve as the input layer
        input_layer = paste0("layer",prior_baselayer)

        current_layer = paste0("layer", b)

        activ_fn = model[[b+1]]$class
        if (activ_fn=="linear"){
          activ_backprop = activation_Linear$backward(
            d_layer = layers_back[[input_layer]]$dense_backprop)
        } else if (activ_fn=="relu"){
          activ_backprop = activation_ReLU$backward(
            d_layer = layers_back[[input_layer]]$dense_backprop,
            layer = layers[[current_layer]]$activ)
        } else if (activ_fn=="sigmoid"){
          activation_Sigmoid$backward(
            d_layer = layers_back[[input_layer]]$dense_backprop,
            layer = layers[[current_layer]]$activ)
        } #else if "softmax" once pure softmax is fixed

        if (model[[baselayers[b]]]$dropout_rate == 0){
          #dropout_rt = FALSE
          d_layer_todense=activ_backprop
        } else {
          #dropout_rt = TRUE
          dropout_backprop = layer_dropout$backward(
            d_layer = activ_backprop,
            layer_dropout = layers[[current_layer]]$dropout
          )
          d_layer_todense=dropout_backprop
        }

        dense_backprop = layer_dense$backward(d_layer = d_layer_todense,
                                              layer = layers[[current_layer]]$dense)
        layer_b = list(dense_backprop=dense_backprop)
        layers_back = c(layers_back, "layer" = list(layer_b))
        names(layers_back)[[length(layers_back)]] = paste0("layer",b)
      }

      # Optimize the Parameters ----
      for (b in baselayers){
        current_layer = paste0("layer", b)

        if (optimizer=="sgd"){
          optimal_params_b = optimizer_SGD$update_params(
            layer_forward = layers[[current_layer]]$dense,
            layer_backward = layers_back[[current_layer]]$dense_backprop,
            learning_rate = learning_rate,
            lr_decay = lr_decay, iteration = epoch,
            momentum_rate = optim_args$momentum_rate
          )
        } else if (optimizer=="adagrad"){
          optimal_params_b = optimizer_AdaGrad$update_params(
            layer_forward = layers[[current_layer]]$dense,
            layer_backward = layers_back[[current_layer]]$dense_backprop,
            learning_rate = learning_rate,
            lr_decay = lr_decay, iteration = epoch,
            epsilon = epsilon
          )
        } else if (optimizer=="rmsprop"){
          optimal_params_b = optimizer_RMSProp$update_params(
            layer_forward = layers[[current_layer]]$dense,
            layer_backward = layers_back[[current_layer]]$dense_backprop,
            learning_rate = learning_rate,
            lr_decay = lr_decay, iteration = epoch,
            epsilon = epsilon,
            rho = optim_args$rho
          )
        } else if (optimizer=="adam"){
          optimal_params_b = optimizer_Adam$update_params(
            layer_forward = layers[[current_layer]]$dense,
            layer_backward = layers_back[[current_layer]]$dense_backprop,
            learning_rate = learning_rate,
            lr_decay = lr_decay, iteration = epoch,
            epsilon = epsilon,
            beta_1 = optim_args$beta_1,
            beta_2 = optim_args$beta_2
          )
        }

        layer_parameters[[current_layer]] = optimal_params_b
      }

      ## Calculate loss for the batch per sample
      ## accumulate over the batches and we will use the average for the epoch
      step_loss = step_loss + sum(loss_layer$sample_losses)
      ##each step, we sum up the loss for all of the samples. we accumulate
      ##this sum over all the steps. thus, to get the average for the epoch
      ##we will divide this accumulated sum by the total # of sample_losses,
      ##which will be equivalent to the total number of inputs
      accumulated_count = accumulated_count + length(loss_layer$sample_losses)
      ##this should be about equal but may differ due to overlapping

    }##end batch steps loop

    # Calculate loss for the epoch
    ##First calculate the average sample loss
    avg_sample_loss = step_loss / accumulated_count
    avg_reg_loss = step_reg_loss / train_steps ##will = 0 if no regularization

    data_loss = avg_sample_loss
    loss = avg_sample_loss + avg_reg_loss

    # Calculate the epoch-wise metrics
    for (metric in metric_list){
      ##sum up the losses/accuracies across the batches
      ##(each column in batch_metrics is a metric, each row a batch step)
      ##divide by the total number of inputs to get the average
      epoch_metric = sum(batch_metrics[[paste0(metric)]]) /
        sum(batch_metrics[["count"]])

      # mean_metric = mean(batch_metrics[[paste0(metric)]])
      metrics[epoch, paste0(metric)] = epoch_metric

    }


    # Validation Steps ----
    # (If batching the inputs, probably want to batch the validation data too)
    # (If validation data is small, this will all happen in one step anyways)
    if(!is.null(validation_X)){
      batch_valid_loss = c()

      for (step in validation_steps){
        if (validation_steps == 1){
          valid_batch_X = validation_X
          valid_batch_y = validation_y

          ##otherwise slice a batch
        } else {
          if (step != validation_steps){
            valid_batch_X =
              validation_X[((step-1)*batch_size +1):(step*batch_size), ]
            valid_batch_y =
              validation_y[((step-1)*batch_size +1):(step*batch_size) ]
          } else {
            ##for the last step, need to index precisely to the last row
            valid_batch_X =
              validation_X[((step-1)*batch_size +1):(nrow(validation_X)), ]
            valid_batch_y =
              validation_y[((step-1)*batch_size +1):(nrow(validation_X)) ]
          }
        }

        ###Utilize the test_model() fn to test on validation data, if provided
        current_fit = list("parameters" = layer_parameters)

        if (is.null(validation_y)){
          warning("provide both validation_X and validation_y")
        } else{
          validation = test_model(model = model,
                                  trained_model = current_fit,
                                  X_test = valid_batch_X,
                                  y_test = valid_batch_y,
                                  metric_list = metric_list)
          valid_loss_i = validation$loss
          ##includes regularization if used
        }
        batch_valid_loss = c(batch_valid_loss, valid_loss_i)

      } #end validation steps loop

      validation_loss = mean(batch_valid_loss)

    }

    # Status Report----
    if (epoch == 1){
      report_head = c("epoch", "loss")
      if (!is.null(validation_X)) report_head = c(report_head, "validation_loss")
      report_head = c(report_head, metric_list)
      print(report_head)
    }
    if (epoch %in% seq(0,epochs,by=print_every)){
      report = c(epoch, loss)
      if (!is.null(validation_X)) report = c(report, validation_loss)
      for (metric in metric_list){
        value = metrics[epoch, paste0(metric)]
        report = c(report, value)
      }


      report = sapply(report, round, 7)
      print(report)
      #report_history = rbind(report_history, report)
    }

    ## Send the losses to the metrics dataframe for tracking
    metrics[epoch, "loss"] = loss
    metrics[epoch, "data_loss"] = data_loss
    if (!is.null(validation_X)){
      metrics[epoch, "validation_loss"] = validation_loss
    }

  }##end epoch loop

  ##Final Output----
  final_metrics = list()
  validation_metrics = list()
  final_metrics[["loss"]] = loss
  final_metrics[["data_loss"]] = data_loss
  if (!is.null(validation_X)){
    final_metrics[["validation_loss"]] = validation_loss
  }
  for (metric in metric_list){
    final_metrics[[paste0(metric)]] = metrics[epochs, paste0(metric)]
    if (!is.null(validation_X)){
      validation_metrics[[paste0(metric)]] =
        validation$metrics[[paste0(metric)]]
    }
  }

  ##Final Predictions
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


  ##Save final model parameters
  parameters = list()
  for (b in baselayers){
    params_b = layer_parameters[[paste0("layer",b)]]
    parameters = c(parameters, "layer"=list(params_b))
    names(parameters)[[length(parameters)]] = paste0("layer",b)
  }

  ##Returns
  list(final_metrics=final_metrics, validation_metrics=validation_metrics,
       metrics=metrics,
       predictions=predictions, raw_predictions=raw_predictions,
       parameters=parameters)
}
