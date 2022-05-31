# Initialize Layer Parameters----


#' @export
init_params = function(n_inputs = "# of features",
                       n_neurons = "desired # of neurons",
                       momentum = FALSE,
                       cache = FALSE){

  weights = matrix(data = (0.1 * rnorm(n = n_inputs*n_neurons)),
                   nrow = n_inputs, ncol = n_neurons)
  #Number of weights = the number of inputs*number of neurons,
  #since every connection between the previous neurons (from input) and
  #the current neurons have an associated weight
  biases = matrix(data = 0, nrow = 1, ncol = n_neurons)
  #Number of biases = the number

  #momentum initialization
  weight_momentums = matrix(data = 0,
                            nrow = nrow(weights),
                            ncol = ncol(weights))
  bias_momentums = matrix(data = 0,
                          nrow = nrow(biases),
                          ncol = ncol(biases))
  #cache initialization
  weight_cache = matrix(data = 0,
                        nrow = nrow(weights),
                        ncol = ncol(weights))
  bias_cache = matrix(data = 0,
                      nrow = nrow(biases),
                      ncol = ncol(biases))

  #saving:
  if (momentum == TRUE & cache == FALSE){ ##momentums only
    list("weights"=weights,"biases"=biases,
         "weight_momentums"=weight_momentums,
         "bias_momentums"=bias_momentums,
         "n_neurons"=n_neurons)
  } else if (momentum == FALSE & cache == TRUE){ ##cache only
    list("weights"=weights,"biases"=biases,
         "weight_cache"=weight_cache,
         "bias_cache"=bias_cache,
         "n_neurons"=n_neurons)
  } else if (momentum == TRUE &  cache == TRUE){ ##momentums and cache
    list("weights"=weights,"biases"=biases,
         "weight_momentums"=weight_momentums,
         "bias_momentums"=bias_momentums,
         "weight_cache"=weight_cache,
         "bias_cache"=bias_cache,
         "n_neurons"=n_neurons)
  } else if (momentum == FALSE & cache == FALSE){ ##no momentums or cache
    list("weights"=weights,"biases"=biases,"n_neurons"=n_neurons)
  }
}
