## Creating Spiral Data for Classification##
#Source: https://cs231n.github.io/neural-networks-case-study/


#' Simulate Spiral Data
#'
#' Simulate spiral data for multi-class classification tasks.
#' Source: https://cs231n.github.io/neural-networks-case-study/
#'
#' @param N the number of data points per class
#' @param K the number of classes
#' @export
sim_spiral_data = function(
  N = 100, # number of points per class
  D = 2,   # "dimensionality" (number of features)
  K = 3,   # number of classes
  random_order = FALSE #T: random order of classes (harder task)
){
  X = data.frame() # data matrix (each row = single sample)
  y = data.frame() # class labels

  for (j in (1:K)){
    r = seq(0.05,1,length.out = N) # radius
    t = seq((j-1)*4.7,j*4.7, length.out = N) + rnorm(N, sd = 0.3) # theta
    Xtemp = data.frame(x =r*sin(t) , y = r*cos(t))
    ytemp = data.frame(matrix(j, N, 1))
    X = rbind(X, Xtemp)
    y = rbind(y, ytemp)
  }
  spiral_data = cbind(X,y)
  colnames(spiral_data) = c(colnames(X), 'label')
  spiral_data$label = spiral_data$label

  # Want randomly ordered labels?
  if (random_order==TRUE) {spiral_data$label = sample(1:(K), size = N*K,
                                                      replace = TRUE)}
  return(spiral_data)
}

## Create Sine Wave Data for Regression ##

#' @export
sim_sine_data = function(samples, multiple){
  x  = seq(0, multiple*pi, length.out = samples)
  sine_data = data.frame(x = x/max(x), ##normalized
                         y = sin(x))
  ##shuffle up the samples
  sine_data = sine_data[sample(1:nrow(sine_data)), ]
  return(sine_data)
}
