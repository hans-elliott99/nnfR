#Classess: Assess Classification Performance

#' Classification Assessment
#'
#' This function takes as input the groud truth labels and predicted labels from
#' a binary or multi-class classification problem. For binary tasks, it computes
#' a simple confusion matrix and calculates classification metrics.
#' For multi-class tasks, it computes a multi-class confusion matrix and computes
#' the classification metrics for each class.
#' Current metrics are accuracy, sensitivity/recall, specificity, precision,
#' and F1 score.
#'
#' @param truths A vector of true labels (indexed as numbers from 1 to k)
#' @param predictions A vector of predicted labels (with same indexing from truths)
#' @return A confusion matrix and dataframe storing classification metrics
#' @export
classess = function(truths, predictions){
  y_true = as.vector(truths)
  y_hat = as.vector(predictions)

  #Binary Confusion Matrix
  if (length(unique(y_true)) <= 2){
    #Note: positive class will be whichever class is assigned to label 1
    #Confusion Matrix
    ##Need to initialize a matrix of the correct length with zeros due to
    ##possibility of predictions not containing every possible class
    y_true = y_true + 1 ##cant index using 0 and 1
    y_hat = y_hat + 1
    # conf_mat = matrix(0, ##create matrix of 0s of appropriate length
    #        nrow = length(unique(y_true)),
    #        ncol = length(unique(y_true)))
    conf_mat = table(y_hat, y_true) #tabulate the predictions and truths
    diff = setdiff(y_true, y_hat) ##figure out which if any class not predicted
    #fill out table if missing class, but order depends on which class
    for (class in diff){
      newrow = array(0, dim = length(unique(y_true)))
      conf_mat = rbind(conf_mat,newrow)
      conf_mat = conf_mat[order(c(1:(nrow(conf_mat)-1),class-0.5)),]
    }

    #Format Confusion Matrix
    dimnames(conf_mat) = list("pred"=c(sort(unique(y_true)-1)),
                              "true"=c(sort(unique(y_true)-1))
    )
    tp = conf_mat[1,1]
    fp = conf_mat[1,2]
    tn = conf_mat[2,2]
    fn = conf_mat[2,1]
    total = tp + fp + tn + fn

    #Calculate metrics
    m = data.frame(
      accuracy = (tp + tn) / total, #share of correctly predicted labels
      sensitivity = tp / (tp + fn), #share of true pos we predict correct
      recall = tp / (tp + fn), #recall = sensitivity
      specificity = tn / (tn + fp), #share of true no's we predicted correct
      precision = tp / (tp + fp) #share of predicted pos that are correct
    )
    m = cbind(m,
              #F1 Score: the harmonic mean of precision and recall
              f1_score = 2 * (m$precision * m$recall) / (m$precision + m$recall)
    )
    #return
    list(conf_mat = conf_mat, metrics = m)

    #Multiclass Confusion Matrix
  } else if (length(unique(y_true)) > 2){
    # conf_mat = matrix(0,
    #        nrow = length(unique(y_true)), ncol = length(unique(y_true)))
    conf_mat = table(y_hat, y_true)

    diff = setdiff(y_true, y_hat)
    for (class in diff){
      newrow = array(0, dim = length(unique(y_true)))
      conf_mat = rbind(conf_mat,newrow)
      conf_mat = conf_mat[order(c(1:(nrow(conf_mat)-1),class-0.5)),]
    }
    #Idea source: https://stackoverflow.com/questions/11561856/add-new-row-to-dataframe-at-specific-row-index-not-appended.

    #Format Confusion Matrix
    dimnames(conf_mat) = list("pred"=c(sort(unique(y_true))),
                              "true"=c(sort(unique(y_true)))
    )

    #Find TP, TN, FP and FN for each individual class
    #Add metrics to a dataframe
    metrics = data.frame(class = 1:length(unique(y_true)))
    #if a class is not in y_hat, it will produce NA. good for user to see
    for (class in unique(y_hat)){##iterate through classes actually predicted
      tp = conf_mat[class, class]
      fp = sum(conf_mat[class, ]) - tp
      tn = sum(conf_mat[-class, -class])
      fn = sum(conf_mat[class, ]) - tp
      total = tp + fp + tn + fn

      accuracy = (tp + tn) / total
      metrics[class, "accuracy"] = accuracy
      sensitivity = tp / (tp + fn)
      metrics[class, "sensitivity"] = sensitivity
      recall = sensitivity
      metrics[class, "recall"] = recall
      specificity = tn / (tn + fp)
      metrics[class, "specificity"] = specificity
      precision = tp / (tp + fp)
      metrics[class, "precision"] = precision

      f1_score = 2 * (precision * recall) / (precision + recall)
      metrics[class, "f1_score"] = f1_score
    }
    overall_accuracy = sum(y_hat == y_true) / length(y_true)
    metrics = cbind(metrics, overall_accuracy)
    #return
    list(conf_mat = conf_mat, metrics = metrics)
  }
}
