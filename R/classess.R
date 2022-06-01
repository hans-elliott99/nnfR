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

  if (length(unique(y_true)) <= 2){
    ##Binary Confusion Matrix

    ##Note: positive class will be whichever class is assigned to label 1
    #Confusion Matrix
    conf_mat = table(y_hat, y_true, dnn = c("pred", "true"))

    tp = conf_mat[1,1]
    fp = conf_mat[1,2]
    tn = conf_mat[2,2]
    fn = conf_mat[2,1]
    total = tp + fp + tn + fn

    metrics = data.frame(
      accuracy = (tp + tn) / total, #share of correctly predicted labels
      sensitivity = tp / (tp + fn), #share of true pos we predict correct
      recall = sensitivity,
      specificity = tn / (tn + fp), #share of true no's we predicted correct
      precision = tp / (tp + fp), #share of predicted pos that are correct

      #F1 Score: the harmonic mean of precision and recall
      f1_score = 2 * (precision * recall) / (precision + recall)
    )
    #return
    list(conf_mat = conf_mat, metrics = metrics)


  } else if (length(unique(y_true)) > 2){
    ##Multiclass Confusion Matrix

    #Confusion Matrix
    conf_mat = table(y_hat, y_true, dnn = c("pred", "true"))

    #Find TP, TN, FP and FN for each individual class
    #Add metrics to a dataframe
    metrics = data.frame(class = 1:length(unique(y_true)))
    for (class in unique(y_true)){
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
    #return
    list(conf_mat = conf_mat, metrics = metrics)
  }
}
