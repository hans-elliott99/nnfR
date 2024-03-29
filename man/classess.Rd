% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/classess.R
\name{classess}
\alias{classess}
\title{Classification Assessment}
\usage{
classess(truths, predictions)
}
\arguments{
\item{truths}{A vector of true labels. For binary/2 class tasks, these
2 classes may be indexed as either (0, 1) or (1, 2). For multi-class tasks,
the classes should be indexed starting at 1 (from 1 to K).}

\item{predictions}{A vector of predicted labels (with the same indexing as the truths)}
}
\value{
A confusion matrix, dataframe storing classification metrics, and the
overall accuracy of the predictions.
}
\description{
This function takes as input the groud truth labels and predicted labels from
a binary or multi-class classification problem. For binary tasks, it computes
a simple confusion matrix and calculates classification metrics.
For multi-class tasks, it computes a multi-class confusion matrix and computes
the classification metrics for each class.
Current metrics are accuracy, sensitivity/recall, specificity, precision,
and F1 score.
}
