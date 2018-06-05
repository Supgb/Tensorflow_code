import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, \
    precision_score, recall_score, precision_recall_curve, \
    roc_curve, roc_auc_score, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def plot_precision_recall_of_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0,1])


def plot_precision_against_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1])


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# Data prepared.
mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# Train a 5 and non-5 binary classifier
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_5 = (y_train==5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Precision/Recall Tradeoff And ROC
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
precisions, recalls, thresholds_pr = precision_recall_curve(y_train_5, y_scores)

"""
When chose a good thresholds, calling the classifier's decision_function([some_digit])
to get the scores, and then make a prediction based on what threshold you prefer.

Such as,
>>> y_scores = sgd_clf.decision_function([some_digit])
>>> threshold = ???
>>> y_some_digit_pred = (y_scores > threshold)
and then get the output.
"""

"""
  The ROC Curve, plots the 'true positive rate'(recall) against the 'false positive rate'.
  (TNR---specificity)true negative rate: the ratio of negative instances that are correctly classified \
                           as negative.
  (FPR)false positive rate: the ratio of positive instances that are incorrectly classified as \
                           positive. And it's equal to one minus the TNR.(1-TNR)
  (TPR---sensitivity)true positive rate: the ratio of positive instances that are correctly classified \
                                         as positive.
                                         
  Hence, the ROC curve also can be comprehended as \
  plotting sensitivity(recall) versus (1-specificity). 
  
  The dotted line represents a purely random classifier.
  
  The standard of incredible classifier:
     A good classifier stays as far as away from the dotted line.(toward the top-left corner)
     
  ROC AUC(receiver operating characteristic area under curve):
      a perfect classifier will have ROC AUC equal to 1, whereas a purely random classifier \
      will have ROC AUC equal to 0.5.
     
     
  
"""
fpr, tpr, thresholds_roc = roc_curve(y_train_5, y_scores)

forest_clf = RandomForestClassifier(random_state=42)
y_prob_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                  method='predict_proba')
y_scores_forest = y_prob_forest[:, 1] # using the positive probability as the score
fpr_forest, tpr_forest, thresholds_forest_roc = roc_curve(y_train_5, y_scores_forest)

""" KNeighborsClassifier for Multilabel Classification """
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 != 0)
y_mult = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_mult)

""" KNN for Multioutput Classification """
noise1 = np.random.randint(0, 100, (len(X_train), 784))
noise2 = np.random.randint(0, 100, (len(X_test), 784))
X_train_mod = X_train + noise1
X_test_mod = X_test + noise2
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)