import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC 

import pickle

from document_features import test_set, train_set

#train the naive bayes classifier
print("Training the Naive Bayes Classifier...")
NaiveBayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
print("Calculating Naive Bayes accuracy...")
print(nltk.classify.accuracy(NaiveBayes_classifier, test_set)*100)
f = open('naive_bayes_classifier.pickle', 'wb')
pickle.dump(NaiveBayes_classifier, f)
f.close()

# train the linear support vector classifier.
print("Training the Linear SVC classifier...")
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(train_set)
print("Calculating Linear SVC accuracy...")
print(nltk.classify.accuracy(LinearSVC_classifier, test_set)*100)
f = open('linear_svc_classifier.pickle', 'wb')
pickle.dump(LinearSVC_classifier, f)
f.close()

# train the logistic regression classifier
print("Training the Logistic Regression Classifier...")
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(train_set)
print("Calculating Logistic Regression accuracy...")
print(nltk.classify.accuracy(LogisticRegression_classifier, test_set)*100)
f = open('logistic_regression_classifier.pickle', 'wb')
pickle.dump(LogisticRegression_classifier, f)
f.close()
