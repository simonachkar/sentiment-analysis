import pickle
from document_features import document_features

f = open('naive_bayes_classifier.pickle', 'rb')
naive_bayes_classifier = pickle.load(f)
f.close()

f = open('linear_svc_classifier.pickle', 'rb')
linear_svc_classifier = pickle.load(f)
f.close()

f = open('logistic_regression_classifier.pickle', 'rb')
logistic_regression_classifier = pickle.load(f)
f.close()

reviews = [
    "You suck!",
    "This is a really great product",
    "Amazing stuff",
    "This is the worst product ever!",
    "Total waste of money",
]

for r in reviews:
    print(r)
    print("\tnaive bayes classifier")
    print(naive_bayes_classifier.classify(
        document_features({'sentence': r})))
    print("\tlinear svc classifier")
    print(linear_svc_classifier.classify(
        document_features({'sentence': r})))
    print("\tlogistic regression classifier")
    print(logistic_regression_classifier.classify(
        document_features({'sentence': r})))
