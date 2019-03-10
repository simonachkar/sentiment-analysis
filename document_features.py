import random
import re

import pickle

f = open('word_features.pickle', 'rb')
word_features = pickle.load(f)
f.close()

f = open('sentences.pickle', 'rb')
sentences = pickle.load(f)
f.close()

def document_features(document):
    if not document.get('words'):
        document['words'] = re.split(r'\W+', document['sentence'])
    # find all unique words
    document_words = set(document['words'])
    features = {}
    for w in word_features:
        # simply finding and formatting the features
        features[w] = (w in document_words) # a boolean value
    return features

random.shuffle(sentences)

featuresets = [(document_features(d), d['category']) for d in sentences]

train_set, test_set = featuresets[:2000], featuresets[1000:]
