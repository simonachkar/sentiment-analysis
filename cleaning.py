import nltk
from nltk.corpus import stopwords
import re

import pickle

sentences = []

read_files = ["./data/amazon_cells_labelled.txt",
              "./data/imdb_labelled.txt", "./data/yelp_labelled.txt"]

# create dicts of sentences and category (1 for positive & 0 for negative)
print("Reading files...")
for read_file in read_files:
    with open(read_file, "r") as r:
        c = 0
        for line in r:
            splitted = line.strip().split('\t')  # split sentence and class
            msg = (' ').join(splitted[:-1])
            is_class = splitted[-1]
            sentences.extend([dict(sentence=msg.lower(), category=is_class)])

print(f"# of sentences: {len(sentences)}")

all_words = []

# create a node in the dict with all the words and cleaning the words
print("Cleaning the data...")
for n in range(len(sentences)):
    sentences[n]['words'] = re.split(r'\W+', sentences[n]['sentence'])
    for word in sentences[n]['words']:
        """Data Cleaning
            removing:
                * stop words like 'the', 'are' and 'is'
                * void spaces
                * digits and words containing digits
        """
        if (word not in stopwords.words()) & (word != '') & (not bool(re.search(r'\d', word))):
            all_words.append(word)

all_words = nltk.FreqDist(all_words)

# Get the top 3000 words
word_features = list(all_words.keys())[:3000]

f = open('word_features.pickle', 'wb')
pickle.dump(word_features, f)
f.close()

f = open('sentences.pickle', 'wb')
pickle.dump(sentences, f)
f.close()
