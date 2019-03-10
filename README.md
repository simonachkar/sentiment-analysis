# Sentiment Analysis

After following [this](https://pythonprogramming.net/naive-bayes-classifier-nltk-tutorial/) and [this](https://pythonprogramming.net/sklearn-scikit-learn-nltk-tutorial/), I followed this approach for the sentiment analysis problem:

## Data preparation (`cleaning.py`)

### Reading the data 
First, I read all the data of reviews provided from the three data sets (Amazon, IMBD & Yelp). And then I created a dictionary of the reviews' sentences composed of the sentence and its category (0 for negative and 1 for positive)

> This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015

### Cleaning the data
I split the sentences in an array of words, then I did some cleaning. I removed: 
* Stop words like 'the', 'are' and 'is'
* Void spaces `''` and `' '`
* Digits and words containing digits

I stored the results in an array called `all_words`. Then I converted the array to a FreqDist using the `nltk` library, so the words will be grouped by frequency from the most frequent to the least frequent. I then picked the top 3000 most frequent words and stored them in `word_features`

> I "pickeled" the results of reading the data (`sentences`)and cleaning the data (`word_features`) so I don't have to do this all over on every run

## Getting the featuresets (`document_features.py`)

Now, after cleaning, I shuffled the sentences and created  `featuresets`, an array of sentences composed of combining the words that match with `word_features` in the sentence  and thencategorizing these words (0 or 1). 
I split the `featuresets` into two, `train_set` and `test_set`.

## Training and testing the classifier (`classifiers.py`)

The following code train the classifier using the train set and then tests it using the test set and returns its accuracy (percentage)

```python
nltk.<classifier>.train(train_set)
nltk.classify.accuracy(<classifier>, test_set)*100
```

## The app (`app.py`)

Finaly the app loads the classifiers, runs them on some sentences, and then returns if the sentence is a negative or a positive review (0 or 1)
