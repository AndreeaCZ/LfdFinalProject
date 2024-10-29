import random
import numpy as np
import nltk
import os
import ssl
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import emoji
from wordsegment import load, segment
import re


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def read_corpus(corpus_file):
    """
    Reads the corpus file with each line containing a tweet and its label, separated by a tab.

    :param corpus_file: The name of the corpus file to be processed
    :return: Lists of tweet texts and labels
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tweet, label = line.strip().split('\t')
            documents.append(tweet)
            labels.append(label)
    return documents, labels


def write_results_to_file(results, model_name):
    """
    Writes the results of the model evaluation to a demojize_results.txt file.

    :param results: The results to be written (classification report)
    :param model_name: The name of the model for identification
    """
    with open('demojize_results.txt', 'a') as f:
        f.write(f"Results for {model_name}:\n")
        f.write("Classification Report:\n")
        f.write(results)
        f.write("\n")


def get_svm():
    """
    Returns an SVM model with optimized hyperparameters.

    :return model: The optimized SVM model.
    """
    return SVC(C=10, kernel='rbf', gamma='scale')


def preprocess_tweet(tweet):
    """
    Preprocesses the tweet by expanding emojis and hashtags.

    :param tweet:
    :return tweet: The preprocessed tweet.
    """
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":", "").replace("_", " ")
    tweet = re.sub(r'#(\w+)', lambda match: ' '.join(segment(match.group(1))), tweet)

    return tweet


def main():
    # Initialize wordsegment
    load()
    set_seeds()

    X_train, Y_train = read_corpus('../train.tsv')
    X_test, Y_test = read_corpus('../dev.tsv')
    # Preprocess the tweets to expand emojis and hashtags
    X_train = [preprocess_tweet(tweet) for tweet in X_train]
    X_test = [preprocess_tweet(tweet) for tweet in X_test]

    vectorizer = CountVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    model = get_svm()
    model.fit(X_train_vect, Y_train)
    Y_pred = model.predict(X_test_vect)
    report = classification_report(Y_test, Y_pred)
    print("Classification Report:\n", report)
    write_results_to_file(report, 'Basic SVM with expanded emojis and hashtags')


if __name__ == "__main__":
    main()
