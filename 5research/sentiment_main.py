import json

from model import set_seeds, read_corpus, demojize_tweet, train_and_evaluate, write_results_to_file
from wordsegment import load
from textblob import TextBlob
import tensorflow as tf
import pandas as pd


def get_sentiment_score(tweet):
    """
    Get sentiment score from TextBlob (or use an alternative sentiment analysis tool).
    """
    return TextBlob(tweet).sentiment.polarity


def main():
    load()
    set_seeds()

    X_train, Y_train = read_corpus('../train.tsv')
    X_dev, Y_dev = read_corpus('../dev.tsv')
    X_train = [demojize_tweet(tweet) for tweet in X_train]
    X_dev = [demojize_tweet(tweet) for tweet in X_dev]

    # Run the model without extra features
    report, model_name = train_and_evaluate(
        "roberta-base",
        X_train,
        Y_train,
        X_dev,
        Y_dev,
        64,
        5e-6,
        128,
        4
    )
    write_results_to_file('sentiment_results.txt', report, model_name)

    # Add sentiment analysis as an extra feature
    X_train_with_sentiment = [f"[SENTIMENT] {get_sentiment_score(tweet)} {tweet}" for tweet in X_train]
    X_dev_with_sentiment = [f"[SENTIMENT] {get_sentiment_score(tweet)} {tweet}" for tweet in X_dev]

    # Run the model with extra features to compare the results
    report, model_name = train_and_evaluate(
        "roberta-base",
        X_train_with_sentiment,
        Y_train,
        X_dev_with_sentiment,
        Y_dev,
        64,
        5e-6,
        128,
        4
    )
    model_name += '\nThis model uses sentiment analysis as an extra feature.'
    write_results_to_file('sentiment_results.txt', report, model_name)


if __name__ == '__main__':
    main()
