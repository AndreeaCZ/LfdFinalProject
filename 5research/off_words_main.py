import json

from model import set_seeds, read_corpus, demojize_tweet, train_and_evaluate, write_results_to_file
from wordsegment import load
import tensorflow as tf
import pandas as pd


def percentage_offensive_words(tweet, offensive_words):
    """
    Calculate the percentage of offensive words in the given tweet.

    :param tweet: The tweet text to analyze.
    :return: Percentage of offensive words in the tweet.
    """
    words = tweet.split()
    if len(words) == 0:
        return 0  # Avoid division by zero

    offensive_count = sum(1 for word in words if word in offensive_words)
    return offensive_count / len(words) * 100


def has_offensive_words(tweet, offensive_words):
    """
    Check if the given tweet contains any offensive words.

    :param tweet: The tweet text to analyze.
    :return: True if the tweet contains offensive words, False otherwise.
    """
    return any(word in tweet for word in offensive_words)


def main():
    load()
    set_seeds()

    X_train, Y_train = read_corpus('../train.tsv')
    X_dev, Y_dev = read_corpus('../dev.tsv')
    X_train = [demojize_tweet(tweet) for tweet in X_train]
    X_dev = [demojize_tweet(tweet) for tweet in X_dev]

    with open('../words.json') as f:
        offensive_words = json.load(f)
    X_train_with_percentage = [f"[OFFENSIVE_PERCENTAGE] {percentage_offensive_words(tweet, offensive_words)} {tweet}" for tweet in X_train]
    X_dev_with_percentage = [f"[OFFENSIVE_PERCENTAGE] {percentage_offensive_words(tweet, offensive_words)} {tweet}" for tweet in X_dev]
    X_train_with_boolean = [f"[HAS_OFFENSIVE_WORDS] {has_offensive_words(tweet, offensive_words)} {tweet}" for tweet in X_train]
    X_dev_with_boolean = [f"[HAS_OFFENSIVE_WORDS] {has_offensive_words(tweet, offensive_words)} {tweet}" for tweet in X_dev]

    # # Run the model without extra features
    # report, model_name = train_and_evaluate(
    #     "roberta-base",
    #     X_train,
    #     Y_train,
    #     X_dev,
    #     Y_dev,
    #     64,
    #     5e-6,
    #     128,
    #     4
    # )
    # write_results_to_file('offensive_words_results.txt', report, model_name)

    # # Run the model with extra features to compare the results
    # report, model_name = train_and_evaluate(
    #     "roberta-base",
    #     X_train_with_percentage,
    #     Y_train,
    #     X_dev_with_percentage,
    #     Y_dev,
    #     64,
    #     5e-6,
    #     128,
    #     4
    # )
    # model_name += '\nThis model uses a set of offensive words to determine the percentage of offensive words in the tweet and use it as a new feature.'
    # write_results_to_file('offensive_words_results.txt', report, model_name)

    # Run the model with extra features to compare the results
    report, model_name = train_and_evaluate(
        "roberta-base",
        X_train_with_boolean,
        Y_train,
        X_dev_with_boolean,
        Y_dev,
        64,
        5e-6,
        128,
        4
    )
    model_name += '\nThis model uses a set of offensive words to determine if the tweet contains any offensive words and use it as a new feature.'
    write_results_to_file('offensive_words_results.txt', report, model_name)


if __name__ == '__main__':
    main()
