import json

from model import set_seeds, read_corpus, demojize_tweet, train_and_evaluate, write_results_to_file
from wordsegment import load
from textblob import TextBlob
import tensorflow as tf
import pandas as pd


def paraphrase_tweet(tweet, synonym_mapping):
    return ' '.join(synonym_mapping.get(word, word) for word in tweet.split())


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
    write_results_to_file('paraphrased_results.txt', report, model_name)

    # Paraphrase tweets to see if it affects the model's performance
    with open('../inoffensive_synonym_mapping.json', 'r') as f:
        synonym_mapping = json.load(f)

    X_dev_paraphrased = [paraphrase_tweet(tweet, synonym_mapping) for tweet in X_dev]

    # Run the model with extra features to compare the results
    report, model_name = train_and_evaluate(
        "roberta-base",
        X_train,
        Y_train,
        X_dev_paraphrased,
        Y_dev,
        64,
        5e-6,
        128,
        4
    )
    model_name += '\nThis model was tested with paraphrased tweets to see if it affects the model\'s performance.'
    write_results_to_file('paraphrased_results.txt', report, model_name)


if __name__ == '__main__':
    main()
