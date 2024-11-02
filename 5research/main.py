from model import set_seeds, read_corpus, demojize_tweet, train_and_evaluate, write_results_to_file
from wordsegment import load


def main():
    load()
    set_seeds()

    X_train, Y_train = read_corpus('../train.tsv')
    X_dev, Y_dev = read_corpus('../dev.tsv')
    X_train = [demojize_tweet(tweet) for tweet in X_train]
    X_dev = [demojize_tweet(tweet) for tweet in X_dev]

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
    model_name += '\n This model has sentiment analysis as an extra feature.'
    write_results_to_file('sentiment_results.txt', report, model_name)


if __name__ == '__main__':
    main()
