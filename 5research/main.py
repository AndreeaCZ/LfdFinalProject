from model import set_seeds, read_corpus, demojize_tweet, train_and_evaluate
from wordsegment import load


def main():
    load()
    set_seeds()

    X_train, Y_train = read_corpus('../train.tsv')
    X_dev, Y_dev = read_corpus('../dev.tsv')
    X_train = [demojize_tweet(tweet) for tweet in X_train]
    X_dev = [demojize_tweet(tweet) for tweet in X_dev]

    train_and_evaluate(X_train, Y_train, X_dev, Y_dev, 64, 5e-5, 128, 6)


if __name__ == '__main__':
    main()
