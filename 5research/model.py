from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import random as python_random
import numpy as np
import tensorflow as tf
import emoji
import re
from wordsegment import load, segment


def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    python_random.seed(seed)


def read_corpus(corpus_file):
    """
    Reads the corpus file with the label as the last token and provides tokenized documents and labels.

    :param corpus_file: The name of the corpus file to be processed
    :return: The tokenized documents and labels
    """
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as in_file:
        for line in in_file:
            tokens = line.strip().split()
            label = tokens[-1]
            document = ' '.join(tokens[:-1])
            documents.append(document)
            labels.append(label)
    return documents, labels


def to_categorical(y):
    """
    Converts a list of labels to a one-hot encoded matrix.
    """
    y = np.asarray(y)  # Ensure `y` is a NumPy array
    num_classes = len(np.unique(y))
    return np.eye(num_classes)[y.reshape(-1)]


def demojize_tweet(tweet):
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":", "").replace("_", " ")
    tweet = re.sub(r'#(\w+)', lambda match: ' '.join(segment(match.group(1))), tweet)
    return tweet.lower()


def write_results_to_file(file, report, model_name):
    with open(file, 'a') as f:
        f.write(f"Results for {model_name}:\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\n")


def train_and_evaluate(lm, X_train, Y_train, X_dev, Y_dev, max_seq_len, learning_rate, batch_size, epochs):
    encoder = LabelBinarizer()
    Y_train_bin = encoder.fit_transform(Y_train)
    Y_dev_bin = encoder.transform(Y_dev)

    Y_train_bin = to_categorical(Y_train_bin)
    Y_dev_bin = to_categorical(Y_dev_bin)

    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm, num_labels=2, from_pt=True)
    tokens_train = tokenizer(X_train, padding=True, max_length=max_seq_len, truncation=True, return_tensors="tf").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=max_seq_len, truncation=True, return_tensors="tf").data

    loss_function = CategoricalCrossentropy(from_logits=True)
    optim = Adam(learning_rate=learning_rate)

    model.compile(loss=loss_function, optimizer=optim, metrics=['accuracy'])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epochs, batch_size=batch_size, validation_data=(tokens_dev, Y_dev_bin))

    Y_pred = model.predict(tokens_dev)["logits"]

    model_name = f"{lm} with max_seq_len: {max_seq_len}, learning_rate: {learning_rate}, batch_size: {batch_size}, epochs: {epochs}"
    report = classification_report(Y_dev_bin.argmax(axis=1), Y_pred.argmax(axis=1))
    print(model_name)
    print(report)
    return report, model_name


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
    write_results_to_file('results.txt', report, model_name)


if __name__ == '__main__':
    main()
