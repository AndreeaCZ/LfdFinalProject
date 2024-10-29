import random
import re

import emoji
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from wordsegment import load, segment

# Constants
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 100
LSTM_UNITS = 32
BATCH_SIZE = 32
EPOCHS = 10


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
            labels.append(1 if label == 'OFF' else 0)
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


def demojize_tweet(tweet):
    """
    Preprocesses the tweet by expanding emojis and hashtags. Also sets the tweet to lowercase.

    :param tweet:
    :return tweet: The preprocessed tweet.
    """
    tweet = emoji.demojize(tweet)
    tweet = tweet.replace(":", "").replace("_", " ")
    tweet = re.sub(r'#(\w+)', lambda match: ' '.join(segment(match.group(1))), tweet)
    tweet = tweet.lower()

    return tweet


def load_embeddings(embedding_path, word_index):
    embeddings_index = {}
    with open(embedding_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coeffs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coeffs

    # Initialize embedding matrix
    num_words = min(len(word_index) + 1, MAX_SEQUENCE_LENGTH)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_SEQUENCE_LENGTH:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# Preprocess text
def tokenize_tweets(tweets):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequences, tokenizer


# Define the LSTM model
def build_and_fit_lstm_model(embedding_matrix, X_train_seq, Y_train):
    model = Sequential()
    model.add(Embedding(input_dim=embedding_matrix.shape[0],
                        output_dim=EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))  # Static embeddings
    model.add(Bidirectional(LSTM(LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train the model
    model.fit(X_train_seq, np.array(Y_train),
              validation_split=0.2,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[EarlyStopping(monitor='val_loss', patience=3)])
              
    return model


# Main function
def main():
    load() # Initialize wordsegment
    set_seeds()

    X_train, Y_train = read_corpus('../train.tsv')
    X_test, Y_test = read_corpus('../dev.tsv')
    X_train = [demojize_tweet(tweet) for tweet in X_train]
    X_test = [demojize_tweet(tweet) for tweet in X_test]

    # Prepare sequences and tokenizer
    X_train_seq, tokenizer = tokenize_tweets(X_train)
    X_test_seq, _ = tokenize_tweets(X_test)

    # Load embeddings and build model
    embedding_matrix = load_embeddings(f'glove.twitter.27B.{EMBEDDING_DIM}d.txt', tokenizer.word_index)
    model = build_and_fit_lstm_model(embedding_matrix, X_train_seq, Y_train)

    # Predict and evaluate
    Y_pred_prob = model.predict(X_test_seq)
    Y_pred = (Y_pred_prob > 0.5).astype("int32")
    report = classification_report(Y_test, Y_pred)
    print("Classification Report:\n", report)


if __name__ == "__main__":
    main()
