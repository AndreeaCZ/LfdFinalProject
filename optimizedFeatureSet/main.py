import random
import numpy as np
import nltk
import os
import ssl
from nltk import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# Define a directory for NLTK data
nltk_data_path = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

# Set NLTK's data path
nltk.data.path.append(nltk_data_path)

# Download the required resources
# necessary because nltk.download throws errors otherwise
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)
nltk.download('averaged_perceptron_tagger_eng', download_dir=nltk_data_path)


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


def run_grid_search_and_evaluate(X_train, Y_train, X_test, Y_test, pipeline, param_grid):
    """
    Used for running and printing the results of GridSearchCV.

    :param X_train: Training set
    :param Y_train: Training labels
    :param X_test: Testing set
    :param Y_test: Testing labels
    :param pipeline: The pipeline to be used
    :param param_grid: The param grid to be tested
    :return:
    """
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')

    grid_search.fit(X_train, Y_train)

    print("Best parameters found:")
    print(grid_search.best_params_)

    Y_pred = grid_search.predict(X_test)
    report = classification_report(Y_test, Y_pred)
    print("\nClassification Report:")
    print(report)
    accuracy = accuracy_score(Y_test, Y_pred)

    return grid_search.best_params_, accuracy, report


def write_results_to_file(results, model_name):
    """
    Writes the results of the model evaluation to a results.txt file.

    :param results: The results to be written (the best parameters, accuracy, classification report)
    :param model_name: The name of the model for identification
    """
    with open('results.txt', 'a') as f:
        f.write(f"Results for {model_name}:\n")
        f.write(f"Best Parameters: {results[0]}\n")
        f.write(f"Accuracy: {results[1]:.4f}\n")
        f.write("Classification Report:\n")
        f.write(results[2])
        f.write("\n")


def get_svm():
    """
    Returns an SVM model with optimized hyperparameters.

    :return: The optimized SVM model
    """
    return SVC(C=10, kernel='rbf', gamma='scale')


def basic_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    return tokens


def stem_tokens(tokens):
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def pos_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = pos_tag(tokens)
    return [f"{word}_{tag}" for word, tag in pos_tags]


def main():
    set_seeds()

    X_train, Y_train = read_corpus('../train.tsv')
    X_test, Y_test = read_corpus('../dev.tsv')

    pipeline = Pipeline([
        ('vec', CountVectorizer(token_pattern=None)),
        ('cls', get_svm())
    ])

    param_grid = [
        {
            'vec': [CountVectorizer(token_pattern=None), TfidfVectorizer(token_pattern=None)],
            'vec__max_df': [1.0, 0.95, 0.90, 0.85],
            'vec__ngram_range': [(1, 1), (1, 2), (1, 3)],
            'vec__max_features': [None, 100, 1000, 10000],
            'vec__tokenizer': [basic_tokenizer, stem_tokens, lemmatize_tokens, pos_tokenizer],
        }
    ]

    results = run_grid_search_and_evaluate(X_train, Y_train, X_test, Y_test, pipeline, param_grid)
    write_results_to_file(results, 'SVM with optimized features')


if __name__ == "__main__":
    main()
