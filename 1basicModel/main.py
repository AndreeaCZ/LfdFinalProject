import random

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC


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


def run_grid_search_and_evaluate(model, param_grid, X_train, Y_train, X_test, Y_test):
    """
    Runs GridSearchCV on the given model with the specified parameter grid and evaluates the model.

    :param model: The model to be optimized
    :param param_grid: The hyperparameter grid for the model
    :param X_train: Training features
    :param Y_train: Training labels
    :param X_test: Test features
    :param Y_test: Test labels
    :return: The best parameters found, the accuracy of the model, and the classification report
    """
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, Y_train)
    print("Best parameters found:", grid_search.best_params_)

    best_model = grid_search.best_estimator_
    Y_pred = best_model.predict(X_test)
    report = classification_report(Y_test, Y_pred)
    print("Classification Report:\n", report)
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


def main():
    set_seeds()
    # Load the corpus
    X_train, Y_train = read_corpus('../train.tsv')
    X_test, Y_test = read_corpus('../dev.tsv')

    # Convert text data into numerical data
    vectorizer = CountVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    nb_model = MultinomialNB()
    nb_param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False],
    }
    print("Running Grid Search for Naive Bayes...")
    nb_results = run_grid_search_and_evaluate(nb_model, nb_param_grid, X_train_vect, Y_train, X_test_vect, Y_test)
    write_results_to_file(nb_results, "Naive Bayes")

    # SVM model
    svm_model = SVC()
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
    }
    print("Running Grid Search for SVM...")
    svm_results = run_grid_search_and_evaluate(svm_model, svm_param_grid, X_train_vect, Y_train, X_test_vect, Y_test)
    write_results_to_file(svm_results, "SVM")


if __name__ == "__main__":
    main()
