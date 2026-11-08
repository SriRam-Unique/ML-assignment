from decision_tree import train as train_decision_tree
from knn import train as train_knn
from logistic_regression import train as train_logistic_regression
from naive_bayes import train as train_naive_bayes
from support_vector_machine import train as train_svm


def main():
    train_logistic_regression()
    train_decision_tree()
    train_knn()
    train_naive_bayes()
    train_svm()
    print("All five models saved. Run python evaluation.py to evaluate them.")


if __name__ == "__main__":
    main()