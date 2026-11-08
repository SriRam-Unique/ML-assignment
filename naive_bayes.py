import pickle

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import ROOT, TARGET, validate_data


def train():
    features, target = validate_data(pd.read_csv(ROOT / "data" / "train.csv"))
    features, test_features, target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    test_data = test_features.copy()
    test_data[TARGET] = test_target
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    model_dir = ROOT / "model"
    model_dir.mkdir(exist_ok=True)

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", GaussianNB()),
    ])
    parameters = {"classifier__var_smoothing": [1e-11, 1e-9, 1e-7]}
    search = GridSearchCV(pipeline, parameters, cv=5, scoring="f1_macro")
    search.fit(features, target)
    best_model = search.best_estimator_
    with open(model_dir / "naive_bayes.pkl", "wb") as file:
        pickle.dump(best_model, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Naive Bayes parameters:", search.best_params_)
    print(f"Best cross-validation F1: {search.best_score_:.4f}")
    return best_model


if __name__ == "__main__":
    train()