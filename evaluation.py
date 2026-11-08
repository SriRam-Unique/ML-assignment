import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET = "price_range"
CLASSES = [0, 1, 2, 3]
CLASS_NAMES = ["Low cost", "Medium cost", "High cost", "Very high cost"]
FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
ROOT = Path(__file__).resolve().parent


def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        with open(ROOT / "model" / filename, "rb") as file:
            models[name] = pickle.load(file)
    return models


def validate_data(frame):
    if frame.empty:
        raise ValueError("The CSV contains no data rows.")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names are not allowed.")
    missing = [column for column in FEATURES + [TARGET] if column not in frame]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    unexpected = sorted(set(frame.columns) - set(FEATURES + [TARGET, "id"]))
    if unexpected:
        raise ValueError("Unexpected columns: " + ", ".join(unexpected))
    try:
        numeric = frame[FEATURES + [TARGET]].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError("All features and price_range must be numeric.") from error
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing values and infinite values are not allowed.")
    if not numeric[TARGET].isin(CLASSES).all():
        raise ValueError("price_range must contain only integer labels 0, 1, 2, 3.")
    return numeric[FEATURES], numeric[TARGET].astype(int)


def evaluate_model(model, features, target):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    if list(model.classes_) != CLASSES:
        raise ValueError("The saved model must contain classes 0, 1, 2, 3 in order.")
    auc = float("nan")
    if set(target.unique()) == set(CLASSES):
        auc = roc_auc_score(
            target, probabilities, labels=CLASSES, multi_class="ovr", average="macro"
        )
    scores = {
        "Accuracy": accuracy_score(target, predictions),
        "AUC": auc,
        "Precision": precision_score(
            target, predictions, labels=CLASSES, average="macro", zero_division=0
        ),
        "Recall": recall_score(
            target, predictions, labels=CLASSES, average="macro", zero_division=0
        ),
        "F1": f1_score(
            target, predictions, labels=CLASSES, average="macro", zero_division=0
        ),
        "MCC": matthews_corrcoef(target, predictions),
    }
    return scores, predictions


def main():
    features, target = validate_data(pd.read_csv(ROOT / "test_data.csv"))
    results = []
    for name, model in load_models().items():
        scores, _ = evaluate_model(model, features, target)
        results.append({"ML Model Name": name, **scores})
    comparison = pd.DataFrame(results)
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    winner = comparison.sort_values(["F1", "MCC", "Accuracy"], ascending=False).iloc[0]
    print(f"\nHighest held-out macro F1: {winner['ML Model Name']}.")


if __name__ == "__main__":
    main()