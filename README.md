# Mobile Price Detectioon

## a. Problem statement

Predict a mobile phone's price category from its hardware and connectivity specifications.
Compare five classifiers on the same held-out data using accuracy, multiclass ROC AUC,
macro precision, macro recall, macro F1, and Matthews correlation coefficient (MCC).
The goal is price-tier classification, not estimation of a monetary price.

## b. Dataset description

Source: [Mobile Price Classification on Kaggle](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).

- Labeled source: [data/train.csv](data/train.csv), 2,000 rows and 20 numeric input features.
- Target: `price_range`, with 0 = low cost, 1 = medium cost, 2 = high cost, 3 = very high cost.
- Each class has 500 labeled observations. The dataset exceeds the required 500 instances and 12 features.
- The generated [test_data.csv](test_data.csv) contains 400 held-out rows, all 20 features and the target, with 100 observations per class.
- All required values in the labeled source are numeric and finite. Validation rejects missing values, non-numeric values, unsupported target labels, and unexpected columns; an optional `id` column is ignored.


## c. GitHub Repository Link And Streamlit link

Git Hub : https://github.com/SriRam-Unique/ML-assignment
Streamlit : https://ml-assignment-cqj48fxqq8qeaz7yoxbdbv.streamlit.app/

## d. Models used

| ML Model Name | Separate training file | Selected settings |
| --- | --- | --- |
| Logistic Regression | [logistic_regression.py](logistic_regression.py) | `C=10`, `max_iter=3000`, `random_state=42`, with scaling |
| Decision Tree | [decision_tree.py](decision_tree.py) | `max_depth=8`, `min_samples_leaf=1`, `random_state=42` |
| KNN | [knn.py](knn.py) | `n_neighbors=21`, `weights="distance"`, with scaling |
| Naive Bayes | [naive_bayes.py](naive_bayes.py) | Gaussian NB, `var_smoothing=1e-11`, with scaling |
| Support Vector Machine | [support_vector_machine.py](support_vector_machine.py) | `C=10`, `gamma=0.01`, RBF kernel, `probability=True`, `random_state=42`, with scaling |

### Evaluation comparison

Measured from the saved pipelines on the 400-row holdout. Values are rounded to four
decimal places. Run [evaluation.py](evaluation.py) to print the comparison table again.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9750 | 0.9990 | 0.9752 | 0.9750 | 0.9750 | 0.9667 |
| Decision Tree | 0.8500 | 0.9125 | 0.8523 | 0.8500 | 0.8501 | 0.8007 |
| KNN | 0.6175 | 0.8343 | 0.6225 | 0.6175 | 0.6196 | 0.4902 |
| Naive Bayes | 0.8100 | 0.9506 | 0.8113 | 0.8100 | 0.8105 | 0.7468 |
| Support Vector Machine | 0.9400 | 0.9940 | 0.9398 | 0.9400 | 0.9398 | 0.9201 |

Precision, recall and F1 are macro-averaged across the four fixed labels; undefined
per-class precision/recall is assigned zero. AUC uses `predict_proba`, macro averaging,
and one-vs-rest multiclass ROC AUC, not hard predicted labels. When an uploaded CSV
lacks any target class, AUC is displayed as N/A rather than an invented score. Other
metrics and a four-class confusion matrix remain available.

### Observations on model performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Highest held-out accuracy and macro F1: 390/400 correct, with 10 adjacent-tier mistakes. AUC 0.9990 indicates very strong class ranking on this split. The result is consistent with useful approximately linear decision boundaries.
| Decision Tree | 340/400 correct and F1 0.8501. All 60 mistakes were between adjacent tiers. The selected depth-limited tree trails Logistic Regression and SVM; attributing this specifically to overfitting would require additional training/validation evidence. |
| KNN | 247/400 correct, F1 0.6196 and MCC 0.4902, lowest among the five selected models. Of 153 errors, 3 span more than one tier. Equal weighting of all standardized features may weaken neighborhood quality; feature relevance or alternative distances would need training-only validation. |
| Naive Bayes | 324/400 correct, F1 0.8105 and AUC 0.9506. All 76 mistakes are adjacent-tier errors. |
| Support Vector Machine | Second-highest accuracy and F1: 376/400 correct, F1 0.9398 and AUC 0.9940. All 24 mistakes are adjacent-tier errors. The training-selected RBF model does not outperform Logistic Regression here; the small grid is not an exhaustive kernel or parameter search. |
| Overall| Logistic Regression wins this fixed holdout comparison by macro F1, with the highest accuracy and MCC as well. |


## Commands

Use Python From the project root, create an environment on first setup:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -m venv .venv`. In PowerShell,
activate it with `.\.venv\Scripts\Activate.ps1`; in Command Prompt, use
`.venv\Scripts\activate.bat`.

### Train, evaluate, and launch


```sh
python train_models.py
python -m streamlit run app.py
```

To only open the app with existing models, run `python -m streamlit run app.py`.
The default local URL is `http://localhost:8501`. Stop the server with Ctrl+C.