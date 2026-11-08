import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay

from evaluation import (
    CLASSES, CLASS_NAMES, METRICS, MODEL_FILES, ROOT, evaluate_model, load_models, validate_data,
)

st.set_page_config(page_title="Mobile Price Lab", page_icon=":material/smartphone:", layout="wide")


def main():
    st.title("Mobile Price Lab")
    st.caption("Four price tiers | Five classifiers | Held-out evaluation")
    try:
        models = load_models()
    except (OSError, ValueError, EOFError, pickle.UnpicklingError) as error:
        st.error(f"Saved model artifacts are unavailable: {error}")
        st.stop()

    with st.sidebar:
        st.header("Evaluation")
        selected = st.selectbox("Classifier", list(MODEL_FILES))
        sample_path = ROOT / "test_data.csv"
        if sample_path.exists():
            st.download_button(
                "Download test CSV", sample_path.read_bytes(),
                file_name="test_data.csv", mime="text/csv", icon=":material/download:",
            )
        uploaded = st.file_uploader("Labeled test CSV", type=["csv"])

    try:
        if uploaded is not None and uploaded.size > 5 * 1024 * 1024:
            raise ValueError("Test CSVs are limited to 5 MB.")
        input_file = uploaded if uploaded is not None else sample_path
        source_label = uploaded.name if uploaded is not None else "Bundled holdout"
        frame = pd.read_csv(input_file)
        if len(frame) > 5000:
            raise ValueError("Test CSVs are limited to 5,000 rows.")
        features, target = validate_data(frame)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as error:
        st.error(f"Cannot evaluate this CSV: {error}")
        st.stop()

    results = []
    predictions_by_model = {}
    for name, model in models.items():
        scores, predictions = evaluate_model(model, features, target)
        results.append({"ML Model Name": name, **scores})
        predictions_by_model[name] = predictions
    comparison = pd.DataFrame(results).set_index("ML Model Name")
    st.subheader(selected)
    st.caption(f"{source_label} | {len(features):,} phones | {target.nunique()} observed price tiers")
    for column, metric in zip(st.columns(len(METRICS)), METRICS):
        value = comparison.loc[selected, metric]
        column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")
    if target.nunique() < len(CLASSES):
        st.warning("AUC is unavailable because the test CSV does not contain all four classes.")

    st.subheader("Model comparison")
    st.dataframe(
        comparison,
        column_config={metric: st.column_config.NumberColumn(format="%.4f") for metric in METRICS},
        width="stretch",
    )
    st.caption("Precision, recall and F1: macro average. AUC: macro one-vs-rest.")
    st.subheader("Confusion matrix")
    predictions = predictions_by_model[selected]
    figure, axis = plt.subplots(figsize=(6, 4.8))
    ConfusionMatrixDisplay.from_predictions(
        target, predictions, labels=CLASSES, display_labels=CLASS_NAMES,
        cmap="Greens", colorbar=False, ax=axis,
    )
    axis.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    st.pyplot(figure, width="content")
    plt.close(figure)


if __name__ == "__main__":
    main()