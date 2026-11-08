# This app shows the saved model evaluation results in a simple dashboard.
from pathlib import Path

import pandas as pd
import streamlit as st


# These paths point to the trained model files and the saved metric table.
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.csv"
MODEL_FILES = [
    "preprocessor.joblib",
    "logistic_regression.joblib",
    "decision_tree.joblib",
    "knn.joblib",
    "naive_bayes.joblib",
    "support_vector_machine.joblib",
]
METRIC_COLUMNS = [
    "Accuracy",
    "AUC Score",
    "Precision",
    "Recall",
    "F1 Score",
    "MCC Score",
]


# Load the fixed test dataset from CSV to display the evaluation set size.
@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)


# Load the saved performance table created by the training script.
@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


def main():
    # Set up the page title and full-width layout for the dashboard.
    st.set_page_config(
        page_title="Bank Marketing Model Comparison",
        page_icon="📊",
        layout="wide",
    )
    st.title("Bank Marketing Classification")
    st.caption(
        "Five classifiers evaluated on the same stratified holdout. "
        "The positive class is subscription: yes."
    )

    # Check that the model files and metric file exist before showing the page.
    try:
        missing_models = [
            filename
            for filename in MODEL_FILES
            if not (MODEL_DIR / filename).is_file()
        ]
        if missing_models or not METRICS_PATH.is_file():
            raise FileNotFoundError
        test_data = load_test_data()
        results_df = load_metrics().sort_values(
            by="Accuracy",
            ascending=False,
        )
    except FileNotFoundError:
        st.error("Model artifacts or test_data.csv are missing.")
        st.code("python test_model.py", language="powershell")
        st.stop()

    # Show a ranked table of the models with their performance scores.
    st.subheader("Model comparison")
    st.dataframe(
        results_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            metric: st.column_config.NumberColumn(metric, format="%.4f")
            for metric in METRIC_COLUMNS
        },
    )

    # Explain the dataset size and the target class used in the metrics.
    st.caption(
        f"Evaluation set: {len(test_data):,} records. "
        "Metrics use the yes class for precision, recall, and F1."
    )


if __name__ == "__main__":
    # Start the dashboard when this Python file is run directly.
    main()
