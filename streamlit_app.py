from pathlib import Path

import pandas as pd
import streamlit as st


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


@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_PATH)


def main():
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

    st.caption(
        f"Evaluation set: {len(test_data):,} records. "
        "Metrics use the yes class for precision, recall, and F1."
    )


if __name__ == "__main__":
    main()
