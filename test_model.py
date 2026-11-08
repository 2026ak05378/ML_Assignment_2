from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "bank-full.csv"
TEST_DATA_PATH = ROOT / "test_data.csv"
MODEL_DIR = ROOT / "model"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.csv"
TARGET = "y"
POSITIVE_LABEL = "yes"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


def main():
    data = pd.read_csv(DATA_PATH, sep=";")
    train_data, expected_test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data[TARGET],
    )
    test_data = pd.read_csv(TEST_DATA_PATH)
    expected_test_data = expected_test_data.reset_index(drop=True)
    pd.testing.assert_frame_equal(
        test_data,
        expected_test_data,
        check_dtype=False,
        check_exact=True,
    )

    X_train = train_data.drop(columns=TARGET)
    y_train = train_data[TARGET]
    X_test = test_data.drop(columns=TARGET)
    y_test = test_data[TARGET]

    categorical_features = X_train.select_dtypes(include="object").columns
    numerical_features = X_train.select_dtypes(exclude="object").columns
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            ("numerical", StandardScaler(), numerical_features),
        ]
    )
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", random_state=42),
    }

    results = []
    for model_name, model in models.items():
        if isinstance(model, GaussianNB):
            X_train_model = (
                X_train_encoded.toarray()
                if hasattr(X_train_encoded, "toarray")
                else X_train_encoded
            )
            X_test_model = (
                X_test_encoded.toarray()
                if hasattr(X_test_encoded, "toarray")
                else X_test_encoded
            )
        else:
            X_train_model = X_train_encoded
            X_test_model = X_test_encoded

        model.fit(X_train_model, y_train)
        joblib.dump(model, MODEL_DIR / MODEL_FILES[model_name])
        predictions = model.predict(X_test_model)
        if hasattr(model, "predict_proba"):
            positive_index = list(model.classes_).index(POSITIVE_LABEL)
            scores = model.predict_proba(X_test_model)[:, positive_index]
        else:
            scores = model.decision_function(X_test_model)

        results.append(
            {
                "ML Model Name": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC Score": roc_auc_score(
                    y_test == POSITIVE_LABEL,
                    scores,
                ),
                "Precision": precision_score(
                    y_test,
                    predictions,
                    pos_label=POSITIVE_LABEL,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_test,
                    predictions,
                    pos_label=POSITIVE_LABEL,
                    zero_division=0,
                ),
                "F1 Score": f1_score(
                    y_test,
                    predictions,
                    pos_label=POSITIVE_LABEL,
                    zero_division=0,
                ),
                "MCC Score": matthews_corrcoef(y_test, predictions),
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by="Accuracy",
        ascending=False,
    )
    results_df.to_csv(METRICS_PATH, index=False)
    print("\nMODEL PERFORMANCE COMPARISON\n")
    print(
        results_df.to_string(
            index=False,
            formatters={
                metric: "{:.4f}".format
                for metric in [
                    "Accuracy",
                    "AUC Score",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "MCC Score",
                ]
            },
        )
    )


if __name__ == "__main__":
    main()