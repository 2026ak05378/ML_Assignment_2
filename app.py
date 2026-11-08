# ==========================================
# ML MODELS & EVALUATION METRICS
# ==========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("bank-full.csv", sep=';')

print("Dataset Shape:", df.shape)
df.head()

# ==========================================
# TARGET ENCODING
# ==========================================

label_encoder = LabelEncoder()
df['y'] = label_encoder.fit_transform(df['y'])

# no = 0
# yes = 1

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df.drop('y', axis=1)
y = df['y']

# ==========================================
# IDENTIFY CATEGORICAL & NUMERIC FEATURES
# ==========================================

categorical_features = X.select_dtypes(
    include=['object']
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=['object']
).columns.tolist()

# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_features
        )
    ],
    remainder='passthrough'
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================================
# DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "Naive Bayes":
        GaussianNB(),

    "Support Vector Machine":
        SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
}

# ==========================================
# EVALUATION FUNCTION
# ==========================================

results = []

for model_name, model in models.items():

    if model_name == "Naive Bayes":

        X_train_encoded = preprocessor.fit_transform(X_train)
        X_test_encoded = preprocessor.transform(X_test)

        model.fit(
            X_train_encoded.toarray(),
            y_train
        )

        y_pred = model.predict(
            X_test_encoded.toarray()
        )

        y_prob = model.predict_proba(
            X_test_encoded.toarray()
        )[:, 1]

    else:

        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(X_test)

        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    results.append([
        model_name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

# ==========================================
# RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=[
        "ML Model Name",
        "Accuracy",
        "AUC Score",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC Score"
    ]
)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\nMODEL PERFORMANCE COMPARISON\n")

display(
    results_df.style.format({
        "Accuracy": "{:.4f}",
        "AUC Score": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1 Score": "{:.4f}",
        "MCC Score": "{:.4f}"
    })
)
