# Bank Marketing Classification

## a. Problem statement

This project predicts whether a bank customer will subscribe to a term deposit. The model uses customer details and information about the bank's contact with the customer. In the target column `y`, `yes` means the customer subscribed and `no` means they did not.

## b. Dataset description

The project uses the Bank Marketing `bank-full.csv` dataset, containing 45,211 records, 16 input features, and the binary target `y`. It has no missing values. The target is imbalanced: 39,922 clients did not subscribe and 5,289 did.

The 20% holdout in `test_data.csv` was created with stratification and `random_state=42` (9,043 records). The remaining 80% is used for training. Categorical inputs are one-hot encoded and numeric inputs are standardized; these preprocessing steps are fitted on training data only.
- Dataset Name: Bank Marketing Dataset (bank-full.csv) | Source: UCI Machine Learning Repository.

**Dataset Size:** The dataset contains 45,211 customer records and 16 input features, meeting the minimum dataset size requirement.
**Target Variable:** The target variable is y, which indicates whether a customer subscribed to a term deposit. A total of 5,289 customers (11.7%) subscribed (yes), while 39,922 customers (88.3%) did not subscribe (no). This shows that the dataset has an imbalanced class distribution.
**Feature Details:** The dataset includes information about customers, such as age, job, marital status, education, account balance, housing loan, and personal loan. It also contains details about marketing campaigns, including contact method, call duration, number of contacts, and previous campaign outcomes.
**Data Splitting and Preprocessing:** The dataset was divided into 80% training data (36,168 records) and 20% testing data (9,043 records) using stratified sampling with random_state=42. Categorical features were converted into numerical values using one-hot encoding, while numerical features were standardized. These preprocessing steps were applied using the training data to avoid data leakage. The dataset does not contain any missing values.
**Important Note:** The duration feature represents the length of the phone call. Since this information is available only after the call is completed, models using this feature may not reflect how accurately the bank can identify potential customers before making a call.

## c. GitHub Repository Link

[ML_Assignment_2](https://github.com/2026ak05378/ML_Assignment_2)

## d. Models used

All models use the same train/test split. Precision, recall, and F1 treat `yes` as the positive class. The SVM AUC is calculated from its decision scores.

Model configurations:

- **Logistic Regression:** `max_iter=1000`.
- **Decision Tree:** `max_depth=5`, `random_state=42`.
- **K-Nearest Neighbors:** `n_neighbors=5`.
- **Gaussian Naive Bayes:** Gaussian likelihood; encoded input is converted to a dense matrix for fitting.
- **Support Vector Machine:** RBF kernel, `random_state=42`.

The evaluator calculates accuracy, ROC AUC, precision, recall, F1, and Matthews correlation coefficient (MCC). AUC uses predicted probabilities where available and the SVM decision function otherwise.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8995 | 0.8572 | 0.6499 | 0.3053 | 0.4154 | 0.3998 |
| KNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes (Gaussian) | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Support Vector Machine | 0.9049 | 0.9036 | 0.6875 | 0.3431 | 0.4578 | 0.4420 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieves the highest AUC (0.9056), indicating the strongest ranking performance, with accuracy close to the SVM. Recall is modest at 0.3478. |
| Decision Tree | Has 0.8995 accuracy, but lower AUC (0.8572) and the lowest recall (0.3053), so it misses more actual subscribers at the default threshold. |
| KNN | Reaches 0.8962 accuracy, but its AUC (0.8277) is the lowest among the models. Precision and recall are both moderate. |
| Naive Bayes | Has the highest recall (0.5198) and F1 (0.4559), but lower accuracy and precision. It catches more subscribers at the cost of more false positives. |
| Support Vector Machine | Leads in accuracy (0.9049), precision (0.6875), and MCC (0.4420), with AUC close to Logistic Regression. Its recall (0.3431) remains modest. |

**Overall winner:** Support Vector Machine by accuracy, precision, and MCC. There is no single winner across every metric: Logistic Regression has the best AUC, while Gaussian Naive Bayes has the best recall and F1. Given the class imbalance, accuracy alone should not determine model choice.

## Step-by-step workflow

### Step 1: Set up Python

Open PowerShell in the project root. Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependencies are declared in `requirements.txt`: Streamlit, scikit-learn, pandas, and joblib. Use `python -m ...` so the commands run through the active environment's interpreter.

### Step 2: Prepare and verify the data split

The raw dataset is `data/bank-full.csv`, a semicolon-separated CSV. `test_data.csv` contains the stratified 20% test split (9,043 records); the other 80% (36,168 records) is used for training. Both splits use `random_state=42` and preserve the original `yes`/`no` target labels.

To recreate `test_data.csv` after replacing the source dataset, run this from the project root:

```python
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(Path("data") / "bank-full.csv", sep=";")
_, test_data = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["y"],
)
test_data.reset_index(drop=True).to_csv("test_data.csv", index=False)
```

The evaluator checks that the committed `test_data.csv` matches this split before fitting models. If it does not match, correct the holdout rather than evaluating against a different split.

### Step 3: Train and save the classifiers

Run the training and evaluation script:

```powershell
python test_model.py
```

The script one-hot encodes categorical features and standardizes numeric features. The preprocessor is fitted using training data only, then reused to transform the test data. All five classifiers are fitted on the same transformed training split. The script prints a comparison table and saves the fitted preprocessing transformer and classifiers as joblib artifacts under `model/`.

### Step 4: Review evaluation results

The metrics table is sorted by accuracy and also saved as `model/evaluation_metrics.csv`. For this dataset, the SVM leads in accuracy, precision, and MCC; Logistic Regression has the best AUC; and Gaussian Naive Bayes has the best recall and F1. Because the target is imbalanced, compare recall, precision, F1, AUC, and MCC along with accuracy.

### Step 5: Run the dashboard locally

Start the app from the project root after the model artifacts have been generated:

```powershell
python -m streamlit run app.py
```

Open the localhost URL printed by Streamlit (normally `http://localhost:8501`). `app.py` is the entry point; it launches the dashboard in `streamlit_app.py`. The dashboard displays the saved test metrics and checks that all model and preprocessing artifacts are present. It does not retrain the classifiers on each page load.

## Notes and limitations

- The target is imbalanced: 39,922 records are `no` and 5,289 are `yes`. The default classification threshold favors overall accuracy and can miss subscribers.
- The `duration` feature is only known after the call. Since it is included in this dataset's model inputs, these scores should not be interpreted as pre-call targeting performance.
- The supplied `test_data.csv` is part of the reproducible evaluation. Do not use it to tune model settings if it must remain an unbiased final test set.

## Project files

```text
project-folder/
├── app.py
├── streamlit_app.py
├── test_model.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── bank-full.csv
└── model/
    ├── preprocessor.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── evaluation_metrics.csv
```
