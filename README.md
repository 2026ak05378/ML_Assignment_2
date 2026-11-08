# Bank Marketing Classification

## a. Problem statement

Predict whether a bank client subscribes to a term deposit (`y = yes`) using client, campaign, and previous-contact information. The positive class is `yes`.

## b. Dataset description

The project uses the Bank Marketing `bank-full.csv` dataset, containing 45,211 records, 16 input features, and the binary target `y`. It has no missing values. The target is imbalanced: 39,922 clients did not subscribe and 5,289 did.

The 20% holdout in `test_data.csv` was created with stratification and `random_state=42` (9,043 records). The remaining 80% is used for training. Categorical inputs are one-hot encoded and numeric inputs are standardized; these preprocessing steps are fitted on training data only.

**Interpretation note:** the feature `duration` is the length of the call and is only known after the call. Results that use it should not be interpreted as pre-call targeting performance.

## c. GitHub Repository Link

[ML_Assignment_2](https://github.com/2026ak05378/ML_Assignment_2)

## d. Models used

All models use the same train/test split. Precision, recall, and F1 treat `yes` as the positive class. The SVM AUC is calculated from its decision scores.

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

## Run the project

From the project directory, install dependencies, train and evaluate the models, then launch the dashboard:

```powershell
python -m pip install -r requirements.txt
python test_model.py
streamlit run app.py
```

`test_model.py` verifies the holdout against the source dataset, prints the comparison table, saves the fitted preprocessing transformer and five classifiers under `model/`, and writes the evaluation table to `model/evaluation_metrics.csv`. The Streamlit dashboard checks that all model artifacts exist and displays the saved metrics for `test_data.csv`.

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
