# Titanic Survival Prediction

A machine-learning classification project that predicts whether a Titanic passenger survived using Logistic Regression.

## Dataset
Download the Kaggle Titanic competition files:
- `train.csv` (required)
- `test.csv` (optional, used to create a Kaggle submission)

Place them in the `data/` folder.

## Features used
Passenger class, sex, age, number of siblings/spouses aboard, number of parents/children aboard, fare and embarkation port.

Missing numeric values are filled using the median. Missing categorical values are filled using the most frequent value. Categorical columns are one-hot encoded and numeric columns are standardized.

## Run
```bash
pip install -r requirements.txt
python main.py
```

## Example output format
```text
Training rows: [number]
Validation rows: [number]
Validation accuracy: [score]

Confusion matrix:
[[... ...]
 [... ...]]

Classification report:
              precision    recall  f1-score   support
           0       ...
           1       ...

5-fold cross-validation accuracy: [...]
Mean CV accuracy: [score]

Created submission.csv
```
Exact metrics depend on the dataset and split. `submission.csv` is created only when `data/test.csv` is present.

## Repository structure
```text
titanic-survival-prediction-ml/
├── data/
│   ├── train.csv
│   └── test.csv
├── main.py
├── requirements.txt
└── README.md
```
