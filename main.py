import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = Path("data")
train_path = DATA_DIR / "train.csv"
test_path = DATA_DIR / "test.csv"

if not train_path.exists():
    raise FileNotFoundError("Place Kaggle Titanic train.csv in the data/ folder.")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path) if test_path.exists() else None

target = "Survived"
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train[features].copy()
y = train[target].astype(int)

numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
valid_predictions = model.predict(X_valid)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_valid))
print("Validation accuracy:", round(accuracy_score(y_valid, valid_predictions), 4))
print("\nConfusion matrix:")
print(confusion_matrix(y_valid, valid_predictions))
print("\nClassification report:")
print(classification_report(y_valid, valid_predictions, digits=4))

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("5-fold cross-validation accuracy:", cv_scores.round(4))
print("Mean CV accuracy:", round(cv_scores.mean(), 4))

# If Kaggle's test.csv is present, create a submission file.
if test is not None:
    model.fit(X, y)
    test_predictions = model.predict(test[features])
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": test_predictions
    })
    submission.to_csv("submission.csv", index=False)
    print("\nCreated submission.csv")
