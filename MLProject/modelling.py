import os
import pandas as pd
import mlflow
import mlflow.sklearn  # This is needed for autolog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# File Location: mlruns 
current_dir = os.path.dirname(os.path.abspath(__file__))
mlflow.set_tracking_uri(f"file://{os.path.join(current_dir, 'mlruns')}")

mlflow.sklearn.autolog() # type: ignore

# Load data
dataset_path = "wine_preprocessing.csv"
dataset = pd.read_csv(dataset_path)
X = dataset.drop(columns=["Class"])
y = dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Set experiment
mlflow.set_experiment("Wine_Classification_Experiment")

# Start run
with mlflow.start_run(run_name="RandomForest_Basic"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", float(acc))

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
