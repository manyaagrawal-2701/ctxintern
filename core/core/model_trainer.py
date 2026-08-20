# ==============================================================================
# CLASS: ModelTrainer
# PURPOSE: Manages the machine learning classification models.
#          Initializes models (Random Forest, Gradient Boosting, SVM, XGBoost),
#          trains them, evaluates their accuracy/F1 metrics, and saves the winner.
# ==============================================================================

import joblib
import logging
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# Imports evaluation metric functions from scikit-learn:
# - Accuracy: % of correct predictions.
# - Precision: Out of all positive predictions, how many were actually correct?
# - Recall: Out of all actual positives, how many did the model find?
# - F1 Score: Balanced average of Precision and Recall (best overall metric for noisy data).
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Safe imports for optional XGBoost. 
# It falls back to standard models if XGBoost is not installed on the system.
from sklearn.preprocessing import LabelEncoder

# Safe imports for optional XGBoost. 
# It falls back to standard models if XGBoost is not installed on the system.
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostWrapper:
    """Wrapper around XGBClassifier to handle string target labels seamlessly."""

    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X, y):
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.model.fit(X, y_encoded)
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ModelTrainer:
    """
    Manages model initialization, training, evaluation, comparison, and serialization.
    """

    def __init__(self):
        """
        Constructor. Sets up the classifier algorithms list.
        """
        self.models = {
            # Random Forest: An ensemble of decision trees voting on the segment.
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ),
            # Gradient Boosting: Trees built sequentially, learning from past errors.
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42
            ),
            # Support Vector Machine (SVM): Finds optimal hyperplane boundaries.
            "SVM": SVC(
                probability=True,
                random_state=42
            )
        }

        # If XGBoost is installed, add it to the comparison list
        if XGBOOST_AVAILABLE:
            self.models["XGBoost"] = XGBoostWrapper(
                eval_metric="mlogloss",
                random_state=42
            )

        self.best_model = None
        self.best_model_name = None
        self.results = []

    # --------------------------------------------------------------------------
    # INDIVIDUAL TRAINING METHOD MODULES (Strict OOP)
    # --------------------------------------------------------------------------

    def train_random_forest(self, X_train, y_train):
        """Trains only the Random Forest Classifier on training inputs."""
        model = self.models["Random Forest"]
        model.fit(X_train, y_train)
        return model

    def train_gradient_boosting(self, X_train, y_train):
        """Trains only the Gradient Boosting Classifier on training inputs."""
        model = self.models["Gradient Boosting"]
        model.fit(X_train, y_train)
        return model

    def train_svm(self, X_train, y_train):
        """Trains only the Support Vector Machine (SVM) on training inputs."""
        model = self.models["SVM"]
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train):
        """Trains only the XGBoost Classifier on training inputs."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed.")
        model = self.models["XGBoost"]
        model.fit(X_train, y_train)
        return model

    # --------------------------------------------------------------------------
    # EVALUATION & COMPARISON
    # --------------------------------------------------------------------------

    def evaluate_model(self, model, X_test, y_test):
        """
        Tests a model against unseen holdout test data and returns scores.
        - average='weighted' adjusts for class size imbalances.
        - zero_division=0 prevents crashes if a category has no predictions.
        """
        prediction = model.predict(X_test)

        return {
            "Accuracy": accuracy_score(y_test, prediction),
            "Precision": precision_score(y_test, prediction, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, prediction, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, prediction, average="weighted", zero_division=0)
        }

    def compare_models(self, X_train, X_test, y_train, y_test):
        """
        Trains and compares all models in our dictionary.
        Returns a sorted DataFrame comparing metrics, selecting the best model 
        based on the highest F1 Score.
        """
        self.results = []
        best_f1 = -1

        # Loop through each model name and model object
        for name, model in self.models.items():
            logging.info(f"Training {name}...")
            
            # 1. Fit (Train) the model
            model.fit(X_train, y_train)
            
            # 2. Evaluate scores
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics["Model"] = name
            self.results.append(metrics)

            # 3. Track the winner based on F1 Score
            if metrics["F1 Score"] > best_f1:
                best_f1 = metrics["F1 Score"]
                self.best_model = model
                self.best_model_name = name

        # 4. Convert results list to a neat Pandas DataFrame for display
        results_df = pd.DataFrame(self.results)
        results_df = results_df[
            [
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ]

        # Sort the table so the best performing model sits at row index 0.
        return results_df.sort_values(by="F1 Score", ascending=False)

    # --------------------------------------------------------------------------
    # SAVING MODEL (Serialization)
    # --------------------------------------------------------------------------

    def save_best_model(self, path="models/best_model.pkl"):
        """
        Saves the best trained model to a file using 'joblib'.
        Think of this as pausing a video game: it serializes the model state so we
        can load it instantly on the server without having to train it again.
        """
        if self.best_model is None:
            raise ValueError("No trained model available.")

        # Serialize model file
        joblib.dump(self.best_model, path)
        logging.info(f"Best model saved as {path}")
        print(f"\nBest Model : {self.best_model_name}")
        print("Model saved successfully!")
