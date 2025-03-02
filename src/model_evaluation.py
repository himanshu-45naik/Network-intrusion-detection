import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from abc import ABC, abstractmethod

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EvaluateModelStrategy(ABC):
    """Strategy for evaluating model."""

    @abstractmethod
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, model):
        """Performs evaluation on the given model.

        Args:
            X_test (pd.DataFrame): The test data.
            y_test (pd.series): The test data.
            model (_type_): The model to be evaluated.
        """
        pass


class LogisticRegressionEvaluation(EvaluateModelStrategy):
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, model):
        """
        Evaluates the given model on the test data and logs evaluation metrics.

        Args:
            model: The trained model to evaluate.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True test target values.
        """
        logging.info("Evaluating the model on the test set.")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        logging.info("Test Accuracy: %.4f", acc)
        logging.info("Classification Report:\n%s", report)
        logging.info("Confusion Matrix:\n%s", cm)

        return acc, report, cm


class ModelEvaluator:
    def __init__(self, strategy: EvaluateModelStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: EvaluateModelStrategy):
        self._strategy = strategy

    def execute_strategy(self, X_test: pd.DataFrame, y_test: pd.Series, model):
        """Executes the given strategy.

        Args:
            X_test (pd.DataFrame): The test data.
            y_test (pd.Series): The test data.
            model : The model to be evaluated.
        """

        return self._strategy.evaluate_model(X_test, y_test, model)


if __name__ == "__main__":
    pass
