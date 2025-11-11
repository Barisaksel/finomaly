import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator

class MLAnomalyModels:
    """
    MLAnomalyModels provides a unified interface for anomaly detection and classification models.
    Supports Isolation Forest (unsupervised), Random Forest, and XGBoost (supervised, binary).
    """
    def __init__(self, method='isolation_forest', random_state=42):
        """
        Args:
            method (str): Model type ('isolation_forest', 'random_forest', 'xgboost')
            random_state (int): Random seed for reproducibility
        """
        self.method = method
        self.model = None
        self.random_state = random_state
        self._build_model()

    def _build_model(self):
        """
        Initializes the model instance based on the selected method.
        Raises:
            ValueError: If method is not supported
        """
        match self.method:
            case 'isolation_forest':
                # Unsupervised anomaly detection
                self.model = IsolationForest(n_estimators=100, contamination=0.01, random_state=self.random_state)
            case 'random_forest':
                # Supervised classification
                self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            case 'xgboost':
                # Supervised binary classification (requires y in {0,1})
                self.model = XGBClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    objective='binary:logistic',
                    base_score=0.5
                )
            case _:
                raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X, y=None):
        """
        Trains the model on the provided data.
        Args:
            X: Feature matrix
            y: Target vector (required for supervised models)
        Raises:
            ValueError: If y is missing for supervised models, or if y is not binary for XGBoost
        """
        match self.method:
            case 'isolation_forest':
                # Unsupervised: no labels required
                self.model.fit(X)
            case 'random_forest' | 'xgboost':
                # Supervised: labels required
                if y is None:
                    raise ValueError(
                        "Supervised models (RandomForest, XGBoost) require labeled (y) data. Please provide y_train. Only unsupervised models (IsolationForest) can be used without labels."
                    )
                # Ensure y is integer numpy array
                if hasattr(y, 'values'):
                    y_fit = y.values.astype(int)
                else:
                    y_fit = np.array(y).astype(int)
                # XGBoost requires binary labels (0/1)
                if self.method == 'xgboost':
                    if y_fit.min() < 0 or y_fit.max() > 1:
                        raise ValueError(f"XGBoost requires y_train to contain only 0 and 1. Current min: {y_fit.min()} max: {y_fit.max()}")
                    if X.shape[0] != y_fit.shape[0]:
                        raise ValueError(f"X and y must have the same number of rows! X: {X.shape[0]}, y: {y_fit.shape[0]}")
                self.model.fit(X, y_fit)
            case _:
                raise ValueError(f"Unknown method: {self.method}")

    def predict(self, X):
        """
        Predicts using the trained model.
        Args:
            X: Feature matrix
        Returns:
            np.ndarray: Model predictions
        Raises:
            ValueError: If method is not supported
        """
        match self.method:
            case 'isolation_forest' | 'random_forest' | 'xgboost':
                return self.model.predict(X)
            case _:
                raise ValueError(f"Unknown method: {self.method}")
