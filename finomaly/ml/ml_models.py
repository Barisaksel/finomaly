from sklearn.ensemble import IsolationForest, RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

import joblib


class MLAnomalyModels:
    """
    MLAnomalyModels provides a unified interface for training and using different anomaly detection and classification models.
    Supported methods: Isolation Forest, Random Forest, XGBoost (if installed).
    """
    def __init__(self, method='isolation_forest', **kwargs):
        """
        Args:
            method (str): Model type ('isolation_forest', 'random_forest', 'xgboost')
            **kwargs: Additional parameters for the model constructor
        """
        self.method = method
        self.model = None
        self.kwargs = kwargs
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Trains the selected model on the provided data.
        Args:
            X: Feature matrix
            y: Target vector (required for supervised models)
        Returns:
            self
        Raises:
            ImportError: If XGBoost is selected but not installed
            ValueError: If method is not supported
        """
        match self.method:
            case 'isolation_forest':
                self.model = IsolationForest(**self.kwargs)
                self.model.fit(X)
                self.is_fitted = True
            case 'random_forest':
                self.model = RandomForestClassifier(**self.kwargs)
                self.model.fit(X, y)
                self.is_fitted = True
            case 'xgboost':
                if XGBClassifier is None:
                    raise ImportError('xgboost is not installed')
                self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **self.kwargs)
                self.model.fit(X, y)
                self.is_fitted = True
            case _:
                raise ValueError(f"Unsupported method: {self.method}")
        return self

    def predict(self, X):
        """
        Predicts using the trained model.
        Args:
            X: Feature matrix
        Returns:
            np.ndarray: Model predictions
        Raises:
            RuntimeError: If model is not trained
        """
        if self.model is None or not self.is_fitted:
            raise RuntimeError('Model is not trained.')
        return self.model.predict(X)

    def save(self, path):
        """
        Saves the trained model to disk.
        Args:
            path (str): File path (without extension for XGBoost)
        """
        if self.method == 'xgboost' and self.model is not None:
            self.model.save_model(path + '.json')
        else:
            joblib.dump(self.model, path)

    def load(self, path):
        """
        Loads a trained model from disk.
        Args:
            path (str): File path (without extension for XGBoost)
        Raises:
            ImportError: If XGBoost is selected but not installed
        """
        if self.method == 'xgboost':
            if XGBClassifier is None:
                raise ImportError('xgboost is not installed')
            self.model = XGBClassifier()
            self.model.load_model(path + '.json')
            self.is_fitted = True
        else:
            self.model = joblib.load(path)
            self.is_fitted = True
