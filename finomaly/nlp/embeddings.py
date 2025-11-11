
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import json

 # TextEmbeddingAnomalyDetector: Embedding-based text anomaly detection model
 # Uses TF-IDF vectorization and IsolationForest. All error and user messages are centrally managed and multilingual.
class TextEmbeddingAnomalyDetector:
    def __init__(self, contamination=0.01, random_state=42, lang='en', messages_path=None):
        """
        Args:
            contamination (float): Anomaly ratio (IsolationForest parameter)
            random_state (int): Seed for reproducibility
            lang (str): Language for user-facing messages (e.g., 'en', 'tr')
            messages_path (str): Path to the JSON file containing messages
        """
        # Vectorize texts using TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
        # Detect anomalies with IsolationForest
        self.model = IsolationForest(contamination=contamination, random_state=random_state)
        self.is_fitted = False  # Has the model been trained?
        self.lang = lang  # Message language
        # Load message configuration file
        if messages_path is None:
            messages_path = os.path.join(os.path.dirname(__file__), '../core/messages_config.json')
        with open(messages_path, encoding='utf-8') as f:
            self.messages = json.load(f)

    def _extract_texts(self, texts_or_path, column=None):
        """
        Converts the input into a suitable list of texts.
        - If an Excel file path and column name are provided, reads the column.
        - If a list, numpy array, or pandas Series, converts all to string.
        - If a single string, wraps it in a list.
        - If the input is not valid, raises a language-supported error message.
        """
        if isinstance(texts_or_path, str) and column is not None:
            # Read texts from Excel file
            df = pd.read_excel(texts_or_path)
            return df[column].astype(str).tolist()
        elif isinstance(texts_or_path, (list, np.ndarray, pd.Series)):
            # Convert list, array, or Series to string
            return [str(t) for t in texts_or_path]
        elif isinstance(texts_or_path, str):
            # Wrap single string in a list
            return [texts_or_path]
        else:
            # Invalid input type, return language-supported error message
            msg = self.messages.get('input_type_error', {}).get(self.lang, "Input must be a list, Series, single string, or Excel file path with column name.")
            raise ValueError(msg)

    def fit(self, texts, column=None):
        """
        Trains the model on the provided texts.
        Args:
            texts: Training data (list, string, Excel file, etc.)
            column: Column name for Excel file
        Returns:
            self
        """
        texts_list = self._extract_texts(texts, column)
        X = self.vectorizer.fit_transform(texts_list)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict(self, texts, column=None):
        """
        Predicts anomalies using the trained model.
        Args:
            texts: Test data (list, string, Excel file, etc.)
            column: Column name for Excel file
        Returns:
            np.ndarray: Language-supported label for detected anomalies, None for normal cases
        """
        if not self.is_fitted:
            # Raise error if prediction is called before fitting
            msg = self.messages.get('not_trained', {}).get(self.lang, "Model must be fitted before prediction.")
            raise RuntimeError(msg)
        texts_list = self._extract_texts(texts, column)
        X = self.vectorizer.transform(texts_list)
        preds = self.model.predict(X)
        anomaly_label = self.messages.get('anomaly', {}).get(self.lang, 'TextAnomaly')
        return np.where(preds == -1, anomaly_label, None)

    def fit_predict(self, texts, column=None):
        """
        Trains the model and predicts on the same data (for pipeline convenience).
        Args:
            texts: Training and test data
            column: Column name for Excel file
        Returns:
            np.ndarray: Language-supported label for detected anomalies, None for normal cases
        """
        self.fit(texts, column)
        return self.predict(texts, column)
