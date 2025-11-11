import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


import os
import json

class TransactionDescriptionClassifier:
    """
    A text classifier for transaction descriptions using TF-IDF and Logistic Regression.
    All error and user messages are centrally managed and multilingual.
    """
    def __init__(self, categories=None, random_state=42, lang='en', messages_path=None):
        """
        Args:
            categories (list): List of possible categories (optional, for reference)
            random_state (int): Seed for reproducibility
            lang (str): Language for user-facing messages (e.g., 'en', 'tr')
            messages_path (str): Path to the JSON file containing messages
        """
        self.categories = categories
        # Pipeline: TF-IDF vectorizer + Logistic Regression classifier
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=500)),
            ("clf", LogisticRegression(max_iter=200, random_state=random_state))
        ])
        self.label_encoder = LabelEncoder()  # Encodes string labels as integers
        self.is_fitted = False  # Has the model been trained?
        self.lang = lang  # Message language
        # Load message configuration file
        if messages_path is None:
            messages_path = os.path.join(os.path.dirname(__file__), '../core/messages_config.json')
        with open(messages_path, encoding='utf-8') as f:
            self.messages = json.load(f)

    def fit(self, texts, labels):
        """
        Trains the classifier on the provided texts and labels.
        Args:
            texts: List of input texts (transaction descriptions)
            labels: List of category labels
        Returns:
            self
        Raises:
            ValueError: If texts or labels are missing
        """
        if texts is None or labels is None:
            msg = self.messages.get('fit_missing', {}).get(self.lang, "Both texts and labels are required for fitting.")
            raise ValueError(msg)
        y = self.label_encoder.fit_transform(labels)
        self.pipeline.fit(texts, y)
        self.is_fitted = True
        return self

    def predict(self, texts):
        """
        Predicts categories for the given texts using the trained model.
        Args:
            texts: List of input texts (transaction descriptions)
        Returns:
            np.ndarray: Predicted category labels
        Raises:
            RuntimeError: If model is not trained
            ValueError: If texts are missing
        """
        if not self.is_fitted:
            msg = self.messages.get('not_trained', {}).get(self.lang, "Model must be fitted before prediction.")
            raise RuntimeError(msg)
        if texts is None:
            msg = self.messages.get('predict_missing', {}).get(self.lang, "Texts are required for prediction.")
            raise ValueError(msg)
        y_pred = self.pipeline.predict(texts)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, texts):
        """
        Returns class probabilities for the given texts using the trained model.
        Args:
            texts: List of input texts (transaction descriptions)
        Returns:
            np.ndarray: Probability estimates for each class
        Raises:
            RuntimeError: If model is not trained
            ValueError: If texts are missing
        """
        if not self.is_fitted:
            msg = self.messages.get('not_trained', {}).get(self.lang, "Model must be fitted before prediction.")
            raise RuntimeError(msg)
        if texts is None:
            msg = self.messages.get('predict_missing', {}).get(self.lang, "Texts are required for prediction.")
            raise ValueError(msg)
        proba = self.pipeline.predict_proba(texts)
        return proba

    def fit_predict(self, texts, labels):
        """
        Trains the model and predicts on the same data (for pipeline convenience).
        Args:
            texts: List of input texts (transaction descriptions)
            labels: List of category labels
        Returns:
            np.ndarray: Predicted category labels
        """
        self.fit(texts, labels)
        return self.predict(texts)

    def get_categories(self):
        """
        Returns the list of category labels learned by the model.
        Returns:
            list: Category label names
        """
        return list(self.label_encoder.classes_)
