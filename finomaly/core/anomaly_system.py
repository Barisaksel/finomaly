
from finomaly.core.data_handler import DataHandler
from finomaly.rules.rule_engine import RuleEngine
from finomaly.ml.ml_models import MLAnomalyModels
from finomaly.profile.profile_engine import ProfileEngine
from finomaly.report.reporter import Reporter
import joblib
import os
import json


class CorporateAnomalySystem:
    """
    CorporateAnomalySystem orchestrates the full anomaly detection pipeline for financial data.
    Integrates data handling, rule-based and ML-based anomaly detection, profiling, and reporting.
    """
    def __init__(self, features, rules_path=None, ml_method='isolation_forest', lang='en', model_path=None, messages_path=None):
        """
        Args:
            features (list): List of feature column names to use
            rules_path (str): Path to rule definitions (JSON/Excel)
            ml_method (str): ML model type ('isolation_forest', 'random_forest', 'xgboost')
            lang (str): Language for user-facing messages
            model_path (str): Path to save/load trained ML model
            messages_path (str): Path to messages config JSON
        """
        self.data_handler = DataHandler(required_columns=features)
        self.rule_engine = RuleEngine(rules_path)
        self.ml_model = MLAnomalyModels(method=ml_method)
        self.profile_engine = ProfileEngine()
        self.reporter = Reporter(lang=lang)
        self.features = features
        self.lang = lang
        self.profiles = None
        self.model_path = model_path
        self.ml_method = ml_method
        # Load messages config (for localization)
        if messages_path is None:
            messages_path = os.path.join(os.path.dirname(__file__), 'messages_config.json')
        with open(messages_path, encoding='utf-8') as f:
            self.messages = json.load(f)
        # Load model if path is provided and file exists
        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path)

    def fit(self, excel_path, y=None, customer_col=None, amount_col=None, save_model=True):
        """
        Trains the ML model and builds customer profiles from the provided Excel data.
        Args:
            excel_path (str): Path to input Excel file
            y: Target labels (for supervised models)
            customer_col (str): Customer ID column name (for profiling)
            amount_col (str): Transaction amount column name (for profiling)
            save_model (bool): Whether to save the trained model
        Returns:
            self
        """
        df = self.data_handler.load_excel(excel_path)
        df_proc = self.data_handler.preprocess(df, fit_scaler=True)
        X = df_proc[self.features].values
        X_scaled = self.data_handler.scale_features(X, fit_scaler=True)
        self.ml_model.fit(X_scaled, y)
        # Build customer profiles if columns provided
        if customer_col and amount_col:
            self.profiles = self.profile_engine.build_profile(df, customer_col, amount_col)
        # Optionally save the trained model
        if save_model and self.model_path:
            self.save_model(self.model_path)
        return self

    def predict(self, excel_path, output_path=None, customer_col=None, amount_col=None):
        """
        Runs anomaly detection and reporting on new data.
        Args:
            excel_path (str): Path to input Excel file
            output_path (str): Path to save the output report (defaults to input file)
            customer_col (str): Customer ID column name (for profiling)
            amount_col (str): Transaction amount column name (for profiling)
        Returns:
            str: Path to the generated report
        """
        df = self.data_handler.load_excel(excel_path)
        df_proc = self.data_handler.preprocess(df, fit_scaler=False)
        X = df_proc[self.features].values
        X_scaled = self.data_handler.scale_features(X, fit_scaler=False)
        ml_preds = self.ml_model.predict(X_scaled)
        # Apply rule-based anomaly detection if rules are loaded
        rule_results = self.rule_engine.apply(df) if self.rule_engine.rules else [[] for _ in range(len(df))]
        # Determine anomaly label (localized if available)
        anomaly_label = self.messages['anomaly'][self.lang] if 'anomaly' in self.messages else 'Anomaly'
        if self.ml_method == 'isolation_forest':
            ml_anomaly = [anomaly_label if p == -1 else '' for p in ml_preds]
        else:
            ml_anomaly = [anomaly_label if p == 1 else '' for p in ml_preds]
        # Profile and behavioral anomaly analysis
        if self.profiles is not None and customer_col and amount_col:
            profile_results = self.profile_engine.detect_deviation(df, self.profiles, customer_col, amount_col)
            ts_anomaly = self.profile_engine.time_series_anomaly(df, customer_col, amount_col)
            behavior_dev = self.profile_engine.behavior_pattern_deviation(df, self.profiles, customer_col, amount_col, freq_col='Saat')
        else:
            profile_results = [''] * len(df)
            ts_anomaly = [''] * len(df)
            behavior_dev = [''] * len(df)
        # Use localized column names if available
        ml_col = self.messages['result_column'][self.lang] if 'result_column' in self.messages else 'Anomaly'
        rule_col = 'Rule_Anomaly'
        profile_col = 'Profile_Anomaly'
        ts_col = 'TS_Anomaly'
        behavior_col = 'Behavior_Deviation'
        # Add results to DataFrame
        df[ml_col] = ml_anomaly
        df[rule_col] = [','.join(r) if r else '' for r in rule_results]
        df[profile_col] = profile_results
        df[ts_col] = ts_anomaly
        df[behavior_col] = behavior_dev
        # Save or overwrite report
        if output_path is None:
            output_path = excel_path
        self.reporter.generate_report(df, output_path)
        return output_path

    def save_model(self, path):
        """
        Saves the trained ML model to disk.
        Args:
            path (str): File path
        """
        self.ml_model.save(path)
        self.model_path = path

    def load_model(self, path):
        """
        Loads a trained ML model from disk.
        Args:
            path (str): File path
        """
        self.ml_model.load(path)
        self.model_path = path
