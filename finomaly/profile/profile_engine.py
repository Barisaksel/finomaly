import pandas as pd
import json
import os

class ProfileEngine:
    """
    ProfileEngine provides customer profiling and anomaly detection utilities for financial transactions.
    All error and user messages are centrally managed and multilingual.
    """
    def __init__(self, lang='en', messages_path=None):
        """
        Args:
            lang (str): Language for user-facing messages (e.g., 'en', 'tr')
            messages_path (str): Path to the JSON file containing messages
        """
        self.lang = lang
        if messages_path is None:
            # Default to core/messages_config.json relative to this file
            base = os.path.dirname(os.path.dirname(__file__))
            messages_path = os.path.join(base, 'core', 'messages_config.json')
        with open(messages_path, 'r', encoding='utf-8') as f:
            self.messages = json.load(f)

    def get_message(self, key):
        """
        Retrieves a language-supported message for the given key.
        Args:
            key (str): Message key
        Returns:
            str: Message in the selected language, or the key if not found
        """
        return self.messages.get(key, {}).get(self.lang, key)

    def time_series_anomaly(self, df, customer_col='MusteriID', amount_col='Tutar', window=10, threshold=3):
        """
        Detects time series anomalies for each customer using rolling Z-score.
        Args:
            df (pd.DataFrame): Input transaction data
            customer_col (str): Customer ID column name
            amount_col (str): Transaction amount column name
            window (int): Rolling window size
            threshold (float): Z-score threshold for anomaly
        Returns:
            list: List of anomaly labels ("TS_Anomaly" or empty string)
        """
        # Sort by customer and time for correct rolling calculation
        df = df.sort_values([customer_col, 'Saat'])
        anomalies = []
        for cust, group in df.groupby(customer_col):
            # Compute rolling mean and std for each customer
            amounts = group[amount_col].rolling(window, min_periods=1).mean()
            stds = group[amount_col].rolling(window, min_periods=1).std().fillna(0)
            zscores = (group[amount_col] - amounts) / (stds + 1e-6)
            # Flag as anomaly if Z-score exceeds threshold
            for z in zscores:
                if abs(z) > threshold:
                    anomalies.append('TS_Anomaly')
                else:
                    anomalies.append('')
        return anomalies

    def behavior_pattern_deviation(self, df, profiles, customer_col='MusteriID', amount_col='Tutar', freq_col='Saat', threshold=3):
        """
        Detects deviations from customer behavior patterns (amount, frequency, time, etc.).
        Args:
            df (pd.DataFrame): Input transaction data
            profiles (pd.DataFrame): Customer profile statistics (mean, std, max, min)
            customer_col (str): Customer ID column name
            amount_col (str): Transaction amount column name
            freq_col (str): Transaction time/frequency column name
            threshold (float): Z-score threshold for anomaly
        Returns:
            list: List of language-supported anomaly messages or empty string
        """
        results = []
        for _, row in df.iterrows():
            if row[customer_col] in profiles.index:
                mean = profiles.loc[row[customer_col], 'mean']
                std = profiles.loc[row[customer_col], 'std']
                max_ = profiles.loc[row[customer_col], 'max']
                min_ = profiles.loc[row[customer_col], 'min']
                # Example: flag if amount deviates from mean by threshold*std
                if std > 0 and abs(row[amount_col] - mean) > threshold * std:
                    results.append(self.get_message('behavior_deviation'))
                # Example: flag if transaction occurs at unusual time (e.g., night)
                elif row[freq_col] < 6 and mean > 1000:  # Example rule
                    results.append(self.get_message('unusual_time'))
                else:
                    results.append('')
            else:
                results.append('')
        return results

    def build_profile(self, df, customer_col='MusteriID', amount_col='Tutar'):
        """
        Builds customer profiles by aggregating statistics (mean, std, max, min).
        Args:
            df (pd.DataFrame): Input transaction data
            customer_col (str): Customer ID column name
            amount_col (str): Transaction amount column name
        Returns:
            pd.DataFrame: Profile statistics indexed by customer
        """
        return df.groupby(customer_col)[amount_col].agg(['mean', 'std', 'max', 'min'])

    def detect_deviation(self, df, profiles, customer_col='MusteriID', amount_col='Tutar', threshold=3):
        """
        Detects profile-based anomalies using Z-score deviation from mean.
        Args:
            df (pd.DataFrame): Input transaction data
            profiles (pd.DataFrame): Customer profile statistics (mean, std)
            customer_col (str): Customer ID column name
            amount_col (str): Transaction amount column name
            threshold (float): Z-score threshold for anomaly
        Returns:
            list: List of anomaly labels ("ProfileAnomaly" or empty string)
        """
        results = []
        for _, row in df.iterrows():
            if row[customer_col] in profiles.index:
                mean = profiles.loc[row[customer_col], 'mean']
                std = profiles.loc[row[customer_col], 'std']
                if std > 0 and abs(row[amount_col] - mean) > threshold * std:
                    results.append('ProfileAnomaly')
                else:
                    results.append('')
            else:
                results.append('')
        return results
