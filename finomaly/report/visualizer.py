import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    """
    Visualizer provides plotting utilities for anomaly detection results and feature relationships.
    """
    def __init__(self):
        pass

    def plot_anomaly_distribution(self, df, amount_col='Tutar', anomaly_col='ML_Anomaly', show=False, return_fig=False):
        """
        Plots the distribution of transaction amounts, highlighting anomalies.
        Args:
            df (pd.DataFrame): Data containing amounts and anomaly labels
            amount_col (str): Column name for transaction amounts
            anomaly_col (str): Column name for anomaly labels
            show (bool): Whether to display the plot
            return_fig (bool): Whether to return the matplotlib Figure object
        Returns:
            matplotlib.figure.Figure or None
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[amount_col], bins=30, kde=True, color='gray', label='All', ax=ax)
        if anomaly_col in df.columns:
            anomalies = df[df[anomaly_col] == 'Anomaly']
            sns.histplot(anomalies[amount_col], bins=30, color='red', label='Anomaly', ax=ax)
        ax.legend()
        ax.set_title('Anomaly Distribution')
        ax.set_xlabel(amount_col)
        ax.set_ylabel('Count')
        if show:
            plt.show()
        else:
            plt.close(fig)
        if return_fig:
            return fig

    def plot_feature_scatter(self, df, x_col, y_col, anomaly_col='ML_Anomaly', show=False, return_fig=False):
        """
        Plots a scatterplot of two features, coloring anomalies.
        Args:
            df (pd.DataFrame): Data containing features and anomaly labels
            x_col (str): Column name for x-axis
            y_col (str): Column name for y-axis
            anomaly_col (str): Column name for anomaly labels
            show (bool): Whether to display the plot
            return_fig (bool): Whether to return the matplotlib Figure object
        Returns:
            matplotlib.figure.Figure or None
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=anomaly_col, palette={'Anomaly':'red', '':'blue'}, ax=ax)
        ax.set_title(f'{x_col} vs {y_col} (Anomalies Highlighted)')
        if show:
            plt.show()
        else:
            plt.close(fig)
        if return_fig:
            return fig
