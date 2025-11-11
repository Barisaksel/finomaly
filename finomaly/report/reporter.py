import pandas as pd

class Reporter:
    """
    Reporter generates Excel reports from pandas DataFrames.
    """
    def __init__(self, lang='en'):
        """
        Args:
            lang (str): Language for future localization (not used in this basic implementation)
        """
        self.lang = lang

    def generate_report(self, df, output_path):
        """
        Generates an Excel report from a DataFrame.
        Args:
            df (pd.DataFrame): Data to include in the report
            output_path (str): File path to save the Excel file
        Returns:
            str: Path to the generated Excel file
        """
        df.to_excel(output_path, index=False)
        return output_path
