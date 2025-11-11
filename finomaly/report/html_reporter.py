import pandas as pd

class HTMLReporter:
    """
    HTMLReporter generates HTML reports from pandas DataFrames.
    """
    def __init__(self):
        pass

    def generate_html_report(self, df, output_path):
        """
        Generates an HTML report from a DataFrame.
        Args:
            df (pd.DataFrame): Data to include in the report
            output_path (str): File path to save the HTML
        Returns:
            str: Path to the generated HTML file
        """
        html = df.to_html(index=False)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return output_path
