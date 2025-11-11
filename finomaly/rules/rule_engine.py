import json


import pandas as pd

class RuleEngine:
    """
    RuleEngine applies rule-based anomaly detection to tabular data.
    Rules can be loaded from JSON or Excel files, or added programmatically.
    """
    def __init__(self, rules_path=None):
        """
        Args:
            rules_path (str): Path to rules file (.json or .xlsx/.xls)
        """
        self.rules = []
        if rules_path:
            if rules_path.endswith('.json'):
                self.load_rules_json(rules_path)
            elif rules_path.endswith('.xlsx') or rules_path.endswith('.xls'):
                self.load_rules_excel(rules_path)

    def load_rules_json(self, path):
        """
        Loads rules from a JSON file.
        Args:
            path (str): Path to JSON file
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.rules = json.load(f)

    def load_rules_excel(self, path):
        """
        Loads rules from an Excel file.
        Args:
            path (str): Path to Excel file
        """
        df = pd.read_excel(path)
        self.rules = df.to_dict(orient='records')

    def add_rule(self, rule):
        """
        Adds a single rule to the engine.
        Args:
            rule (dict): Rule definition (must contain 'column', 'op', 'value', optional 'label')
        """
        self.rules.append(rule)

    def apply(self, df):
        """
        Applies all loaded rules to each row of the DataFrame.
        Args:
            df (pd.DataFrame): Input data
        Returns:
            list: List of lists, each containing labels of triggered rules for each row
        """
        results = []
        for _, row in df.iterrows():
            row_result = []
            for rule in self.rules:
                col, op, val = rule['column'], rule['op'], rule['value']
                label = rule.get('label', 'RuleAnomaly')
                try:
                    # Use match-case for rule operation
                    match op:
                        case '>':
                            if row[col] > val:
                                row_result.append(label)
                        case '<':
                            if row[col] < val:
                                row_result.append(label)
                        case '==':
                            if row[col] == val:
                                row_result.append(label)
                        case '!=':
                            if row[col] != val:
                                row_result.append(label)
                        case 'in':
                            if row[col] in val:
                                row_result.append(label)
                        case 'not in':
                            if row[col] not in val:
                                row_result.append(label)
                        case 'contains':
                            if isinstance(row[col], str) and str(val) in row[col]:
                                row_result.append(label)
                        case 'startswith':
                            if isinstance(row[col], str) and row[col].startswith(str(val)):
                                row_result.append(label)
                        case 'endswith':
                            if isinstance(row[col], str) and row[col].endswith(str(val)):
                                row_result.append(label)
                        case _:
                            pass  # Unknown operation, skip
                except Exception:
                    # Ignore errors for missing columns or invalid operations
                    continue
            results.append(row_result)
        return results
