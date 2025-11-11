
# Finomaly
## PyPI & Source Code

**PyPI:** https://pypi.org/project/finomaly/

**Source Code:** https://github.com/Barisaksel/finomaly



Finomaly is a modular, open-source Python library for anomaly detection in financial transactions. It supports both rule-based and machine learning-based detection, multi-language reporting, and professional reporting formats.

## 🆕 NLP Modules & Language-Aware Message Management


### TextEmbeddingAnomalyDetector
Provides text anomaly detection using TF-IDF vectorization and IsolationForest. All error and user messages are centrally managed and support multiple languages.

**Usage:**
```python
from finomaly.nlp.embeddings import TextEmbeddingAnomalyDetector

# You can provide a list of texts, or an Excel file path and column name.
texts = ["Money transfer completed", "Credit card purchase", "Suspicious transaction"]
detector = TextEmbeddingAnomalyDetector(lang='en')
detector.fit(texts)
anomalies = detector.predict(["Very large amount transfer"])
print(anomalies)

# Or load texts directly from an Excel file:
# detector.fit('transactions.xlsx', column='Description')
# anomalies = detector.predict('new_transactions.xlsx', column='Description')
```


### TransactionDescriptionClassifier
Categorizes transaction descriptions using TF-IDF and Logistic Regression. All messages are centrally managed and support multiple languages.

**Usage:**
```python
from finomaly.nlp.text_classifier import TransactionDescriptionClassifier

# You can provide lists, or load from Excel file with text and label columns.
texts = ["Grocery shopping", "Bill payment", "ATM withdrawal"]
labels = ["Shopping", "Bill", "Cash"]
clf = TransactionDescriptionClassifier(lang='en')
clf.fit(texts, labels)
pred = clf.predict(["Electricity bill"])
print(pred)

# Or load texts and labels from Excel:
# import pandas as pd
# df = pd.read_excel('train.xlsx')
# clf.fit(df['Description'], df['Category'])
# pred = clf.predict(['Electricity bill'])
```

### Multi-Language Support & Message Management
- All error and user messages are centrally managed in `finomaly/core/messages_config.json` according to the selected language.
- If a message key or language is missing, the system automatically returns the default English message without raising an error.

### Requirements
- scikit-learn, pandas, numpy
- (For NLP) openpyxl (Excel support), language file: messages_config.json

## Features
- Rule-based anomaly detection (JSON-configurable, customer-specific rules)
- Machine learning models: IsolationForest, RandomForest, XGBoost
- NLP modules: TextEmbeddingAnomalyDetector (text anomaly), TransactionDescriptionClassifier (text classification)
- Profile-based analysis (behavioral deviation, unusual time, etc.)
- Multi-language support (TR/EN) for all messages and reports
- Centralized message and rule management (with fallback to default language)
- Professional reporting: Excel, HTML, PDF (with optional charts)
- Visual analytics: anomaly distribution, scatter plots
- Easy integration, clean API, and extensible modular structure

## Installation
```bash
pip install finomaly
```

## Quick Start
```python
import pandas as pd
from finomaly.core.anomaly_system import CorporateAnomalySystem

# Load your data
train_df = pd.read_excel('train.xlsx')
predict_df = pd.read_excel('predict.xlsx')

# Define features and rules
features = ['Tutar', 'Saat']
rules_path = 'rules.json'

# Initialize system
system = CorporateAnomalySystem(features, rules_path=rules_path, ml_method='isolation_forest', lang='en')

# Train model
system.fit('train.xlsx', customer_col='MusteriID', amount_col='Tutar')

# Predict anomalies
output_path = system.predict('predict.xlsx', customer_col='MusteriID', amount_col='Tutar')
result = pd.read_excel(output_path)
print(result.head())
```

## Reporting & Visualization
```python
from finomaly.report.visualizer import Visualizer
from finomaly.report.pdf_reporter import PDFReporter

visualizer = Visualizer()
visualizer.plot_anomaly_distribution(result, amount_col='Tutar', anomaly_col='ML_Anomaly')

pdf_reporter = PDFReporter()
pdf_reporter.generate_pdf_report(result, 'report.pdf')
```

## Project Structure
- `core/` : Rule engine, model management, utilities
- `ml/` : ML models (IsolationForest, RandomForest, XGBoost)
- `profile/` : Profile-based analysis (behavioral, time-based)
- `report/` : Reporting and visualization (Excel, HTML, PDF, charts)

## Contributing
Finomaly is open-source and welcomes contributions. Please open issues or pull requests for improvements, bug fixes, or new features.

## License
MIT License

## Author
Barış
