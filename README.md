# Task 2: End-to-End ML Pipeline with Scikit-learn

## Objective
Build a reusable and production-ready machine learning pipeline for predicting customer churn using the Telco Churn Dataset.

## Methodology / Approach
1. **Dataset Loading**: Loaded Telco Churn dataset with 7,043 customers and 20 features
2. **Preprocessing**: Built comprehensive pipeline using ColumnTransformer with:
   - Numerical features: Median imputation + StandardScaler
   - Categorical features: Most frequent imputation + OneHotEncoder
3. **Models**: Trained Logistic Regression and Random Forest classifiers
4. **Hyperparameter Tuning**: Used GridSearchCV with 5-fold cross-validation
5. **Evaluation**: Evaluated using accuracy, precision, recall, F1-score, and ROC AUC
6. **Export**: Saved complete pipeline using joblib for production deployment

## Key Results / Observations
- Successfully built end-to-end ML pipeline with proper preprocessing
- Random Forest typically outperformed Logistic Regression (F1-score ~0.60-0.65)
- Top predictive features: tenure, MonthlyCharges, contract type, and payment method
- Pipeline encapsulates all preprocessing steps for consistent inference
- Churn rate of ~26.5% required careful evaluation (F1-score over accuracy)
- Exported pipeline is production-ready and can be loaded for predictions

## Installation
```bash
pip install -r requirements.txt
```

## Usage
### Training
Run the Jupyter notebook to train and evaluate the pipeline:
```bash
jupyter notebook ml_pipeline_churn.ipynb
```

### Prediction
Load the exported pipeline and make predictions:
```bash
python predict.py
```

## Files
- `ml_pipeline_churn.ipynb` - Complete pipeline building and evaluation notebook
- `predict.py` - Script to load pipeline and make predictions
- `churn_prediction_pipeline.joblib` - Exported pipeline (generated after training)
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Skills Gained
- ML pipeline construction with scikit-learn Pipeline API
- Data preprocessing (scaling, encoding, imputation)
- Hyperparameter tuning with GridSearchCV
- Model evaluation and comparison
- Feature importance analysis
- Model export and reusability with joblib
- Production-readiness practices
