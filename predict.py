"""
Churn Prediction Script
Load the exported pipeline and make predictions on new data
"""

import pandas as pd
import joblib

def load_pipeline():
    """Load the exported churn prediction pipeline"""
    try:
        pipeline = joblib.load('churn_prediction_pipeline.joblib')
        print("Pipeline loaded successfully!")
        return pipeline
    except FileNotFoundError:
        print("Error: Pipeline file not found. Please train the model first.")
        return None

def predict_churn(pipeline, customer_data):
    """
    Predict churn for customer data
    
    Args:
        pipeline: Loaded scikit-learn pipeline
        customer_data: Dictionary or DataFrame with customer features
    
    Returns:
        Prediction and probability
    """
    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])
    
    prediction = pipeline.predict(customer_data)[0]
    probability = pipeline.predict_proba(customer_data)[0, 1]
    
    return {
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'probability': probability
    }

if __name__ == "__main__":
    # Load pipeline
    pipeline = load_pipeline()
    
    if pipeline is not None:
        # Example prediction
        sample_customer = {
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 24,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 79.5,
            'TotalCharges': 1908.0
        }
        
        result = predict_churn(pipeline, sample_customer)
        print(f"\nPrediction: {result['prediction']}")
        print(f"Churn Probability: {result['probability']:.4f}")
