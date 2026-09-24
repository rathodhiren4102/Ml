import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

def load_and_clean_data(filepath='insurance_fraud_data.csv'):
    """Loads and preprocesses raw insurance fraud dataset according to SOP."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file '{filepath}' not found.")
    
    df_raw = pd.read_csv(filepath)
    df = df_raw.copy()
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Coerce '*' placeholders to NaN
    numeric_fixes = ['marital_status', 'witness_present', 'age_of_vehicle', 'injury_claim']
    for col in numeric_fixes:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'claim_date' in df.columns:
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
    
    # Impute missing values
    for col in numeric_fixes:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    if 'claim_date' in df.columns and not df['claim_date'].dropna().empty:
        df['claim_date'] = df['claim_date'].fillna(df['claim_date'].mode()[0])
        
    if 'fraud_reported' in df.columns:
        df['fraud_reported'] = df['fraud_reported'].fillna(df['fraud_reported'].mode()[0])
        
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Outlier removal via IQR on continuous numerical columns
    continuous_cols = [
        'age_of_driver', 'safety_rating', 'annual_income', 'vehicle_price',
        'total_claim', 'injury_claim', 'annual_premium', 'days_open', 'form_defects'
    ]
    for col in continuous_cols:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
    # Extract date features
    if 'claim_date' in df.columns:
        df['claim_year'] = df['claim_date'].dt.year.astype(float)
        df['claim_month'] = df['claim_date'].dt.month.astype(float)
        df['claim_day'] = df['claim_date'].dt.day.astype(float)
        df = df.drop(columns=['claim_date'])
        
    return df_raw, df

def build_preprocessor(X):
    """Builds the scikit-learn ColumnTransformer for scaling and encoding."""
    numerical_cols = [
        'age_of_driver', 'safety_rating', 'annual_income', 'vehicle_price',
        'total_claim', 'injury_claim', 'annual_premium', 'days_open',
        'form_defects', 'claim_year', 'claim_month', 'claim_day',
        'marital_status', 'witness_present', 'age_of_vehicle',
        'high_education', 'address_change', 'zip_code', 'past_num_of_claims',
        'policy_deductible', 'liab_prct', 'police_report'
    ]
    categorical_cols = [
        'gender', 'property_status', 'claim_day_of_week', 'accident_site',
        'channel', 'vehicle_category', 'vehicle_color'
    ]
    
    num_present = [c for c in numerical_cols if c in X.columns]
    cat_present = [c for c in categorical_cols if c in X.columns]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_present),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_present)
        ],
        remainder='passthrough'
    )
    return preprocessor

def train_all_models():
    """Trains and benchmarks all 5 algorithms, returning pipelines and evaluation stats."""
    _, df = load_and_clean_data()
    
    # Feature matrix and target
    X = df.drop(columns=['claim_number', 'fraud_reported'], errors='ignore')
    y = df['fraud_reported'].map({'N': 0, 'Y': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = build_preprocessor(X)
    
    # Define models
    model_definitions = {
        'Tuned Random Forest (GridSearchCV)': RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_split=5, class_weight='balanced', random_state=42
        ),
        'Random Forest (Bagging)': RandomForestClassifier(
            n_estimators=50, max_depth=4, class_weight='balanced', random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=4, class_weight='balanced', random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50, random_state=42
        )
    }
    
    trained_pipelines = {}
    metrics_records = []
    confusion_matrices = {}
    roc_data = {}
    classification_reports_dict = {}
    
    for name, clf in model_definitions.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        
        train_pred = pipe.predict(X_train)
        test_pred = pipe.predict(X_test)
        
        # Probabilities for ROC
        if hasattr(pipe, "predict_proba"):
            y_probs = pipe.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_probs)
            roc_auc_val = auc(fpr, tpr)
            roc_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc_val}
        else:
            y_probs = None
            roc_data[name] = {'fpr': [0, 1], 'tpr': [0, 1], 'auc': 0.5}
            
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        prec = precision_score(y_test, test_pred, zero_division=0)
        rec = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)
        cm = confusion_matrix(y_test, test_pred)
        cr = classification_report(y_test, test_pred, output_dict=True)
        
        # Overfitting diagnosis
        diff = train_acc - test_acc
        if diff > 0.12:
            fit_status = 'Overfitting'
        elif train_acc < 0.55 and test_acc < 0.55:
            fit_status = 'Underfitting'
        else:
            fit_status = 'Good Fit'
            
        # 5-Fold Cross Validation
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        confusion_matrices[name] = cm
        classification_reports_dict[name] = cr
        
        metrics_records.append({
            'Model': name,
            'Accuracy': round(test_acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(roc_data[name]['auc'], 4),
            'Train Score': round(train_acc, 4),
            'Test Score': round(test_acc, 4),
            'CV Mean (5-Fold)': round(cv_mean, 4),
            'CV Std Spread': round(cv_std, 4),
            'Fit Status': fit_status
        })
        
    comparison_df = pd.DataFrame(metrics_records)
    
    return {
        'pipelines': trained_pipelines,
        'comparison_df': comparison_df,
        'confusion_matrices': confusion_matrices,
        'roc_data': roc_data,
        'classification_reports': classification_reports_dict,
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'feature_cols': X.columns.tolist()
    }

def format_single_input(inputs):
    """Converts user input dictionary to DataFrame with all required columns."""
    claim_date = inputs.get('claim_date', datetime.today())
    if isinstance(claim_date, str):
        claim_date = pd.to_datetime(claim_date)
        
    gender_mapped = "M" if inputs.get('gender') == "Male" else "F"
    marital_mapped = 1.0 if inputs.get('marital_status') == "Married" else 0.0
    high_edu_mapped = 1 if inputs.get('high_education') == "Yes" else 0
    addr_map = 1 if inputs.get('address_change') == "Yes" else 0
    witness_mapped = 1.0 if inputs.get('witness_present') == "Yes" else 0.0
    police_mapped = 1 if inputs.get('police_report') == "Yes" else 0

    input_data = {
        'age_of_driver':      [float(inputs.get('age_of_driver', 40))],
        'gender':             [gender_mapped],
        'marital_status':     [marital_mapped],
        'safety_rating':      [float(inputs.get('safety_rating', 75))],
        'annual_income':      [float(inputs.get('annual_income', 35000))],
        'high_education':     [high_edu_mapped],
        'address_change':     [addr_map],
        'property_status':    [inputs.get('property_status', 'Own')],
        'zip_code':           [int(inputs.get('zip_code', 50027))],
        'claim_day_of_week':  [claim_date.strftime('%A')],
        'accident_site':      [inputs.get('accident_site', 'Local')],
        'past_num_of_claims': [int(inputs.get('past_num_of_claims', 1))],
        'witness_present':    [witness_mapped],
        'liab_prct':          [int(inputs.get('liab_prct', 50))],
        'channel':            [inputs.get('channel', 'Broker')],
        'police_report':      [police_mapped],
        'age_of_vehicle':     [float(inputs.get('age_of_vehicle', 5))],
        'vehicle_category':   [inputs.get('vehicle_category', 'Medium')],
        'vehicle_price':      [float(inputs.get('vehicle_price', 22000))],
        'vehicle_color':      [inputs.get('vehicle_color', 'white')],
        'total_claim':        [float(inputs.get('total_claim', 5000))],
        'injury_claim':       [float(inputs.get('injury_claim', 1000))],
        'policy_deductible':  [int(inputs.get('policy_deductible', 500))],
        'annual_premium':     [float(inputs.get('annual_premium', 1000))],
        'days_open':          [float(inputs.get('days_open', 8))],
        'form_defects':       [float(inputs.get('form_defects', 4))],
        'claim_year':         [float(claim_date.year)],
        'claim_month':        [float(claim_date.month)],
        'claim_day':          [float(claim_date.day)],
    }
    return pd.DataFrame(input_data)
