# ==============================================================================
# Week 5 / Task 5: Model Evaluation, Validation, Comparison & Hyperparameter Tuning
# Vehicle Insurance Fraud Detection Project (Semester 5 ML)
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Models to evaluate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# Evaluation Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("="*70)
print("  TASK 5: MODEL EVALUATION, COMPARISON & HYPERPARAMETER TUNING")
print("="*70)

# ------------------------------------------------------------------------------
# STEP 1: Load and Clean Dataset
# ------------------------------------------------------------------------------
print("\n[Step 1] Loading and cleaning dataset...")
df = pd.read_csv('insurance_fraud_data.csv')
print(f"  -> Raw dataset shape: {df.shape}")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Handle missing values
df['marital_status'] = pd.to_numeric(df['marital_status'], errors='coerce')
df['witness_present'] = pd.to_numeric(df['witness_present'], errors='coerce')
df['age_of_vehicle'] = pd.to_numeric(df['age_of_vehicle'], errors='coerce')
df['injury_claim'] = pd.to_numeric(df['injury_claim'], errors='coerce')
df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')

df['marital_status'] = df['marital_status'].fillna(df['marital_status'].median())
df['witness_present'] = df['witness_present'].fillna(df['witness_present'].median())
df['age_of_vehicle'] = df['age_of_vehicle'].fillna(df['age_of_vehicle'].median())
df['injury_claim'] = df['injury_claim'].fillna(df['injury_claim'].median())
df['claim_date'] = df['claim_date'].fillna(df['claim_date'].mode()[0])
df['fraud_reported'] = df['fraud_reported'].fillna(df['fraud_reported'].mode()[0])

# Drop duplicates
df = df.drop_duplicates()

# IQR Outlier Removal
continuous_cols = ['age_of_driver', 'safety_rating', 'annual_income', 'vehicle_price', 
                   'total_claim', 'injury_claim', 'annual_premium', 'days_open', 'form_defects']
for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Extract date components
df['claim_year'] = df['claim_date'].dt.year
df['claim_month'] = df['claim_date'].dt.month
df['claim_day'] = df['claim_date'].dt.day
df = df.drop(columns=['claim_date'])

print(f"  -> Cleaned dataset shape: {df.shape}")

# ------------------------------------------------------------------------------
# STEP 2: Train-Test Split (80/20) & Preprocessing Pipeline
# ------------------------------------------------------------------------------
print("\n[Step 2] Splitting features and setting up Preprocessor...")
X = df.drop(columns=['claim_number', 'fraud_reported'])
y = df['fraud_reported'].map({'N': 0, 'Y': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  -> Training set: {X_train.shape}, Testing set: {X_test.shape}")

numerical_cols = ['age_of_driver', 'safety_rating', 'annual_income', 'vehicle_price', 
                  'total_claim', 'injury_claim', 'annual_premium', 'days_open', 
                  'form_defects', 'claim_year', 'claim_month', 'claim_day', 
                  'marital_status', 'witness_present', 'age_of_vehicle', 
                  'high_education', 'address_change', 'zip_code', 'past_num_of_claims', 'police_report_available']
categorical_cols = ['gender', 'property_status', 'claim_day_of_week', 'accident_site', 
                    'channel', 'vehicle_category', 'vehicle_color']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), [c for c in numerical_cols if c in X.columns]),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), [c for c in categorical_cols if c in X.columns])
    ],
    remainder='passthrough'
)

# ------------------------------------------------------------------------------
# STEP 3: Train Models, Evaluate, Check Overfitting & 5-Fold CV
# ------------------------------------------------------------------------------
print("\n[Step 3] Training and Evaluating all models (Checklist items 1, 2, 3, 6)...")

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42),
    'Random Forest (Bagging)': RandomForestClassifier(n_estimators=50, max_depth=4, class_weight='balanced', random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42)
}

results_list = []

for name, clf in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    # Fit model on training data
    pipe.fit(X_train, y_train)
    
    # Predictions
    train_pred = pipe.predict(X_train)
    test_pred = pipe.predict(X_test)
    
    # 1. Overfitting / Underfitting Check
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    diff = train_acc - test_acc
    if diff > 0.12:
        fit_status = 'Overfitting'
    elif train_acc < 0.55 and test_acc < 0.55:
        fit_status = 'Underfitting'
    else:
        fit_status = 'Good Fit'
        
    # 2. Evaluation Metrics (Classification)
    acc = accuracy_score(y_test, test_pred)
    prec = precision_score(y_test, test_pred, zero_division=0)
    rec = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    
    # 3. 5-Fold Cross Validation
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results_list.append({
        'Model': name,
        'Train Score': round(train_acc, 4),
        'Test Score': round(test_acc, 4),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-score': round(f1, 4),
        'CV Mean': round(cv_mean, 4),
        'CV Spread (Std)': round(cv_std, 4),
        'Fit Status': fit_status
    })

# ------------------------------------------------------------------------------
# STEP 4: All Models Comparison Table (Checklist item 4)
# ------------------------------------------------------------------------------
print("\n" + "="*40 + " ALL MODELS COMPARISON TABLE " + "="*40)
comparison_df = pd.DataFrame(results_list)
print(comparison_df.to_string(index=False))
print("="*109)

# ------------------------------------------------------------------------------
# STEP 5: Hyperparameter Tuning using GridSearchCV (Checklist item 5)
# ------------------------------------------------------------------------------
print("\n[Step 5] Performing Hyperparameter Tuning on Best Model (Random Forest)...")
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, 8],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, scoring='f1', n_jobs=1)
grid_search.fit(X_train, y_train)

print(f"  -> Best Hyperparameters: {grid_search.best_params_}")
print(f"  -> Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

# Re-test tuned model on test set
best_tuned_model = grid_search.best_estimator_
tuned_pred = best_tuned_model.predict(X_test)

tuned_acc = accuracy_score(y_test, tuned_pred)
tuned_prec = precision_score(y_test, tuned_pred, zero_division=0)
tuned_rec = recall_score(y_test, tuned_pred, zero_division=0)
tuned_f1 = f1_score(y_test, tuned_pred, zero_division=0)
tuned_cm = confusion_matrix(y_test, tuned_pred)

print("\n--- Final Re-Test of Tuned Random Forest Model ---")
print(f"  Accuracy:  {tuned_acc:.4f}")
print(f"  Precision: {tuned_prec:.4f}")
print(f"  Recall:    {tuned_rec:.4f}")
print(f"  F1-Score:  {tuned_f1:.4f}")
print("\nConfusion Matrix:")
print(tuned_cm)
print("\nClassification Report:")
print(classification_report(y_test, tuned_pred))
print("\nTask 5 completed successfully!")
