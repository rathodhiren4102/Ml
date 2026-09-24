import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

print("--- Step 1: Loading Dataset ---")
df = pd.read_csv('insurance_fraud_data.csv')
print(f"Initial shape: {df.shape}")

print("--- Step 2: Cleaning Column Names ---")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print("--- Step 3: Fixing Incorrect Data Types (coercing '*' to NaN) ---")
df['marital_status'] = pd.to_numeric(df['marital_status'], errors='coerce')
df['witness_present'] = pd.to_numeric(df['witness_present'], errors='coerce')
df['age_of_vehicle'] = pd.to_numeric(df['age_of_vehicle'], errors='coerce')
df['injury_claim'] = pd.to_numeric(df['injury_claim'], errors='coerce')
df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')

print("--- Step 4: Imputing Missing Values ---")
df['marital_status'] = df['marital_status'].fillna(df['marital_status'].median())
df['witness_present'] = df['witness_present'].fillna(df['witness_present'].median())
df['age_of_vehicle'] = df['age_of_vehicle'].fillna(df['age_of_vehicle'].median())
df['injury_claim'] = df['injury_claim'].fillna(df['injury_claim'].median())
df['claim_date'] = df['claim_date'].fillna(df['claim_date'].mode()[0])
df['fraud_reported'] = df['fraud_reported'].fillna(df['fraud_reported'].mode()[0])

print("--- Step 5: Dropping Duplicates ---")
df = df.drop_duplicates()
print(f"Shape after duplicate removal: {df.shape}")

print("--- Step 6: Removing Outliers (IQR on continuous numerical columns) ---")
continuous_cols = ['age_of_driver', 'safety_rating', 'annual_income', 'vehicle_price', 'total_claim', 'injury_claim', 'annual_premium', 'days_open', 'form_defects']
for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
print(f"Shape after outlier removal: {df.shape}")

print("--- Step 7: Extracting Date Features ---")
df['claim_year'] = df['claim_date'].dt.year
df['claim_month'] = df['claim_date'].dt.month
df['claim_day'] = df['claim_date'].dt.day
df = df.drop(columns=['claim_date'])

print("--- Step 8: Splitting Features and Target ---")
# Drop unique identifier 'claim_number' and target 'fraud_reported'
X = df.drop(columns=['claim_number', 'fraud_reported'])
y = df['fraud_reported'].map({'N': 0, 'Y': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

print("--- Step 9: Creating Preprocessing Pipeline ---")
# Numerical columns to scale
numerical_cols = ['age_of_driver', 'safety_rating', 'annual_income', 'vehicle_price', 'total_claim', 'injury_claim', 'annual_premium', 'days_open', 'form_defects', 'claim_year', 'claim_month', 'claim_day']

# Categorical columns to encode
categorical_cols = ['gender', 'property_status', 'claim_day_of_week', 'accident_site', 'channel', 'vehicle_category', 'vehicle_color']

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols)
    ],
    remainder='passthrough'
)

print("--- Step 10: Model Training ---")
clf = RandomForestClassifier(max_depth=3, class_weight='balanced', random_state=42)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', clf)
])

pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nRandom Forest (max_depth=3) Evaluation:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  Confusion Matrix:\n{cm}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n--- Step 11: Exporting Complete Pipeline to Pickle ---")
model_path = 'insurance_fraud_model.pkl'
joblib.dump(pipeline, model_path)
print(f"Complete pipeline successfully saved to {model_path}!")
