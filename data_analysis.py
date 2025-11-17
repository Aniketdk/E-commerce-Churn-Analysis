import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the data (Assuming you have run the script and saved 'e-commerce_data.csv')
file_name = 'e-commerce_data.csv'
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: File '{file_name}' not found. Please run the data generation script first.")
    exit()

# Drop ID column
df.drop('customerID', axis=1, inplace=True)

# 2. Encode Categorical Features
# Target Variable (Is_Churned): Yes=1, No=0
df['Is_Churned'] = df['Is_Churned'].replace({'Yes': 1, 'No': 0})

# Subscription: Yes=1, No=0
df['Subscription'] = df['Subscription'].replace({'Yes': 1, 'No': 0})

# 3. Separate Features (X) and Target (y)
X = df.drop('Is_Churned', axis=1)
y = df['Is_Churned']

# 4. Scale Numerical Features (for consistent model training, though less critical for Random Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Train the Model
# Use class_weight='balanced' to handle the typical imbalance (fewer churners than non-churners)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', max_depth=10)
model.fit(X_train, y_train)

# 7. Predict Probabilities and Classes on the Test Set
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# 8. Evaluate Model Performance
print("## Model Evaluation Results")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 9. Get Feature Importance (Business Insight!)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_drivers = feature_importances.sort_values(ascending=False).head(5)

print("\n## Top 5 Churn Drivers (Feature Importance)")
print(top_drivers)

# 10. Create Churn Risk Score for the Entire Dataset (Required for Power BI)
# Score the original un-split, un-scaled data after fit (using the scaled features)
# To get the raw data rows aligned with the prediction, we can use the full scaled data:
df_full_scaled = pd.DataFrame(scaler.transform(df.drop('Is_Churned', axis=1)), columns=X.columns)

# Predict probabilities (Churn Risk Score)
df['Churn_Probability'] = model.predict_proba(df_full_scaled)[:, 1].round(4)
df['Churn_Risk_Segment'] = pd.cut(
    df['Churn_Probability'], 
    bins=[0, 0.3, 0.7, 1.0], 
    labels=['Low Risk', 'Medium Risk', 'High Risk'], 
    right=True, 
    include_lowest=True
)

# Save the enriched data for Power BI
df.to_csv('ecommerce_churn_scored_for_BI.csv', index=False)
print("\nScored data saved to 'ecommerce_churn_scored_for_BI.csv' for Power BI.")