# Step 1: Import required libraries
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML Models
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

print("Libraries imported successfully ✅")

# Step 2: Load CSV files
accounts = pd.read_csv(r"C:\Users\pvjay\Desktop\projects\done_by_me\Customer_Churn_Prediction_for_Subscription_Services\ravenstack_accounts.csv")
subscriptions = pd.read_csv(r"C:\Users\pvjay\Desktop\projects\done_by_me\Customer_Churn_Prediction_for_Subscription_Services\ravenstack_subscriptions.csv")
churn_events = pd.read_csv(r"C:\Users\pvjay\Desktop\projects\done_by_me\Customer_Churn_Prediction_for_Subscription_Services\ravenstack_churn_events.csv")
feature_usage = pd.read_csv(r"C:\Users\pvjay\Desktop\projects\done_by_me\Customer_Churn_Prediction_for_Subscription_Services\ravenstack_feature_usage.csv")
support_tickets = pd.read_csv(r"C:\Users\pvjay\Desktop\projects\done_by_me\Customer_Churn_Prediction_for_Subscription_Services\ravenstack_support_tickets.csv")

# Quick check
print("Accounts shape:", accounts.shape)
print("Subscriptions shape:", subscriptions.shape)
print("Churn events shape:", churn_events.shape)
print("Feature usage shape:", feature_usage.shape)
print("Support tickets shape:", support_tickets.shape)

# Step 3: Merge datasets to create master dataframe
df = accounts.copy()

# Merge subscriptions
df = pd.merge(df, subscriptions[['account_id','subscription_id','start_date','end_date',
                                 'plan_tier','seats','mrr_amount','arr_amount',
                                 'is_trial','upgrade_flag','downgrade_flag',
                                 'billing_frequency','auto_renew_flag','churn_flag']],
              on="account_id", how="left")

# Merge churn_events
churn_events['churned'] = 1
df = pd.merge(df, churn_events[['account_id','churned']], on="account_id", how="left")
df['churned'] = df['churned'].fillna(0)

# Aggregate feature usage
usage_agg = feature_usage.groupby('subscription_id').agg({
    'usage_count':'sum',
    'usage_duration_secs':'sum',
    'error_count':'sum'
}).reset_index()
df = pd.merge(df, usage_agg, on='subscription_id', how='left')
df[['usage_count','usage_duration_secs','error_count']] = df[['usage_count','usage_duration_secs','error_count']].fillna(0)

# Aggregate support tickets
tickets_agg = support_tickets.groupby('account_id').agg({'ticket_id':'count'}).reset_index()
tickets_agg.rename(columns={'ticket_id':'support_tickets_count'}, inplace=True)
df = pd.merge(df, tickets_agg, on='account_id', how='left')
df['support_tickets_count'] = df['support_tickets_count'].fillna(0)

print("Master dataset shape:", df.shape)
print(df.head())

# Step 4: Data Preprocessing
df.rename(columns={
    'plan_tier_y':'plan_tier',
    'seats_y':'seats',
    'is_trial_y':'is_trial'
}, inplace=True)

df.drop(['plan_tier_x','seats_x','is_trial_x','churn_flag_x','churn_flag_y'], axis=1, inplace=True, errors='ignore')

# Handle missing values
num_cols = ['seats','mrr_amount','arr_amount','usage_count','usage_duration_secs','error_count','support_tickets_count']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ['industry','country','plan_tier','billing_frequency']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature engineering
df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])
df['subscription_duration'] = (df['end_date'] - df['start_date']).dt.days
df['subscription_duration'] = df['subscription_duration'].fillna(0)

# Encode categorical variables
le = LabelEncoder()
for col in cat_cols + ['is_trial','upgrade_flag','downgrade_flag','auto_renew_flag']:
    df[col] = le.fit_transform(df[col].astype(str))

# Select features & target
features = ['seats','mrr_amount','arr_amount','usage_count','usage_duration_secs','error_count',
            'support_tickets_count','subscription_duration','industry','country','plan_tier',
            'billing_frequency','is_trial','upgrade_flag','downgrade_flag','auto_renew_flag']
X = df[features]
y = df['churned']

# Scale only numeric columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Save the scaler
import pickle
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
print("Scaler saved as scaler.pkl ✅")

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", pd.Series(y_train_res).value_counts())

# Step 7: Train XGBoost with hyperparameter tuning
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1],
    'colsample_bytree': [0.7, 0.8, 1],
    'gamma': [0, 0.1, 0.2]
}

xgb_model = XGBClassifier(random_state=42, eval_metric='logloss')

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='roc_auc',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_res, y_train_res)

print("Best Parameters:", random_search.best_params_)
print("Best ROC-AUC:", random_search.best_score_)

best_xgb = random_search.best_estimator_
y_pred_tuned = best_xgb.predict(X_test)

print("Tuned XGBoost Accuracy:", accuracy_score(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))

# Step 8: Save final model and preprocessing artifacts
with open("xgb_churn_model.pkl", "wb") as file:
    pickle.dump(best_xgb, file)
print("Model saved as xgb_churn_model.pkl ✅")

with open("columns.pkl", "wb") as file:
    pickle.dump(X.columns.tolist(), file)
print("Columns saved as columns.pkl ✅")

# Step 9: Evaluation - Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuned XGBoost")
plt.show()

# ROC Curve
y_prob = best_xgb.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Tuned XGBoost')
plt.legend(loc="lower right")
plt.show()

# Feature Importance
fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': best_xgb.feature_importances_})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette="viridis")
plt.title("Feature Importance - Tuned XGBoost")
plt.show()
