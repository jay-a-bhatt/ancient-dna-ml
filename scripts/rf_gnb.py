import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

data_path = '../data/generated/training-features/'

def calculate_relative_age(age):
    if age == 0:
        return 0
    return 2026 - age

print("Loading data...")

train_df = pd.read_csv(data_path + 'train_features_with_age.csv')
test_df  = pd.read_csv(data_path + 'test_features_with_age.csv')
val_df   = pd.read_csv(data_path + 'val_features_with_age.csv')

DEFAULT_THRESHOLDS = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
feature_cols = ['NRC_AVERAGE_AGE', 'CG_CONTENT', 'N_CONTENT', 'RELATIVE_SIZE']

train_df['AGE'] = train_df['AGE'].apply(calculate_relative_age)
test_df['AGE']  = test_df['AGE'].apply(calculate_relative_age)
val_df['AGE']   = val_df['AGE'].apply(calculate_relative_age)

X_train = train_df[feature_cols]
X_test = test_df[feature_cols]
X_val = val_df[feature_cols]

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
gnb_model = GaussianNB()

results = []

for threshold in tqdm(DEFAULT_THRESHOLDS, desc="Testing Threshold"):
    cutoff_years = threshold * 100

    # calc labels for according to each threshold.
    # fit the models to train data
    y_train = (train_df['AGE'] > cutoff_years).astype(int)
    y_test  = (test_df['AGE'] > cutoff_years).astype(int)
    y_val   = (val_df['AGE'] > cutoff_years).astype(int)

    # Traing time
    rf_model.fit(X_train, y_train)
    rf_val_preds = rf_model.predict(X_val)
    rf_f1 = f1_score(y_val, rf_val_preds, zero_division=0)

    # test
    rf_test_preds = rf_model.predict(X_test)
    rf_test_f1 = f1_score(y_test, rf_test_preds, zero_division=0)

    gnb_model.fit(X_train, y_train)
    gnb_val_preds = gnb_model.predict(X_val)
    gnb_f1 = f1_score(y_val, gnb_val_preds, zero_division=0)

    # GNB test
    gnb_test_preds = gnb_model.predict(X_test)
    gnb_test_f1 = f1_score(y_test, gnb_test_preds, zero_division=0)

    results.append({
            'Threshold_Centuries': threshold,
            'RF_Val_F1': rf_f1,
            'RF_Test_F1': rf_test_f1,
            'GNB_Val_F1': gnb_f1,
            'GNB_Test_F1': gnb_test_f1
    })

    print(f"\nThreshold: {cutoff_years} years | RF F1: {rf_f1:.4f} | GNB F1: {gnb_f1:.4f}")

# Optional: Convert to a DataFrame so it looks like a nice table if you print it!
results_df = pd.DataFrame(results)
print("\n--- Summary Table ---")
print(results_df.to_string(index=False))

csv_output = results_df[['Threshold_Centuries', 'RF_Test_F1', 'GNB_Test_F1']]
output_path = '../data/generated/test_f1_scores.csv'
csv_output.to_csv(output_path, index=False)

print(f"\nSuccessfully saved test scores to: {output_path}")
