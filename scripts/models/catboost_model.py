import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score


DEFAULT_THRESHOLDS = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]

train_data = "data/train.csv"
val_data = "data/validation.csv"
test_data = "data/test.csv"
f1_scores = {}

def make_model(threshold):
    # Load datasets
    train_df = pd.read_csv(train_data).drop(columns=['ID', 'SEQUENCE'])
    val_df = pd.read_csv(val_data).drop(columns=['ID', 'SEQUENCE'])
    test_df = pd.read_csv(test_data).drop(columns=['ID', 'SEQUENCE'])

    # Offset age so it is the number of years before 2026 (ignore age=0 which is for the modern dataset)
    train_df.loc[train_df['AGE'] != 0, 'AGE'] = 2026 - train_df['AGE']
    val_df.loc[val_df['AGE'] != 0, 'AGE'] = 2026 - val_df['AGE']
    test_df.loc[test_df['AGE'] != 0, 'AGE'] = 2026 - test_df['AGE']

    train_df['AGE'] = (train_df['AGE'] > threshold*100).astype(int)
    val_df['AGE'] = (val_df['AGE'] > threshold*100).astype(int)
    test_df['AGE'] = (test_df['AGE'] > threshold*100).astype(int)

    target_col = 'AGE'

    # Split into X and y
    X_train, y_train = train_df.drop(target_col, axis=1), train_df[target_col]
    X_val, y_val = val_df.drop(target_col, axis=1), val_df[target_col]
    X_test, y_test = test_df.drop(target_col, axis=1), test_df[target_col]

    # Prepare Pools
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # Initialize and Train the Model
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        eval_metric='F1',
        verbose=100
    )

    model.fit(
        train_pool, 
        eval_set=val_pool, 
        early_stopping_rounds=50
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_true = y_test

    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_scores[threshold] = f1

    print(f"Threshold: {threshold*100} years")
    print(f"Test F1 Score: {f1:.4f}")

    return model

for threshold in DEFAULT_THRESHOLDS:
    make_model(threshold)

# Output f1 scores to CSV
# results_df = pd.DataFrame(list(f1_scores.items()), columns=['Threshold_Centuries', 'F1_Score'])
# results_df.to_csv('cb_threshold_performance.csv', index=False)

print(f1_scores)