import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


TRAIN_PATH = "/Users/manmeetkaur/Downloads/train_features_with_age.csv"
VAL_PATH   = "/Users/manmeetkaur/Downloads/val_features_with_age.csv"
TEST_PATH  = "/Users/manmeetkaur/Downloads/test_features_with_age.csv"


DEFAULT_THRESHOLDS = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]

=
FEATURES = ["NRC_AVERAGE_AGE", "CG_CONTENT", "N_CONTENT", "RELATIVE_SIZE"]

REFERENCE_YEAR = 2026



# LOAD DATA

train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)
test_df  = pd.read_csv(TEST_PATH)



def compute_years_ago(df: pd.DataFrame) -> pd.Series:
    age = pd.to_numeric(df["AGE"], errors="coerce")
    years_ago = REFERENCE_YEAR - age

   
    years_ago.loc[age == 0] = 0
    return years_ago

train_years = compute_years_ago(train_df)
val_years   = compute_years_ago(val_df)
test_years  = compute_years_ago(test_df)


def make_labels(years_ago: pd.Series, threshold_centuries: int) -> np.ndarray:
    cutoff = threshold_centuries * 100
    # ancient=1 if years_ago > cutoff else modern=0
    return (years_ago > cutoff).astype(int).to_numpy()

def make_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X

X_train = make_X(train_df)
X_val   = make_X(val_df)
X_test  = make_X(test_df)

X_trainval = pd.concat([X_train, X_val], axis=0, ignore_index=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def make_svm(kernel: str, C: float, gamma) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(kernel=kernel, C=C, gamma=gamma))
    ])

def make_logreg(C: float) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=C, max_iter=5000, solver="lbfgs"))
    ])

SVM_C_GRID = [0.1, 1, 10, 100]
SVM_GAMMA_GRID = ["scale", 0.01, 0.1, 1]

LOGREG_C_GRID = [0.01, 0.1, 1, 10, 100]


def tune_on_val(model_name: str, kernel: str | None,
                Xtr, ytr, Xv, yv):
    """
    Returns: best_model, best_params, best_val_metrics
    Selection criterion: highest VAL binary F1
    """
    best = {
        "model": None,
        "params": None,
        "val_metrics": None,
        "val_f1_binary": -1.0
    }

    if model_name.startswith("SVM"):
        for C in SVM_C_GRID:
            for gamma in SVM_GAMMA_GRID:
                model = make_svm(kernel=kernel, C=C, gamma=gamma)
                model.fit(Xtr, ytr)
                pred = model.predict(Xv)
                m = compute_metrics(yv, pred)

                if m["f1_binary"] > best["val_f1_binary"]:
                    best["model"] = model
                    best["params"] = {"model": model_name, "kernel": kernel, "C": C, "gamma": gamma}
                    best["val_metrics"] = m
                    best["val_f1_binary"] = m["f1_binary"]

    elif model_name == "LOGISTIC_REGRESSION":
        for C in LOGREG_C_GRID:
            model = make_logreg(C=C)
            model.fit(Xtr, ytr)
            pred = model.predict(Xv)
            m = compute_metrics(yv, pred)

            if m["f1_binary"] > best["val_f1_binary"]:
                best["model"] = model
                best["params"] = {"model": model_name, "C": C}
                best["val_metrics"] = m
                best["val_f1_binary"] = m["f1_binary"]

    else:
        raise ValueError("Unknown model_name")

    return best["model"], best["params"], best["val_metrics"]

results_rows = []
best_params_rows = []

for threshold in DEFAULT_THRESHOLDS:
    cutoff = threshold * 100

    y_train = make_labels(train_years, threshold)
    y_val   = make_labels(val_years, threshold)
    y_test  = make_labels(test_years, threshold)

    # If training is single-class at some threshold, skip
    if len(np.unique(y_train)) < 2:
        print(f"[SKIP] threshold={threshold} (cutoff={cutoff}): training has only one class")
        continue

    print("\n" + "="*50)
    print(f"Threshold: {threshold} centuries (cutoff={cutoff} years)")
    print("="*50)

    # ---- Tune each model on VAL ----
    tuned = []

    svm_rbf_model, svm_rbf_params, svm_rbf_valm = tune_on_val(
        model_name="SVM_RBF", kernel="rbf",
        Xtr=X_train, ytr=y_train, Xv=X_val, yv=y_val
    )
    tuned.append(("SVM_RBF", svm_rbf_model, svm_rbf_params, svm_rbf_valm))

    svm_sig_model, svm_sig_params, svm_sig_valm = tune_on_val(
        model_name="SVM_SIGMOID", kernel="sigmoid",
        Xtr=X_train, ytr=y_train, Xv=X_val, yv=y_val
    )
    tuned.append(("SVM_SIGMOID", svm_sig_model, svm_sig_params, svm_sig_valm))

    log_model, log_params, log_valm = tune_on_val(
        model_name="LOGISTIC_REGRESSION", kernel=None,
        Xtr=X_train, ytr=y_train, Xv=X_val, yv=y_val
    )
    tuned.append(("LOGISTIC_REGRESSION", log_model, log_params, log_valm))

    # ---- Evaluate best params on TEST (retrain on TRAIN+VAL) ----
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    for name, _, params, valm in tuned:
        # rebuild model using best params then fit on train+val
        if name.startswith("SVM"):
            model = make_svm(kernel=params["kernel"], C=params["C"], gamma=params["gamma"])
        else:
            model = make_logreg(C=params["C"])

        model.fit(X_trainval, y_trainval)

        test_pred = model.predict(X_test)
        testm = compute_metrics(y_test, test_pred)

        print(f"\n{name} BEST PARAMS: {params}")
        print(f"VAL  F1 Binary:   {valm['f1_binary']:.4f} | VAL  F1 Weighted:   {valm['f1_weighted']:.4f}")
        print(f"TEST F1 Binary:   {testm['f1_binary']:.4f} | TEST F1 Weighted:   {testm['f1_weighted']:.4f}")
        print(f"VAL  Acc: {valm['accuracy']:.4f} | TEST Acc: {testm['accuracy']:.4f}")

        results_rows.append({
            "threshold_centuries": threshold,
            "cutoff_years": cutoff,
            "model": name,

            "val_accuracy": valm["accuracy"],
            "val_precision": valm["precision"],
            "val_recall": valm["recall"],
            "val_f1_binary": valm["f1_binary"],
            "val_f1_weighted": valm["f1_weighted"],

            "test_accuracy": testm["accuracy"],
            "test_precision": testm["precision"],
            "test_recall": testm["recall"],
            "test_f1_binary": testm["f1_binary"],
            "test_f1_weighted": testm["f1_weighted"],
        })

        best_params_rows.append({
            "threshold_centuries": threshold,
            "cutoff_years": cutoff,
            **params
        })


results_df = pd.DataFrame(results_rows)
params_df  = pd.DataFrame(best_params_rows)

results_df.to_csv("tuned_results_per_threshold.csv", index=False)
params_df.to_csv("best_params_per_threshold.csv", index=False)

print("\nSaved:")
print(" - tuned_results_per_threshold.csv")
print(" - best_params_per_threshold.csv")

def plot_metric(results: pd.DataFrame, metric_col: str, title: str, out_file: str):
    plt.figure()
    for model_name in results["model"].unique():
        sub = results[results["model"] == model_name].sort_values("threshold_centuries")
        plt.plot(sub["threshold_centuries"], sub[metric_col], marker="o", label=model_name)
    plt.xlabel("Threshold (centuries)")
    plt.ylabel(metric_col.replace("_", " ").upper())
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=200)
    plt.close()

if not results_df.empty:
    plot_metric(
        results_df,
        metric_col="val_f1_binary",
        title="Validation Binary F1 vs Threshold (Tuned Models)",
        out_file="f1_binary_vs_threshold_val.png"
    )
    plot_metric(
        results_df,
        metric_col="test_f1_binary",
        title="Test Binary F1 vs Threshold (Tuned Models)",
        out_file="f1_binary_vs_threshold_test.png"
    )
    plot_metric(
        results_df,
        metric_col="val_f1_weighted",
        title="Validation Weighted F1 vs Threshold (Tuned Models)",
        out_file="f1_weighted_vs_threshold_val.png"
    )
    plot_metric(
        results_df,
        metric_col="test_f1_weighted",
        title="Test Weighted F1 vs Threshold (Tuned Models)",
        out_file="f1_weighted_vs_threshold_test.png"
    )

    print("Saved plots:")
    print(" - f1_binary_vs_threshold_val.png")
    print(" - f1_binary_vs_threshold_test.png")
    print(" - f1_weighted_vs_threshold_val.png")
    print(" - f1_weighted_vs_threshold_test.png")
else:
    print("No results to plot (all thresholds skipped).")
