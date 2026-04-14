#!/usr/bin/env python3
"""
Models:
    XGBoost             - gradient boosting
    LightGBM            - fast gradient boosting
    CatBoost            - gradient boosting w/ builtin categorical handling
    Random Forest       - bagged decision trees
    SVM (RBF)           - support vector machine, radial basis function kernel
    SVM (Sigmoid)       - support vector machine, sigmoid kernel
    Logistic Regression - linear probabilistic classifier
    KNN                 - k-nearest neighbours
    Gaussian NB         - naive Bayes probabilistic classifier
Usage:
    python classify.py
    python classify.py --train-csv path/to/train.csv \\
                       --val-csv   path/to/val.csv   \\
                       --test-csv  path/to/test.csv  \\
                       --outdir    path/to/results   \\
                       --threshold 300
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from sklearn.preprocessing   import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.pipeline         import Pipeline
from sklearn.inspection       import permutation_importance
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.svm              import SVC
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (
    accuracy_score, classification_report, f1_score,
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    precision_score, recall_score
)
import sklearn.base as skbase
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore', category=UserWarning, module='catboost')

# Used as threshold so we don't discard any ancient samples 
# as this is the most recent sample from the dataset
# 
# If sample age > NEWEST_ANCIENT_AGE_YEARS = ancient
# If sample age < NEWEST_ANCIENT_AGE_YEARS = modern
NEWEST_ANCIENT_AGE_YEARS = 2000

PRESENT_YEAR = 2026
FEATURES     = ['NRC_AVERAGE_AGE', 'CG_CONTENT', 'N_CONTENT', 'RELATIVE_SIZE']

# Number of permutation repeats for importance (higher = more stable, slower)
N_PERMUTATION_REPEATS = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'generated', 'features', 'drive_features')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'generated', 'results')

MODEL_COLORS = {
    'XGBoost':             '#E63946',
    'LightGBM':            '#FF8C69',
    'CatBoost':            '#FFBA08',
    'RandomForest':        '#2DC653',
    'SVM_RBF':             '#457B9D',
    'SVM_Sigmoid':         '#A8DADC',
    'LogisticRegression':  '#1D3557',
    'KNN':                 '#9B5DE5',
    'GaussianNB':          '#F15BB5',
}

def ancient_precision_recall(y_test, y_pred):
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall    = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    return precision, recall

def load_split(csv_path):
    df = pd.read_csv(csv_path, dtype={'ID': str})
    if 'AGE' not in df.columns:
        raise ValueError(f'No AGE column in {csv_path}. Run add_age_column.py first.')

    ages_calendar = pd.to_numeric(df['AGE'], errors='coerce')
    ages_ago      = PRESENT_YEAR - ages_calendar
    ages_ago[ages_calendar == 0] = 0 # modern sentinel = 0 years ago

    n_missing = ages_ago.isna().sum()
    if n_missing:
        print(f'WARNING: dropping {n_missing} rows with missing AGE in '
              f'{os.path.basename(csv_path)}')
        mask     = ages_ago.notna()
        df       = df[mask].reset_index(drop=True)
        ages_ago = ages_ago[mask].reset_index(drop=True)

    return df, ages_ago

def make_labels(ages_ago, threshold_years):
    """Binary labels: 1=ancient if older than threshold_years, and if not 0=modern."""
    return (ages_ago > threshold_years).astype(int).values


# SVM, KNN, and LogReg use sklearn Pipelines with their own internal scalers
# so they have needs_scaling=False
def get_models():
    return [
        (
            'XGBoost',
            xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=1.0, eval_metric='logloss', random_state=42,
            ),
            False,
        ),
        (
            'LightGBM',
            lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05,
                num_leaves=31, random_state=42, verbose=-1,
            ),
            False,
        ),
        (
            'CatBoost',
            CatBoostClassifier(
                iterations=300, learning_rate=0.05,
                depth=6, eval_metric='F1',
                verbose=0, random_state=42,
            ),
            False,
        ),
        (
            'RandomForest',
            RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1,
            ),
            False,
        ),
        (
            'SVM_RBF',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('model',   SVC(kernel='rbf', C=10, gamma='scale',
                                probability=True, random_state=42)),
            ]),
            False,
        ),
        (
            'SVM_Sigmoid',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('model',   SVC(kernel='sigmoid', C=1, gamma='scale',
                                probability=True, random_state=42)),
            ]),
            False,
        ),
        (
            'LogisticRegression',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('model',   LogisticRegression(C=1, max_iter=5000,
                                               solver='lbfgs', random_state=42)),
            ]),
            False,
        ),
        (
            'KNN',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler',  StandardScaler()),
                ('model',   KNeighborsClassifier(n_neighbors=5, weights='uniform',
                                                  metric='minkowski')),
            ]),
            False,
        ),
        (
            'GaussianNB',
            GaussianNB(),
            True,
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def _get_builtin_importance(model, feature_names):
    """
    Extract native feature importances where available.
    - Tree models (XGBoost, LightGBM, CatBoost, RandomForest): feature_importances_
    - Linear models (LogisticRegression): abs(coef_), normalized
    - Pipeline: unwraps to the final estimator first
    Returns a dict {feature: score} or None if unsupported (SVM, KNN, GaussianNB).
    """
    estimator = model[-1] if isinstance(model, Pipeline) else model

    if hasattr(estimator, 'feature_importances_'):
        scores = estimator.feature_importances_
        return dict(zip(feature_names, scores / (scores.sum() + 1e-12)))

    if hasattr(estimator, 'coef_'):
        coef = np.abs(estimator.coef_)
        if coef.ndim > 1:
            coef = coef.mean(axis=0)
        return dict(zip(feature_names, coef / (coef.sum() + 1e-12)))

    return None  # KNN, SVM, GaussianNB have no native importance


def _get_permutation_importance(model, X_test, y_test, feature_names,
                                 n_repeats=N_PERMUTATION_REPEATS):
    """
    Model-agnostic importance: shuffle each feature and measure accuracy drop.
    Works for every model including SVM and KNN.
    Negative values (feature hurts when present) are clipped to 0.
    Returns a dict {feature: normalized_score}.
    """
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring='accuracy',
        n_jobs=-1,
    )
    scores = np.clip(result.importances_mean, 0, None)
    return dict(zip(feature_names, scores / (scores.sum() + 1e-12)))


def compute_feature_importance(fitted_models, X_test, y_test, feature_names,
                                needs_scaling_map, scaler):
    """
    For every fitted model collect:
      - built-in importance  (tree / linear models only)
      - permutation importance (all models)

    fitted_models : list of (name, fitted_clf, needs_scaling)
    X_test        : raw (unscaled) test features, numpy array
    needs_scaling_map : dict {name: bool} - True only for GaussianNB
    scaler        : the global StandardScaler fitted on train+val

    Returns a DataFrame: rows=features, columns=model×method scores.
    """
    records = {f: {} for f in feature_names}

    for name, clf, needs_scaling in fitted_models:
        # Models that are sklearn Pipelines carry their own scaler internally,
        # so always pass raw X. Only GaussianNB uses the global scaler.
        X_eval = scaler.transform(X_test) if needs_scaling else X_test

        # Permutation
        perm = _get_permutation_importance(clf, X_eval, y_test, feature_names)
        for f in feature_names:
            records[f][f'{name}'] = perm[f]

    df = pd.DataFrame(records).T
    df.index.name = 'feature'
    df['aggregate_score'] = df.mean(axis=1)
    df['rank'] = df['aggregate_score'].rank(ascending=False).astype(int)
    return df.sort_values('rank')


def print_importance_table(df_imp):
    print('\n' + '═' * 55)
    print('  FEATURE IMPORTANCE RANKING (most → least important)')
    print('═' * 55)
    summary = df_imp[['rank', 'aggregate_score']].copy()
    summary.columns = ['Rank', 'Aggregate Score']
    print(summary.to_string())
    print('═' * 55)
    top = df_imp.index[0]
    print(f'\n★  Most important feature: "{top}"  '
          f'(aggregate score: {df_imp.loc[top, "aggregate_score"]:.4f})\n')


def plot_feature_importance(df_imp, threshold_years, outdir):
    """
    Two-panel figure:
      Left:  Aggregate importance bar chart (all features)
      Right: Heatmap of importance per model/method
    """
    features  = df_imp.index.tolist()
    score_cols = [c for c in df_imp.columns if c not in ('aggregate_score', 'rank')]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(features) * 0.7)))
    fig.suptitle(
        f'Permutation Feature Importance (threshold={threshold_years} years ago)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    # ── Left: aggregate bar chart ──────────────────────────────────────────
    ax1 = axes[0]
    scores = df_imp['aggregate_score'].values
    colors = cm.viridis(scores / (scores.max() + 1e-12))
    bars   = ax1.barh(features[::-1], scores[::-1], color=colors[::-1],
                      edgecolor='white')
    ax1.set_xlabel('Aggregate Importance Score (normalized)', fontsize=10)
    ax1.set_title('Overall Feature Ranking', fontsize=11, fontweight='bold')
    ax1.spines[['top', 'right']].set_visible(False)
    for bar, val in zip(bars, scores[::-1]):
        ax1.text(bar.get_width() + 0.002,
                 bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', ha='left', fontsize=9)

    # ── Right: heatmap ──────────────────────────────────────────────────────
    ax2 = axes[1]
    heat_data = df_imp[score_cols].values
    im = ax2.imshow(heat_data, aspect='auto', cmap='YlOrRd',
                    vmin=0, vmax=heat_data.max())
    ax2.set_yticks(range(len(features)))
    ax2.set_yticklabels(features, fontsize=9)
    ax2.set_xticks(range(len(score_cols)))
    ax2.set_xticklabels(score_cols, rotation=40, ha='right', fontsize=7)
    ax2.set_title('Feature Importance by Model', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Normalized importance', shrink=0.7)
    for i in range(len(features)):
        for j in range(len(score_cols)):
            val = heat_data[i, j]
            text_color = 'white' if val > heat_data.max() * 0.6 else 'black'
            ax2.text(j, i, f'{val:.3f}', ha='center', va='center',
                     fontsize=7, color=text_color)

    plt.tight_layout()
    path = os.path.join(outdir, 'feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(train_df, train_ages, val_df, val_ages,
                        test_df, test_ages, threshold_years):
    """
    Fits every model on train+val combined, evaluates on test.
    Returns (results dict, fitted_models list, X_test array, y_test array, scaler).
    """
    y_train = make_labels(train_ages, threshold_years)
    y_val   = make_labels(val_ages,   threshold_years)
    y_test  = make_labels(test_ages,  threshold_years)

    X_train = train_df[FEATURES].values.astype(np.float32)
    X_val   = val_df[FEATURES].values.astype(np.float32)
    X_test  = test_df[FEATURES].values.astype(np.float32)

    X_fit = np.vstack([X_train, X_val])
    y_fit = np.concatenate([y_train, y_val])

    if len(np.unique(y_fit)) < 2:
        raise ValueError(
            f'Training data has only one class at threshold={threshold_years} years. '
            'Change NEWEST_ANCIENT_AGE_YEARS.'
        )
    if len(np.unique(y_test)) < 2:
        raise ValueError(
            f'Test data has only one class at threshold={threshold_years} years. '
            'Change NEWEST_ANCIENT_AGE_YEARS.'
        )

    scaler = StandardScaler().fit(X_fit)
    X_fit_s  = scaler.transform(X_fit)
    X_test_s = scaler.transform(X_test)

    n_pos = int(y_test.sum())
    n_neg = int(len(y_test) - n_pos)
    print(f'  Test set: {n_pos} ancient, {n_neg} modern '
          f'({n_pos/(n_pos+n_neg)*100:.1f}% ancient)')

    results       = {}
    fitted_models = []   # (name, fitted_clf, needs_scaling)

    for name, model, needs_scaling in get_models():
        clf  = skbase.clone(model)
        X_tr = X_fit_s  if needs_scaling else X_fit
        X_te = X_test_s if needs_scaling else X_test

        clf.fit(X_tr, y_fit)
        y_pred = clf.predict(X_te)

        proba_raw = clf.predict_proba(X_te)
        y_proba   = (proba_raw[:, 1] if proba_raw.shape[1] > 1
                     else np.full(len(y_test), float(clf.classes_[0])))

        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_b = f1_score(y_test, y_pred, average='binary',   zero_division=0)
        acc  = accuracy_score(y_test, y_pred)
        prec_ancient, rec_ancient = ancient_precision_recall(y_test, y_pred)

        results[name] = {
            'f1_weighted':       f1_w,
            'f1_binary':         f1_b,
            'accuracy':          acc,
            'auroc':             roc_auc_score(y_test, y_proba),
            'auprc':             average_precision_score(y_test, y_proba),
            'precision_ancient': prec_ancient,
            'recall_ancient':    rec_ancient,
            'y_test':            y_test,
            'y_proba':           y_proba,
            'y_pred':            y_pred,
        }
        fitted_models.append((name, clf, needs_scaling))

        print(f'{name:<22} | F1w={f1_w:.4f}  F1b={f1_b:.4f}  '
              f'Acc={acc:.4f}  AUROC={results[name]["auroc"]:.4f}')

    return results, fitted_models, X_test, y_test, scaler

def print_summary_table(results, threshold_years):
    print(f'\n')
    print(f'Results at threshold = {threshold_years} years ago'
          f'({threshold_years/100:.1f} centuries)')
    print(f'\n')
    header = f'{"Model":<22}  {"F1 Weighted":>11}  {"F1 Binary":>9}'
    header += f'{"Accuracy":>8}  {"AUROC":>7}  {"AUPRC":>7}'
    print(header)
    print(f'\n')
    for name, m in results.items():
        print(f'{name:<22}  {m["f1_weighted"]:>11.4f}  {m["f1_binary"]:>9.4f}  '
              f'{m["accuracy"]:>8.4f}  {m["auroc"]:>7.4f}  {m["auprc"]:>7.4f}')
    print(f'\n')

def print_summary_table_2(results, threshold_years):
    print(f'\nResults at threshold = {threshold_years} years ago ({threshold_years/100:.1f} centuries)\n')
    header = f'{"Model":<22}  {"Accuracy":>9}  {"AUC-ROC":>7}  {"Prec (Anc)":>10}  {"Recall (Anc)":>12}  {"F1 Weighted":>11}  {"F1 Binary":>9}'
    print(header)
    print('-' * len(header))
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for name, m in sorted_results:
        print('('
            f'"{name + '",':<22}  '
            f'{m["accuracy"]:>8.4f},  '
            f'{m["auroc"]:>6.4f},  '
            f'{m["precision_ancient"]:>9.4f},  '
            f'{m["recall_ancient"]:>11.4f},  '
            f'{m["f1_weighted"]:>10.4f},  '
            f'{m["f1_binary"]:>8.4f},  '
            '),'
        )
    print()

def plot_roc_curves(results, threshold_years, outdir):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    for name, m in results.items():
        fpr, tpr, _ = roc_curve(m['y_test'], m['y_proba'])
        ax.plot(fpr, tpr,
                color=MODEL_COLORS.get(name, 'grey'),
                linewidth=2,
                label=f"{name} (AUROC={m['auroc']:.3f})")

    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_title(f'ROC Curves - threshold={threshold_years} years ago',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to: {path}')

def plot_pr_curves(results, threshold_years, outdir):
    fig, ax = plt.subplots(figsize=(8, 7))
    pos_rate = results[next(iter(results))]['y_test'].mean()
    ax.axhline(pos_rate, color='grey', linestyle=':', linewidth=1, label='No-skill')

    for name, m in results.items():
        prec, rec, _ = precision_recall_curve(m['y_test'], m['y_proba'])
        ax.plot(rec, prec,
                color=MODEL_COLORS.get(name, 'grey'),
                linewidth=2,
                label=f"{name} (AUPRC={m['auprc']:.3f})")

    ax.set_xlabel('Recall',    fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f'Precision-Recall Curves - threshold={threshold_years} years ago',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'pr_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to: {path}')

def plot_f1_bar(results, threshold_years, outdir):
    names = list(results.keys())
    f1w = [results[n]['f1_weighted'] for n in names]
    f1b = [results[n]['f1_binary']   for n in names]
    colors = [MODEL_COLORS.get(n, 'grey') for n in names]

    x     = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 5))
    bars_w = ax.bar(x - width/2, f1w, width, label='F1 Weighted',
                    color=colors, alpha=0.9, edgecolor='white')
    bars_b = ax.bar(x + width/2, f1b, width, label='F1 Binary',
                    color=colors, alpha=0.5, edgecolor='white', hatch='//')

    for bar, val in zip(bars_w, f1w):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    for bar, val in zip(bars_b, f1b):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title(f'F1 Scores by Model — threshold={threshold_years} years ago',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'f1_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to: {path}')


def plot_accuracy_bar(results, threshold_years, outdir):
    names = list(results.keys())
    accs = [results[n]['accuracy'] for n in names]
    colors = [MODEL_COLORS.get(n, 'grey') for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_title(f'Accuracy by Model — threshold={threshold_years} years ago',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'accuracy_bar.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved to: {path}')

def main():
    parser = argparse.ArgumentParser(
        description='Train all classifiers at a fixed ancient/modern threshold.'
    )
    parser.add_argument('--train-csv', default=os.path.join(DATA_DIR, 'train_features_with_age.csv'))
    parser.add_argument('--val-csv', default=os.path.join(DATA_DIR, 'val_features_with_age.csv'))
    parser.add_argument('--test-csv', default=os.path.join(DATA_DIR, 'test_features_with_age.csv'))
    parser.add_argument('--outdir', default=OUT_DIR)
    parser.add_argument('--threshold', type=int, default=None,
                        help='Override NEWEST_ANCIENT_AGE_YEARS from the command line '
                             '(years ago). Default: uses NEWEST_ANCIENT_AGE_YEARS variable.')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --threshold overrides the variable
    threshold_years = args.threshold if args.threshold is not None else NEWEST_ANCIENT_AGE_YEARS

    print(f'Threshold: {threshold_years} years ago '
          f'({threshold_years/100:.1f} centuries)')
    print(f'Definition: samples older than {threshold_years} years = ancient (1)\n')

    print('Loading data')
    train_df, train_ages = load_split(args.train_csv)
    val_df, val_ages = load_split(args.val_csv)
    test_df, test_ages = load_split(args.test_csv)
    print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')

    print('\nTraining & evaluating all models')
    results, fitted_models, X_test, y_test, scaler = train_and_evaluate(
        train_df, train_ages,
        val_df,   val_ages,
        test_df,  test_ages,
        threshold_years,
    )

    print_summary_table_2(results, threshold_years)

    rows = [{'model': n, **{k: v for k, v in m.items()
                             if k not in ('y_test', 'y_proba', 'y_pred')}}
            for n, m in results.items()]
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, 'results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'Saved results table to: {csv_path}')

    print('\nComputing feature importance across all models...')
    df_imp = compute_feature_importance(
        fitted_models, X_test, y_test, FEATURES,
        needs_scaling_map={name: ns for name, _, ns in fitted_models},
        scaler=scaler,
    )
    print_importance_table(df_imp)
    imp_csv = os.path.join(args.outdir, 'feature_importance.csv')
    df_imp.to_csv(imp_csv)
    print(f'Saved feature importance table to: {imp_csv}')

    print('\nGenerating plots')
    plot_roc_curves(results, threshold_years, args.outdir)
    plot_pr_curves(results, threshold_years, args.outdir)
    plot_f1_bar(results, threshold_years, args.outdir)
    plot_accuracy_bar(results, threshold_years, args.outdir)
    plot_feature_importance(df_imp, threshold_years, args.outdir)

    print(f'\nOutputs saved to: {args.outdir}')


if __name__ == '__main__':
    main()