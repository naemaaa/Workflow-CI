import os
import sys
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import shap

# ── Argparse ──────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Sepsis ICU Model Training')
    p.add_argument('--n_estimators',     type=int,   default=100)
    p.add_argument('--max_depth',        type=int,   default=10)
    p.add_argument('--learning_rate',    type=float, default=0.1)
    p.add_argument('--n_trials',         type=int,   default=30,
                   help='Jumlah Optuna trials')
    p.add_argument('--model_type',       type=str,   default='xgboost',
                   choices=['xgboost', 'random_forest', 'both'],
                   help='Model yang dilatih')
    p.add_argument('--data_dir',         type=str,   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sepsis_preprocessing'),
                   help='Folder berisi file CSV preprocessing')
    p.add_argument('--output_dir',       type=str,   default='./artifacts',
                   help='Folder output untuk artefak')
    p.add_argument('--dagshub_username', type=str,   default='',
                   help='DagsHub username (kosongkan untuk simpan lokal)')
    p.add_argument('--dagshub_repo',     type=str,   default='',
                   help='DagsHub repository name')
    p.add_argument('--no_dagshub',       action='store_true',
                   help='Force simpan MLflow lokal')
    return p.parse_args()


# ── DagsHub Setup ──────────────────────────────────────────────────────────────
def setup_tracking(args):
    use_dagshub = (
        not args.no_dagshub
        and args.dagshub_username
        and args.dagshub_repo
    )
    if use_dagshub:
        try:
            import dagshub
            dagshub.init(
                repo_owner=args.dagshub_username,
                repo_name=args.dagshub_repo,
                mlflow=True
            )
            print(f"✅ DagsHub: {args.dagshub_username}/{args.dagshub_repo}")
            return True
        except Exception as e:
            print(f"⚠️  DagsHub gagal ({e}), lanjut lokal...")
    return False


# ── Load Data ──────────────────────────────────────────────────────────────────
def load_data(data_dir):
    train_path = os.path.join(data_dir, 'sepsis_preprocessing_train.csv')
    test_path  = os.path.join(data_dir, 'sepsis_preprocessing_test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"File tidak ditemukan: {train_path}\n"
            f"Jalankan automate_NamaKamu.py terlebih dahulu."
        )

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    X_train = train_df.drop('SepsisLabel', axis=1)
    y_train = train_df['SepsisLabel']
    X_test  = test_df.drop('SepsisLabel', axis=1)
    y_test  = test_df['SepsisLabel']

    print(f"Train : {X_train.shape}  Sepsis: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
    print(f"Test  : {X_test.shape}   Sepsis: {y_test.sum():,}  ({y_test.mean()*100:.1f}%)")
    return X_train, X_test, y_train, y_test


# ── Artefak Helpers ─────────────────────────────────────────────────────────────
def save_confusion_matrix(y_true, y_pred, name, output_dir):
    cm   = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['Non-Sepsis', 'Sepsis']).plot(
        ax=ax, cmap='Blues', colorbar=False
    )
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax.set_title(f'{name}\nSensitivity: {sensitivity:.3f}', fontweight='bold')
    path = os.path.join(output_dir, f'confusion_matrix_{name}.png')
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    return path


def save_roc_pr(y_true, y_prob, name, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, color='#E74C3C', lw=2, label=f'AUC={auc:.3f}')
    axes[0].plot([0,1],[0,1],'k--', lw=1); axes[0].legend()
    axes[0].set(title=f'ROC — {name}', xlabel='FPR', ylabel='TPR')

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[1].plot(rec, prec, color='#2E75B6', lw=2, label=f'AP={ap:.3f}')
    axes[1].axhline(y_true.mean(), color='gray', linestyle='--', label='Baseline')
    axes[1].legend()
    axes[1].set(title=f'PR Curve — {name}', xlabel='Recall', ylabel='Precision')

    path = os.path.join(output_dir, f'roc_pr_{name}.png')
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    return path


def save_threshold_plot(y_true, y_prob, name, output_dir):
    thresholds  = np.arange(0.1, 0.9, 0.01)
    f1s         = [f1_score(y_true, (y_prob>=t).astype(int), zero_division=0)
                   for t in thresholds]
    recalls     = [recall_score(y_true, (y_prob>=t).astype(int), zero_division=0)
                   for t in thresholds]
    precisions  = [precision_score(y_true, (y_prob>=t).astype(int), zero_division=0)
                   for t in thresholds]
    best_t      = thresholds[np.argmax(f1s)]
    best_f1     = max(f1s)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, f1s,        label='F1',        color='#2E75B6', lw=2)
    ax.plot(thresholds, recalls,    label='Recall',    color='#E74C3C', lw=2)
    ax.plot(thresholds, precisions, label='Precision', color='#27AE60', lw=2)
    ax.axvline(best_t, color='purple', linestyle='--', lw=1.5,
               label=f'Best F1 threshold={best_t:.2f}')
    ax.axvline(0.5, color='gray', linestyle=':', lw=1, label='Default 0.5')
    ax.set(title=f'Threshold Optimization — {name}', xlabel='Threshold', ylabel='Score')
    ax.legend(); ax.set_xlim(0.1, 0.9)

    path = os.path.join(output_dir, f'threshold_{name}.png')
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    return path, best_t, best_f1


def save_shap(model, X_sample, name, output_dir):
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # Bar plot
        plt.figure(figsize=(9, 7))
        shap.summary_plot(sv, X_sample, show=False, max_display=20, plot_type='bar')
        plt.title(f'SHAP Importance — {name}', fontweight='bold')
        path_bar = os.path.join(output_dir, f'shap_bar_{name}.png')
        plt.tight_layout(); plt.savefig(path_bar, dpi=120, bbox_inches='tight'); plt.close()

        # Beeswarm
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_sample, show=False, max_display=20)
        plt.title(f'SHAP Beeswarm — {name}', fontweight='bold')
        path_bee = os.path.join(output_dir, f'shap_beeswarm_{name}.png')
        plt.tight_layout(); plt.savefig(path_bee, dpi=120, bbox_inches='tight'); plt.close()

        return path_bar, path_bee
    except Exception as e:
        print(f"  ⚠️  SHAP error: {e}")
        return None, None


def save_feature_importance(model, feature_names, name, output_dir, top_n=25):
    if not hasattr(model, 'feature_importances_'):
        return None
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 8))
    colors = ['#E74C3C' if any(k in f for k in ['Lactate','WBC','sofa','HR_trend','pH'])
              else '#4A90D9' for f in fi_df['feature']]
    ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1],
            color=colors[::-1], edgecolor='white')
    ax.set(title=f'Feature Importance — {name}', xlabel='Importance')
    path = os.path.join(output_dir, f'feature_importance_{name}.png')
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches='tight'); plt.close()
    return path


# ── Train XGBoost ──────────────────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test, args, output_dir):
    print("\n⚡ Training XGBoost...")
    spw = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        p = {
            'n_estimators'     : trial.suggest_int('n_estimators', 100, 500),
            'max_depth'        : trial.suggest_int('max_depth', 3, 12),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
            'gamma'            : trial.suggest_float('gamma', 0, 5),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 0, 5),
            'scale_pos_weight' : spw,
            'use_label_encoder': False,
            'eval_metric'      : 'auc',
            'random_state'     : 42,
            'n_jobs'           : -1,
        }
        m = xgb.XGBClassifier(**p)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        'scale_pos_weight' : spw,
        'use_label_encoder': False,
        'eval_metric'      : 'auc',
        'random_state'     : 42,
        'n_jobs'           : -1,
    })

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    thresh_path, best_t, best_f1 = save_threshold_plot(y_test, y_prob, 'XGBoost', output_dir)
    y_pred_opt = (y_prob >= best_t).astype(int)

    return model, y_prob, y_pred_opt, best_t, best_f1, best_params, study, thresh_path


# ── Train Random Forest ─────────────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test, args, output_dir):
    print("\n🌲 Training Random Forest...")

    def objective(trial):
        p = {
            'n_estimators'     : trial.suggest_int('n_estimators', 50, 300),
            'max_depth'        : trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features'     : trial.suggest_categorical('max_features',
                                                           ['sqrt', 'log2', 0.5]),
            'class_weight'     : 'balanced',
            'random_state'     : 42,
            'n_jobs'           : -1,
        }
        m = RandomForestClassifier(**p)
        m.fit(X_train, y_train)
        return roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1})

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    thresh_path, best_t, best_f1 = save_threshold_plot(y_test, y_prob, 'RF', output_dir)
    y_pred_opt = (y_prob >= best_t).astype(int)

    return model, y_prob, y_pred_opt, best_t, best_f1, best_params, study, thresh_path


# ── Log ke MLflow ───────────────────────────────────────────────────────────────
def log_to_mlflow(model, model_name, y_test, y_prob, y_pred_opt,
                  best_t, best_f1, best_params, study,
                  X_train, X_test, output_dir, args):

    with mlflow.start_run(run_name=f'{model_name}_Optuna') as run:

        # ── Parameters ──
        log_params = {k: v for k, v in best_params.items()
                      if k not in ['use_label_encoder', 'eval_metric',
                                   'n_jobs', 'random_state']}
        mlflow.log_params(log_params)
        mlflow.log_param('model_type',       model_name)
        mlflow.log_param('n_trials',         args.n_trials)
        mlflow.log_param('optimal_threshold', best_t)
        mlflow.log_param('tuning_method',    'optuna_tpe')
        mlflow.log_param('train_shape',      str(X_train.shape))
        mlflow.log_param('test_shape',       str(X_test.shape))

        # ── Metrics ──
        mlflow.log_metric('accuracy',              accuracy_score(y_test, y_pred_opt))
        mlflow.log_metric('precision',             precision_score(y_test, y_pred_opt, zero_division=0))
        mlflow.log_metric('recall',                recall_score(y_test, y_pred_opt, zero_division=0))
        mlflow.log_metric('f1_score',              best_f1)
        mlflow.log_metric('roc_auc',               roc_auc_score(y_test, y_prob))
        mlflow.log_metric('avg_precision_score',   average_precision_score(y_test, y_prob))
        mlflow.log_metric('specificity',           _specificity(y_test, y_pred_opt))
        mlflow.log_metric('optuna_best_auc',       study.best_value)
        mlflow.log_metric('false_negative_rate',   _fnr(y_test, y_pred_opt))
        mlflow.log_metric('false_positive_rate',   _fpr_metric(y_test, y_pred_opt))

        # ── Artefak ──
        safe_name = model_name.replace(' ', '_')
        artifacts = []

        artifacts.append(save_confusion_matrix(y_test, y_pred_opt, safe_name, output_dir))
        artifacts.append(save_roc_pr(y_test, y_prob, safe_name, output_dir))

        thresh_path = os.path.join(output_dir, f'threshold_{safe_name}.png')
        if os.path.exists(thresh_path):
            artifacts.append(thresh_path)

        fi_path = save_feature_importance(model, X_train.columns.tolist(),
                                          safe_name, output_dir)
        if fi_path: artifacts.append(fi_path)

        shap_bar, shap_bee = save_shap(model, X_test.iloc[:500], safe_name, output_dir)
        for p in [shap_bar, shap_bee]:
            if p: artifacts.append(p)

        # Optuna summary JSON
        summary = {
            'best_auc'    : study.best_value,
            'best_params' : study.best_params,
            'n_trials'    : args.n_trials,
            'model_type'  : model_name,
        }
        summary_path = os.path.join(output_dir, f'optuna_{safe_name}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        artifacts.append(summary_path)

        for path in artifacts:
            if path and os.path.exists(path):
                mlflow.log_artifact(path)

        # ── Log Model ──
        if 'XGBoost' in model_name:
            mlflow.xgboost.log_model(model, f'{safe_name}_model',
                                      registered_model_name=f'Sepsis_{safe_name}')
        else:
            mlflow.sklearn.log_model(model, f'{safe_name}_model',
                                      registered_model_name=f'Sepsis_{safe_name}')

        run_id = run.info.run_id
        print(f"  MLflow Run ID: {run_id}")
        return run_id


# ── Metric Helpers ──────────────────────────────────────────────────────────────
def _specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def _fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0

def _fpr_metric(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0


# ── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*65)
    print("  SEPSIS ICU — CI TRAINING PIPELINE")
    print("="*65)
    print(f"  Model     : {args.model_type}")
    print(f"  Trials    : {args.n_trials}")
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")

    # Setup tracking
    setup_tracking(args)
    mlflow.set_experiment('Sepsis_ICU_CI')

    # Load data
    print("\n📂 Loading data...")
    X_train, X_test, y_train, y_test = load_data(args.data_dir)

    results = {}

    if args.model_type in ('xgboost', 'both'):
        model, y_prob, y_pred_opt, best_t, best_f1, best_params, study, _ = \
            train_xgboost(X_train, X_test, y_train, y_test, args, args.output_dir)
        run_id = log_to_mlflow(model, 'XGBoost', y_test, y_prob, y_pred_opt,
                               best_t, best_f1, best_params, study,
                               X_train, X_test, args.output_dir, args)
        results['xgboost'] = {
            'roc_auc': roc_auc_score(y_test, y_prob),
            'f1'     : best_f1,
            'recall' : recall_score(y_test, y_pred_opt, zero_division=0),
            'run_id' : run_id
        }

    if args.model_type in ('random_forest', 'both'):
        model, y_prob, y_pred_opt, best_t, best_f1, best_params, study, _ = \
            train_random_forest(X_train, X_test, y_train, y_test, args, args.output_dir)
        run_id = log_to_mlflow(model, 'RandomForest', y_test, y_prob, y_pred_opt,
                               best_t, best_f1, best_params, study,
                               X_train, X_test, args.output_dir, args)
        results['random_forest'] = {
            'roc_auc': roc_auc_score(y_test, y_prob),
            'f1'     : best_f1,
            'recall' : recall_score(y_test, y_pred_opt, zero_division=0),
            'run_id' : run_id
        }

    # Simpan results summary
    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*65)
    print("  ✅ TRAINING SELESAI")
    print("="*65)
    for model_name, res in results.items():
        print(f"  {model_name:<20} ROC AUC: {res['roc_auc']:.4f}  "
              f"F1: {res['f1']:.4f}  Recall: {res['recall']:.4f}")
    print(f"\n  Artefak tersimpan di: {args.output_dir}/")


if __name__ == '__main__':
    main()