"""
classic_pipeline.py — Sklearn pipeline for Classic ML Brugada classification.
V2 overhaul: scoring='f1' (minority class), probability calibration,
             threshold optimisation, patient-level evaluation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, learning_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    brier_score_loss, average_precision_score,
    roc_curve, auc, precision_recall_curve,
    matthews_corrcoef, f1_score, roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, \
    learning_curve, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


def build_classic_ml_pipeline(model_name='random_forest'):
    """Returns a Pipeline and hyperparameter grid for RandomizedSearchCV."""
    if model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10],
        }
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=42, class_weight='balanced')
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': ['scale', 'auto'],
        }
    elif model_name == 'knn':
        model = KNeighborsClassifier()
        param_grid = {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
        }
    elif model_name == 'xgboost' and XGBClassifier is not None:
        model = XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42
        )
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
        }
    elif model_name == 'lightgbm' and LGBMClassifier is not None:
        model = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__num_leaves': [31, 50, 100],
        }
    elif model_name == 'isolation_forest':
        model = IsolationForest(random_state=42, contamination=0.1)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__contamination': [0.05, 0.1, 0.15],
        }
    else:
        raise ValueError(f"Model {model_name} is not supported or missing deps.")

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('model', model),
    ])
    param_grid['feature_selection__k'] = [5, 10, 20, 'all']
    return pipeline, param_grid


def plot_learning_curve(estimator, title, X, y, cv=5):
    """Generates training/validation learning curves."""
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='f1_macro', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                     alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                     alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validation Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def compute_shap_or_importance(best_model, X_train, feature_columns=None):
    """Extracts tree-based feature importances or runs SHAP."""
    global HAS_SHAP
    model_obj = best_model.named_steps['model']

    if HAS_SHAP and hasattr(model_obj, 'predict'):
        try:
            print("\n[SHAP] Generating SHAP summary plot...")
            X_t = best_model.named_steps['imputer'].transform(X_train)
            X_t = best_model.named_steps['scaler'].transform(X_t)
            X_t = best_model.named_steps['feature_selection'].transform(X_t)
            support = best_model.named_steps['feature_selection'].get_support()
            sel_feats = ([feature_columns[i] for i, m in enumerate(support) if m]
                         if feature_columns else None)
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_t)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            shap.summary_plot(shap_values, X_t, feature_names=sel_feats)
        except Exception as e:
            print(f"[SHAP] Error: {e}. Falling back to standard importances.")
            HAS_SHAP = False

    if not HAS_SHAP and hasattr(model_obj, 'feature_importances_'):
        print("\n[Explainability] Plotting Standard Feature Importances...")
        importances = model_obj.feature_importances_
        support = best_model.named_steps['feature_selection'].get_support()
        if feature_columns is None:
            sel_feats = [f"Feature {i}" for i in range(len(importances))]
        else:
            sel_feats = [feature_columns[i] for i, m in enumerate(support) if m]
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances (Top 15)")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)),
                   [sel_feats[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       feature_columns=None, model_name='random_forest',
                       n_iter=10, cv=5):
    """
    Trains model with RandomizedSearchCV (scoring='f1'), calibrates probabilities,
    finds optimal threshold via PR curve, and evaluates on test set.

    Returns: (best_model, optimal_threshold, probs_test, preds_test)
    """
    print(f"\n=======================================")
    print(f"--- Training {model_name.upper()} ---")
    print(f"=======================================")

    # Ensure labels are integer-typed for sklearn binary scoring
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    pipeline, param_grid = build_classic_ml_pipeline(model_name)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    # Use make_scorer with pos_label=1 to avoid multiclass error on float labels
    f1_binary_scorer = make_scorer(f1_score, pos_label=1, zero_division=0)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, n_iter=n_iter,
        scoring=f1_binary_scorer,
        cv=skf, random_state=42, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best Parameters: {search.best_params_}")

    # Probability calibration
    best_model = search.best_estimator_
    model_step = best_model.named_steps['model']
    if hasattr(model_step, 'decision_function') or hasattr(model_step, 'predict_proba'):
        try:
            calibrated_model = CalibratedClassifierCV(
                best_model, cv='prefit', method='isotonic'
            )
            calibrated_model.fit(X_train, y_train)
            calibrated = True
        except Exception:
            calibrated_model = best_model
            calibrated = False
    else:
        calibrated_model = best_model
        calibrated = False

    # Find optimal threshold on training set PR curve
    optimal_threshold = 0.5
    if hasattr(calibrated_model, 'predict_proba'):
        try:
            # Use CV probabilities on training set — not overfit training predictions
            cv_probs = cross_val_predict(
                best_model, X_train, y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
                method='predict_proba', n_jobs=-1
            )[:, 1]
            precision_cv, recall_cv, thresholds_cv = precision_recall_curve(
                y_train, cv_probs
            )
            f1_scores = (2 * (precision_cv * recall_cv)
                        / (precision_cv + recall_cv + 1e-8))
            optimal_threshold = float(thresholds_cv[np.argmax(f1_scores[:-1])])
            print(f"[Threshold] Optimal (CV PR): {optimal_threshold:.4f}"
                f" | Calibrated: {calibrated}")
        except Exception as e:
            print(f"[Threshold] CV failed ({e}), using 0.5")

    # Test-set evaluation with optimal threshold
    print("\n[Test Set Evaluation]")
    if hasattr(calibrated_model, 'predict_proba'):
        probs_test = calibrated_model.predict_proba(X_test)[:, 1]
        preds = (probs_test >= optimal_threshold).astype(int)
    else:
        preds = calibrated_model.predict(X_test)
        probs_test = preds.astype(float)

    print(classification_report(y_test, preds, target_names=['Normal', 'Brugada']))

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    mcc = matthews_corrcoef(y_test, preds)
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity:          {specificity:.4f}")
    print(f"MCC: {mcc:.4f}")

    if hasattr(calibrated_model, 'predict_proba'):
        brier = brier_score_loss(y_test, probs_test)
        pr_auc_val = average_precision_score(y_test, probs_test)
        roc_auc_val = roc_auc_score(y_test, probs_test)
        print(f"Brier Score: {brier:.4f}")
        print(f"ROC-AUC: {roc_auc_val:.4f}")
        print(f"PR-AUC:  {pr_auc_val:.4f}")

    # Learning curve
    print("\n[Training Curves]")
    plot_learning_curve(best_model, f"Learning Curve: {model_name.upper()}",
                        X_train, y_train, cv=skf)

    # SHAP
    compute_shap_or_importance(best_model, X_train, feature_columns)

    return calibrated_model, optimal_threshold, probs_test, preds


# ──────────────────────────────────────────────────────────────────────────────
# Patient-level evaluation  (Step 5c)
# ──────────────────────────────────────────────────────────────────────────────

def patient_level_evaluate(y_test, preds_test, probs_test, patient_ids_test):
    """
    Aggregates beat-level predictions to patient level.
    Rule: if EITHER beat (median or outlier) is predicted positive → patient positive.
    """
    patient_df = pd.DataFrame({
        'patient_id': patient_ids_test,
        'y_true': y_test,
        'pred': preds_test,
        'prob': probs_test,
    })

    patient_agg = patient_df.groupby('patient_id').agg(
        y_true=('y_true', 'max'),
        pred_positive=('pred', 'max'),
        max_prob=('prob', 'max'),
    ).reset_index()

    y_true_p = patient_agg['y_true'].values
    y_pred_p = patient_agg['pred_positive'].values
    y_prob_p = patient_agg['max_prob'].values

    print("\n[Patient-Level Metrics]")
    print(classification_report(y_true_p, y_pred_p,
                                target_names=['Normal', 'Brugada']))

    cm = confusion_matrix(y_true_p, y_pred_p)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    mcc = matthews_corrcoef(y_true_p, y_pred_p)
    roc_auc_val = roc_auc_score(y_true_p, y_prob_p) if len(np.unique(y_true_p)) > 1 else 0.0
    pr_auc_val = average_precision_score(y_true_p, y_prob_p) if len(np.unique(y_true_p)) > 1 else 0.0

    print(f"TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}")
    print(f"Sensitivity (Recall): {sens:.4f}")
    print(f"Specificity:          {spec:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"ROC-AUC: {roc_auc_val:.4f}")
    print(f"PR-AUC:  {pr_auc_val:.4f}")

    return patient_agg
