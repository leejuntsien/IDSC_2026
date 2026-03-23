import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, learning_curve
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix,
    brier_score_loss, average_precision_score,
    roc_curve, auc, precision_recall_curve,
    matthews_corrcoef
)
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
    """
    Returns a scalable Pipeline object and a hyperparameter grid for RandomizedSearchCV.
    """
    if model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [None, 10, 20, 30],
            'model__min_samples_split': [2, 5, 10]
        }
    elif model_name == 'svm':
        model = SVC(probability=True, random_state=42, class_weight='balanced')
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': ['scale', 'auto']
        }
    elif model_name == 'knn':
        model = KNeighborsClassifier()
        param_grid = {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance']
        }
    elif model_name == 'xgboost' and XGBClassifier is not None:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__max_depth': [3, 5, 7]
        }
    elif model_name == 'lightgbm' and LGBMClassifier is not None:
        model = LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__learning_rate': [0.01, 0.1, 0.2],
            'model__num_leaves': [31, 50, 100]
        }
    elif model_name == 'isolation_forest':
        model = IsolationForest(random_state=42, contamination=0.1)
        param_grid = {
            'model__n_estimators': [50, 100, 200],
            'model__contamination': [0.05, 0.1, 0.15]
        }
    else:
        raise ValueError(f"Model {model_name} is not supported or missing dependencies.")

    # Create Pipeline handling NaNs (e.g. missing U or P waves) and scaling
    # Include a feature selection step to reduce noise and prevent overfitting
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()), 
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('model', model)
    ])
    
    # Add feature selection k as a hyperparameter to loop over
    param_grid['feature_selection__k'] = [5, 10, 20, 'all']
    
    return pipeline, param_grid

def plot_learning_curve(estimator, title, X, y, cv=5):
    """
    Generates a simple plot of the test and training learning curve.
    """
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score (Macro)")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='f1_macro', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5))
        
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Validation (CV Test) Score")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def compute_shap_or_importance(best_model, X_train, feature_columns=None):
    """
    Extracts tree-based Feature Importances or runs SHAP if available.
    """
    global HAS_SHAP
    model_obj = best_model.named_steps['model']
    
    if HAS_SHAP and hasattr(model_obj, 'predict'):
        try:
            print("\\n[SHAP] Generating SHAP summary plot...")
            # We must transform X_train through the pipeline preprocessing first
            # to feed it purely into the model object
            X_transformed = best_model.named_steps['imputer'].transform(X_train)
            X_transformed = best_model.named_steps['scaler'].transform(X_transformed)
            X_transformed = best_model.named_steps['feature_selection'].transform(X_transformed)
            
            # Get the retained feature names
            support = best_model.named_steps['feature_selection'].get_support()
            selected_features = [feature_columns[i] for i, mask in enumerate(support) if mask]
            if not selected_features:
                selected_features = None # Fallback
                
            explainer = shap.TreeExplainer(model_obj)
            shap_values = explainer.shap_values(X_transformed)
            
            # If binary classification, shap_values might be a list of two arrays
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
                
            shap.summary_plot(shap_values, X_transformed, feature_names=selected_features)
        except Exception as e:
            print(f"[SHAP] Error during SHAP computation: {e}. Falling back to standard importances.")
            HAS_SHAP = False # trigger fallback
            
    if not HAS_SHAP and hasattr(model_obj, 'feature_importances_'):
        print("\\n[Explainability] Plotting Standard Feature Importances...")
        importances = model_obj.feature_importances_
        
        # Determine subset of features used
        support = best_model.named_steps['feature_selection'].get_support()
        if feature_columns is None:
            selected_features = [f"Feature {i}" for i in range(len(importances))]
        else:
            selected_features = [feature_columns[i] for i, mask in enumerate(support) if mask]
            
        indices = np.argsort(importances)[::-1][:15] # Top 15
        
        plt.figure(figsize=(10, 6))
        plt.title("Classic ML Feature Importances (Top 15)")
        plt.bar(range(len(indices)), importances[indices], align="center")
        plt.xticks(range(len(indices)), [selected_features[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def train_and_evaluate(X_train, y_train, X_test, y_test, feature_columns=None, 
                       model_name='random_forest', n_iter=10, cv=5):
    print(f"\n=======================================")
    print(f"--- Training {model_name.upper()} ---")
    print(f"=======================================")
    pipeline, param_grid = build_classic_ml_pipeline(model_name)
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, n_iter=n_iter,
        scoring='f1_macro', cv=skf, random_state=42, n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    print(f"Best Parameters: {search.best_params_}")
    
    print("\n[Test Set Evaluation]")
    preds = best_model.predict(X_test)
    print(classification_report(y_test, preds))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # Derived metrics
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    mcc = matthews_corrcoef(y_test, preds)        # Matthews Correlation Coefficient
    
    print(f"PPV (Precision): {ppv:.4f}")
    print(f"NPV: {npv:.4f}")
    print(f"MCC: {mcc:.4f}")
    
    if hasattr(best_model.named_steps['model'], "predict_proba"):
        probs = best_model.predict_proba(X_test)[:, 1]
        
        # Brier Score
        brier = brier_score_loss(y_test, probs)
        print(f"Brier Score Loss: {brier:.4f} (Lower is better)")
        
        # PR-AUC
        pr_auc = average_precision_score(y_test, probs)
        print(f"PR-AUC: {pr_auc:.4f}")
        
        # Plot ROC and PR curves
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(y_test, probs)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")
        
        ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend(loc="lower left")
        plt.show()
        
    print("\n[Training Curves]")
    plot_learning_curve(best_model, f"Learning Curve: {model_name.upper()}", X_train, y_train, cv=skf)
    
    # SHAP or Feature Importances
    compute_shap_or_importance(best_model, X_train, feature_columns)
        
    return best_model
