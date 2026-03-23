"""
migrate_model.py — Extract model predictions to version-agnostic format.
Run once in the training environment, then the app uses the migrated version.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Load the original PKL (works in training env)
with open('models/best_classic_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

# Load the full feature cache
df_all = pd.read_csv('extracted_features_all_leads.csv')
feature_columns = pkg['feature_columns']

# Rebuild the representative dataset to get X_full
from ml_pipeline.beat_selector import build_representative_dataset
df_repr = build_representative_dataset(df_all)

X_full  = df_repr[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
y_full  = df_repr['label'].values.astype(int)
pids    = df_repr['patient_id'].values

# Extract the underlying model in a sklearn-version-agnostic way
# For tree ensembles: extract the trees themselves as ONNX or just save probabilities
# Simplest robust option: save as ONNX
# Replace the try/except in migrate_model.py with this:
pkg_migrated = {k: v for k, v in pkg.items() if k != 'model'}

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx

    inner = pkg['model']
    if hasattr(inner, 'calibrated_classifiers_'):
        inner = inner.calibrated_classifiers_[0].estimator

    n_features = X_full.shape[1]
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(inner, initial_types=initial_type, target_opset=17)

    with open('models/best_classic_model.onnx', 'wb') as f:
        onnx.save(onnx_model, f)

    pkg_migrated['model_format'] = 'onnx'
    pkg_migrated['model_path']   = 'models/best_classic_model.onnx'
    print("ONNX migration successful.")

except Exception as e:
    print(f"ONNX failed ({e}), saving prob_cache instead.")
    probs_full = pkg['model'].predict_proba(X_full)[:, 1]
    pkg_migrated['model_format'] = 'prob_cache'
    pkg_migrated['prob_cache']   = {
        'X_train':         X_full,
        'probs_train':     probs_full,
        'patient_ids':     pids,
        'feature_columns': feature_columns,
    }

# Always save — regardless of which path succeeded
with open('models/best_classic_model_migrated.pkl', 'wb') as f:
    pickle.dump(pkg_migrated, f)
print(f"Saved migrated model (format: {pkg_migrated['model_format']})")
