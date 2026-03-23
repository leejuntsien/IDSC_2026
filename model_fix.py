"""
migrate_model.py — Patches sklearn version mismatch and saves prob_cache.
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Monkey-patch for sklearn 1.5.1 → 1.8.0 attribute rename
_SI_orig = SimpleImputer.__getattribute__
def _si_compat(self, name):
    if name == '_fill_dtype':
        try:
            return _SI_orig(self, '_fill_dtype')
        except AttributeError:
            return _SI_orig(self, '_fit_dtype')
    return _SI_orig(self, name)
SimpleImputer.__getattribute__ = _si_compat

# ── Step 1: Load PKL (will warn about version mismatch — that's expected) ────
with open('models/best_classic_model.pkl', 'rb') as f:
    pkg = pickle.load(f)

# ── Step 2: Patch all SimpleImputer objects in the nested pipeline ────────────
# sklearn 1.8.0 renamed _fill_dtype → _fit_dtype
# Walk the entire object tree and fix any imputer missing the new attribute

def patch_imputers(obj, visited=None):
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(obj, SimpleImputer):
        if hasattr(obj, '_fill_dtype') and not hasattr(obj, '_fit_dtype'):
            obj._fit_dtype = obj._fill_dtype
            print(f"  Patched SimpleImputer: _fill_dtype → _fit_dtype")
        elif not hasattr(obj, '_fit_dtype'):
            # Neither exists — set a safe default
            obj._fit_dtype = np.float64
            print(f"  Patched SimpleImputer: added _fit_dtype = np.float64")

    # Recurse into common sklearn container attributes
    for attr in ['steps', 'calibrated_classifiers_', 'estimators_',
                 'estimator', 'base_estimator']:
        child = getattr(obj, attr, None)
        if child is None:
            continue
        if isinstance(child, list):
            for item in child:
                if hasattr(item, '__dict__'):
                    patch_imputers(item, visited)
                # Handle (name, estimator) tuples from Pipeline.steps
                if isinstance(item, tuple) and len(item) == 2:
                    patch_imputers(item[1], visited)
        elif hasattr(child, '__dict__'):
            patch_imputers(child, visited)

print("Patching SimpleImputer attributes...")
patch_imputers(pkg['model'])
print("Patching complete.")

# ── Step 3: Verify the patch works ───────────────────────────────────────────
print("Testing model.predict_proba on dummy data...")
try:
    dummy = np.zeros((1, len(pkg['feature_columns'])))
    _ = pkg['model'].predict_proba(dummy)
    print("predict_proba test PASSED.")
    patch_ok = True
except Exception as e:
    print(f"predict_proba still failing: {e}")
    patch_ok = False

# ── Step 4: Build prob_cache using patched model ──────────────────────────────
from ml_pipeline.beat_selector import build_representative_dataset

df_all  = pd.read_csv('extracted_features_all_leads.csv')
df_repr = build_representative_dataset(df_all)

feature_columns = pkg['feature_columns']
available = [c for c in feature_columns if c in df_repr.columns]
X_full    = df_repr[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0).values
y_full    = df_repr['label'].values.astype(int)
pids      = df_repr['patient_id'].values

pkg_migrated = {k: v for k, v in pkg.items() if k != 'model'}

if patch_ok:
    print("Generating probability cache from patched model...")
    probs_full = pkg['model'].predict_proba(X_full)[:, 1]
    pkg_migrated['model_format'] = 'prob_cache'
    pkg_migrated['prob_cache']   = {
        'X_train':         X_full,
        'probs_train':     probs_full,
        'patient_ids':     pids,
        'feature_columns': feature_columns,
    }
    print(f"Probability cache built: {len(probs_full)} samples, "
          f"mean prob={probs_full.mean():.4f}")
else:
    # Last resort — save patched model directly and hope 1.8.0 tolerates it
    pkg_migrated['model']        = pkg['model']
    pkg_migrated['model_format'] = 'sklearn_patched'
    print("Saving patched sklearn model directly.")

with open('models/best_classic_model_migrated.pkl', 'wb') as f:
    pickle.dump(pkg_migrated, f)
print(f"Saved: models/best_classic_model_migrated.pkl "
      f"(format: {pkg_migrated['model_format']})")