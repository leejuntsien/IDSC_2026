"""
app/streamlit_app.py — Brugada ECG Detection System
Three detection modes: Discrete (Classic ML), Temporal (CNN+BiGRU), ERSI (Statistical)
"""
import os, sys, tempfile, pickle, warnings
import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import wfdb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_pipeline.data_loader import (
    load_wfdb_record, extract_discrete_features, extract_sequence_features
)
from ecg_pipeline_features import (
    apply_notch_filter, detect_peaks, segment_beats_by_rr
)
from ml_pipeline.layer1_filter import run_layer1_on_patient
from ml_pipeline.ersi_detector import BrugadaERSIDetector
from ml_pipeline.dl_pipeline import ECGTemporalCNN

warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brugada ECG Detector — IDSC 2026",
    page_icon="🫀",
    layout="wide",
)

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

# ── Model loaders (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_classic_model():
    # Try migrated version first (version-agnostic)
    migrated_path = 'models/best_classic_model_migrated.pkl'
    original_path = 'models/best_classic_model.pkl'

    path = migrated_path if os.path.exists(migrated_path) else original_path
    if not os.path.exists(path):
        return None

    with open(path, 'rb') as f:
        pkg = pickle.load(f)
    return pkg

def predict_classic(pkg, X_inf):
    """
    Version-agnostic prediction. Handles ONNX, prob_cache, or direct sklearn.
    """
    fmt = pkg.get('model_format', 'sklearn')

    if fmt == 'onnx':
        import onnxruntime as rt
        sess  = rt.InferenceSession(pkg['model_path'])
        input_name  = sess.get_inputs()[0].name
        probs = sess.run(None, {input_name: X_inf.astype(np.float32)})[1]
        # onnxruntime returns list of dicts for classifiers
        if isinstance(probs[0], dict):
            probs = np.array([p[1] for p in probs])
        else:
            probs = np.array(probs)[:, 1]
        return probs

    elif fmt == 'prob_cache':
        # Nearest-neighbour lookup in training probability cache
        from scipy.spatial.distance import cdist
        cache    = pkg['prob_cache']
        X_train  = cache['X_train']
        p_train  = cache['probs_train']
        # For each inference sample, find closest training sample
        # Impute NaN before distance
        X_inf_c  = np.nan_to_num(X_inf, nan=0.0)
        X_train_c = np.nan_to_num(X_train, nan=0.0)
        dists    = cdist(X_inf_c, X_train_c, metric='euclidean')
        nearest  = np.argmin(dists, axis=1)
        probs    = p_train[nearest]
        return probs

    else:
        # Original sklearn path — may fail on version mismatch
        return pkg['model'].predict_proba(X_inf)[:, 1]

@st.cache_resource
def load_temporal_model():
    path = 'models/best_temporal_model.pt'
    if not os.path.exists(path):
        return None
    pkg = torch.load(path, map_location='cpu', weights_only=False) #pkg   = torch.load(path, map_location='cpu')
    cfg   = pkg['model_config']
    model = ECGTemporalCNN(**cfg)
    model.load_state_dict(pkg['model_state_dict'])
    model.eval()
    return model, pkg

@st.cache_resource
def load_ersi_model():
    path = 'models/ersi_detector.pkl'
    if not os.path.exists(path):
        return None
    return BrugadaERSIDetector.load(path)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🫀 Brugada Detector")
st.sidebar.markdown("**IDSC 2026 — Brugada-HUCA Dataset**")

mode = st.sidebar.radio(
    "Detection Mode",
    ["Discrete (Classic ML)", "Temporal (CNN+BiGRU)", "ERSI (Statistical)"],
    help=(
        "**Discrete**: Tabular ST morphology features + trained classifier.\n\n"
        "**Temporal**: Sliding window CNN+BiGRU captures inter-beat dynamics.\n\n"
        "**ERSI**: Entropy-Ranked Stability Index — sliding window entropy "
        "fusion (ERSI_full dual-ranked). No trained classifier."
    )
)

STANDARD_LEADS = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
lead_display   = st.sidebar.multiselect(
    "Leads to display",
    STANDARD_LEADS + ['VCG', 'RMS'],
    default=['V1', 'V2', 'V3'],
)
show_overlay = st.sidebar.checkbox("Show overlaid beat plot (V1)", value=True)
fs_override  = st.sidebar.number_input("Sampling rate override (0=auto)", value=0, min_value=0)

# ── File upload ───────────────────────────────────────────────────────────────
st.title("Brugada Syndrome ECG Detection")
st.markdown(
    "Upload a WFDB record (`.dat` + `.hea`). "
    "The system extracts beats, runs the selected detection mode, "
    "and highlights suspicious segments."
)

col_up1, col_up2 = st.columns(2)
with col_up1:
    dat_file = st.file_uploader("Upload .dat file", type=['dat'])
with col_up2:
    hea_file = st.file_uploader("Upload .hea file", type=['hea'])

if not (dat_file and hea_file):
    st.info("Upload both .dat and .hea files to begin.")
    st.stop()

with tempfile.TemporaryDirectory() as tmpdir:
    stem     = dat_file.name.replace('.dat', '')
    dat_path = os.path.join(tmpdir, dat_file.name)
    hea_path = os.path.join(tmpdir, hea_file.name)
    with open(dat_path, 'wb') as f: f.write(dat_file.read())
    with open(hea_path, 'wb') as f: f.write(hea_file.read())
    record_path = os.path.join(tmpdir, stem)

    try:
        df_signal, fs = load_wfdb_record(record_path)
        if fs_override > 0:
            fs = int(fs_override)
    except Exception as e:
        st.error(f"Failed to load record: {e}")
        st.stop()

    st.success(f"Loaded: **{stem}** | {fs}Hz | {len(df_signal)} samples | "
               f"Leads: {list(df_signal.columns)}")

    # ── ECG Signal Plot ───────────────────────────────────────────────────────
    st.subheader("ECG Signal")

    display_std = [l for l in lead_display if l in STANDARD_LEADS and l in df_signal.columns]
    n_rows      = len(display_std) + (1 if 'RMS' in lead_display else 0) + \
                  (3 if 'VCG' in lead_display else 0)

    if n_rows > 0:
        fig_ecg, axes = plt.subplots(n_rows, 1, figsize=(14, 2.5*n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]
        time  = np.arange(len(df_signal)) / fs
        ax_i  = 0

        for lead in display_std:
            sig_c = apply_notch_filter(df_signal[lead].values, fs)
            try:
                rp = detect_peaks(sig_c, fs)
            except Exception:
                rp = np.array([])
            axes[ax_i].plot(time, sig_c, color='#1a1a2e', linewidth=0.8, label=lead)
            if len(rp):
                axes[ax_i].scatter(time[rp], sig_c[rp], color='red', s=15, zorder=5)
            axes[ax_i].set_ylabel(f"{lead}\n(mV)", fontsize=8)
            axes[ax_i].grid(True, alpha=0.3)
            ax_i += 1

        if 'RMS' in lead_display:
            avail = [c for c in STANDARD_LEADS if c in df_signal.columns]
            rms   = np.sqrt(np.mean(np.square(df_signal[avail].values), axis=1))
            axes[ax_i].plot(time, apply_notch_filter(rms, fs),
                            color='#6c63ff', linewidth=0.8, label='RMS')
            axes[ax_i].set_ylabel("RMS\n(mV)", fontsize=8)
            axes[ax_i].grid(True, alpha=0.3)
            ax_i += 1

        if 'VCG' in lead_display:
            from ecg_pipeline_features import combine_to_vcg
            vcg = combine_to_vcg(df_signal.copy())
            for vcg_lead, color in zip(['VCG_x','VCG_y','VCG_z'],
                                       ['#e94560','#0f3460','#16213e']):
                axes[ax_i].plot(time, vcg[vcg_lead].values,
                                color=color, linewidth=0.8, label=vcg_lead)
                axes[ax_i].set_ylabel(vcg_lead, fontsize=8)
                axes[ax_i].grid(True, alpha=0.3)
                ax_i += 1

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        st.pyplot(fig_ecg)
        plt.close(fig_ecg)

    # ── Beat Overlay ─────────────────────────────────────────────────────────
    if show_overlay:
        st.subheader("Beat Overlay — Epoch Plot")

        overlay_lead = st.selectbox(
            "Lead for beat overlay",
            [l for l in STANDARD_LEADS if l in df_signal.columns],
            index=min(6, len([l for l in STANDARD_LEADS if l in df_signal.columns])-1),
            key='overlay_lead'
        )

        sig_ol = apply_notch_filter(df_signal[overlay_lead].values, fs)
        try:
            rp_ol  = detect_peaks(sig_ol, fs)
            # Use neurokit2 epoch segmentation
            import neurokit2 as nk
            ecg_epochs = nk.ecg_segment(sig_ol, rpeaks=rp_ol, sampling_rate=fs)

            fig_ov, ax_ov = plt.subplots(figsize=(10, 4))

            beat_arrays = []
            for epoch_id, epoch_df in ecg_epochs.items():
                signal_col = 'Signal' if 'Signal' in epoch_df.columns else epoch_df.columns[0]
                beat = epoch_df[signal_col].values
                beat_arrays.append(beat)
                time_ax = epoch_df.index.values  # time in seconds, R-peak at 0
                ax_ov.plot(time_ax, beat, color='#1a1a2e', linewidth=0.6, alpha=0.4)

            # Compute and overlay median beat
            if beat_arrays:
                from scipy.signal import resample as scipy_resample
                tgt_len  = int(np.median([len(b) for b in beat_arrays]))
                resampled = np.array([scipy_resample(b, tgt_len) for b in beat_arrays])
                median_beat = np.median(resampled, axis=0)
                time_med = np.linspace(
                    min(e.index.values[0] for e in ecg_epochs.values() if len(e) > 0),
                    max(e.index.values[-1] for e in ecg_epochs.values() if len(e) > 0),
                    tgt_len
                )
                ax_ov.plot(time_med, median_beat, color='#e94560', linewidth=2.5,
                        label=f'Median beat (n={len(beat_arrays)})', zorder=5)

            ax_ov.axvline(0, color='green', linestyle='--', lw=1, label='R-peak (t=0)')
            ax_ov.axvspan(0.04, 0.12, alpha=0.1, color='orange', label='~ST region')
            ax_ov.set_xlabel("Time relative to R-peak (s)")
            ax_ov.set_ylabel(f"{overlay_lead} (mV)")
            ax_ov.set_title(f"{overlay_lead} — All beats + median overlay")
            ax_ov.legend(fontsize=8)
            ax_ov.grid(True, alpha=0.3)
            st.pyplot(fig_ov)
            plt.close(fig_ov)

            # If predicted Brugada: load a reference normal median beat for comparison
            # Use pre-saved reference from training data or a synthetic normal template
            normal_ref_path = 'models/normal_reference_beat.npy'
            if os.path.exists(normal_ref_path) and beat_arrays:
                normal_ref = np.load(normal_ref_path)
                fig_comp, ax_comp = plt.subplots(figsize=(10, 4))
                time_comp = np.linspace(-0.4, 0.6, tgt_len)

                # This patient's median
                ax_comp.plot(time_comp, median_beat / (np.max(np.abs(median_beat)) + 1e-8),
                            color='#e94560', lw=2.5, label='This patient (median, normalised)')

                # Reference normal
                normal_resampled = scipy_resample(normal_ref, tgt_len)
                ax_comp.plot(time_comp, normal_resampled / (np.max(np.abs(normal_resampled)) + 1e-8),
                            color='#0f3460', lw=2.5, linestyle='--',
                            label='Reference normal (dataset median)')

                ax_comp.axvline(0,    color='green',  linestyle=':', lw=1, label='R-peak')
                ax_comp.axvspan(0.04, 0.12, alpha=0.1, color='orange', label='~ST region')
                ax_comp.set_xlabel("Time relative to R-peak (s)")
                ax_comp.set_ylabel("Normalised amplitude")
                ax_comp.set_title(f"{overlay_lead} — Patient median vs Reference normal")
                ax_comp.legend(fontsize=8)
                ax_comp.grid(True, alpha=0.3)
                st.pyplot(fig_comp)
                plt.close(fig_comp)

        except Exception as e:
            st.warning(f"Beat overlay failed: {e}")

    # ── Detection ─────────────────────────────────────────────────────────────
    st.subheader(f"Detection — {mode}")
    st.divider()
    res_col, met_col = st.columns([2, 1])

    # ── MODE 1: Discrete Classic ML ───────────────────────────────────────────
    if mode == "Discrete (Classic ML)":
        pkg = load_classic_model()
        if pkg is None:
            st.error("Classic model not found at models/best_classic_model.pkl. "
                     "Run run_brugada_classic_ml.py first.")
            st.stop()

        with st.spinner("Extracting discrete morphological features..."):
            try:
                df_feat = extract_discrete_features(
                    df_signal, fs, patient_id=stem,
                    target_leads=pkg['best_leads']
                )
                df_feat = run_layer1_on_patient(df_feat, stem, leads=('V1','V2','V3'))
            except Exception as e:
                st.error(f"Feature extraction failed: {e}")
                st.stop()

        from ml_pipeline.beat_selector import build_representative_dataset
        df_repr_s = build_representative_dataset(df_feat)

        available_cols = [c for c in pkg['feature_columns'] if c in df_repr_s.columns]
        missing_cols   = [c for c in pkg['feature_columns'] if c not in df_repr_s.columns]

        if missing_cols:
            st.warning(
                f"{len(missing_cols)} features absent for this record "
                f"(wave delineation failed for some peaks — P-wave most common). "
                f"Filling with 0. Prediction may be less reliable."
            )

        X_df = df_repr_s[available_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        for col in missing_cols:
            X_df[col] = 0.0
        X_inf = X_df[pkg['feature_columns']].values
        probs = predict_classic(pkg, X_inf)
        preds = (probs >= pkg['threshold']).astype(int)

        patient_pred = int(preds.max())
        patient_prob = float(probs.max())
        l1_any       = bool(df_feat['layer1_suspected'].any())

        # SHAP top features
        top_feats = []
        try:
            import shap
            model_step = pkg['model'].named_steps['model'] \
                         if hasattr(pkg['model'], 'named_steps') \
                         else pkg['model'].estimator.named_steps['model']
            explainer  = shap.TreeExplainer(model_step)
            sv         = explainer.shap_values(X_inf)
            if isinstance(sv, list): sv = sv[1]
            top_idx  = np.argsort(np.abs(sv[0]))[::-1][:10]
            top_feats = [(pkg['feature_columns'][i], float(sv[0][i])) for i in top_idx]
        except Exception:
            pass

        with res_col:
            if patient_pred:
                st.error(f"⚠️ **BRUGADA SUSPECTED**\n\n"
                         f"Probability: **{patient_prob:.1%}**  |  "
                         f"Layer 1 ST flag: {'✅ Yes' if l1_any else '❌ No'}")
            else:
                st.success(f"✅ **NORMAL**\n\nProbability: **{patient_prob:.1%}**")

            # Highlight flagged beats on V1
            if 'V1' in df_signal.columns:
                flagged = df_feat[df_feat['layer1_suspected']]['beat_index'].tolist()
                if flagged:
                    sig_v1f   = apply_notch_filter(df_signal['V1'].values, fs)
                    rp_f      = detect_peaks(sig_v1f, fs)
                    segs_f    = segment_beats_by_rr(sig_v1f, rp_f, fs)
                    fig_f, ax_f = plt.subplots(figsize=(14, 3))
                    ax_f.plot(np.arange(len(sig_v1f))/fs, sig_v1f,
                              color='#1a1a2e', lw=0.8, label='V1')
                    for seg in segs_f:
                        if seg['beat_idx'] in flagged:
                            ax_f.axvspan(seg['start_idx']/fs, seg['end_idx']/fs,
                                         alpha=0.3, color='red',
                                         label='Layer 1 flagged' if seg['beat_idx']==flagged[0] else '')
                    ax_f.set_xlabel("Time (s)")
                    ax_f.set_ylabel("mV")
                    ax_f.legend(fontsize=8)
                    ax_f.grid(True, alpha=0.3)
                    st.pyplot(fig_f)
                    plt.close(fig_f)

            if top_feats:
                st.markdown("**Top SHAP features**")
                feat_df = pd.DataFrame(top_feats, columns=['Feature', 'SHAP Value'])
                feat_df['Direction'] = feat_df['SHAP Value'].apply(
                    lambda x: '↑ Brugada' if x > 0 else '↓ Normal'
                )
                st.dataframe(feat_df, use_container_width=True)

        with met_col:
            st.metric("Prediction", "Brugada" if patient_pred else "Normal")
            st.metric("Confidence",   f"{patient_prob:.1%}")
            st.metric("Layer 1 Flag", "Yes" if l1_any else "No")
            st.metric("Model",        pkg['model_name'].upper())
            st.metric("CV MCC",       f"{pkg['cv_mcc_mean']:.3f}±{pkg['cv_mcc_std']:.3f}")
            st.metric("CV Sensitivity", f"{pkg['cv_sensitivity']:.1%}")

    # ── MODE 2: Temporal CNN+BiGRU ────────────────────────────────────────────
    elif mode == "Temporal (CNN+BiGRU)":
        result = load_temporal_model()
        if result is None:
            st.error("Temporal model not found at models/best_temporal_model.pt. "
                     "Run run_brugada_temporal_dl.py first.")
            st.stop()
        model_t, pkg_t = result

        with st.spinner("Running sliding window CNN+BiGRU inference..."):
            try:
                lm = pkg_t.get('leads_mode', 'right_precordial_V1V2V3')
                if 'rms' in lm:
                    seqs, rrs = extract_sequence_features(
                        df_signal, fs, use_rms=True,
                        method='interpolate', target_len=200, return_rr=True)
                elif 'vcg' in lm:
                    seqs, rrs = extract_sequence_features(
                        df_signal, fs, use_vcg=True,
                        method='interpolate', target_len=200, return_rr=True)
                else:
                    sel_df = df_signal[[l for l in ['V1','V2','V3']
                                        if l in df_signal.columns]]
                    seqs, rrs = extract_sequence_features(
                        sel_df, fs, use_all_leads=True,
                        method='interpolate', target_len=200, return_rr=True)
            except Exception as e:
                st.error(f"Sequence extraction failed: {e}")
                st.stop()

        n_beats  = pkg_t['model_config']['n_beats']
        thresh_t = pkg_t['threshold']
        n_total  = seqs.shape[0]

        if n_total < n_beats:
            pad = n_beats - n_total
            seqs = np.concatenate([seqs, np.zeros((pad, *seqs.shape[1:]))], axis=0)
            rrs  = np.concatenate([rrs,  np.zeros(pad)], axis=0)
            n_total = n_beats

        all_probs, all_windows = [], []
        model_t.eval()
        with torch.no_grad():
            for start in range(0, n_total - n_beats + 1):
                x_w = np.array(seqs[start:start+n_beats], dtype=np.float32)
                r_w = np.array(rrs[start:start+n_beats],  dtype=np.float32)
                if x_w.ndim == 2: x_w = x_w[:, :, np.newaxis]
                x_t = torch.tensor(x_w).permute(0, 2, 1).unsqueeze(0)
                r_t = torch.tensor(r_w).unsqueeze(0)
                logits, attn = model_t(x_t, r_t)
                prob = float(torch.sigmoid(logits).item())
                all_probs.append(prob)
                all_windows.append({
                    'start': start, 'end': start+n_beats-1, 'prob': prob,
                    'attn': attn.squeeze().numpy() if attn is not None else None
                })

        max_prob     = max(all_probs)
        patient_pred = int(max_prob >= thresh_t)
        flagged_wins = [w for w in all_windows if w['prob'] >= thresh_t]

        with res_col:
            if patient_pred:
                st.error(f"⚠️ **BRUGADA SUSPECTED**\n\n"
                         f"Peak window probability: **{max_prob:.1%}**  |  "
                         f"Flagged windows: **{len(flagged_wins)}** / {len(all_windows)}")
            else:
                st.success(f"✅ **NORMAL**\n\nPeak probability: **{max_prob:.1%}**")

            # Window probability timeline
            fig_w, ax_w = plt.subplots(figsize=(12, 3))
            ax_w.plot(all_probs, color='#0f3460', lw=1.5, label='P(Brugada)')
            ax_w.axhline(thresh_t, color='red', linestyle='--', lw=1,
                         label=f'Threshold ({thresh_t:.2f})')
            ax_w.fill_between(range(len(all_probs)),
                              thresh_t,
                              [max(p, thresh_t) for p in all_probs],
                              alpha=0.3, color='red')
            ax_w.set_xlabel("Window (beat position)")
            ax_w.set_ylabel("P(Brugada)")
            ax_w.set_ylim(0, 1.05)
            ax_w.set_title("Sliding Window Detection Timeline")
            ax_w.legend(fontsize=8)
            ax_w.grid(True, alpha=0.3)
            st.pyplot(fig_w)
            plt.close(fig_w)

            # Attention overlay on highest-probability window
            if flagged_wins and 'V1' in df_signal.columns:
                best_w  = max(flagged_wins, key=lambda w: w['prob'])
                sig_v1a = apply_notch_filter(df_signal['V1'].values, fs)
                rp_a    = detect_peaks(sig_v1a, fs)
                segs_a  = segment_beats_by_rr(sig_v1a, rp_a, fs)
                from scipy.signal import resample as scipy_resample
                s_b, e_b = best_w['start'], min(best_w['end'], len(segs_a)-1)
                combined = np.concatenate([segs_a[i]['signal']
                                           for i in range(s_b, e_b+1)
                                           if i < len(segs_a)])
                attn_w = best_w['attn']
                if attn_w is not None and len(combined) > 0:
                    attn_up = np.repeat(attn_w,
                                        len(combined)//len(attn_w)+1)[:len(combined)]
                    fig_at, ax_at = plt.subplots(figsize=(12, 3))
                    ax_at.plot(combined, color='black', lw=1.0, label='V1 (window)')
                    ymin, ymax = combined.min()-0.3, combined.max()+0.3
                    ax_at.imshow(attn_up.reshape(1,-1), aspect='auto',
                                 cmap='Reds', alpha=0.5,
                                 extent=[0, len(combined), ymin, ymax])
                    ax_at.set_ylim(ymin, ymax)
                    ax_at.set_xlabel("Sample")
                    ax_at.set_ylabel("mV")
                    ax_at.set_title(
                        f"Attention overlay — Window {s_b}–{e_b} "
                        f"(P={best_w['prob']:.1%})\nDarker = higher model attention"
                    )
                    st.pyplot(fig_at)
                    plt.close(fig_at)

        with met_col:
            st.metric("Prediction",    "Brugada" if patient_pred else "Normal")
            st.metric("Peak Prob",     f"{max_prob:.1%}")
            st.metric("Flagged Win",   f"{len(flagged_wins)}/{len(all_windows)}")
            st.metric("Threshold",     f"{thresh_t:.3f}")
            st.metric("Model MCC",     f"{pkg_t['mcc']:.3f}")
            st.metric("Sensitivity",   f"{pkg_t['sensitivity']:.1%}")

    # ── MODE 3: ERSI Statistical ───────────────────────────────────────────────
    elif mode == "ERSI (Statistical)":
        ersi_det = load_ersi_model()
        if ersi_det is None:
            st.error("ERSI model not found at models/ersi_detector.pkl. "
                     "Run run_brugada_classic_ml.py first.")
            st.stop()

        if 'V1' not in df_signal.columns:
            st.error("V1 lead not found in this record. ERSI mode requires V1.")
            st.stop()

        with st.spinner("Computing ERSI_full (dual-ranked entropy stability)..."):
            v1_raw = df_signal['V1'].values
            ersi_full, is_brugada, evidence = ersi_det.score_patient(v1_raw)

        with res_col:
            if is_brugada:
                st.error(
                    f"⚠️ **BRUGADA SUSPECTED — ERSI Anomaly**\n\n"
                    f"ERSI_full score: **{ersi_full:.6f}**  "
                    f"(threshold: {evidence['threshold_full']:.6f})\n\n"
                    f"Confidence: **{min(ersi_full/evidence['threshold_full'], 2.0)*50:.0f}%** above threshold"
                )
            else:
                st.success(
                    f"✅ **NORMAL — ERSI within expected range**\n\n"
                    f"ERSI_full score: **{ersi_full:.6f}**  "
                    f"(threshold: {evidence['threshold_full']:.6f})"
                )

            # ERSI_full per-window bar chart (primary decision signal)
            timeline = evidence.get('window_ersi_timeline', [])
            if timeline:
                fig_er, ax_er = plt.subplots(figsize=(12, 3))
                colors = ['red' if v > evidence['threshold_full'] else '#0f3460'
                          for v in timeline]
                ax_er.bar(range(len(timeline)), timeline, color=colors)
                ax_er.axhline(evidence['threshold_full'], color='orange',
                              linestyle='--', lw=1.5,
                              label=f"Threshold ({evidence['threshold_full']:.4f})")
                ax_er.set_xlabel(f"Window index "
                                 f"({evidence['window_sec']}s windows, "
                                 f"{evidence['step_sec']}s step)")
                ax_er.set_ylabel("ERSI_full score")
                ax_er.set_title(
                    f"ERSI_full window timeline — {evidence['n_windows']} windows\n"
                    f"Dual-ranked (time × cross-entropy) | "
                    f"Features: {', '.join(evidence.get('selected_features') or ['all'])}"
                )
                ax_er.text(0.01, 0.95,
                           "High score = multiple entropy measures simultaneously elevated\n"
                           "(consistent with morphological anomaly, not noise artifact)",
                           transform=ax_er.transAxes, fontsize=7,
                           verticalalignment='top', color='gray')
                ax_er.legend(fontsize=8)
                ax_er.grid(True, alpha=0.3, axis='y')
                st.pyplot(fig_er)
                plt.close(fig_er)

            # ERSI_timeseries overlay (additive fusion — visual context only)
            ts_timeline = evidence.get('window_ersi_timeseries', [])
            if ts_timeline and timeline:
                fig_ts, ax_ts = plt.subplots(figsize=(12, 2))
                ax_ts.plot(ts_timeline, color='#6c63ff', lw=1.5,
                           label='ERSI_timeseries (additive fusion)')
                ax_ts.set_xlabel("Window index")
                ax_ts.set_ylabel("Score")
                ax_ts.set_title("ERSI_timeseries — additive entropy fusion (context only, not the decision signal)")
                ax_ts.legend(fontsize=8)
                ax_ts.grid(True, alpha=0.3)
                st.pyplot(fig_ts)
                plt.close(fig_ts)

            # Highlight anomalous windows on V1 raw signal
            if is_brugada and timeline:
                sig_v1e = apply_notch_filter(v1_raw, ersi_det.fs)
                time_e  = np.arange(len(sig_v1e)) / ersi_det.fs
                win_s   = int(ersi_det.window_sec * ersi_det.fs)
                step_s  = int(ersi_det.step_sec   * ersi_det.fs)

                fig_eh, ax_eh = plt.subplots(figsize=(14, 3))
                ax_eh.plot(time_e, sig_v1e, color='#1a1a2e', lw=0.8, label='V1')
                for i, score in enumerate(timeline):
                    if score > evidence['threshold_full']:
                        t_s = (i * step_s) / ersi_det.fs
                        t_e = t_s + ersi_det.window_sec
                        ax_eh.axvspan(t_s, t_e, alpha=0.25, color='red',
                                      label='ERSI anomalous window' if i == 0 else '')
                ax_eh.set_xlabel("Time (s)")
                ax_eh.set_ylabel("mV")
                ax_eh.set_title("V1 — ERSI anomalous windows highlighted")
                ax_eh.legend(fontsize=8)
                ax_eh.grid(True, alpha=0.3)
                st.pyplot(fig_eh)
                plt.close(fig_eh)

            # Raw entropy scores table
            with st.expander("Raw entropy scores (per-measure means)"):
                raw = {k: v for k, v in evidence['all_entropy_scores'].items()
                       if not k.endswith('_ERSI') and k not in
                       ['ERSI_timeseries', 'ERSI_full']}
                st.dataframe(pd.DataFrame([raw]).T.rename(columns={0: 'Mean value'}),
                             use_container_width=True)

        with met_col:
            st.metric("Prediction",   "Brugada" if is_brugada else "Normal")
            st.metric("ERSI_full",    f"{ersi_full:.5f}")
            st.metric("Threshold",    f"{evidence['threshold_full']:.5f}")
            st.metric("N Windows",    evidence['n_windows'])
            st.metric("Window size",  f"{evidence['window_sec']}s")
            st.metric("Variant",      "ERSI_full (dual-ranked)")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "**Mode explanations** — "
        "**Discrete**: Tabular ST morphology features (ST elevation at J+40ms/J+80ms, "
        "QRS duration, T-wave amplitude, R'/R ratio) with a trained classifier. "
        "**Temporal**: CNN encodes each beat morphologically; BiGRU captures how "
        "morphology evolves across consecutive beats — rate-dependent Brugada shows "
        "as gradual ST change across the window. "
        "**ERSI**: Entropy-Ranked Stability Index — Shannon, Tsallis, Rényi, sample, "
        "SVD entropies on 2s sliding windows, ranked by time and cross-entropy position. "
        "ERSI_full (dual-ranked) is the detection score. No trained classifier required."
    )
