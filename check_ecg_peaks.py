"""
check_peaks.py — Standalone peak detection audit across all patients.
Run independently to diagnose which records fail and why.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import neurokit2 as nk
from scipy.signal import find_peaks

from ecg_pipeline_features import (
    apply_notch_filter, detect_peaks
)

metadata = pd.read_csv("brugada-huca/metadata.csv")
metadata['brugada'] = (metadata['brugada'] > 0).astype(int)

results = []
failed_pids = []

for _, row in metadata.iterrows():
    pid   = str(row['patient_id'])
    label = int(row['brugada'])
    path  = f"brugada-huca/files/{pid}/{pid}"

    if not os.path.exists(path + ".dat"):
        continue

    rec    = wfdb.rdrecord(path)
    fs     = rec.fs
    signal = rec.p_signal[:, rec.sig_name.index('V1')]  # check V1 only

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sig_clean = apply_notch_filter(signal, fs)
            peaks     = detect_peaks(sig_clean, fs)
            n_peaks   = len(peaks)
            mean_rr   = float(np.mean(np.diff(peaks)) / fs) if len(peaks) > 1 else np.nan
            hr_bpm    = 60.0 / mean_rr if not np.isnan(mean_rr) else np.nan
            status    = 'ok'
        except Exception as e:
            n_peaks = 0
            mean_rr = np.nan
            hr_bpm  = np.nan
            status  = str(e)[:80]
            failed_pids.append(pid)

    results.append({
        'patient_id': pid,
        'label':      label,
        'n_peaks':    n_peaks,
        'mean_rr_s':  round(mean_rr, 3) if not np.isnan(mean_rr) else np.nan,
        'hr_bpm':     round(hr_bpm, 1)  if not np.isnan(hr_bpm)  else np.nan,
        'status':     status,
    })

df = pd.DataFrame(results)

print(f"\nTotal patients:  {len(df)}")
print(f"OK:              {(df['status']=='ok').sum()}")
print(f"Failed:          {(df['status']!='ok').sum()}")
print(f"\nFailed Brugada:  {df[(df['status']!='ok') & (df['label']==1)].shape[0]}")
print(f"Failed Normal:   {df[(df['status']!='ok') & (df['label']==0)].shape[0]}")

print("\nFailure reasons:")
print(df[df['status']!='ok']['status'].value_counts())

print("\nPeak count distribution (successful records):")
ok = df[df['status']=='ok']
print(ok['n_peaks'].describe())

# Flag physiologically suspicious even if not failed
suspicious = df[
    (df['status']=='ok') &
    ((df['hr_bpm'] < 30) | (df['hr_bpm'] > 120))
]
print(f"\nPhysiologically suspicious (HR <40 or >150 BPM): {len(suspicious)}")
print(suspicious[['patient_id','label','n_peaks','hr_bpm']].to_string())

df.to_csv("peak_detection_audit.csv", index=False)
print("\nSaved peak_detection_audit.csv")

# Plot worst 4 failures for visual inspection
if failed_pids:
    fig, axes = plt.subplots(min(4, len(failed_pids)), 1, figsize=(14, 10))
    if len(failed_pids) == 1:
        axes = [axes]
    for ax, pid in zip(axes, failed_pids[:4]):
        path   = f"brugada-huca/files/{pid}/{pid}"
        rec    = wfdb.rdrecord(path)
        signal = rec.p_signal[:, rec.sig_name.index('V1')]
        ax.plot(signal, linewidth=0.8)
        ax.set_title(f"Patient {pid} — V1 (FAILED peak detection)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("mV")
    plt.tight_layout()
    plt.savefig("figures/failed_peak_detection.png", dpi=150)
    print("Saved figures/failed_peak_detection.png")