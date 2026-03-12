# Brugada Syndrome ECG Dataset

## Overview

This dataset contains electrocardiographic (ECG) recordings from 363 individuals with suspected Brugada Syndrome. Brugada syndrome is a rare but potentially life-threatening cardiac arrhythmia disorder, marked by distinctive ECG abnormalities and an elevated risk of sudden cardiac death.

## Background

Brugada syndrome is characterized by a coved-type ST-segment elevation in the right precordial leads (V1–V3), frequently accompanied by a right bundle branch block pattern. Diagnosis is primarily clinical and is based on the identification of this ECG pattern — either occurring spontaneously or induced by sodium channel blockers — along with clinical criteria such as a history of syncope, documented ventricular arrhythmias, or a family history of sudden cardiac death.

## Data Acquisition

- **Sampling Frequency**: 100 Hz
- **Recording Duration**: 12 seconds per subject
- **Number of Leads**: 12 standard ECG leads
- **Total Subjects**: 363 individuals

## Folder Structure

```
├── metadata.csv                 # Clinical info about subjects
├── metadata_dictionary.csv      # Dictionary explaining metadata variables
├── RECORDS                      # List of all patient IDs
├── files/                       # ECG data files organized by patient ID
│   ├── 188981/
│   │   ├── 188981.dat         # ECG signal data
│   │   └── 188981.hea         # Header file with recording metadata
│   ├── 251972/
│   │   ├── 251972.dat
│   │   └── 251972.hea
│   └── [...]
```

## Data Characteristics

### Subject Distribution

Based on the `brugada` field in metadata:
- **0**: Healthy individuals
- **1**: Confirmed Brugada Syndrome diagnosis
- **2**: Other/atypical cases

### Clinical Variables

- **basal_pattern**: Indicates pathological baseline ECG patterns (independent of Brugada diagnosis)
- **sudden_death**: Critical outcome variable for risk assessment
- **brugada**: Primary diagnostic label

## Loading the Dataset

### Reading Metadata

```python
import pandas as pd

# Load metadata
metadata = pd.read_csv('metadata.csv')

# Load data dictionary
data_dict = pd.read_csv('metadata_dictionary.csv')

# Display basic statistics
print(metadata.head())
print(f"Total subjects: {len(metadata)}")
print(f"Brugada patients: {(metadata['brugada'] > 0).sum()}")
print(f"Healthy subjects: {(metadata['brugada'] == 0).sum()}")
```

### Reading ECG Signal Data

The ECG data is stored in WFDB (WaveForm DataBase) format, which is commonly used for physiological signals. You can use the `wfdb` Python package to read these files:

```python
import wfdb
import matplotlib.pyplot as plt

# Read a single patient's ECG
patient_id = '188981'
record = wfdb.rdrecord(f'files/{patient_id}/{patient_id}')

# Access the signal data
signals = record.p_signal  # Shape: (1200, 12) for 12s at 100Hz, 12 leads
lead_names = record.sig_name  # Lead names (I, II, III, aVR, aVL, aVF, V1-V6)
sampling_freq = record.fs  # Sampling frequency (100 Hz)

# Plot a specific lead
plt.figure(figsize=(12, 4))
plt.plot(signals[:, 0])  # Plot first lead
plt.title(f'Patient {patient_id} - {lead_names[0]}')
plt.xlabel('Sample')
plt.ylabel('Amplitude (mV)')
plt.show()
```