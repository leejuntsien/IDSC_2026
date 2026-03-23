import wfdb, os, pandas as pd
from collections import Counter

metadata = pd.read_csv("brugada-huca/metadata.csv")
all_leads = []
missing = []

for _, row in metadata.iterrows():
    pid = str(row['patient_id'])
    path = f"brugada-huca/files/{pid}/{pid}"
    if os.path.exists(path + ".dat"):
        rec = wfdb.rdrecord(path)
        all_leads.extend(rec.sig_name)
    else:
        missing.append(pid)

print("Lead name counts:", Counter(all_leads).most_common(20))
print(f"Missing .dat files: {len(missing)}")

import pandas as pd

metadata = pd.read_csv("brugada-huca/metadata.csv")
metadata['brugada'] = (metadata['brugada'] > 0).astype(int)

cached = pd.read_csv("extracted_features_all_leads.csv")
extracted_pids = set(cached['patient_id'].astype(str).unique())
all_pids = set(metadata['patient_id'].astype(str))

dropped_pids = all_pids - extracted_pids
dropped_df = metadata[metadata['patient_id'].astype(str).isin(dropped_pids)]

print(f"Total patients:   {len(metadata)}")
print(f"Extracted:        {len(extracted_pids)}")
print(f"Dropped:          {len(dropped_pids)}")
print(f"Dropped Brugada:  {dropped_df['brugada'].sum()}")
print(f"Dropped Normal:   {(dropped_df['brugada']==0).sum()}")
print(f"Brugada dropout rate: {dropped_df['brugada'].sum() / metadata['brugada'].sum():.1%}")