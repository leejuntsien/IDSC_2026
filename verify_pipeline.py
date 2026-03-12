import numpy as np
from ersi.ersi_pipeline import benchmark_ersi

# Generate some fake ECG signals (e.g. 5 healthy, 5 brugada), shape (N_samples,)
fs = 1000
duration = 10  # 10 seconds

healthy = [np.random.randn(fs * duration) for _ in range(5)]
brugada = [np.random.randn(fs * duration) * 1.5 + 0.2 for _ in range(5)]

print("Running test benchmark...")
benchmark_ersi(healthy, brugada, fs=fs)
