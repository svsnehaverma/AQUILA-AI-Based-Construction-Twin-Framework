import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import random

# Set seeds for reproducibility
os.environ["PYTHONHASHSEED"] = "42"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -----------------------------
# 1. Load and Filter Data
# -----------------------------
db_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/Telemetry_data_IOT_Plug_and_play/Aqila_App_robotic_arm/Komastsu Testing/Pixel1_first_round_10Hz.db"
conn = sqlite3.connect(db_path)
accel = pd.read_sql("SELECT timestamp, worldXms2, worldYms2, worldZms2 FROM accelerometer;", conn)
conn.close()

accel['timestamp'] = pd.to_datetime(accel['timestamp'].astype('int64'), unit='ms')
start_time = pd.to_datetime("2025-05-08 10:52:00")
end_time = pd.to_datetime("2025-05-08 11:30:00")
accel = accel[(accel['timestamp'] >= start_time) & (accel['timestamp'] <= end_time)].copy()

# -----------------------------
# 2. Compute Signal Magnitude Vector (SMV)
# -----------------------------
accel['smv'] = np.sqrt(accel['worldXms2']**2 + accel['worldYms2']**2 + accel['worldZms2']**2)

# -----------------------------
# 3. Bandpass Filtering (fs = 10 Hz)
# -----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass(data, lowcut=0.1, highcut=2.5, fs=10.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

accel['smv_filtered'] = apply_bandpass(accel['smv'], fs=10.0)

# -----------------------------
# 4. Sliding Window Feature Extraction
# -----------------------------
def extract_features(df, window_size=600, step_size=5):
    features = []
    timestamps = []
    for start in range(0, len(df) - window_size, step_size):
        window = df['smv_filtered'].iloc[start:start + window_size]
        ts = df['timestamp'].iloc[start + window_size // 2]
        feat = [
            window.mean(),
            window.std(),
            np.sum(window**2),  # energy
            np.median(window),
            window.max(),
            window.min(),
            np.ptp(window),     # peak-to-peak
        ]
        features.append(feat)
        timestamps.append(ts)
    return pd.DataFrame(features, columns=['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp'], index=timestamps)

features_df = extract_features(accel)

# -----------------------------
# 5. Clustering (KMeans)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df)

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)
features_df['cluster'] = labels

# -----------------------------
# 6. Auto-label clusters -> states
# -----------------------------
cluster_stats = features_df.groupby('cluster')['std'].mean().sort_values()
sorted_clusters = cluster_stats.index.tolist()
label_map = {
    sorted_clusters[0]: 'off',
    sorted_clusters[1]: 'idle',
    sorted_clusters[2]: 'working'
}
features_df['state'] = features_df['cluster'].map(label_map)
#print(features_df.groupby('cluster')[['mean', 'std', 'ptp', 'energy']].median())
# -----------------------------
# 7. Voting-Based State Refinement
# -----------------------------
def vote_based_state(row):
    votes = []
    # Rule 1: STD-based
    if row['std'] > 0.4:
        votes.append('working')
    elif row['std'] < 0.015:
        votes.append('off')
    else:
        votes.append('idle')
    # Rule 2: Energy-based
    if row['energy'] > 900:
        votes.append('working')
    elif row['energy'] < 40:
        votes.append('idle')
    else:
        votes.append('off')
    # Rule 3: Cluster prediction
    votes.append(row['state'])
    return max(set(votes), key=votes.count)

features_df['voted_state'] = features_df.apply(vote_based_state, axis=1)

# -----------------------------
# 7. Rolling Energy and STD (Better Watermarks)
# -----------------------------
features_df['rolling_energy_max'] = features_df['energy'].rolling(window=10, min_periods=1).max()
features_df['rolling_energy_min'] = features_df['energy'].rolling(window=10, min_periods=1).min()
features_df['rolling_std_max'] = features_df['std'].rolling(window=10, min_periods=1).max()
features_df['rolling_std_min'] = features_df['std'].rolling(window=10, min_periods=1).min()

# -----------------------------
# 8. Smooth cluster labels
# -----------------------------
def smooth_series(series, window=3):
    return series.rolling(window=window, min_periods=1).apply(lambda x: x.mode().iloc[0])

features_df['cluster_smooth'] = smooth_series(features_df['cluster'], window=5)
features_df['cluster_smooth'] = features_df['cluster_smooth'].round().astype(int)
features_df['state_smooth'] = features_df['cluster_smooth'].map(label_map)

# -----------------------------
# Visualization: Voted States with Rolling Energy
# -----------------------------
import matplotlib.pyplot as plt

state_colors = {'off': 'blue', 'idle': 'red', 'working': 'green'}

plt.figure(figsize=(14, 6))

for state in ['off', 'idle', 'working']:
    subset = features_df[features_df['voted_state'] == state]
    plt.scatter(subset.index, subset['energy'], label=state, s=20, alpha=0.7, color=state_colors[state])

plt.plot(features_df.index, features_df['rolling_energy_max'], linestyle='--', label='Rolling Energy Max', color='black')
plt.plot(features_df.index, features_df['rolling_energy_min'], linestyle='--', label='Rolling Energy Min', color='gray')

plt.title('State Segmentation via Voting with Energy Markers')
plt.xlabel('Time')
plt.ylabel('SMV Energy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''


###############################################################################
# -----------------------------
# 10. Save Outputs
# -----------------------------
features_df.to_csv("features_with_improved_states.csv")
features_df[['state_smooth']].to_csv("detected_states_smoothed.csv")

# Save calibrated model, scaler, and cluster-to-state mapping
import joblib

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_df[['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp']])
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

joblib.dump(scaler, 'scaler_baseline.joblib')
joblib.dump(kmeans, 'kmeans_baseline.joblib')

# Save cluster-to-state mapping
cluster_stats = features_df.groupby('cluster')['std'].mean().sort_values()
sorted_clusters = cluster_stats.index.tolist()
label_map = {
    sorted_clusters[0]: 'off',
    sorted_clusters[1]: 'idle',
    sorted_clusters[2]: 'working'
}
joblib.dump(label_map, 'cluster_to_state_map_baseline.joblib')

# -----------------------------
# Interpretation Aids for State Identification
# -----------------------------

print("\n--- Cluster-wise Feature Medians (Real Scale) ---")
print(features_df.groupby('cluster')[['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp']].median())

print("\n--- Cluster Centroids (Standardized Feature Space) ---")
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp'])
print(centroids)

print("\n--- Cluster to State Mapping (based on ascending STD) ---")
for i, state in label_map.items():
    print(f"Cluster {i} → '{state}'")

print("\n--- Representative Samples from Each State (Smoothed) ---")
for state in ['off', 'idle', 'working']:
    samples = features_df[features_df['state_smooth'] == state][['mean', 'std', 'energy', 'ptp']].head(3)
    print(f"\nState: {state.upper()}")
    print(samples)
'''
