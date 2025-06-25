import sqlite3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, dates
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pandas import DataFrame, Timestamp
import joblib
import pytz
import random
import numpy as np
import os
os.environ["PYTHONHASHSEED"] = "42"

import random
import numpy as np
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

db_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/Telemetry_data_IOT_Plug_and_play/Aqila_App_robotic_arm/Komastsu Testing/Pixel1_second_round_20Hz-recovered.db"
# start_time = pd.to_datetime("2025-05-08 13:11:00 +0000", utc=True)
# end_time = pd.to_datetime("2025-05-08 13:25:00 +0000", utc=True)
start_time = pd.to_datetime("2025-05-08 14:26:00 +0100", utc=True)
end_time = pd.to_datetime("2025-05-08 14:32:00 +0100", utc=True)
my_tz = pytz.timezone('Europe/London')

#################### Pulling Pixel at second Round at 20Hz (making sure it down sampled at 10Hz) #######################   
# -----------------------------
# 1. Load 20 Hz Data
# -----------------------------
def acquire_data(db_path: str, start_time: Timestamp, end_time: Timestamp) -> DataFrame: 
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    accel = pd.read_sql(f"SELECT timestamp, worldXms2, worldYms2, worldZms2 FROM accelerometer where timestamp BETWEEN {start_time.timestamp()} AND {end_time.timestamp()};", conn)
    conn.close()

    accel['timestamp'] = pd.to_datetime(accel['timestamp'].astype('int64'), unit='ms')

    # -----------------------------
    # 2. Downsample to 10 Hz
    # -----------------------------
    downsample = accel.set_index('timestamp').resample('100ms').mean().interpolate().reset_index()

    # -----------------------------
    # 3. Compute SMV - Magnitude
    # -----------------------------
    downsample['smv'] = np.sqrt(downsample['worldXms2']**2 +
                                downsample['worldYms2']**2 +
                                downsample['worldZms2']**2)
    
    return downsample
# -------------------------------------
# 4. Bandpass Filter (10 Hz)
# -------------------------------------

def apply_bandpass(data, lowcut=0.0005, highcut=4.5, fs=10.0, order=4):
    def butter_bandpass(lowcut, highcut, fs, order=6):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype='band')
    
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)


# -------------------------------------
# 5. Feature Extraction (10Hz)
# -------------------------------------
def extract_features(df, window_size=300, step_size=5):
    features = []
    timestamps = []
    for start in range(0, len(df) - window_size, step_size):
        window = df['smv_filtered'].iloc[start:start + window_size]
        ts = df['timestamp'].iloc[start + window_size // 2]
        feat = [
            window.mean(),
            window.std(),
            np.sum(window**2),
            np.median(window),
            window.max(),
            window.min(),
            np.ptp(window),
        ]
        features.append(feat)
        timestamps.append(ts)
    return pd.DataFrame(features, columns=['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp'], index=timestamps)


def print_diagnostic(results):
    print(results.groupby('cluster')[['mean', 'std', 'ptp', 'energy']].median())
    
    # -----------------------------
    # 11. Inspect Clustering Decisions
    # -----------------------------
    '''
    # Features used for clustering (first 5 samples)
    print("\n--- Sample Feature Vectors Used for Clustering ---")
    print(results[['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp']].head())

    # Cluster centroids (interpreted in standardized feature space)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp'])
    print("\n--- Cluster Centroids (Standardized Feature Space) ---")
    print(centroids)
    '''
    # Median values per cluster (real scale, helps interpret actual separation)
    print("\n--- Cluster-wise Feature Medians (Real Scale) ---")
    print(results.groupby('cluster')[['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp']].median())

    # Cluster-to-state mapping logic (based on std)
    print("\n--- Cluster to State Mapping ---")
    for i, state in label_map.items():
        print(f"Cluster {i} → '{state}'")

    # Frequency of states (smoothed)
    print("\n--- Frequency of Smoothed States ---")
    print(results['state_smooth'].value_counts())
    '''
    # First few timestamps and state predictions for each state
    for state in ['off', 'idle', 'working']:
        print(f"\n--- First Entries for State: {state.upper()} ---")
        print(results[results['state_smooth'] == state][['mean', 'std', 'energy', 'state_smooth']].head())

    '''
    # --------------------------------------
    # 9. Visualization
    # --------------------------------------
    state_colors = {'off': 'blue', 'idle': 'red', 'working': 'green'}

    plt.figure(figsize=(14, 6))
    for state in ['off', 'idle', 'working']:
        subset = features_df[features_df['state_smooth'] == state]
        plt.scatter(subset.index, subset['energy'], label=state, s=20, alpha=0.7, color=state_colors[state])

    plt.title("Predicted Activity States Using Pretrained Model")
    plt.xlabel("Time")
    plt.ylabel("SMV Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    ax= plt.subplot()
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S', my_tz))
    ax.xaxis.set_minor_locator(dates.MinuteLocator())
    plt.show()

def save_to_csv(results):
    results.to_csv("features_predicted_by_model_testing.csv")
    results[['state_smooth']].to_csv("states_predicted_by_model.csv")

# ---------------------------------------
# 8. Smooth cluster labels
# ---------------------------------------

def apply_smoothing(results):
    def smooth_series(series, window=1):
        return series.rolling(window=window, min_periods=1).apply(lambda x: x.mode().iloc[0])
    results['cluster_smooth'] = smooth_series(results['cluster'])
    results['cluster_smooth'] = results['cluster_smooth'].round().astype(int)
    results['state_smooth'] = results['cluster_smooth'].map(label_map)



## ======================================================================
# Main logic
## ======================================================================

## Load data from DB etc

accel_10hz = acquire_data(db_path, start_time, end_time)

# Apply Bandpass filter
accel_10hz['smv_filtered'] = apply_bandpass(accel_10hz['smv'], fs=10.0)

# Derive features (mean, energy, std etc)
features_df = extract_features(accel_10hz, window_size=300, step_size=5)

# -------------------------------------
# 5. Clustering (KMeans)
# 6. Load Pretrained Scaler and KMeans
# -------------------------------------
scaler: StandardScaler = joblib.load("./scaler_baseline.joblib")
kmeans: KMeans = joblib.load("./kmeans_baseline.joblib")
label_map: dict[int, str] = joblib.load("./cluster_to_state_map_baseline.joblib")

# -------------------------------------
# 7. Predict Clusters and Map States
# -------------------------------------
X_scaled = scaler.transform(features_df[['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp']])
features_df['cluster'] = kmeans.predict(X_scaled)
features_df['state'] = features_df['cluster'].map(label_map)
apply_smoothing(features_df)

print_diagnostic(features_df)

# -----------------------------
# 10. Save Results
# -----------------------------
save_to_csv(features_df)




