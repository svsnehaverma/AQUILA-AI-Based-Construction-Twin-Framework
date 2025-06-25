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
import os

os.environ["PYTHONHASHSEED"] = "42"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

db_path = "/home/campus.ncl.ac.uk/nsv53/Sneha/Telemetry_data_IOT_Plug_and_play/Aqila_App_robotic_arm/Komastsu Testing/Pixel1_second_round_20Hz-recovered.db"
start_time = pd.to_datetime("2025-05-08 14:26:00 +0100", utc=True)
end_time = pd.to_datetime("2025-05-08 14:28:00 +0100", utc=True)
my_tz = pytz.timezone('Europe/London')

def acquire_data(db_path: str, start_time: Timestamp, end_time: Timestamp) -> DataFrame: 
    conn: sqlite3.Connection = sqlite3.connect(db_path)
    accel = pd.read_sql(f"""
        SELECT timestamp, worldXms2, worldYms2, worldZms2 
        FROM accelerometer 
        WHERE timestamp BETWEEN {start_time.timestamp()} AND {end_time.timestamp()};
    """, conn)
    conn.close()

    accel['timestamp'] = pd.to_datetime(accel['timestamp'].astype('int64'), unit='ms')
    downsample = accel.set_index('timestamp').resample('100ms').mean().interpolate().reset_index()
    downsample['smv'] = np.sqrt(downsample['worldXms2']**2 + downsample['worldYms2']**2 + downsample['worldZms2']**2)
    return downsample

def apply_bandpass(data, lowcut=0.0005, highcut=4.5, fs=10.0, order=4):
    def butter_bandpass(lowcut, highcut, fs, order=6):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype='band')
    
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data)

def extract_features(df, window_size=300, step_size=5):
    features = []
    timestamps = []
    for start in range(0, len(df) - window_size, step_size):
        window = df['smv_filtered'].iloc[start:start + window_size]
        ts = df['timestamp'].iloc[start + window_size // 2]
        feat = [
            window.mean(), window.std(), np.sum(window**2), 
            np.median(window), window.max(), window.min(), np.ptp(window),
        ]
        features.append(feat)
        timestamps.append(ts)
    return pd.DataFrame(features, columns=['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp'], index=timestamps)

def apply_smoothing(results):
    def smooth_series(series, window=1):
        return series.rolling(window=window, min_periods=1).apply(lambda x: x.mode().iloc[0])
    results['cluster_smooth'] = smooth_series(results['cluster'])
    results['cluster_smooth'] = results['cluster_smooth'].round().astype(int)
    results['state_smooth'] = results['cluster_smooth'].map(label_map)

def print_diagnostic(results):
    print("\n--- Cluster-wise Feature Medians (Real Scale) ---")
    print(results.groupby('cluster')[['mean', 'std', 'energy', 'ptp']].median())
    
    print("\n--- Cluster to State Mapping ---")
    for i, state in label_map.items():
        print(f"Cluster {i} → '{state}'")
    
    print("\n--- Frequency of Smoothed States ---")
    print(results['state_smooth'].value_counts())

    print("\n--- Final Voted State Frequencies ---")
    print(results['final_state'].value_counts())

    # Plot final_state
    state_colors = {'off': 'blue', 'idle': 'orange', 'working': 'green'}

    plt.figure(figsize=(14, 6))
    for state in ['off', 'idle', 'working']:
        subset = results[results['final_state'] == state]
        plt.scatter(subset.index, subset['energy'], label=state, s=20, alpha=0.7, color=state_colors[state])

    plt.title("Final Voted Activity States")
    plt.xlabel("Time")
    plt.ylabel("SMV Energy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    ax = plt.subplot()
    ax.xaxis.set_major_formatter(dates.DateFormatter('%H:%M:%S', my_tz))
    ax.xaxis.set_minor_locator(dates.MinuteLocator())
    plt.show()


def rule_based_classification(row):
    votes = []
    if row['std'] < 0.02:
        votes.append('off')
    elif row['std'] < 0.25:
        votes.append('idle')
    else:
        votes.append('working')

    if row['energy'] < 0.1:
        votes.append('off')
    elif row['energy'] < 10:
        votes.append('idle')
    else:
        votes.append('working')

    if row['ptp'] < 0.1:
        votes.append('off')
    elif row['ptp'] < 4:
        votes.append('idle')
    else:
        votes.append('working')

    return max(set(votes), key=votes.count)

def final_decision(row):
    if row['state_smooth'] == row['state_rule_based']:
        return row['state_smooth']
    elif row['state_smooth'] == 'off' and row['state_rule_based'] != 'off':
        return row['state_rule_based']
    else:
        return row['state_smooth']

def save_to_csv(results):
    results.to_csv("features_predicted_by_model_testing.csv")
    results[['state_smooth', 'state_rule_based', 'final_state']].to_csv("states_predicted_final.csv")

# ===================== MAIN WORKFLOW ======================

# 1. Load and preprocess data
accel_10hz = acquire_data(db_path, start_time, end_time)
accel_10hz['smv_filtered'] = apply_bandpass(accel_10hz['smv'], fs=10.0)

# 2. Extract features
features_df = extract_features(accel_10hz, window_size=300, step_size=5)

# 3. Load models
scaler: StandardScaler = joblib.load("./scaler_baseline.joblib")
kmeans: KMeans = joblib.load("./kmeans_baseline.joblib")
label_map: dict[int, str] = joblib.load("./cluster_to_state_map_baseline.joblib")

# 4. Predict clusters and map to state
X_scaled = scaler.transform(features_df[['mean', 'std', 'energy', 'median', 'max', 'min', 'ptp']])
features_df['cluster'] = kmeans.predict(X_scaled)
features_df['state'] = features_df['cluster'].map(label_map)
apply_smoothing(features_df)

# 5. Apply rule-based logic
features_df['state_rule_based'] = features_df.apply(rule_based_classification, axis=1)

# 6. Combine both for final decision
features_df['final_state'] = features_df.apply(final_decision, axis=1)

# 7. Print diagnostics
print_diagnostic(features_df)

print("\n--- Agreement Between Rule-Based and Clustering ---")
features_df['agreement'] = features_df['state_smooth'] == features_df['state_rule_based']
print(features_df['agreement'].value_counts())

print("\n--- Final State Distribution ---")
print(features_df['final_state'].value_counts())

# 8. Save results
save_to_csv(features_df)

