"""
feature_engineering.py
=======================
What this file does:
    Takes raw sensor data and creates NEW columns that are smarter and
    more useful for the AI model. This is called "feature engineering."

Real-world analogy:
    Raw data: "temperature was 620K at 2pm"
    Engineered feature: "temperature was 18K ABOVE the rolling average
                         in the 30 minutes before the stop"

    The second version is MUCH more useful for detecting failures.

The 5 types of features we create:
    1. Rolling statistics   → "what was the average over the last 10 readings?"
    2. Rate of change       → "how fast is this sensor CHANGING?"
    3. Deviation from mean  → "how far is this from normal?"
    4. Time-based features  → "is this a night shift? end of week?"
    5. Interaction features → "temp × pressure combined"
"""

import pandas as pd
import numpy as np


def add_rolling_features(df, sensors, windows=[5, 10, 30]):
    """
    Rolling mean and std for each sensor.

    What is a rolling mean?
        Imagine you have readings: [100, 102, 98, 110, 108, 112, 115]
        Rolling mean with window=3 gives: [_, _, 100, 103.3, 105.3, 110, 111.7]
        Each value = average of the last 3 readings.

        This smooths out noise and reveals the TREND.

    Why rolling STD?
        Standard deviation = how "spread out" recent readings are.
        High STD before a stop = erratic, unstable sensor = bad sign.
    """
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        for window in windows:
            # Group by machine so we don't mix different machines' history
            grp = df.groupby("machine_id")[sensor]
            
            col_mean = f"{sensor}_roll_mean_{window}"
            col_std  = f"{sensor}_roll_std_{window}"
            
            df[col_mean] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[col_std] = grp.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
    return df


def add_rate_of_change(df, sensors):
    """
    How fast is each sensor changing?

    Formula: diff = current_reading - previous_reading

    Why this matters:
        A temperature jumping from 580K to 612K in ONE cycle = 32K spike
        That spike is a much stronger signal than the absolute value of 612K.

    In pandas: .diff() subtracts the previous row's value from the current one.
    """
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        df[f"{sensor}_roc"] = (
            df.groupby("machine_id")[sensor]
              .transform(lambda x: x.diff().fillna(0))
        )
    return df


def add_deviation_from_baseline(df, sensors):
    """
    How far is each reading from the machine's own average?

    Formula: deviation = (value - machine_mean) / machine_std

    This is called "Z-score normalization."

    Why it matters:
        Machine CTL-04 might normally run at 620K.
        Machine CNC-11 might normally run at 580K.
        A reading of 600K means different things for each!
        
        The Z-score tells you: "this is 1.5 standard deviations above normal"
        — which IS comparable across machines.
    """
    for sensor in sensors:
        if sensor not in df.columns:
            continue
        
        # Calculate mean and std PER MACHINE (not global)
        machine_stats = df.groupby("machine_id")[sensor].agg(["mean", "std"])
        machine_stats.columns = ["base_mean", "base_std"]
        
        df = df.join(machine_stats, on="machine_id", rsuffix="_stat")
        
        df[f"{sensor}_zscore"] = (
            (df[sensor] - df["base_mean"]) / df["base_std"].replace(0, 1)
        )
        df.drop(columns=["base_mean", "base_std"], inplace=True)
    
    return df


def add_time_features(df):
    """
    Extract time-based patterns from timestamps.

    Why this matters:
        Night shifts often have more downtime (fewer experienced operators)
        End-of-week machines might rush production (more micro-stops)
        Machines that stop every 3 days → time feature helps detect this
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    df["hour_of_day"]    = df["timestamp"].dt.hour
    df["day_of_week"]    = df["timestamp"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["is_weekend"]     = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night_shift"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)).astype(int)
    
    # Hours since last stop (per machine)
    # This helps detect recurring stop cycles
    df = df.sort_values(["machine_id", "timestamp"])
    
    def hours_since_last_stop(group):
        stop_times = group[group["is_stop"] == 1]["timestamp"]
        result = []
        last_stop = None
        for _, row in group.iterrows():
            if last_stop is None:
                result.append(np.nan)
            else:
                hours_diff = (row["timestamp"] - last_stop).total_seconds() / 3600
                result.append(round(hours_diff, 1))
            if row["is_stop"] == 1:
                last_stop = row["timestamp"]
        return pd.Series(result, index=group.index)
    
    df["hours_since_last_stop"] = (
        df.groupby("machine_id", group_keys=False)
          .apply(hours_since_last_stop)
    )
    df["hours_since_last_stop"].fillna(0, inplace=True)
    
    return df


def add_stop_pattern_features(df):
    """
    Aggregated stop history per machine.

    This creates features like:
        - How many stops in the last 7 days?
        - What's the average stop duration?
        - Is there a periodic pattern?
    
    These features help the ML model recognize "this machine is about
    to enter another stop cycle."
    """
    df = df.sort_values(["machine_id", "cycle"])
    
    # Cumulative stops per machine
    df["cumulative_stops"] = (
        df.groupby("machine_id")["is_stop"]
          .transform("cumsum")
    )
    
    # Rolling stop count (last 30 cycles = ~5 days)
    df["stops_last_30_cycles"] = (
        df.groupby("machine_id")["is_stop"]
          .transform(lambda x: x.rolling(30, min_periods=1).sum())
    )
    
    # Rolling downtime hours (last 30 cycles)
    df["downtime_last_30_cycles_hrs"] = (
        df.groupby("machine_id")["stop_duration_min"]
          .transform(lambda x: x.rolling(30, min_periods=1).sum()) / 60
    ).round(2)
    
    return df


def engineer_all_features(df):
    """
    Master function — runs ALL feature engineering steps in order.
    
    Call this on your raw data before feeding it to the ML model.
    """
    print("Starting feature engineering...")
    
    # Define which sensors to create features for
    # We pick the most informative ones (not all 21 — that would be too much noise)
    KEY_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s15"]
    
    print("  → Adding rolling mean / std features...")
    df = add_rolling_features(df, KEY_SENSORS, windows=[5, 10, 30])
    
    print("  → Adding rate-of-change features...")
    df = add_rate_of_change(df, KEY_SENSORS)
    
    print("  → Adding Z-score deviation features...")
    df = add_deviation_from_baseline(df, KEY_SENSORS)
    
    print("  → Adding time-based features...")
    df = add_time_features(df)
    
    print("  → Adding stop pattern features...")
    df = add_stop_pattern_features(df)
    
    print(f"  Done! Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.append("..")
    from data.generate_data import generate_all_data
    
    raw = generate_all_data()
    featured = engineer_all_features(raw)
    print("\nNew columns added:")
    original_cols = set(raw.columns)
    new_cols = [c for c in featured.columns if c not in original_cols]
    for c in new_cols[:20]:
        print(f"  {c}")
    print(f"  ... and {max(0, len(new_cols)-20)} more")
