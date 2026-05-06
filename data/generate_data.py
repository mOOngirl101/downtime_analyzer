"""
generate_data.py
================
What this file does:
    Generates synthetic machine sensor data that mimics the NASA CMAPSS
    turbofan dataset (FD001). Instead of jet engines, we label them as
    Cummins turbocharger machines so the project feels real.
 
Think of it like this:
    A real dataset would come from Cummins' SCADA system.
    We can't access that, so we *generate* realistic data ourselves.
    The patterns we bake in (overheating, bearing wear, etc.) are based
    on how real turbofan sensors behave before failure.
 
NASA CMAPSS dataset columns we borrow:
    - op_setting_1/2/3  → operating conditions (like load, speed, temp)
    - sensor_1 to sensor_21 → various sensor readings
    - cycle               → how many operating cycles have passed
    - RUL                 → Remaining Useful Life (cycles until failure)
"""
 
import numpy as np
import pandas as pd
import random
 
# ── Seed for reproducibility ──────────────────────────────────────────────────
# This means every time you run this, you get the EXACT same "random" data.
# Think of it as saving the random number generator's starting point.
np.random.seed(42)
random.seed(42)
 
# ── Machine definitions ───────────────────────────────────────────────────────
# Each machine has a "failure mode" — the pattern of how it degrades.
# In real life, you'd learn this from maintenance history.
MACHINES = {
    "CTL-04": {"line": "Balancing",          "failure_mode": "overheating",    "rul_start": 120},
    "CNC-11": {"line": "Precision Machining", "failure_mode": "bearing_wear",   "rul_start": 150},
    "ASM-02": {"line": "Assembly",            "failure_mode": "vibration",      "rul_start": 100},
    "CNC-03": {"line": "Turbine Machining",   "failure_mode": "overheating",    "rul_start": 180},
    "TST-07": {"line": "Hot Test Rig",        "failure_mode": "sensor_fault",   "rul_start": 200},
    "FRG-01": {"line": "Forging",             "failure_mode": "operator_stop",  "rul_start": 300},
}
 
# ── Sensor names (borrowed from NASA CMAPSS, relabeled for manufacturing) ────
SENSOR_NAMES = {
    "s1":  "T2_fan_inlet_temp",       # Fan inlet temperature (K)
    "s2":  "T24_lpc_outlet_temp",     # Low pressure compressor outlet temp (K)
    "s3":  "T30_hpc_inlet_temp",      # High pressure compressor inlet temp (K)
    "s4":  "T50_fan_exit_temp",       # Fan exit temperature (K)
    "s5":  "P2_total_pressure",       # Total inlet pressure (psia)
    "s6":  "P15_total_pressure",      # Total pressure at fan inlet (psia)
    "s7":  "P30_hpc_pressure",        # High pressure compressor pressure (psia)
    "s8":  "Nf_fan_speed",            # Fan rotational speed (rpm)
    "s9":  "Nc_core_speed",           # Core rotational speed (rpm)
    "s10": "epr_ratio",               # Engine pressure ratio
    "s11": "Ps30_static_pressure",    # Static pressure at HPC outlet (psia)
    "s12": "phi_fuel_ratio",          # Ratio of fuel flow to Ps30
    "s13": "NRf_fan_corrected_speed", # Fan corrected speed (rpm)
    "s14": "NRc_core_corrected_speed",# Core corrected speed (rpm)
    "s15": "BPR_bypass_ratio",        # Bypass ratio
    "s16": "farB_burner_fuel_ratio",  # Burner fuel-air ratio
    "s17": "htBleed_bleed_enthalpy",  # Bleed enthalpy
    "s18": "Nf_demand_fan_speed",     # Demanded fan speed (rpm)
    "s19": "PCNfR_demand",            # Demanded corrected fan speed
    "s20": "W31_hpt_coolant",         # HPT coolant bleed (lbm/s)
    "s21": "W32_lpt_coolant",         # LPT coolant bleed (lbm/s)
}
 
def add_degradation_signal(df, failure_mode, rul_col="RUL"):
    """
    This is the most important function in the file.
    
    It ADDS degradation signals to the sensor data.
    
    How it works:
        - 'health' goes from 1.0 (healthy) → 0.0 (failed)
        - health = RUL / max_RUL
        - Each failure mode affects DIFFERENT sensors
        
    Real-world analogy:
        Bearing wear → vibration sensor goes UP
        Overheating  → temperature sensor goes UP
        Sensor fault → random dropouts appear
    """
    max_rul = df[rul_col].max()
    
    # health goes from 1 (new) to 0 (failed)
    health = df[rul_col] / max_rul
    noise = lambda scale: np.random.normal(0, scale, len(df))
 
    if failure_mode == "overheating":
        # Temperature sensors rise as machine degrades
        # The (1 - health) part means: when health=1, degradation=0. When health=0, degradation=1.
        df["s4"] += (1 - health) * 35 + noise(2)   # T50 fan exit temp rises
        df["s3"] += (1 - health) * 15 + noise(1.5) # HPC inlet rises
        df["s7"] += (1 - health) * 8  + noise(0.5) # Pressure also rises slightly
        
    elif failure_mode == "bearing_wear":
        # Vibration and speed drift
        df["s8"] -= (1 - health) * 80 + noise(5)   # Fan speed drops (bearing friction)
        df["s11"] -= (1 - health) * 5 + noise(0.3) # Pressure drop
        # Vibration spike — this is a NEW synthetic sensor we add
        df["vibration_rms"] = 2.0 + (1 - health) * 3.5 + noise(0.2)
        
    elif failure_mode == "vibration":
        # Axial vibration + shaft runout
        df["vibration_axial"] = 3.0 + (1 - health) * 4.0 + noise(0.3)
        df["shaft_runout_um"]  = 10  + (1 - health) * 12  + noise(0.8)
        df["s8"] -= (1 - health) * 40 + noise(3)
 
    elif failure_mode == "sensor_fault":
        # Sensor dropout — random NaN values appearing more frequently as health drops
        # Think of it as: bad wiring causes intermittent signal loss
        dropout_prob = (1 - health) * 0.15  # max 15% chance of dropout per reading
        mask = np.random.random(len(df)) < dropout_prob
        df.loc[mask, "s5"] = np.nan   # P2 sensor drops out
        df["s15"] -= (1 - health) * 1.5 + noise(0.1)  # Bypass ratio drifts
 
    elif failure_mode == "operator_stop":
        # Micro-stops — not sensor degradation, but frequent short stops
        # We simulate this with random "stop" events in a separate column
        df["operator_micro_stop"] = (np.random.random(len(df)) < 0.08 * (1 - health * 0.5)).astype(int)
 
    return df
 
 
def generate_machine_data(machine_id, config, n_days=90):
    """
    Generates a full time-series dataset for one machine over n_days.
    
    Steps:
    1. Create a timestamp series (every 4 hours = 6 readings per day)
    2. Create sensor readings with realistic base values + random noise
    3. Calculate RUL (counts down from max to 0)
    4. Add degradation signal based on failure mode
    5. Add stop events (when did the machine actually stop?)
    """
    # Each cycle = 4 hours. 6 cycles per day.
    n_cycles = n_days * 6
    
    # ── Base sensor values (from NASA CMAPSS FD001 mean values) ──────────────
    # These are the "normal" readings when the machine is healthy.
    base_sensors = {
        "s1": 518.67, "s2": 642.68, "s3": 1583.9, "s4": 1400.6,
        "s5": 14.62,  "s6": 21.61,  "s7": 554.36,  "s8": 2388.1,
        "s9": 9046.2, "s10": 1.3,   "s11": 47.47,  "s12": 521.66,
        "s13": 2388.1,"s14": 8138.6, "s15": 8.4195, "s16": 0.03,
        "s17": 392.0, "s18": 2388.0, "s19": 100.0,  "s20": 38.81,
        "s21": 23.42,
    }
    
    # Small random noise on each sensor (simulates real-world measurement variation)
    sensor_noise = {k: v * 0.01 for k, v in base_sensors.items()}  # 1% noise
 
    rows = []
    for cycle in range(1, n_cycles + 1):
        row = {"machine_id": machine_id, "line": config["line"], "cycle": cycle}
        
        # Timestamps — start from 90 days ago
        row["timestamp"] = pd.Timestamp.now() - pd.Timedelta(hours=(n_cycles - cycle) * 4)
        
        # Operating conditions (3 discrete settings in the real dataset)
        row["op_setting_1"] = round(random.choice([-0.0087, 0.0218, 0.0, 0.0005]), 4)
        row["op_setting_2"] = round(random.choice([0.0, 0.0003, -0.0003]), 4)
        row["op_setting_3"] = random.choice([60, 80, 100])  # % of rated load
 
        # Add sensor readings with noise
        for s, base in base_sensors.items():
            noise_scale = sensor_noise[s]
            row[s] = round(base + np.random.normal(0, noise_scale * 10), 3)
        
        # RUL counts down: machine starts at max and heads toward 0
        rul_start = config["rul_start"]
        row["RUL"] = max(0, rul_start - int(cycle * rul_start / n_cycles))
        
        # Shift (which shift is this reading from?)
        hour_of_day = (cycle * 4) % 24
        if 6 <= hour_of_day < 14:
            row["shift"] = "Day"
        elif 14 <= hour_of_day < 22:
            row["shift"] = "Evening"
        else:
            row["shift"] = "Night"
            
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Apply degradation patterns
    df = add_degradation_signal(df, config["failure_mode"])
    
    # Add stop events based on degradation
    # A "stop" happens when sensors hit threshold OR randomly near end of life
    df["is_stop"] = 0
    threshold_cycle = int(n_cycles * 0.6)  # degradation starts causing issues at 60% life used
    for idx in df[df["cycle"] > threshold_cycle].index:
        # More likely to stop as RUL drops
        prob = (1 - df.loc[idx, "RUL"] / config["rul_start"]) * 0.12
        if np.random.random() < prob:
            df.loc[idx, "is_stop"] = 1
    
    # Stop duration in minutes (short = micro-stop, long = breakdown)
    df["stop_duration_min"] = df["is_stop"] * np.random.choice([5, 15, 45, 120, 240], len(df))
    
    # Time since last maintenance (resets every ~300 cycles)
    df["cycles_since_maintenance"] = df["cycle"] % 300
    
    return df
 
 
def generate_all_data():
    """
    Master function — generates data for ALL machines and saves to CSV.
    """
    all_dfs = []
    for machine_id, config in MACHINES.items():
        print(f"Generating data for {machine_id} ({config['line']})...")
        df = generate_machine_data(machine_id, config, n_days=90)
        all_dfs.append(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    # Save CSV next to this file (works no matter where you run from)
    import os
    save_path = os.path.join(os.path.dirname(__file__), "machine_logs.csv")
    full_df.to_csv(save_path, index=False)
    print(f"\nSaved {len(full_df):,} rows to {save_path}")
    print(f"Machines: {full_df['machine_id'].nunique()}")
    print(f"Date range: {full_df['timestamp'].min()} → {full_df['timestamp'].max()}")
    return full_df
 
 
if __name__ == "__main__":
    df = generate_all_data()
    print("\nSample:")
    print(df[["machine_id", "cycle", "RUL", "s4", "s8", "is_stop"]].head(10))