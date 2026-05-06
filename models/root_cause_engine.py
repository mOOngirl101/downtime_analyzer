"""
root_cause_engine.py
====================
The BRAIN of the project. This is where the actual diagnosis happens.

Three layers working together:
    1. Rule-Based Engine   → Fast, always works, human-readable logic
    2. ML Classifier       → Learns from labeled data (if you have it)
    3. Pattern Detector    → Finds recurring cycles using autocorrelation

How they combine:
    Rule engine gives an initial diagnosis.
    ML classifier gives a probability-weighted diagnosis.
    Pattern detector finds the "every 71 hours" type insights.
    
    The final output merges all three into a plain-English explanation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict


# =============================================================================
# LAYER 1: RULE-BASED ENGINE
# =============================================================================

class RuleBasedEngine:
    """
    The simplest and most reliable layer.
    
    Think of it like a doctor's checklist:
        "If temperature is high AND pressure is rising → overheating"
        "If vibration is high → bearing wear"
        "If stop was short AND happened 3+ times → micro-stop"
    
    Rules are written by domain experts (maintenance engineers).
    They're not learned from data — they're hand-coded knowledge.
    """
    
    # Sensor thresholds — beyond these values, we flag an issue.
    # These are based on NASA CMAPSS degraded sensor patterns.
    THRESHOLDS = {
        "s4_zscore":          1.8,    # T50 fan exit temp (Z-score > 1.8 = anomaly)
        "s3_zscore":          1.5,    # HPC inlet temp
        "s8_roc":            -30.0,   # Fan speed rate-of-change (dropping fast)
        "vibration_rms":      4.0,    # Vibration (mm/s) above safe limit
        "vibration_axial":    5.0,    # Axial vibration threshold
        "s5_missing_pct":     0.08,   # If >8% of P2 readings are NaN → sensor fault
        "micro_stop_count":   3,      # 3+ stops under 10 min in a row → operator issue
        "temp_spike_30min":   12.0,   # K rise in 30 minutes before stop
    }
    
    def diagnose(self, machine_df):
        """
        Takes a machine's recent data and returns probable root causes.
        
        Input:  DataFrame of recent sensor readings for ONE machine
        Output: List of causes with confidence percentages
        
        Each cause = {name, confidence, evidence, recommendation}
        """
        causes = []
        recent = machine_df.tail(30)  # Look at last 30 cycles (~5 days)
        
        # ── Rule 1: Overheating ─────────────────────────────────────────────
        # Temperature Z-score tells us how far above "normal" the temp is.
        if "s4_zscore" in recent.columns:
            temp_z = recent["s4_zscore"].mean()
            if temp_z > self.THRESHOLDS["s4_zscore"]:
                confidence = min(95, int(40 + temp_z * 20))
                causes.append({
                    "cause": "Overheating",
                    "confidence": confidence,
                    "evidence": f"T50 temp {temp_z:.1f}σ above baseline (avg last 5 days)",
                    "recommendation": "Check coolant flow to main bearing. Inspect oil pump for partial blockage.",
                    "priority": "immediate" if temp_z > 2.5 else "within_48h",
                    "rule": "temp_zscore_threshold",
                })
        
        # ── Rule 2: Bearing Wear ────────────────────────────────────────────
        # Fan speed dropping + vibration rising = bearing friction
        if "vibration_rms" in recent.columns:
            vib = recent["vibration_rms"].tail(10).mean()
            speed_roc = recent.get("s8_roc", pd.Series([0])).mean()
            
            if vib > self.THRESHOLDS["vibration_rms"]:
                confidence = min(92, int(35 + vib * 12 + abs(min(0, speed_roc)) * 0.5))
                causes.append({
                    "cause": "Bearing wear / imbalance",
                    "confidence": confidence,
                    "evidence": f"Vibration RMS at {vib:.1f} mm/s (ISO limit: 4.0 mm/s). Fan speed drifting {speed_roc:.0f} rpm/cycle.",
                    "recommendation": "Inspect inner bearing races. Schedule replacement within 48h.",
                    "priority": "within_48h",
                    "rule": "vibration_threshold",
                })
        
        # ── Rule 3: Axial Vibration / Imbalance ────────────────────────────
        if "vibration_axial" in recent.columns:
            axial = recent["vibration_axial"].tail(10).mean()
            if axial > self.THRESHOLDS["vibration_axial"]:
                causes.append({
                    "cause": "Vibration anomaly / imbalance",
                    "confidence": min(88, int(30 + axial * 9)),
                    "evidence": f"Axial vibration at {axial:.1f} mm/s — 2× above safe threshold.",
                    "recommendation": "Dynamic balance check. Verify shaft runout within 20 µm tolerance.",
                    "priority": "immediate",
                    "rule": "axial_vibration_threshold",
                })
        
        # ── Rule 4: Sensor Fault ────────────────────────────────────────────
        # Count what % of recent P2 sensor readings are missing (NaN)
        if "s5" in recent.columns:
            missing_pct = recent["s5"].isna().mean()
            if missing_pct > self.THRESHOLDS["s5_missing_pct"]:
                causes.append({
                    "cause": "Sensor fault / dropout",
                    "confidence": min(85, int(missing_pct * 600)),
                    "evidence": f"P2 pressure sensor missing {missing_pct*100:.0f}% of readings (normal: <1%).",
                    "recommendation": "Replace P2 total pressure sensor. Recalibrate BPR measurement system.",
                    "priority": "within_1_week",
                    "rule": "sensor_dropout_threshold",
                })
        
        # ── Rule 5: Operator Micro-Stop Pattern ─────────────────────────────
        # Many very short stops = operator intervention / process issue
        if "stop_duration_min" in recent.columns and "is_stop" in recent.columns:
            micro_stops = ((recent["is_stop"] == 1) & (recent["stop_duration_min"] < 10)).sum()
            if micro_stops >= self.THRESHOLDS["micro_stop_count"]:
                causes.append({
                    "cause": "Operator intervention / micro-stop pattern",
                    "confidence": min(75, int(40 + micro_stops * 8)),
                    "evidence": f"{micro_stops} stops under 10 minutes in last 30 cycles.",
                    "recommendation": "Review operator procedure adherence. Check if material feed jams are recurring.",
                    "priority": "within_1_week",
                    "rule": "micro_stop_count",
                })
        
        # Sort by confidence descending
        causes.sort(key=lambda x: x["confidence"], reverse=True)
        
        return causes if causes else [{
            "cause": "No clear pattern detected",
            "confidence": 0,
            "evidence": "Sensor readings within normal range. Monitor closely.",
            "recommendation": "Continue scheduled maintenance. No immediate action required.",
            "priority": "monitor",
            "rule": "no_pattern",
        }]


# =============================================================================
# LAYER 2: ML CLASSIFIER  
# =============================================================================

class MLClassifier:
    """
    Learns to classify failure causes from historical labeled data.
    
    "Labeled data" means: past records where we KNOW what caused the stop.
    (e.g., maintenance log says "bearing replaced" → we label those stops "bearing_wear")
    
    Algorithm: Random Forest
        - Trains hundreds of "decision trees" on the data
        - Each tree votes on the most likely cause
        - Final answer = majority vote
    
    Why Random Forest?
        → Works on small datasets (even 200-300 records)
        → Doesn't need data scaling/normalization
        → You can ask WHY it made a decision (feature importance)
        → Hard to overfit by accident
    
    If you don't have labeled data yet:
        → Use the RuleBasedEngine alone
        → Once you collect ~100 labeled stops, train this
    """
    
    FEATURE_COLS = [
        "s4_zscore", "s3_zscore", "s8_roc", "s7_zscore",
        "s4_roll_mean_10", "s8_roll_mean_10",
        "stops_last_30_cycles", "cycles_since_maintenance",
        "is_night_shift", "hour_of_day",
        "hours_since_last_stop", "cumulative_stops",
    ]
    
    CAUSE_LABELS = ["overheating", "bearing_wear", "vibration", "sensor_fault", "operator_stop"]
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def _prepare_features(self, df):
        """
        Select and clean feature columns.
        Missing columns get filled with 0.
        """
        available = [c for c in self.FEATURE_COLS if c in df.columns]
        X = df[available].fillna(0)
        return X
    
    def train(self, df, label_col="failure_mode"):
        """
        Train the classifier on labeled stop events.
        
        Parameters:
            df         : DataFrame with sensor features + label column
            label_col  : column name containing the cause labels
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            print("scikit-learn not installed. Run: pip install scikit-learn")
            return None
        
        # Only train on rows that are stop events
        stop_df = df[df["is_stop"] == 1].copy()
        
        if len(stop_df) < 20:
            print(f"  Warning: Only {len(stop_df)} labeled stop events. Need ~20+ for reliable training.")
            return None
        
        X = self._prepare_features(stop_df)
        y = stop_df[label_col].fillna("unknown")
        
        # Encode string labels to numbers (sklearn needs numbers)
        # e.g., "overheating" → 0, "bearing_wear" → 1, etc.
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # n_estimators=100 means 100 decision trees
        # max_depth=8 prevents overfitting (trees can't get too complex)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            class_weight="balanced",  # handles imbalanced classes
        )
        self.model.fit(X_train, y_train)
        
        train_acc = self.model.score(X_train, y_train)
        test_acc  = self.model.score(X_test, y_test)
        
        self.is_trained = True
        print(f"  Model trained: train accuracy={train_acc:.1%}, test accuracy={test_acc:.1%}")
        print(f"  Classes: {list(self.label_encoder.classes_)}")
        
        return self
    
    def predict(self, machine_df):
        """
        Predict probabilities for each failure cause.
        Returns: dict of {cause_name: probability}
        """
        if not self.is_trained or self.model is None:
            return {}
        
        X = self._prepare_features(machine_df.tail(1))
        proba = self.model.predict_proba(X)[0]
        
        result = {}
        for i, cls in enumerate(self.label_encoder.classes_):
            result[cls] = round(float(proba[i]), 3)
        
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    
    def get_feature_importance(self):
        """
        Which features matter most for the predictions?
        Returns a ranked list — useful for telling engineers what to watch.
        """
        if not self.is_trained:
            return {}
        
        importances = self.model.feature_importances_
        available = [c for c in self.FEATURE_COLS if c in self.model.feature_names_in_]
        
        return dict(sorted(
            zip(available, importances),
            key=lambda x: x[1],
            reverse=True
        ))


# =============================================================================
# LAYER 3: PATTERN DETECTOR
# =============================================================================

class PatternDetector:
    """
    Finds recurring stop cycles — the "every 71 hours" type insight.
    
    How it works:
        Method A — Simple: Plot stop frequency vs time. Look for peaks.
        Method B — Advanced: Autocorrelation (ACF)
        
    What is Autocorrelation?
        It asks: "How similar is the stop pattern to itself, shifted by N hours?"
        
        If stops happen every 71 hours:
            → shift by 71 hours → patterns line up perfectly → high correlation
            → shift by 35 hours → patterns misalign → low correlation
        
        The lag (hours) with the HIGHEST correlation = the cycle period.
    
    This is the feature that turns your project from a "dashboard" into
    a "predictive maintenance system."
    """
    
    def detect_periodicity(self, machine_df, machine_id):
        """
        Detect if stops happen on a regular cycle.
        
        Returns:
            {
                "has_cycle":    True/False,
                "period_hours": 71.2,
                "confidence":   0.82,
                "interpretation": "likely lubrication dry-out cycle"
            }
        """
        df = machine_df.sort_values("timestamp").copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        stops = df[df["is_stop"] == 1]["timestamp"].sort_values().reset_index(drop=True)
        
        if len(stops) < 4:
            return {"has_cycle": False, "reason": "Not enough stop events (need ≥4)"}
        
        # Calculate intervals between consecutive stops (in hours)
        intervals = []
        for i in range(1, len(stops)):
            delta = (stops[i] - stops[i-1]).total_seconds() / 3600
            if delta < 500:  # Ignore gaps > 500 hours (planned shutdowns)
                intervals.append(delta)
        
        if len(intervals) < 3:
            return {"has_cycle": False, "reason": "Too few consecutive stops"}
        
        intervals = np.array(intervals)
        mean_interval = np.mean(intervals)
        std_interval  = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else 1  # Coefficient of variation
        
        # Low CV (< 0.3) means very regular intervals → strong cycle
        # High CV (> 0.5) means irregular → no clear cycle
        has_cycle   = cv < 0.35
        confidence  = max(0, min(1, 1 - cv * 2))
        
        result = {
            "has_cycle":       has_cycle,
            "period_hours":    round(mean_interval, 1),
            "period_std":      round(std_interval, 1),
            "confidence":      round(confidence, 2),
            "n_intervals":     len(intervals),
            "cv":              round(cv, 3),
        }
        
        if has_cycle:
            result["interpretation"] = self._interpret_cycle(mean_interval)
            result["next_stop_estimate"] = self._estimate_next_stop(stops, mean_interval)
        
        return result
    
    def _interpret_cycle(self, hours):
        """
        Map cycle lengths to likely physical causes.
        Based on industry knowledge of turbocharger manufacturing.
        """
        if 60 <= hours <= 80:
            return "likely lubrication dry-out cycle (grease lasts ~3 days under load)"
        elif 45 <= hours <= 55:
            return "likely coolant filter saturation cycle (~2-day interval)"
        elif 90 <= hours <= 110:
            return "likely bearing fatigue cycle (~4-day load accumulation)"
        elif 168 <= hours <= 180:
            return "weekly pattern — possibly tied to shift changeover or scheduled cleandown"
        elif 20 <= hours <= 30:
            return "daily pattern — likely temperature/thermal cycling effect"
        else:
            return f"periodic cycle of {hours:.0f}h — correlate with maintenance log"
    
    def _estimate_next_stop(self, stop_times, period_hours):
        """
        When is the next stop likely to happen?
        
        Simple: last_stop + period_hours
        """
        last_stop = stop_times.iloc[-1]
        next_stop = last_stop + pd.Timedelta(hours=period_hours)
        hours_away = (next_stop - pd.Timestamp.now()).total_seconds() / 3600
        return {
            "estimated_time": str(next_stop.strftime("%Y-%m-%d %H:%M")),
            "hours_from_now": round(hours_away, 1),
        }
    
    def compute_acf(self, machine_df, max_lag_hours=200, resolution_hours=4):
        """
        Compute autocorrelation of stop events over time.
        
        This is the math behind the "peak at 71h" chart you saw in the web app.
        
        How:
            1. Create a time grid (every 4 hours)
            2. Mark 1 where a stop happened, 0 elsewhere
            3. Correlate this binary series with a shifted copy of itself
            4. The shift with highest correlation = cycle period
        """
        df = machine_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Create uniform time grid
        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        grid = pd.date_range(t_min, t_max, freq=f"{resolution_hours}H")
        
        # Create stop indicator series on the grid
        stop_series = pd.Series(0, index=grid)
        for ts in df[df["is_stop"] == 1]["timestamp"]:
            # Find nearest grid point
            nearest = grid[np.argmin(np.abs(grid - ts))]
            stop_series[nearest] = 1
        
        y = stop_series.values.astype(float)
        y -= y.mean()  # De-mean (required for autocorrelation)
        
        max_lag = int(max_lag_hours / resolution_hours)
        acf_values = []
        
        for lag in range(max_lag + 1):
            if lag == 0:
                acf_values.append(1.0)
            else:
                # Pearson correlation between original and lagged series
                r, _ = stats.pearsonr(y[:-lag], y[lag:])
                acf_values.append(round(r, 4))
        
        lags_hours = [i * resolution_hours for i in range(max_lag + 1)]
        
        # Find peak (ignore lag=0 which is always 1.0)
        acf_array = np.array(acf_values[1:])
        peak_lag_idx = np.argmax(acf_array) + 1
        peak_lag_hours = lags_hours[peak_lag_idx]
        peak_value = acf_values[peak_lag_idx]
        
        return {
            "lags_hours": lags_hours,
            "acf_values": acf_values,
            "dominant_period_hours": peak_lag_hours,
            "peak_acf": round(peak_value, 3),
            "has_significant_period": peak_value > 0.4,
        }


# =============================================================================
# MASTER DIAGNOSIS FUNCTION
# =============================================================================

def diagnose_machine(machine_df, machine_id, ml_classifier=None):
    """
    Single function to call for a complete diagnosis.
    
    Runs all three layers and merges results into one report.
    
    Usage:
        result = diagnose_machine(df[df.machine_id == "CTL-04"], "CTL-04")
        print(result["top_cause"])
        print(result["pattern"])
        print(result["recommendations"])
    """
    rule_engine     = RuleBasedEngine()
    pattern_detector = PatternDetector()
    
    # Layer 1: Rules
    rule_causes = rule_engine.diagnose(machine_df)
    
    # Layer 2: ML (if trained)
    ml_probs = {}
    if ml_classifier and ml_classifier.is_trained:
        ml_probs = ml_classifier.predict(machine_df)
    
    # Layer 3: Patterns
    pattern = pattern_detector.detect_periodicity(machine_df, machine_id)
    
    # Calculate total downtime
    total_stops    = machine_df["is_stop"].sum()
    total_down_hrs = machine_df["stop_duration_min"].sum() / 60
    
    # RUL (remaining useful life)
    current_rul = machine_df["RUL"].iloc[-1] if "RUL" in machine_df.columns else None
    
    return {
        "machine_id":       machine_id,
        "top_cause":        rule_causes[0] if rule_causes else None,
        "all_causes":       rule_causes,
        "ml_probabilities": ml_probs,
        "pattern":          pattern,
        "summary": {
            "total_stops":     int(total_stops),
            "total_down_hrs":  round(float(total_down_hrs), 1),
            "current_rul":     current_rul,
        },
        "recommendations":  [c["recommendation"] for c in rule_causes[:3]],
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    import sys
    sys.path.append("..")
    from data.generate_data import generate_machine_data, MACHINES
    from models.feature_engineering import engineer_all_features
    import pandas as pd
    
    print("Testing root cause engine on CTL-04...")
    df = generate_machine_data("CTL-04", MACHINES["CTL-04"])
    df_feat = engineer_all_features(df)
    
    result = diagnose_machine(df_feat, "CTL-04")
    
    print(f"\nMachine: {result['machine_id']}")
    print(f"Top cause: {result['top_cause']['cause']} ({result['top_cause']['confidence']}%)")
    print(f"Evidence: {result['top_cause']['evidence']}")
    print(f"Pattern: {result['pattern']}")
    print(f"Summary: {result['summary']}")
