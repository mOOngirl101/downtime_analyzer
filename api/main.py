"""
main.py  (FastAPI Backend Server)
==================================
What this file does:
    This is the SERVER — it listens for requests from the web app
    and responds with analysis results.

What is an API?
    Think of a restaurant:
        - YOU are the web app (the customer)
        - The KITCHEN is the root cause engine (the logic)
        - The WAITER is the API (takes your order, brings back the result)

    You say: "GET /api/diagnose/CTL-04"
    Server says: "Here's the JSON analysis of machine CTL-04"

What is FastAPI?
    A Python library for building APIs quickly.
    It automatically creates docs at http://localhost:8000/docs
    
What is JSON?
    JavaScript Object Notation — the standard format for sending
    data between a server and a web browser.
    Example: {"machine": "CTL-04", "cause": "Overheating", "confidence": 64}

How to run this:
    pip install fastapi uvicorn pandas numpy scipy scikit-learn
    python main.py
    → Go to http://localhost:8000/docs to see all endpoints
"""

import os
import sys
import json
from typing import Optional
from datetime import datetime

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install fastapi uvicorn pandas numpy scipy scikit-learn")
    sys.exit(1)

from data.generate_data import generate_all_data, MACHINES
from models.feature_engineering import engineer_all_features
from models.root_cause_engine import diagnose_machine, PatternDetector

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="Downtime Root Cause Analyzer API",
    description="""
    AI-powered machine downtime diagnosis for Cummins Turbo Technologies.
    
    Uses NASA CMAPSS turbofan sensor patterns to:
    - Detect anomalies in sensor readings
    - Identify probable root causes
    - Discover recurring failure cycles
    - Estimate Remaining Useful Life (RUL)
    """,
    version="1.0.0",
)

# ── CORS: Allow the web app to call this API ──────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# Without this, browsers BLOCK requests from different domains.
# Since our frontend runs on port 3000 and backend on 8000 → different origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # In production: specify exact frontend URL
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# =============================================================================
# DATA LOADING (runs once at startup)
# =============================================================================

# We load data ONCE when the server starts, then keep it in memory.
# This is faster than reading from disk on every request.
print("Loading machine data...")
RAW_DATA = generate_all_data()
FEATURED_DATA = engineer_all_features(RAW_DATA)
print(f"Data loaded: {len(FEATURED_DATA):,} rows, {FEATURED_DATA['machine_id'].nunique()} machines")


def get_machine_data(machine_id: str, days: int = 30) -> pd.DataFrame:
    """Helper: get recent data for one machine."""
    if machine_id not in MACHINES:
        raise HTTPException(status_code=404, detail=f"Machine '{machine_id}' not found")
    
    df = FEATURED_DATA[FEATURED_DATA["machine_id"] == machine_id].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Filter to recent N days
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df = df[df["timestamp"] >= cutoff]
    
    return df.sort_values("timestamp")


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint — confirms server is running."""
    return {
        "status": "running",
        "service": "Downtime Root Cause Analyzer",
        "machines": list(MACHINES.keys()),
        "docs": "/docs",
    }


@app.get("/api/fleet/overview")
async def fleet_overview():
    """
    Returns summary stats for ALL machines.
    
    The web app calls this to populate the Fleet Overview tab.
    
    Example response:
        {
          "total_machines": 6,
          "critical_machines": 2,
          "total_stops_30d": 48,
          "machines": [...]
        }
    """
    machines_summary = []
    
    for machine_id, config in MACHINES.items():
        df = FEATURED_DATA[FEATURED_DATA["machine_id"] == machine_id]
        recent = df.tail(180)  # last 30 days (6 readings/day * 30 = 180)
        
        total_stops    = int(recent["is_stop"].sum())
        total_down_hrs = round(float(recent["stop_duration_min"].sum() / 60), 1)
        current_rul    = int(df["RUL"].iloc[-1]) if len(df) > 0 else None
        
        # Status based on RUL
        if current_rul is not None and current_rul < 30:
            status = "critical"
        elif current_rul is not None and current_rul < 60:
            status = "warning"
        else:
            status = "ok"
        
        machines_summary.append({
            "machine_id":   machine_id,
            "line":         config["line"],
            "failure_mode": config["failure_mode"],
            "stops_30d":    total_stops,
            "downtime_hrs": total_down_hrs,
            "rul":          current_rul,
            "status":       status,
        })
    
    # Sort by RUL (most critical first)
    machines_summary.sort(key=lambda x: x["rul"] or 999)
    
    critical_count = sum(1 for m in machines_summary if m["status"] == "critical")
    
    return {
        "total_machines":    len(machines_summary),
        "critical_machines": critical_count,
        "total_stops_30d":   sum(m["stops_30d"] for m in machines_summary),
        "total_downtime_hrs":sum(m["downtime_hrs"] for m in machines_summary),
        "machines":          machines_summary,
        "timestamp":         datetime.now().isoformat(),
    }


@app.get("/api/diagnose/{machine_id}")
async def diagnose(
    machine_id: str,
    days: int = Query(default=30, ge=1, le=90, description="Look-back window in days"),
):
    """
    Full diagnosis for a single machine.
    
    Runs all 3 analysis layers:
        1. Rule-based cause detection
        2. Pattern/cycle detection
        3. RUL estimation
    
    Parameters:
        machine_id : e.g. "CTL-04"
        days       : how many days of history to analyze (default 30)
    
    Usage:
        GET /api/diagnose/CTL-04
        GET /api/diagnose/CTL-04?days=60
    """
    df = get_machine_data(machine_id, days)
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for {machine_id} in last {days} days")
    
    # Run the diagnosis
    result = diagnose_machine(df, machine_id)
    
    # Convert numpy types to Python types (JSON serialization)
    # NumPy int64 / float64 aren't directly serializable to JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return round(float(obj), 3)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj
    
    # Manually build clean response
    top = result["top_cause"]
    pattern = result["pattern"]
    
    response = {
        "machine_id":  machine_id,
        "line":        MACHINES[machine_id]["line"],
        "analysis_window_days": days,
        "diagnosis": {
            "top_cause":    top["cause"] if top else "Unknown",
            "confidence":   top["confidence"] if top else 0,
            "evidence":     top["evidence"] if top else "",
            "priority":     top.get("priority", "monitor") if top else "monitor",
            "all_causes": [
                {k: convert(v) for k, v in c.items()}
                for c in result["all_causes"]
            ],
        },
        "pattern": {
            "has_cycle":     pattern.get("has_cycle", False),
            "period_hours":  convert(pattern.get("period_hours")),
            "period_std":    convert(pattern.get("period_std")),
            "confidence":    convert(pattern.get("confidence")),
            "interpretation":pattern.get("interpretation", ""),
            "next_stop":     pattern.get("next_stop_estimate"),
        },
        "summary": {
            "total_stops":    convert(result["summary"]["total_stops"]),
            "total_down_hrs": convert(result["summary"]["total_down_hrs"]),
            "rul":            convert(result["summary"]["current_rul"]),
        },
        "recommendations": result["recommendations"],
        "timestamp": datetime.now().isoformat(),
    }
    
    return JSONResponse(content=response)


@app.get("/api/sensors/{machine_id}")
async def get_sensor_trends(
    machine_id: str,
    sensor: str = Query(default="s4", description="Sensor code (e.g. s4, s8, s11)"),
    days: int = Query(default=14, ge=1, le=90),
):
    """
    Returns time-series data for a specific sensor on a machine.
    
    The web app uses this to draw the sensor trend charts.
    
    Parameters:
        machine_id : Machine ID
        sensor     : Sensor code (s4 = fan exit temp, s8 = fan speed, etc.)
        days       : Days of history to return
    
    Usage:
        GET /api/sensors/CTL-04?sensor=s4&days=14
    """
    df = get_machine_data(machine_id, days)
    
    if sensor not in df.columns:
        raise HTTPException(status_code=400, detail=f"Sensor '{sensor}' not found. Available: {[c for c in df.columns if c.startswith('s') and len(c) <= 3]}")
    
    # Downsample to max 200 points for performance
    # (90 days * 6 readings = 540 rows — too many for a chart)
    step = max(1, len(df) // 200)
    df_sampled = df.iloc[::step].copy()
    
    return {
        "machine_id":  machine_id,
        "sensor":      sensor,
        "sensor_name": {
            "s4": "T50 Fan Exit Temp (K)", "s8": "Fan Speed (rpm)",
            "s3": "HPC Inlet Temp (K)",   "s7": "HPC Pressure (psia)",
            "s11": "Static Pressure (psia)", "s15": "Bypass Ratio",
        }.get(sensor, sensor),
        "data": [
            {
                "timestamp": str(row["timestamp"]),
                "value":     round(float(row[sensor]), 3) if not pd.isna(row[sensor]) else None,
                "is_stop":   int(row["is_stop"]),
                "rul":       int(row["RUL"]),
            }
            for _, row in df_sampled.iterrows()
        ],
    }


@app.get("/api/pattern/{machine_id}")
async def get_pattern_analysis(machine_id: str):
    """
    Autocorrelation analysis for a machine.
    Returns ACF values for the stop-cycle chart in the web app.
    
    Usage:
        GET /api/pattern/CTL-04
    """
    df = FEATURED_DATA[FEATURED_DATA["machine_id"] == machine_id].copy()
    
    if len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data for {machine_id}")
    
    detector = PatternDetector()
    
    # Periodicity test
    period_result = detector.detect_periodicity(df, machine_id)
    
    # ACF computation
    acf_result = detector.compute_acf(df)
    
    return {
        "machine_id": machine_id,
        "periodicity": {
            "has_cycle":     period_result.get("has_cycle", False),
            "period_hours":  period_result.get("period_hours"),
            "std_hours":     period_result.get("period_std"),
            "confidence":    period_result.get("confidence"),
            "interpretation":period_result.get("interpretation", ""),
            "n_intervals":   period_result.get("n_intervals"),
        },
        "acf": {
            "lags_hours":    acf_result["lags_hours"],
            "acf_values":    acf_result["acf_values"],
            "dominant_period_hours": acf_result["dominant_period_hours"],
            "peak_acf":      acf_result["peak_acf"],
        },
    }


@app.get("/api/cost_impact")
async def cost_impact(
    cost_per_hour: float = Query(default=28000, description="Cost per downtime hour in ₹"),
):
    """
    Calculates potential cost savings if detected patterns are addressed.
    
    Formula:
        Preventable downtime = stops_with_pattern × avg_stop_duration
        Savings = preventable_downtime × cost_per_hour
    
    Usage:
        GET /api/cost_impact
        GET /api/cost_impact?cost_per_hour=50000
    """
    savings_breakdown = []
    
    for machine_id, config in MACHINES.items():
        df = FEATURED_DATA[FEATURED_DATA["machine_id"] == machine_id]
        recent = df.tail(180)
        
        detector = PatternDetector()
        period = detector.detect_periodicity(df, machine_id)
        
        total_down_hrs = recent["stop_duration_min"].sum() / 60
        stops = int(recent["is_stop"].sum())
        
        # Estimate preventable % based on whether there's a detectable pattern
        # If there's a clear cycle, ~70% of stops are preventable
        # If no pattern, only ~20% (random failures are harder to prevent)
        preventable_pct = 0.70 if period.get("has_cycle") else 0.20
        preventable_hrs = round(total_down_hrs * preventable_pct, 1)
        savings_inr     = round(preventable_hrs * cost_per_hour)
        
        savings_breakdown.append({
            "machine_id":      machine_id,
            "line":            config["line"],
            "total_down_hrs":  round(total_down_hrs, 1),
            "preventable_hrs": preventable_hrs,
            "has_pattern":     period.get("has_cycle", False),
            "pattern_period_hrs": period.get("period_hours"),
            "savings_inr":     savings_inr,
            "preventable_pct": int(preventable_pct * 100),
        })
    
    total_savings = sum(m["savings_inr"] for m in savings_breakdown)
    savings_breakdown.sort(key=lambda x: x["savings_inr"], reverse=True)
    
    return {
        "cost_per_hour_inr":    cost_per_hour,
        "monthly_savings_inr":  total_savings,
        "annual_savings_inr":   total_savings * 12,
        "breakdown":            savings_breakdown,
    }


# =============================================================================
# RUN THE SERVER
# =============================================================================

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("Install uvicorn: pip install uvicorn")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  Downtime Root Cause Analyzer — Backend Server")
    print("="*60)
    print("  API docs:  http://localhost:8000/docs")
    print("  Fleet:     http://localhost:8000/api/fleet/overview")
    print("  Diagnose:  http://localhost:8000/api/diagnose/CTL-04")
    print("="*60 + "\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",   # Listen on all interfaces
        port=8000,
        reload=True,       # Auto-restart when you save code changes
        log_level="info",
    )
