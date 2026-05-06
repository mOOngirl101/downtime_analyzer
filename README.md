# 🛠️ AI-Powered Downtime Root Cause Analyzer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest-green.svg)

An intelligent diagnostic engine designed for high-precision manufacturing environments (like **Cummins Turbo Technologies**). This system moves beyond simple dashboards by ingesting machine sensor logs to predict failures, identify root causes, and recommend maintenance actions before downtime occurs.

## 📖 Overview
In precision manufacturing, a single machine stoppage on a turbocharger production line can cost thousands of dollars per hour. This project uses the **NASA Turbofan Jet Engine Degradation Dataset** to simulate industrial sensor data and provides:
- **Predictive Analytics:** Estimating Remaining Useful Life (RUL).
- **Anomaly Detection:** Identifying sensor spikes that precede failure.
- **Root Cause Diagnosis:** Mapping sensor patterns to specific failure modes (e.g., Bearing Wear vs. Overheating).
- **Actionable Recommendations:** Suggesting maintenance tasks to prevent downtime.

---

## 🚀 Interactive Web App
The project includes a **Streamlit** interface where users can:
1. **Upload** machine logs (CSV/Excel).
2. **Visualize** real-time sensor health (Temperature, Pressure, Vibration).
3. **Get Diagnosis:** View "Plain English" root cause analysis and failure probability.



---

## 🏗️ Technical Architecture
The system follows a 4-stage pipeline:

1.  **Data Ingestion:** Loads sensor data from 21 different telemetry points (simulating SCADA/MES logs).
2.  **Feature Engineering:** 
    *   Rolling averages and standard deviations.
    *   Sensor trend analysis (Delta between current and baseline).
    *   Operational cycle counting.
3.  **The Intelligence Engine:**
    *   **Regression (RUL):** Predicts how many cycles are left before failure.
    *   **Classification:** Categorizes the type of failure based on sensor thresholds.
4.  **Output Layer:** Streamlit UI presenting the "Junior Maintenance Engineer" view.

---

## 🛠️ Tech Stack
| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Web Interface** | Streamlit |
| **Visualization** | Plotly, Matplotlib |

---

## 📂 Project Structure
```text
├── data/                   # NASA C-MAPSS Dataset
├── notebooks/              # Exploratory Data Analysis (EDA)
├── src/
│   ├── model_trainer.py    # Training RUL and Classification models
│   ├── processor.py        # Data cleaning & feature engineering
│   └── app.py              # Streamlit Web Application
├── requirements.txt        # Project dependencies
└── README.md
⚙️ Installation & Usage
Clone the repository:

Bash
git clone [https://github.com/your-username/downtime-analyzer.git](https://github.com/your-username/downtime-analyzer.git)
cd downtime-analyzer
Install dependencies:

Bash
pip install -r requirements.txt
Run the App:

Bash
streamlit run src/app.py
💡 Industrial Impact (The "Cummins" Angle)
Reduced MTTR (Mean Time to Repair): Diagnosis is provided instantly, reducing "guessing" time.

Transition to Industry 4.0: Shifting from reactive "fix it when it breaks" to proactive "fix it because the data says so."

Cost Savings: Preventing an unplanned stoppage on a CNC Machining or Balancing line can save upwards of $500 - $5,000 per hour in lost productivity.

📝 License
Distributed under the MIT License. See LICENSE for more information.

Next Steps for your Project:

To make the code match the README, here is a tiny snippet of the "Logic" you should include in your app.py:

Python
# Quick logic example for your Streamlit app
def diagnose_root_cause(sensor_data):
    if sensor_data['T24_Temp'].mean() > threshold_high:
        return "🔴 Root Cause: Overheating in Bearing Housing"
    elif sensor_data['Vibration'].std() > vibration_limit:
        return "🟡 Root Cause: Dynamic Unbalance / Bearing Wear"
    else:
        return "🟢 Status: Optimal Performance"
