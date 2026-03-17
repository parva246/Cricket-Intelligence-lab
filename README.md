# IPL 2026 Match Predictor

AI-powered IPL Match Predictor that analyses 17 years of ball-by-ball cricket data to predict match outcomes, compare player performance, and break down matches phase by phase.

## Features

- **Match Winner Prediction** — Win probability powered by XGBoost ML model
- **Phase-by-Phase Breakdown** — Powerplay, Middle overs, Death overs analysis
- **Playing XI Selection** — Pick 11 from actual IPL 2026 squads
- **Individual Player Stats** — Batting avg, strike rate, phase performance
- **Batsman vs Bowler Matchups** — Head-to-head records and dominance
- **Venue Analysis** — Chase vs defend win rates

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | XGBoost (300 decision trees) |
| Data Processing | Pandas, NumPy |
| Web Interface | Streamlit |
| Charts | Plotly |
| Language | Python |

## Setup

### 1. Install dependencies
```bash
pip install streamlit xgboost pandas scikit-learn plotly seaborn matplotlib numpy
```

### 2. Extract data
```bash
python setup_data.py
```
This extracts `deliveries.csv` from the included zip file.

### 3. Run the app
```bash
streamlit run ipl_predictor_v2.py
```

The app opens at `http://localhost:8501`

## Project Files

| File | Purpose |
|------|---------|
| `ipl_predictor_v2.py` | Main Streamlit app (latest version) |
| `squads_data.py` | IPL 2026 squad data |
| `setup_data.py` | Data extraction script |
| `matches.csv` | Historical match results (2008-2024) |
| `deliveries.csv.zip` | Ball-by-ball data (compressed) |
| `ipl_2026_squads.txt` | Squad reference |

## Model Accuracy

| Source | Accuracy |
|--------|----------|
| Coin flip | 50% |
| Average fan | 52-55% |
| **This model** | **65-70%** |
| CricViz (professional) | 70-75% |
| Betting markets | 72-78% |

## Disclaimer

Built for data analysis and fun — not for gambling.
