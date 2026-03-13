# ABC-HP — Adaptive Bias-Corrected Hotspot Prediction System

**Road Accident Risk Detection Using Multi-Source Data Fusion and Spatio-Temporal Machine Learning**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

The **ABC-HP system** is a software-based, data-driven road safety intelligence platform. It processes historical accident records alongside live contextual data streams, corrects for systematic reporting bias, and outputs predictive risk maps identifying future accident hotspots.

The system is organized into **7 functional modules** operating in a sequential pipeline:

| Module | Function | Output |
|--------|----------|--------|
| M1 — Data Acquisition | Collect accident records + contextual datasets | Raw multi-source dataset |
| M2 — Preprocessing | Clean, align, normalize all data sources | Unified feature-ready dataset |
| M3 — Bias Correction | Adjust under-reported accident counts | Bias-corrected accident weights |
| M4 — Spatial Grid Engine | Divide study area into adaptive cells | Gridded spatial units |
| M5 — Feature Engineering | Extract spatio-temporal + contextual features | Feature matrix per grid cell |
| M6 — ML Prediction Engine | Classify/regress risk level per grid cell | Risk probability scores |
| M7 — Visualization & Reporting | Generate risk maps and dashboards | Interactive hotspot maps |

### Data Flow

```
Raw Data Sources → Preprocessing → Bias Correction → Spatial Gridding
→ Feature Matrix → ML Ensemble → Risk Scores → Hotspot Map
```

---

## Installation

```bash
# Core installation (data processing + ML)
pip install -e .

# Full installation (includes geo, deep learning, visualisation)
pip install -e ".[all]"

# Development installation
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- NumPy, Pandas, scikit-learn (core)
- GeoPandas, Shapely, H3 (geospatial — `geo` extra)
- XGBoost, Optuna (ML — `ml` extra)
- TensorFlow (LSTM — `dl` extra)
- Folium, Matplotlib, Plotly, Dash (visualisation — `viz` extra)

---

## Quick Start

```python
from abc_hp import (
    DataAcquisition,
    Preprocessor,
    BiasCorrector,
    SpatialGridEngine,
    FeatureEngineer,
    MLPredictionEngine,
    HotspotVisualizer,
)

# M1 — Load accident data
da = DataAcquisition(data_dir="data/raw")
df = da.load_csv("data/raw/us_accidents.csv", source="kaggle_us")
df = da.validate(df)

# M2 — Preprocess
pre = Preprocessor(max_road_distance_m=100.0)
df = pre.run(df)

# M3 — Bias correction (encode categoricals first)
df = BiasCorrector.encode_categoricals(df)
bc = BiasCorrector()
bc.fit(df, reported_col="is_reported")
df = bc.apply_weights(df)

# M4 — Assign H3 grid cells
grid = SpatialGridEngine(resolution=8, grid_type="h3")
df = grid.assign_cells(df)
summary = grid.build_grid_summary(df, weight_col="bias_weight")

# M5 — Build feature matrix
fe = FeatureEngineer(grid_engine=grid)
feature_matrix = fe.build_feature_matrix(df, summary)

# M6 — Train ensemble & predict
engine = MLPredictionEngine(use_xgboost=True)
labels = engine.build_labels(feature_matrix)
engine.fit(feature_matrix, labels, drop_cols=["corrected_count"])
risk_scores = engine.predict_proba(feature_matrix)

# M7 — Visualise
risk_df = feature_matrix.copy()
risk_df["risk_score"] = risk_scores
viz = HotspotVisualizer(output_dir="output")
viz.generate_summary_report(risk_df)
viz.export_risk_scores(risk_df, fmt="csv")
```

---

## Module Details

### M1 — Data Acquisition (`abc_hp.data_acquisition`)

Loads and normalises accident records from heterogeneous public sources:

- **Kaggle US Accidents** (7.7M records, GPS + weather)
- **UK STATS19** (severity, time, location)
- **US FARS** (fatal crashes, GPS, road type)

```python
da = DataAcquisition()
df = da.load_csv("accidents.csv", source="kaggle_us")   # auto-renames columns
df = da.validate(df)                                     # drops missing GPS rows
```

### M2 — Preprocessing (`abc_hp.preprocessing`)

1. Coordinate standardisation — validates WGS-84 bounds
2. Temporal features — hour, day_of_week, month, is_weekend, is_public_holiday
3. Missing values — mode imputation (severity), forward/backward fill (weather)
4. Outlier detection — removes records > 100 m from road network
5. IDW weather interpolation utility

```python
pre = Preprocessor(max_road_distance_m=100.0, public_holidays={"2024-01-01"})
df = pre.run(df)
```

### M3 — Bias Correction (`abc_hp.bias_correction`)

Core novelty: **Poisson Inverse Probability Weighting** to correct systematic under-reporting.

```
p(i) = P(reported | pop_density, road_type, weather, hour)
w(i) = 1 / p(i)
```

Addresses four bias types: geographic, severity, temporal, and socioeconomic.

```python
bc = BiasCorrector(weight_clip_max=100.0)
bc.fit(df, reported_col="is_reported")
df = bc.apply_weights(df)
counts = bc.aggregate_weighted_counts(df, grid_col="grid_id")
```

### M4 — Spatial Grid Engine (`abc_hp.spatial_grid`)

Divides the study area into prediction units using:
- **H3 hexagonal grid** (recommended, resolution 8 ≈ 0.74 km²)
- **Square grid** (pure-Python fallback)

```python
engine = SpatialGridEngine(resolution=8, grid_type="h3")
df = engine.assign_cells(df)
summary = engine.build_grid_summary(df, weight_col="bias_weight")
neighbours = engine.get_neighbours(cell_id, k=1)
```

### M5 — Feature Engineering (`abc_hp.feature_engineering`)

Builds 21+ features per grid cell:

| Category | Features |
|----------|----------|
| Accident history | corrected_count, accident_density |
| Temporal | peak_hour_ratio, night_ratio, weekend_ratio |
| Weather | rain_accident_ratio, fog_ratio, severe_weather_ratio |
| Infrastructure | junction_count, road_length_km |
| Population | pop_density |
| POI proximity | school_within_500m, hospital_within_500m |
| Spatial lag | neighbour_corrected_count |

```python
fe = FeatureEngineer(grid_engine=engine)
fm = fe.build_feature_matrix(df, summary, contextual=contextual_df)
```

### M6 — ML Prediction Engine (`abc_hp.ml_prediction`)

Stacking ensemble of three models:
- **Random Forest** — stable ensemble baseline
- **XGBoost** — handles class imbalance, SHAP-explainable
- **LSTM** — temporal patterns (opt-in, requires TensorFlow)

Meta-model: Logistic Regression combining out-of-fold predictions.

```python
engine = MLPredictionEngine(use_xgboost=True, risk_percentile=75)
labels = engine.build_labels(feature_matrix)
engine.fit(feature_matrix, labels, drop_cols=["corrected_count"])
risk_proba = engine.predict_proba(feature_matrix)
metrics = engine.evaluate(test_fm, test_labels)
```

Evaluation metrics: Precision, Recall, F1, AUC-ROC, AUC-PR (best for imbalanced data).

### M7 — Visualization & Reporting (`abc_hp.visualization`)

```python
viz = HotspotVisualizer(output_dir="output")

# Interactive Folium heatmap + circle markers
m = viz.folium_risk_map(risk_df)

# Static research figures
viz.plot_risk_distribution(risk_df)
viz.plot_feature_importance(importances)

# Export
viz.export_risk_scores(risk_df, fmt="csv")
viz.export_risk_scores(risk_df, fmt="geojson")   # requires geopandas
viz.generate_summary_report(risk_df)
```

---

## Data Sources

| Dataset | Coverage | Access | Key Fields |
|---------|----------|--------|------------|
| UK STATS19 | Great Britain | Public | Location, severity, time |
| USA FARS | US fatal crashes | Public | GPS, road type, weather |
| India iRAD | Indian highways | Research access | Road, vehicle, person |
| Kaggle US Accidents | 7.7M records | Free | Coordinates, weather |
| OpenStreetMap | Global roads | Free API | Road geometry |

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/ --cov=abc_hp --cov-report=term-missing
```

---

## Implementation Roadmap

| Phase | Duration | Description |
|-------|----------|-------------|
| 1 | Month 1–2 | Dataset setup, Python GIS environment, OSM spatial join |
| 2 | Month 2–3 | Bias correction — train reporting probability model |
| 3 | Month 3–4 | Feature matrix — H3 grid + 21 spatial features |
| 4 | Month 4–6 | Model training — XGBoost, Random Forest, LSTM ensemble |
| 5 | Month 6–7 | Visualisation — interactive maps + SHAP explainability |
| 6 | Month 7–10 | Patent filing + IEEE paper publication |

---

## Technology Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Spatial | GeoPandas, OSMnx, H3, Shapely |
| ML | scikit-learn, XGBoost |
| Deep Learning | TensorFlow (LSTM) |
| Visualisation | Folium, Plotly, Dash, Matplotlib |
| Optimisation | Optuna |

---

## Patent Claims

**Claim A1** — Bias-corrected accident hotspot prediction system using inverse probability weighting.

**Claim A2** — Multi-source fusion of accident, weather, traffic, and infrastructure data.

**Claim B1** — Spatio-temporal accident risk prediction using hexagonal spatial grids and ML ensemble.

**Claim B2** — Adaptive hotspot maps updated with live contextual data.

---

*ABC-HP System — Adaptive Bias-Corrected Hotspot Prediction*
