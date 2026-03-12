"""
ABC-HP: Adaptive Bias-Corrected Hotspot Prediction System
==========================================================
A road accident risk detection platform using multi-source data fusion
and spatio-temporal machine learning.

Modules
-------
data_acquisition   : M1 — collect accident records and contextual datasets
preprocessing      : M2 — clean, align, and normalise all data sources
bias_correction    : M3 — adjust under-reported accident counts
spatial_grid       : M4 — divide study area into adaptive H3 cells
feature_engineering: M5 — extract spatio-temporal + contextual features
ml_prediction      : M6 — classify/regress risk level per grid cell
visualization      : M7 — generate risk maps and dashboards
"""

from .data_acquisition import DataAcquisition
from .preprocessing import Preprocessor
from .bias_correction import BiasCorrector
from .spatial_grid import SpatialGridEngine
from .feature_engineering import FeatureEngineer
from .ml_prediction import MLPredictionEngine
from .visualization import HotspotVisualizer

__all__ = [
    "DataAcquisition",
    "Preprocessor",
    "BiasCorrector",
    "SpatialGridEngine",
    "FeatureEngineer",
    "MLPredictionEngine",
    "HotspotVisualizer",
]

__version__ = "1.0.0"
