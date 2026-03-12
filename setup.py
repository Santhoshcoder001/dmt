"""
Setup configuration for the ABC-HP package.
"""

from setuptools import setup, find_packages

setup(
    name="abc-hp",
    version="1.0.0",
    description=(
        "Adaptive Bias-Corrected Hotspot Prediction System — "
        "road accident risk detection using multi-source data fusion "
        "and spatio-temporal machine learning."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="ABC-HP Contributors",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
    ],
    extras_require={
        "geo": [
            "geopandas>=0.14",
            "shapely>=2.0",
            "pyproj>=3.6",
            "osmnx>=1.6",
            "h3>=3.7",
        ],
        "ml": [
            "xgboost>=2.0",
            "optuna>=3.3",
        ],
        "dl": [
            "tensorflow>=2.13",
        ],
        "viz": [
            "folium>=0.15",
            "matplotlib>=3.7",
            "plotly>=5.15",
            "dash>=2.12",
        ],
        "all": [
            "geopandas>=0.14",
            "shapely>=2.0",
            "pyproj>=3.6",
            "osmnx>=1.6",
            "h3>=3.7",
            "xgboost>=2.0",
            "optuna>=3.3",
            "tensorflow>=2.13",
            "folium>=0.15",
            "matplotlib>=3.7",
            "plotly>=5.15",
            "dash>=2.12",
            "libpysal>=4.8",
        ],
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
