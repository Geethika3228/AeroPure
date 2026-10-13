import os
import sys
import glob
import pandas as pd
import numpy
import seaborn as sns
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from data_loader import load_data

# ---------------------------------------------------------------------------
# Loading the dataset (12 site CSVs concatenated)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = load_data()

# Folder where every plot gets saved as a PNG
PLOT_DIR = os.path.join(BASE_DIR, "Output", "plot")
os.makedirs(PLOT_DIR, exist_ok=True)

# Delete any PNGs left over from a previous run before generating fresh ones
for old_png in glob.glob(os.path.join(PLOT_DIR, "*.png")):
    os.remove(old_png)

# ---------------------------------------------------------------------------
# Dataset structure understanding
# ---------------------------------------------------------------------------
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df["station"].value_counts())

# ---------------------------------------------------------------------------
# Univariate Analysis
# ---------------------------------------------------------------------------

# Histogram - PM2.5 distribution
plt.figure(figsize=(6, 4))
plt.hist(df["PM2.5"].dropna(), bins=40, edgecolor="black")
plt.title("PM2.5 Distribution")
plt.xlabel("PM2.5 (µg/m³)")
plt.ylabel("Number of Readings")
plt.savefig(os.path.join(PLOT_DIR, "univariate_pm25_histogram.png"), dpi=150)
plt.show()

# Box plot - outlier detection for PM2.5
plt.figure(figsize=(6, 4))
sns.boxplot(y="PM2.5", data=df)
plt.title("PM2.5 Box Plot (Outlier Detection)")
plt.savefig(os.path.join(PLOT_DIR, "univariate_pm25_boxplot.png"), dpi=150)
plt.show()

# Count plot - readings per monitoring station
plt.figure(figsize=(9, 4))
sns.countplot(x="station", data=df, order=df["station"].value_counts().index)
plt.title("Readings per Monitoring Station")
plt.xlabel("Station")
plt.ylabel("Number of Readings")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "univariate_station_countplot.png"), dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Time Series - core to a forecasting project
# ---------------------------------------------------------------------------

# Monthly average PM2.5 trend across the full 2013-2017 period
monthly_pm25 = df.set_index("datetime")["PM2.5"].resample("ME").mean()
plt.figure(figsize=(10, 4))
monthly_pm25.plot()
plt.title("Monthly Average PM2.5 (2013-2017)")
plt.xlabel("Month")
plt.ylabel("PM2.5 (µg/m³)")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "timeseries_monthly_pm25.png"), dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Bivariate Analysis
# ---------------------------------------------------------------------------

# Box plot - PM2.5 by station
plt.figure(figsize=(9, 4))
sns.boxplot(x="station", y="PM2.5", data=df)
plt.title("PM2.5 by Station")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "bivariate_pm25_vs_station_box.png"), dpi=150)
plt.show()

# Scatter plot - Temperature vs PM2.5
sample_df = df.sample(min(5000, len(df)), random_state=42)
plt.figure(figsize=(6, 4))
sns.scatterplot(x="TEMP", y="PM2.5", data=sample_df, color="teal", alpha=0.4)
plt.title("Temperature vs PM2.5")
plt.savefig(os.path.join(PLOT_DIR, "bivariate_temp_vs_pm25_scatter.png"), dpi=150)
plt.show()

# Count plot - wind direction distribution
plt.figure(figsize=(9, 4))
sns.countplot(x="wd", data=df, order=df["wd"].value_counts().index)
plt.title("Wind Direction Distribution")
plt.xlabel("Wind Direction")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "univariate_winddirection_countplot.png"), dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Multivariate Analysis
# ---------------------------------------------------------------------------

# Correlation heatmap - pollutants + weather variables
numeric_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
plt.figure(figsize=(9, 7))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap: Pollutants & Weather")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "multivariate_correlation_heatmap.png"), dpi=150)
plt.show()

# Pair plot - PM2.5, PM10, TEMP, WSPM
pair_sample = df[["PM2.5", "PM10", "TEMP", "WSPM"]].dropna().sample(2000, random_state=42)
pair_grid = sns.pairplot(pair_sample, diag_kind="kde")
pair_grid.savefig(os.path.join(PLOT_DIR, "multivariate_pairplot.png"), dpi=150)
plt.show()
