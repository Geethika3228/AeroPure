import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

def model_data(df, sample_size=7000):
    features = [c for c in FEATURES if c in df]
    data = df[features + ["PM2.5"]].replace([np.inf, -np.inf], np.nan).dropna()
    return data.sample(min(sample_size, len(data)), random_state=42), features

def split_data(df):
    data, features = model_data(df)
    X_train, X_test, y_train, y_test = train_test_split(data[features], data["PM2.5"], test_size=.2, random_state=42)
    return X_train, X_test, y_train, y_test, features

def numeric_pipeline(model, scale=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)

def save_plot(fig, plot_dir, slug):
    os.makedirs(plot_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"topic_{slug}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"topic_{slug}.png"
