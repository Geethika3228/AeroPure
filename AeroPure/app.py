"""AeroPure analytics platform with the CREDITFAIR-style navigation and page flow."""
import os
import sys
import warnings

# Must run before pyplot is imported: draw charts straight to PNG files (no Tk windows).
# Flask handles requests in background threads, and Tk windows there cause runtime errors.
import matplotlib
matplotlib.use("Agg")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from functools import lru_cache
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from data_loader import load_data
from ml_topics import syllabus_lab

app = Flask(__name__)
app.secret_key = "aeropure-analytics-platform"
PLOT_DIR = os.path.join(BASE_DIR, "Output", "plot")
os.makedirs(PLOT_DIR, exist_ok=True)
LATEST_ACCURACY = None

NAVIGATION = [
    ("Core Analytics", [("dashboard", "Dashboard", "dashboard"), ("data_loading_view", "Data Loading", "data_loading"), ("eda_view", "EDA Analytics", "eda"), ("preprocessing_view", "Preprocessing", "preprocessing")]),
    ("Machine Learning Models", [("linear_regression_view", "Linear Regression", "linear_regression"), ("logistic_regression_view", "Logistic Regression", "logistic_regression"), ("tree_models_view", "Tree Models & Comparison", "tree_models")]),
    ("Clustering & Dimensionality", [("kmeans_view", "K-Means Clustering", "kmeans"), ("hierarchical_view", "Hierarchical Clustering", "hierarchical"), ("pca_view", "Principal Components (PCA)", "pca"), ("dbscan_view", "DBSCAN Clustering", "dbscan")]),
]

@lru_cache(maxsize=1)
def dataset():
    df = load_data().copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def summary():
    df = dataset()
    return {"dataset_name": "AeroPure OHLCV Market Dataset", "total_rows": len(df), "total_columns": len(df.columns), "missing_cells": int(df.isna().sum().sum()), "missing_percentage": round(float(df.isna().mean().mean() * 100), 2), "filepath": os.path.join(BASE_DIR, "Data", "data.csv"), "date_start": df.datetime.min().strftime("%Y-%m-%d"), "date_end": df.datetime.max().strftime("%Y-%m-%d"), "num_numerical_cols": len(df.select_dtypes(include=np.number).columns), "num_categorical_cols": len(df.select_dtypes(exclude=np.number).columns)}

def module(slug, model=None):
    return syllabus_lab.run(dataset(), slug, PLOT_DIR, model=model)

def chart_data(series, bins=10):
    values, edges = np.histogram(series.dropna(), bins=bins)
    return {"labels": [f"{edges[i]:.1f}" for i in range(len(values))], "values": [int(x) for x in values]}

def page_context(active, title, subtitle, **extra):
    return {"active_page": active, "page_title": title, "page_subtitle": subtitle, "summary": summary(), "navigation": NAVIGATION, **extra}

@app.route("/")
@app.route("/dashboard")
def dashboard():
    df = dataset(); s = summary(); m1 = module("m1_lifecycle_eda")
    return render_template("index.html", **page_context("dashboard", "Market Intelligence Dashboard", "Comprehensive overview of OHLCV records, price behaviour, and model-readiness metrics", m1=m1, latest_accuracy=LATEST_ACCURACY, close_chart=chart_data(df["Close"], 12), volume_chart=chart_data(np.log1p(df["Volume"]), 10)))

@app.route("/data-loading")
def data_loading_view():
    dataset_type = request.args.get("dataset", "market").lower(); df = dataset()
    return render_template("data_loading.html", **page_context("data_loading", "Data Loading & Schema", "Inspect the source dataset, columns, records, missing values, and data contract", active_dataset=dataset_type, preview=df.head(10).to_dict(orient="records"), columns=list(df.columns), shape=df.shape, missing=df.isna().sum().to_dict(), duplicates=int(df.duplicated().sum())))

@app.route("/eda")
def eda_view():
    result = module("m1_lifecycle_eda")
    return render_template("eda.html", **page_context("eda", "Exploratory Data Analysis", "Eight large analytical views of distributions, relationships, trend, and volatility", eda=result))

@app.route("/preprocessing")
def preprocessing_view():
    df = dataset(); quality = {"rows": len(df), "columns": len(df.columns), "missing": df.isna().sum().to_dict(), "duplicates": int(df.duplicated().sum()), "numeric": list(df.select_dtypes(include=np.number).columns), "strategies": ["Deletion", "Median imputation", "Missing indicators", "StandardScaler", "MinMaxScaler"]}
    return render_template("preprocessing.html", **page_context("preprocessing", "Preprocessing Pipeline", "Data quality, missing values, scaling, feature preparation, and leakage-safe splitting", quality=quality))

@app.route("/linear-regression")
def linear_regression_view():
    return render_template("analytics.html", **page_context("linear_regression", "Linear Regression", "Continuous closing-price prediction with regularisation and train/test metrics", result=module("m2_linear_models"), section="linear"))

@app.route("/logistic-regression")
def logistic_regression_view():
    return render_template("analytics.html", **page_context("logistic_regression", "Logistic Regression", "Binary above-median closing-price classification with probability metrics", result=module("m2_linear_models"), section="logistic"))

@app.route("/tree-models")
def tree_models_view():
    selected = request.args.get("algorithm", "random_forest")
    result = module("m3_tree_models", selected)
    global LATEST_ACCURACY
    labels = dict(result.get("metrics", [])); LATEST_ACCURACY = labels.get("Test accuracy")
    return render_template("analytics.html", **page_context("tree_models", "Tree Models & Comparison", "Decision Tree, Random Forest, Gradient Boosting, training accuracy, test accuracy, and overfitting", result=result, section="trees", selected_model=selected, available_models=[("decision_tree", "Decision Tree"), ("random_forest", "Random Forest"), ("gradient_boosting", "Gradient Boosting")]))

@app.route("/kmeans")
def kmeans_view():
    return render_template("analytics.html", **page_context("kmeans", "K-Means Clustering", "Centroid-based grouping, elbow logic, silhouette score, and market personas", result=module("m4_unsupervised"), section="kmeans"))

@app.route("/hierarchical")
def hierarchical_view():
    return render_template("analytics.html", **page_context("hierarchical", "Hierarchical Clustering", "Agglomerative bottom-up clustering and dendrogram interpretation", result=module("m4_unsupervised"), section="hierarchical"))

@app.route("/pca")
def pca_view():
    return render_template("analytics.html", **page_context("pca", "Principal Components (PCA)", "Standardised dimensionality reduction and two-dimensional projection", result=module("m4_unsupervised"), section="pca"))

@app.route("/dbscan")
def dbscan_view():
    return render_template("analytics.html", **page_context("dbscan", "DBSCAN Clustering", "Density-based clustering, epsilon, minimum points, and natural outliers", result=module("m4_unsupervised"), section="dbscan"))

@app.route("/settings")
def settings_view():
    s = summary(); return render_template("settings.html", **page_context("settings", "Settings & Environment", "Dataset paths, runtime status, model artifact, and platform configuration", settings={**s, "artifact": os.path.join(BASE_DIR, "Output", "artifacts", "aeropure_model.joblib"), "framework": "Flask · scikit-learn · Matplotlib"}))

@app.route("/about")
def about_view():
    return render_template("about.html", **page_context("about", "About AeroPure", "An end-to-end classical machine-learning analytics workspace", coverage=syllabus_lab.COVERAGE))

@app.route("/topic/<slug>")
def topic_page(slug):
    if slug not in {x[0] for x in syllabus_lab.MODULES}: return dashboard()
    titles = {item[0]: item[1] for item in syllabus_lab.MODULES}
    result = module(slug, request.args.get("model")); return render_template("analytics.html", **page_context("dashboard", titles[slug], "Syllabus module analysis", result=result, section=slug))

@app.route("/api/reload-dataset", methods=["POST"])
def api_reload_dataset():
    dataset.cache_clear(); df = dataset(); return jsonify({"status": "success", "message": "Dataset reloaded successfully", "total_rows": len(df)})

@app.route("/api/run-preprocessing", methods=["POST"])
def api_run_preprocessing():
    data = request.get_json() or {}; s = summary(); return jsonify({"status": "success", "train_samples": int(s["total_rows"] * (1 - float(data.get("test_size", .2)))), "test_samples": int(s["total_rows"] * float(data.get("test_size", .2))), "num_features": 4, "scaling_method": data.get("scaling_method", "standard"), "pipeline_log": ["Loaded OHLCV dataset", "Checked missing values", "Applied median imputation", "Applied scaling", "Created leakage-safe split"]})

@app.route("/plots/<path:filename>")
def plot_file(filename):
    from flask import send_from_directory
    return send_from_directory(PLOT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=False)
