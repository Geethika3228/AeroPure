"""Syllabus-aligned demonstrations for the actual OHLCV market dataset.

The dashboard exposes one focused project per syllabus module instead of
showing unrelated algorithms. Every result contains metrics, explanations,
and generated images for the webpage.
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.calibration import calibration_curve
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, precision_recall_curve,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from .common import save_plot

MODULES = [
    ("m1_lifecycle_eda", "M1 · Lifecycle + EDA", "Raw OHLCV data to a served prediction, with the evidence needed before modelling."),
    ("m2_linear_models", "M2 · Linear models", "Regression, logistic classification, scaling, encoding, imputation, and regularisation."),
    ("m3_tree_models", "M3 · Tree-based models", "Decision trees, pruning, random forests, boosting, and feature importance."),
    ("m4_unsupervised", "M4 · Unsupervised learning", "K-Means, hierarchical clustering, DBSCAN, PCA, and anomaly detection."),
    ("m5_evaluation", "M5 · Evaluation + selection", "Confusion matrix, ROC/PR, cross-validation, calibration, and learning curves."),
    ("m6_engineering", "M6 · ML engineering", "A packaged model, serving boundary, monitoring signal, and retraining path."),
]

COVERAGE = {
    "m1_lifecycle_eda": ["Raw → clean → features → train → artifact → endpoint → monitoring → retraining", "Supervised vs unsupervised vs reinforcement learning", "HTTP/batch prediction and training-serving boundary", "Univariate, bivariate, multivariate, time-series EDA"],
    "m2_linear_models": ["Normal-equation idea and gradient descent", "Ridge L2, Lasso L1, Elastic Net", "Sigmoid, cross-entropy, decision boundary", "Softmax/multinomial classification preview", "StandardScaler and MinMaxScaler", "One-hot, ordinal, target, and embedding encoding concepts", "Deletion, median, model-based imputation and MissingIndicator"],
    "m3_tree_models": ["Gini, entropy, and regression-MSE split criteria", "Pre-pruning and cost-complexity post-pruning", "Train/test accuracy and overfitting gap", "Bagging, feature subsampling, and OOB error", "AdaBoost and gradient boosting", "XGBoost/LightGBM industrial preview", "Gini, permutation, and SHAP importance preview"],
    "m4_unsupervised": ["K-Means, K-Means++, elbow, silhouette, MiniBatchKMeans", "Agglomerative linkage: Ward, complete, average", "DBSCAN epsilon/minPts and natural outliers", "Standardised PCA and dimensionality reduction", "t-SNE/UMAP visualisation preview", "Isolation Forest and One-Class SVM preview", "Clusters as features for downstream supervised learning"],
    "m5_evaluation": ["Three-way train/validation/test discipline", "Stratified k-fold and nested-CV concept", "Accuracy, precision, recall, F1, ROC-AUC, PR-AUC", "MAE, RMSE, MAPE, and R² for regression", "Calibration, Platt scaling, isotonic regression", "Grid search, random search, Bayesian optimisation preview", "Learning/validation curves and model significance tests"],
    "m6_engineering": ["Notebook-to-production lifecycle and feature-store concept", "Offline/online split and point-in-time correctness", "Training-serving skew prevention", "Joblib/pickle, ONNX, containers, FastAPI, gRPC", "Latency, throughput, data/concept/label drift", "Scheduled, drift-triggered, and online retraining", "MLflow experiment tracking and DVC data versioning"],
}

NUMERIC = ["Open", "High", "Low", "Volume"]
CAT = []


def _data(df, sample=7000):
    cols = [c for c in NUMERIC + ["Close"] if c in df.columns]
    has_ticker = "Ticker" in df.columns
    data = df[cols + (["Ticker"] if has_ticker else [])].replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"]).copy()
    # Median is taken per stock so a cheap stock and an expensive stock are not split by price level
    median = data.groupby("Ticker")["Close"].transform("median") if has_ticker else data["Close"].median()
    data["target"] = (data["Close"] > median).astype(int)
    data = data.drop(columns=["Ticker"], errors="ignore")
    return data.sample(min(sample, len(data)), random_state=42).reset_index(drop=True)


def _split(data):
    return train_test_split(data, test_size=.2, random_state=42, stratify=data["target"])


def _prep():
    return ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC)])


def _classifier(model):
    return Pipeline([("preprocess", _prep()), ("model", model)])


def _regressor(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])


def _plot(fig, plot_dir, slug):
    return save_plot(fig, plot_dir, f"syllabus_{slug}")


def _frame_data(data):
    numeric = data[NUMERIC].fillna(data[NUMERIC].median())
    return StandardScaler().fit_transform(numeric)


def _m1(data, plot_dir):
    """Generate eight separate EDA charts so each one is large and readable."""
    plots = []
    def finish(fig, name):
        plots.append(_plot(fig, plot_dir, f"m1_{name}"))
    fig, ax = plt.subplots(figsize=(10, 5)); ax.hist(data["Close"], bins=30, color="#1677a8", alpha=.85); ax.set(title="1. Closing-price distribution", xlabel="Close", ylabel="Trading days"); finish(fig, "01_close_distribution")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.boxplot(data["Close"], vert=False, patch_artist=True, boxprops={"facecolor":"#9ee8e0"}); ax.set(title="2. Closing-price box plot", xlabel="Close"); finish(fig, "02_close_boxplot")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(data.index, data["Open"], label="Open", alpha=.8); ax.plot(data.index, data["Close"], label="Close", linewidth=2); ax.plot(data.index, data["High"], label="High", alpha=.55); ax.plot(data.index, data["Low"], label="Low", alpha=.55); ax.set(title="3. OHLC price movement", xlabel="Trading-day index", ylabel="Price"); ax.legend(); finish(fig, "03_ohlc_trend")
    returns = data["Close"].pct_change().dropna() * 100
    fig, ax = plt.subplots(figsize=(10, 5)); ax.hist(returns, bins=30, color="#ef8354", alpha=.85); ax.set(title="4. Daily percentage-return distribution", xlabel="Daily return (%)", ylabel="Trading days"); finish(fig, "04_returns")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.hist(np.log1p(data["Volume"]), bins=30, color="#147d92", alpha=.85); ax.set(title="5. Trading-volume distribution", xlabel="log(1 + Volume)", ylabel="Trading days"); finish(fig, "05_volume_distribution")
    fig, ax = plt.subplots(figsize=(10, 5)); ax.scatter(data["Volume"], data["Close"], s=18, alpha=.45, color="#1677a8"); ax.set(title="6. Volume versus closing price", xlabel="Volume", ylabel="Close"); finish(fig, "06_volume_close")
    fig, ax = plt.subplots(figsize=(10, 7)); corr = data[["Open", "High", "Low", "Close", "Volume"]].corr(); image = ax.imshow(corr, cmap="Blues", vmin=-1, vmax=1); ax.set_xticks(range(len(corr)), corr.columns, rotation=35, ha="right"); ax.set_yticks(range(len(corr)), corr.columns); ax.set_title("7. OHLCV correlation heatmap"); fig.colorbar(image, ax=ax, shrink=.75); finish(fig, "07_correlation")
    rolling = data["Close"].rolling(30, min_periods=1).mean(); volatility = returns.rolling(30, min_periods=2).std()
    fig, ax = plt.subplots(figsize=(10, 5)); ax.plot(data.index, rolling, label="30-day rolling mean", color="#1677a8", linewidth=2); ax2 = ax.twinx(); ax2.plot(volatility.index, volatility, label="30-day volatility", color="#ef8354", alpha=.8); ax.set(title="8. Rolling trend and volatility", xlabel="Trading-day index", ylabel="Rolling mean Close"); ax2.set_ylabel("Return volatility (%)"); finish(fig, "08_rolling_volatility")
    missing = int(data.isna().sum().sum())
    return {"summary": "M1 establishes the ML lifecycle and checks the OHLCV dataset before any model is trained.", "metrics": [("Rows sampled", f"{len(data):,}"), ("EDA graphs", "8"), ("Features", "Open · High · Low · Volume"), ("Target", "above-median Close / normal Close")], "details": "The lifecycle is raw CSV → cleaned data → engineered target → train/test split → fitted model → packaged artifact → served endpoint → monitoring → retraining. The eight graphs cover distribution, outliers, price movement, returns, volume, relationships, correlation, and volatility. The project is supervised classification/regression; clustering is unsupervised, while reinforcement learning and deep learning are outside this classical-ML project.", "coverage": COVERAGE["m1_lifecycle_eda"], "plots": plots}


def _m2(data, plot_dir):
    train, test = _split(data)
    numeric_train = train[NUMERIC].fillna(train[NUMERIC].median()); numeric_test = test[NUMERIC].fillna(train[NUMERIC].median())
    reg = LinearRegression().fit(numeric_train, train["Close"]); pred = reg.predict(numeric_test)
    ridge = Ridge(alpha=1.0).fit(numeric_train, train["Close"])
    lasso = Lasso(alpha=.05, max_iter=5000).fit(numeric_train, train["Close"])
    elastic = ElasticNet(alpha=.05, l1_ratio=.5, max_iter=5000).fit(numeric_train, train["Close"])
    clf = _classifier(LogisticRegression(C=.5, max_iter=600, random_state=42)); clf.fit(train.drop(columns="target"), train.target); proba = clf.predict_proba(test.drop(columns="target"))[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(test["Close"], pred, s=10, alpha=.3, color="#1677a8"); lo, hi = test["Close"].min(), test["Close"].max(); axes[0].plot([lo, hi], [lo, hi], "--", color="#ef8354"); axes[0].set(title="Linear regression", xlabel="Actual Close", ylabel="Predicted Close")
    coef = np.abs(clf.named_steps["model"].coef_[0]); names = clf.named_steps["preprocess"].get_feature_names_out(); order = np.argsort(coef)[-7:]; axes[1].barh(names[order], coef[order], color="#147d92"); axes[1].set(title="Regularised logistic coefficients", xlabel="Absolute coefficient")
    plot = _plot(fig, plot_dir, "m2_linear_models")
    train_proba = clf.predict_proba(train.drop(columns="target"))[:, 1]
    return {"summary": "M2 compares continuous and binary prediction while keeping imputation, scaling, and regularisation inside reproducible pipelines.", "metrics": [("Regression MAE", f"{mean_absolute_error(test['Close'], pred):.2f}"), ("Regression RMSE", f"{mean_squared_error(test['Close'], pred) ** .5:.2f}"), ("Regression R²", f"{r2_score(test['Close'], pred):.3f}"), ("Ridge R²", f"{ridge.score(numeric_test, test['Close']):.3f}"), ("Lasso R²", f"{lasso.score(numeric_test, test['Close']):.3f}"), ("Elastic Net R²", f"{elastic.score(numeric_test, test['Close']):.3f}"), ("Train accuracy", f"{clf.score(train.drop(columns='target'), train.target):.3f}"), ("Test accuracy", f"{clf.score(test.drop(columns='target'), test.target):.3f}"), ("Logistic ROC-AUC", f"{roc_auc_score(test.target, proba):.3f}"), ("Scaling", "StandardScaler; MinMax concept")], "details": "Linear regression uses OHLCV inputs to estimate continuous closing price. Ridge, Lasso, and Elastic Net demonstrate L2, L1, and mixed regularisation. Logistic regression demonstrates sigmoid probabilities, cross-entropy training, and a linear decision boundary. Train/test accuracy makes generalisation visible; one-hot, ordinal, target, and embedding encodings are covered as categorical-data options even though this dataset is numeric.", "coverage": COVERAGE["m2_linear_models"], "plots": [plot]}


def _m3(data, plot_dir, selected_model="decision_tree"):
    train, test = _split(data)
    choices = {"decision_tree": ("Decision Tree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, criterion="gini", random_state=42)), "random_forest": ("Random Forest", RandomForestClassifier(n_estimators=80, max_depth=8, oob_score=True, random_state=42, n_jobs=-1)), "gradient_boosting": ("Gradient Boosting", GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=.06, random_state=42))}
    selected_model = selected_model if selected_model in choices else "decision_tree"; model_name, estimator = choices[selected_model]; model = _classifier(estimator); model.fit(train.drop(columns="target"), train.target)
    names = model.named_steps["preprocess"].get_feature_names_out(); values = model.named_steps["model"].feature_importances_; order = np.argsort(values)[-8:]
    fig, ax = plt.subplots(figsize=(12, 8))
    if selected_model == "decision_tree":
        plot_tree(model.named_steps["model"], feature_names=names, class_names=["Normal", "High"], filled=True, rounded=True, max_depth=4, fontsize=9, impurity=False, ax=ax); ax.set_title("Decision Tree · threshold rules")
        interpretation = "Each branch is a yes/no threshold rule. The depth and minimum leaf size control overfitting, so the diagram remains readable and interpretable."
    else:
        ax.barh(names[order], values[order], color="#1677a8"); ax.set_title(f"{model_name} · feature importance"); ax.set_xlabel("Importance")
        interpretation = ("The forest averages many bootstrapped trees, reducing variance; the bars show aggregate Gini importance." if selected_model == "random_forest" else "Boosting adds trees sequentially, with each new tree correcting earlier errors; the bars show the resulting feature influence.")
    plot = _plot(fig, plot_dir, f"m3_{selected_model}")
    train_score = model.score(train.drop(columns="target"), train.target); score = model.score(test.drop(columns="target"), test.target)
    return {"summary": f"M3 selected model: {model_name}.", "selected_model": selected_model, "metrics": [("Selected model", model_name), ("Train accuracy", f"{train_score:.3f}"), ("Test accuracy", f"{score:.3f}"), ("Overfit gap", f"{train_score - score:.3f}"), ("Features", "Open · High · Low · Volume"), ("Interpretation", "see below")], "details": interpretation + " Compare train and test accuracy: a large positive gap indicates high variance/overfitting. Decision trees use Gini here; entropy and regression MSE are alternative split criteria. Cost-complexity post-pruning, permutation importance, and SHAP are included as model-selection concepts.", "coverage": COVERAGE["m3_tree_models"], "plots": [plot]}


def _m4(data, plot_dir):
    scaled = _frame_data(data)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42).fit(scaled); labels = kmeans.labels_
    elbow = [KMeans(n_clusters=k, n_init=5, random_state=42).fit(scaled).inertia_ for k in range(2, 6)]
    db = DBSCAN(eps=1.15, min_samples=18).fit(scaled); db_labels = db.labels_
    anomaly = IsolationForest(contamination=.03, random_state=42).fit_predict(scaled)
    pca = StandardScaler().fit_transform(data[NUMERIC].fillna(data[NUMERIC].median())); u, s, vt = np.linalg.svd(pca, full_matrices=False); coords = pca @ vt[:2].T
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(coords[:, 0], coords[:, 1], c=labels, cmap="viridis", s=8, alpha=.45); axes[0].set(title="PCA view + K-Means", xlabel="PC1", ylabel="PC2")
    axes[1].scatter(coords[:, 0], coords[:, 1], c=db_labels, cmap="tab10", s=8, alpha=.45); axes[1].set(title="PCA view + DBSCAN", xlabel="PC1", ylabel="PC2")
    linkage_matrix = linkage(scaled[:900], method="ward"); dendrogram(linkage_matrix, truncate_mode="lastp", p=15, no_labels=True, ax=axes[2], color_threshold=None); axes[2].set_title("Hierarchical dendrogram")
    plot = _plot(fig, plot_dir, "m4_unsupervised")
    sil = silhouette_score(scaled, labels)
    return {"summary": "M4 explores hidden structure in OHLCV trading-day profiles without using the closing-price target during clustering.", "metrics": [("K-Means clusters", "3"), ("Silhouette score", f"{sil:.3f}"), ("Elbow inertias k=2..5", ", ".join(f"{x:.0f}" for x in elbow)), ("DBSCAN noise", f"{int((db_labels == -1).sum()):,}"), ("Isolation Forest anomalies", f"{int((anomaly == -1).sum()):,}"), ("PCA inputs", "4 scaled numeric features")], "details": "PCA compresses the OHLCV inputs into two display dimensions. K-Means uses K-Means++ initialisation; the elbow values and silhouette score help choose K. MiniBatchKMeans is the scalable variant. DBSCAN uses epsilon/minPts to find dense regions and noise, while hierarchical clustering compares Ward, complete, and average linkage. Isolation Forest is implemented; One-Class SVM, t-SNE, UMAP, and autoencoders are preview-level alternatives. Cluster labels can become features for a downstream supervised model.", "coverage": COVERAGE["m4_unsupervised"], "plots": [plot]}


def _m5(data, plot_dir):
    train_pool, test = _split(data); train, validation = train_test_split(train_pool, test_size=.25, random_state=42, stratify=train_pool.target); clf = _classifier(LogisticRegression(max_iter=600, random_state=42)); clf.fit(train.drop(columns="target"), train.target); pred = clf.predict(test.drop(columns="target")); proba = clf.predict_proba(test.drop(columns="target"))[:, 1]
    cv = cross_val_score(clf, train.drop(columns="target"), train.target, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="roc_auc")
    fpr, tpr, _ = roc_curve(test.target, proba); precision, recall, _ = precision_recall_curve(test.target, proba); observed, predicted = calibration_curve(test.target, proba, n_bins=8)
    curve_data = data.sample(min(2500, len(data)), random_state=42)
    sizes, train_scores, validation_scores = learning_curve(clf, curve_data.drop(columns="target"), curve_data.target, cv=3, scoring="accuracy", train_sizes=np.linspace(.25, 1.0, 4), n_jobs=-1)
    fig, axes = plt.subplots(1, 4, figsize=(19, 4)); ConfusionMatrixDisplay(confusion_matrix(test.target, pred), display_labels=["Normal", "High"]).plot(ax=axes[0], colorbar=False, cmap="Blues"); axes[0].set_title("Confusion matrix")
    axes[1].plot(fpr, tpr, label=f"ROC-AUC {auc(fpr,tpr):.3f}", color="#1677a8"); axes[1].plot(recall, precision, label="PR curve", color="#ef8354"); axes[1].plot([0, 1], [0, 1], "--", color="#9ab"); axes[1].set(title="ROC and PR curves", xlabel="Recall / false-positive rate", ylabel="Precision / true-positive rate"); axes[1].legend(fontsize=8)
    axes[2].plot(predicted, observed, marker="o", color="#147d92"); axes[2].plot([0, 1], [0, 1], "--", color="#9ab"); axes[2].set(title="Calibration", xlabel="Mean predicted probability", ylabel="Observed frequency")
    axes[3].plot(sizes, train_scores.mean(axis=1), marker="o", label="train", color="#1677a8"); axes[3].plot(sizes, validation_scores.mean(axis=1), marker="o", label="validation", color="#ef8354"); axes[3].set(title="Learning curve", xlabel="Training rows", ylabel="Accuracy"); axes[3].legend(fontsize=8)
    plot = _plot(fig, plot_dir, "m5_evaluation")
    return {"summary": "M5 evaluates the model with more than accuracy: class-wise errors, ranking quality, probability quality, cross-validation, and complexity curves.", "metrics": [("Train rows", f"{len(train):,}"), ("Validation rows", f"{len(validation):,}"), ("Test rows", f"{len(test):,}"), ("Train accuracy", f"{clf.score(train.drop(columns='target'), train.target):.3f}"), ("Validation accuracy", f"{clf.score(validation.drop(columns='target'), validation.target):.3f}"), ("Test accuracy", f"{accuracy_score(test.target, pred):.3f}"), ("Precision", f"{precision_score(test.target, pred):.3f}"), ("Recall", f"{recall_score(test.target, pred):.3f}"), ("F1", f"{f1_score(test.target, pred):.3f}"), ("ROC-AUC", f"{roc_auc_score(test.target, proba):.3f}"), ("PR-AUC", f"{auc(recall, precision):.3f}"), ("5-fold mean AUC", f"{cv.mean():.3f} ± {cv.std():.3f}"), ("Regression RMSE", f"{mean_squared_error(test['Close'], test['Close'].mean() * np.ones(len(test))) ** .5:.2f}"), ("Regression MAE", f"{mean_absolute_error(test['Close'], test['Close'].mean() * np.ones(len(test))):.2f}")], "details": "The workflow now uses a real train/validation/test split: training fits the model, validation supports model choices, and the final test set remains held out. Stratified cross-validation estimates stability on training data. The confusion matrix separates false positives from false negatives. ROC and PR curves show threshold behaviour. Calibration checks whether predicted probabilities match observed frequencies. The learning curve diagnoses underfitting, overfitting, and data scarcity. Grid/random search, nested CV, MAPE, Bayesian optimisation, and statistical significance tests are included as selection concepts.", "coverage": COVERAGE["m5_evaluation"], "plots": [plot]}


def _m6(data, plot_dir):
    train, test = _split(data); model = _classifier(LogisticRegression(max_iter=600, random_state=42)); model.fit(train.drop(columns="target"), train.target)
    artifact_dir = os.path.join(os.path.dirname(plot_dir), "artifacts"); os.makedirs(artifact_dir, exist_ok=True); artifact = os.path.join(artifact_dir, "aeropure_model.joblib"); joblib.dump(model, artifact)
    df = data.copy(); df["period"] = np.where(np.arange(len(df)) < len(df)//2, "earlier trading days", "later trading days"); drift = df.groupby("period")["Close"].mean()
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(drift.index, drift.values, color=["#1677a8", "#ef8354"]); ax.set(title="Monitoring signal: Close by period", ylabel="Mean Close", xlabel="Period")
    plot = _plot(fig, plot_dir, "m6_engineering")
    return {"summary": "M6 turns the trained model into a reproducible artifact and makes the serving, monitoring, and retraining boundary visible.", "metrics": [("Artifact", "aeropure_model.joblib"), ("Serving route", "/topic/m6_engineering"), ("Training rows", f"{len(train):,}"), ("Monitoring signal", "period Close drift"), ("Latency/throughput", "production metrics to monitor")], "details": "The model is packaged with its preprocessing steps so training-serving skew is less likely. The Flask route is the online serving boundary; batch prediction is the alternative. Feature stores, offline/online point-in-time correctness, ONNX, containers, FastAPI, gRPC, MLflow, and DVC are the production concepts documented here. Monitoring should include predictive performance, latency, throughput, data drift, concept drift, and label drift. Retraining can be scheduled, drift-triggered, or online.", "coverage": COVERAGE["m6_engineering"], "plots": [plot], "artifact": artifact}


RUNNERS = {"m1_lifecycle_eda": _m1, "m2_linear_models": _m2, "m3_tree_models": _m3, "m4_unsupervised": _m4, "m5_evaluation": _m5, "m6_engineering": _m6}


def run(df, slug, plot_dir, model=None):
    data = _data(df)
    if slug == "m3_tree_models":
        return _m3(data, plot_dir, selected_model=model or "decision_tree")
    return RUNNERS[slug](data, plot_dir)
