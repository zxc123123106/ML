import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "project-matplotlib-cache")
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import shap
except ImportError:
    shap = None


TARGET_COL = "death_count"
DEFAULT_EXCLUDED_COLS = ("date", "pond_id")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, output_path):
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_feature_columns(path):
    path = Path(path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def prepare_features_and_target(
    df,
    target_col=TARGET_COL,
    excluded_cols=DEFAULT_EXCLUDED_COLS,
    feature_columns=None,
    drop_missing_target=True,
):
    """
    Splits a processed feature table into model-ready numeric features and target.
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    model_df = df.copy()
    if drop_missing_target:
        model_df = model_df[model_df[target_col].notna()].copy()

    y = model_df[target_col].astype(float)
    excluded = set(excluded_cols) | {target_col}
    candidate_cols = [col for col in model_df.columns if col not in excluded]
    X = model_df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    X = X.select_dtypes(include=[np.number])

    if feature_columns is not None:
        for col in feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[list(feature_columns)]

    return X, y


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "n_samples": int(len(y_true)),
    }


def predict_non_negative(model, X):
    return np.maximum(model.predict(X), 0)


def get_model_feature_columns(model):
    booster = model.get_booster()
    if booster.feature_names:
        return list(booster.feature_names)

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names)

    return None


def plot_prediction_vs_actual(y_true, y_pred, output_path):
    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.75, edgecolor="black", linewidth=0.4)

    upper = max(float(np.max(y_true)), float(np.max(y_pred)), 1.0)
    ax.plot([0, upper], [0, upper], color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Actual death_count")
    ax.set_ylabel("Predicted death_count")
    ax.set_title("Prediction vs Actual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_residuals(y_true, y_pred, output_path):
    ensure_dir(Path(output_path).parent)
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.scatter(y_pred, residuals, alpha=0.75, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Predicted death_count")
    ax.set_ylabel("Residual actual - predicted")
    ax.set_title("Residual Analysis")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(model, feature_columns, output_path, top_n=20):
    ensure_dir(Path(output_path).parent)
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    top_df = importance_df.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.34 * len(top_df))))
    ax.barh(top_df["feature"], top_df["importance"], color="#2f6f9f")
    ax.set_xlabel("XGBoost gain-based importance")
    ax.set_title("Top Feature Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    return importance_df


def plot_learning_curve(evals_result, output_path):
    if not evals_result:
        return

    ensure_dir(Path(output_path).parent)
    fig, ax = plt.subplots(figsize=(8, 5))

    for data_name, metrics in evals_result.items():
        for metric_name, values in metrics.items():
            ax.plot(values, label=f"{data_name} {metric_name}")

    ax.set_xlabel("Boosting round")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_shap_summary(shap_values, feature_names, method):
    summary_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
        "method": method,
    }).sort_values("mean_abs_shap", ascending=False)

    return summary_df


def compute_shap_contributions(model, X):
    """
    Computes SHAP contributions, preferring the shap package and falling back to
    XGBoost's native pred_contribs when shap is unavailable or incompatible.
    """
    if shap is not None:
        try:
            explainer = shap.TreeExplainer(model)
            raw_values = explainer.shap_values(X)
            if isinstance(raw_values, list):
                raw_values = raw_values[0]

            shap_values = np.asarray(raw_values)
            contribution_df = pd.DataFrame(shap_values, columns=X.columns, index=X.index)
            summary_df = _build_shap_summary(
                shap_values,
                X.columns,
                method="shap.TreeExplainer",
            )
            return contribution_df, summary_df
        except Exception as exc:
            print(f"Warning: shap.TreeExplainer failed; falling back to XGBoost pred_contribs. Reason: {exc}")

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=list(X.columns), missing=np.nan)
    contributions = booster.predict(dmatrix, pred_contribs=True)
    feature_contribs = contributions[:, :-1]

    contribution_df = pd.DataFrame(feature_contribs, columns=X.columns, index=X.index)
    summary_df = _build_shap_summary(
        feature_contribs,
        X.columns,
        method="xgboost.pred_contribs",
    )

    return contribution_df, summary_df


def plot_shap_summary(summary_df, output_path, top_n=20):
    ensure_dir(Path(output_path).parent)
    top_df = summary_df.head(top_n).sort_values("mean_abs_shap")

    fig, ax = plt.subplots(figsize=(9, max(5, 0.34 * len(top_df))))
    ax.barh(top_df["feature"], top_df["mean_abs_shap"], color="#587246")
    ax.set_xlabel("Mean absolute SHAP contribution")
    ax.set_title("SHAP Feature Contribution Summary")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_top_feature_shap_dependence(X, shap_values, summary_df, output_path):
    ensure_dir(Path(output_path).parent)
    if summary_df.empty:
        return None

    top_feature = summary_df.iloc[0]["feature"]
    feature_values = X[top_feature]
    contributions = shap_values[top_feature]
    finite_mask = feature_values.notna() & np.isfinite(contributions)

    if finite_mask.sum() == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        feature_values[finite_mask],
        contributions[finite_mask],
        alpha=0.75,
        edgecolor="black",
        linewidth=0.4,
        color="#6f5f90",
    )
    ax.axhline(0, color="#d62728", linestyle="--", linewidth=1.2)
    ax.set_xlabel(top_feature)
    ax.set_ylabel("SHAP contribution")
    ax.set_title(f"SHAP Dependence: {top_feature}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    positive_values = feature_values[finite_mask & (contributions > 0)]
    threshold = float(positive_values.min()) if len(positive_values) else None
    return {
        "feature": top_feature,
        "approx_positive_contribution_threshold": threshold,
        "method": "Minimum observed value with positive SHAP contribution.",
    }


def evaluate_model(
    model,
    df,
    figure_dir="reports/figures",
    report_dir="reports",
    feature_columns=None,
    prefix="",
):
    """
    Evaluates a trained model and writes report artifacts.
    """
    ensure_dir(report_dir)
    ensure_dir(figure_dir)

    feature_columns = feature_columns or get_model_feature_columns(model)
    X, y = prepare_features_and_target(df, feature_columns=feature_columns)
    y_pred = predict_non_negative(model, X)
    metrics = calculate_metrics(y, y_pred)

    file_prefix = f"{prefix}_" if prefix else ""
    plot_prediction_vs_actual(
        y,
        y_pred,
        Path(figure_dir) / f"{file_prefix}prediction_vs_actual.png",
    )
    plot_residuals(y, y_pred, Path(figure_dir) / f"{file_prefix}residuals.png")

    importance_df = plot_feature_importance(
        model,
        list(X.columns),
        Path(figure_dir) / f"{file_prefix}feature_importance.png",
    )
    importance_df.to_csv(Path(report_dir) / f"{file_prefix}feature_importance.csv", index=False)

    shap_values, shap_summary = compute_shap_contributions(model, X)
    shap_summary.to_csv(Path(report_dir) / f"{file_prefix}shap_feature_contributions.csv", index=False)
    plot_shap_summary(
        shap_summary,
        Path(figure_dir) / f"{file_prefix}shap_summary.png",
    )

    threshold_info = plot_top_feature_shap_dependence(
        X,
        shap_values,
        shap_summary,
        Path(figure_dir) / f"{file_prefix}top_feature_shap_dependence.png",
    )
    if threshold_info:
        pd.DataFrame([threshold_info]).to_csv(
            Path(report_dir) / f"{file_prefix}shap_thresholds.csv",
            index=False,
        )

    return {
        "metrics": metrics,
        "predictions": pd.DataFrame({
            "actual": y.to_numpy(),
            "predicted": y_pred,
            "residual": y.to_numpy() - y_pred,
        }),
        "feature_importance": importance_df,
        "shap_summary": shap_summary,
    }


def load_model(model_path):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained XGBoost mortality model.")
    parser.add_argument("--model-path", default="models/final_best_model.json")
    parser.add_argument("--test-path", default="data/processed/test_set.csv")
    parser.add_argument("--feature-columns-path", default="models/feature_columns.json")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--figure-dir", default="reports/figures")
    args = parser.parse_args()

    model = load_model(args.model_path)
    feature_columns = load_feature_columns(args.feature_columns_path)
    test_df = pd.read_csv(args.test_path)
    result = evaluate_model(
        model,
        test_df,
        figure_dir=args.figure_dir,
        report_dir=args.report_dir,
        feature_columns=feature_columns,
    )
    save_json(result["metrics"], Path(args.report_dir) / "test_metrics.json")
    result["predictions"].to_csv(Path(args.report_dir) / "test_predictions.csv", index=False)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
