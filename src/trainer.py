import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from xgboost import XGBRegressor

try:
    import optuna
except ImportError:
    optuna = None
else:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from src.evaluator import (
        calculate_metrics,
        ensure_dir,
        evaluate_model,
        plot_learning_curve,
        prepare_features_and_target,
        predict_non_negative,
        save_json,
    )
except ModuleNotFoundError:
    from evaluator import (
        calculate_metrics,
        ensure_dir,
        evaluate_model,
        plot_learning_curve,
        prepare_features_and_target,
        predict_non_negative,
        save_json,
    )


DEFAULT_TRAIN_PATH = "data/processed/train_set.csv"
DEFAULT_TEST_PATH = "data/processed/test_set.csv"
DEFAULT_MODEL_DIR = "models"
DEFAULT_REPORT_DIR = "reports"
DEFAULT_FIGURE_DIR = "reports/figures"


BASE_MODEL_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "verbosity": 0,
}


FALLBACK_PARAM_SPACE = {
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.12],
    "n_estimators": [80, 120, 180, 260, 360, 500],
    "subsample": [0.65, 0.75, 0.85, 0.95, 1.0],
    "colsample_bytree": [0.65, 0.75, 0.85, 0.95, 1.0],
    "min_child_weight": [1, 2, 4, 6, 8],
    "gamma": [0, 0.05, 0.1, 0.25, 0.5, 1.0],
    "reg_alpha": [0, 0.01, 0.05, 0.1, 0.5],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
}


def _model_params(params, random_state, early_stopping_rounds=None):
    model_params = {
        **BASE_MODEL_PARAMS,
        **params,
        "random_state": random_state,
        "n_jobs": 2,
    }
    if early_stopping_rounds:
        model_params["early_stopping_rounds"] = early_stopping_rounds

    return model_params


def fit_xgb_model(
    params,
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
    random_state=42,
    early_stopping_rounds=30,
):
    has_validation = X_valid is not None and y_valid is not None and len(X_valid) > 0
    model = XGBRegressor(
        **_model_params(
            params,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds if has_validation else None,
        )
    )

    if has_validation:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def make_time_series_splitter(n_samples, requested_splits):
    if n_samples < 3:
        raise ValueError("At least 3 labeled samples are required for time-series validation.")

    n_splits = min(requested_splits, n_samples - 1)
    n_splits = max(2, n_splits)
    return TimeSeriesSplit(n_splits=n_splits)


def cross_validate_params(
    params,
    X,
    y,
    n_splits=4,
    random_state=42,
    early_stopping_rounds=30,
):
    splitter = make_time_series_splitter(len(X), n_splits)
    fold_metrics = []

    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

        model = fit_xgb_model(
            params,
            X_train,
            y_train,
            X_valid,
            y_valid,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
        y_pred = predict_non_negative(model, X_valid)
        metrics = calculate_metrics(y_valid, y_pred)
        metrics["fold"] = fold_id
        fold_metrics.append(metrics)

    mean_rmse = float(np.mean([fold["rmse"] for fold in fold_metrics]))
    mean_mae = float(np.mean([fold["mae"] for fold in fold_metrics]))
    mean_r2 = float(np.nanmean([fold["r2"] for fold in fold_metrics]))

    return {
        "mean_rmse": mean_rmse,
        "mean_mae": mean_mae,
        "mean_r2": mean_r2,
        "folds": fold_metrics,
    }


def suggest_optuna_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 80, 700),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
    }


def tune_with_optuna(X, y, n_trials, n_splits, random_state, early_stopping_rounds):
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial):
        params = suggest_optuna_params(trial)
        cv_result = cross_validate_params(
            params,
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
        trial.set_user_attr("mean_mae", cv_result["mean_mae"])
        trial.set_user_attr("mean_r2", cv_result["mean_r2"])
        return cv_result["mean_rmse"]

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    trials_df = study.trials_dataframe()

    return study.best_params, {
        "method": "optuna",
        "best_rmse": float(study.best_value),
        "trials": trials_df,
    }


def tune_with_fallback_search(X, y, n_trials, n_splits, random_state, early_stopping_rounds):
    sampler = ParameterSampler(
        FALLBACK_PARAM_SPACE,
        n_iter=n_trials,
        random_state=random_state,
    )
    rows = []
    best_params = None
    best_rmse = float("inf")

    for trial_id, params in enumerate(sampler, start=1):
        cv_result = cross_validate_params(
            params,
            X,
            y,
            n_splits=n_splits,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )
        row = {
            "trial": trial_id,
            "mean_rmse": cv_result["mean_rmse"],
            "mean_mae": cv_result["mean_mae"],
            "mean_r2": cv_result["mean_r2"],
            **params,
        }
        rows.append(row)

        if cv_result["mean_rmse"] < best_rmse:
            best_rmse = cv_result["mean_rmse"]
            best_params = params

    return best_params, {
        "method": "fallback_parameter_sampler",
        "best_rmse": best_rmse,
        "trials": pd.DataFrame(rows).sort_values("mean_rmse"),
    }


def tune_hyperparameters(
    X,
    y,
    n_trials=25,
    n_splits=4,
    random_state=42,
    early_stopping_rounds=30,
):
    if optuna is not None:
        return tune_with_optuna(
            X,
            y,
            n_trials=n_trials,
            n_splits=n_splits,
            random_state=random_state,
            early_stopping_rounds=early_stopping_rounds,
        )

    return tune_with_fallback_search(
        X,
        y,
        n_trials=n_trials,
        n_splits=n_splits,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )


def make_final_validation_split(X, y, validation_ratio=0.2):
    if len(X) < 5:
        return X, y, None, None

    split_idx = max(1, int(len(X) * (1 - validation_ratio)))
    split_idx = min(split_idx, len(X) - 1)
    return (
        X.iloc[:split_idx],
        y.iloc[:split_idx],
        X.iloc[split_idx:],
        y.iloc[split_idx:],
    )


def train_final_model(
    best_params,
    X,
    y,
    random_state=42,
    early_stopping_rounds=30,
):
    X_fit, y_fit, X_valid, y_valid = make_final_validation_split(X, y)
    early_model = fit_xgb_model(
        best_params,
        X_fit,
        y_fit,
        X_valid,
        y_valid,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )

    best_iteration = getattr(early_model, "best_iteration", None)
    final_params = dict(best_params)
    if best_iteration is not None:
        final_params["n_estimators"] = max(1, int(best_iteration) + 1)

    final_model = fit_xgb_model(
        final_params,
        X,
        y,
        random_state=random_state,
        early_stopping_rounds=None,
    )

    return final_model, early_model, final_params


def train_and_evaluate(
    train_path=DEFAULT_TRAIN_PATH,
    test_path=DEFAULT_TEST_PATH,
    model_dir=DEFAULT_MODEL_DIR,
    report_dir=DEFAULT_REPORT_DIR,
    figure_dir=DEFAULT_FIGURE_DIR,
    n_trials=25,
    n_splits=4,
    random_state=42,
    early_stopping_rounds=30,
):
    ensure_dir(model_dir)
    ensure_dir(report_dir)
    ensure_dir(figure_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "date" in train_df.columns:
        train_df["date"] = pd.to_datetime(train_df["date"])
        train_df = train_df.sort_values(["date", "pond_id"])
    if "date" in test_df.columns:
        test_df["date"] = pd.to_datetime(test_df["date"])
        test_df = test_df.sort_values(["date", "pond_id"])

    X_train, y_train = prepare_features_and_target(train_df)
    X_test, y_test = prepare_features_and_target(
        test_df,
        feature_columns=list(X_train.columns),
    )

    best_params, tuning_result = tune_hyperparameters(
        X_train,
        y_train,
        n_trials=n_trials,
        n_splits=n_splits,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )

    tuning_result["trials"].to_csv(Path(report_dir) / "cv_results.csv", index=False)

    model, early_model, final_params = train_final_model(
        best_params,
        X_train,
        y_train,
        random_state=random_state,
        early_stopping_rounds=early_stopping_rounds,
    )

    model_path = Path(model_dir) / "final_best_model.json"
    feature_columns_path = Path(model_dir) / "feature_columns.json"
    model.save_model(model_path)
    save_json(list(X_train.columns), feature_columns_path)
    save_json(final_params, Path(report_dir) / "best_params.json")

    train_pred = predict_non_negative(model, X_train)
    test_pred = predict_non_negative(model, X_test)
    train_metrics = calculate_metrics(y_train, train_pred)
    test_metrics = calculate_metrics(y_test, test_pred)

    eval_result = evaluate_model(
        model,
        test_df,
        figure_dir=figure_dir,
        report_dir=report_dir,
        feature_columns=list(X_train.columns),
    )
    eval_result["predictions"].to_csv(Path(report_dir) / "test_predictions.csv", index=False)

    plot_learning_curve(
        early_model.evals_result() if hasattr(early_model, "evals_result") else None,
        Path(figure_dir) / "learning_curve.png",
    )

    metrics_payload = {
        "tuning_method": tuning_result["method"],
        "cv_best_rmse": float(tuning_result["best_rmse"]),
        "train": train_metrics,
        "test": test_metrics,
        "labeled_train_samples": int(len(X_train)),
        "labeled_test_samples": int(len(X_test)),
        "model_path": str(model_path),
    }
    save_json(metrics_payload, Path(report_dir) / "metrics.json")

    return {
        "model": model,
        "best_params": final_params,
        "metrics": metrics_payload,
        "model_path": model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate an XGBoost mortality model.")
    parser.add_argument("--train-path", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test-path", default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    args = parser.parse_args()

    result = train_and_evaluate(
        train_path=args.train_path,
        test_path=args.test_path,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        figure_dir=args.figure_dir,
        n_trials=args.n_trials,
        n_splits=args.n_splits,
        random_state=args.random_state,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
