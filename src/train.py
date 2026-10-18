"""
train.py: forecasts hourly departures and arrivals per station.

For each target it trains:
- seasonal naive baseline: "same hour last week"
- linear ridge regression
- gradient boosting, tuned with Bayesian optimization (Optuna)

Split by TIME: train April to September, test October and November.

Run:  python -m src.train              (20 tuning trials per target)
      python -m src.train --trials 0   (skip tuning, fast first run)
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATASET, FEATURES, METRICS_FILE, MODELS_DIR, RANDOM_STATE, TARGETS,
    TEST_PREDICTIONS, TEST_START, VALID_START,
)

CATEGORICAL = ["station_id"]  # boosting treats this as a category, not a number

DEFAULT_PARAMS = {"learning_rate": 0.08, "max_leaf_nodes": 63, "min_samples_leaf": 40,
                  "l2_regularization": 1.0, "max_iter": 500}


def load_dataset():
    return pd.read_csv(DATASET, parse_dates=["timestamp"])


def time_split(df):
    train = df[df["timestamp"] < TEST_START]
    test = df[df["timestamp"] >= TEST_START]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(y_true, y_pred):
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def make_gbm(params):
    """Gradient boosting with Poisson loss, designed for counts (0, 1, 2 ... trips)."""
    return HistGradientBoostingRegressor(
        loss="poisson",
        categorical_features=CATEGORICAL,
        early_stopping=False,          # its built in split is random, which would leak time
        random_state=RANDOM_STATE,
        **params,
    )


def make_ridge():
    """Linear model. Hour, weekday and station become one hot columns; numbers are scaled."""
    one_hot = ["station_id", "hour", "dow"]
    numeric = [c for c in FEATURES if c not in one_hot]
    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), one_hot),
        ("scale", StandardScaler(), numeric),
    ])
    return make_pipeline(pre, Ridge(alpha=1.0))


def tune(train, target, n_trials):
    """Bayesian optimization with Optuna. Trains before VALID_START, scores on the
    3 weeks after, so tuning never touches the test set."""
    if n_trials == 0:
        return DEFAULT_PARAMS
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fit = train[train["timestamp"] < VALID_START]
    val = train[train["timestamp"] >= VALID_START]

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 5.0),
            "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
        }
        model = make_gbm(params).fit(fit[FEATURES], fit[target])
        return mean_absolute_error(val[target], model.predict(val[FEATURES]))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  best validation MAE {study.best_value:.3f} with {study.best_params}")
    return study.best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials per target (0 = skip)")
    args = parser.parse_args()

    df = load_dataset()
    train, test = time_split(df)
    print(f"train: {len(train):,} rows until {TEST_START}, test: {len(test):,} rows after")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = test[["timestamp", "station_id"] + TARGETS].copy()
    metrics = {}

    for target in TARGETS:
        print(f"\n=== {target} ===")
        results = {}
        results["seasonal_naive"] = evaluate(test[target], test[f"{target}_lag_168"])

        ridge = make_ridge().fit(train[FEATURES], train[target])
        results["linear_ridge"] = evaluate(test[target], ridge.predict(test[FEATURES]).clip(0))

        params = tune(train, target, args.trials)
        gbm = make_gbm(params).fit(train[FEATURES], train[target])
        pred = gbm.predict(test[FEATURES])
        results["gradient_boosting"] = evaluate(test[target], pred)
        results["gradient_boosting"]["params"] = params

        predictions[f"pred_{target}"] = pred
        joblib.dump({"model": gbm, "features": FEATURES, "target": target},
                    MODELS_DIR / f"{target}_model.joblib")
        metrics[target] = results

        print(f"{'model':<20}{'MAE':>8}{'RMSE':>8}{'R2':>8}")
        for name, m in results.items():
            print(f"{name:<20}{m['mae']:>8}{m['rmse']:>8}{m['r2']:>8}")

    predictions.to_csv(TEST_PREDICTIONS, index=False)
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved models, {METRICS_FILE.name} and test predictions")


if __name__ == "__main__":
    main()