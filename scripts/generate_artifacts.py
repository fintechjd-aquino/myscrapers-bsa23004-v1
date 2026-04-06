import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


def main():
    data_path = "data/listings_master_llm.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    for col in ["price", "year", "mileage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["vehicle_age"] = 2026 - df["year"]

    df = df[(df["price"] > 500) & (df["price"] < 100000)]
    df = df[(df["mileage"] > 0) & (df["mileage"] < 300000)]

    df["log_price"] = np.log1p(df["price"])

    features = ["mileage", "vehicle_age", "drivetrain", "state", "color"]
    df = df.dropna(subset=["log_price", "mileage", "vehicle_age"])

    X = df[features]
    y = df["log_price"]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    grid = GridSearchCV(
        Ridge(),
        param_grid={"alpha": [0.01, 0.1, 1, 10, 100]},
        cv=5
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    run_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    data_dir = f"artifacts/runs/{run_stamp}/data"
    plots_dir = f"artifacts/runs/{run_stamp}/plots"

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    predictions_df = pd.DataFrame({
        "actual_log_price": y_test.values,
        "predicted_log_price": y_pred,
        "actual_price": np.expm1(y_test.values),
        "predicted_price": np.expm1(y_pred)
    })
    predictions_df.to_csv(f"{data_dir}/predictions.csv", index=False)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    bias = np.mean(y_pred - y_test)

    metrics_df = pd.DataFrame([{
        "run_id": run_stamp,
        "model": "ridge_gridsearch",
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "bias": bias
    }])
    metrics_df.to_csv(f"{data_dir}/metrics_summary.csv", index=False)

    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=10, random_state=42
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)

    importance_df.to_csv(f"{data_dir}/feature_importance.csv", index=False)

    top3_features = importance_df["feature"].head(3).tolist()

    for feature in top3_features:
        fig, ax = plt.subplots(figsize=(8, 5))
        PartialDependenceDisplay.from_estimator(best_model, X_test, [feature], ax=ax)
        plt.title(f"PDP for {feature}")
        plt.tight_layout()

        safe_name = feature.replace("/", "_").replace(" ", "_")
        plt.savefig(f"{plots_dir}/pdp_{safe_name}.png", dpi=200, bbox_inches="tight")
        plt.close()

    print(f"Artifacts written to artifacts/runs/{run_stamp}/")


if __name__ == "__main__":
    main()
