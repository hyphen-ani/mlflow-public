## imports

import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import label
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn


def parse_args():
    p = argparse.ArgumentParser("MLFlow Session")
    p.add_argument("--csv", default="data/sample.csv")
    p.add_argument("--target", default="performance_score")
    p.add_argument("--experiment", default="student-performance-prediction-model")
    p.add_argument("--run", default="SPM:8")

    p.add_argument("--n-estimators", type=int, default=85)
    p.add_argument("--max-depth", type=int, default=80)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--random-state", type=int, default=100)

    return p.parse_args()


def main():

    # optional
    mlflow.sklearn.autolog()

    args = parse_args()

    ## MLFLOW INITIALIZATION
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:7006")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment)

    ## Load CSV

    if not os.path.exists(args.csv):
        raise SystemError(f"File {args.csv} does not exist")

    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise SystemError(f"Target column {args.target} does not exist")


    ## Prepare Data

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size,
                                                        random_state=args.random_state)

    ### Train and log with MLFlow

    with mlflow.start_run(run_name=args.run) as run:

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        ### Train Model

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
        )

        model.fit(X_train, y_train)


        ### Predict + Metric (Evaluation)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = float(math.sqrt(mse))
        r2 = float(r2_score(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))

        ### Log Metric

        mlflow.log_metric("Mean Squared Error", mse)
        mlflow.log_metric("Mean Absolute Error", mae)
        mlflow.log_metric("R2 Score", r2)
        mlflow.log_metric("Root-Mean Squared Error",rmse)

        ### Artifacts

        plt.figure()
        plt.scatter(y_test, preds, label="Predictions", alpha=0.7)
        plt.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 linestyle="--", color="red", label="Perfect Prediction")

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.legend()
        plt.grid(True)

        plot_path = "pred.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)


if __name__ == "__main__":
    main()