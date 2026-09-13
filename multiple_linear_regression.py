import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def multiple_linear_regression(data, target):
    X = data.drop(columns=[target]).values
    y = data[target].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mse_scores = []
    rmse_scores = []
    r2_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1

        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        n_features = X_train.shape[1]

        w = np.zeros(n_features)
        b = 0

        learning_rate = 0.01
        epochs = 1000

        m = len(X_train)
        for _ in range(epochs):

            y_pred = np.dot(X_train, w) + b

            dw = (-2 / m) * np.dot(X_train.T, (y_train - y_pred))
            db = (-2 / m) * np.sum(y_train - y_pred)

            w -= learning_rate * dw
            b -= learning_rate * db
        y_test_pred = np.dot(X_test, w) + b
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_test_pred)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"\nFold {fold}")
        print(f"MSE  = {mse:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"R²   = {r2:.4f}")

    print(f"MSE mean     : {np.mean(mse_scores):.4f}")
    print(f"MSE variance : {np.var(mse_scores):.4f}")
    print(f"RMSE mean     : {np.mean(rmse_scores):.4f}")
    print(f"RMSE variance : {np.var(rmse_scores):.4f}")
    print(f"R² mean     : {np.mean(r2_scores):.4f}")
    print(f"R² variance : {np.var(r2_scores):.4f}")

def cost_per_iteration(df, target,simple_feature):
    X_simple = df[simple_feature].values
    y = df[target].values

    X_simple = (X_simple - X_simple.mean()) / X_simple.std() #standardization

    w_simple = 0.0
    b_simple = 0.0

    learning_rate = 0.01
    epochs = 1000

    m = len(X_simple)

    simple_costs = []

    for _ in range(epochs):
        y_pred = w_simple * X_simple + b_simple

        cost = np.mean((y - y_pred) ** 2)
        simple_costs.append(cost)

        dw = (-2 / m) * np.sum(X_simple * (y - y_pred))
        db = (-2 / m) * np.sum(y - y_pred)

        w_simple -= learning_rate * dw
        b_simple -= learning_rate * db

    #multiple regreession
    X_multi = df.drop(columns=[target]).values
    X_multi = (X_multi - X_multi.mean(axis=0)) / X_multi.std(axis=0)
    X_multi = np.nan_to_num(X_multi)

    n_features = X_multi.shape[1]

    w_multi = np.zeros(n_features)
    b_multi = 0.0

    multi_costs = []

    for _ in range(epochs):
        y_pred = np.dot(X_multi, w_multi) + b_multi

        cost = np.mean((y - y_pred) ** 2)
        multi_costs.append(cost)

        dw = (-2 / len(X_multi)) * np.dot(
            X_multi.T,
            (y - y_pred)
        )

        db = (-2 / len(X_multi)) * np.sum(y - y_pred)

        w_multi -= learning_rate * dw
        b_multi -= learning_rate * db

    plt.figure(figsize=(10, 6))

    plt.plot(
        simple_costs,
        label=f"Simple LR ({simple_feature})",
        linewidth=2
    )

    plt.plot(
        multi_costs,
        label="Multiple LR (All Features)",
        linewidth=2
    )

    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost vs Iteration Comparison")
    plt.legend()
    plt.grid(True)

    plt.show()

#multiple_linear_regression(pd.read_csv("train.csv"),"critical_temp")
#cost_per_iteration(pd.read_csv("train.csv"), "critical_temp", "wtd_std_ThermalConductivity")