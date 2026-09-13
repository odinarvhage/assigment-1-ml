import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("train.csv") #data

def train_test_split(feature, target):
    X = df[feature].values
    y = df[target].values
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    r2_scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        train_mean = X_train.mean()
        train_std = X_train.std() #standardization

        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        w = 0.0
        b = 0.0
        learning_rate = 0.01
        epochs = 1000

        m = len(X_train)
        for _ in range(epochs):
            y_pred = w * X_train + b
            dw = (-2 / m) * np.sum(X_train * (y_train - y_pred))
            db = (-2 / m) * np.sum(y_train - y_pred)
            w -= learning_rate * dw
            b -= learning_rate * db


        y_test_pred = w * X_test + b
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_test_pred)
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"\nFold {fold}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

    print("Mean and variance")
    print(f"MSE mean     : {np.mean(mse_scores):.4f}")
    print(f"MSE variance : {np.var(mse_scores):.4f}")
    print(f"RMSE mean     : {np.mean(rmse_scores):.4f}")
    print(f"RMSE variance : {np.var(rmse_scores):.4f}")
    print(f"R² mean     : {np.mean(r2_scores):.4f}")
    print(f"R² variance : {np.var(r2_scores):.4f}")


#train_test_split("wtd_std_ThermalConductivity", "critical_temp") #strong predictor
#train_test_split("wtd_mean_Valence","critical_temp") #weak predictor