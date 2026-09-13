import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv") #data

def simple_regression(target, feature, title):
    X = df[feature].values #feature
    y = df[target].values #target

    #standardization
    X = (X - X.mean()) / X.std()

    m = len(X)
    w = 0.0  # slope
    b = 0.0  # intercept

    learning_rate = 0.01
    epochs = 1000

    #plot using gradient descent
    for _ in range(epochs):
        y_pred = w * X + b
        dw = (-2 / m) * np.sum(X * (y - y_pred))
        db = (-2 / m) * np.sum(y - y_pred)

        w -= learning_rate * dw
        b -= learning_rate * db

    print(f"Reg coefficient: {w:.4f}")
    print(f"Intercept: {b:.4f}")

    #linear prediction
    y_pred = w * X + b

    #plotting of the graph
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, alpha=0.5, label="Data")
    plt.plot(X, y_pred, color="red", label=f"Predicted {target}")
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(title)
    plt.legend()
    plt.show()

#simple_regression(target = "critical_temp", feature = "wtd_mean_Valence", title = "Linreg (weak predictor)")
#simple_regression(target = "critical_temp", feature = "wtd_std_ThermalConductivity", title = "Linreg (strong predictor)")