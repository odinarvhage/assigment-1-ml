import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("train.csv")
def top_correlations(data, target, amount):
    correlation_matrix = data.corr(numeric_only=True)
    target_correlations = correlation_matrix[target].drop(target)
    top = (
        target_correlations.abs()
        .sort_values(ascending=False)
        .head(amount)
    )

    print(f"Top features most correlated with {target}:\n")
    for feature in top.index:
        corr = target_correlations[feature]
        print(f"{feature:<35} {corr:.4f}")


def regression_coeffient(data, target, amount):
    X = data.drop(columns=[target])
    y = data[target]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Feature importance ranking
    importance = pd.Series(
        np.abs(model.coef_),
        index=X.columns
    ).sort_values(ascending=False)
    print("Highest regression coefficients\n")
    print(importance.head(amount))

def polynomial_regression(data, features):
    X = data[features]
    poly = PolynomialFeatures(
        degree=2,
        include_bias=False
    )
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(features)
    X_poly = pd.DataFrame(
        X_poly,
        columns=feature_names
    )

    print(X_poly.head())
    print("\nTotal features:", len(feature_names))

#top_correlations(data, "critical_temp", 20)
#regression_coeffient(data, "critical_temp",20)
#polynomial_regression(data,["wtd_std_ThermalConductivity","range_ThermalConductivity","entropy_fie","wtd_mean_Valence","entropy_Valence","gmean_Valence"])