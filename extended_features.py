import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("train.csv")

target = "critical_temp"

selected_features = [
    "entropy_fie",
    "entropy_atomic_radius",
    "mean_Valence",
    "wtd_mean_Valence",
    "wtd_gmean_fie",
    "std_Valence"
]

X = df[selected_features]
y = df[target]
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
mse_before = []
rmse_before = []
r2_before = []

mse_after = []
rmse_after = []
r2_after = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    y_pred = linear_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mse_before.append(mse)
    rmse_before.append(rmse)
    r2_before.append(r2)
    poly = PolynomialFeatures(
        degree=2,
        include_bias=False
    )

    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    poly_model = LinearRegression()

    poly_model.fit(X_train_poly, y_train)

    y_pred_poly = poly_model.predict(X_test_poly)

    mse_poly = mean_squared_error(y_test, y_pred_poly)
    rmse_poly = np.sqrt(mse_poly)
    r2_poly = r2_score(y_test, y_pred_poly)

    mse_after.append(mse_poly)
    rmse_after.append(rmse_poly)
    r2_after.append(r2_poly)
print("\n\n")
print("=" * 60)
print("BEFORE FEATURE EXPANSION")
print("=" * 60)

print(f"Mean MSE      : {np.mean(mse_before):.4f}")
print(f"Variance MSE  : {np.var(mse_before, ddof=1):.4f}")

print(f"Mean RMSE     : {np.mean(rmse_before):.4f}")
print(f"Variance RMSE : {np.var(rmse_before, ddof=1):.4f}")

print(f"Mean R²       : {np.mean(r2_before):.4f}")
print(f"Variance R²   : {np.var(r2_before, ddof=1):.4f}")

print("\n")
print("=" * 60)
print("AFTER FEATURE EXPANSION")
print("=" * 60)

print(f"Mean MSE      : {np.mean(mse_after):.4f}")
print(f"Variance MSE  : {np.var(mse_after, ddof=1):.4f}")

print(f"Mean RMSE     : {np.mean(rmse_after):.4f}")
print(f"Variance RMSE : {np.var(rmse_after, ddof=1):.4f}")

print(f"Mean R²       : {np.mean(r2_after):.4f}")
print(f"Variance R²   : {np.var(r2_after, ddof=1):.4f}")
mse_reduction = (
    (np.mean(mse_before) - np.mean(mse_after))
    / np.mean(mse_before)
) * 100

rmse_reduction = (
    (np.mean(rmse_before) - np.mean(rmse_after))
    / np.mean(rmse_before)
) * 100

r2_improvement = (
    (np.mean(r2_after) - np.mean(r2_before))
    / np.mean(r2_before)
) * 100

print("\n")
print("=" * 60)
print("PERFORMANCE CHANGE")
print("=" * 60)

print(f"MSE Reduction      : {mse_reduction:.2f}%")
print(f"RMSE Reduction     : {rmse_reduction:.2f}%")
print(f"R² Improvement     : {r2_improvement:.2f}%")
poly = PolynomialFeatures(
    degree=2,
    include_bias=False
)

poly.fit(X)

feature_names = poly.get_feature_names_out(selected_features)

print("\n")
print("=" * 60)
print("FEATURE SPACE EXPANSION")
print("=" * 60)

print(f"Original Features : {len(selected_features)}")
print(f"Expanded Features : {len(feature_names)}")

print("\nGenerated Features:")
for feature in feature_names:
    print(feature)