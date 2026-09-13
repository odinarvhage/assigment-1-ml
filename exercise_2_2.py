from sklearn.linear_model import LassoCV
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("train.csv")

def ridge(data):
    X = data.drop(columns=["critical_temp"])
    y = data["critical_temp"]
    alphas = np.logspace(-3, 3, 50)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mse_scores = []
    rmse_scores = []
    r2_scores = []
    best_alphas = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        ridge = RidgeCV(
            alphas=alphas,
            cv=5,
            scoring="r2"
        )

        ridge.fit(X_train_scaled, y_train)

        y_pred = ridge.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        best_alphas.append(ridge.alpha_)

        print(f"\nFold {fold}")
        print(f"Best alpha: {ridge.alpha_:.4f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
    # Train final model on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ridge_final = RidgeCV(
        alphas=alphas,
        cv=5
    )

    ridge_final.fit(X_scaled, y)

    importance = pd.Series(
        np.abs(ridge_final.coef_),
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 Ridge Features:")
    print(importance.head(10))

def lasso(data):
    X = data.drop(columns=["critical_temp"])
    y = data["critical_temp"]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mse_scores = []
    rmse_scores = []
    r2_scores = []
    best_alphas = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Standardize features
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # LASSO with internal CV for alpha selection
        lasso = LassoCV(
            cv=5,
            random_state=42,
            max_iter=10000
        )

        lasso.fit(X_train_scaled, y_train)

        # Test predictions
        y_pred = lasso.predict(X_test_scaled)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)
        best_alphas.append(lasso.alpha_)

        # Feature elimination analysis
        coefficients = pd.Series(
            lasso.coef_,
            index=X.columns
        )

        eliminated_features = coefficients[
            coefficients == 0
            ].index.tolist()

        selected_features = coefficients[
            coefficients != 0
            ].sort_values(key=np.abs, ascending=False)

        print(f"\n{'=' * 50}")
        print(f"Fold {fold}")
        print(f"{'=' * 50}")

        print(f"Best alpha: {lasso.alpha_:.6f}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")

        print(f"\nNumber of selected features: {len(selected_features)}")
        print(f"Number of eliminated features: {len(eliminated_features)}")

        print("\nEliminated Features:")
        for feature in eliminated_features:
            print(feature)

    # Summary metrics
    print(f"\n{'=' * 50}")
    print("OVERALL RESULTS")
    print(f"{'=' * 50}")

    print(f"Mean MSE:  {np.mean(mse_scores):.4f}")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Mean R²:   {np.mean(r2_scores):.4f}")

    print(f"\nMean Alpha: {np.mean(best_alphas):.6f}")


def random_forest(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    r2_scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Train model
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"\nFold {fold}")
        print(f"MSE  = {mse:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"R²   = {r2:.4f}")

    print("\n========== Overall Results ==========")
    print(f"Mean MSE  : {np.mean(mse_scores):.4f}")
    print(f"Mean RMSE : {np.mean(rmse_scores):.4f}")
    print(f"Mean R²   : {np.mean(r2_scores):.4f}")

    print(f"\nMSE Variance  : {np.var(mse_scores, ddof=1):.4f}")
    print(f"RMSE Variance : {np.var(rmse_scores, ddof=1):.4f}")
    print(f"R² Variance   : {np.var(r2_scores, ddof=1):.4f}")
#ridge(df)
#lasso(df)
random_forest(df,"critical_temp")