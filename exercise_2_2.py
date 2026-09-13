import pandas as pd
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("train.csv")

def ridge(data):
    import pandas as pd
    import numpy as np

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score

    # Load data
    df = pd.read_csv("train.csv")

    # Features and target
    X = df.drop(columns=["critical_temp"])
    y = df["critical_temp"]

    # Candidate regularization strengths
    alphas = np.logspace(-3, 3, 50)

    # Outer 5-fold CV for model evaluation
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

        # Standardize using training data only
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # RidgeCV selects the best alpha via internal CV
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
ridge(df)