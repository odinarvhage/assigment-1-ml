import pandas as pd


def top_ten_correlations(data, target):
    correlation_matrix = data.corr(numeric_only=True)
    target_correlations = correlation_matrix[target].drop(target)
    top_10 = (
        target_correlations.abs()
        .sort_values(ascending=False)
        .head(10)
    )

    print(f"Top 10 features most correlated with {target}:\n")
    for feature in top_10.index:
        corr = target_correlations[feature]
        print(f"{feature:<35} {corr:.4f}")


top_ten_correlations(pd.read_csv("train.csv"), "critical_temp")