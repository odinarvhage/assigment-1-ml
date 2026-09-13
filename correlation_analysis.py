import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")
target = "critical_temp"
def task_1():
    correlation_matrix = df.corr(numeric_only=True)

    target_correlation = correlation_matrix[target].drop(target) # drop target correlation (critical temp)

    positive_correlation = target_correlation[target_correlation > 0] # remove negative correlations

    top_15_correlations = positive_correlation.sort_values(ascending=False).head(15).index.tolist() #get the top 15 correlations

    matrix = df[top_15_correlations].corr() #create matrix with the top 15 correlations

    #create the figure and heatmap
    plt.figure(figsize=(15, 15))
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5
    )
    #figure/heatmap settings to make it more readable
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.title(f"Top 15 correlation coefficients for {target}")
    plt.show()

def task_3():
    corr_matrix = df.corr(numeric_only=True)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]

            if abs(corr) > 0.9:
                high_corr_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr
                    )
                )

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True) #sort by absolute correlation
    print("Highly correlated feature pairs (|r| > 0.9):\n")
    for feature1, feature2, corr in high_corr_pairs[:5]:
        print(f"{feature1} <-> {feature2}: {corr:.3f}")


#task_1()
#task_3()