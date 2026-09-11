import pandas as pd

data_name = "train.csv" #csv file name here

df = pd.read_csv(data_name)

stats = pd.DataFrame({
    "mean": df.mean(),
    "std": df.std(),
    "min": df.min(),
    "q1": df.quantile(0.25),
    "median": df.median(),
    "q3": df.quantile(0.75),
    "max": df.max()
})

def print_all_summaries(dataframe):
    for col in dataframe.index:
        print(f"\nColumn: {col}")
        print(f"  Mean   = {stats.loc[col, 'mean']:.4f}")
        print(f"  Std    = {stats.loc[col, 'std']:.4f}")
        print(f"  Min    = {stats.loc[col, 'min']:.4f}")
        print(f"  Q1     = {stats.loc[col, 'q1']:.4f}")
        print(f"  Median = {stats.loc[col, 'median']:.4f}")
        print(f"  Q3     = {stats.loc[col, 'q3']:.4f}")
        print(f"  Max    = {stats.loc[col, 'max']:.4f}")

def print_std(dataframe):
    sorted_df = dataframe.sort_values(by="std")

    print("5 Lowest values:")
    for col in sorted_df.head(5).index:
        print(f"\nColumn: {col}")
        print(f"  Std    = {sorted_df.loc[col, 'std']:.4f}")

    print("\n\n5 Highest values:")
    for col in sorted_df.tail(5).index:
        print(f"\nColumn: {col}")
        print(f"  Std    = {sorted_df.loc[col, 'std']:.4f}")

#print_all_summaries(stats)
print_std(stats)