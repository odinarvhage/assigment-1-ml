import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_name = "train.csv" #csv file name here

df = pd.read_csv(data_name)

critical_temp = df["critical_temp"]
log_critical_temp = np.log(critical_temp)

def plot_distribution(column):
    plt.figure(figsize=(8, 5))

    plt.hist(column, bins=20)

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Amount")

    plt.show()

plot_distribution(critical_temp)
plot_distribution(log_critical_temp)