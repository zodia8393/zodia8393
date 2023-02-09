import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda(data):
    # convert the input data into a Pandas dataframe
    df = pd.DataFrame(data)

    # print the first 5 rows of the data
    print("First 5 rows of the data:")
    print(df.head())

    # print the descriptive statistics of the data
    print("\nDescriptive Statistics:")
    print(df.describe())

    # plot the histograms of the numerical columns
    df.hist(bins=30, figsize=(10,7))
    plt.tight_layout()
    plt.show()

    # plot the pairplot of the numerical columns
    sns.pairplot(df)
    plt.show()

    # plot the boxplot of the numerical columns
    plt.figure(figsize=(10,7))
    sns.boxplot(data=df)
    plt.tight_layout()
    plt.show()

    # plot the countplot of the categorical columns
    categorical_columns = df.select_dtypes(include=np.object).columns
    if categorical_columns.size > 0:
        plt.figure(figsize=(10,7))
        sns.countplot(x=categorical_columns[0], data=df)
        plt.tight_layout()
        plt.show()

    # plot the correlation matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(df.corr(), annot=True)
    plt.tight_layout()
    plt.show()

# example input data
data = {
    'column1': [1, 2, 3, 4, 5],
    'column2': [2, 3, 4, 5, 6],
    'column3': [3, 4, 5, 6, 7]
}

# perform the exploratory data analysis
perform_eda(data)
