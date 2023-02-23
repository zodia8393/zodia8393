#Auto Data Analysis Tool

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create the main window
root = tk.Tk()
root.title("Data Analysis Tool")

# Add a label to the window
label = tk.Label(root, text="Select data files to analyze")
label.pack()

# Add a button to select the data files
file_paths = []

def select_files():
    file_paths_temp = filedialog.askopenfilenames()
    file_paths.extend(file_paths_temp)
    for file_path in file_paths:
        print(f"Selected file: {file_path}")
    analyze_data()

file_button = tk.Button(root, text="Select Files", command=select_files)
file_button.pack()

# Add a function to perform EDA on the data
def eda(data):
    # Summary statistics
    print("Summary Statistics:")
    print(data.describe())

    # Distribution plots
    print("Distribution Plots:")
    for col in data.columns:
        if data[col].dtype == "float" or data[col].dtype == "int":
            sns.histplot(data=data, x=col)
            plt.show()
        else:
            sns.countplot(data=data, x=col)
            plt.show()

    # Correlation heatmap
    print("Correlation Heatmap:")
    corr = data.corr()
    sns.heatmap(corr, cmap="coolwarm", annot=True)
    plt.show()

    # Pairwise scatterplots
    print("Pairwise Scatterplots:")
    sns.pairplot(data)
    plt.show()

    # Boxplots
    print("Boxplots:")
    for col in data.columns:
        if data[col].dtype == "float" or data[col].dtype == "int":
            sns.boxplot(data=data, x=col)
            plt.show()

# Add a function to recommend a data analysis method
def recommend_analysis(data):
    if data.dtypes.unique().size == 1:
        if data.dtypes.unique()[0] == "object":
            return "Frequency Analysis"
        else:
            return "Distribution Analysis"
    elif data.dtypes.unique().size == 2:
        if "object" in data.dtypes.unique():
            return "Frequency Analysis"
        else:
            return "Scatterplot Analysis"
    else:
        return "Correlation Analysis"

# Add a dropdown menu to select the analysis method
analysis_methods = ["Method 1", "Method 2", "Method 3", "Recommended Analysis"]

method_var = tk.StringVar()
method_var.set(analysis_methods[0])

method_menu = tk.OptionMenu(root, method_var, *analysis_methods)
method_menu.pack()

# Function to analyze the data
def analyze_data():
    # Read the data files
    try:
        data = pd.concat([pd.read_csv(file_path) for file_path in file_paths], ignore_index=True)
    except:
        print("Error: Invalid file format")
        return

    # Perform EDA on the data
    eda(data)

    # Get the selected analysis method
    method = method_var.get()

    # Perform the analysis
    if method == "Method 1":
        # Code for method 1
        pass
    elif method == "Method 2":
        # Code for method 2
        pass
    elif method == "Method 3":
        # Code for method 3
        pass
    elif method == "Recommended Analysis":
        # Recommend an analysis method based on the data type
        recommended_method = recommend_analysis(data)
        print(f"Recommended Analysis: {recommended_method}")

# Run the main loop
root.mainloop()
