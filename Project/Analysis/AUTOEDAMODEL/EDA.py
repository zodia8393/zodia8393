import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import scipy.stats as stats
import json

class EDA_GUI:
    def __init__(self, master):
        self.master = master
        master.title("Exploratory Data Analysis")

        # Create a frame for selecting data file
        self.file_frame = tk.Frame(master)
        self.file_frame.pack(padx=10, pady=10)
        self.file_label = tk.Label(self.file_frame, text="Select data file:")
        self.file_label.pack(side="left")
        self.file_button = tk.Button(self.file_frame, text="Browse...", command=self.select_file)
        self.file_button.pack(side="left")

        # Create a frame for selecting EDA tasks
        self.task_frame = tk.Frame(master)
        self.task_frame.pack(padx=10, pady=10)
        self.task_label = tk.Label(self.task_frame, text="Select EDA tasks to perform:")
        self.task_label.pack(side="left")
        self.summary_var = tk.BooleanVar()
        self.summary_box = tk.Checkbutton(self.task_frame, text="Calculate summary statistics", variable=self.summary_var)
        self.summary_box.pack(side="left")
        self.correlation_var = tk.BooleanVar()
        self.correlation_box = tk.Checkbutton(self.task_frame, text="Calculate correlation coefficients", variable=self.correlation_var)
        self.correlation_box.pack(side="left")
        self.distribution_var = tk.BooleanVar()
        self.distribution_box = tk.Checkbutton(self.task_frame, text="Create distribution plots", variable=self.distribution_var)
        self.distribution_box.pack(side="left")
        self.outlier_var = tk.BooleanVar()
        self.outlier_box = tk.Checkbutton(self.task_frame, text="Detect outliers", variable=self.outlier_var)
        self.outlier_box.pack(side="left")

        # Create a button to perform EDA analysis
        self.run_button = tk.Button(master, text="Run EDA analysis", command=self.run_analysis)
        self.run_button.pack(pady=10)

        # Create a label for displaying status messages
        self.status_label = tk.Label(master, text="")
        self.status_label.pack()

        # Initialize data and file path
        self.data = None
        self.file_path = None

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        self.file_label.config(text=self.file_path)


def run_analysis(self):
        # Check if data file has been selected
        if not self.file_path:
            self.status_label.config(text="Please select a data file.")
            return

        # Load data from file
        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            return

        # Perform selected EDA tasks
        eda_results = {}
        if self.summary_var.get():
            eda_results['summary_statistics'] = self.data.describe(include='all').to_dict()
        if self.correlation_var.get():
            corr_matrix = self.data.corr()
            corr_results = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
            corr_results = corr_results.stack().reset_index()
            corr_results.columns = ['Variable 1', 'Variable 2', 'Correlation Coefficient']
            eda_results['correlation_results'] = corr_results.to_dict('records')
        if self.distribution_var.get():
            for column in self.data.columns:
                plot_data = self.data[column].dropna()
                if plot_data.dtype == np.float64 or plot_data.dtype == np.int64:
                    plot_type = 'histogram'
                    plot_values, plot_bins = np.histogram(plot_data)
                    plot_values = plot_values.tolist()
                    plot_bins = plot_bins.tolist()
                else:
                    plot_type = 'bar'
                    plot_values = plot_data.value_counts(normalize=True).tolist()
                    plot_bins = plot_data.unique().tolist()
                eda_results[f'{column}_distribution'] = {
                    'plot_type': plot_type,
                    'values': plot_values,
                    'bins': plot_bins
                }
        if self.outlier_var.get():
            z_scores = np.abs(stats.zscore(self.data))
            outlier_indices = np.argwhere(z_scores > 3)
            outlier_results = []
            for row, col in outlier_indices:
                outlier_results.append({
                    'variable': self.data.columns[col],
                    'value': self.data.iloc[row, col],
                    'z_score': z_scores[row, col]
                })
            eda_results['outlier_detection'] = outlier_results

        # Display results
        self.status_label.config(text=json.dumps(eda_results, indent=4))
