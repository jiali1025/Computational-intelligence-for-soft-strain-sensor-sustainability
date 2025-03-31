import pandas as pd
import numpy as np
import csv


def calculate_mse_between_columns(csv_file, output_file):

    df = pd.read_csv(csv_file)

    df.iloc[:, 3] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    df.iloc[:, 8] = pd.to_numeric(df.iloc[:, 8], errors='coerce')
    df = df.dropna(subset=[df.columns[3], df.columns[8]])

    mse_values = (df.iloc[:, 3] - df.iloc[:, 8]) ** 2


    df['MSE'] = mse_values
    df.to_csv(output_file, index=False)

    print(f"MSE values saved to {output_file}")

input_file = 'linear_cw2_testing.csv'
output_file = 'sigmoid_prediction_mse.csv'
calculate_mse_between_columns(input_file, output_file)
