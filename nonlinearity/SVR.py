from sklearn import svm
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import csv
import joblib

# load your data and handle missing or non-numeric values
def load_data(path):
    valid_rows = []
    with open(path, 'r') as fp:
        reader = csv.reader(fp)
        next(reader)  # Skip header row
        for row in reader:
            try:
                X_val = float(row[6])
                y_val = float(row[3])
                valid_rows.append([X_val, y_val])
            except ValueError:
                continue

    valid_rows = np.array(valid_rows)
    X = valid_rows[:, 0]
    y = valid_rows[:, 1]
    return X, y

path_train = './linear_cw2_training.csv'
path_test = './linear_cw2_testing.csv'
X_train, y_train = load_data(path_train)
X_test, y_test = load_data(path_test)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

model = svm.SVR(kernel='linear')

# # train
# model.fit(X_train, y_train)

model_filename = 'svm_model.pkl'
# joblib.dump(model, model_filename)
# print(f"Model saved to {model_filename}")

loaded_model = joblib.load(model_filename)

# validate
train_score = loaded_model.score(X_train, y_train)
test_score = loaded_model.score(X_test, y_test)
print('Train score:', train_score)
print('Test score:', test_score)

# predict
predictions = loaded_model.predict(X_test)

# compute MSE for each prediction
mse_values = (y_test - predictions) ** 2

# print MSE for each prediction
for i, mse_value in enumerate(mse_values):
    print(f"Prediction {i+1}: MSE = {mse_value}")

# you can also save these values to a CSV file
output_file = 'svr_prediction_mse.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Prediction', 'MSE'])
    for i, mse_value in enumerate(mse_values):
        writer.writerow([predictions[i], mse_value])
print(f"MSE values saved to {output_file}")
