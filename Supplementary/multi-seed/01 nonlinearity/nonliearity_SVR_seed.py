from sklearn import svm
from sklearn.metrics import mean_squared_error
import numpy as np
import csv
import joblib
import random

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
    valid_rows = np.array(valid_rows, dtype=float)
    X = valid_rows[:, 0].reshape(-1, 1)
    y = valid_rows[:, 1]
    return X, y

path_train = './linear_cw2_training.csv'
path_test  = './linear_cw2_testing.csv'
X_train, y_train = load_data(path_train)
X_test,  y_test  = load_data(path_test)


seeds = [42, 99, 3407, 2023, 7]
summary_rows = []

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

for seed in seeds:
    print(f"\n===== SEED = {seed} =====")
    set_seed(seed)

    model = svm.SVR(kernel='linear')

    # training
    model.fit(X_train, y_train)

    # save the model
    ckpt_path = f"svm_model_seed{seed}.pkl"
    joblib.dump(model, ckpt_path)
    print(f"Model saved to {ckpt_path}")

    # evaluate
    train_score = model.score(X_train, y_train)  # R^2
    test_score  = model.score(X_test, y_test)    # R^2
    preds       = model.predict(X_test)
    test_mse    = mean_squared_error(y_test, preds)

    print(f"Train R^2: {train_score:.6f}")
    print(f"Test  R^2: {test_score:.6f}")
    print(f"Test  MSE: {test_mse:.6f}")

    mse_values = (y_test - preds) ** 2
    per_pred_csv = f"svr_prediction_mse_seed{seed}.csv"
    with open(per_pred_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Prediction', 'MSE'])
        for i, mse_value in enumerate(mse_values):
            writer.writerow([preds[i], mse_value])
    print(f"MSE values saved to {per_pred_csv}")

    summary_rows.append([seed, train_score, test_score, test_mse, ckpt_path])

summary_csv = "svr_multi_seed_summary.csv"
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['seed', 'train_r2', 'test_r2', 'test_mse', 'ckpt_path'])
    writer.writerows(summary_rows)

print("\n===== Multi-seed Summary =====")
for row in summary_rows:
    print(f"seed={row[0]} | train_r2={row[1]:.6f} | test_r2={row[2]:.6f} | test_mse={row[3]:.6f} | ckpt={row[4]}")
print(f"\nSummary saved to {summary_csv}")
