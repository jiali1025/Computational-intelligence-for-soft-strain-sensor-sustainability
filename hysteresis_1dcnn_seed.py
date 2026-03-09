# pytorch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import time
from torch.utils.data import BatchSampler
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import csv
import math
import os
import random

torch.cuda.empty_cache()

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(f"[INFO] device = {device}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_std_ci95(values):
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    try:
        from scipy import stats
        tval = stats.t.ppf(0.975, df=n-1) if n > 1 else 1.96
    except Exception:
        tval = 1.96
    ci = tval * sd / math.sqrt(n) if n > 1 else 0.0
    return mean, sd, ci

# Dataset
class StrainDataset(Dataset):
    '''Dataset for loading and preprocessing the strain_sensor dataset '''
    def __init__(self, path, mode='train', sequence_length=12, transforms=None):
        self.mode = mode
        self.path = path
        self.seq_len = sequence_length
        self.transforms = transforms
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data)

            features = data[1:, 3:4]
            target = data[1:, 4:5]
            cyc_time = data[1:, 5:6]

            y = target.astype(float)
            self.y = torch.tensor(y)
            X = features.astype(float)
            self.X = torch.tensor(X)
            T = cyc_time.astype(float)
            self.T = torch.tensor(T)
        print('Finished reading the {} set of Strain Dataset ({} samples found)'.format(mode, len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if index >= self.seq_len - 1:
            i_start = index - self.seq_len + 1
            x = self.X[i_start:(index + 1), :]
            t = self.T[i_start:(index + 1), :]
        else:
            padding = self.X[0].repeat(self.seq_len - index - 1, 1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((padding, x), 0)

            padding_time = self.T[0].repeat(self.seq_len - index - 1, 1)
            t = self.T[0:(index + 1), :]

            t = torch.cat((padding_time, t), 0)
        return x, self.y[index], t

# -------------------------------- model -----------------------------------
class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1d = nn.Conv1d(1, 256, kernel_size=20)
        self.relu = nn.ReLU(inplace=True)
        self.pool1d = nn.MaxPool1d(kernel_size=50 - 20 + 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool1d(x)
        fc_inp = x.view(-1, x.size(1))
        x = self.fc1(fc_inp)
        x = self.relu(x)
        x = self.fc2(x)
        return x

loss_funtion = nn.MSELoss()


def train(model_tr_data, dv_set, model, config, device):
    '''1d_CNN training'''
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    min_mse = 1e9
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    num_batche_tr = len(model_tr_data)
    print("[INFO] num_train_batches =", num_batche_tr)

    while epoch < n_epochs:
        print(f"[Epoch] {epoch}")
        allbat_mse_loss = 0.0
        model.train()
        for x, y, t in model_tr_data:
            model.zero_grad()
            x = x.float().to(device)
            pred = model(x)
            y = y.float().to(device)
            mse_loss = loss_funtion(pred, y)
            allbat_mse_loss += mse_loss.item()
            mse_loss.backward()
            optimizer.step()

        evepoch_mse_loss = allbat_mse_loss / max(1, num_batche_tr)
        print(f"[Train] MSE={evepoch_mse_loss:.6f}")
        loss_record['train'].append(evepoch_mse_loss)

        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {}, loss = {:.6f})'.format((epoch + 1), (min_mse)))
            os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            print("[EarlyStop] Stop at epoch", epoch)
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(dv_set)
    for i, y, t in dv_set:
        x = i.float().to(device)
        with torch.no_grad():
            pred = model(x)
            y = y.float().to(device)
            mse_loss = loss_funtion(pred, y)
        total_loss += mse_loss.detach().cpu().item()
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss

def test(tt_set, model, device):
    model.eval()
    preds = []
    test_loss = 0.0
    num_batches = len(tt_set)
    for x, y, t in tt_set:
        x = x.float().to(device)
        y = y.float().to(device)
        with torch.no_grad():
            pred = model(x)
            batch_loss = loss_funtion(pred, y)
            preds.append(pred.detach().cpu())
        test_loss += batch_loss.detach().cpu().item()
    avg_loss_test = test_loss / max(1, num_batches)
    preds = torch.cat(preds, dim=0).numpy() if len(preds) > 0 else np.array([])
    print('Finished testing predictions!')
    return preds, avg_loss_test

def plot_learning_curve(loss_record, title=''):
    total_epochs = len(loss_record['train'])
    x_1 = range(total_epochs)
    x_2 = x_1[::max(1, len(loss_record['train']) // max(1, len(loss_record['dev'])))]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, max(1.0, max(loss_record['train'] + loss_record['dev'])))
    plt.xlabel('Training epochs')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_x, model, device, lim_x=60, lim_y=4000., preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        resist = []
        for i, y, t in dv_x:
            x = i.float().to(device)
            y = y.float().to(device)
            t = t.float().to(device)
            with torch.no_grad():
                pred = model(x)
                resist.append(x[:, -1, :].detach().cpu())
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        resist_values = torch.cat(resist, dim=0).numpy()
    # create plot
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('resistance')
    ax1.set_ylabel('ground truth value', color='red')
    ax1.scatter(resist_values, targets, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    # adding twin Axes
    ax2 = ax1.twinx()
    ax2.set_ylabel('predicted value', color='blue')
    ax2.scatter(resist_values, preds, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

def plot_pred_time(dv_x, model, device, preds=None, targets=None):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        cyc_time = []
        for i, y, t in dv_x:
            x = i.float().to(device)
            y = y.float().to(device)
            t = t.float().to(device)
            with torch.no_grad():
                pred = model(x)
                cyc_time.append(t[:, -1, :].detach().cpu())
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        time_values = torch.cat(cyc_time, dim=0).numpy()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('cycle time')
    ax1.set_ylabel('ground truth value', color='red')
    ax1.scatter(time_values, targets, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('predicted value', color='blue')
    ax2.scatter(time_values, preds, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()

# ------------------------------train/dev------------------------
def train_dev_data(x_set):
    X, deX = [], []
    for i in range(len(x_set)):
        if i % 10 != 0:
            X.append(x_set[i])
        else:
            deX.append(x_set[i])
    return X, deX


def run_multi_seed(
        seed_base=2025,
        num_runs=5,
        train_csv='./cycle_180_190_time.csv',
        test_csv='./cycle_190_195_test.csv',
        seq_len=50,
        batch_size=36,
        save_dir='1DCNN_multi_seed'):

    os.makedirs(save_dir, exist_ok=True)

    dataset = StrainDataset(train_csv, 'train', seq_len, transforms=None)
    dataset_tt = StrainDataset(test_csv, 'test', seq_len, transforms=None)

    training_data, valid_data = train_dev_data(dataset)
    model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
    model_tt_data = DataLoader(dataset_tt, batch_size=batch_size, shuffle=False, drop_last=True)

    base_config = {
        'n_epochs': 100,
        'batch_size': batch_size,
        'optimizer': 'Adam',
        'optim_hparas': {'lr': 1e-4},
        'early_stop': 100,
        'save_path': os.path.join(save_dir, 'model_1d_CNN_seedX.pth')
    }

    scores = []
    ckpts = []
    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "=" * 60)
        print(f"[Run {i+1}/{num_runs}] seed = {seed}")
        set_seed(seed)

        model = CNNnetwork().to(device)

        cfg = dict(base_config)
        cfg['save_path'] = os.path.join(save_dir, f"model_1d_CNN_seed{seed}.pth")
        min_dev, rec = train(model_tr_data, model_dev_data, model, cfg, device)
        # model.load_state_dict(torch.load(cfg['save_path'], map_location='cpu'))
        preds, test_mse = test(model_tt_data, model, device)
        print(f"[Seed {seed}] DEV best MSE={min_dev:.6f} | TEST MSE={test_mse:.6f}")

        scores.append(test_mse)
        ckpts.append(cfg['save_path'])

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (1D-CNN) =====")
    print(f"Runs (n): {num_runs}")
    print(f"Test MSE: {mean_mse:.6f} ± {ci95:.6f} (95% CI)")
    print(f"Std(MSE): {sd_mse:.6f}")
    print(f"ckpt example: {ckpts[0]} ... {ckpts[-1]}")

    # Save per-seed results
    csv_path = os.path.join(save_dir, "scores_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i in range(num_runs):
            w.writerow([i, seed_base + i, scores[i], ckpts[i]])
    print(f"[INFO] Scores saved to: {csv_path}")

    # Error bar plot
    plt.figure(figsize=(4,4))
    plt.bar(["1D-CNN"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"1D-CNN mean ± 95% CI (n={num_runs})")
    plt.tight_layout()
    plt.show()

    return {
        "scores": scores,
        "mean_mse": mean_mse,
        "std_mse": sd_mse,
        "ci95": ci95,
        "ckpts": ckpts,
        "csv": csv_path
    }

def run_multi_seed_eval_only(
        seed_base=2025,
        num_runs=5,
        test_csv='./cycle_190_195_test.csv',
        seq_len=50,
        batch_size=36,
        save_dir='1DCNN',
        ckpt_tmpl='model_1d_CNN_seed{seed}.pth'
    ):
    os.makedirs(save_dir, exist_ok=True)

    dataset_tt = StrainDataset(test_csv, 'test', seq_len, transforms=None)
    model_tt_data = DataLoader(dataset_tt, batch_size=batch_size, shuffle=False, drop_last=True)

    scores = []
    ckpts = []
    used_seeds = []

    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "=" * 60)
        print(f"[Run {i+1}/{num_runs}] seed = {seed}")
        set_seed(seed)

        model = CNNnetwork().to(device)

        ckpt_path = os.path.join(save_dir, ckpt_tmpl.format(seed=seed))
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] ckpt not found: {ckpt_path}  -> skip this seed")
            continue

        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        # Save test predictions
        preds, test_mse = test(model_tt_data, model, device)
        print(f"[Seed {seed}] TEST MSE={test_mse:.6f}")

        scores.append(test_mse)
        ckpts.append(ckpt_path)
        used_seeds.append(seed)

    if len(scores) == 0:
        print("[ERROR] No valid checkpoints found. Please check save_dir / ckpt_tmpl.")
        return None

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (1D-CNN, EVAL ONLY) =====")
    print(f"Runs (n): {len(scores)}")
    print(f"Test MSE: {mean_mse:.6f} ± {ci95:.6f} (95% CI)")
    print(f"Std(MSE): {sd_mse:.6f}")
    print(f"ckpt example: {ckpts[0]} ... {ckpts[-1]}")

    # save scores
    csv_path = os.path.join(save_dir, "scores_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i, (sd, mse, path) in enumerate(zip(used_seeds, scores, ckpts)):
            w.writerow([i, sd, mse, path])
    print(f"[INFO] Scores saved to: {csv_path}")

    # Error bar plot
    plt.figure(figsize=(4,4))
    plt.bar(["1D-CNN"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"1D-CNN mean ± 95% CI (n={len(scores)})")
    plt.tight_layout()
    plt.show()

    return {
        "scores": scores,
        "mean_mse": mean_mse,
        "std_mse": sd_mse,
        "ci95": ci95,
        "ckpts": ckpts,
        "csv": csv_path
    }

# Main
if __name__ == "__main__":
    # summary = run_multi_seed(
    #     seed_base=2025,
    #     num_runs=5,
    #     train_csv='./cycle_180_190_time.csv',
    #     test_csv='./cycle_190_195_test.csv',
    #     seq_len=50,
    #     batch_size=36,
    #     save_dir='1DCNN'
    # )

    summary = run_multi_seed_eval_only(
        seed_base=2025,
        num_runs=5,
        test_csv='./cycle_190_195_test.csv',
        seq_len=50,
        batch_size=36,
        save_dir='1DCNN',
        ckpt_tmpl='model_1d_CNN_seed{seed}.pth'
    )
    print(summary)
