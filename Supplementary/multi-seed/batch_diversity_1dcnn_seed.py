import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import csv
import os
import math
import random

torch.cuda.empty_cache()

# -------------------------------------- Dataset --------------------------------
class StrainDataset(Dataset):
    def __init__(self, path, mode='train', sequence_length=500, transforms=None):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.seq_len = sequence_length
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data)
            features     = data[1::30, 2:5]
            target       = data[1::30, 5:6]
            all_time     = data[1::30, 2:3]
            resistance_f = data[1::30, 2:3]
            cycle_num    = data[1::30, 3:4]

            y = target.astype(float)
            self.y = torch.tensor(y)
            X = features.astype(float)
            self.X = torch.tensor(X)
            time = all_time.astype(float)
            self.time = torch.tensor(time)
            resis_f = resistance_f.astype(float)
            self.resis_f = torch.tensor(resis_f)
            cyc_n = cycle_num.astype(float)
            self.cyc_n = torch.tensor(cyc_n)

            self.X[:, :] = \
                (self.X[:, :] - self.X[:, :].mean(dim=0, keepdim=True)) \
                / self.X[:, :].std(dim=0, keepdim=True)
        print('Finished reading the {} set of Strain Dataset ({} samples found)'.format(mode, len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if index >= self.seq_len - 1:
            i_start = index - self.seq_len + 1
            x     = self.X[i_start:(index + 1), :]
            time  = self.time[i_start:(index + 1), :]
            resist= self.resis_f[i_start:(index + 1), :]
            cycle = self.cyc_n[i_start:(index + 1), :]
        else:
            padding = self.X[0].repeat(self.seq_len - index - 1, 1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((padding, x), 0)

            padding_time = self.time[0].repeat(self.seq_len - index - 1, 1)
            time = self.time[0:(index + 1), :]
            time = torch.cat((padding_time, time), 0)

            padding_resist = self.resis_f[0].repeat(self.seq_len - index - 1, 1)
            resist = self.resis_f[0:(index + 1), :]
            resist = torch.cat((padding_resist, resist), 0)

            padding_cycle = self.cyc_n[0].repeat(self.seq_len - index - 1, 1)
            cycle = self.cyc_n[0:(index + 1), :]
            cycle = torch.cat((padding_cycle, cycle), 0)
        return x, self.y[index], time, resist, cycle

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(3, 256, kernel_size=5)
        self.relu   = nn.ReLU(inplace=True)
        self.pool1d = nn.MaxPool1d(kernel_size=30 - 5 + 1)
        self.fc1    = nn.Linear(256, 128)
        self.fc2    = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)           # -> [B, C=3, T]
        x = self.conv1d(x)               # -> [B, 256, T-4]
        x = self.relu(x)
        x = self.pool1d(x)               # -> [B, 256, 1]
        fc_inp = x.view(-1, x.size(1))   # -> [B, 256]
        x = self.fc1(fc_inp)
        x = self.relu(x)
        x = self.fc2(x)
        return x

loss_funtion = nn.MSELoss()

def train(model_tr_data, dv_set, model, config, device):
    '''1dCNN training'''
    n_epochs  = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    min_mse = 1000
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0
    num_batche_tr = len(model_tr_data)
    print(num_batche_tr)
    while epoch < n_epochs:
        print(epoch)
        allbat_mse_loss = 0
        model.train()
        for x, y, t, r, c in model_tr_data:
            model.zero_grad()
            x = x.float().to(device)
            pred = model(x)
            y = y.float().to(device)
            mse_loss = loss_funtion(pred, y)
            allbat_mse_loss += mse_loss
            mse_loss.backward()
            optimizer.step()
        evepoch_mse_loss = allbat_mse_loss / num_batche_tr
        print(evepoch_mse_loss)
        loss_record['train'].append(evepoch_mse_loss.detach().cpu().item())

        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {},loss = {:.4f}'.format((epoch + 1), (min_mse)))

            os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt = early_stop_cnt + 1
        epoch = epoch + 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    num_batches = len(dv_set)
    for i, y, t, r, c in dv_set:
        x = i.float().to(device)
        with torch.no_grad():
            pred = model(x)
            y = y.float().to(device)
            mse_loss = loss_funtion(pred, y)
        total_loss += mse_loss.detach().cpu().item()
    avg_loss = total_loss / num_batches
    return avg_loss

def test(tt_set, model, device):
    model.eval()
    preds = []
    test_loss = 0.0
    num_batches = len(tt_set)
    for x, y, t, r, c in tt_set:
        x = x.float().to(device)
        y = y.float().to(device)
        with torch.no_grad():
            pred = model(x)
            batch_loss = loss_funtion(pred, y)
            preds.append(pred.detach().cpu())
        test_loss += batch_loss.detach().cpu().item()
    avg_loss_test = test_loss / num_batches
    preds = torch.cat(preds, dim=0).numpy()
    print('Finished testing predictions!')
    return preds, avg_loss_test

def plot_learning_curve(loss_record, title=''):
    total_epochs = len(loss_record['train'])
    x_1 = range(total_epochs)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red',  label='train')
    plt.plot(x_2, loss_record['dev'],   c='tab:cyan', label='dev')
    plt.ylim(0.0, 100.0)
    plt.xlabel('Training epochs')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def plot_pred(dv_x, model, device, lim_x=60, lim_y=4000., preds=None, targets=None, title=''):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        resist = []
        for i, y, t, r, c in dv_x:
            x = i.float().to(device)
            y = y.float().to(device)
            t = t.float().to(device)
            r = r.float().to(device)
            with torch.no_grad():
                pred = model(x)
                resist.append(r[:, -1, :].detach().cpu())
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        resist_values = torch.cat(resist, dim=0).numpy()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('resistance')
    ax1.set_ylabel('ground truth value', color='red')
    ax1.scatter(resist_values, targets, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('predicted value', color='blue')
    ax2.scatter(resist_values, preds, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title('Ground Truth v.s. Prediction of {}'.format(title))
    plt.show()

def plot_pred_time(dv_x, model, device, preds=None, targets=None, title=''):
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        all_time = []
        for i, y, t, r, c in dv_x:
            x = i.float().to(device)
            y = y.float().to(device)
            t = t.float().to(device)
            with torch.no_grad():
                pred = model(x)
                all_time.append(t[:, -1, :].detach().cpu())
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        time_values = torch.cat(all_time, dim=0).numpy()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('all time')
    ax1.set_ylabel('ground truth value', color='red')
    ax1.scatter(time_values, targets, color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx()
    ax2.set_ylabel('predicted value', color='blue')
    ax2.scatter(time_values, preds, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    plt.title('Ground Truth v.s. Prediction of {}'.format(title))
    plt.show()

def train_dev_data(x_set):
    X, deX = [], []
    for i in range(len(x_set)):
        if i % 10 != 0:
            X.append(x_set[i])
        else:
            deX.append(x_set[i])
    return X, deX

os.makedirs('BatchDiversity_1DCNN_models', exist_ok=True)
config = {
    'n_epochs': 100,
    'batch_size': 36,
    'optimizer': 'Adam',
    'optim_hparas': {'lr': 0.0001},
    'early_stop': 100,
    'save_path': 'BatchDiversity_1DCNN_models/BatchDiversity_1DCNN_01.pth'
}
train_csv = './Training_batchdiversity_1589.csv'
test_csv  = './Testing_batchdiversity_10.csv'
seq_len   = 30
batch_size= 36

def set_seed(seed:int):
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
    sd   = float(np.std(arr, ddof=1)) if n>1 else 0.0
    try:
        from scipy import stats
        tval = stats.t.ppf(0.975, df=n-1) if n>1 else 1.96
    except Exception:
        tval = 1.96
    ci = tval * sd / math.sqrt(n) if n>1 else 0.0
    return mean, sd, ci


def run_multi_seed(seed_base=2025, num_runs=5, save_dir='BatchDiversity_1DCNN_models'):
    os.makedirs(save_dir, exist_ok=True)

    dataset_tr = StrainDataset(train_csv, 'train', seq_len, transforms=None)
    dataset_te = StrainDataset(test_csv,  'test',  seq_len, transforms=None)
    training_data, valid_data = train_dev_data(dataset_tr)
    model_tr_data  = DataLoader(training_data, batch_size=batch_size, shuffle=True,  drop_last=True)
    model_dev_data = DataLoader(valid_data,   batch_size=batch_size, shuffle=True,  drop_last=True)
    model_tt_data  = DataLoader(dataset_te,   batch_size=batch_size, shuffle=False, drop_last=True)

    scores = []
    ckpts  = []

    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "="*60)
        print(f"[Run {i+1}/{num_runs}] seed = {seed}")
        set_seed(seed)

        model = CNNnetwork().to(device)

        cfg = dict(config)
        cfg['save_path'] = os.path.join(save_dir, f"cnn1d_seed{seed}.pth")

        min_dev, rec = train(model_tr_data, model_dev_data, model, cfg, device)

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

    csv_path = os.path.join(save_dir, "scores_seed_cnn1d.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i in range(num_runs):
            w.writerow([i, seed_base + i, scores[i], ckpts[i]])
    print(f"[INFO] Scores saved to: {csv_path}")

    plt.figure(figsize=(4,4))
    plt.bar(["1D-CNN"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"1D-CNN mean ± 95% CI (n={num_runs})")
    plt.tight_layout()
    plt.show()

    return {"scores": scores, "mean_mse": mean_mse, "std_mse": sd_mse, "ci95": ci95, "ckpts": ckpts, "csv": csv_path}

def run_multi_seed_eval_only(
        seed_base=2025,
        num_runs=10,
        test_csv_path=test_csv,
        seq_len_eval=seq_len,
        batch_size_eval=batch_size,
        save_dir='BatchDiversity_1DCNN_models',
        ckpt_tmpl="cnn1d_seed{seed}.pth"
    ):

    os.makedirs(save_dir, exist_ok=True)
    dataset_te = StrainDataset(test_csv_path, 'test', seq_len_eval, transforms=None)
    tt_loader  = DataLoader(dataset_te, batch_size=batch_size_eval,
                            shuffle=False, drop_last=True)
    scores, ckpts, used_seeds = [], [], []

    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "=" * 60)
        print(f"[Eval] seed = {seed}")
        set_seed(seed)
        model = CNNnetwork().to(device)
        ckpt_path = os.path.join(save_dir, ckpt_tmpl.format(seed=seed))
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] ckpt not found: {ckpt_path} -> skip this seed")
            continue

        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        preds, test_mse = test(tt_loader, model, device)
        print(f"[Seed {seed}] TEST MSE = {test_mse:.6f}")

        pred_csv_path = os.path.join(save_dir, f"preds_seed{seed}.csv")
        with open(pred_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "pred"])
            for idx, p in enumerate(preds):
                w.writerow([idx, float(p)])
        print(f"[INFO] Predictions of seed {seed} saved to: {pred_csv_path}")

        scores.append(test_mse)
        ckpts.append(ckpt_path)
        used_seeds.append(seed)

    if len(scores) == 0:
        print("[ERROR] No valid checkpoints found. Check save_dir/ckpt_tmpl.")
        return None

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (1D-CNN, EVAL ONLY) =====")
    print(f"Runs (n)   : {len(scores)}")
    print(f"Test MSE   : {mean_mse:.6f} ± {ci95:.6f}  (95% CI)")
    print(f"Std (MSE)  : {sd_mse:.6f}")
    print(f"First ckpt : {ckpts[0]}")
    print(f"Last ckpt  : {ckpts[-1]}")

    csv_path = os.path.join(save_dir, "scores_seed_eval_cnn1d.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i, (sd, mse, path) in enumerate(zip(used_seeds, scores, ckpts)):
            w.writerow([i, sd, mse, path])
    print(f"[INFO] Eval scores saved to: {csv_path}")

    # Draw error bar
    plt.figure(figsize=(4,4))
    plt.bar(["1D-CNN"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"1D-CNN (n={len(scores)}) mean ± 95% CI (EVAL ONLY)")
    plt.tight_layout()
    plt.show()

    return {
        "scores": scores,
        "mean_mse": mean_mse,
        "std_mse": sd_mse,
        "ci95": ci95,
        "ckpts": ckpts,
        "csv_path": csv_path
    }

# ===================== directly run =====================
if __name__ == "__main__":
    # summary = run_multi_seed(
    #     seed_base=2025,
    #     num_runs=5,
    #     save_dir='BatchDiversity_1DCNN_models'
    # )
    summary = run_multi_seed_eval_only(
        seed_base=2025,
        num_runs=5,
        test_csv_path=test_csv,
        seq_len_eval=seq_len,
        batch_size_eval=batch_size,
        save_dir='BatchDiversity_1DCNN_models',
        ckpt_tmpl="cnn1d_seed{seed}.pth"
    )

    print(summary)
