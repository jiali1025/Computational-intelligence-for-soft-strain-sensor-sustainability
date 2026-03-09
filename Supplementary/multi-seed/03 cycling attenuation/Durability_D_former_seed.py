#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv
import os
import math
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, Tensor
from torch.optim import AdamW
import causal_convolution_layer
from RoFormer import modeling_roformer, RoFormerConfig

class StrainDataset(Dataset):
    def __init__(self, path, mode='train', sequence_length=500, transforms=None):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.seq_len = sequence_length

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data)
            features = data[1::3, 2:4]
            target = data[1::3, 4:5]
            all_time = data[1::3, 1:2]
            resistance_f = data[1::3, 2:3]
            cycle_num = data[1::3, 3:4]

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
            x = self.X[i_start:(index + 1), :]
            time = self.time[i_start:(index + 1), :]
            resist = self.resis_f[i_start:(index + 1), :]
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

# splitting training data into train & val sets
def train_dev_data(x_set):
    X = []
    deX = []
    for i in range(len(x_set)):
        if i % 10 != 0:
            x_data = x_set[i]
            X.append(x_data)
        else:
            devX = x_set[i]
            deX.append(devX)
    return X, deX

torch.cuda.empty_cache()
class D_Former(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.prenet = nn.Linear(2, d_model)
        self.input_embedding = causal_convolution_layer.context_embedding(500, d_model, 5)
        config = RoFormerConfig()
        self.roformerEnc = modeling_roformer.RoFormerEncoder(config)
        self.decoder = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, src: Tensor) -> Tensor:
        src = self.prenet(src)
        z_embedding = self.input_embedding(src)
        z_embeoutput = z_embedding.reshape(32, -1, 24)
        output = self.roformerEnc(z_embeoutput)
        output = output[0]
        output = output[:, -1, :]
        output = self.decoder(output)

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pos_encoding)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] using device: {device}")

d_model = 12

# training config
base_config = {
    'n_epochs': 120,
    'early_stop': 20,
    'save_path': 'Durability_D_former_01/Durability_D_Former_01.pth',
    'lr': 1e-5,
}

criterion = nn.MSELoss()

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

# dev/train/test
def dev(dv_set, model, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(dv_set)
    for i, y, t, r, c in dv_set:
        x = i.float().to(device)
        with torch.no_grad():
            pred = model(x)
            y = y.float().to(device)
            mse_loss = criterion(pred, y)
        total_loss += mse_loss.detach().cpu().item()
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss

def train(tr_loader, dv_set, model, config, device, save_path=None):
    n_epochs = config['n_epochs']
    optimizer = AdamW(model.parameters(), lr=config.get('lr', 1e-5))
    # warmup / total steps
    min_mse = 2000
    loss_record = {'train': [], 'dev': []}
    lrs = []
    early_stop_cnt = 0
    epoch = 0
    num_batche_tr = len(tr_loader)
    while epoch < n_epochs:
        allbat_mse_loss = 0
        model.train()
        for x, y, t, r, c in tr_loader:
            x = x.float().to(device)
            pred = model(x)
            y = y.float().to(device)
            mse_loss = criterion(pred, y)
            allbat_mse_loss += mse_loss
            mse_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        evepoch_mse_loss = allbat_mse_loss / max(1, num_batche_tr)
        loss_record['train'].append(evepoch_mse_loss.detach().cpu().item())
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            if save_path:
                torch.save(model.state_dict(), save_path)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            break
    return min_mse, loss_record

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
            batch_loss = criterion(pred, y)
            preds.append(pred.detach().cpu())
        test_loss += batch_loss.detach().cpu().item()
    avg_loss_test = test_loss / max(1, num_batches)
    preds = torch.cat(preds, dim=0).numpy() if len(preds) > 0 else np.array([])
    print('Finished testing predictions!')
    return preds, avg_loss_test

# Utility: set seeds
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# statistics
def mean_std_ci95(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    mean = float(vals.mean())
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    try:
        from scipy import stats
        tval = stats.t.ppf(0.975, df=n-1) if n > 1 else 1.96
    except Exception:
        tval = 1.96
    ci = tval * sd / math.sqrt(n) if n > 1 else 0.0
    return mean, sd, ci

# run one seed: init model, train with scheduler, test, save ckpt
def run_one_seed(seed, tr_loader, dv_loader, tt_loader, config, save_dir):
    set_seed(seed)
    model = D_Former(d_model).to(device)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"Durability_D_Former_seed{seed}.pth")

    # optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=config.get('lr', 1e-5))
    total_steps = len(tr_loader) * config['n_epochs']
    warmup_steps = max(1, len(tr_loader) * 2)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    min_dev = float('inf')
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0

    for epoch in range(config['n_epochs']):
        model.train()
        total_train_loss = 0.0
        for x, y, t, r, c in tr_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.detach().cpu().item()
        avg_train = total_train_loss / max(1, len(tr_loader))
        avg_dev = dev(dv_loader, model, device)
        loss_record['train'].append(avg_train)
        loss_record['dev'].append(avg_dev)

        if avg_dev < min_dev - 1e-8:
            min_dev = avg_dev
            torch.save(model.state_dict(), ckpt_path)
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        print(f"[Seed {seed}] Epoch {epoch+1:03d} train={avg_train:.6f} dev={avg_dev:.6f} best_dev={min_dev:.6f} es={early_stop_cnt}")
        if early_stop_cnt > config['early_stop']:
            print(f"[Seed {seed}] Early stopping at epoch {epoch+1}")
            break

    # model.load_state_dict(torch.load(ckpt_path))
    preds, test_mse = test(tt_loader, model, device)
    return test_mse, ckpt_path, loss_record

# multi-seed runner
def run_multi_seed_transformer(seed_base=2025, num_runs=5,
                               train_csv='./01_training_data_4000.csv',
                               test_csv='./01_testing_data_16000.csv',
                               seq_len=500, batch_size=32, save_dir='Durability_D_Former_01'):

    dataset_for_train = StrainDataset(train_csv, 'train', seq_len, transforms=None)
    dataset_for_test_01 = StrainDataset(test_csv, 'test', seq_len, transforms=None)
    training_data, valid_data = train_dev_data(dataset_for_train)

    tr_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
    dv_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
    tt_loader = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)

    config = base_config.copy()
    config['n_epochs'] = base_config['n_epochs']
    config['early_stop'] = base_config['early_stop']
    config['lr'] = base_config['lr']

    scores = []
    ckpts = []
    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "="*60)
        print(f"[Run {i+1}/{num_runs}] seed={seed}")
        test_mse, ckpt_path, rec = run_one_seed(seed, tr_loader, dv_loader, tt_loader, config, save_dir)
        scores.append(test_mse)
        ckpts.append(ckpt_path)

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (D_Former) =====")
    print(f"Runs: {num_runs}")
    print(f"Test MSE: {mean_mse:.6f} ± {ci95:.6f} (95% CI)")
    print(f"Std(MSE): {sd_mse:.6f}")
    print(f"ckpt sample: {ckpts[0]} ... {ckpts[-1]}")

    # save CSV
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "scores_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i in range(num_runs):
            w.writerow([i, seed_base + i, scores[i], ckpts[i]])
    print(f"[INFO] Scores saved to: {csv_path}")

    # plot error bar
    plt.figure(figsize=(4,4))
    plt.bar(["D_Former"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"D_Former mean ± 95% CI (n={num_runs})")
    plt.tight_layout()
    plt.show()

    return {"scores": scores, "mean_mse": mean_mse, "std_mse": sd_mse, "ci95": ci95, "ckpts": ckpts, "csv": csv_path}

def run_multi_seed_eval_only(
        seed_base=2025,
        num_runs=5,
        test_csv='./01_testing_data_16000.csv',
        seq_len=500,
        batch_size=32,
        ckpt_dir='Durability_D_Former_01',
        ckpt_tmpl='Durability_D_Former_seed{seed}.pth'
    ):

    os.makedirs(ckpt_dir, exist_ok=True)

    dataset_for_test_01 = StrainDataset(test_csv, 'test', seq_len, transforms=None)
    tt_loader = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)

    scores, ckpts, used_seeds = [], [], []

    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "=" * 60)
        print(f"[Eval] seed = {seed}")
        set_seed(seed)

        model = D_Former(d_model).to(device)

        ckpt_path = os.path.join(ckpt_dir, ckpt_tmpl.format(seed=seed))
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] ckpt not found: {ckpt_path} -> skip this seed")
            continue

        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        _, test_mse = test(tt_loader, model, device)
        print(f"[Seed {seed}] TEST MSE={test_mse:.6f}")

        scores.append(test_mse)
        ckpts.append(ckpt_path)
        used_seeds.append(seed)

    if len(scores) == 0:
        print("[ERROR] No valid checkpoints found. Check ckpt_dir/ckpt_tmpl.")
        return None

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (D_Former, EVAL ONLY) =====")
    print(f"Runs (n)   : {len(scores)}")
    print(f"Test MSE   : {mean_mse:.6f} ± {ci95:.6f}  (95% CI)")
    print(f"Std (MSE)  : {sd_mse:.6f}")
    print(f"First ckpt : {ckpts[0]}")
    print(f"Last ckpt  : {ckpts[-1]}")

    csv_path = os.path.join(ckpt_dir, "scores_seed_eval.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i, (sd, mse, path) in enumerate(zip(used_seeds, scores, ckpts)):
            w.writerow([i, sd, mse, path])
    print(f"[INFO] Eval scores saved to: {csv_path}")

    plt.figure(figsize=(4,4))
    plt.bar(["D_Former"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"D_Former (n={len(scores)}) mean ± 95% CI (EVAL ONLY)")
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

# ------------------------------------
if __name__ == "__main__":
    # summary = run_multi_seed_transformer(
    #     seed_base=2025,
    #     num_runs=5,
    #     train_csv='./01_training_data_4000.csv',
    #     test_csv='./01_testing_data_16000.csv',
    #     seq_len=500,
    #     batch_size=32,
    #     save_dir='Durability_D_Former_01'
    # )
    summary = run_multi_seed_eval_only(
        seed_base=2025,
        num_runs=5,
        test_csv='./01_testing_data_16000.csv',
        seq_len=500,
        batch_size=32,
        ckpt_dir='Durability_D_Former_01',
        ckpt_tmpl='Durability_D_Former_seed{seed}.pth'
    )
    print(summary)
