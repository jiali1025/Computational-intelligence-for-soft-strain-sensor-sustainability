import os
import csv
import math
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

torch.cuda.empty_cache()

# Dataset
class StrainDataset(Dataset):
    """
    Turn a 1D signal into overlapping sequences of length `sequence_length`.
    For sample index `idx`, I return:
      x_seq: [seq_len, 1]  feature sequence ending at idx
      y:     [1]           target at idx
      t_seq: [seq_len, 1]  time sequence ending at idx (kept for compatibility; not used by the GRU here)
    """
    def __init__(self, path, mode='train', sequence_length=12, transforms=None, encoding='utf-8'):
        self.mode = mode
        self.path = path
        self.seq_len = sequence_length
        self.transforms = transforms

        with open(path, 'r', encoding=encoding) as fp:
            data = list(csv.reader(fp))

        data = np.array(data)

        features = data[1:, 3:4]
        target   = data[1:, 4:5]
        cyc_time = data[1:, 5:6]

        self.X = torch.tensor(features.astype(float))
        self.y = torch.tensor(target.astype(float))
        self.T = torch.tensor(cyc_time.astype(float))

        print(f'Finished reading "{mode}" dataset from {path} ({len(self.X)} samples)')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # Build a window ending at `index`
        if index >= self.seq_len - 1:
            i_start = index - self.seq_len + 1
            x = self.X[i_start:(index + 1), :]
            t = self.T[i_start:(index + 1), :]
        else:
            pad_len = self.seq_len - index - 1

            x_pad = self.X[0].repeat(pad_len, 1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((x_pad, x), dim=0)
            t_pad = self.T[0].repeat(pad_len, 1)
            t = self.T[0:(index + 1), :]
            t = torch.cat((t_pad, t), dim=0)

        return x, self.y[index], t

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()
# Model
class GruRNN(nn.Module):
    """
    GRU encoder + 2-layer MLP head.
    I take the last hidden state as the sequence representation.
    """
    def __init__(self, input_size, hidden_size=64, output_size=1, num_layers=2, batch_first=True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.linear1 = nn.Linear(hidden_size, 16)
        self.linear2 = nn.Linear(16, output_size)

    def forward(self, x):
        # x: [B, T, C]
        h, _ = self.gru(x)
        h_last = h[:, -1, :]          # [B, hidden]
        out = self.linear2(self.linear1(h_last))
        return out

loss_function = nn.MSELoss()

# Train / Dev / Test
def train(train_loader, dev_loader, model, config, device):
    """
    Train the model and save the best checkpoint based on dev MSE.
    """
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])

    best_dev = float('inf')
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0

    num_batches_tr = len(train_loader)
    print(f"[INFO] #train batches: {num_batches_tr}")

    for epoch in range(n_epochs):
        model.train()
        total_tr_loss = 0.0

        for x, y, _t in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            mse = loss_function(pred, y)
            mse.backward()
            optimizer.step()

            total_tr_loss += mse.detach().cpu().item()

        avg_tr = total_tr_loss / max(1, num_batches_tr)
        loss_record['train'].append(avg_tr)

        dev_mse = dev(dev_loader, model, device)
        loss_record['dev'].append(dev_mse)

        print(f"[Epoch {epoch+1:03d}] train MSE={avg_tr:.6f} | dev MSE={dev_mse:.6f}")

        # Save best-by-dev
        if dev_mse < best_dev:
            best_dev = dev_mse
            os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
            torch.save(model.state_dict(), config['save_path'])
            print(f"[INFO] Saved best checkpoint -> {config['save_path']} (dev MSE={best_dev:.6f})")
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt > config['early_stop']:
            print(f"[INFO] Early stop triggered (patience={config['early_stop']})")
            break

    print(f"[INFO] Training finished. Best dev MSE={best_dev:.6f}")
    return best_dev, loss_record


def dev(dev_loader, model, device):
    """
    Compute average dev MSE.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dev_loader)

    with torch.no_grad():
        for x, y, _t in dev_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            mse = loss_function(pred, y)
            total_loss += mse.detach().cpu().item()

    return total_loss / max(1, num_batches)

def test(test_loader, model, device):
    """
    Evaluate on the test set and return predictions + average test MSE.
    """
    model.eval()
    preds = []
    total_loss = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for x, y, _t in test_loader:
            x = x.float().to(device)
            y = y.float().to(device)

            pred = model(x)
            mse = loss_function(pred, y)

            preds.append(pred.detach().cpu())
            total_loss += mse.detach().cpu().item()

    preds = torch.cat(preds, dim=0).numpy()
    avg_mse = total_loss / max(1, num_batches)
    print("[INFO] Testing finished.")
    return preds, avg_mse

# Plots
def plot_learning_curve(loss_record, title=''):
    """
    Plot train/dev loss curves across epochs.
    """
    epochs = len(loss_record['train'])
    x = list(range(epochs))
    figure(figsize=(6, 4))
    plt.plot(x, loss_record['train'], label='train')
    plt.plot(x, loss_record['dev'], label='dev')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'Learning Curve {title}'.strip())
    plt.legend()
    plt.tight_layout()
    plt.show()

# Data split
def train_dev_data(dataset):
    train_items, dev_items = [], []
    for i in range(len(dataset)):
        if i % 10 != 0:
            train_items.append(dataset[i])
        else:
            dev_items.append(dataset[i])
    return train_items, dev_items

# Reproducibility
def set_seed(seed: int):
    # Set RNG seeds for reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_std_ci95(values):
    """
    Return mean, sample std (ddof=1), and 95% CI half-width.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)

    mean = float(arr.mean())
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    try:
        from scipy import stats
        tval = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
    except Exception:
        tval = 1.96

    ci = tval * sd / math.sqrt(n) if n > 1 else 0.0
    return mean, sd, ci


def _normalize_seeds(seeds):

    if seeds is None:
        raise ValueError("Please provide a seed list, e.g., seeds=[2025, 2027, 3001].")

    if not isinstance(seeds, (list, tuple)):
        raise TypeError("seeds must be a list/tuple, e.g., seeds=[2025, 2027, 2033].")

    seeds = [int(s) for s in seeds]
    seen = set()
    uniq = []
    for s in seeds:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq

# Multi-seed runners
def run_multi_seed_train_and_test(
        seeds,
        config,
        train_csv,
        test_csv,
        seq_len,
        batch_size,
        save_dir='GRU',
        train_encoding='utf-8',
        test_encoding='iso-8859-1',
    ):

    seeds = _normalize_seeds(seeds)
    os.makedirs(save_dir, exist_ok=True)

    dataset_tr = StrainDataset(train_csv, mode='train', sequence_length=seq_len, encoding=train_encoding)
    dataset_tt = StrainDataset(test_csv,  mode='test',  sequence_length=seq_len, encoding=test_encoding)

    train_items, dev_items = train_dev_data(dataset_tr)

    train_loader = DataLoader(train_items, batch_size=batch_size, shuffle=True,  drop_last=True)
    dev_loader   = DataLoader(dev_items,   batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(dataset_tt,  batch_size=batch_size, shuffle=False, drop_last=True)

    scores, ckpts, used_seeds = [], [], []

    for run_id, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 60)
        print(f"[RUN {run_id}/{len(seeds)}] seed={seed}")
        set_seed(seed)

        model = GruRNN(input_size=1, hidden_size=64).to(device)

        cfg = dict(config)
        cfg['save_path'] = os.path.join(save_dir, f"model_gru_seed{seed}.pth")

        best_dev, loss_rec = train(train_loader, dev_loader, model, cfg, device)

        state = torch.load(cfg['save_path'], map_location=device)
        model.load_state_dict(state)

        preds, test_mse = test(test_loader, model, device)

        print(f"[RESULT] seed={seed} | best dev MSE={best_dev:.6f} | test MSE={test_mse:.6f}")

        scores.append(test_mse)
        ckpts.append(cfg['save_path'])
        used_seeds.append(seed)

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (TRAIN+TEST) =====")
    print(f"Seeds (n) : {len(scores)}")
    print(f"Test MSE  : {mean_mse:.6f} ± {ci95:.6f} (95% CI)")
    print(f"Std (MSE) : {sd_mse:.6f}")

    # Save per-seed results
    out_csv = os.path.join(save_dir, "scores_seed.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i, (sd, mse, path) in enumerate(zip(used_seeds, scores, ckpts)):
            w.writerow([i, sd, mse, path])
    print(f"[INFO] Saved scores -> {out_csv}")

    # Error bar plot
    plt.figure(figsize=(4, 4))
    plt.bar(["GRU"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"GRU mean ± 95% CI (n={len(scores)})")
    plt.tight_layout()
    plt.show()

    return {
        "seeds": used_seeds,
        "scores": scores,
        "mean_mse": mean_mse,
        "std_mse": sd_mse,
        "ci95": ci95,
        "ckpts": ckpts,
        "csv": out_csv
    }

def run_multi_seed_eval_only(
        seeds,
        test_csv,
        seq_len,
        batch_size,
        save_dir='GRU',
        ckpt_tmpl="model_gru_seed{seed}.pth",
        test_encoding='iso-8859-1',
        strict_missing=False,
    ):
    seeds = _normalize_seeds(seeds)
    os.makedirs(save_dir, exist_ok=True)

    dataset_tt = StrainDataset(test_csv, mode='test', sequence_length=seq_len, encoding=test_encoding)
    test_loader = DataLoader(dataset_tt, batch_size=batch_size, shuffle=False, drop_last=True)

    scores, ckpts, used_seeds = [], [], []

    for run_id, seed in enumerate(seeds, start=1):
        print("\n" + "=" * 60)
        print(f"[EVAL {run_id}/{len(seeds)}] seed={seed}")
        set_seed(seed)

        model = GruRNN(input_size=1, hidden_size=64).to(device)

        ckpt_path = os.path.join(save_dir, ckpt_tmpl.format(seed=seed))
        if not os.path.exists(ckpt_path):
            msg = f"[WARNING] checkpoint not found -> {ckpt_path}"
            if strict_missing:
                raise FileNotFoundError(msg)
            print(msg + " (skip)")
            continue

        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        preds, test_mse = test(test_loader, model, device)
        print(f"[RESULT] seed={seed} | test MSE={test_mse:.6f}")

        # Save predictions
        pred_csv = os.path.join(save_dir, f"preds_seed{seed}.csv")
        with open(pred_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_idx", "pred"])
            for idx, p in enumerate(preds):
                w.writerow([idx, float(p)])
        print(f"[INFO] Saved predictions -> {pred_csv}")

        scores.append(test_mse)
        ckpts.append(ckpt_path)
        used_seeds.append(seed)

    if len(scores) == 0:
        print("[ERROR] No valid checkpoints were evaluated. Please check save_dir/ckpt_tmpl.")
        return None

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (EVAL ONLY) =====")
    print(f"Seeds (n) : {len(scores)}")
    print(f"Test MSE  : {mean_mse:.6f} ± {ci95:.6f} (95% CI)")
    print(f"Std (MSE) : {sd_mse:.6f}")

    out_csv = os.path.join(save_dir, "scores_seed_eval.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for run_id, (seed, mse, path) in enumerate(zip(used_seeds, scores, ckpts), start=1):
            w.writerow([run_id, seed, mse, path])
    print(f"[INFO] Saved eval scores -> {out_csv}")

    # Error bar plot
    plt.figure(figsize=(4, 4))
    plt.bar(["GRU"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"GRU mean ± 95% CI (n={len(scores)}) [EVAL ONLY]")
    plt.tight_layout()
    plt.show()

    return {
        "seeds": used_seeds,
        "scores": scores,
        "mean_mse": mean_mse,
        "std_mse": sd_mse,
        "ci95": ci95,
        "ckpts": ckpts,
        "csv_path": out_csv
    }

# Main
if __name__ == "__main__":
    seeds = [36, 3011, 99, 2025, 2029]

    config = {
        'n_epochs': 100,
        'batch_size': 36,
        'optimizer': 'Adam',
        'optim_hparas': {'lr': 0.0001},
        'early_stop': 100,
        'save_path': 'GRU1/model_gru_placeholder.pth'
    }

    train_csv = './cycle_180_190_time.csv'
    test_csv  = './cycle_190_195_test.csv'
    seq_len   = 35
    batch_size = 36

    # Option 1: eval-only
    summary = run_multi_seed_eval_only(
        seeds=seeds,
        test_csv=test_csv,
        seq_len=seq_len,
        batch_size=batch_size,
        save_dir='GRU1',
        ckpt_tmpl="model_gru_seed{seed}.pth",
        strict_missing=False
    )

    # # Option 2: train+test
    # summary = run_multi_seed_train_and_test(
    #     seeds=seeds,
    #     config=config,
    #     train_csv=train_csv,
    #     test_csv=test_csv,
    #     seq_len=seq_len,
    #     batch_size=batch_size,
    #     save_dir='GRU1'
    # )

    print(summary)
