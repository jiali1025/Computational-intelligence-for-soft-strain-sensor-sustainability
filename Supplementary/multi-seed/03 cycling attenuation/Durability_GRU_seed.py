import torch
torch.cuda.empty_cache()
import os, csv, math, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()
print(f"[INFO] device = {device}")

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

def train_dev_split_by_10(x_set):
    X_tr, X_dv = [], []
    for i in range(len(x_set)):
        if i % 10 != 0:
            X_tr.append(x_set[i])
        else:
            X_dv.append(x_set[i])
    return X_tr, X_dv

class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1, batch_first=True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 8)
        self.linear2 = nn.Linear(8, output_size)
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, _x):
        x, _ = self.gru(_x)
        x = self.linear1(x[:, -1, :])
        x = self.linear2(x)
        return x

model = GruRNN(2, 64)
loss_function = nn.MSELoss()

def dev(dv_loader, model, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(dv_loader)
    with torch.no_grad():
        for x, y, t, r, c in dv_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            mse_loss = loss_function(pred, y)
            total_loss += mse_loss.detach().cpu().item()
    return total_loss / max(num_batches, 1)

# training
def train(tr_loader, dv_loader, model, config, device, save_path=None):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(model.parameters(), **config['optim_hparas'])
    early_stop_patience = config['early_stop']

    min_mse = float('inf')
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        num_batches = len(tr_loader)

        for x, y, t, r, c in tr_loader:
            optimizer.zero_grad()
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            mse_loss = loss_function(pred, y)
            mse_loss.backward()
            optimizer.step()
            total_train_loss += mse_loss.detach().cpu().item()

        avg_train_loss = total_train_loss / max(num_batches, 1)
        avg_dev_loss = dev(dv_loader, model, device)

        loss_record['train'].append(avg_train_loss)
        loss_record['dev'].append(avg_dev_loss)

        if avg_dev_loss < min_mse - 1e-8:
            min_mse = avg_dev_loss
            early_stop_cnt = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
        else:
            early_stop_cnt += 1

        print(f"[Epoch {epoch+1:03d}] train MSE={avg_train_loss:.6f} | dev MSE={avg_dev_loss:.6f} | best={min_mse:.6f} | es_cnt={early_stop_cnt}")

        if early_stop_cnt > early_stop_patience:
            print(f"[EarlyStop] Stop at epoch {epoch+1}")
            break

    return min_mse, loss_record

def test(tt_loader, model, device):
    model.eval()
    preds = []
    total_loss = 0.0
    num_batches = len(tt_loader)
    with torch.no_grad():
        for x, y, t, r, c in tt_loader:
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            loss = loss_function(pred, y)
            preds.append(pred.detach().cpu())
            total_loss += loss.detach().cpu().item()
    avg_loss = total_loss / max(num_batches, 1)
    preds = torch.cat(preds, dim=0).numpy() if len(preds) > 0 else np.array([])
    return preds, avg_loss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mean_std_ci95(values):
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    mean = float(vals.mean())
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    try:
        from scipy import stats
        tval = stats.t.ppf(0.975, df=n-1) if n > 1 else 0.0
    except Exception:
        tval = 1.96
    ci = tval * sd / math.sqrt(n) if n > 1 else 0.0
    return mean, sd, ci

def run_one_seed(seed, config, tr_loader, dv_loader, tt_loader, ckpt_dir):
    set_seed(seed)
    model = GruRNN(2, 64).to(device)
    ckpt_path = os.path.join(ckpt_dir, f"Durability_GruRNN_seed{seed}.pth")
    min_dev_mse, _ = train(tr_loader, dv_loader, model, config, device, save_path=ckpt_path)

    _, test_mse = test(tt_loader, model, device)
    print(f"[Seed {seed}] DEV best MSE={min_dev_mse:.6f} | TEST MSE={test_mse:.6f}")
    return test_mse, ckpt_path

def run_multi_seed(seed_base=2025, num_runs=10,
                   train_csv='./01_training_data_4000.csv',
                   test_csv='./01_testing_data_16000.csv',
                   seq_len=30,
                   batch_size=64,
                   save_dir="Durability_GruRNN_multi_seed"):
    # DataLoader
    os.makedirs(save_dir, exist_ok=True)
    dataset_tr = StrainDataset(train_csv, 'train', seq_len, transforms=None)
    dataset_tt = StrainDataset(test_csv, 'test', seq_len, transforms=None)

    train_list, dev_list = train_dev_split_by_10(dataset_tr)
    tr_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, drop_last=True)
    dv_loader = DataLoader(dev_list, batch_size=batch_size, shuffle=True, drop_last=True)
    tt_loader = DataLoader(dataset_tt, batch_size=batch_size, shuffle=False, drop_last=True)

    # configuration
    config = {
        'n_epochs': 120,
        'batch_size': batch_size,
        'optimizer': 'SGD',
        'optim_hparas': {'lr': 0.000001},
        'early_stop': 100,
    }

    scores = []
    ckpts = []
    for i in range(num_runs):
        seed = seed_base + i
        test_mse, ckpt_path = run_one_seed(seed, config, tr_loader, dv_loader, tt_loader, save_dir)
        scores.append(test_mse)
        ckpts.append(ckpt_path)

    # summary
    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (GruRNN) =====")
    print(f"Runs (n)   : {num_runs}")
    print(f"Test MSE   : {mean_mse:.6f} ± {ci95:.6f}  (95% CI)")
    print(f"Std (MSE)  : {sd_mse:.6f}")
    print(f"First ckpt : {ckpts[0]}")
    print(f"Last ckpt  : {ckpts[-1]}")


    csv_path = os.path.join(save_dir, "scores_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i in range(num_runs):
            w.writerow([i, seed_base + i, scores[i], ckpts[i]])
    print(f"[INFO] Scores saved to: {csv_path}")

    plt.figure(figsize=(4,4))
    plt.bar(["GruRNN"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"GruRNN (n={num_runs}) mean ± 95% CI")
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
def run_multi_seed_eval_only(
        seed_base=2025,
        num_runs=5,
        test_csv='./01_testing_data_16000.csv',
        seq_len=30,
        batch_size=64,
        ckpt_dir="Durability_GRU_01",
        ckpt_tmpl="Durability_GruRNN_seed{seed}.pth"
    ):
    os.makedirs(ckpt_dir, exist_ok=True)

    dataset_tt = StrainDataset(test_csv, 'test', seq_len, transforms=None)
    tt_loader = DataLoader(dataset_tt, batch_size=batch_size, shuffle=False, drop_last=True)

    scores, ckpts, used_seeds = [], [], []

    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "=" * 60)
        print(f"[Eval] seed = {seed}")
        set_seed(seed)

        model = GruRNN(2, 64).to(device)

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
    print("\n===== Multi-seed Summary (GruRNN, EVAL ONLY) =====")
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
    plt.bar(["GruRNN"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"GruRNN (n={len(scores)}) mean ± 95% CI (EVAL ONLY)")
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
# ==========================================
if __name__ == "__main__":
    # summary = run_multi_seed(
    #     seed_base=2025,
    #     num_runs=5,
    #     train_csv='./01_training_data_4000.csv',
    #     test_csv='./01_testing_data_16000.csv',
    #     seq_len=30,
    #     batch_size=64,
    #     save_dir="Durability_GRU_01"
    # )

    summary = run_multi_seed_eval_only(
        seed_base=2025,
        num_runs=5,
        test_csv='./01_testing_data_16000.csv',
        seq_len=30,
        batch_size=64,
        ckpt_dir="Durability_GRU_01",
        ckpt_tmpl="Durability_GruRNN_seed{seed}.pth"
    )

    print(summary)

