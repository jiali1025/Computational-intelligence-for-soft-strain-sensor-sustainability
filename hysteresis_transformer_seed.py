import numpy as np
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import os, math, random

# --------------------- Dataset ---------------------
class StrainDataset(Dataset):
    def __init__(self, path, mode='train', sequence_length=500, transforms=None):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.seq_len = sequence_length

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data)
            features = data[1:, 3:4]
            target  = data[1:, 4:5]
            cyc_time = data[1:, 5:6]
            y = target.astype(float)
            self.y = torch.tensor(y)
            X = features.astype(float)
            self.X = torch.tensor(X)
            time = cyc_time.astype(float)
            self.time = torch.tensor(time)
        print('Finished reading the {} set of Strain Dataset ({} samples found)'.format(mode, len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if index >= self.seq_len - 1:
            i_start = index - self.seq_len + 1
            x = self.X[i_start:(index + 1), :]
            time = self.time[i_start:(index + 1), :]
        else:
            padding = self.X[0].repeat(self.seq_len - index - 1, 1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((padding, x), 0)
            padding_time = self.time[0].repeat(self.seq_len - index - 1, 1)
            time = self.time[0:(index + 1), :]
            time = torch.cat((padding_time, time), 0)
        return x, self.y[index], time

def train_dev_data(x_set):
    X, deX = [], []
    for i in range(len(x_set)):
        if i % 10 != 0:
            X.append(x_set[i])
        else:
            deX.append(x_set[i])
    return X, deX

# --------------------- data set ---------------------
batch_size = 32
torch.manual_seed(99)
dataset_for_train = StrainDataset('./cycle_180_190_time.csv', 'train', 200, transforms=None)
dataset_for_test  = StrainDataset('./cycle_190_195_test.csv',  'test',  200, transforms=None)
training_data, valid_data = train_dev_data(dataset_for_train)
model_tr_data   = DataLoader(training_data, batch_size=batch_size, shuffle=True,  drop_last=True)
print(len(model_tr_data))
model_dev_data  = DataLoader(valid_data,   batch_size=batch_size, shuffle=True,  drop_last=True)
model_tt_dataX  = DataLoader(dataset_for_test, batch_size=batch_size, shuffle=False, drop_last=True)

# --------------------- model ---------------------
class TransformerModel(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.prenet = nn.Linear(1, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1))

    def forward(self, src: Tensor) -> Tensor:
        src = self.prenet(src)
        output = self.transformer_encoder(src)
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

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_model = 100
nhead   = 10
d_hid   = 1024
nlayers = 12
dropout = 0

model   = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)

# ---------------------------------------
config = {
    'n_epochs': 15,
    'early_stop': 100,
    'save_path': 'Transformer/model_transformer_01.pth'
}
criterion = nn.MSELoss()

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
def get_cosine_schedule_with_warmup(
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(model_tr_data, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = AdamW(model.parameters(), lr=1e-5)
    warmup_steps = (len(dataset_for_train)//batch_size)*2
    total_steps  = (len(dataset_for_train)//batch_size)*config['n_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    min_mse = 2000
    loss_record = {'train': [], 'dev': []}
    lrs=[]
    early_stop_cnt = 0
    epoch = 0
    num_batche_tr = len(model_tr_data)
    while epoch < n_epochs:
        print(epoch)
        allbat_mse_loss = 0
        model.train()
        for x, y, t in model_tr_data:
            x = x.float().to(device)
            pred = model(x)
            y = y.float().to(device)
            mse_loss = criterion(pred, y)
            allbat_mse_loss += mse_loss
            mse_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

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
    plt.plot(lrs)
    plt.show()
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    num_batches = len(dv_set)
    for i, y, t in dv_set:
        x = i.float().to(device)
        with torch.no_grad():
            pred = model(x)
            y = y.float().to(device)
            mse_loss = criterion(pred, y)
        total_loss += mse_loss.detach().cpu().item()
    avg_loss = total_loss / num_batches
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
            batch_loss = criterion(pred, y)
            preds.append(pred.detach().cpu())
        test_loss += batch_loss.detach().cpu().item()
    avg_loss_test = test_loss / num_batches
    preds = torch.cat(preds, dim=0).numpy()
    print('Finished testing predictions!')
    return preds, avg_loss_test

# -----------------------------
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
    sd = float(np.std(arr, ddof=1)) if n>1 else 0.0
    try:
        from scipy import stats
        tval = stats.t.ppf(0.975, df=n-1) if n>1 else 1.96
    except Exception:
        tval = 1.96
    ci = tval * sd / math.sqrt(n) if n>1 else 0.0
    return mean, sd, ci

def run_multi_seed(seed_base=2025, num_runs=10, save_dir='Transformer'):
    os.makedirs(save_dir, exist_ok=True)
    scores = []
    ckpts = []
    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "="*60)
        print(f"[Run {i+1}/{num_runs}] seed = {seed}")
        set_seed(seed)

        model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)

        cfg = dict(config)
        cfg['save_path'] = os.path.join(save_dir, f"model_transformer_seed{seed}.pth")

        min_dev, rec = train(model_tr_data, model_dev_data, model, cfg, device)

        preds, test_mse = test(model_tt_dataX, model, device)
        print(f"[Seed {seed}] DEV best MSE={min_dev:.6f} | TEST MSE={test_mse:.6f}")
        scores.append(test_mse)
        ckpts.append(cfg['save_path'])

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (Transformer) =====")
    print(f"Runs (n): {num_runs}")
    print(f"Test MSE: {mean_mse:.6f} ± {ci95:.6f} (95% CI)")
    print(f"Std(MSE): {sd_mse:.6f}")
    print(f"ckpt example: {ckpts[0]} ... {ckpts[-1]}")

    csv_path = os.path.join(save_dir, "scores_seed2020.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i in range(num_runs):
            w.writerow([i, seed_base + i, scores[i], ckpts[i]])
    print(f"[INFO] Scores saved to: {csv_path}")

    return {"scores": scores, "mean_mse": mean_mse, "std_mse": sd_mse, "ci95": ci95, "ckpts": ckpts, "csv": csv_path}

def run_multi_seed_eval_only(seed_base=2025, num_runs=10, save_dir='Transformer',
                   ckpt_tmpl="model_transformer_seed{seed}.pth"):

    os.makedirs(save_dir, exist_ok=True)

    scores = []
    ckpts = []
    used_seeds = []

    for i in range(num_runs):
        seed = seed_base + i
        print("\n" + "="*60)
        print(f"[Run {i+1}/{num_runs}] seed = {seed}")
        set_seed(seed)

        model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)

        ckpt_path = os.path.join(save_dir, ckpt_tmpl.format(seed=seed))
        if not os.path.exists(ckpt_path):
            print(f"[WARNING] ckpt not found: {ckpt_path}  -> skip this seed")
            continue

        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

        preds, test_mse = test(model_tt_dataX, model, device)
        print(f"[Seed {seed}] TEST MSE={test_mse:.6f}")

        scores.append(test_mse)
        ckpts.append(ckpt_path)
        used_seeds.append(seed)

    if len(scores) == 0:
        print("[ERROR] No valid checkpoints found. Please check ckpt_tmpl / save_dir.")
        return None

    mean_mse, sd_mse, ci95 = mean_std_ci95(scores)
    print("\n===== Multi-seed Summary (Transformer, EVAL ONLY) =====")
    print(f"Runs (n): {len(scores)}")
    print(f"Test MSE: {mean_mse:.6f} ± {ci95:.6f} (95% CI)")
    print(f"Std(MSE): {sd_mse:.6f}")
    print(f"ckpt example: {ckpts[0]} ... {ckpts[-1]}")

    csv_path = os.path.join(save_dir, "scores_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "test_mse", "ckpt_path"])
        for i, (seed, mse, path) in enumerate(zip(used_seeds, scores, ckpts)):
            w.writerow([i, seed, mse, path])
    print(f"[INFO] Scores saved to: {csv_path}")

    plt.figure(figsize=(4,4))
    plt.bar(["Transformer"], [mean_mse], yerr=[ci95], capsize=6)
    plt.ylabel("Test MSE")
    plt.title(f"Transformer mean ± 95% CI (n={len(scores)})")
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

# ------------------------------------------
if __name__ == "__main__":
    summary = run_multi_seed(
        seed_base=2022,
        num_runs=5,
        save_dir='Transformer'
    )

    # summary = run_multi_seed_eval_only(
    #     seed_base=2022,
    #     num_runs=5,
    #     save_dir='Transformer',
    #     ckpt_tmpl="model_transformer_seed{seed}.pth"
    # )
    print(summary)
