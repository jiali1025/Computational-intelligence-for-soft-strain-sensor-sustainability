import os, csv, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device()
torch.cuda.empty_cache()

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# -----------Dataset------------
class StrainDataset(Dataset):
    def __init__(self, path, mode='train', sequence_length=35):
        self.mode = mode
        self.seq_len = sequence_length
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
        data = np.array(data)
        R = data[1:, 3:4].astype(float)   # resistance
        eps = data[1:, 4:5].astype(float) # strain label
        t   = data[1:, 5:6].astype(float) # time

        self.X = torch.tensor(R)    # (N,1)
        self.y = torch.tensor(eps)  # (N,1)
        self.T = torch.tensor(t)    # (N,1)
        print(f'Finished reading {mode} set ({len(self.X)} samples)')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if index >= self.seq_len - 1:
            i0 = index - self.seq_len + 1
            x = self.X[i0:index+1, :]
            t = self.T[i0:index+1, :]
        else:
            pad = self.X[0].repeat(self.seq_len - index - 1, 1)
            x = torch.cat((pad, self.X[0:index+1, :]), dim=0)
            t = torch.cat((pad, self.T[0:index+1, :]), dim=0)
        return x, self.y[index], t

# Model: LSTM predicts ε̂, and also learns k
# --------------------
class PINNLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=256, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        self.k_raw = nn.Parameter(torch.tensor(0.1))
        self.softplus = nn.Softplus()

    @property
    def k(self):
        return self.softplus(self.k_raw)

    def forward(self, x):
        out, _ = self.lstm(x)
        eps_hat = self.head(out[:, -1, :])
        return eps_hat, self.k

# --------------------
# Physics residual: ΔR/R_ref - k * ε̂
# --------------------
def physics_residual_delta(R_seq, eps_hat, k):
    R_ref  = R_seq[:, 0, 0:1]   # (B,1)
    R_last = R_seq[:, -1, 0:1]  # (B,1)
    rel_change = (R_last - R_ref) / (R_ref + 1e-12)
    residual = rel_change - k * eps_hat
    return residual

# ----------Train / eval / test----------
@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    mse = nn.MSELoss(reduction='sum')
    total, n = 0.0, 0
    for x, y, _ in loader:
        x = x.float().to(device)
        y = y.float().to(device)
        eps_hat, _ = model(x)
        total += mse(eps_hat, y).item()
        n += y.numel()
    return total / max(1, n)

def train_one(model, train_loader, dev_loader, cfg):
    mse = nn.MSELoss()
    opt = getattr(torch.optim, cfg['optimizer'])(model.parameters(), **cfg['optim_hparas'])
    best_val = float('inf'); early = 0

    for ep in range(1, cfg['n_epochs']+1):
        model.train()
        for x, y, _ in train_loader:
            x = x.float().to(device)   # (B,L,1) R
            y = y.float().to(device)   # (B,1)   ε
            eps_hat, k = model(x)
            data_loss = mse(eps_hat, y)
            phys_loss = (physics_residual_delta(x, eps_hat, k)**2).mean()
            loss = data_loss + cfg['lambda_phys'] * phys_loss
            opt.zero_grad(); loss.backward(); opt.step()
        val = evaluate(dev_loader, model, device)
        if (ep % 5)==0 or ep==1:
            print(f"[Epoch {ep:03d}] val_mse={val:.6f} | k={model.k.item():.6f}")
        if val < best_val - 1e-12:
            best_val = val; early = 0
            torch.save(model.state_dict(), cfg['save_path'])
            with open(os.path.join(os.path.dirname(cfg['save_path']), 'learned_k.txt'), 'w') as f:
                f.write(f"{model.k.item():.8f}\n")
        else:
            early += 1
            if early > cfg['early_stop']:
                print(f"Early stop at epoch {ep}")
                break
    return best_val

@torch.no_grad()
def test(loader, model):
    model.eval()
    mse = nn.MSELoss(reduction='sum')
    total, n = 0.0, 0
    preds = []
    for x, y, _ in loader:
        x = x.float().to(device)
        y = y.float().to(device)
        eps_hat, _ = model(x)
        preds.append(eps_hat.detach().cpu())
        total += mse(eps_hat, y).item()
        n += y.numel()
    preds = torch.cat(preds, 0).numpy()
    return preds, total / max(1, n)


def build_loaders(train_csv, test_csv, seq_len, batch_size, dev_ratio=0.1, seed=0):
    tr_ds = StrainDataset(train_csv, 'train', seq_len)
    tt_ds = StrainDataset(test_csv,  'test',  seq_len)

    N = len(tr_ds)
    idx = np.arange(N)
    rng = np.random.default_rng(seed); rng.shuffle(idx)
    n_dev = max(1, int(round(N*dev_ratio)))
    dv_idx = idx[:n_dev]; tr_idx = idx[n_dev:]

    train_list = [tr_ds[i] for i in tr_idx]
    dev_list   = [tr_ds[i] for i in dv_idx]

    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True,  drop_last=True, generator=g)
    dev_loader   = DataLoader(dev_list,   batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = DataLoader(tt_ds,      batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, dev_loader, test_loader

def run_seed(seed, base_out, cfg, train_csv, test_csv):
    print("\n" + "="*70)
    print(f"[SEED {seed}]")
    set_global_seed(seed)
    outdir = os.path.join(base_out, f"seed_{seed}")
    os.makedirs(outdir, exist_ok=True)

    tr_loader, dv_loader, tt_loader = build_loaders(
        train_csv, test_csv, cfg['seq_len'], cfg['batch_size'], cfg['dev_ratio'], seed
    )

    model = PINNLSTM(1, cfg['hidden_size'], cfg['num_layers']).to(device)
    cfg_local = dict(cfg); cfg_local['save_path'] = os.path.join(outdir, "model.pth")
    best_val = train_one(model, tr_loader, dv_loader, cfg_local)

    # Reload the best checkpoint
    del model
    model = PINNLSTM(1, cfg['hidden_size'], cfg['num_layers']).to(device)
    state = torch.load(cfg_local['save_path'], map_location='cpu')
    model.load_state_dict(state); model.to(device)

    preds, test_mse = test(tt_loader, model)
    learned_k = float(model.k.item())

    # Save predictions
    with open(os.path.join(outdir, "pred.csv"), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['id','epsilon_hat'])
        for i,p in enumerate(preds): w.writerow([i, float(p)])

    # Save metrics
    with open(os.path.join(outdir, "metrics.csv"), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed','val_mse','test_mse','learned_k','lambda_phys'])
        w.writerow([seed, best_val, test_mse, learned_k, cfg['lambda_phys']])

    print(f"[SEED {seed}] val_mse={best_val:.6f} | test_mse={test_mse:.6f} | k={learned_k:.6f}")
    return {'seed':seed, 'val_mse':best_val, 'test_mse':test_mse, 'learned_k':learned_k}

def main():
    cfg = {
        'n_epochs': 120,
        'early_stop': 100,
        'optimizer': 'Adam',
        'optim_hparas': {'lr': 1e-4},
        'batch_size': 36,
        'seq_len': 35,
        'hidden_size': 256,
        'num_layers': 2,
        'lambda_phys': 5.0,   # physics loss weight; tune ~0.1 to 5
        'dev_ratio': 0.1,
    }
    train_csv = './cycle_180_190_time.csv'
    test_csv  = './cycle_190_195_test.csv'
    seeds = [0,1,2,3,4]
    base_out = 'runs_pinn_lstm_simple_lp5'
    os.makedirs(base_out, exist_ok=True)

    rows = []
    for s in seeds:
        rows.append(run_seed(s, base_out, cfg, train_csv, test_csv))

    summ = os.path.join(base_out, 'summary.csv')
    with open(summ, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed','val_mse','test_mse','learned_k'])
        for r in rows:
            w.writerow([r['seed'], r['val_mse'], r['test_mse'], r['learned_k']])
        w.writerow([])
        for key in ['val_mse','test_mse','learned_k']:
            arr = np.array([r[key] for r in rows], dtype=np.float64)
            w.writerow([f'__{key}_mean__', float(arr.mean())])
            w.writerow([f'__{key}_std__',  float(arr.std(ddof=1))])

    print(f"\n[DONE] Summary saved: {summ}")

if __name__ == '__main__':
    main()
