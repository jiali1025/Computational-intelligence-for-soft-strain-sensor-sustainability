import csv
import os
import math
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import Dataset, DataLoader
import time as ttt
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
            x      = self.X[i_start:(index + 1), :]
            time   = self.time[i_start:(index + 1), :]
            resist = self.resis_f[i_start:(index + 1), :]
            cycle  = self.cyc_n[i_start:(index + 1), :]
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

def train_dev_data(x_set):
    X, deX = [], []
    for i in range(len(x_set)):
        if i % 10 != 0:
            X.append(x_set[i])
        else:
            deX.append(x_set[i])
    return X, deX


torch.cuda.empty_cache()
batch_size = 32

dataset_for_train   = StrainDataset('./Training_batchdiversity_1589.csv', 'train', 500, transforms=None)
dataset_for_test_01 = StrainDataset('./Testing_batchdiversity_10.csv', 'test', 500, transforms=None)
training_data, valid_data = train_dev_data(dataset_for_train)
model_tr_data   = DataLoader(training_data, batch_size=batch_size, shuffle=True,  drop_last=True)
print(len(model_tr_data))
model_dev_data  = DataLoader(valid_data,   batch_size=batch_size, shuffle=True,  drop_last=True)
model_tt_dataX01= DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.prenet = nn.Linear(3, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)  # batch_first=True
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

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

        # encoding
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
d_model = 12
nhead   = 4
d_hid   = 128
nlayers = 10
dropout = 0
# model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)

os.makedirs('BatchDiversity_Transformer_models', exist_ok=True)
config = {
    'n_epochs': 100,
    'early_stop': 100,
    'save_path': 'BatchDiversity_Transformer_models/BatchDiversity_Transformer_01.pth'
}
criterion = nn.MSELoss()

class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, predictions, targets):
        return torch.mean(torch.abs((targets - predictions) / (50)))

criterion1 = MAPELoss()

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
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(model_tr_data, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = AdamW(model.parameters(), lr=1e-5)
    warmup_steps = (len(dataset_for_train)//batch_size)*2
    total_steps  = (len(dataset_for_train)//batch_size)*config['n_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    min_mse = 2000
    loss_record = {'train': [], 'dev': []}
    lrs = []
    early_stop_cnt = 0
    epoch = 0
    num_batche_tr = len(model_tr_data)
    t1 = ttt.time()
    while epoch < n_epochs:
        t0 = ttt.time()
        print(epoch)
        allbat_mse_loss = 0
        model.train()
        for x, y, t, r, c in model_tr_data:
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
        time_elapsed = ttt.time() - t0
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print(evepoch_mse_loss)

        loss_record['train'].append(evepoch_mse_loss.detach().cpu().item())
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {},loss = {:.4f}'.format((epoch + 1), (min_mse)))
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
    time_stopped = ttt.time() - t1
    print('Training 100 epoches completed in {:.0f}m {:.0f}s'.format(time_stopped // 60, time_stopped % 60))
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
            mse_loss = criterion(pred, y)
        total_loss += mse_loss.detach().cpu().item()
    avg_loss = total_loss / num_batches
    return avg_loss

def plot_learning_curve(loss_record, title=''):
    total_epochs = len(loss_record['train'])
    x_1 = range(total_epochs)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'],   c='tab:cyan', label='dev')
    plt.ylim(0.0, 1000.0)
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

def test(tt_set, model, device):
    model.eval()
    preds = []
    targets = []
    test_loss = 0.0
    test_mape_loss = 0.0
    num_batches = len(tt_set)
    for x, y, t, r, c in tt_set:
        x = x.float().to(device)
        y = y.float().to(device)
        with torch.no_grad():
            pred = model(x)
            batch_loss = criterion(pred, y)
            batch_mape_loss = criterion1(pred, y)
            targets.append(y.detach().cpu())
            preds.append(pred.detach().cpu())
        test_loss += batch_loss.detach().cpu().item()
        test_mape_loss += batch_mape_loss.detach().cpu().item()
    avg_loss_test = test_loss / num_batches
    avg_loss_mape = test_mape_loss / num_batches
    preds = torch.cat(preds, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    print('Finished testing predictions!')
    return preds, targets, avg_loss_test, avg_loss_mape


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_one_seed(seed: int):
    print(f"\n[SEED] = {seed}")
    set_seed(seed)

    model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)

    base_ckpt_path = config['save_path']
    seed_ckpt_path = base_ckpt_path.replace('.pth', f'_seed{seed}.pth')

    best_dev_mse, _loss_record = train(model_tr_data, model_dev_data, model,
                                       {**config, 'save_path': seed_ckpt_path}, device)

    del model
    model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)
    ckpt = torch.load(seed_ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt)
    _preds, _targets, test_mse, test_mape = test(model_tt_dataX01, model, device)

    print(f"[SEED {seed}] saved ckpt: {seed_ckpt_path}")
    print(f"[SEED {seed}] best_dev_mse={best_dev_mse:.6f} | test_mse={test_mse:.6f} | test_mape={test_mape:.6f}")
    return seed, best_dev_mse, test_mse, test_mape, seed_ckpt_path

def main():
    seeds = [2025, 2026, 2027, 2028, 2029]

    summary_rows = []
    for s in seeds:
        seed, best_dev_mse, test_mse, test_mape, ckpt_path = run_one_seed(s)
        summary_rows.append([seed, best_dev_mse, test_mse, test_mape, ckpt_path])

    summary_csv = 'BatchDiversity_D_Former_multi_seed_summary.csv'
    with open(summary_csv, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['seed', 'best_dev_mse', 'test_mse', 'test_mape', 'ckpt_path'])
        writer.writerows(summary_rows)

    print(f"\n[SUMMARY] saved -> {summary_csv}")
    print("ALL Done!")

if __name__ == "__main__":
    main()
