import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import csv
import os
import random

torch.cuda.empty_cache()

# ------------------------------------- Dataset-----------------------------------
class StrainDataset(Dataset):
    def __init__(self, path, mode='train', sequence_length=500, transforms=None):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.seq_len = sequence_length
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data)
            features = data[1::30, 2:5]
            target = data[1::30, 5:6]
            all_time = data[1::30, 2:3]
            resistance_f = data[1::30, 2:3]
            cycle_num = data[1::30, 3:4]

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
        print(f'Finished reading the {mode} set of Strain Dataset ({len(self.X)} samples found)')

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


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

# --------------------------------- Model-----------------------------------
class LSTMpred(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True,
                 batch_first=False, dropout=0, output_size=1):
        super().__init__()
        self.input_dim = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 128)
        self.linear2 = nn.Linear(128, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(self.num_layers, 36, self.hidden_size).to(device),
                torch.zeros(self.num_layers, 36, self.hidden_size).to(device))

    def forward(self, x):
        lstm_out, _ = self.lstm(x, self.hidden)
        pred = self.linear1(lstm_out[:, -1, :])
        pred = self.linear2(pred)
        return pred


loss_funtion = nn.MSELoss()

# ---------------------------- Train / Dev / Test---------------------------
def train(model_tr_data, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])
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

os.makedirs('BatchDiversity_LSTM_models/BatchDiversity_LSTM_01_1589Preds3', exist_ok=True)
config = {
    'n_epochs': 100,
    'batch_size': 36,
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.0001,
    },
    'early_stop': 100,
    'save_path': 'BatchDiversity_LSTM_models/BatchDiversity_LSTM_01_1589Preds3.pth'
}

dataset = StrainDataset('./Training_batchdiversity_1589.csv', 'train', 30, transforms=None)
dataset_for_test_01 = StrainDataset('./Testing_batchdiversity_10.csv', 'test', 30, transforms=None)

def train_dev_data(x_set):
    X, deX = [], []
    for i in range(len(x_set)):
        if i % 10 != 0:
            X.append(x_set[i])
        else:
            deX.append(x_set[i])
    return X, deX

training_data, valid_data = train_dev_data(dataset)
batch_size = 36

model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX01 = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)


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

    model = LSTMpred(input_size=3, hidden_size=256, num_layers=2, batch_first=True)
    model = model.to(device)

    base_ckpt = config['save_path']
    ckpt_path = base_ckpt.replace('.pth', f'_seed{seed}.pth')

    best_dev_mse, _loss_rec = train(model_tr_data, model_dev_data, model,
                                    {**config, 'save_path': ckpt_path}, device)

    del model
    model = LSTMpred(input_size=3, hidden_size=256, num_layers=2, batch_first=True).to(device)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    _preds, test_mse = test(model_tt_dataX01, model, device)

    print(f"[SEED {seed}] ckpt: {ckpt_path}")
    print(f"[SEED {seed}] best_dev_mse={best_dev_mse:.6f} | test_mse={test_mse:.6f}")
    return seed, best_dev_mse, test_mse, ckpt_path

def main():
    seeds = [42, 99, 3407, 2023, 7]

    results = []
    for s in seeds:
        seed, best_dev_mse, test_mse, ckpt_path = run_one_seed(s)
        results.append([seed, best_dev_mse, test_mse, ckpt_path])

    summary_csv = 'BatchDiversity_LSTM_multi_seed_summary.csv'
    with open(summary_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'best_dev_mse', 'test_mse', 'ckpt_path'])
        w.writerows(results)

    print("\n=== Multi-seed Summary ===")
    for row in results:
        print(f"seed={row[0]:>5} | best_dev_mse={row[1]:.6f} | test_mse={row[2]:.6f} | ckpt={row[3]}")
    print(f"\n[SUMMARY CSV] -> {summary_csv}")
    print("ALL Done!")

if __name__ == '__main__':
    main()
