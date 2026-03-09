import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import csv
import os
torch.cuda.empty_cache()

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

def get_device():
    return 'cuda'if torch.cuda.is_available() else 'cpu'
device = get_device()

class _NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_dim=256, theta_dim=64, backcast_size=None, forecast_size=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, theta_dim)
        )
        self.backcast = nn.Linear(theta_dim, backcast_size if backcast_size is not None else input_size)
        self.forecast = nn.Linear(theta_dim, forecast_size)
    def forward(self, x):  # x: [B, input_size]
        theta = self.fc(x)
        return self.backcast(theta), self.forecast(theta)

class NBeatsRegressor(nn.Module):

    def __init__(self, input_dim=2, seq_len=30, stacks=3, hidden_dim=256, theta_dim=64):
        super().__init__()
        self.input_size = input_dim * seq_len
        self.blocks = nn.ModuleList([
            _NBeatsBlock(self.input_size, hidden_dim, theta_dim,
                         backcast_size=self.input_size, forecast_size=1)
            for _ in range(stacks)
        ])
    def forward(self, x):             # x: [B, T, F]
        B, T, F = x.shape
        z = x.reshape(B, T * F)
        residual = z
        forecast_sum = 0
        for blk in self.blocks:
            backcast, forecast = blk(residual)
            residual = residual - backcast
            forecast_sum = forecast_sum + forecast
        return forecast_sum           # [B, 1]

model = NBeatsRegressor(input_dim=2, seq_len=30, stacks=3, hidden_dim=128)
loss_funtion = nn.MSELoss()
# training
def train(model_tr_data,dv_set,model,config,device):
    n_epochs = config['n_epochs']
    # Setup optimizer
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
            torch.save(model.state_dict(), config ['save_path'])
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

# setup hyper-parameters
os.makedirs('Durability_NBeats_01', exist_ok=True)
config = {
        'n_epochs': 100,
        'batch_size': 36,
        'optimizer': 'SGD',
        'optim_hparas': {
        'lr': 1e-6,
        },
        'early_stop': 100,
        'save_path': 'Durability_NBeats_01/Durability_NBeats_01.pth'
}
dataset = StrainDataset('./01_training_data_4000.csv', 'train', 30, transforms=None)
dataset_for_test_01 = StrainDataset('./01_testing_data_16000.csv', 'test', 30, transforms=None)
print(len(dataset))

# splitting training data into train & dev sets
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
training_data ,valid_data = train_dev_data(dataset)

batch_size = 36
torch.manual_seed(99)

model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX01 = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)

print('done')
model = model.to(device)

# # start training
# model_loss, model_loss_record = train(model_tr_data, model_dev_data,model, config, device)
# total_params = sum(p.numel() for p in model.parameters())
# print('total params = ', total_params)
# print(model_loss)
# len_tr = model_loss_record['train']
# print(len_tr)
# len_dev = model_loss_record['dev']
# print(len(len_dev))

del model
model = NBeatsRegressor(input_dim=2, seq_len=30, stacks=3, hidden_dim=128)
model = model.to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)

def save_pred(preds, file):
    '''Save predictions to specified file'''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'strain'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds01, test_loss01 = test(model_tt_dataX01, model, device)
print(preds01)
print(test_loss01)
# save_pred(preds01, 'Durability_NBeats_01_Preds.csv')  # save prediction file to pred.csv
