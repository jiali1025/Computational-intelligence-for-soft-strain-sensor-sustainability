# pytorch
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import time
from torch.utils.data import BatchSampler
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# data preprocess
import numpy as np
import csv
import math
import os
torch.cuda.empty_cache()

class StrainDataset(Dataset):
    def __init__(self, path, mode='train',sequence_length=12, transforms=None):
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
            t = torch.cat((padding, t), 0)
        return x, self.y[index], t

def get_device():
    return 'cuda'if torch.cuda.is_available() else 'cpu'
device = get_device()

class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=2, batch_first=True):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 16)
        self.linear2 = nn.Linear(16, output_size)
        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, _x):

        x, _ = self.gru(_x)
        x = self.linear1(x[:, -1, :])
        x = self.linear2(x)
        return x

model = GruRNN(1, 64)
loss_funtion = nn.MSELoss()

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
        for x, y, t in model_tr_data:
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
    for i, y, t in dv_set:
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
    for x, y, t in tt_set:
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
# os.makedirs('gru_models/model_gru_01', exist_ok=True)
config = {
        'n_epochs': 100,
        'batch_size': 36,
        'optimizer': 'Adam',
        'optim_hparas': {
        'lr': 0.0001,
        },
        'early_stop': 100,
        'save_path': 'GRU/model_gru_01.pth'
}
dataset = StrainDataset('./cycle_180_190_time.csv', 'train', 35, transforms=None)
dataset_tt = StrainDataset('./cycle_190_195_test.csv', 'test', 35, transforms=None)

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
model_tt_data = DataLoader(dataset_tt, batch_size=batch_size, shuffle=False, drop_last=True)
print(len(model_tr_data))
print('done')
model = model.to(device)

# # start training
# model_loss, model_loss_record = train(model_tr_data, model_dev_data, model, config, device)
#
# total_params = sum(p.numel() for p in model.parameters())
# print('total params = ', total_params)
#
# print(model_loss)
# len_tr = model_loss_record['train']
# print(len_tr)
# len_dev = model_loss_record['dev']
# print(len(len_dev))
# plot_learning_curve(model_loss_record, title='GRU model')

del model
model = GruRNN(1, 64)
model = model.to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)

def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'strain'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
preds, test_loss = test(model_tt_data , model, device)

# save_pred(preds, 'pred_gru_01.csv')
print(test_loss)
# print('ALL Done!')