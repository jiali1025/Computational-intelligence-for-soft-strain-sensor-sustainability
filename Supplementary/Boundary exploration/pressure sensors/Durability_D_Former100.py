import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time as ttt
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import AdamW
import causal_convolution_layer
from RoFormer import modeling_roformer, RoFormerConfig
import os
class StrainDataset(Dataset):
    def __init__(self, path, mode='train', sequence_length=500, transforms=None):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.seq_len = sequence_length

        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data)
            features = data[1::, 2:4]
            target = data[1::, 4:5]
            all_time = data[1::, 2:3]
            resistance_f = data[1::, 2:3]
            cycle_num = data[1::, 3:4]

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

            self.y_mean = y.mean()
            self.y_std = y.std()
            y = (y - self.y_mean) / self.y_std
            self.y = torch.tensor(y)
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
batch_size = 32
torch.manual_seed(99)
dataset_for_train = StrainDataset('./pressure_train_cycles_0-99.csv', 'train', 500, transforms=None)
dataset_for_test_01 = StrainDataset('./pressure_test_cycles_100-2478.csv', 'test', 500, transforms=None)
training_data, valid_data = train_dev_data(dataset_for_train)
model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
print(len(model_tr_data))
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX01 = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)

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
d_model = 12

model = D_Former(d_model).to(device)

os.makedirs('Pressure100_D_Former_01', exist_ok=True)
config = {
    'n_epochs': 120,
    'early_stop': 100,
    'save_path': 'Pressure100_D_Former_01/Durability_D_Former_01.pth'
}
criterion = nn.MSELoss()
class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        loss = torch.mean(torch.abs((targets - predictions) / (2000)))
        return loss

criterion1 = MAPELoss()

import math
import torch
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

# training
def train(model_tr_data, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4)
    warmup_steps = (len(dataset_for_train)//batch_size)*2
    total_steps = (len(dataset_for_train)//batch_size)*config['n_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    min_mse = 140000
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

# model_loss, model_loss_record = train(model_tr_data, model_dev_data, model, config, device)
# total_params = sum(p.numel() for p in model.parameters())
# print('total params = ', total_params)
# print(model_loss)
# print(model_loss_record['train'])
# print(model_loss_record['dev'])

# def test(tt_set, model, device):
#     model.eval()
#     preds = []
#     targets = []
#     test_loss = 0.0
#     test_mape_loss = 0.0
#     num_batches = len(tt_set)
#     for x, y, t, r, c in tt_set:
#         x = x.float().to(device)
#         y = y.float().to(device)
#         with torch.no_grad():
#             pred = model(x)
#             batch_loss = criterion(pred, y)
#             batch_mape_loss = criterion1(pred, y)
#             targets.append(y.detach().cpu())
#             preds.append(pred.detach().cpu())
#         test_loss += batch_loss.detach().cpu().item()
#         test_mape_loss += batch_mape_loss.detach().cpu().item()
#     avg_loss_test = test_loss / num_batches
#     avg_loss_mape = test_mape_loss / num_batches
#     preds = torch.cat(preds, dim=0).numpy()
#     targets = torch.cat(targets, dim=0).numpy()
#
#     preds = preds * dataset_for_train.y_std + dataset_for_train.y_mean
#     targets = targets * dataset_for_train.y_std + dataset_for_train.y_mean
#     print('Finished testing predictions!')
#     return preds, targets, avg_loss_test, avg_loss_mape

del model
model = D_Former(d_model).to(device)
model = model.to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)

# preds01, targets01, test_loss01, test_mape01 = test(model_tt_dataX01, model, device)
#
# print(preds01)
# print(test_loss01)
# print(test_mape01)

def test_real(tt_loader, model, device):
    model.eval()
    ds_test = tt_loader.dataset

    y_mean_train = torch.tensor(float(dataset_for_train.y_mean), device=device)
    y_std_train  = torch.tensor(float(dataset_for_train.y_std),  device=device)
    y_mean_test  = torch.tensor(float(ds_test.y_mean), device=device)
    y_std_test   = torch.tensor(float(ds_test.y_std),  device=device)

    preds_raw = []
    targets_raw = []

    sse = 0.0
    sae_div2000 = 0.0
    n = 0

    with torch.no_grad():
        for x, y, t, r, c in tt_loader:
            x = x.float().to(device)
            y = y.float().to(device)

            pred = model(x)

            pred_pa = pred * y_std_train + y_mean_train
            y_pa    = y    * y_std_test  + y_mean_test

            err = pred_pa - y_pa

            sse += (err ** 2).sum().item()
            sae_div2000 += (err.abs() / 2000.0).sum().item()
            n += err.numel()

            preds_raw.append(pred_pa.detach().cpu())
            targets_raw.append(y_pa.detach().cpu())

    preds_raw = torch.cat(preds_raw, dim=0).numpy()
    targets_raw = torch.cat(targets_raw, dim=0).numpy()

    test_mse_raw = sse / max(1, n)
    test_mae_div2000_raw = sae_div2000 / max(1, n)

    print('Finished testing predictions!')
    return preds_raw, targets_raw, test_mse_raw, test_mae_div2000_raw


preds01_raw, targets01_raw, test_mse_raw, test_mae2000_raw = test_real(model_tt_dataX01, model, device)

# print("Test MSE (pa^2, RAW):", test_mse_raw)
print("Test mean(|err|/2000) (RAW):", test_mae2000_raw)

def save_pred(preds, targets, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'prediction', 'target'])
        for i, (p, t) in enumerate(zip(preds, targets)):
            writer.writerow([i, p, t])

# save_pred(preds01_raw, targets01_raw, 'Pressure100_D_Former_01_Preds1.csv')
