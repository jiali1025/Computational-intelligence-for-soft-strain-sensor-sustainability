import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from torch.utils.data import Dataset, DataLoader
import time as ttt
import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import AdamW
import causal_convolution_layer
from RoFormer import modeling_roformer, RoFormerConfig
import os
import math
import random

# --------------------- Dataset ---------------------
class StrainDataset(Dataset):

    def __init__(self, path, mode='train', sequence_length=500, transforms=None):
        """
           CSV dataset for sequence modeling.
        """
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

            # standardization for input features
            self.X[:, :] = (self.X[:, :] - self.X[:, :].mean(dim=0, keepdim=True)) \
                           / (self.X[:, :].std(dim=0, keepdim=True) + 1e-8)
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

# Split a dataset into train/dev subsets by index.
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

# --------------------- Device & seeds ---------------------
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 99
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# --------------------- DataLoader ---------------------
batch_size = 32
dataset_for_train = StrainDataset('./01_training_data_4000.csv', 'train', 500, transforms=None)
dataset_for_test_01 = StrainDataset('./01_testing_data_16000.csv', 'test', 500, transforms=None)
training_data, valid_data = train_dev_data(dataset_for_train)
model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX01 = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)
print('num train batches:', len(model_tr_data))


# --------------------- Model ---------------------
class D_Former(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.prenet = nn.Linear(2, d_model)
        # causal convolution embedding (your module)
        self.input_embedding = causal_convolution_layer.context_embedding(500, d_model, 5)
        config = RoFormerConfig()
        self.roformerEnc = modeling_roformer.RoFormerEncoder(config)
        self.decoder = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, src: Tensor) -> Tensor:
        # src: [B, T, 2]
        src = self.prenet(src)   # [B, T, d_model]
        z_embedding = self.input_embedding(src)

        B = z_embedding.size(0)

        z_embeoutput = z_embedding.reshape(B, -1, 24)
        output = self.roformerEnc(z_embeoutput)

        output = output[0]
        output = output[:, -1, :]
        output = self.decoder(output)
        return output

# --------------------- Config & losses ---------------------
d_model = 12
nhead = 5
d_hid = 256
nlayers = 8
dropout = 0

model = D_Former(d_model, nhead, d_hid, nlayers, dropout).to(device)

os.makedirs('Durability_D_Former_ablation01', exist_ok=True)
config = {
    'n_epochs': 10,
    'early_stop': 100,
    'save_path': 'Durability_D_Former_ablation01/Durability_D_Former_01.pth'
}

criterion = nn.MSELoss()

class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        loss = torch.mean(torch.abs((targets - predictions) / (50)))
        return loss

criterion1 = MAPELoss()

# --------------------- Scheduler helper ---------------------
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

# --------------------- Training function (param-groups) ---------------------
def train(model_tr_data, dv_set, model, config, device):
    n_epochs = config['n_epochs']

    conv_lr = 3e-5
    conv_wd = 1e-5
    other_lr = 1e-5
    other_wd = 1e-3

    if hasattr(model, 'input_embedding'):
        conv_params = list(model.input_embedding.parameters())
        other_params = [p for n, p in model.named_parameters() if not n.startswith('input_embedding')]

        print('conv params count:', sum(p.numel() for p in conv_params))
        print('other params count:', sum(p.numel() for p in other_params))

        if sum(p.numel() for p in conv_params) == 0:
            print('Warning')
        optimizer = AdamW([
            {'params': conv_params,  'lr': conv_lr,  'weight_decay': conv_wd},
            {'params': other_params, 'lr': other_lr, 'weight_decay': other_wd},
        ])
    else:
        # fallback
        print('Warning: model has no attribute input_embedding')
        optimizer = AdamW(model.parameters(), lr=other_lr, weight_decay=other_wd)

    # scheduler
    warmup_steps = (len(dataset_for_train) // batch_size) * 2
    total_steps = (len(dataset_for_train) // batch_size) * config['n_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    min_mse = 2000
    loss_record = {'train': [], 'dev': []}
    lrs = []
    early_stop_cnt = 0
    epoch = 0
    num_batche_tr = len(model_tr_data)
    t1 = ttt.time()

    for i, g in enumerate(optimizer.param_groups):
        print(f'param group {i}: lr={g["lr"]}, weight_decay={g.get("weight_decay", None)}')

    while epoch < n_epochs:
        t0 = ttt.time()
        print('Epoch', epoch)
        allbat_mse_loss = 0.0
        model.train()

        for x, y, t, r, c in model_tr_data:
            x = x.float().to(device)
            y = y.float().to(device)

            pred = model(x)
            mse_loss = criterion(pred, y)
            allbat_mse_loss += mse_loss.detach().cpu().item()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()
            lrs.append(optimizer.param_groups[0]["lr"])
            scheduler.step()

        evepoch_mse_loss = allbat_mse_loss / num_batche_tr
        time_elapsed = ttt.time() - t0
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Epoch {} train avg mse loss: {:.6f}'.format(epoch, evepoch_mse_loss))
        loss_record['train'].append(evepoch_mse_loss)

        dev_mse = dev(dv_set, model, device)
        print('Epoch {} dev mse: {:.6f}'.format(epoch, dev_mse))
        if dev_mse < min_mse:
            min_mse = dev_mse
            print('Saving model (epoch = {}, loss = {:.6f})'.format((epoch + 1), (min_mse)))
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            print('Early stopping triggered.')
            break

    if len(lrs) > 0:
        plt.plot(lrs)
        plt.title('Learning rate schedule')
        plt.show()

    time_stopped = ttt.time() - t1
    print('Training completed in {:.0f}m {:.0f}s'.format(time_stopped // 60, time_stopped % 60))
    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record

# --------------------- dev / test ---------------------
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
    avg_loss = total_loss / num_batches
    return avg_loss

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

# # --------------------- Run training & test ---------------------
# model_loss, model_loss_record = train(model_tr_data, model_dev_data, model, config, device)
# total_params = sum(p.numel() for p in model.parameters())
# print('total params = ', total_params)
# print('best dev loss (min_mse):', model_loss)
# print('train loss record:', model_loss_record['train'])
# print('dev loss record:', model_loss_record['dev'])


def plot_learning_curve(loss_record, title=''):
    total_epochs = len(loss_record['train'])
    x_1 = range(total_epochs)
    if len(loss_record['dev']) > 0:
        step = max(1, len(loss_record['train']) // len(loss_record['dev']))
        x_2 = x_1[::step]
    else:
        x_2 = x_1
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], label='train')
    plt.plot(x_2, loss_record['dev'], label='dev')
    plt.ylim(0.0, max(max(loss_record['train']), max(loss_record['dev']) if loss_record['dev'] else 1.0) * 1.1)
    plt.xlabel('Training epochs')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

# plot_learning_curve(model_loss_record, title='D_Former_with_causal_conv')

# ------------------ Load best model & test -------------------
del model
model = D_Former(d_model, nhead, d_hid, nlayers, dropout).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)

preds01, targets01, test_loss01, test_mape01 = test(model_tt_dataX01, model, device)
print('test MSE:', test_loss01)
print('test MAPE:', test_mape01)

def save_pred(preds, targets, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'prediction', 'target'])
        for i, (p, t) in enumerate(zip(preds, targets)):
            writer.writerow([i, float(np.squeeze(p)), float(np.squeeze(t))])

save_pred(preds01, targets01, 'Durability_D_Former_ablation01_Preds_1.csv')
