import csv
import os
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
import json

class StrainDataset(Dataset):
    def __init__(self, path, mode='train', sequence_length=500, transforms=None):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.seq_len = sequence_length
        # read data into numpy arrays
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

# splitting training data into train & val sets
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
dataset_for_train = StrainDataset('./Training_batchdiversity_1589.csv', 'train', 500, transforms=None)
dataset_for_test_01 = StrainDataset('./Testing_batchdiversity_10.csv', 'test', 500, transforms=None)
training_data, valid_data = train_dev_data(dataset_for_train)
model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
print(len(model_tr_data))
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX01 = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)

class D_Former(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.prenet = nn.Linear(3, d_model)
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_model = 12
nhead = 5
d_hid = 256
nlayers = 8
dropout = 0
model = D_Former(d_model, nhead, d_hid, nlayers, dropout).to(device)

# os.makedirs('BatchDiversity_D_Former_01_1589Preds3', exist_ok=True)
config = {
    'n_epochs': 100,
    'early_stop': 100,
    'save_path': 'BatchDiversity_D_Former_01_1589Preds3/BatchDiversity_D_Former_01_1589Preds3.pth'
}
criterion = nn.MSELoss()
class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        loss = torch.mean(torch.abs((targets - predictions) / (targets + self.epsilon)))
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
    optimizer = AdamW(model.parameters(), lr=1e-5)
    warmup_steps = (len(dataset_for_train)//batch_size)*2
    total_steps = (len(dataset_for_train)//batch_size)*config['n_epochs']
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

# model_loss, model_loss_record = train(model_tr_data, model_dev_data, model, config, device)
# total_params = sum(p.numel() for p in model.parameters())
# print('total params = ', total_params)
# print(model_loss)
# print(model_loss_record['train'])
# print(model_loss_record['dev'])

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

def count_params(model: nn.Module):

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def model_param_bytes(model: nn.Module) -> int:

    total = 0
    for p in list(model.parameters()) + list(model.buffers()):
        total += p.numel() * p.element_size()
    return total

def byte_str(n_bytes: int) -> str:

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n_bytes)
    for u in units:
        if x < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} EB"

@torch.no_grad()
def benchmark_on_testset(tt_loader,
                         model,
                         device,
                         test_mse,
                         test_mape,
                         save_path="BatchDiversity_D_Former_benchmark.json"):

    if not isinstance(device, torch.device):
        device = torch.device(device)

    model.eval().to(device)

    n_batches = 0
    n_samples = 0

    total_params, trainable_params = count_params(model)
    param_mem_bytes = model_param_bytes(model)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    t_start = ttt.time()
    for x, y, t, r, c in tt_loader:
        n_batches += 1
        n_samples += x.size(0)
        x = x.float().to(device)
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_end = ttt.time()

    elapsed = t_end - t_start
    avg_batch_ms = elapsed / max(1, n_batches) * 1000.0
    avg_sample_ms = elapsed / max(1, n_samples) * 1000.0

    if device.type == "cuda":
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    else:
        peak_mem_bytes = 0

    result = {
        "model_name": "D_Former(d_model=12, nhead=5, d_hid=256, nlayers=8)_BatchDiversity",
        "device": str(device),
        "batch_size": tt_loader.batch_size,
        "num_batches": n_batches,
        "num_samples": n_samples,
        "test_mse": float(test_mse),
        "test_mape": float(test_mape),
        "params_total": int(total_params),
        "params_trainable": int(trainable_params),
        "param_mem_bytes": int(param_mem_bytes),
        "param_mem_human": byte_str(param_mem_bytes),
        "peak_mem_bytes": int(peak_mem_bytes),
        "peak_mem_human": byte_str(peak_mem_bytes),
        "latency_total_seconds": float(elapsed),
        "latency_avg_batch_ms": float(avg_batch_ms),
        "latency_avg_sample_ms": float(avg_sample_ms),
    }

    print("\n===== BatchDiversity D_Former | Test-set Inference / Params / Peak Memory / Latency =====")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Benchmark results saved to:  {save_path}")
    return result

del model
model = D_Former(d_model, nhead, d_hid, nlayers, dropout).to(device)
model = model.to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)

preds01, targets01, test_loss01, test_mape01 = test(model_tt_dataX01, model, device)

print(preds01)
print(test_loss01)
print(test_mape01)

def save_pred(preds, targets, file):

    print('Saving results to {}'.format(file))
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'prediction', 'target'])
        for i, (p, t) in enumerate(zip(preds, targets)):
            writer.writerow([i, p, t])

save_pred(preds01, targets01, 'BatchDiversity_D_Former_01_1589Preds10.csv')
print('ALL Done!')

benchmark_on_testset(
    tt_loader=model_tt_dataX01,
    model=model,
    device=device,
    test_mse=test_loss01,
    test_mape=test_mape01,
    save_path="BatchDiversity_D_Former_models/BatchDiversity_D_Former_01_1589_benchmark.json"
)
