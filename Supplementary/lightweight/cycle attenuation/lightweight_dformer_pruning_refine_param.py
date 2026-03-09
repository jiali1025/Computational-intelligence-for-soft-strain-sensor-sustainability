import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import os, time, math, csv, random, json
from torch import Tensor
import causal_convolution_layer
from RoFormer import modeling_roformer, RoFormerConfig
import numpy as np


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
            all_time = data[1::3, 2:3]
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


class LightweightDFormer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.prenet = nn.Linear(2, d_model)
        self.input_embedding = causal_convolution_layer.context_embedding(500, d_model, 5)
        # config = RoFormerConfig()
        config = RoFormerConfig(
            hidden_size= 24,  # 24
            num_attention_heads = nhead,
            num_hidden_layers = nlayers,
            intermediate_size = d_hid,
        )
        self.roformerEnc = modeling_roformer.RoFormerEncoder(config)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, src: Tensor) -> Tensor:

        src = self.prenet(src)
        z_embedding = self.input_embedding(src)
        z_embedding = z_embedding.reshape(32, -1, 24)

        output = self.roformerEnc(z_embedding)
        output = output[0]
        output = output[:, -1, :]
        output = self.decoder(output)

        return output


def prune_model(model, pruning_amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Pruning {name}")
            # Prune the weights of the layer (removing the smallest weights)
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            prune.remove(module, 'weight')  # Optional: Remove the pruning mask after pruning
    return model

def fine_tune_student_model(student_model, train_loader, dev_loader, teacher_model, device, epochs=10, lr=1e-4):
    student_model.to(device)
    teacher_model.to(device)

    teacher_model.eval()

    optimizer = AdamW(student_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_dev_loss = float('inf')

    for epoch in range(epochs):
        student_model.train()
        train_loss = 0.0
        for x, y, _, _, _ in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            student_output = student_model(x)
            with torch.no_grad():
                teacher_output = teacher_model(x)

            distillation_loss = criterion(student_output, teacher_output)
            supervised_loss = criterion(student_output, y)

            loss = distillation_loss + supervised_loss
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        dev_loss = evaluate_loss(student_model, dev_loader, device)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader)}, Dev Loss: {dev_loss}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(student_model.state_dict(), 'best_student_model_after_pruning0.3.pth')

    print("Fine-tuning finished.")
    return student_model

def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y, _, _, _ in dataloader:
            x = x.float().to(device)
            y = y.float().to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def count_params(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def benchmark(model, device, example_batch, warmup=20, repeat=100, energy_meter=None):
    model.eval().to(device)
    x, *_ = example_batch
    x = x.float().to(device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for _ in range(warmup):
        _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def _run():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeat):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()
        return (t1 - t0) / repeat * 1000.0  # ms/iter

    if energy_meter is not None:
        lat_ms, energy_j = energy_meter.integrate(_run)
    else:
        lat_ms = _run()
        energy_j = None

    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0  # MiB
    else:
        peak_mem = None

    return lat_ms, peak_mem, energy_j


class MAPELoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred, tgt):
        return torch.mean(torch.abs((tgt - pred)/(50)))

@torch.no_grad()
def test_metrics(model, loader, device):
    model.eval()
    mse, mape = nn.MSELoss(), MAPELoss()
    n, mse_sum, mape_sum = 0, 0.0, 0.0
    preds, tgts = [], []
    for batch in loader:
        x, y, *_ = batch
        x = x.float().to(device); y = y.float().to(device)
        yhat = model(x)
        preds.append(yhat.detach().cpu()); tgts.append(y.detach().cpu())
        mse_sum += mse(yhat, y).item()
        mape_sum += mape(yhat, y).item()
        n += 1
    preds = torch.cat(preds, 0).numpy()
    tgts  = torch.cat(tgts,  0).numpy()
    return preds, tgts, mse_sum/max(n,1), mape_sum/max(n,1)

def run_training_with_pruning():
    seed = 2025
    batch_size = 32

    dataset_for_train = StrainDataset('./01_training_data_4000.csv', 'train', 500, transforms=None)
    dataset_for_test = StrainDataset('./01_testing_data_16000.csv', 'test', 500, transforms=None)

    train_dataset, dev_dataset = train_dev_data(dataset_for_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_for_test, batch_size=batch_size, shuffle=False, drop_last=True)

    teacher_model = LightweightDFormer(d_model=12, nhead=12, d_hid=3072, nlayers=12)
    student_model = LightweightDFormer(d_model=12, nhead=2, d_hid=64, nlayers=4)

    try:
        teacher_model.load_state_dict(torch.load(f'Durability_D_Former/Durability_D_Former_01.pth'), strict=False)
        print("Teacher model loaded.")
    except Exception as e:
        print("Error loading teacher model:", e)


    student_model = fine_tune_student_model(student_model, train_loader, dev_loader, teacher_model, device='cuda')

    student_model = prune_model(student_model, pruning_amount=0.3)

    student_model = fine_tune_student_model(student_model, train_loader, dev_loader, teacher_model, device='cuda')

    torch.save(student_model.state_dict(), 'best_student_model_final_pruning0.3.pth')
    print("Student model saved after pruning.")

    student_params = count_params(student_model)
    lat_ms, peak_mem, energy_j = benchmark(student_model, device='cuda', example_batch=next(iter(test_loader)))

    preds, tgts, test_mse, test_mape = test_metrics(student_model, test_loader, device='cuda')

    result = {
        "params": student_params,
        "latency_ms": lat_ms,
        "peak_mem_MiB": peak_mem,
        "test_mse": test_mse,
        "test_mape": test_mape,
        "energy_J_train": energy_j,
    }

    with open("training_results_after_pruning0.3.json", "w") as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    run_training_with_pruning()
