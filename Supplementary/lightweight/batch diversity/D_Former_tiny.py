import os, time, math, csv, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

import causal_convolution_layer
from RoFormer import modeling_roformer, RoFormerConfig


# -------------------------------- Dataset -----------------------------------------
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
            time_arr = all_time.astype(float)
            self.time = torch.tensor(time_arr)
            resis_f = resistance_f.astype(float)
            self.resis_f = torch.tensor(resis_f)
            cyc_n = cycle_num.astype(float)
            self.cyc_n = torch.tensor(cyc_n)

            self.X[:, :] = (
                self.X[:, :] - self.X[:, :].mean(dim=0, keepdim=True)
            ) / self.X[:, :].std(dim=0, keepdim=True)

        print(f"Finished reading the {mode} set of Strain Dataset ({len(self.X)} samples found)")

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
            X.append(x_set[i])
        else:
            deX.append(x_set[i])
    return X, deX


# ----------------------------------lightweight D-Former -------------------------------------
class LightweightDFormer(nn.Module):
    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.0):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.prenet = nn.Linear(3, d_model)
        self.input_embedding = causal_convolution_layer.context_embedding(500, d_model, 5)

        config = RoFormerConfig(
            hidden_size=24,
            num_attention_heads=nhead,
            num_hidden_layers=nlayers,
            intermediate_size=d_hid,
        )
        self.roformerEnc = modeling_roformer.RoFormerEncoder(config)

        # Decoder: 24 -> 128 -> 1
        self.decoder = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, src: Tensor) -> Tensor:
        # src: [B, T, 3]
        B = src.size(0)
        src = self.prenet(src)                 # [B, T, d_model]
        z_embedding = self.input_embedding(src)
        z_embedding = z_embedding.reshape(B, -1, 24)
        # RoFormer
        output = self.roformerEnc(z_embedding)
        output = output[0]                      # [B, L, 24]
        output = output[:, -1, :]               # [B, 24]
        output = self.decoder(output)           # [B, 1]
        return output


# ------------------------------- Pruning and distillation --------------------------------------
def prune_model(model, pruning_amount=0.5):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"Pruning {name}")
            prune.l1_unstructured(module, name="weight", amount=pruning_amount)
            prune.remove(module, "weight")
    return model


def fine_tune_student_model(student_model, train_loader, dev_loader, teacher_model, device, epochs=10, lr=1e-4):
    student_model.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    optimizer = AdamW(student_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_dev_loss = float("inf")

    for epoch in range(epochs):
        student_model.train()
        train_loss = 0.0

        for x, y, *_ in train_loader:
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()

            # student prediction
            student_output = student_model(x)

            # teacher prediction
            with torch.no_grad():
                teacher_output = teacher_model(x)

            # distillation loss + supervised loss
            distillation_loss = criterion(student_output, teacher_output)
            supervised_loss = criterion(student_output, y)
            loss = distillation_loss + supervised_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        dev_loss = evaluate_loss(student_model, dev_loader, device)
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss / max(1, len(train_loader)):.6f}, "
              f"Dev Loss: {dev_loss:.6f}")

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(student_model.state_dict(), "BatchDiversity_best_student_after_pruning.pth")
            print(f"  ↳ New best student model saved. Dev Loss = {best_dev_loss:.6f}")

    print("Fine-tuning finished.")
    return student_model


def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for x, y, *_ in dataloader:
            x = x.float().to(device)
            y = y.float().to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

def count_params(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def benchmark(model, device, example_batch, warmup=20, repeat=100, energy_meter=None):
    if not isinstance(device, torch.device):
        device = torch.device(device)

    model.eval().to(device)
    x, *_ = example_batch
    x = x.float().to(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(warmup):
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def _run():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.time()
        for _ in range(repeat):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.time()
        return (t1 - t0) / repeat * 1000.0  # ms

    if energy_meter is not None:
        lat_ms, energy_j = energy_meter.integrate(_run)
    else:
        lat_ms = _run()
        energy_j = None

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024.0 / 1024.0  # MiB
    else:
        peak_mem = None

    return lat_ms, peak_mem, energy_j


class MAPELoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, tgt):
        return torch.mean(torch.abs((tgt - pred) / (50)))


@torch.no_grad()
def test_metrics(model, loader, device):
    if not isinstance(device, torch.device):
        device = torch.device(device)

    model.eval().to(device)
    mse, mape = nn.MSELoss(), MAPELoss()
    n, mse_sum, mape_sum = 0, 0.0, 0.0
    preds, tgts = [], []

    for batch in loader:
        x, y, *_ = batch
        x = x.float().to(device)
        y = y.float().to(device)
        yhat = model(x)
        preds.append(yhat.detach().cpu())
        tgts.append(y.detach().cpu())
        mse_sum += mse(yhat, y).item()
        mape_sum += mape(yhat, y).item()
        n += 1

    preds = torch.cat(preds, 0).numpy()
    tgts = torch.cat(tgts, 0).numpy()
    return preds, tgts, mse_sum / max(1, n), mape_sum / max(1, n)

def run_training_with_pruning():
    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    dataset_for_train = StrainDataset("./Training_batchdiversity_1589.csv", "train", 500, transforms=None)
    dataset_for_test = StrainDataset("./Testing_batchdiversity_10.csv", "test", 500, transforms=None)

    train_dataset, dev_dataset = train_dev_data(dataset_for_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_for_test, batch_size=batch_size, shuffle=False, drop_last=True)

    teacher_model = LightweightDFormer(d_model=12, nhead=12, d_hid=3072, nlayers=12)
    student_model = LightweightDFormer(d_model=12, nhead=2, d_hid=64, nlayers=4)

    # try:
    #     teacher_ckpt_path = "BatchDiversity_D_Former_models/BatchDiversity_D_Former_01_1589Preds10.pth"
    #     teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location="cpu"), strict=False)
    #     print(f"Teacher model loaded from {teacher_ckpt_path}.")
    # except Exception as e:
    #     print("Warning: could not load teacher model checkpoint, using randomly initialized teacher.")
    #     print("  Error:", e)

    # student_model = fine_tune_student_model(
    #     student_model, train_loader, dev_loader, teacher_model,
    #     device=device, epochs=10, lr=1e-4
    # )

    # student_model = prune_model(student_model, pruning_amount=0.3)

    # student_model = fine_tune_student_model(
    #     student_model, train_loader, dev_loader, teacher_model,
    #     device=device, epochs=5, lr=5e-5
    # )

    # final_student_ckpt = "BatchDiversity_best_student_after_pruning_final.pth"
    # torch.save(student_model.state_dict(), final_student_ckpt)
    # print(f"Student model saved after pruning and fine-tuning: {final_student_ckpt}")
    
    best_state = torch.load("BatchDiversity_best_student_after_pruning_final.pth")
    student_model.load_state_dict(best_state)

    student_params = count_params(student_model)
    lat_ms, peak_mem, energy_j = benchmark(student_model, device=device, example_batch=next(iter(test_loader)))
    preds, tgts, test_mse, test_mape = test_metrics(student_model, test_loader, device=device)

    result = {
        "params": int(student_params),
        "latency_ms": float(lat_ms),
        "peak_mem_MiB": float(peak_mem) if peak_mem is not None else None,
        "test_mse": float(test_mse),
        "test_mape": float(test_mape),
        "energy_J_train": float(energy_j) if energy_j is not None else None,
    }

    os.makedirs("BatchDiversity_Lightweight_results", exist_ok=True)
    out_json = os.path.join(
        "BatchDiversity_Lightweight_results",
        "BatchDiversity_student_after_pruning.json"
    )
    with open(out_json, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Lightweight student metrics saved to: {out_json}")


if __name__ == "__main__":
    run_training_with_pruning()
