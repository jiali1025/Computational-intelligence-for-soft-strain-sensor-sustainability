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


# -------------------------------------Dataset -----------------------------------------
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


# train / dev
def train_dev_data(x_set):
    X = []
    deX = []
    for i in range(len(x_set)):
        if i % 10 != 0:
            X.append(x_set[i])
        else:
            deX.append(x_set[i])
    return X, deX


# ------------------------------------------- Lightweight D-Former --------------------------------------
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
        output = self.roformerEnc(z_embedding)
        output = output[0]                      # [B, L, 24]

        # 取最后时间步
        output = output[:, -1, :]               # [B, 24]

        # 解码
        output = self.decoder(output)           # [B, 1]
        return output


@torch.no_grad()
def save_predictions_to_csv(model, loader, device, filename='predictions.csv'):
    model.eval()
    preds = []
    tgts = []

    for batch in loader:
        x, y, *_ = batch
        x = x.float().to(device)
        y = y.float().to(device)

        yhat = model(x)

        preds.append(yhat.detach().cpu().numpy())
        tgts.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    tgts = np.concatenate(tgts, axis=0)


    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['True Values', 'Predictions'])
        for true, pred in zip(tgts, preds):
            writer.writerow([true, pred])

    print(f"Predictions and true values saved to {filename}")


def run_inference():
    seed = 2026
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32


    # dataset_for_train = StrainDataset("./Training_batchdiversity_1589.csv", "train", 500, transforms=None)
    dataset_for_test = StrainDataset("./Testing_batchdiversity_10.csv", "test", 500, transforms=None)

    # train_dataset, dev_dataset = train_dev_data(dataset_for_train)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    # dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset_for_test, batch_size=batch_size, shuffle=False, drop_last=True)

    # teacher_model = LightweightDFormer(d_model=12, nhead=12, d_hid=3072, nlayers=12)

    student_model = LightweightDFormer(d_model=12, nhead=2, d_hid=64, nlayers=4)
    student_model.to(device)

    try:
        student_model.load_state_dict(torch.load(f'BatchDiversity_best_student_after_pruning_final.pth'), strict=False)
        print("Student model loaded.")

    except Exception as e:
        print("Error loading student model:", e)

    save_predictions_to_csv(student_model, test_loader, device='cuda', filename='student_model_predictions.csv')


if __name__ == "__main__":
    run_inference()
