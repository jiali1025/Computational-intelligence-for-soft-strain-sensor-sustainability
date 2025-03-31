import torch, gc
from torch import nn, Tensor
from torch.optim import AdamW
import os
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
class StrainDataset(Dataset):
    '''Dataset for loading and preprocessing the strain_sensor dataset'''
    def __init__(self, path, mode='train', sequence_length=500, transforms=None):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.seq_len = sequence_length
        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data)
            features = data[1:, 6].astype(float)  # Get resistance feature
            target = data[1:, 3].astype(float)    # Get strain
            self.y = torch.tensor(target).float()  # Convert target to tensor
            self.X = torch.tensor(features).float().reshape(-1, 1)  # Reshape input data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index].reshape(-1, 1)  # Return specific sample
        y = self.y[index].reshape(-1, 1)  # Return corresponding target value
        return x, y


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

class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 1
output_size = 1
batch_size = 64
criterion = nn.MSELoss()
# os.makedirs('DNNlinear/DNNlinear_01', exist_ok=True)
config = {
    'n_epochs': 400,
    'early_stop': 100,
    'save_path': 'DNNlinear_01/DNNlinear_01.pth'
}

dataset_for_train = StrainDataset('./linear_cw2_training.csv', 'train', 100, transforms=None)
training_data, valid_data = train_dev_data(dataset_for_train)
dataset_for_test = StrainDataset('./linear_cw2_testing.csv', 'train', 100, transforms=None)
model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX = DataLoader(dataset_for_test, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=False)

model = DNN(input_size = input_size, output_size= output_size)

def train(model_tr_data, dv_set, model, config, device):
    n_epochs = config['n_epochs']
    optimizer = AdamW(model.parameters(), lr=1e-5)
    min_mse = float('inf')
    loss_record = {'train': [], 'dev': []}
    early_stop_cnt = 0
    epoch = 0

    while epoch < n_epochs:
        print(f'Epoch {epoch + 1}/{n_epochs}')
        allbat_mse_loss = 0
        model.train()
        for x, y in model_tr_data:
            x = x.float().to(device)
            y = y.float().to(device)
            pred = model(x)
            mse_loss = criterion(pred, y)
            allbat_mse_loss += mse_loss.item()
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

        avg_loss = allbat_mse_loss / len(model_tr_data)
        print(f'Training Loss: {avg_loss:.4f}')
        loss_record['train'].append(avg_loss)
        # Validate
        dev_mse = dev(dv_set, model, device)
        loss_record['dev'].append(dev_mse)
        if dev_mse < min_mse:
            min_mse = dev_mse
            print(f'Saving model at epoch {epoch + 1} with loss {min_mse:.4f}')
            torch.save(model.state_dict(), config['save_path'])
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt > config['early_stop']:
            print('Early stopping...')
            break

        epoch += 1

    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()
    total_loss = 0
    num_batches = len(dv_set)

    with torch.no_grad():
        for x, y in dv_set:
            x = x.float().to(device)
            y = y.float().to(device)

            pred = model(x)
            mse_loss = criterion(pred, y)
            total_loss += mse_loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


# del model
# model = DNN(input_size = input_size, output_size= output_size)
# model = model.to(device)
# model_loss, model_loss_record = train(model_tr_data, model_dev_data, model, config, device)
# total_params = sum(p.numel() for p in model.parameters())
# print('total params = ', total_params)
# print(model_loss)
# print(model_loss_record['train'])
# print(model_loss_record['dev'])

# gc.collect()
# torch.cuda.empty_cache()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
del model
model = DNN(input_size = input_size, output_size= output_size)
model = model.to(device)
print(f'Using device: {device}')
ckpt = torch.load(config['save_path'], map_location='cpu')  # load your best model
model.load_state_dict(ckpt)


def test(tt_set, model, device):
    model.eval()
    preds = []
    mses = []
    test_loss = 0.0

    with torch.no_grad():
        for x, y in tt_set:
            x = x.float().to(device)
            y = y.float().to(device)

            pred = model(x)
            batch_loss = criterion(pred, y)
            preds.append(pred.detach().cpu())

            mse_values = (pred - y) ** 2
            mses.append(mse_values.detach().cpu())

            test_loss += batch_loss.item()

    avg_loss_test = test_loss / len(tt_set)
    preds = torch.cat(preds, dim=0).numpy()
    mses = torch.cat(mses, dim=0).numpy()
    print('Finished testing predictions!')
    return preds, mses, avg_loss_test

# Testing
preds01, mses01, test_loss01 = test(model_tt_dataX, model, device)  # predict strain cases with your model

print(f'Test loss: {test_loss01}')

def save_pred(preds, mses, file):
    '''Save predictions and MSE to specified file'''
    print(f'Saving results to {file}')
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'strain', 'mse'])
        for i in range(len(preds)):
            writer.writerow([i, preds[i], mses[i]])

save_pred(preds01, mses01, 'DNN_prediction_mse.csv')

