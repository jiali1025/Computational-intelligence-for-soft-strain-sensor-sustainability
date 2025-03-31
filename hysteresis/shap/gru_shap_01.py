import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import math
import os
import shap
shap.initjs()
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
            t = self.T[0:(index + 1), :]
            t = torch.cat((padding, t), 0)
        return x, self.y[index], t


class GruRNN(nn.Module):

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=2, batch_first=True):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 16)
        self.linear2 = nn.Linear(16, output_size)
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, _x):

        x, _ = self.gru(_x)
        x = self.linear1(x[:, -1, :])
        x = self.linear2(x)

        return x

model = GruRNN(1, 256)
loss_funtion = nn.MSELoss()
config = {
        'n_epochs': 120,
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
device = torch.device('cuda')

batch_size = 36
torch.manual_seed(99)

model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_data = DataLoader(dataset_tt, batch_size=batch_size, shuffle=False, drop_last=True)
print('done')
model = model.to(device)

del model
model = GruRNN(1, 256)
model = model.to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
model.eval()

background_01 = []
for X_train, Y_train, T_train in model_tr_data:
    background = X_train.to(device)
    background = background.detach().cpu().numpy()
    background = background.reshape(36, -1)
    background_01.append(background)
bakcgroundshape = background_01[2:3]
background_arr = np.array(bakcgroundshape)
background_arr_reshape = background_arr.reshape(-1, 35)
background_arr_reshape = background_arr_reshape[0:2, :]

pred_test_shape = []
for X_test, Y_test, T_test in model_tt_data:
    pred_test_02 = X_test.to(device)
    pred_test_02 = pred_test_02.detach().cpu().numpy()
    pred_test_02 = np.reshape(pred_test_02, (36, -1))
    # print(pred_test_02.shape)
    pred_test_shape.append(pred_test_02)
pred_test_02 = pred_test_02.reshape(-1, 35)
n_samples = pred_test_02.shape[0]
num_random_samples = 10
random_indices = np.random.choice(n_samples, num_random_samples, replace=False)
random_samples = pred_test_02[random_indices, :]

def model_prediction(input_feature):

    input_feature = np.reshape(input_feature, (-1, 35, 1))
    feature = torch.tensor(input_feature)
    feature = feature.float().to(device)
    x = model(feature)
    data_frame = x.detach().cpu().numpy()

    return data_frame

explainer_01 = shap.KernelExplainer(model = model_prediction, data = background_arr_reshape, link ='identity')
shap_values_f = explainer_01.shap_values(X=random_samples)
shap.summary_plot(shap_values_f[0], random_samples, plot_type="bar")
print('cool')