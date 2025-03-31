import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader
import shap
shap.initjs()
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class StrainDataset(Dataset):
    def __init__(self, path, mode='train', sequence_length=100, transforms=None):
        self.path = path
        self.mode = mode
        self.transforms = transforms
        self.seq_len = sequence_length

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
            time = cyc_time.astype(float)
            self.time = torch.tensor(time)
        print('Finished reading the {} set of Strain Dataset ({} samples found)'.format(mode, len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if index >= self.seq_len - 1:
            i_start = index - self.seq_len + 1
            x = self.X[i_start:(index + 1), :]
            time = self.time[i_start:(index + 1), :]
        else:
            padding = self.X[0].repeat(self.seq_len - index - 1, 1)
            x = self.X[0:(index + 1), :]
            x = torch.cat((padding, x), 0)
            padding_time = self.time[0].repeat(self.seq_len - index - 1, 1)
            time = self.time[0:(index + 1), :]
            time = torch.cat((padding_time, time), 0)
        return x, self.y[index], time

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
batch_size = 32
torch.manual_seed(99)
dataset_for_train = StrainDataset('./cycle_180_190_time.csv', 'train', 200, transforms=None)
dataset_for_test = StrainDataset('./cycle_190_195_test.csv', 'test', 200, transforms=None)
training_data, valid_data = train_dev_data(dataset_for_train)
model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
print(len(model_tr_data))
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX = DataLoader(dataset_for_test, batch_size=batch_size, shuffle=False, drop_last=True)

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.prenet = nn.Linear(1, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1))

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.prenet(src)
        output = self.transformer_encoder(src)
        output = output[:, -1, :]
        output = self.decoder(output)
        return output


device = torch.device('cpu')
d_model = 100
nhead = 10
d_hid = 1024
nlayers = 12
dropout = 0
model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)

config = {
    'n_epochs': 100,
    'early_stop': 100,
    'save_path': 'Transformer/model_transformer_01.pth'
}
criterion = nn.MSELoss()

import torch

del model
model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)
model = model.to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
model.eval()
background_01 = []
for X_train, Y_train, T_train in model_tr_data:
    background = X_train.to(device)
    background = background.detach().cpu().numpy()
    background = background.reshape(32, -1)
    background_01.append(background)
bakcgroundshape = background_01[2:3]
background_arr = np.array(bakcgroundshape)
background_arr_reshape = background_arr.reshape(-1, 200)
background_arr_reshape = background_arr_reshape[0:1, :]
pred_test_shape = []
for X_test, Y_test, T_test in model_tt_dataX:
    pred_test_02 = X_test.to(device)
    pred_test_02 = pred_test_02.detach().cpu().numpy()
    pred_test_02 = np.reshape(pred_test_02, (32, -1))

    pred_test_shape.append(pred_test_02)

pred_test_02 = pred_test_02.reshape(-1, 200)
n_samples = pred_test_02.shape[0]
num_random_samples = 10
random_indices = np.random.choice(n_samples, num_random_samples, replace=False)
random_samples = pred_test_02[random_indices, :]

def model_prediction(input_feature):

    input_feature = np.reshape(input_feature, (-1, 200, 1))
    feature = torch.tensor(input_feature)
    feature = feature.float().to(device)
    x = model(feature)
    data_frame = x.detach().numpy()

    return data_frame

explainer_01 = shap.KernelExplainer(model = model_prediction, data = background_arr_reshape, link ='identity')
shap_values_f = explainer_01.shap_values(X=random_samples)
shap_values_avg = np.mean(shap_values_f, axis=0)
print(shap_values_avg)
def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow('shap')
        for i, p in enumerate(preds):
            writer.writerow([i, p])
shap_val = shap_values_avg[0]
save_pred(shap_values_avg, 'Hysteresis_transformer_Shap_01.csv')
shap.summary_plot(shap_values_avg[0], random_samples, plot_type="bar")
print('cool')