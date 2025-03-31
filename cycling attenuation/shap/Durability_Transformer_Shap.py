from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, Tensor
import numpy as np
import csv
import math
torch.cuda.empty_cache()
import shap
shap.initjs()
from torch.nn import TransformerEncoder, TransformerEncoderLayer

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

def get_device():
    '''get device(if GPU is available, use GPU'''
    return 'cuda'if torch.cuda.is_available() else 'cpu'
# device = get_device()

device = torch.device('cpu')
class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.prenet = nn.Linear(2, d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1))

    def forward(self, src: Tensor) -> Tensor:

        src = self.prenet(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
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
nhead = 4
d_hid = 256
nlayers = 8
dropout = 0
model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)
loss_funtion = nn.MSELoss()
config = {
    'n_epochs': 100,
    'early_stop': 100,
    'save_path': 'Durability_Transformer_01/Durability_Transformer_01.pth'
}

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
dataset_for_train = StrainDataset('./01_training_data_4000.csv', 'train', 500, transforms=None)
dataset_for_test_01 = StrainDataset('./01_testing_data_16000.csv', 'test', 500, transforms=None)
training_data, valid_data = train_dev_data(dataset_for_train)
model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
print(len(model_tr_data))
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX01 = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)

print('done')
del model

model = TransformerModel(d_model, nhead, d_hid, nlayers, dropout).to(device)
model = model.to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
model.eval()
background_01 = []
for X_train, Y_train, T_train, R_train, C_train in model_tr_data:
    background = X_train.to(device)
    background = background.detach().cpu().numpy()
    background = background.reshape(32, -1)
    background_01.append(background)
bakcgroundshape = background_01[1:2]
background_arr = np.array(bakcgroundshape)
background_arr_reshape = background_arr.reshape(-1, 1000)
background_arr_reshape = background_arr_reshape[1:2, :]
pred_test_shape = []
with torch.no_grad():
    for X_test, Y_test, T_test, R_test, C_test in model_tt_dataX01:
        pred_test_02 = X_test.to(device)
        pred_test_02 = pred_test_02.detach().cpu().numpy()
        pred_test_02 = np.reshape(pred_test_02, (32, -1))
        pred_test_shape.append(pred_test_02)

pred_test_02 = pred_test_02.reshape(-1, 30)
n_samples = pred_test_02.shape[0]
num_random_samples = 10
random_indices = np.random.choice(n_samples, num_random_samples, replace=False)
random_samples = pred_test_02[random_indices, :]

def model_prediction(input_feature):

    input_feature = np.reshape(input_feature, (-1, 500, 2))
    feature = torch.tensor(input_feature)
    feature = feature.float().to(device)
    torch.cuda.empty_cache()
    with torch.no_grad():
        x = model(feature)
    data_frame = x.detach().cpu().numpy()

    return data_frame

explainer_01 = shap.KernelExplainer(model = model_prediction, data = background_arr_reshape, link ='identity')
torch.cuda.empty_cache()
shap_values_f = explainer_01.shap_values(X=random_samples, nsamples=1000)

def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['index', 'shap_value'])
        for i, p in enumerate(preds):
            if isinstance(p, (list, np.ndarray)):
                p = np.array(p).flatten()
                for val in p:
                    writer.writerow([i, val])
            else:
                writer.writerow([i, p])

shap_val = shap_values_f[0]
save_pred(shap_val, 'Durability_Transformer_Shap_01.csv')

print(shap_values_f[0].shape)
# shap.force_plot(explainer_01.expected_value[0], shap_values_f[0], pred_test_02_shape, matplotlib=True)
shap.summary_plot(shap_values_f[0], random_samples, plot_type="bar")

print('cool')