from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, Tensor
import numpy as np
import csv
torch.cuda.empty_cache()
import shap
shap.initjs()


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
            all_time = data[1::3, 1:2]
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
    return 'cuda'if torch.cuda.is_available() else 'cpu'
device = get_device()

class GruRNN(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1, batch_first=True):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 8)
        self.linear2 = nn.Linear(8, output_size)
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, _x):

        x, _ = self.gru(_x)
        x = self.linear1(x[:, -1, :])
        x = self.linear2(x)

        return x

model = GruRNN(2, 64)
loss_funtion = nn.MSELoss()
model = model.to(device)
config = {
        'n_epochs': 100,
        'batch_size': 64,
        'optimizer': 'SGD',
        'optim_hparas': {
        'lr':  0.000001,
        },
        'early_stop': 100,
        'save_path': 'Durability_GRU_01/Durability_GRU_01.pth'
}
dataset = StrainDataset('./01_training_data_4000.csv', 'train', 30, transforms=None)
dataset_for_test_01 = StrainDataset('./01_testing_data_16000.csv', 'test', 30, transforms=None)

print(len(dataset))

# splitting training data into train & dev sets
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

batch_size = 64
torch.manual_seed(99)

model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_dataX01 = DataLoader(dataset_for_test_01, batch_size=batch_size, shuffle=False, drop_last=True)

print('done')
del model

model = GruRNN(2, 64)
model = model.to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')
model.load_state_dict(ckpt)
model.eval()
background_01 = []
for X_train, Y_train, T_train, R_train, C_train in model_tr_data:
    background = X_train.to(device)
    background = background.detach().cpu().numpy()
    background = background.reshape(64, -1)
    background_01.append(background)
bakcgroundshape = background_01[2:3]
background_arr = np.array(bakcgroundshape)
background_arr_reshape = background_arr.reshape(-1, 60)
background_arr_reshape = background_arr_reshape[0:2, :]
pred_test_shape = []
for X_test, Y_test, T_test, R_test, C_test in model_tt_dataX01:
    pred_test_02 = X_test.to(device)
    pred_test_02 = pred_test_02.detach().cpu().numpy()
    pred_test_02 = np.reshape(pred_test_02, (64, -1))
    pred_test_shape.append(pred_test_02)
pred_test_02 = pred_test_02.reshape(-1, 30)
n_samples = pred_test_02.shape[0]
num_random_samples = 10
random_indices = np.random.choice(n_samples, num_random_samples, replace=False)
random_samples = pred_test_02[random_indices, :]

def model_prediction(input_feature):
    input_feature = np.reshape(input_feature, (-1, 30, 2))
    feature = torch.tensor(input_feature)
    feature = feature.float().to(device)
    with torch.no_grad():
        x = model(feature)
    data_frame = x.detach().cpu().numpy()
    return data_frame

explainer_01 = shap.KernelExplainer(model = model_prediction, data = background_arr_reshape, link ='identity')
torch.cuda.empty_cache()
shap_values_f = explainer_01.shap_values(X=random_samples)
print(shap_values_f)
def save_pred(preds, file):
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow('shap')
        for i, p in enumerate(preds):
            writer.writerow([i, p])
shap_val = shap_values_f[0]
save_pred(shap_val, 'Durability_GRU_Shap_01.csv')
# shap.force_plot(explainer_01.expected_value[0], shap_values_f[0], pred_test_02_shape, matplotlib=True)
shap.summary_plot(shap_values_f[0], random_samples, plot_type="bar")

print('cool')
