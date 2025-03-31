import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
import shap
shap.initjs()
torch.cuda.empty_cache()

class StrainDataset(Dataset):

    def __init__(self, path, mode='train', sequence_length=12, transforms=None):
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

            padding_time = self.T[0].repeat(self.seq_len - index - 1, 1)
            t = self.T[0:(index + 1), :]
            t = torch.cat((padding, t), 0)
        return x, self.y[index], t


device = torch.device('cpu')

class CNNnetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Conv1d(1, 256, kernel_size=20)
        self.relu = nn.ReLU(inplace=True)
        self.pool1d = nn.MaxPool1d(kernel_size=30 - 20 + 1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.pool1d(x)
        fc_inp = x.view(-1, x.size(1))
        x = self.fc1(fc_inp)
        x = self.relu(x)
        x = self.fc2(x)

        return x

model = CNNnetwork()
loss_funtion = nn.MSELoss()
config = {
    'n_epochs': 120,
    'batch_size': 36,
    'optimizer': 'Adam',
    'optim_hparas': {
        'lr': 0.0001,
    },
    'early_stop': 100,
    'save_path': '1DCNN/model_1d_CNN_01.pth'
}
dataset = StrainDataset('./cycle_180_190_time.csv', 'train', 30, transforms=None)
dataset_tt = StrainDataset('./cycle_190_195_test.csv', 'test', 30, transforms=None)

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

training_data, valid_data = train_dev_data(dataset)
batch_size = 36
torch.manual_seed(99)
model_tr_data = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_dev_data = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)
model_tt_data = DataLoader(dataset_tt, batch_size=batch_size, shuffle=False, drop_last=True)

del model
model = CNNnetwork()
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
background_arr_reshape = background_arr.reshape(-1, 30)
background_arr_reshape = background_arr_reshape[0:2, :]
pred_test_shape = []
for X_test, Y_test, T_test in model_tt_data:
    pred_test_02 = X_test.to(device)
    pred_test_02 = pred_test_02.detach().cpu().numpy()
    pred_test_02 = np.reshape(pred_test_02, (36, -1))
    pred_test_shape.append(pred_test_02)

pred_test_02 = pred_test_02.reshape(-1, 30)
n_samples = pred_test_02.shape[0]
num_random_samples = 10
random_indices = np.random.choice(n_samples, num_random_samples, replace=False)

random_samples = pred_test_02[random_indices, :]

def model_prediction(input_feature):

    input_feature = np.reshape(input_feature, (-1, 30, 1))
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
    '''Save predictions to specified file'''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow('shap')
        for i, p in enumerate(preds):
            writer.writerow([i, p])
shap_val = shap_values_avg[0]
save_pred(shap_values_avg, 'Hysteresis_1dcnn_Shap_01.csv')
shap.summary_plot(shap_values_f[0],random_samples, plot_type="bar")
