
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler


def load_batch_data(batch_num):
    filename = f'combined_0{batch_num}_2000_.csv'
    df = pd.read_csv(filename, header=0)
    resistance_columns = range(0, 10)
    batch_data = df.iloc[:, resistance_columns].values

    return batch_data

data = []
labels = []
batch_indices = []

batch_nums = [1, 5, 8, 9]
# 循环处理每个批次数据
for batch_num in batch_nums:
    batch_data = load_batch_data(batch_num)
    data.append(batch_data)
    labels.extend([f'Batch {batch_num}'] * batch_data.shape[0])
    batch_indices.extend([batch_num] * batch_data.shape[0])


data_stacked = np.vstack(data)


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_stacked)


reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, n_components=2, random_state=42)
data_2d = reducer.fit_transform(data_scaled)


df_plot = pd.DataFrame({
    'Dim1': data_2d[:, 0],
    'Dim2': data_2d[:, 1],
    'Batch': labels,
    'Batch_Num': batch_indices
})


for batch_num in batch_nums:
    subset = df_plot[df_plot['Batch_Num'] == batch_num]
    filename = f'batch_ood{batch_num}_coordinates.csv'
    subset[['Dim1', 'Dim2']].to_csv(filename, index=False)
    print(f'batch {batch_num} coordinates save to {filename}')


plt.figure(figsize=(10, 8))
for batch in df_plot['Batch'].unique():
    subset = df_plot[df_plot['Batch'] == batch]
    plt.scatter(subset['Dim1'], subset['Dim2'], label=batch, alpha=0.7, s=50)
plt.legend()
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Sensor Resistance Data Dimensionality Reduction using UMAP')
plt.show()



