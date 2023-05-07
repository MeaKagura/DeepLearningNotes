import collections

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import numpy
import pandas as pd
from d2l import torch as d2l
import plotly.express as px

# X = torch.normal(0, 1, (16, 2))
# w1 = torch.normal(0, 1, (2, 1))
# w2 = w1.reshape(2)
# b = torch.ones(1)
#
# y_hat1 = torch.matmul(X, w1) + b
# y_hat2 = torch.matmul(X, w2) + b
# print(y_hat1)
#
# print(w1)
# print(w1.sum())
# float(w1.sum())

data_path = './data/california-house-prices/'
train_data = pd.read_csv(data_path + 'train.csv')
test_data = pd.read_csv(data_path + 'test.csv')

train_data=train_data.drop([3674,6055,32867,34876,43398,44091,44633])
data = pd.concat([train_data['Sold Price'], train_data['Listed Price']], axis=1)
fig = px.scatter(data, x='Listed Price', y='Sold Price')
fig.show()

# 处理数值特征
numeric_cols = train_data[1:].dtypes[train_data.dtypes != 'object'].index[1:]

# 过滤各个特征取值不正常的记录
for col in numeric_cols:
    train_data = train_data.drop(train_data[col].nlargest(int(0.01 * len(train_data))).index)
    train_data = train_data.drop(train_data[col].nsmallest(int(0.01 * len(train_data))).index)
# a = train_data['Lot'] < 10000
# train_data = train_data.drop(train_data.index[a])

# 添加新的数值特征
max_value = 10 ** 9
count = 0
zip_prices = collections.defaultdict(list)
for zip_value, sold_price in zip(train_data['Zip'], train_data['Sold Price']):
    zip_prices[zip_value].append(sold_price)
zip_price = collections.defaultdict(float)
for zip_value, price_list in zip_prices.items():
    zip_price[zip_value] = np.percentile(price_list, 50)
train_data['Near Price'] = pd.Series(zip_price[zip_value] for zip_value in train_data['Zip'])

numeric_cols = train_data[1:].dtypes[train_data.dtypes != 'object'].index[1:]

# 计算相关系数，选择高相关性的特征
cols_corr = train_data[numeric_cols].corr()
numeric_cols = []
target_cols = collections.deque(['Sold Price'])
while target_cols:
    target_col = target_cols.popleft()
    for col, corr in cols_corr[target_col].items():
        if abs(corr) > 0.1 and col not in numeric_cols:
            numeric_cols.append(col)
            target_cols.append(col)
numeric_cols.remove('Sold Price')
print(numeric_cols)

# 处理非数值特征
string_cols = ['Type', 'Heating', 'Cooling']
selected_cols = numeric_cols + string_cols
all_features = pd.concat((train_data[selected_cols], test_data[selected_cols]))

# 特征编码
# def label_encoder(data: pd.Series):
#     """
#     生成训练集中的某个特征的编码器（LabelEncoder对象）并存储到文件中
#     """
#     if data.name not in pre_model_files:
#         print(f"{data.name} not support.")
#         return
#     # LabelEncoder构造的编码器可以将离散型的数据转换成 0 到 n − 1 之间的数，n为离散取值数
#     le = LabelEncoder()
#     le.fit(data.astype(str).values)
#     # 将LabelEncoder构造的编码器le序列化后存储到文件中
#     with open(pre_model_files[data.name], "wb") as f:
#         pickle.dump(le, f)
#         print(f"dump {data.name} down.")
#     return le
all_features = pd.get_dummies(all_features, dummy_na=True, columns=string_cols)

# 对各个特征进行标准化、缺失填充
all_features[numeric_cols] = all_features[numeric_cols].apply(
    lambda x: ((x - x.mean()) / x.std())
)
all_features = all_features.fillna(0)

# 处理为迭代对象
batch_size = 32
all_features = torch.tensor(all_features.values, dtype=torch.float32)
all_labels = torch.tensor(train_data['Sold Price'].values.reshape((-1, 1)), dtype=torch.float32)
# train_num = int(train_data.shape[0] * 0.9)
# train_set = data.TensorDataset(all_features[:train_num], all_labels[:train_num])
# val_set = data.TensorDataset(all_features[train_num: len(train_data)], all_labels[train_num:])
test_set = data.TensorDataset(all_features[len(train_data):])
# train_iter = data.DataLoader(train_set, batch_size, shuffle=True)
# val_iter = data.DataLoader(val_set, batch_size, shuffle=False)
test_iter = data.DataLoader(test_set, batch_size, shuffle=False)
train_val_data = (all_features[:len(train_data)], all_labels)


def get_k_fold_data(data_set, k, i):
    assert k > 1
    num_samples = len(data_set[0])
    fold_size = num_samples // k
    features, labels = data_set
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, min((j + 1) * fold_size, num_samples - 1))
        X_part, y_part = features[idx], labels[idx]
        if j == i:
            X_val, y_val = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_val, y_val


k_fold = 5
train_features, train_labels, val_features, val_labels = \
    get_k_fold_data(train_val_data, k_fold, 4)
train_set = data.TensorDataset(train_features, train_labels)
val_set = data.TensorDataset(val_features, val_labels)
train_iter = data.DataLoader(train_set, batch_size, shuffle=True)
val_iter = data.DataLoader(val_set, batch_size, shuffle=False)


class Model(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        super(Model, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.linear3 = nn.Sequential(
            nn.Linear(64, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.linear4 = nn.Linear(16, out_features)

    def forward(self, X):
        X = self.linear1(X)
        X = self.linear2(X)
        X = self.linear3(X)
        y = self.linear4(X)
        return y


class log_rmse:
    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def __call__(self, y_hat, y):
        y_hat = torch.clamp(y_hat, 1, float('inf'))
        loss = torch.sqrt(self.loss_fn(torch.log(y_hat), torch.log(y)))
        return loss


def train(model, train_iter, val_iter, loss_fn, optimizer, num_epochs):
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, num_epochs],
                            legend=['train_loss', 'val_loss'])
    metrics = d2l.Accumulator(4)
    for epoch in range(num_epochs):
        metrics.reset()
        model.train()
        for X, y in train_iter:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            metrics.add(loss, 1)
        model.eval()
        for X, y in val_iter:
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            metrics.add(0, 0, loss, 1)
        animator.add(epoch + 1, (metrics[0] / metrics[1], metrics[2] / metrics[3]))
    print(f'log_rmse train loss: {metrics[0] / metrics[1]}')
    print(f'log_rmse val loss: {metrics[2] / metrics[3]}')


def predict(models, test_iter, test_data):
    for model in models:
        model.eval()
    predicts = []
    with torch.no_grad():
        for X, in test_iter:
            y_hat = np.zeros((X.shape[0], 1))
            for model in models:
                y_hat += (model(X).detach().numpy())
            predicts.append(y_hat.reshape(-1) / len(models))
    predicts = np.concatenate(predicts, axis=0)
    test_data['SalePrice'] = pd.Series(predicts)
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('./submission.csv', index=False)


model1 = Model(len(train_features[0]), 1)
model2 = Model(len(train_features[0]), 1)
predict([model1, model2], test_iter, test_data)
