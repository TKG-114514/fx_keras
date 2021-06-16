import pandas as pd
import seaborn as sns
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
sns.set()

'''
1. データの準備
'''
np.random.seed(123)
tf.random.set_seed(123)

data_type = {'time': str, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
name_list = ['time', 'open', 'high', 'low', 'close', 'volume']
file_name = 'USDJPY_30 Mins_Bid_2021.02.01_2021.04.30.csv'
df = pd.read_csv(filepath_or_buffer=file_name, dtype=data_type, header=0, names=name_list, parse_dates=['time'])
train, test = train_test_split(df, test_size=0.2, shuffle=False)
del train['time']
del test['time']

window_len = 5

train_lstm_in = []
for i in range(len(train) - window_len):
    temp = train[i:(i + window_len)].copy()
    for col in train:
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    train_lstm_in.append(temp)
lstm_train_out = (train['close'][window_len:].values / train['close'][:-window_len].values) - 1

test_lstm_in = []
for i in range(len(test) - window_len):
    temp = test[i:(i + window_len)].copy()
    for col in test:
        temp.loc[:, col] = temp[col] / temp[col].iloc[0] - 1
    test_lstm_in.append(temp)
lstm_test_out = (test['close'][window_len:].values / test['close'][:-window_len].values) - 1

train_lstm_in = [np.array(train_lstm_input) for train_lstm_input in train_lstm_in]
train_lstm_in = np.array(train_lstm_in)

test_lstm_in = [np.array(test_lstm_input) for test_lstm_input in test_lstm_in]
test_lstm_in = np.array(test_lstm_in)

print(train_lstm_in.shape)
print(test_lstm_in.shape)

'''
2. モデルの構築
'''
model = Sequential()
model.add(LSTM(50, input_shape=(train_lstm_in.shape[1], train_lstm_in.shape[2]), activation='tanh', recurrent_activation='sigmoid',
               kernel_initializer='glorot_normal', recurrent_initializer='orthogonal'))
model.add(Dense(1, activation='linear'))

'''
3. モデルの学習
'''
optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
model.compile(optimizer=optimizer, loss='mean_squared_error')
es = EarlyStopping(monitor='loss', patience=10, verbose=1)
yen_history = model.fit(train_lstm_in, lstm_train_out, epochs=500, batch_size=100, verbose=2, shuffle=True, callbacks=[es])

'''
4. モデルの評価
'''
fig1, ax1 = plt.subplots(1, 1)
ax1.plot(yen_history.epoch, yen_history.history['loss'])
ax1.set_title('TrainingError')
if model.loss == 'mean_squared_error':
    ax1.set_ylabel('mean_squared_error', fontsize=12)
    fig1.subplots_adjust(left=0.2)
    plt.figure(figsize=(30, 15))
else:
    ax1.set_ylabel('Model Loss', fontsize=12)
ax1.set_xlabel('# Epochs', fontsize=12)
fig1.subplots_adjust(left=0.2)
plt.figure(figsize=(20, 8))

fig2, ax2 = plt.subplots(1,1)
ax2.plot(df[df['time']<df['time'][train.shape[0]]]['time'][window_len:],
         train['close'][window_len:], label='Actual', color='blue')
ax2.plot(df[df['time']<df['time'][train.shape[0]]]['time'][window_len:],
         ((np.transpose(model.predict(train_lstm_in))+1) * train['close'].values[:-window_len])[0],
         label='Predicted', color='red')
plt.xticks(rotation=45)
fig2.subplots_adjust(bottom=0.2)
plt.figure(figsize=(20,15))

fig3, ax3 = plt.subplots(1,1)
ax3.plot(df[df['time']>=df['time'][train.shape[0]]]['time'][window_len:],
         test['close'][window_len:], label='Actual', color='blue')
ax3.plot(df[df['time']>=df['time'][train.shape[0]]]['time'][window_len:],
         ((np.transpose(model.predict(test_lstm_in))+1) * test['close'].values[:-window_len])[0],
         label='Predicted', color='red')
plt.xticks(rotation=45)
fig3.subplots_adjust(bottom=0.2)
plt.figure(figsize=(20,15))
ax3.grid(True)

"""
fig1.savefig("img1.png")
fig2.savefig("img2.png")
fig3.savefig("img3.png")
"""
plt.show()