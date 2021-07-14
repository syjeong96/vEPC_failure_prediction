import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


dataset_file = pd.read_csv('dataset/data.txt', sep='\t', header=None)
# 0: stress_period, 2: cpu_usage (%), 3: mem_usage (%), 4: swap_space_usage (%), 5: #indications (T3413 expiration), 6: failure_label ("broken pipe")
dataset_features = dataset_file.iloc[:, [0, 2, 3, 4, 5]].values
dataset_labels = dataset_file.iloc[:, [6]].values

training_X, testing_X, training_y, testing_y = train_test_split(dataset_features, dataset_labels, test_size=0.2, shuffle=False, stratify=None)
# training_X, testing_X, training_y, testing_y = train_test_split(dataset_features, dataset_labels, test_size=0.2, shuffle=True, stratify=dataset_labels)
print(training_X.shape)
print(training_y.shape)
print(testing_X.shape)
print(testing_y.shape)

sc = MinMaxScaler(feature_range=(0, 1))
training_X_scaled = sc.fit_transform(training_X)
# print(training_X_scaled)

training_features = []
training_labels = []
win_size = 60      # time steps. 60 * 5sec = 300sec
for i in range(win_size, len(training_X)):
    # 0: stress-period, 1: cpu, 2: mem, 3: swap, 4: #indications
    training_features.append(training_X_scaled[i-win_size:i, 0:5])       # features of [T-60, ..., T-1]
    # 0: failure_label
    training_labels.append(training_y[i, 0])                      # label at time T
training_features, training_labels = np.array(training_features), np.array(training_labels)
print(training_features.shape)
print(training_labels.shape)

# LSTM build
regressor = Sequential()

# input_shape = (time_steps, # of indicators/features)
num_units=32
regressor.add(LSTM(units=num_units, return_sequences=True, input_shape=[training_features.shape[1], training_features.shape[2]]))
regressor.add(Dropout(0.2))
# regressor.add(LSTM(units=num_units, return_sequences=True))
# regressor.add(Dropout(0.2))
# regressor.add(LSTM(units=num_units, return_sequences=True))
# regressor.add(Dropout(0.2))
regressor.add(LSTM(units=num_units))
regressor.add(Dropout(0.2))

regressor.add(Dense(units=1))

regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# # https://stackoverflow.com/questions/51047676/how-to-get-accuracy-of-model-using-keras
# # model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001,
# #                                        beta_1=0.9,
# #                                        beta_2=0.999,
# #                                        epsilon=1e-07,
# #                                        amsgrad=False,
# #                                        name='Adam'
# #                                        ),
# #           loss='sparse_categorical_crossentropy',
# #           metrics=['accuracy']

history = regressor.fit(training_features, training_labels, epochs=10, batch_size=32)
print("Model accuracy: " + str(history.history['accuracy']))

# Testing phase
# real_values = testing_y
# print(real_values)

testing_X_scaled = sc.fit_transform(testing_X)
# print(testing_X_scaled)

testing_features = []
for i in range(win_size, len(testing_X)):
    # 0: stress-period, 1: cpu, 2: mem, 3: swap, 4: #indications
    testing_features.append(testing_X_scaled[i-win_size:i, 0:5])
testing_features = np.array(testing_features)
# print(features_test.shape)

# features_test = np.reshape(features_test, (features_test.shape[0], features_test.shape[1], 1))
# # print(X_test.shape)

predictions = regressor.predict(testing_features)
# predictions = sc.inverse_transform(predictions)

# Visualize
plt.plot(testing_y, color='black', label='ground-truth ')
plt.plot(predictions, color='red', label='prediction ')
# plt.title('Failure Prediction')
plt.xlabel('Time unit')
plt.xlim([0, 1200])
plt.ylabel('Failure score')
plt.ylim([0, 1.2])
plt.tick_params(axis='x', direction='in')
plt.tick_params(axis='y', direction='in')
plt.legend(loc='lower right')
plt.show()

f = open("predictions.txt", 'w')
for i in range(predictions.shape[0]):
    f.write(np.array_str(predictions[i]).strip("[]") + "\n")
f.close()
