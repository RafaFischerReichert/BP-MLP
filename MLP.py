import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('processed_prices.csv')

X = data.drop('sellingprice', axis=1)
y = data['sellingprice']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

neuron_number = 256
epoch_number = 100
batch_size = 32

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# PARA DEBUG
# print("X_train shape:", X_train.shape)
# print("X_test shape:", X_test.shape)
# print("y_train shape:", y_train.shape)
# print("y_test shape:", y_test.shape)

model = Sequential()
model.add(Dense(neuron_number, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1)) 

model.compile(optimizer='adam', loss='mean_absolute_error')

history = model.fit(X_train, y_train, epochs=epoch_number, batch_size=batch_size, validation_data=(X_test, y_test))
