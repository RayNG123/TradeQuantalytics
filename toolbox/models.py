import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.optimizers import Adam

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor


def Conv(X_train, y_train, X_test): 
    # Assuming X_train, y_train are your training data and labels
    n_timesteps, n_features = X_train.shape[1],1
  
    # Define the model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
  
    # Compile the model
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mse')
  
    # Fit the model
    history = model.fit(X_train, y_train, epochs=5, verbose=2)
    X_test_reshaped = X_test.reshape((1, len(X_test), 1))
    y_pred = model.predict(X_test_reshaped)
    test = np.concatenate((X_test, y_pred[0]))
    test = test.reshape(-1, 1)
    test = test.T
    test = scaler.inverse_transform(test)
    return test[0][-1]

def rf(X_train, y_train, X_test): 
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X_train.squeeze(2),y_train)
    y_pred = model.predict(X_test.reshape(1,-1))
    test = np.concatenate((X_test, y_pred))
    test = test.reshape(-1, 1)
    test = test.T
    test = scaler.inverse_transform(test)
    return test[0][-1]

def gradient_boosting(X_train, y_train, X_test): 
    model = GradientBoostingRegressor(n_estimators=50)
    model.fit(X_train.squeeze(2),y_train)
    y_pred = model.predict(X_test.reshape(1,-1))
    test = np.concatenate((X_test, y_pred))
    test = test.reshape(-1, 1)
    test = test.T
    test = scaler.inverse_transform(test)
    return test[0][-1]

def lstm(X_train, y_train, X_test):
    X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(X_train.shape[1], 1), activation='tanh',return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    #lstm.summary()
    lstm.fit(X_train, y_train, epochs=10, batch_size=1, verbose = 1)
    X_test_reshaped = X_test.reshape((1, len(X_test), 1))
    y_pred = lstm.predict(X_test_reshaped)
    print(y_pred)
    test = np.concatenate((X_test, y_pred[0]))
    test = test.reshape(-1, 1)
    test = test.T
    test = scaler.inverse_transform(test)
    return test[0][-1]
