from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.constraints import nonneg
from matplotlib import pyplot
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json
from os.path import isfile
import numpy as np 
modelName = "modelNoPrecip"
population = 8112505 #DEBUG VERACRUZ
# population = 15203934 #DEBUG BAHIA

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

dataset = read_csv('data/Weekly-Veracruz_28-11-2015-24-03-2018.csv', header=0, index_col=1) # DEBUG

dataset[["Searches"]] /= 100

#Try converting cases to cases per 100K 
dataset[["Cases"]] = dataset[["cases"]].apply(lambda x: x*100000/population, axis=1)


values = dataset.values
# ensure all data is float
values = values.astype('float32')

total_features = len(values[0])

#normalize features per column 
# print("SCALER VAL SHAPE", values.shape)
# min_max_scaler = MinMaxScaler(feature_range=(0, 1))

# scaled = min_max_scaler.fit_transform(values)
# print("Scaled ", scaled[0])

scaled = values #TEMP

#Set number of lag weeks
n_weeks = 3
n_features = 2

# Convert to supervised learning
reframed = series_to_supervised(scaled, n_weeks, 1)
print("Reframed Shape: ", reframed.shape) #Returns an array with shape (n, 32) 32 = 8 features per week and 4 weeks

# split into train and test sets
values = reframed.values
n_train_weeks = int(reframed.shape[0]*0.2)
print("Using {} train hours".format(n_train_weeks))
train = values[:n_train_weeks, :]
test = values [n_train_weeks:, :]


# split into input and outputs TEST
n_obs = n_weeks * n_features


train_X, train_y = train[:, :-1], train[:, -1] #takes last 3 weeks of input as train and last cases as test
test_X, test_y = test[:, :-1], test[:, -1]
print(train_X.shape, len(train_X), train_y.shape)

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("NEWSHAPE", train_X.shape, train_y.shape, test_X.shape, test_y.shape)

model = None

if(isfile("{}.json".format(modelName)) and isfile("{}.h5".format(modelName))):
	json_file = open("{}.json".format(modelName), "r")
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)

	# load weights
	model.load_weights("{}.h5".format(modelName))
	model.compile(loss="mse", optimizer="Nadam")
	print("Loaded model from disk")

	#Make predictions on test data
	yhat = model.predict(test_X)

	print("PRED", yhat)
	test_y = test_y.reshape((len(test_y), 1))
	print("TESTY", test_y)
	# inv_yhat = outputScaler.inverse_transform(yhat)
	# inv_testY = outputScaler.inverse_transform(test_y)

	inv_yhat = np.apply_along_axis(lambda x: x * population / 100000, 1, yhat)
	inv_testY = np.apply_along_axis(lambda x: x * population / 100000, 1, test_y)

	# calculate RMSE
	rmse = sqrt(mean_squared_error(inv_testY, inv_yhat))
	print('Test RMSE: %.3f' % rmse)
	print("Total", sum(inv_testY))
	print("len", len(inv_testY))
	# print("REAL", inv_testY)
	# print("Predicted: ", inv_yhat)

	pyplot.plot(inv_testY)
	pyplot.plot(inv_yhat)
	pyplot.show()

else:

	#Set network 
	model = Sequential()
	# model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
	model.add(LSTM(512, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=False))
	# model.add(Dropout(0.25))
	# model.add(LSTM(50))
	# model.add(Dense(1, activation='linear', kernel_constraint=nonneg())) 
	model.add(Dense(1, activation="relu")) #DEBUG

	model.compile(loss="mse", optimizer="adam", metrics=["mse"])
	history = model.fit(train_X, train_y, epochs = 50, batch_size=52, validation_data=(test_X, test_y), verbose=2, shuffle=False)

	#Save model 
	model_json = model.to_json()
	with open("{}.json".format(modelName), "w") as json_file:
		json_file.write(model_json)
	#seralize weights to HDF5
	model.save_weights("{}.h5".format(modelName))
	print("Saved model to disk")

	# plot history

	if("loss" in history.history):
		pyplot.plot(history.history['loss'], label='train')

	if("val_loss" in history.history):
		pyplot.plot(history.history['val_loss'], label='test')

	if("mean_squared_error" in history.history):
		pyplot.plot(history.history["mean_squared_error"], label="mse")

	if("val_mean_squared_error" in history.history):
		pyplot.plot(history.history["val_mean_squared_error"], label="val_mse")

	pyplot.legend()
	pyplot.show()