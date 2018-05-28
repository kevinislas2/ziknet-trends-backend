from pandas import read_csv, DataFrame, concat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.optimizers import Adam
from keras.constraints import nonneg
from matplotlib import pyplot
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error
from keras.models import model_from_json
from os.path import isfile
import numpy as np 
import sys

# modelName = "modelNoPrecip"

trainStates = {
	"Veracruz" : "data/Train/Weekly-Veracruz_28-11-2015-24-03-2018.csv",
	"Yucatan" : "data/Train/Weekly-Yucatan_28-11-2015-24-03-2018.csv",
	"Guerrero" : "data/Train/Weekly-Guerrero_28-11-2015-24-03-2018.csv",
	"QuintanaRoo" : "data/Train/Weekly-QuintanaRoo_28-11-2015-24-03-2018.csv",
	
}

testStates = {
	"NuevoLeon": "data/Test/Weekly-NuevoLeon_28-11-2015-24-03-2018.csv",
	"MatoGrosso" : "data/Test/Weekly-Mato_Grosso_10-01-2015-20-02-2016.csv",
	"Bahia" : "data/Test/Weekly-Bahia_10-01-2015-15-05-2016.csv",
	"Chiapas" : "data/Test/Weekly-Chiapas_28-11-2015-24-03-2018.csv",
}

populations = {
	"data/Train/Weekly-Veracruz_28-11-2015-24-03-2018.csv" : 8112505,
	"data/Train/Weekly-Yucatan_28-11-2015-24-03-2018.csv" : 2097175,
	"data/Train/Weekly-Guerrero_28-11-2015-24-03-2018.csv" : 3533251,
	"data/Train/Weekly-QuintanaRoo_28-11-2015-24-03-2018.csv": 1501562,
	"data/Test/Weekly-Chiapas_28-11-2015-24-03-2018.csv" : 5217908,

	"data/Test/Weekly-NuevoLeon_28-11-2015-24-03-2018.csv" : 5119504,
	"data/Test/Weekly-Mato_Grosso_10-01-2015-20-02-2016.csv" : 3344544,
	"data/Test/Weekly-Bahia_10-01-2015-15-05-2016.csv" : 15344447,
	
}
# population = 8112505 #DEBUG VERACRUZ

# population = 5119504 #DEBUG NUEVO LEON

# population = 5217908 #DEBUG CHIAPAS

# population = 2097175 #DEBUG YUCATAN

# population = 3533251 #DEBUG GUERRERO

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

def getXY(filename):
	global populations
	scale = populations[filename]

	dataset = read_csv(filename, header=0, index_col=0)
	dataset[["Searches"]] /= 100
	dataset[["Cases"]] = dataset[["Cases"]].apply(lambda x: x*100000/scale, axis=1)


	values = dataset.values.astype("float32")
	total_features = len(values[0])

	n_weeks = 4
	n_features = 2

	reframed = series_to_supervised(values, n_weeks, 1)
	values = reframed.values
	print("Reframed Shape: ", reframed.shape)
	totalFeatures = reframed.shape[1]
	n_obs = n_weeks * n_features

	x, y = values[:, :-2], values[:, -1] # Pick last week's cases as y and drop last week's 

	x = x.reshape((x.shape[0], n_weeks, n_features)) # Reshape as 3-D
	return x, y

def saveModel(model, modelName):
	jsonName = "{}.json".format(modelName)
	h5Name = "{}.h5".format(modelName)

	model_json = model.to_json()
	with open(jsonName, "w") as json_file:
		json_file.write(model_json)
	#seralize weights to HDF5
	model.save_weights(h5Name)


def loadOrCreateModel(modelName, x): 

	jsonName = "{}.json".format(modelName)
	h5Name = "{}.h5".format(modelName)

	if(isfile(jsonName) and isfile(h5Name)):

		loaded_model_json = None
		with open(jsonName, "r") as json_file:
			loaded_model_json = json_file.read()

		model = model_from_json(loaded_model_json)
		model.load_weights(h5Name)
		model.compile(loss="mse", optimizer="rmsprop", metrics=["mse"])
		return model
	else:
		model = Sequential()
		model.add(LSTM(128, input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
		model.add(LSTM(64, activation="relu", return_sequences=True))
		model.add(LSTM(32, activation="relu", return_sequences=False))
		
		# model.add(Dense(128, activation="relu"))
		model.add(Dense(1, activation='linear', kernel_constraint=nonneg()))
		# model.add(Dense(1, activation="relu", kernel_constraint=nonneg()))
		model.compile(loss="mse", optimizer="rmsprop", metrics=["mse"])
		model.summary()
		return model

def formatFilename(filename):
	fFilename = filename.split(".")[0]
	fFilename = fFilename.split("/")[-1]
	fFilename = fFilename.replace("Weekly-", "")
	return fFilename

def main():
	train = False
	modelName = None
	if len(sys.argv) != 3:
		print("Usage: finalLSTM.py modelName [train,test]")
		exit()
	else:
		modelName = sys.argv[1]

		if(sys.argv[2] == "train"):
			train = True

	# filename = trainStates["Yucatan"]
	# filename = trainStates["Guerrero"]
	filename = trainStates["Veracruz"]

	# filename = trainStates["QuintanaRoo"]
	# filename = trainStates["Chiapas"]
	# filename = trainStates["Rio"]

	x, y = getXY(filename)

	val_x, val_y = getXY(testStates["Bahia"])

	model = loadOrCreateModel(modelName, x)
	
	history = None
	if(train):
		history = model.fit(x, y,
			epochs=10, 
			batch_size=x.shape[0],
			validation_data=(val_x, val_y), 
			verbose=1, shuffle=False)
		saveModel(model, modelName)
	else:
		global populations

		# testFile = trainStates["Veracruz"]
		# testFile = trainStates["Yucatan"]
		# testFile = trainStates["Guerrero"]
		testFile = trainStates["QuintanaRoo"]

		# testFile = testStates["NuevoLeon"]
		# testFile = testStates["MatoGrosso"]
		# testFile = testStates["Bahia"]
		# testFile = testStates["Chiapas"]

		scale = populations[testFile]
		x,y = getXY(testFile)
		predictions = model.predict(x)
		y = y.reshape((len(y), 1))

		inv_yPred = np.apply_along_axis(lambda x: x * scale / 100000, 1, predictions)
		inv_y = np.apply_along_axis(lambda x: x * scale / 100000, 1, y)

		rmse = sqrt(mean_squared_error(inv_y, inv_yPred))
		print('Test RMSE: %.3f' % rmse)
		print("Total", sum(inv_y))
		print("len", len(inv_y))
		pyplot.title("Cases {} RMSE: {:.2f}".format(formatFilename(testFile), rmse))
		pyplot.ylabel("Cases")
		pyplot.xlabel("Week #")
		pyplot.plot(inv_y, label="Cases")
		pyplot.plot(inv_yPred, label="Predictions")
		pyplot.legend()
		pyplot.show()

if __name__ == '__main__':
	main()