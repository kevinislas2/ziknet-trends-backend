from finalLSTM import loadOrCreateModel
from keras.utils import plot_model
def main():
	model = loadOrCreateModel("TESTMODEL", [])
	plot_model(model, show_shapes=True, to_file='model.png')

if __name__ == '__main__':
	main()