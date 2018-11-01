from keras.models import model_from_json
from keras.utils import plot_model
json_file = open("MODEL.json", "r")
loaded_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_json)

plot_model(loaded_model, to_file='model.png', show_shapes=True)