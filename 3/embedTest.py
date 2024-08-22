import numpy as np
from tensorflow import keras

model = keras.Sequential()
model.add(keras.layers.Embedding(1000, 64))
# The model will take as input an integer matrix of size (batch,
# input_length), and the largest integer (i.e. word index) in the input
# should be no larger than 999 (vocabulary size).
# Now model.output_shape is (None, 10, 64), where `None` is the batch
# dimension.
input_array = np.random.randint(1000, size=(32, 10))
print(input_array.shape)
model.compile("rmsprop", "mse")
output_array = model.predict(input_array)
print(output_array.shape)

