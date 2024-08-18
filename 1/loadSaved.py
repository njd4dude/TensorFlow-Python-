# Import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Class names for the dataset
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# load the model
model = tf.keras.models.load_model("prob_model.keras")
# model.summary()

# Load the Fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# predict
predictions = model.predict(test_images)

# evaluate predictions
index = 2050
indexPrediction = np.argmax(predictions[index])
indexActual = test_labels[index]

# print out results
print(f"Prediction: {indexPrediction} - {class_names[indexPrediction]}")
print(f"Actual: {indexActual} - {class_names[indexActual]}")

# look at the image the model is trying to predict
plt.figure()
plt.imshow(test_images[index])
plt.colorbar()
plt.grid(False)
plt.show()
