# Exercise: modify to -> multi-class classification on Stack Overflow question task left off here
# task left off here 8/21 getting it work good but I want to learn how to optimize the model or the dataset to get better results and accuracy
# NOTE: -must have Stack Overflow dataset
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


# extract the dataset
dataset_dir = "StackOFData"

# List the directories in the dataset
directories = os.listdir(dataset_dir)
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")


# creates a batched dataset
batch_size = 32
seed = 41
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed,
)

# print the first 3 reviews and labels of the first batch
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        # print("Review", text_batch.numpy()[i])
        # print("Label", label_batch.numpy()[i])
        # print("\n\n")
        pass


# create a validation set(automatically labels data based on the directory structure)
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed,
)

# create a test set(automatically labels data based on the directory structure)
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    test_dir, batch_size=batch_size
)


# strip the html tags
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


# create a text vectorization layer(converts text into numerical representation)
max_features = 10000
sequence_length = 250
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

# calculates vocabulary from training text so that this layer can now encode strings to integer sequences during training
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# vectorize a batch of data .. you acutally can pass batches
def vectorize_text(text, label):
    test = vectorize_layer(text)
    vectorized = vectorize_layer(text), label

    return vectorized


# retrieve a single batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_question, first_label = text_batch[0], label_batch[0]
print("Stack OF Question", first_question)
print("Label", raw_train_ds.class_names[first_label])
vectorizedReview = vectorize_text(first_question, first_label)
text, label = vectorizedReview
print("Vectorized review", vectorizedReview)

# print the vocabulary
print("23 ---> ", vectorize_layer.get_vocabulary()[23])
print(" 702 ---> ", vectorize_layer.get_vocabulary()[702])
print("Vocabulary size: {}".format(len(vectorize_layer.get_vocabulary())))

# vectorize the raw datasets(creates a vector for every example)
print("\n\n")
num_batches = len(raw_train_ds)
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


#  configure the dataset for performance
# .cache() Speeds up data loading by storing the dataset in memory or on disk after the first pass.
# .prefetch()  Optimizes training by preparing the next batch of data while the current one is being processed by the model.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# create the model
embedding_dim = 16
model = tf.keras.Sequential(
    [
        layers.Embedding(max_features, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(4, activation="softmax"),
    ]
)
model.summary()

# compile the model
model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=["accuracy"],
)

# train the model
print("train dataset: ", type(train_ds))
print("validation dataset: ", val_ds)
epochs = 50
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


# evaluate the model
print(f"test_ds shape: {test_ds}")
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# create a plot of accuracy and loss over time
history_dict = history.history
def plot():
    # plot it
    acc = history_dict["accuracy"]
    val_acc = history_dict["val_accuracy"]
    loss = history_dict["loss"]
    val_loss = history_dict["val_loss"]
    epochs = range(1, len(acc) + 1)
    # "bo" is for "blue dot"
    plt.plot(epochs, loss, "bo", label="Training loss")
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(epochs, acc, "bo", label="Training acc")
    plt.plot(epochs, val_acc, "b", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.show()

# ALTERNATIVE OPTION: Include the text vecorization layer inside the model to simplify deployment
export_model = tf.keras.Sequential([vectorize_layer, model])
export_model.summary()
export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=["accuracy"],
)
# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(f"\n\nraw test dataset acc: {accuracy}")


examples = tf.constant(
    [
        "I have a list of lists, and I want to flatten it into a single list. I'm using a list comprehension, but it's returning a nested list instead. nested_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] flattened_list = [x for x in nested_list] print(flattened_list)",
        "async function fetchData() { const response = fetch('https://jsonplaceholder.typicode.com/todos/1'); return response; } const data = fetchData(); console.log(data); // Output: Promise { <pending> } # I’m trying to fetch some data using an `async` function in blank, but instead of the data, I'm getting a pending promise. How can I properly wait for the promise to resolve and access the data?",
        'public class Main { public static void main(String[] args) { String str = null; if (str.equals("Hello")) { System.out.println("Hello World"); } } } # I\'m trying to check if a string equals "Hello". I initialized the string, but my program throws a `NullPointerException`. What\'s causing this, and how can I fix it?',
        "List<int> numbers = new List<int> { 1, 2, 3, 4, 5 }; foreach (int number in numbers) { if (number % 2 == 0) { numbers.Remove(number); } } # I’m trying to remove all even numbers from a list using a `foreach` loop in blank, but I’m getting an `InvalidOperationException`. How can I modify the list while iterating through it?",
        "public system.out.println('hello world') static ",
        "how to download  file in blank i want to download file using blank.  new bu",
    ]
)
print(f"examples shape: {examples.shape}")
preds = export_model.predict(examples)
preds = np.array(preds)

# Set numpy print options to avoid scientific notation
np.set_printoptions(precision=4, suppress=True)

print(preds)
print(f"preds shape: {preds.shape}")

print("\n\n")
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])
print("\n\n")
