# Binary classfier of movie reviews using the IMDB dataset
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf


from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.callbacks import EarlyStopping

print(tf.__version__)

# Download and extract the dataset
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
if not os.path.exists("aclImdb"):
    dataset = tf.keras.utils.get_file(
        "aclImdb_v1", url, untar=True, cache_dir=".", cache_subdir=""
    )
    dataset_dir = os.path.join(os.path.dirname(dataset), "aclImdb")
else:
    dataset_dir = "aclImdb"

# List the directories in the dataset
directories = os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, "train")

sample_file = os.path.join(train_dir, "pos/1181_9.txt")
with open(sample_file) as f:
    # print(f.read())
    pass

# remove the unsup("unsuperivsed") directory
unsup_dir = os.path.join(train_dir, "unsup")
if os.path.exists(os.path.join(train_dir, "unsup")):
    shutil.rmtree(unsup_dir)


# use the text_dataset_from_directory utility to create a labeled dataset 0 for neg directory and 1 for pos directory
# creates a batched dataset
batch_size = 32
seed = 41
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=seed,
)


# print the first 3 reviews and labels
print("\n\n-----------------")
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        # print("Review", text_batch.numpy()[i])
        # print("Label", label_batch.numpy()[i])
        # print("\n\n")
        pass

# check the class names
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

# create a validation set(automatically labels data based on the directory structure)
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=seed,
)

# create a test set(automatically labels data based on the directory structure)
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
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


# vectorize a batch of data
def vectorize_text(text, label):
    text = tf.expand_dims(
        text, -1
    )  # it seems like this line is not needed because the code runs the same without it
    print(f"expand dims {text}")
    return vectorize_layer(text), label


# retrieve a single batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ", vectorize_layer.get_vocabulary()[85])
print(" 17 ---> ", vectorize_layer.get_vocabulary()[17])
print(" 260 ---> ", vectorize_layer.get_vocabulary()[260])
print(" 282 ---> ", vectorize_layer.get_vocabulary()[282])
print("Vocabulary size: {}".format(len(vectorize_layer.get_vocabulary())))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)
print("Train dataset: ", train_ds)
print("Validation dataset: ", val_ds)
print("Test dataset: ", test_ds)

#  configure the dataset for performance
# .cache() Speeds up data loading by storing the dataset in memory or on disk after the first pass.
# .prefetch()  Optimizes training by preparing the next batch of data while the current one is being processed by the model.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("Train dataset: ", train_ds)

# create the model
embedding_dim = 16
model = tf.keras.Sequential(
    [
        layers.Embedding(max_features, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.summary()

# compile the model
model.compile(
    loss=losses.BinaryCrossentropy(),
    optimizer="adam",
    metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)],
)

# train the model
epochs = 10
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)


# evaluate the model
loss, accuracy = model.evaluate(test_ds)
print("Loss: ", loss)
print("Accuracy: ", accuracy)

# create a plot of accuracy and loss over time
history_dict = history.history
print(history_dict.keys())

# plot it
acc = history_dict["binary_accuracy"]
val_acc = history_dict["val_binary_accuracy"]
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

# Include the text vecorization layer inside the model to simplify deployment
export_model = tf.keras.Sequential(
    [vectorize_layer, model, layers.Activation("sigmoid")]
)
export_model.compile(
    loss=losses.BinaryCrossentropy(from_logits=False),
    optimizer="adam",
    metrics=[tf.metrics.BinaryAccuracy(threshold=0.5)],
)
# Test it with `raw_test_ds`, which yields raw strings
print(f"the raw test dataset is {raw_test_ds}")
loss, accuracy = export_model.evaluate(raw_test_ds)
print(f"raw test dataset acc: {accuracy}")


examples = tf.constant(
    [
        "The movie was trash and horrible! I hated it a lot.",
        "The movie was okay.",
        "The movie was terrible...",
        "The movie was bad.",
        "good",
        "bad",
        "The movie was good.",
        "I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge.",
        "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in.",
    ]
)
preds = export_model.predict(examples)
print(preds)
