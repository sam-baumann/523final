import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

def standardize_data(data):
    """
    Standardize the data.
    """
    lowercase = tf.strings.lower(data)
    ret = tf.strings.regex_replace(lowercase, r'[^a-z0-9\s]', '')
    return ret

data = pd.read_csv('dataset.csv', names=['type', 'description'])

#encode the target
enc = LabelEncoder()
data['type'] = enc.fit_transform(data['type'])

x_train, x_test, y_train, y_test = train_test_split(data['description'], data['type'], test_size=0.2)

max_features = 20000
embedding_dim = 128
sequence_length = 500

#data cleanup will go here.

vectorize_layer = TextVectorization(max_tokens=max_features, output_mode='int', standardize=standardize_data)
vectorize_layer.adapt(x_train)

text_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='text_input')
x = vectorize_layer(text_input)
x = layers.Embedding(max_features, embedding_dim)(x)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(text_input, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#fit the model
model.fit(x_train, y_train, epochs=3, validation_split=0.1)

#evaluate the model
model.evaluate(x_test, y_test)