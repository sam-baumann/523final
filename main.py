import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = pd.read_csv('dataset.csv', names=['type', 'description'])

target = data['type']
values = data['description']

#encode the target
enc = LabelEncoder()
target = enc.fit_transform(target)

max_features = 20000
embedding_dim = 128
sequence_length = 500

#data cleanup will go here.

vectorize_layer = TextVectorization(max_tokens=max_features, output_mode='int')
vectorize_layer.adapt(values)

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
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#fit the model
model.fit(values, target, batch_size=32, epochs=10, validation_split=0.1)