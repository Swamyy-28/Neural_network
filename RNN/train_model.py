import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Load Dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Padding
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)

# Build RNN Model
model = Sequential()

model.add(
    Embedding(
        input_dim=10000,
        output_dim=32,
        input_length=200
    )
)

model.add(
    SimpleRNN(32)
)

model.add(
    Dense(
        1,
        activation='sigmoid'
    )
)

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test)
)

# Save Model
model.save("rnn_imdb_model.h5")

print("Model Saved Successfully")