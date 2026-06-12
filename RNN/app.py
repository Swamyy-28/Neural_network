import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load Model
model = tf.keras.models.load_model(
    "rnn_imdb_model.h5"
)

st.title("🎬 IMDb Review Sentiment Analysis using RNN")

st.write(
    "Enter a movie review and predict whether it is Positive or Negative."
)

review = st.text_area(
    "Enter Review"
)

if st.button("Predict"):

    tokenizer = Tokenizer(num_words=10000)

    tokenizer.fit_on_texts([review])

    sequence = tokenizer.texts_to_sequences(
        [review]
    )

    padded = pad_sequences(
        sequence,
        maxlen=200
    )

    prediction = model.predict(
        padded
    )

    if prediction[0][0] > 0.5:
        st.success(
            "Positive Review 😊"
        )
    else:
        st.error(
            "Negative Review 😞"
        )

    st.write(
        f"Prediction Score: {prediction[0][0]:.4f}"
    )