import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    tf.keras.models.load_model(
    "CNN/models/cnn_fashion_model.h5"
)

model = load_model()

# =====================================================
# CLASS LABELS
# =====================================================

classes = [
    "T-Shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]

# =====================================================
# TITLE
# =====================================================

st.title("🧠 CNN Fashion MNIST Classification")

st.markdown(
"""
This application uses a Convolutional Neural Network (CNN)
trained on the Fashion MNIST dataset to classify fashion items.
"""
)

# =====================================================
# DATASET INFORMATION
# =====================================================

st.header("Dataset Information")

dataset_info = pd.DataFrame({
    "Class Index":[0,1,2,3,4,5,6,7,8,9],
    "Class Name":classes
})

st.dataframe(dataset_info)

# =====================================================
# FILE UPLOAD
# =====================================================

st.header("Upload Image")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["png","jpg","jpeg"]
)

# =====================================================
# PREDICTION
# =====================================================

if uploaded_file is not None:

    col1,col2 = st.columns(2)

    with col1:

        image = Image.open(uploaded_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    image_gray = image.convert("L")

    image_resized = image_gray.resize((28,28))

    img_array = np.array(image_resized)

    img_array = img_array / 255.0

    img_array = img_array.reshape(
        1,
        28,
        28,
        1
    )
    st.write("Image Shape:", img_array.shape)
    st.write("Model:", model)

    prediction = model.predict(img_array)

    predicted_class = np.argmax(
        prediction
    )

    confidence = np.max(
        prediction
    ) * 100

    with col2:

        st.header("Prediction Result")

        st.success(
            f"Predicted Class: {classes[predicted_class]}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

    # =================================================
    # PROBABILITY CHART
    # =================================================

    st.header("Prediction Probabilities")

    prob_df = pd.DataFrame({
        "Class": classes,
        "Probability": prediction[0]
    })

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    ax.bar(
        prob_df["Class"],
        prob_df["Probability"]
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    st.pyplot(fig)

    # =================================================
    # TABLE
    # =================================================

    st.header("Probability Table")

    st.dataframe(
        prob_df.sort_values(
            by="Probability",
            ascending=False
        )
    )



# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.write(
    "Developed using TensorFlow, CNN and Streamlit"
)
