
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------
# Page Configuration
# -----------------------------------------

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="MRI",
    layout="wide"
)

# -----------------------------------------
# Model Configuration
# -----------------------------------------

MODEL_PATH = "/content/brain_tumor_app/brain_tumor_classification_model.keras"

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# -----------------------------------------
# Load Model
# -----------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# -----------------------------------------
# Custom CSS
# -----------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 19px;
    text-align: center;
    color: #666666;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    text-align: center;
    margin-top: 20px;
}

.result-title {
    font-size: 18px;
    color: #666666;
}

.result-value {
    font-size: 32px;
    font-weight: 700;
    margin-top: 8px;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Header
# -----------------------------------------

st.markdown(
    '<div class="main-title">Brain Tumor Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">MRI Image Classification Using Deep Learning</div>',
    unsafe_allow_html=True
)

# -----------------------------------------
# About
# -----------------------------------------

with st.expander("About This Project"):

    st.write(
        "This application uses a deep learning model to classify "
        "brain MRI images into four categories: Glioma, Meningioma, "
        "No Tumor, and Pituitary."
    )

    st.write(
        "The model uses MobileNetV2 as the pretrained feature extractor "
        "and was trained using the Brain Tumor MRI Dataset."
    )

# -----------------------------------------
# Upload Section
# -----------------------------------------

st.markdown(
    '<div class="section-title">Upload MRI Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------------------
# Prediction
# -----------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded MRI")

        st.image(
            image,
            caption="Uploaded MRI Image",
            width=450
        )

    # Resize
    image_resized = image.resize((224, 224))

    # Convert to array
    image_array = np.array(image_resized).astype("float32")

    # Normalize once
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[0][predicted_index] * 100
    )

    with col2:

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div class="result-box">

            <div class="result-title">
            Predicted Class
            </div>

            <div class="result-value">
            {predicted_class}
            </div>

            <br>

            <div class="result-title">
            Confidence
            </div>

            <div class="result-value">
            {confidence:.2f}%
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------
    # Class Probabilities
    # -----------------------------------------

    st.subheader("Class Probabilities")

    for class_name, probability in zip(
        CLASS_NAMES,
        predictions[0]
    ):

        percentage = float(probability * 100)

        st.write(
            f"**{class_name}: {percentage:.2f}%**"
        )

        st.progress(
            min(int(percentage), 100)
        )


# -----------------------------------------
# Supported Classes
# -----------------------------------------

st.subheader("Supported Classes")

st.write(
    "Glioma  |  Meningioma  |  No Tumor  |  Pituitary"
)

# -----------------------------------------
# Disclaimer
# -----------------------------------------



# -----------------------------------------
# Footer
# -----------------------------------------

st.divider()

st.caption(
    "AI-Based Brain Tumor Detection and Classification Using MRI Images"
)

