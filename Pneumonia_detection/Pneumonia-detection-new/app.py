import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# Load model from Hugging Face Hub
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="your-username/your-repo", filename="pneumonia_model.h5")
    return tf.keras.models.load_model(model_path)

model = load_model()

# Preprocess function
def preprocess_image(img):
    img = img.convert("L").resize((150, 150))  # Convert to grayscale & resize
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 150, 150, 1)  # Match input shape
    return img_array

# Streamlit UI
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image and the model will predict if pneumonia is present.")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_image = preprocess_image(image)
        prediction = model.predict(input_image)[0][0]
        result = "Pneumonia Detected 😷" if prediction > 0.5 else "Normal 😊"
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {prediction:.2f}")
