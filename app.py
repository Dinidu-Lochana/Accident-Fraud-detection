import pickle as pk
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow import keras  

# Load the pre-trained model
model = pk.load(open('CNNModel.pkl', 'rb'))

# Streamlit app
st.title("🚗 Accident Fraud Detection System")
st.markdown("""
    <style>
    .big-font {
        font-size: 20px !important;
        color: #4F8BF9;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">Upload an image of an accident to check if it\'s real or fake.</p>', unsafe_allow_html=True)

# Add logo to the sidebar
st.sidebar.image("logo.png",  use_container_width=False, width=300)  



# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image for the model
    with st.spinner('Processing image...'):
        image = image.resize((150, 150))  # Resize to the input size expected by the model
        image = np.array(image) / 255.0  # Normalize pixel values (if required)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(image)

    # Display the result
    st.success("Prediction complete!")
    st.write("Prediction Probabilities:", prediction)

    # If your model outputs probabilities, you can display the class with the highest probability
    if len(prediction[0]) > 1:  # For multi-class classification
        predicted_class = np.argmax(prediction[0])
        st.markdown(f"### 🎯 Predicted Class: **{predicted_class}**")
    else:  # For binary classification
        if prediction[0] > 0.5:
            st.markdown("### ✅ **Prediction: Real Accident**")
            st.balloons()  # Celebrate if it's a real accident
        else:
            st.markdown("### ❌ **Prediction: Fake Accident**")
            st.error("This image is likely to be a fake accident.")

# Footer
st.markdown("---")
st.markdown("### 🛠️ Built with ❤️ using [Streamlit](https://streamlit.io)")