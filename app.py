import pickle as pk
import streamlit as st
from PIL import Image
import numpy as np

# Load the models
fraud_model = pk.load(open('CNNModel.pkl', 'rb'))
price_model = pk.load(open('PriceModel_Latest.pkl', 'rb'))



# Sidebar navigation
st.sidebar.image("logo.png",  use_container_width=False, width=300)  

option = st.sidebar.radio("Select an option", ["Fraud Accident Detector", "Auction Car Price"])

if option == "Fraud Accident Detector":

    # Accident fraud detection page

    st.title("🚗 Accident Fraud Detector")
    st.markdown("""
        <style>
        .big-font {
            font-size: 20px !important;
            color: #4F8BF9;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<p class="big-font">Upload the image of the accident to check if it\'s real or fake.</p>', unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
       
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Preprocess the image
        with st.spinner('Processing image...'):
            image = image.resize((150, 150))  # Resize to the input size
            image = np.array(image) / 255.0  # Normalize pixel values
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Prediction
            prediction = fraud_model.predict(image)

       
        st.success("Prediction complete!")
        

        predict = prediction[0]

        if predict > 0.7:
            st.markdown("### ✅ **Prediction: Real Accident**")
              
        else:
            st.markdown("### ❌ **Prediction: Fake Accident**")
            st.error("This image is likely to be a fake accident.")
            st.balloons()
            

elif option == "Auction Car Price":
    # Auction Car Price Prediction page
    st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f0f2f6, #e0e5ec);
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .header {
        font-size: 36px;
        color: #4CAF50;
        font-weight: bold;
        text-align: center;
        margin-bottom: 25px;
    }
    .prediction-box {
        background-color: #4CAF50;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 14px;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header">🚗 Vehicle Auction Price Predictor</div>', unsafe_allow_html=True)

    
    with st.container():
        st.markdown('<div class="main">', unsafe_allow_html=True)

        # Input fields
        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input('📅 Year of Manufacture', min_value=1990, max_value=2024, step=1, value=2020)
            mileage = st.number_input('🚗 Mileage (miles)', min_value=0, step=1000, value=10000)
            tax = st.number_input('💰 Road Tax (£)', min_value=0, step=10, value=150)

        with col2:
            mpg = st.number_input('⛽ Miles Per Gallon (MPG)', min_value=10.0, step=0.1, value=50.0)
            engineSize = st.number_input('🔧 Engine Size (L)', min_value=0.5, step=0.1, value=1.5)

        brand = st.selectbox('🏷 Select Brand', ['Audi', 'BMW', 'Ford', 'Toyota', 'Other'])

        # Conditional model options based on selected brand
        if brand == 'Audi':
            model_options = ['A3', 'A4', 'Q3', 'Other']
        elif brand == 'BMW':
            model_options = ['1 Series', '2 Series', '3 Series', 'Other']
        elif brand == 'Toyota':
            model_options = ['Aygo', 'Yaris', 'Other']
        elif brand == 'Ford':
            model_options = ['EcoSport', 'Fiesta', 'Focus', 'Kuga', 'Other']
        else:
            model_options = ['Other']  

        model_name = st.selectbox('🚘 Select Model', model_options)

        
        transmission = st.selectbox('⚙️ Transmission Type', ['Automatic', 'Manual', 'Semi-Auto'])
        fuel_type = st.selectbox('⛽ Fuel Type', ['Diesel', 'Petrol'])

        # Encoding categorical variables
        brand_dict = {'Audi': [1, 0, 0, 0, 0], 'BMW': [0, 1, 0, 0, 0], 'Ford': [0, 0, 1, 0, 0], 'Toyota': [0, 0, 0, 0, 1], 'Other': [0, 0, 0, 1, 0]}
        model_dict = {
            '1 Series': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            '2 Series': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            '3 Series': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'A3': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            'A4': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 
            'Aygo': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
            'EcoSport': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 
            'Fiesta': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 
            'Focus': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 
            'Kuga': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 
            'Q3': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
            'Yaris': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
            'Other': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  
        }
        transmission_dict = {'Automatic': [1, 0, 0], 'Manual': [0, 1, 0], 'Semi-Auto': [0, 0, 1]}
        fuel_type_dict = {'Diesel': [1, 0], 'Petrol': [0, 1]}

        # Encode inputs
        brand_encoded = brand_dict[brand]
        model_encoded = model_dict[model_name]
        transmission_encoded = transmission_dict[transmission]
        fuel_encoded = fuel_type_dict[fuel_type]

        # Convert inputs to array for prediction
        input_data = np.array([[year, mileage, tax, mpg, engineSize] + brand_encoded + model_encoded + transmission_encoded + fuel_encoded])

        if st.button('🔮 Predict Price'):
            if price_model is not None:
                try:
                    prediction = price_model.predict(input_data)
                    st.markdown(f'<div class="prediction-box">💰 Estimated Price : £ {float(prediction[0]):,.2f}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
            else:
                st.error("Model is not loaded. Please check the model file.")

        st.markdown('</div>', unsafe_allow_html=True)

