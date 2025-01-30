import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import traceback
import requests  

API_URL = "https://api-inference.huggingface.co/models/facebook/opt-350m"  # Switched to a smaller model
API_KEY = "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"  
MODEL_PATH = "trainedmodelfinal.pkl"
XAI_IMAGE_PATH = "feature importance.png"
HEAT_THRESHOLDS = {
    'critical_temp': 38.0,
    'green_cover_min': 25,
    'albedo_min': 0.4,
    'building_height_max': 35,
    'heat_stress_max': 4.0,
    'population_density_max': 15000
}

def generate_suggestions(prompt):
    headers = {"Authorization": f"Bearer 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"}
    payload = {"inputs": prompt, "parameters": {"max_length": 250, "temperature": 0.7}}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        response_json = response.json()
        if isinstance(response_json, list) and len(response_json) > 0 and 'generated_text' in response_json[0]:
            return response_json[0]['generated_text']
        else:
            return "Error: Unexpected response format"
    else:
        return f"API Error: {response.status_code} - {response.text}"

st.set_page_config(page_title="Urban Heat Analyst", layout="wide")
st.title("Urban Heat Analysis with AI")

cities = st.sidebar.selectbox("Select City", ["Delhi", "Mumbai", "Hyderabad"])
st.sidebar.header("Urban Parameters")
inputs = {}

try:
    inputs['Latitude'] = st.number_input("Latitude", 19.0, 19.2, 19.0760, 0.0001)
    inputs['Longitude'] = st.number_input("Longitude", 72.8, 73.0, 72.8777, 0.0001)
    inputs['Population Density'] = st.number_input("Population Density (people/km²)", 1000, 50000, 20000)
    inputs['Albedo'] = st.slider("Albedo", 0.0, 1.0, 0.3, 0.05)
    inputs['Green Cover Percentage'] = st.slider("Green Cover (%)", 0, 100, 25)
    inputs['Relative Humidity'] = st.slider("Humidity (%)", 0, 100, 60)
    inputs['Wind Speed'] = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1)
    inputs['Building Height'] = st.slider("Building Height (m)", 5, 150, 30)
    inputs['Road Density'] = st.slider("Road Density (km/km²)", 0.0, 20.0, 5.0, 0.1)
    inputs['Proximity to Water Body'] = st.slider("Water Proximity (m)", 0, 5000, 1000)
    inputs['Solar Radiation'] = st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
    inputs['Nighttime Surface Temperature'] = st.slider("Night Temp (°C)", 15.0, 40.0, 25.0, 0.1)
    inputs['Distance from Previous Point'] = st.number_input("Distance from Previous Point (m)", 0, 5000, 100)
    inputs['Heat Stress Index'] = st.slider("Heat Stress Index", 0.0, 10.0, 3.5, 0.1)
    inputs['Urban Vegetation Index'] = st.slider("Vegetation Index", 0.0, 1.0, 0.5, 0.01)
    inputs['Carbon Emission Levels'] = st.number_input("CO₂ Levels (ppm)", 300, 1000, 400)
    inputs['Surface Material'] = st.selectbox("Surface Material", ["Concrete", "Asphalt", "Grass", "Water", "Mixed"])
    inputs['Land Cover Type'] = st.selectbox("Land Cover Type", ["Urban", "Residential", "Water", "Vegetation", "Industrial", "Bare Soil"])
    inputs['Cooling Measures Present'] = st.selectbox("Cooling Measures Present", ["None", "Green Roofs", "Reflective Paint", "Rooftop Garden", "Shaded Streets", "Water Features"])
except KeyError as e:
    st.error(f"Missing input field: {str(e)}")
    st.stop()

if st.sidebar.button("Analyze Urban Heat"):
    try:
        model = joblib.load(MODEL_PATH)  # Load model only when needed
        required_features = model.feature_names_in_
        xai_image = Image.open(XAI_IMAGE_PATH)
        
        missing_features = [f for f in required_features if f not in inputs]
        if missing_features:
            st.error(f"Missing features: {', '.join(missing_features)}")
            st.stop()
        
        input_df = pd.DataFrame([inputs], columns=required_features)
        prediction = model.predict(input_df)[0]
        
        st.subheader("Urban Heat Analysis")
        st.metric("Predicted Surface Temperature", f"{prediction:.1f}°C")
        st.image(xai_image, caption="Feature Impact Analysis", use_column_width=True)
        
        prompt = "Suggest ways to reduce urban heat based on green cover, albedo, and cooling measures."
        suggestions = generate_suggestions(prompt)
        
        st.subheader("Recommendations")
        st.write(suggestions)
    except FileNotFoundError:
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"Analysis Failed: {traceback.format_exc()}")
