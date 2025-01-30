import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import traceback
import requests  

API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-1.3B"  
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
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {"inputs": prompt, "parameters": {"max_length": 250, "temperature": 0.7}}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text'] if response.status_code == 200 else f"Error: {response.status_code}"

try:
    model = joblib.load(MODEL_PATH)
    required_features = model.feature_names_in_
    xai_image = Image.open(XAI_IMAGE_PATH)
except Exception as e:
    st.error(f"Initialization Error: {traceback.format_exc()}")
    st.stop()

st.set_page_config(page_title="Urban Heat Analyst", layout="wide")
st.title("Urban Heat Analysis with AI")

cities = st.sidebar.selectbox("Select City", ["Delhi", "Mumbai", "Hyderabad"])
st.sidebar.header("Urban Parameters")

inputs = {}
inputs['Longitude'] = st.number_input("Longitude", 
                                      min_value=float(72.8), 
                                      max_value=float(77.2), 
                                      value=float(77.2090), 
                                      step=float(0.001), 
                                      format="%.6f")

inputs['Latitude'] = st.number_input("Latitude", 
                                     min_value=float(18.9), 
                                     max_value=float(28.6), 
                                     value=float(19.0760), 
                                     step=float(0.001), 
                                     format="%.6f")

inputs['Population Density'] = st.number_input("Population Density (people/km²)", 1000, 50000, 20000)
inputs['Albedo'] = st.slider("Albedo", 0.0, 1.0, 0.3, 0.05)
inputs['Green Cover Percentage'] = st.slider("Green Cover (%)", 0, 100, 25)
inputs['Humidity'] = st.slider("Humidity (%)", 0, 100, 60)
inputs['Wind Speed'] = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1)
inputs['Building Height'] = st.slider("Building Height (m)", 5, 150, 30)
inputs['Road Density'] = st.slider("Road Density (km/km²)", 0.0, 20.0, 5.0, 0.1)
inputs['Water Proximity'] = st.slider("Proximity to Water Body (m)", 0, 5000, 1000)
inputs['Solar Radiation'] = st.slider("Solar Radiation (W/m²)", 0, 1000, 500)
inputs['Heat Stress Index'] = st.slider("Heat Stress Index", 0.0, 10.0, 3.5, 0.1)
inputs['Carbon Emission Levels'] = st.number_input("CO₂ Levels (ppm)", 300, 1000, 400)
inputs['Land Cover Type'] = st.selectbox("Land Cover Type", ["Urban", "Vegetation", "Water", "Bare Soil", "Industrial", "Residential"])
inputs['Cooling Measures'] = st.selectbox("Cooling Measures", ["None", "Water Features", "Reflective Paint", "Rooftop Garden", "Shaded Streets", "Green Roofs"])

if st.sidebar.button("Analyze Urban Heat"):
    try:
        missing_features = [f for f in required_features if f not in inputs]
        if missing_features:
            st.error(f"Missing features: {', '.join(missing_features)}")
            st.stop()
        
        input_df = pd.DataFrame([inputs], columns=required_features)
        prediction = model.predict(input_df)[0]
        
        st.subheader("Urban Heat Analysis")
        st.metric("Predicted Surface Temperature", f"{prediction:.1f}°C")
        st.image(xai_image, caption="Feature Impact Analysis", use_column_width=True)
        
        prompt = f"""
        Analyze urban heat issues in {cities} with these factors:
        - Surface Temperature: {prediction:.1f}°C
        - Green Cover: {inputs['Green Cover Percentage']}%
        - Albedo: {inputs['Albedo']}
        - Population Density: {inputs['Population Density']} people/km²
        - Building Height: {inputs['Building Height']}m
        - Heat Stress Index: {inputs['Heat Stress Index']}
        - Land Cover Type: {inputs['Land Cover Type']}
        - Cooling Measures: {inputs['Cooling Measures']}
        Provide actionable mitigation strategies.
        """
        
        suggestions = generate_suggestions(prompt)
        st.subheader("Recommendations")
        st.write(suggestions)
    except Exception as e:
        st.error(f"Analysis Failed: {traceback.format_exc()}")
