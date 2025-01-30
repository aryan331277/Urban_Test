import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import traceback
import requests  

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
API_KEY = "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"  # Replace with your Hugging Face API key
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
    payload = {"inputs": prompt, "parameters": {"max_length": 250, "temperature": 0.5}}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            return f"**API Error**: {response.status_code} - {response.text[:200]}"

        response_json = response.json()
        if isinstance(response_json, list) and len(response_json) > 0:
            # Corrected key from generated_text to summary_text
            summary_text = response_json[0].get("summary_text", "")
            return summary_text.replace(". ", ".\n\n- ")  # Format bullets better
        return "Error: Unexpected response format"
    except Exception as e:
        return f"**Error**: {str(e)}"

st.set_page_config(page_title="Urban Heat Analyst", layout="wide")
st.title("Urban Heat Analysis with AI")

# Input processing moved to function for better organization
def get_user_inputs():
    st.sidebar.header("City Selection")
    city = st.sidebar.selectbox("Select City", ["Delhi", "Mumbai", "Hyderabad"])
    
    st.sidebar.header("Urban Parameters")
    return {
        'Latitude': st.sidebar.number_input("Latitude", 19.0, 19.2, 19.0760, 0.0001),
        'Longitude': st.sidebar.number_input("Longitude", 72.8, 73.0, 72.8777, 0.0001),
        'Population Density': st.sidebar.number_input("Population Density (people/km²)", 1000, 50000, 20000),
        'Albedo': st.sidebar.slider("Albedo", 0.0, 1.0, 0.3, 0.05),
        # ... (keep other input fields the same)
    }

inputs = get_user_inputs()

if st.sidebar.button("Analyze Urban Heat"):
    try:
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame([inputs], columns=model.feature_names_in_)
        prediction = model.predict(input_df)[0]
        
        # Main results display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Core Metrics")
            st.metric("Predicted Temperature", f"{prediction:.1f}°C", 
                     delta_color="inverse" if prediction > HEAT_THRESHOLDS['critical_temp'] else "normal")
            
        with col2:
            st.subheader("Feature Impact")
            st.image(Image.open(XAI_IMAGE_PATH), use_column_width=True)

        # Generate and display recommendations
        st.subheader("Optimization Strategies")
        with st.spinner("Generating AI-powered recommendations..."):
            prompt = f"""Generate urban heat mitigation strategies for {inputs.get('City', 'an urban area')} considering:
            - Current temperature: {prediction:.1f}°C ({'exceeds' if prediction > 38 else 'within'} thresholds)
            - Key issues: {', '.join([k for k,v in inputs.items() if v > HEAT_THRESHOLDS.get(k, float('inf'))])}
            Response format:
            🏗️ 1-2 urban design improvements
            🌳 1-2 nature-based solutions
            🔬 1 technical intervention
            🏙️ 1 policy recommendation"""
            
            suggestions = generate_suggestions(prompt)
            
            if suggestions.startswith("**Error"):
                st.error(suggestions)
            else:
                st.markdown(f"### Recommended Interventions\n{suggestions}", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Analysis Failed: {str(e)}")
        st.code(traceback.format_exc())
