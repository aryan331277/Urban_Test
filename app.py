import streamlit as st
import joblib
import pandas as pd
from PIL import Image
import traceback
import requests
from typing import Dict, Any
import os

API_KEY = st.secrets.get("HUGGINGFACE_API_KEY")

if not API_KEY:
    st.error("🚨 Missing API key! Please add it to Streamlit Secrets.")
    st.stop()

API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"
MODEL_PATH = "trainedmodelfinal.pkl"
XAI_IMAGE_PATH = "feature importance.png"
HEAT_THRESHOLDS = {
    'critical_temp': 30.0,
    'green_cover_min': 25,
    'albedo_min': 0.4,
    'building_height_max': 35,
    'heat_stress_max': 4.0,
    'population_density_max': 15000
}

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    return joblib.load(MODEL_PATH)
API_KEY = os.getenv("HUGGINGFACE_API_KEY", st.secrets["HUGGINGFACE_API_KEY"])

def generate_suggestions(prompt: str) -> str:
    """Generate mitigation strategies using DeepSeek-R1"""
    
    API_KEY = st.secrets["HUGGINGFACE_API_KEY"]  # Get API key from Streamlit Secrets

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    
    try:
        response = requests.post("https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1", 
                                 headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, dict) and "generated_text" in result:
            return result["generated_text"]
        return "⚠️ Could not generate suggestions. Please try again."
    
    except requests.exceptions.Timeout:
        return "⚠️ API request timed out. Please try again."
    except requests.exceptions.RequestException as e:
        return f"⚠️ API error: {str(e)}"
        
def format_suggestions(text: str) -> str:
    """Format the generated suggestions with proper markdown"""
    sections = {
        "🏗️ Urban Design": [],
        "🌳 Nature Solutions": [],
        "🔬 Technology": [],
        "📜 Policies": []
    }
    
    current_section = None
    for line in text.split('\n'):
        line = line.strip()
        if line.startswith('🏗️'):
            current_section = "🏗️ Urban Design"
        elif line.startswith('🌳'):
            current_section = "🌳 Nature Solutions"
        elif line.startswith('🔬'):
            current_section = "🔬 Technology"
        elif line.startswith('📜'):
            current_section = "📜 Policies"
        elif current_section and line:
            sections[current_section].append(line)
    
    formatted = []
    for section, items in sections.items():
        if items:
            formatted.append(f"### {section}")
            formatted.extend([f"- {item}" for item in items])
    return '\n\n'.join(formatted)

def get_input_parameters() -> Dict[str, Any]:
    """Collect all user inputs from the sidebar"""
    st.sidebar.header("🌍 City Selection")
    city = st.sidebar.selectbox("Select City", ["Delhi", "Mumbai", "Hyderabad"])
    
    st.sidebar.header("📊 Urban Parameters")
    return {
        'Latitude': st.sidebar.number_input("Latitude", 19.0, 19.2, 19.0760, 0.0001),
        'Longitude': st.sidebar.number_input("Longitude", 72.8, 73.0, 72.8777, 0.0001),
        'Population Density': st.sidebar.number_input("Population Density (people/km²)", 1000, 50000, 20000),
        'Albedo': st.sidebar.slider("Albedo", 0.0, 1.0, 0.3, 0.05),
        'Green Cover Percentage': st.sidebar.slider("Green Cover (%)", 0, 100, 25),
        'Relative Humidity': st.sidebar.slider("Humidity (%)", 0, 100, 60),
        'Wind Speed': st.sidebar.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1),
        'Building Height': st.sidebar.slider("Building Height (m)", 5, 150, 30),
        'Road Density': st.sidebar.slider("Road Density (km/km²)", 0.0, 20.0, 5.0, 0.1),
        'Proximity to Water Body': st.sidebar.slider("Water Proximity (m)", 0, 5000, 1000),
        'Solar Radiation': st.sidebar.slider("Solar Radiation (W/m²)", 0, 1000, 500),
        'Nighttime Surface Temperature': st.sidebar.slider("Night Temp (°C)", 15.0, 40.0, 25.0, 0.1),
        'Distance from Previous Point': st.sidebar.number_input("Distance from Previous Point (m)", 0, 5000, 100),
        'Heat Stress Index': st.sidebar.slider("Heat Stress Index", 0.0, 10.0, 3.5, 0.1),
        'Urban Vegetation Index': st.sidebar.slider("Vegetation Index", 0.0, 1.0, 0.5, 0.01),
        'Carbon Emission Levels': st.sidebar.number_input("CO₂ Levels (ppm)", 300, 1000, 400),
        'Surface Material': st.sidebar.selectbox("Surface Material", ["Concrete", "Asphalt", "Grass", "Water", "Mixed"]),
        'Land Cover Type': st.sidebar.selectbox("Land Cover Type", ["Urban", "Residential", "Water", "Vegetation", "Industrial", "Bare Soil"]),
        'Cooling Measures Present': st.sidebar.selectbox("Cooling Measures Present", ["None", "Green Roofs", "Reflective Paint", "Rooftop Garden", "Shaded Streets", "Water Features"])
    }

# Streamlit UI Configuration
st.set_page_config(page_title="Urban Heat Analyst", layout="wide", page_icon="🌡️")
st.title("🌇 Urban Heat Island Effect Analysis")

# Main Application Flow
inputs = get_input_parameters()

if st.sidebar.button("🚀 Analyze Urban Heat", use_container_width=True):
    try:
        model = load_model()
        
        # Create DataFrame with exact feature names from training
        input_df = pd.DataFrame([inputs], columns=model.feature_names_in_)
        
        # Check for feature mismatch
        missing_features = set(model.feature_names_in_) - set(inputs.keys())
        extra_features = set(inputs.keys()) - set(model.feature_names_in_)
        
        if missing_features:
            st.error(f"🚨 Missing features: {', '.join(missing_features)}")
            st.stop()
        if extra_features:
            st.warning(f"⚠️ Extra features ignored: {', '.join(extra_features)}")

        # Make prediction
        prediction = model.predict(input_df)[0]
        threshold_status = "❌ Exceeds" if prediction > HEAT_THRESHOLDS['critical_temp'] else "✅ Within"
        
        # Display results
        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.subheader("📈 Prediction Results")
            st.metric("Surface Temperature", 
                     f"{prediction:.1f}°C", 
                     delta=f"{threshold_status} safe threshold ({HEAT_THRESHOLDS['critical_temp']}°C)")
            
            st.subheader("⚠️ Threshold Violations")
            violations = []
            for param, value in inputs.items():
                if param in HEAT_THRESHOLDS:
                    threshold = HEAT_THRESHOLDS[param]
                    if value > threshold:
                        violations.append(f"{param.replace('_', ' ').title()}: {value} > {threshold}")
            
            if violations:
                st.write("\n".join([f"- {v}" for v in violations]))
            else:
                st.success("All monitored parameters within safe thresholds")

        with col2:
            st.subheader("📊 Feature Importance")
            st.image(Image.open(XAI_IMAGE_PATH), use_column_width=True)

        # Generate suggestions
        st.subheader("💡 Mitigation Strategies")
        with st.spinner("🔍 Analyzing urban parameters and generating recommendations..."):
            prompt = f"""
            Generate urban heat mitigation strategies considering these parameters:
            City: {inputs.get('City', 'Urban Area')}
            Predicted Temperature: {prediction:.1f}°C ({threshold_status} threshold)
            
            Key Parameters:
            - Green Cover: {inputs['Green Cover Percentage']}% (Threshold: {HEAT_THRESHOLDS['green_cover_min']}%)
            - Albedo: {inputs['Albedo']} (Threshold: {HEAT_THRESHOLDS['albedo_min']})
            - Building Height: {inputs['Building Height']}m (Threshold: {HEAT_THRESHOLDS['building_height_max']}m)
            - Heat Stress Index: {inputs['Heat Stress Index']} (Threshold: {HEAT_THRESHOLDS['heat_stress_max']})
            - Population Density: {inputs['Population Density']} (Threshold: {HEAT_THRESHOLDS['population_density_max']})
            
            Additional Context:
            - Surface Material: {inputs['Surface Material']}
            - Cooling Measures: {inputs['Cooling Measures Present']}
            - Land Cover Type: {inputs['Land Cover Type']}
            - Humidity: {inputs['Relative Humidity']}%
            - Wind Speed: {inputs['Wind Speed']} m/s
            
            Provide specific recommendations in these categories:
            🏗️ Urban design improvements
            🌳 Nature-based solutions
            🔬 Technological interventions
            📜 Policy recommendations
            
            Focus on parameters exceeding thresholds and leverage the additional context.
            """
            
            suggestions = generate_suggestions(prompt)
            st.markdown(suggestions, unsafe_allow_html=True)

    except Exception as e:
        st.error("🚨 Analysis failed - Please check input parameters and try again")
        st.error(f"Error details: {str(e)}")
        st.code(traceback.format_exc())
