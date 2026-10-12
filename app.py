import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Configure page styling
st.set_page_config(
    page_title="University Cutoff Predictor",
    page_icon="⌒",
    layout="centered"
)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('improved_model.pkl')
    maps = joblib.load('encoding_maps.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, maps, scaler

try:
    model, maps, scaler = load_artifacts()
    artifacts_loaded = True
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    artifacts_loaded = False

# Header
st.title("⌒ University Cutoff Predictor")
st.write("Predict course cutoff figures based on university, category, seat type, and year using our optimized Machine Learning model with balanced feature scaling.")

if artifacts_loaded:
    st.markdown("--- ")
    st.subheader("Configure Predictor Inputs")
    
    # Retrieve unique options for selectboxes
    unis = maps['unique_values']['university_id']
    cats = maps['unique_values']['category']
    seats = maps['unique_values']['seat_type']
    
    # Inputs
    year = st.slider("Admission Year", min_value=2024, max_value=2030, value=2025, step=1)
    selected_uni = st.selectbox("University ID", options=unis)
    selected_cat = st.selectbox("Category", options=cats)
    selected_seat = st.selectbox("Seat Type", options=seats)
    
    # Map features using the encoded dictionaries
    uni_encoded = maps['university_id']['mappings'].get(selected_uni, maps['university_id']['global_mean'])
    cat_encoded = maps['category']['mappings'].get(selected_cat, maps['category']['global_mean'])
    seat_encoded = maps['seat_type']['mappings'].get(selected_seat, maps['seat_type']['global_mean'])
    
    # Prepare features for prediction
    # Feature order must match training: ['year', 'university_id_encoded', 'category_encoded', 'seat_type_encoded']
    input_data = pd.DataFrame([[
        year,
        uni_encoded,
        cat_encoded,
        seat_encoded
    ]], columns=['year', 'university_id_encoded', 'category_encoded', 'seat_type_encoded'])
    
    # Transform inputs via the fitted scaler to fix scale-mismatch bugs
    input_scaled = scaler.transform(input_data)
    
    # Predict button
    if st.button("Predict Cutoff Score", type="primary"):
        with st.spinner("Running prediction model..."):
            prediction = model.predict(input_scaled)[0]
            
        st.success(f"### Predicted Cutoff Figure: **{prediction:.2f}**")
        
        # Add a helpful expander detailing input conversions
        with st.expander("ℹ️ Technical details on prediction features"):
            st.write("The raw categories were mapped and scaled as follows:")
            st.json({
                "Year (Raw)": year,
                "University (Encoded)": round(uni_encoded, 2),
                "Category (Encoded)": round(cat_encoded, 2),
                "Seat Type (Encoded)": round(seat_encoded, 2)
            })
else:
    st.warning("⚠️ Ensure 'improved_model.pkl', 'scaler.pkl', and 'encoding_maps.pkl' are saved in the same directory as 'app.py' before launching.")