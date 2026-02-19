# User Interface
import streamlit as st

# Machine Learning Model Persistence
import joblib


@st.cache_resource(show_spinner="Loading classifier model...")
def load_model():
    model = joblib.load('random_forest_model.sav')
    return model

# Load once, reuse everywhere
MODEL = load_model()

def predict(data):
    return MODEL.predict(data)