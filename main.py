
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="PrEP Adherence Prediction",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load the trained model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "prep_adherence_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# -----------------------------
# Application interface
# -----------------------------
st.title("PrEP Adherence Prediction System")

st.write(
    "A machine learning application for estimating "
    "PrEP adherence using relevant patient characteristics."
)

st.info(
    "This tool is for research and decision support. "
    "Predictions should not replace clinical assessment."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Add your trained model "
        "as prep_adherence_model.pkl to this project folder."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

# -----------------------------
# Collect patient information
# -----------------------------
st.subheader("Patient Information")

with st.form("adherence_form"):

    age = st.number_input(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=25
    )

    prep_type = st.selectbox(
        "PrEP Type",
        ["Daily oral", "Event-driven"]
    )

    months = st.number_input(
        "Months on PrEP",
        min_value=0,
        max_value=600,
        value=6
    )

    condom_use = st.selectbox(
        "Condom Use",
        ["Yes", "No", "Unknown"]
    )

    sti_status = st.selectbox(
        "STI Status",
        ["Yes", "No", "Unknown"]
    )

    menopause = st.selectbox(
        "Menopause Status",
        ["Yes", "No", "Not applicable", "Unknown"]
    )

    pregnancy = st.selectbox(
        "Pregnancy Status",
        ["Yes", "No", "Not applicable", "Unknown"]
    )

    breastfeeding = st.selectbox(
        "Breastfeeding Status",
        ["Yes", "No", "Not applicable", "Unknown"]
    )

    fp_status = st.selectbox(
        "Family Planning Status",
        ["Yes", "No", "Unknown"]
    )

    chronic_conditions = st.selectbox(
        "Chronic Conditions",
        ["Yes", "No", "Unknown"]
    )

    adverse_reactions = st.selectbox(
        "Reported Adverse Reactions",
        ["Yes", "No", "Unknown"]
    )

    adherence_counselled = st.selectbox(
        "Adherence Counselling Received",
        ["Yes", "No", "Unknown"]
    )

    submitted = st.form_submit_button(
        "Predict PrEP Adherence"
    )

# -----------------------------
# Make a prediction
# -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "prep_type": prep_type,
        "months": months,
        "condom_use": condom_use,
        "sti_status": sti_status,
        "menopause": menopause,
        "pregnancy": pregnancy,
        "breastfeeding": breastfeeding,
        "fp_status": fp_status,
        "chronic_conditions": chronic_conditions,
        "adverse_reactions": adverse_reactions,
        "adherence_counselled": adherence_counselled
    }])

    try:
        # The saved model must include the same preprocessing
        # and feature transformations used during training.
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        # Update these labels to match your actual target encoding.
        if prediction == 1:
            st.success("Predicted class: Adherent")
        elif prediction == 0:
            st.warning("Predicted class: Not adherent")
        else:
            st.write(f"Predicted class: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.write("Model probability estimates:")

            for class_label, probability in zip(
                model.classes_, probabilities
            ):
                st.write(
                    f"Class {class_label}: "
                    f"{probability * 100:.2f}%"
                )

        st.caption(
            "This is a model-generated estimate, not a confirmed "
            "measure of a patient's actual adherence."
        )

    except Exception as e:
        st.error(
            "Prediction failed. Check that the model's input "
            "features and preprocessing match this application."
        )
        st.exception(e)
