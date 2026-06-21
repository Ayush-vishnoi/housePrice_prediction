import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/lite_rf_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🏠 House Price Prediction")

st.markdown("""
### Machine Learning House Price Predictor

Predict residential property prices using a Random Forest Regressor trained on housing data.

**Model Performance**
- Full Random Forest: R² = 0.918
- Lite Random Forest: R² = 0.904
- Features Used: 10
""")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Property Details")

overall_qual = st.sidebar.slider(
    "Overall Quality", 1, 10, 5
)

gr_liv_area = st.sidebar.number_input(
    "Ground Living Area (sq ft)",
    min_value=300,
    max_value=6000,
    value=1500
)

garage_cars = st.sidebar.slider(
    "Garage Capacity",
    min_value=0,
    max_value=5,
    value=2
)

garage_area = st.sidebar.number_input(
    "Garage Area",
    min_value=0,
    max_value=2000,
    value=500
)

first_flr = st.sidebar.number_input(
    "1st Floor Area",
    min_value=300,
    max_value=4000,
    value=1200
)

total_bsmt = st.sidebar.number_input(
    "Total Basement Area",
    min_value=0,
    max_value=4000,
    value=1000
)

year_built = st.sidebar.number_input(
    "Year Built",
    min_value=1900,
    max_value=2025,
    value=2000
)

lot_area = st.sidebar.number_input(
    "Lot Area",
    min_value=1000,
    max_value=100000,
    value=8000
)

overall_cond = st.sidebar.slider(
    "Overall Condition",
    min_value=1,
    max_value=10,
    value=5
)

year_remod = st.sidebar.number_input(
    "Year Remodeled",
    min_value=1950,
    max_value=2025,
    value=2000
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "Overall Qual": [overall_qual],
        "Gr Liv Area": [gr_liv_area],
        "Garage Cars": [garage_cars],
        "Garage Area": [garage_area],
        "1st Flr SF": [first_flr],
        "Total Bsmt SF": [total_bsmt],
        "Year Built": [year_built],
        "Lot Area": [lot_area],
        "Overall Cond": [overall_cond],
        "Year Remod/Add": [year_remod]
    })

    prediction = model.predict(input_data)[0]

    st.success("Prediction Complete ✅")

    st.metric(
        label="Estimated House Price",
        value=f"${prediction:,.0f}"
    )

# -----------------------------
# Model Comparison
# -----------------------------
st.markdown("---")

st.subheader("Model Comparison")

metrics = pd.DataFrame({
    "Model": [
        "Full Random Forest",
        "Lite Random Forest"
    ],
    "R² Score": [
        0.918,
        0.904
    ]
})

st.dataframe(metrics, use_container_width=True)

# -----------------------------
# Project Insights
# -----------------------------
st.markdown("---")

st.subheader("Key Insights")

st.write("""
- Overall Quality is the most important predictor of house prices.
- Living Area strongly influences property value.
- Garage capacity and garage size contribute significantly.
- Basement area has a noticeable impact on pricing.
- The Lite Model uses only 10 features while retaining most of the predictive performance.
""")

# -----------------------------
# Plots
# -----------------------------
st.markdown("---")

if os.path.exists("plots/feature_importance.png"):
    st.subheader("Feature Importance")
    st.image(
        "plots/feature_importance.png",
        use_container_width=True
    )

if os.path.exists("plots/residual_plot.png"):
    st.subheader("Residual Analysis")
    st.image(
        "plots/residual_plot.png",
        use_container_width=True
    )