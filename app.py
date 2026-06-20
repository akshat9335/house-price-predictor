import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ─── Header ────────────────────────────────────────────────────
st.title("🏠 House Price Prediction System")
st.markdown("**AIML Summer Internship 2026 | MNNIT Allahabad**")
st.markdown("---")

# ─── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model   = joblib.load('best_model.pkl')
    scaler  = joblib.load('scaler.pkl')
    le      = joblib.load('label_encoder.pkl')
    return model, scaler, le

try:
    model, scaler, le = load_model()
    st.success("✅ Model loaded successfully!")
except:
    st.warning("⚠️ Model files not found. Please run the notebook first to generate best_model.pkl")
    st.stop()

# ─── Sidebar Inputs ────────────────────────────────────────────
st.sidebar.header("🏡 Enter House Details")

bedrooms    = st.sidebar.slider("Bedrooms",        1, 9, 3)
bathrooms   = st.sidebar.slider("Bathrooms",       1.0, 6.0, 2.0, step=0.25)
sqft_living = st.sidebar.number_input("Living Area (sqft)", 300, 10000, 1800)
sqft_lot    = st.sidebar.number_input("Lot Size (sqft)",    500, 100000, 7000)
floors      = st.sidebar.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0])
waterfront  = st.sidebar.selectbox("Waterfront View", [0, 1], format_func=lambda x: "Yes" if x else "No")
view        = st.sidebar.slider("View Rating (0-4)", 0, 4, 0)
condition   = st.sidebar.slider("Condition (1-5)",   1, 5, 3)
sqft_above  = st.sidebar.number_input("Sqft Above Ground", 300, 9000, 1500)
sqft_basement = st.sidebar.number_input("Sqft Basement",   0, 4000, 300)
yr_built    = st.sidebar.number_input("Year Built",  1900, 2024, 1990)
was_renovated = st.sidebar.selectbox("Renovated?", [0, 1], format_func=lambda x: "Yes" if x else "No")

city_list = ['Seattle', 'Bellevue', 'Redmond', 'Kirkland', 'Shoreline',
             'Renton', 'Kent', 'Sammamish', 'Auburn', 'Burien']
city = st.sidebar.selectbox("City", city_list)

# ─── Feature Engineering (same as notebook) ────────────────────
house_age   = 2026 - yr_built
total_rooms = bedrooms + bathrooms

try:
    city_encoded = le.transform([city])[0]
except:
    city_encoded = 0

# ─── Input Summary ─────────────────────────────────────────────
st.subheader("📋 Your House Details")
col1, col2, col3 = st.columns(3)
col1.metric("🛏️ Bedrooms",  bedrooms)
col2.metric("🚿 Bathrooms", bathrooms)
col3.metric("📐 Living Area", f"{sqft_living:,} sqft")

col4, col5, col6 = st.columns(3)
col4.metric("🏙️ City",      city)
col5.metric("🏗️ Built",     yr_built)
col6.metric("⭐ Condition",  f"{condition}/5")

st.markdown("---")

# ─── Predict Button ────────────────────────────────────────────
if st.button("🔮 Predict House Price", use_container_width=True, type="primary"):

    input_data = np.array([[
        bedrooms, bathrooms, sqft_living, sqft_lot, floors,
        waterfront, view, condition, sqft_above, sqft_basement,
        city_encoded, house_age, was_renovated, total_rooms
    ]])

    input_df = pd.DataFrame(input_data, columns=[
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
        'city', 'house_age', 'was_renovated', 'total_rooms'
    ])

    predicted_price = model.predict(input_df)[0]

    st.markdown("## 💰 Predicted House Price")
    st.markdown(
        f"<h1 style='text-align:center; color:#2e7d32; font-size:3rem;'>"
        f"${predicted_price:,.0f}</h1>",
        unsafe_allow_html=True
    )

    # Price range (±10%)
    low  = predicted_price * 0.90
    high = predicted_price * 1.10
    st.info(f"📊 Estimated Price Range: **${low:,.0f}** — **${high:,.0f}**")

    # Price category
    if predicted_price < 300000:
        st.success("🟢 Category: Budget-Friendly Home")
    elif predicted_price < 600000:
        st.warning("🟡 Category: Mid-Range Home")
    elif predicted_price < 1000000:
        st.warning("🟠 Category: Premium Home")
    else:
        st.error("🔴 Category: Luxury Home")

# ─── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.caption("AIML Summer Internship 2026 | IIHMF, MNNIT Allahabad | House Price Prediction System")
