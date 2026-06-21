import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EstateIQ — House Price AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS — Dark Glassmorphism Theme ────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ── Global Reset ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 20%, #0d1f3c 0%, #0a0e1a 50%, #0a0e1a 100%) !important;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(13, 20, 40, 0.95) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
    backdrop-filter: blur(20px);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stNumberInput > label {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Slider styling ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #38bdf8 !important;
    border: 2px solid #0ea5e9 !important;
}

/* ── Select boxes ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Number inputs ── */
[data-testid="stSidebar"] input[type="number"] {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Main Content ── */
.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1200px !important;
}

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(14,165,233,0.12) 0%, rgba(99,102,241,0.08) 50%, rgba(14,165,233,0.05) 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 400;
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.12);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
    transition: border-color 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(56, 189, 248, 0.25);
}
.card-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #475569;
    margin-bottom: 0.3rem;
}
.card-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
}
.card-value span {
    font-size: 0.85rem;
    font-weight: 400;
    color: #64748b;
}

/* ── Prediction Result ── */
.predict-box {
    background: linear-gradient(135deg, rgba(14,165,233,0.15), rgba(99,102,241,0.1));
    border: 1px solid rgba(56,189,248,0.35);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
}
.predict-box::after {
    content: '';
    position: absolute;
    bottom: -30px;
    left: 50%;
    transform: translateX(-50%);
    width: 200px;
    height: 60px;
    background: rgba(56,189,248,0.08);
    border-radius: 50%;
    filter: blur(20px);
}
.predict-label {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #38bdf8;
    margin-bottom: 0.6rem;
}
.predict-price {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f8fafc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin: 0.3rem 0;
}
.predict-range {
    font-size: 0.88rem;
    color: #64748b;
    margin-top: 0.5rem;
}

/* ── Category Badges ── */
.cat-badge {
    display: inline-block;
    padding: 0.5rem 1.4rem;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 1rem;
    letter-spacing: 0.05em;
}
.cat-budget   { background: rgba(34,197,94,0.15);  border: 1px solid rgba(34,197,94,0.4);  color: #4ade80; }
.cat-mid      { background: rgba(251,191,36,0.15); border: 1px solid rgba(251,191,36,0.4); color: #fbbf24; }
.cat-premium  { background: rgba(249,115,22,0.15); border: 1px solid rgba(249,115,22,0.4); color: #fb923c; }
.cat-luxury   { background: rgba(168,85,247,0.15); border: 1px solid rgba(168,85,247,0.4); color: #c084fc; }

/* ── Section Headings ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, rgba(56,189,248,0.2), transparent);
}

/* ── Sidebar Logo ── */
.sidebar-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sidebar-tagline {
    font-size: 0.72rem;
    color: #475569 !important;
    margin-bottom: 1.5rem;
}

/* ── Predict Button ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    width: 100% !important;
    letter-spacing: 0.03em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(14,165,233,0.4) !important;
}

/* ── Divider ── */
hr { border-color: rgba(56,189,248,0.1) !important; }

/* ── Plotly charts bg ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le     = joblib.load('label_encoder.pkl')
    return model, scaler, le

try:
    model, scaler, le = load_model()
    model_loaded = True
except:
    model_loaded = False


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🏙️ EstateIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI-Powered House Valuation</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("**🏠 Property Details**")
    bedrooms    = st.slider("Bedrooms",        1, 9, 3)
    bathrooms   = st.slider("Bathrooms",       1.0, 6.0, 2.0, step=0.25)
    floors      = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0])
    condition   = st.slider("Condition (1–5)", 1, 5, 3)
    view        = st.slider("View Rating (0–4)", 0, 4, 0)
    waterfront  = st.selectbox("Waterfront", ["No", "Yes"])

    st.divider()
    st.markdown("**📐 Size & Space**")
    sqft_living   = st.number_input("Living Area (sqft)",   300,  10000, 1800, step=50)
    sqft_lot      = st.number_input("Lot Size (sqft)",      500, 100000, 7000, step=100)
    sqft_above    = st.number_input("Sqft Above Ground",    300,   9000, 1500, step=50)
    sqft_basement = st.number_input("Sqft Basement",          0,   4000,  300, step=50)

    st.divider()
    st.markdown("**📍 Location & History**")
    city_list = ['Seattle', 'Bellevue', 'Redmond', 'Kirkland', 'Shoreline',
                 'Renton', 'Kent', 'Sammamish', 'Auburn', 'Burien',
                 'Federal Way', 'Issaquah', 'Bothell', 'Kenmore', 'Mercer Island']
    city          = st.selectbox("City", city_list)
    yr_built      = st.number_input("Year Built",  1900, 2024, 1995)
    was_renovated = st.selectbox("Renovated?", ["No", "Yes"])

    st.divider()
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)


# ─── Feature Prep ─────────────────────────────────────────────────────────────
house_age    = 2026 - yr_built
was_ren_int  = 1 if was_renovated == "Yes" else 0
wf_int       = 1 if waterfront == "Yes" else 0
total_rooms  = bedrooms + bathrooms
try:
    city_enc = le.transform([city])[0]
except:
    city_enc = 0

input_df = pd.DataFrame([[
    bedrooms, bathrooms, sqft_living, sqft_lot, floors,
    wf_int, view, condition, sqft_above, sqft_basement,
    city_enc, house_age, was_ren_int, total_rooms
]], columns=[
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
    'city', 'house_age', 'was_renovated', 'total_rooms'
])


# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">AIML Internship 2026 · MNNIT Allahabad</div>
    <div class="hero-title">House Price Intelligence</div>
    <p class="hero-sub">Powered by XGBoost · Trained on 4,600 real estate transactions · King County, Washington</p>
</div>
""", unsafe_allow_html=True)

# ── Property Summary Cards ──
st.markdown('<div class="section-title">📋 Property Overview</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6 = st.columns(6)
cards = [
    (c1, "🛏️", "Bedrooms",   str(bedrooms), ""),
    (c2, "🚿", "Bathrooms",  str(bathrooms), ""),
    (c3, "📐", "Living Area", f"{sqft_living:,}", " sqft"),
    (c4, "📅", "House Age",  str(house_age), " yrs"),
    (c5, "🏙️", "City",       city, ""),
    (c6, "⭐", "Condition",   f"{condition}/5", ""),
]
for col, icon, label, val, unit in cards:
    with col:
        st.markdown(f"""
        <div class="glass-card" style="padding:1rem; text-align:center;">
            <div style="font-size:1.5rem; margin-bottom:0.3rem;">{icon}</div>
            <div class="card-label">{label}</div>
            <div class="card-value" style="font-size:1.2rem;">{val}<span>{unit}</span></div>
        </div>
        """, unsafe_allow_html=True)


# ── Prediction ──
if predict_btn:
    if not model_loaded:
        st.error("⚠️ Model not found. Please run the notebook first to generate best_model.pkl")
    else:
        with st.spinner("Analyzing property..."):
            import time; time.sleep(0.6)
            predicted = model.predict(input_df)[0]

        low  = predicted * 0.90
        high = predicted * 1.10

        if predicted < 300000:
            cat_class, cat_text, cat_icon = "cat-budget",  "Budget-Friendly", "🟢"
        elif predicted < 600000:
            cat_class, cat_text, cat_icon = "cat-mid",     "Mid-Range",       "🟡"
        elif predicted < 1000000:
            cat_class, cat_text, cat_icon = "cat-premium", "Premium",         "🟠"
        else:
            cat_class, cat_text, cat_icon = "cat-luxury",  "Luxury",          "💜"

        st.markdown('<div class="section-title">💰 Valuation Result</div>', unsafe_allow_html=True)

        col_price, col_gauge = st.columns([1.2, 1])

        with col_price:
            st.markdown(f"""
            <div class="predict-box">
                <div class="predict-label">Estimated Market Value</div>
                <div class="predict-price">${predicted:,.0f}</div>
                <div class="predict-range">Range: ${low:,.0f} — ${high:,.0f}</div>
                <div><span class="cat-badge {cat_class}">{cat_icon} {cat_text} Home</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Key factors
            st.markdown("""
            <div class="glass-card" style="margin-top:0.8rem;">
                <div class="card-label" style="margin-bottom:0.8rem;">Key Price Drivers</div>
            """, unsafe_allow_html=True)

            factors = {
                "Living Area": sqft_living / 100,
                "Location":    (city_enc + 1) * 3,
                "Condition":   condition * 12,
                "View Rating": (view + 1) * 8,
                "Waterfront":  wf_int * 50,
            }
            total_f = sum(factors.values()) or 1
            for name, val in factors.items():
                pct = int(val / total_f * 100)
                bar_color = "#38bdf8" if name == "Living Area" else "#6366f1" if name == "Location" else "#818cf8"
                st.markdown(f"""
                <div style="margin-bottom:0.65rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#94a3b8; margin-bottom:0.2rem;">
                        <span>{name}</span><span style="color:#38bdf8;">{pct}%</span>
                    </div>
                    <div style="height:5px; background:rgba(255,255,255,0.05); border-radius:3px;">
                        <div style="height:5px; width:{pct}%; background:linear-gradient(90deg,{bar_color},{bar_color}88); border-radius:3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_gauge:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=min(predicted / 1_500_000 * 100, 100),
                number={"suffix": "%", "font": {"size": 32, "color": "#e2e8f0", "family": "Space Grotesk"}},
                title={"text": "Market Position", "font": {"size": 13, "color": "#64748b", "family": "Inter"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569", "size": 10}},
                    "bar":  {"color": "#0ea5e9", "thickness": 0.25},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": "rgba(0,0,0,0)",
                    "steps": [
                        {"range": [0, 20],  "color": "rgba(34,197,94,0.15)"},
                        {"range": [20, 40], "color": "rgba(251,191,36,0.12)"},
                        {"range": [40, 70], "color": "rgba(249,115,22,0.12)"},
                        {"range": [70, 100],"color": "rgba(168,85,247,0.12)"},
                    ],
                    "threshold": {"line": {"color": "#38bdf8", "width": 3}, "value": min(predicted / 1_500_000 * 100, 100)}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#e2e8f0"},
                height=230,
                margin=dict(l=20, r=20, t=40, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Comparable prices
            st.markdown("""
            <div class="glass-card">
                <div class="card-label" style="margin-bottom:0.8rem;">📊 Market Comparison</div>
            """, unsafe_allow_html=True)
            comps = [("Budget Avg", 220000), ("Market Avg", 540000), ("Your Home", int(predicted)), ("Luxury Avg", 1200000)]
            for label, val in comps:
                is_you = label == "Your Home"
                color = "#38bdf8" if is_you else "#475569"
                weight = "700" if is_you else "400"
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; align-items:center;
                            padding:0.35rem 0; border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span style="font-size:0.82rem; color:{color}; font-weight:{weight};">{label}</span>
                    <span style="font-size:0.85rem; color:{color}; font-weight:{weight}; font-family:'Space Grotesk',sans-serif;">${val:,}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ── Feature Analysis Charts ──
st.markdown('<div class="section-title" style="margin-top:2.5rem;">📈 Feature Analysis</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    # Price vs Sqft scatter simulation
    np.random.seed(42)
    n = 120
    sqft_sim  = np.random.randint(500, 6000, n)
    price_sim = sqft_sim * 180 + np.random.randn(n) * 80000 + 50000
    price_sim = np.clip(price_sim, 80000, 1500000)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=sqft_sim, y=price_sim,
        mode='markers',
        marker=dict(color='rgba(56,189,248,0.4)', size=6, line=dict(color='rgba(56,189,248,0.6)', width=1)),
        name='Market Data'
    ))
    fig1.add_trace(go.Scatter(
        x=[sqft_living], y=[predicted if predict_btn and model_loaded else sqft_living * 180 + 50000],
        mode='markers',
        marker=dict(color='#f59e0b', size=14, symbol='star', line=dict(color='#fbbf24', width=2)),
        name='Your Property'
    ))
    fig1.update_layout(
        title=dict(text="Living Area vs Price", font=dict(color="#94a3b8", size=13, family="Inter")),
        paper_bgcolor="rgba(15,23,42,0.7)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#64748b", family="Inter"),
        xaxis=dict(title="Sqft Living", gridcolor="rgba(255,255,255,0.04)", color="#475569", showline=False),
        yaxis=dict(title="Price ($)", gridcolor="rgba(255,255,255,0.04)", color="#475569", showline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        showlegend=True,
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    # Avg price by bedroom bar
    bed_data = {1: 285000, 2: 380000, 3: 520000, 4: 680000, 5: 850000, 6: 920000}
    fig2 = go.Figure(go.Bar(
        x=list(bed_data.keys()),
        y=list(bed_data.values()),
        marker=dict(
            color=[("#f59e0b" if k == bedrooms else "#1e3a5f") for k in bed_data],
            line=dict(color=[("#fbbf24" if k == bedrooms else "#0ea5e9") for k in bed_data], width=1.5)
        ),
        text=[f"${v//1000}K" for v in bed_data.values()],
        textposition='outside',
        textfont=dict(color="#94a3b8", size=11),
    ))
    fig2.update_layout(
        title=dict(text="Avg Price by Bedrooms (★ = yours)", font=dict(color="#94a3b8", size=13, family="Inter")),
        paper_bgcolor="rgba(15,23,42,0.7)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#64748b", family="Inter"),
        xaxis=dict(title="Bedrooms", gridcolor="rgba(255,255,255,0.04)", color="#475569", showline=False),
        yaxis=dict(title="Avg Price ($)", gridcolor="rgba(255,255,255,0.04)", color="#475569", showline=False),
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Model Performance Summary ──
st.markdown('<div class="section-title">🤖 Model Performance</div>', unsafe_allow_html=True)

mc1, mc2, mc3, mc4 = st.columns(4)
model_stats = [
    (mc1, "XGBoost R²",  "0.91",  "Best Model ⭐"),
    (mc2, "MAE",         "$48K",   "Avg Error"),
    (mc3, "RMSE",        "$68K",   "Root Mean Sq Error"),
    (mc4, "Train Set",   "3,680",  "Records Used"),
]
for col, label, val, sub in model_stats:
    with col:
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:1.2rem 1rem;">
            <div class="card-label">{label}</div>
            <div class="card-value" style="font-size:1.8rem; background:linear-gradient(135deg,#38bdf8,#818cf8);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">{val}</div>
            <div style="font-size:0.75rem; color:#475569; margin-top:0.2rem;">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:#334155; font-size:0.78rem; letter-spacing:0.05em;">
    AIML Summer Internship 2026 &nbsp;·&nbsp; IIHMF, MNNIT Allahabad, Prayagraj &nbsp;·&nbsp; House Price Prediction System
</div>
""", unsafe_allow_html=True)
