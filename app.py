import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Financial Profiler | MNIT Jaipur",
    page_icon="💰",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [class*="css"] { font-size: 18px !important; }
    h1 { font-size: 2.4rem !important; font-weight: 800; }
    h2 { font-size: 1.8rem !important; font-weight: 700; }
    h3 { font-size: 1.4rem !important; }
    .stButton>button {
        background-color: #00c04b !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        padding: 0.6rem 2rem !important;
        border-radius: 8px !important;
        border: none !important;
    }
    .result-card {
        background: #f0f9f4;
        border-left: 6px solid #00c04b;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .risk-high { border-left-color: #e74c3c !important; background: #fdf0ef !important; }
    .risk-medium { border-left-color: #f39c12 !important; background: #fef9ef !important; }
    .risk-low { border-left-color: #00c04b !important; background: #f0f9f4 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = {}
    try:
        import tensorflow as tf
        m["nn"] = tf.keras.models.load_model("models/spend_nn_model.h5")
        m["oe"] = joblib.load("models/ordinal_encoder.joblib")
        m["ohe"] = joblib.load("models/ohe_encoder.joblib")
        m["mms"] = joblib.load("models/mms_scaler.joblib")
        m["kp"] = joblib.load("models/kproto_model.joblib")
        m["kp_cat_idx"] = joblib.load("models/kproto_cat_indices.joblib")
        m["kp_cols"] = joblib.load("models/kproto_feature_columns.joblib")
        m["rf"] = joblib.load("models/rf_classifier.joblib")
        m["scaler_rf"] = joblib.load("models/scaler_rf.joblib")
        m["rf_cols"] = joblib.load("models/rf_feature_columns.joblib")
    except Exception as e:
        st.error(f"Model loading error: {e}")
    return m

models = load_models()

# ── Survey options ────────────────────────────────────────────────────────────
PLACE_OPTIONS = ["🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"]
IMPORTANCE_OPTIONS = ["Not important", "Slightly important", "Very important"]
TRACK_OPTIONS = [
    "I do not keep the track",
    "I check my bank balance occasionally.",
    "I review my history within payment apps (e.g., UPI, Paytm).",
    "I use a dedicated expense-tracking app or spreadsheet.",
]
GRAPH_OPTIONS = [
    "Uniform Daily Expenses",
    "Irregular and Random Spending",
    "Spend a lot once and then low spending for rest",
    "Steady Weekdays with High Weekends",
    "None",
]
JUSTIFY_OPTIONS = [
    "Emergencies (e.g., phone/laptop repair).",
    "A 50% discount on a brand I highly value.",
    "Social celebrations or parties.",
    "Skill development (workshops, certifications, technical kits).",
    "A planned trip with friends.",
]
BUDGET_OPTIONS = ["Food & Dining", "Travel", "Fashion", "Subscriptions (Netflix, Spotify, etc.)", "Fun & Entertainment"]

# ── App Header ────────────────────────────────────────────────────────────────
st.title("💰 Student Financial Behavior & Risk Profiling System")
st.markdown("**MNIT Jaipur** | Answer the survey below to get your financial intelligence report.")
st.divider()

# ── Survey Form ───────────────────────────────────────────────────────────────
st.header("📋 Financial Behavior Survey")

with st.form("survey_form"):
    st.subheader("Section 1: Background")
    place = st.radio("1. What best describes the place you grew up in?", PLACE_OPTIONS)

    st.subheader("Section 2: Spending Behavior")
    unplanned = st.slider("2. How often do you make purchases you hadn't planned for?", 1, 5, 3,
                          help="1 = Never, 5 = Very often")
    peer_influence = st.slider("3. How much do social events or peer pressure influence your spending?", 1, 5, 3,
                               help="1 = Not at all, 5 = Very much")
    finance_conf = st.slider("4. How confident are you in managing your personal finances?", 1, 5, 3,
                             help="1 = Not confident, 5 = Very confident")

    st.subheader("Section 3: Purchase Priorities")
    price_imp = st.select_slider("5. Importance of Price/Cost", IMPORTANCE_OPTIONS, value="Slightly important")
    brand_imp = st.select_slider("6. Importance of Brand Reputation", IMPORTANCE_OPTIONS, value="Slightly important")
    peer_imp = st.select_slider("7. Importance of Peer Recommendation", IMPORTANCE_OPTIONS, value="Slightly important")
    utility_imp = st.select_slider("8. Importance of Long-term Utility/Value", IMPORTANCE_OPTIONS, value="Very important")

    st.subheader("Section 4: Habits & Patterns")
    track = st.selectbox("9. How do you track your monthly expenditures?", TRACK_OPTIONS)
    graph = st.radio("10. What best describes your expected monthly expenditure pattern?", GRAPH_OPTIONS)

    st.subheader("Section 5: Spending Categories")
    budget_selected = st.multiselect("11. Where do you spend the majority of your budget? (select all that apply)", BUDGET_OPTIONS)
    justify_selected = st.multiselect("12. In which scenarios would you justify an unexpected ₹1,500+ expense?", JUSTIFY_OPTIONS)

    submitted = st.form_submit_button("🔍 Generate My Financial Report")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    st.divider()
    st.header("🧠 Your Financial Intelligence Report")

    # ── Helpers ──────────────────────────────────────────────────────────────
    place_clean = place.replace("🏙️ ", "").replace("🏢 ", "").replace("🏘️ ", "").replace("🌾 ", "").strip()
    budget_flags = {
        "Budget_FoodDining": 1 if "Food & Dining" in budget_selected else 0,
        "Budget_Travel": 1 if "Travel" in budget_selected else 0,
        "Budget_Fashion": 1 if "Fashion" in budget_selected else 0,
        "Budget_Subscriptions": 1 if "Subscriptions (Netflix, Spotify, etc.)" in budget_selected else 0,
        "Budget_Entertainment": 1 if "Fun & Entertainment" in budget_selected else 0,
    }

    # ── MODEL 1: Softmax NN → Monthly Spend Tier ─────────────────────────────
    with st.spinner("Running Neural Network..."):
        try:
            oe_input = models["oe"].transform([[place_clean, price_imp, brand_imp, peer_imp, utility_imp]])
            ohe_input = models["ohe"].transform([[track, graph if graph != "None" else "Uniform Daily Expenses"]])
            jue_input = np.array([[
                1 if "Emergencies" in " ".join(justify_selected) else 0,
                1 if "discount" in " ".join(justify_selected).lower() else 0,
                1 if "celebrations" in " ".join(justify_selected).lower() else 0,
                1 if "Skill development" in " ".join(justify_selected) else 0,
                1 if "planned trip" in " ".join(justify_selected).lower() else 0,
            ]])
            num_input = models["mms"].transform([[unplanned, peer_influence, finance_conf]])
            budget_input = np.array([[v for v in budget_flags.values()]])

            X_nn = np.hstack([oe_input, ohe_input, jue_input, num_input, budget_input])
            probs = models["nn"].predict(X_nn, verbose=0)[0]
            spend_tier = int(np.argmax(probs)) + 1
            expected_spend = float(np.dot(probs, np.arange(1, 11)))
            nn_ok = True
        except Exception as e:
            nn_ok = False
            spend_tier = 5
            st.warning(f"NN prediction error: {e}")

    # ── MODEL 2: K-Prototypes → Persona Cluster ───────────────────────────────
    with st.spinner("Running K-Prototypes clustering..."):
        try:
            track_map = {
                "I do not keep the track": 0,
                "I check my bank balance occasionally.": 1,
                "I review my history within payment apps (e.g., UPI, Paytm).": 2,
                "I use a dedicated expense-tracking app or spreadsheet.": 3,
            }
            graph_map = {
                "Uniform Daily Expenses": 0, "Steady Weekdays with High Weekends": 1,
                "Irregular and Random Spending": 2, "Spend a lot once and then low spending for rest": 3, "None": 1,
            }
            group = "Low Spenders" if spend_tier <= 3 else ("Medium Spenders" if spend_tier <= 6 else "High Spenders")

            # Build row matching kproto feature columns
            kp_row = {
                "Place_Grew_Up": place_clean,
                "Price_Importance": price_imp,
                "Brand_Importance": brand_imp,
                "Peer_Importance": peer_imp,
                "Utility_Importance": utility_imp,
                "Track_Expenditures": float(track_map.get(track, 1)),
                "Expenditure_Graph": graph if graph != "None" else "Uniform Daily Expenses",
                "Unplanned_Purchases": float(unplanned),
                "Peer_Influence": float(peer_influence),
                "Finance_Confidence": float(finance_conf),
                "Budget_FoodDining": "Yes" if budget_flags["Budget_FoodDining"] else "No",
                "Budget_Travel": "Yes" if budget_flags["Budget_Travel"] else "No",
                "Budget_Fashion": "Yes" if budget_flags["Budget_Fashion"] else "No",
                "Budget_Subscriptions": "Yes" if budget_flags["Budget_Subscriptions"] else "No",
                "Budget_Entertainment": "Yes" if budget_flags["Budget_Entertainment"] else "No",
                "Group": group,
            }
            kp_cols = models["kp_cols"]
            row_vals = []
            for col in kp_cols:
                val = kp_row.get(col, 0)
                row_vals.append(val)

            kp_array = np.array([row_vals], dtype=object)
            cluster_id = models["kp"].predict(kp_array, categorical=models["kp_cat_idx"])[0]
            persona_names = {0: "🟢 Cautious Saver", 1: "🟡 Balanced Spender", 2: "🔴 Impulsive Spender"}
            persona = persona_names.get(int(cluster_id), f"Cluster {cluster_id}")
            kp_ok = True
        except Exception as e:
            kp_ok = False
            persona = "N/A"
            st.warning(f"K-Prototypes error: {e}")

    # ── MODEL 3: Random Forest → Risk Score ───────────────────────────────────
    with st.spinner("Running Risk Scorer..."):
        try:
            rf_cols = models["rf_cols"]
            import pandas as pd
            place_dummies = {
                f"Place_Grew_Up_Big metro city": 1 if "Big metro" in place_clean else 0,
                f"Place_Grew_Up_Medium-sized city": 1 if "Medium" in place_clean else 0,
                f"Place_Grew_Up_Small town": 1 if "Small" in place_clean else 0,
                f"Place_Grew_Up_Rural area": 1 if "Rural" in place_clean else 0,
            }
            imp_map = {"Not important": 0, "Slightly important": 1, "Very important": 2}
            rf_row = {
                "Price_Importance": imp_map[price_imp],
                "Brand_Importance": imp_map[brand_imp],
                "Peer_Importance": imp_map[peer_imp],
                "Utility_Importance": imp_map[utility_imp],
                "Track_Expenditures": float(track_map.get(track, 1)),
                "Expenditure_Graph": float(graph_map.get(graph, 1)),
                "Unplanned_Purchases": float(unplanned),
                "Peer_Influence": float(peer_influence),
                "Finance_Confidence": float(finance_conf),
                "Monthly_Spend": float(spend_tier),
                "Num_Unexpected_Justifications": float(len(justify_selected)),
                **{k: float(v) for k, v in budget_flags.items()},
                **{k: float(v) for k, v in place_dummies.items()},
            }
            rf_df = pd.DataFrame([rf_row])
            for col in rf_cols:
                if col not in rf_df.columns:
                    rf_df[col] = 0
            rf_df = rf_df[rf_cols]

            scaled_row = models["scaler_rf"].transform(rf_df)
            persona_rf = models["rf"].predict(rf_df)[0]
            proba_rf = models["rf"].predict_proba(rf_df)[0]
            risk_label_map = {"Saver": 20, "Normal": 50, "Spender": 80}
            risk_score = risk_label_map.get(str(persona_rf), 50)
            rf_ok = True
        except Exception as e:
            rf_ok = False
            risk_score = 50
            persona_rf = "N/A"
            st.warning(f"RF error: {e}")

    # ── Display Results ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="result-card">
            <h3>📊 Spend Tier</h3>
            <h1 style="color:#2c3e50;">{spend_tier}/10</h1>
            <p>Expected monthly spend bracket</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk_class = "risk-high" if risk_score >= 66 else ("risk-medium" if risk_score >= 33 else "risk-low")
        risk_emoji = "🔴" if risk_score >= 66 else ("🟡" if risk_score >= 33 else "🟢")
        st.markdown(f"""
        <div class="result-card {risk_class}">
            <h3>{risk_emoji} Risk Score</h3>
            <h1 style="color:#2c3e50;">{risk_score}/100</h1>
            <p>Financial risk level</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="result-card">
            <h3>🧬 Persona</h3>
            <h2 style="color:#2c3e50;">{persona}</h2>
            <p>Behavioral cluster</p>
        </div>
        """, unsafe_allow_html=True)

    # Risk interpretation
    st.divider()
    st.subheader("📝 Interpretation")
    if risk_score >= 66:
        st.error("⚠️ **High Risk** — Your spending patterns show signs of financial vulnerability. Consider budgeting and tracking expenses more closely.")
    elif risk_score >= 33:
        st.warning("⚡ **Medium Risk** — You're managing fairly, but there's room to build better financial habits.")
    else:
        st.success("✅ **Low Risk** — Great job! Your financial behavior is healthy and disciplined.")

    # Spending breakdown
    if budget_selected:
        st.subheader("💸 Your Spending Categories")
        st.write(", ".join(budget_selected))

    st.divider()
    st.caption("This system is built for academic purposes at MNIT Jaipur. All predictions are model-generated estimates.")
