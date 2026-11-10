import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="EcoTwin-AI",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_model():
    model = joblib.load("EcoTwin_AI_Model.pkl")
    features = joblib.load("EcoTwin_AI_Features.pkl")
    return model, features

model, features = load_model()

# Professional EcoTwin-AI Theme
st.markdown("""
<style>
.stApp {
    background-color: #F3F6FA;
    color: #172B4D;
}
[data-testid="stSidebar"] {
    background-color: #10243A;
}
[data-testid="stSidebar"] * {
    color: #F4F7FB !important;
}
h1, h2, h3 {
    color: #10243A !important;
}
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE4ED;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 7px rgba(16,36,58,0.06);
}
[data-testid="stMetricValue"] {
    color: #138A8A;
}
.stButton > button {
    background-color: #138A8A;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}
.stButton > button:hover {
    background-color: #0E6F70;
    color: white;
}
</style>
""", unsafe_allow_html=True)


st.markdown('''
<style>
/* Make all input labels clearly visible */
.stNumberInput label,
.stTextInput label,
.stSelectbox label,
.stSlider label,
[data-testid="stWidgetLabel"] p {
    color: #10243A !important;
    font-weight: 600 !important;
    opacity: 1 !important;
}

/* Improve input field appearance */
.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #C8D3DF !important;
}

/* Improve sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #FFFFFF !important;
}
</style>
''', unsafe_allow_html=True)

st.title("⚙️ EcoTwin-AI")
st.subheader("Risk-Aware Physics-Informed Digital Twin for Bearing Fault Diagnosis")

st.info(
    "Software-based bearing monitoring using vibration features, "
    "theoretical bearing frequencies and risk-aware maintenance guidance."
)

st.sidebar.header("Operating Conditions")
rpm = st.sidebar.slider("Shaft Speed (RPM)", 500, 3000, 1500, 100)
torque = st.sidebar.slider("Torque (Nm)", 1.0, 10.0, 4.0, 0.5)

st.sidebar.header("Bearing Geometry")
n = 8
d = 6.75
D = 28.55
theta = 0.0

fr = rpm / 60
bpfo = (n / 2) * fr * (1 - (d / D) * np.cos(theta))
bpfi = (n / 2) * fr * (1 + (d / D) * np.cos(theta))
power = torque * 2 * np.pi * rpm / 60

c1, c2, c3, c4 = st.columns(4)
c1.metric("Shaft Speed", f"{rpm} RPM")
c2.metric("Mechanical Power", f"{power:.2f} W")
c3.metric("BPFO", f"{bpfo:.2f} Hz")
c4.metric("BPFI", f"{bpfi:.2f} Hz")

st.caption(
    "Power is calculated from entered torque and RPM. "
    "Bearing geometry is assumed; verify dimensions before practical use."
)

st.divider()
st.header("Bearing Vibration Feature Input")
st.write("Enter the 11 vibration features extracted from a bearing vibration recording.")

defaults = {
    "RMS": 0.2,
    "Standard_Deviation": 0.2,
    "Kurtosis": 3.0,
    "Skewness": 0.0,
    "BPFO_1x": 0.01,
    "BPFO_2x": 0.01,
    "BPFO_3x": 0.01,
    "BPFO_4x": 0.01,
    "BPFO_5x": 0.01,
    "BPFO_6x": 0.01,
    "BPFI": 0.01
}

values = {}
cols = st.columns(3)

for i, feature in enumerate(features):
    with cols[i % 3]:
        values[feature] = st.number_input(
            feature,
            value=float(defaults.get(feature, 0.01)),
            format="%.6f"
        )

if st.button("Run Bearing Diagnosis", type="primary"):
    input_df = pd.DataFrame([[values[f] for f in features]], columns=features)

    prediction = model.predict(input_df)[0]

    st.divider()
    st.header("Diagnosis Result")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability (%)": probabilities * 100
        })
        st.dataframe(prob_df, use_container_width=True)

        fault_probability = 0.0
        for cls, prob in zip(classes, probabilities):
            if "fault" in str(cls).lower():
                fault_probability = float(prob)

    st.subheader(f"Predicted Condition: {prediction}")

    if "fault" in str(prediction).lower():
        st.error("Potential bearing fault detected. Schedule inspection and further vibration analysis.")
    else:
        st.success("Bearing classified as healthy. Continue routine condition monitoring.")

    if hasattr(model, "predict_proba"):
        st.metric("Estimated Fault Probability", f"{fault_probability * 100:.1f}%")

st.divider()
st.header("Theoretical Bearing Frequency vs Shaft Speed")

rpm_range = np.arange(500, 3001, 100)
shaft_frequency = rpm_range / 60
bpfo_range = (n / 2) * shaft_frequency * (1 - (d / D) * np.cos(theta))
bpfi_range = (n / 2) * shaft_frequency * (1 + (d / D) * np.cos(theta))

fig = go.Figure()
fig.add_trace(go.Scatter(x=rpm_range, y=bpfo_range, name="BPFO"))
fig.add_trace(go.Scatter(x=rpm_range, y=bpfi_range, name="BPFI"))
fig.update_layout(
    xaxis_title="Shaft Speed (RPM)",
    yaxis_title="Frequency (Hz)",
    template="plotly_dark",
    height=450
)
st.plotly_chart(fig, use_container_width=True)

st.warning(
    "Limitations: This dashboard uses manually entered vibration features "
    "and a saved Random Forest model. RPM and torque sliders update theoretical "
    "calculations, not the model's vibration inputs. It does not directly process "
    "raw vibration files or measure electrical energy savings."
)

st.caption("EcoTwin-AI | Preliminary academic prototype")
