import streamlit as st
import pandas as pd
import joblib
import math
import streamlit.components.v1 as components


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Machine Failure Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("machine_failure_pipeline.pkl")
threshold = joblib.load("failure_threshold.pkl")


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
/* Remove Streamlit's default white top header */
[data-testid="stHeader"] {
    display: none;
}
    background:
        radial-gradient(
            circle at 85% 5%,
            rgba(105, 45, 190, 0.18),
            transparent 30%
        ),
        radial-gradient(
            circle at 10% 90%,
            rgba(20, 80, 180, 0.12),
            transparent 30%
        ),
        #050914;

    color: #E8EEF7;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 1rem;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

[data-testid="stSidebar"] {
    background: #070C18;
    border-right: 1px solid #1B2940;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 900;
    letter-spacing: 1px;
    line-height: 1.25;
    color: #6EA8FF;
    margin-bottom: 20px;
}

.sidebar-section {
    color: #6AAEFF;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-top: 25px;
    margin-bottom: 10px;
}

.sidebar-text {
    color: #AAB7CA;
    font-size: 13px;
    line-height: 1.6;
}

.live-badge {
    display: inline-block;
    color: #35E995;
    border: 1px solid #087D50;
    background: rgba(20, 220, 130, 0.08);
    border-radius: 20px;
    padding: 7px 13px;
    font-size: 12px;
    font-weight: 800;
}


/* =====================================================
   HEADER
   ===================================================== */

.hero-title {
    font-size: 43px;
    font-weight: 900;
    letter-spacing: -1.5px;
    color: #F4F7FB;
    margin-bottom: 3px;
}

.hero-subtitle {
    font-size: 18px;
    font-weight: 700;
    color: #8E6BFF;
}

.hero-description {
    color: #6F809A;
    font-size: 13px;
    margin-top: 5px;
    margin-bottom: 25px;
}


/* =====================================================
   CARDS
   ===================================================== */

.card {
    background: linear-gradient(
        145deg,
        rgba(12, 23, 42, 0.98),
        rgba(7, 14, 27, 0.98)
    );

    border: 1px solid #1D2B43;
    border-radius: 16px;
    padding: 22px;

    box-shadow:
        0 15px 40px rgba(0, 0, 0, 0.20);
}

.card-title {
    color: #62AFFF;
    font-size: 16px;
    font-weight: 850;
    letter-spacing: 0.4px;
}

.card-subtitle {
    color: #61748F;
    font-size: 12px;
    margin-top: 4px;
    margin-bottom: 18px;
}


/* =====================================================
   INPUTS
   ===================================================== */

label {
    color: #B9C7D9 !important;
    font-weight: 600 !important;
}

div[data-baseweb="select"] > div {
    background: #0A1425 !important;
    border: 1px solid #253653 !important;
    border-radius: 9px !important;
}

div[data-testid="stNumberInput"] input {
    background: #0A1425 !important;
    border: 1px solid #253653 !important;
    border-radius: 9px !important;
    color: white !important;
}


/* =====================================================
   BUTTON
   ===================================================== */

div.stButton > button {
    width: 100%;
    height: 52px;

    border: none;
    border-radius: 10px;

    background:
        linear-gradient(
            90deg,
            #176FFF,
            #9239E8
        );

    color: white;

    font-size: 16px;
    font-weight: 850;

    transition: 0.2s;
}

div.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 10px 28px rgba(111, 65, 230, 0.35);
}


/* =====================================================
   PROBABILITY
   ===================================================== */

.probability-label {
    text-align: center;

    color: #8292AA;

    font-size: 12px;
    font-weight: 800;

    letter-spacing: 1.5px;
}

.probability-value {
    text-align: center;

    font-size: 46px;
    font-weight: 950;

    margin-top: -8px;
}

.safe-value {
    color: #27E39A;

    text-shadow:
        0 0 25px rgba(39, 227, 154, 0.25);
}

.danger-value {
    color: #FF4D61;

    text-shadow:
        0 0 25px rgba(255, 77, 97, 0.25);
}


/* =====================================================
   RESULT
   ===================================================== */

.result-danger {
    border: 1px solid rgba(255, 60, 80, 0.55);

    background:
        rgba(255, 40, 60, 0.07);

    border-radius: 12px;

    padding: 18px;

    text-align: center;
}

.result-safe {
    border: 1px solid rgba(30, 220, 140, 0.40);

    background:
        rgba(20, 220, 140, 0.06);

    border-radius: 12px;

    padding: 18px;

    text-align: center;
}

.result-title {
    font-size: 19px;
    font-weight: 900;
}

.result-description {
    color: #92A2B8;
    font-size: 12px;
    margin-top: 5px;
}


/* =====================================================
   MODEL METRICS
   ===================================================== */

.metric-card {
    background: #0A1425;

    border: 1px solid #1D2B43;

    border-radius: 12px;

    padding: 17px 10px;

    text-align: center;
}

.metric-name {
    color: #71829B;

    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.7px;
}

.metric-value {
    color: #EDF3FA;

    font-size: 25px;

    font-weight: 900;

    margin-top: 4px;
}


/* =====================================================
   ERROR ANALYSIS
   ===================================================== */

.error-card {
    background: #0A1425;

    border: 1px solid #1D2B43;

    border-radius: 12px;

    padding: 20px 10px;

    text-align: center;
}

.error-number {
    font-size: 32px;

    font-weight: 900;
}

.error-label {
    font-size: 11px;

    font-weight: 800;

    letter-spacing: 0.5px;

    margin-top: 4px;
}

.error-description {
    color: #71829B;

    font-size: 10px;

    margin-top: 8px;
}

.matrix-card {
    margin-top: 15px;

    background: #081120;

    border: 1px solid #16243A;

    border-radius: 10px;

    padding: 14px;

    text-align: center;
}

.matrix-title {
    color: #AEBED2;

    font-size: 11px;

    font-weight: 800;
}

.matrix-text {
    color: #667A95;

    font-size: 10px;

    margin-top: 6px;
}


/* =====================================================
   SHAP
   ===================================================== */

.shap-card {
    background: #0A1425;

    border: 1px solid #1D2B43;

    border-radius: 11px;

    padding: 13px;

    margin-bottom: 10px;
}

.shap-header {
    display: flex;

    justify-content: space-between;

    align-items: center;
}

.shap-name {
    color: #B9C7D9;

    font-size: 12px;

    font-weight: 600;
}

.shap-rank {
    color: #647791;

    font-size: 10px;
}

.shap-background {
    height: 7px;

    width: 100%;

    background: #17243A;

    border-radius: 10px;

    margin-top: 9px;

    overflow: hidden;
}

.shap-bar {
    height: 100%;

    border-radius: 10px;

    background:
        linear-gradient(
            90deg,
            #176FFF,
            #A63DFF
        );
}

.shap-note {
    color: #52637B;

    font-size: 10px;

    margin-top: 14px;
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;

    color: #52637B;

    font-size: 11px;

    padding: 25px 0 5px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">'
        '⚙️ PREDICT<br>MAINTENANCE'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="live-badge">'
        '🟢 MODEL: LIVE'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">'
        'ABOUT THIS APP'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-text">'
        'Predict the probability of machine failure '
        'using operational and sensor measurements.'
        '<br><br>'
        'Powered by an optimized XGBoost classifier.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">'
        'MODEL INFORMATION'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-text">'
        '<b>Algorithm</b><br>'
        'XGBoost Classifier'
        '<br><br>'
        '<b>Problem</b><br>'
        'Predictive Maintenance'
        '<br><br>'
        '<b>Decision Threshold</b><br>'
        '0.70 (70%)'
        '<br><br>'
        '<b>Target</b><br>'
        'Failure (1) / No Failure (0)'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption("Built with ❤️ using Streamlit")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hero-title">'
    '⚙️ Machine Failure Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">'
    'AI-Powered Predictive Maintenance using XGBoost'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-description">'
    'Predict the probability of machine failure '
    'based on operating conditions.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MAIN COLUMNS
# =========================================================

input_col, result_col = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# INPUT CARD
# =========================================================

with input_col:

    st.markdown(
        '<div class="card">'
        '<div class="card-title">'
        '① INPUT PARAMETERS'
        '</div>'
        '<div class="card-subtitle">'
        'Enter the current machine operating conditions.'
        '</div>',
        unsafe_allow_html=True
    )

    machine_type = st.selectbox(
        "Machine Type",
        ["L", "M", "H"]
    )

    air_temp = st.number_input(
        "Air temperature [K]",
        min_value=250.0,
        max_value=350.0,
        value=300.0,
        step=0.01
    )

    process_temp = st.number_input(
        "Process temperature [K]",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.01
    )

    rpm = st.number_input(
        "Rotational speed [rpm]",
        min_value=0,
        max_value=5000,
        value=1500,
        step=1
    )

    torque = st.number_input(
        "Torque [Nm]",
        min_value=0.0,
        max_value=100.0,
        value=40.0,
        step=0.1
    )

    tool_wear = st.number_input(
        "Tool wear [min]",
        min_value=0,
        max_value=300,
        value=100,
        step=1
    )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    predict_button = st.button(
        "⚡ PREDICT FAILURE"
    )

    st.markdown(
        '<div style="'
        'color:#52627A;'
        'font-size:10px;'
        'margin-top:8px;">'
        '* All fields are required'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RESULT CARD
# =========================================================

with result_col:

    st.markdown(
        '<div class="card">'
        '<div class="card-title">'
        '② PREDICTION RESULT'
        '</div>'
        '<div class="card-subtitle">'
        'Model prediction based on the supplied operating conditions.'
        '</div>',
        unsafe_allow_html=True
    )

    if predict_button:

        # -------------------------------------------------
        # INPUT DATA
        # -------------------------------------------------

        input_data = pd.DataFrame(
            {
                "Type": [machine_type],

                "Air temperature [K]": [
                    air_temp
                ],

                "Process temperature [K]": [
                    process_temp
                ],

                "Rotational speed [rpm]": [
                    rpm
                ],

                "Torque [Nm]": [
                    torque
                ],

                "Tool wear [min]": [
                    tool_wear
                ]
            }
        )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        probability = model.predict_proba(
            input_data
        )[0, 1]

        prediction = int(
            probability >= threshold
        )

        probability_percent = (
            probability * 100
        )


        # -------------------------------------------------
        # GAUGE
        # -------------------------------------------------

        progress = max(
            0.0,
            min(probability, 1.0)
        )

        angle = math.pi * progress

        needle_x = (
            100
            - 75 * math.cos(angle)
        )

        needle_y = (
            100
            - 75 * math.sin(angle)
        )


        if prediction == 1:

            value_color = "#FF4D61"

        else:

            value_color = "#27E39A"


        gauge_html = f"""
<!DOCTYPE html>

<html>

<head>

<style>

html, body {{

    margin: 0;

    padding: 0;

    background: transparent;

    overflow: hidden;

    font-family: Arial, sans-serif;

}}

.gauge-container {{

    width: 360px;

    height: 245px;

    margin: auto;

    position: relative;

}}

.gauge-svg {{

    width: 360px;

    height: 215px;

    display: block;

    margin: auto;

}}

.gauge-value {{

    position: absolute;

    left: 0;

    right: 0;

    bottom: 0;

    text-align: center;

    font-size: 42px;

    font-weight: 900;

    color: {value_color};

    text-shadow:
        0 0 18px {value_color}55;

}}

</style>

</head>


<body>

<div class="gauge-container">


<svg
    class="gauge-svg"
    viewBox="0 0 200 125"
>


<defs>

<linearGradient
    id="riskGradient"
    x1="0%"
    y1="0%"
    x2="100%"
    y2="0%"
>

<stop
    offset="0%"
    stop-color="#27E39A"
/>

<stop
    offset="50%"
    stop-color="#FFC107"
/>

<stop
    offset="100%"
    stop-color="#FF4058"
/>

</linearGradient>

</defs>


<!-- BACKGROUND -->

<path
    d="M 10 100 A 90 90 0 0 1 190 100"
    fill="none"
    stroke="#18263D"
    stroke-width="11"
    stroke-linecap="round"
/>


<!-- PROGRESS -->

<path
    d="M 10 100 A 90 90 0 0 1 190 100"
    fill="none"
    stroke="url(#riskGradient)"
    stroke-width="11"
    stroke-linecap="round"

    pathLength="100"

    stroke-dasharray="{progress * 100} 100"
/>


<!-- NEEDLE -->

<line
    x1="100"
    y1="100"

    x2="{needle_x}"
    y2="{needle_y}"

    stroke="#FFFFFF"
    stroke-width="3"

    stroke-linecap="round"
/>


<!-- CENTER -->

<circle
    cx="100"
    cy="100"
    r="7"

    fill="#FFFFFF"
/>

<circle
    cx="100"
    cy="100"
    r="3"

    fill="#080E1C"
/>


<!-- SCALE -->

<text
    x="4"
    y="117"

    fill="#71829B"
    font-size="8"
>
0%
</text>


<text
    x="95"
    y="14"

    fill="#71829B"
    font-size="8"
>
50%
</text>


<text
    x="176"
    y="117"

    fill="#71829B"
    font-size="8"
>
100%
</text>


</svg>


<div class="gauge-value">
{probability_percent:.2f}%
</div>


</div>

</body>

</html>
"""


        # Render actual HTML component
        components.html(
            gauge_html,
            height=245,
            scrolling=False
        )


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        if prediction == 1:

            st.markdown(
                f'<div class="result-danger">'
                f'<div class="result-title">'
                f'⚠️ MACHINE FAILURE PREDICTED'
                f'</div>'
                f'<div class="result-description">'
                f'Failure probability is above the '
                f'{threshold:.0%} decision threshold. '
                f'Machine inspection is recommended.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f'<div class="result-safe">'
                f'<div class="result-title">'
                f'✅ NO MACHINE FAILURE PREDICTED'
                f'</div>'
                f'<div class="result-description">'
                f'Failure probability is below the '
                f'{threshold:.0%} decision threshold.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    else:

        st.markdown(
            '<div style="'
            'height:380px;'
            'display:flex;'
            'align-items:center;'
            'justify-content:center;'
            'text-align:center;'
            'color:#52627A;">'

            '<div>'

            '<div style="font-size:48px;">'
            '⚙️'
            '</div>'

            '<div style="'
            'font-size:18px;'
            'font-weight:800;'
            'color:#71829B;">'
            'Awaiting prediction'
            '</div>'

            '<div style="'
            'font-size:12px;'
            'margin-top:6px;">'
            'Enter machine parameters and run the model.'
            '</div>'

            '</div>'

            '</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card-title">'
    '🏆 MODEL PERFORMANCE'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="card-subtitle">'
    'Final evaluation on the held-out test set'
    '</div>',
    unsafe_allow_html=True
)


m1, m2, m3, m4 = st.columns(4)


metrics = [
    ("ACCURACY", "98.65%"),
    ("PRECISION", "82.54%"),
    ("RECALL", "76.47%"),
    ("F1 SCORE", "79.39%")
]


for col, (name, value) in zip(
    [m1, m2, m3, m4],
    metrics
):

    with col:

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-name">'
            f'{name}'
            f'</div>'
            f'<div class="metric-value">'
            f'{value}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )


# =========================================================
# ERROR ANALYSIS + SHAP
# =========================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)


error_col, shap_col = st.columns(
    [1, 1],
    gap="large"
)


# =========================================================
# ERROR ANALYSIS
# =========================================================

with error_col:

    st.markdown(
        '<div class="card">'
        '<div class="card-title">'
        '🔍 ERROR ANALYSIS'
        '</div>'
        '<div class="card-subtitle">'
        'Final test-set classification errors'
        '</div>',
        unsafe_allow_html=True
    )


    e1, e2 = st.columns(2)


    with e1:

        st.markdown(
            '<div class="error-card">'
            '<div class="error-number" '
            'style="color:#FF596B;">'
            '16'
            '</div>'
            '<div class="error-label" '
            'style="color:#FF596B;">'
            'FALSE NEGATIVES'
            '</div>'
            '<div class="error-description">'
            'Actual failure predicted as normal'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    with e2:

        st.markdown(
            '<div class="error-card">'
            '<div class="error-number" '
            'style="color:#FFAE55;">'
            '11'
            '</div>'
            '<div class="error-label" '
            'style="color:#FFAE55;">'
            'FALSE POSITIVES'
            '</div>'
            '<div class="error-description">'
            'Normal machine predicted as failure'
            '</div>'
            '</div>',
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="matrix-card">'
        '<div class="matrix-title">'
        'CONFUSION MATRIX'
        '</div>'
        '<div class="matrix-text">'
        '1921 True Negatives '
        '&nbsp;•&nbsp; '
        '52 True Positives '
        '&nbsp;•&nbsp; '
        '11 False Positives '
        '&nbsp;•&nbsp; '
        '16 False Negatives'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# SHAP
# =========================================================

with shap_col:

    st.markdown(
        '<div class="card">'
        '<div class="card-title">'
        '🧠 TOP FEATURE IMPACT'
        '</div>'
        '<div class="card-subtitle">'
        'Global feature ranking from SHAP analysis'
        '</div>',
        unsafe_allow_html=True
    )


    shap_features = [
        ("Torque [Nm]", 100),
        ("Tool wear [min]", 82),
        ("Rotational speed [rpm]", 68),
        ("Air temperature [K]", 50),
        ("Process temperature [K]", 42)
    ]


    for rank, (feature, width) in enumerate(
        shap_features,
        start=1
    ):

        st.markdown(
            f'<div class="shap-card">'

            f'<div class="shap-header">'

            f'<div class="shap-name">'
            f'{feature}'
            f'</div>'

            f'<div class="shap-rank">'
            f'#{rank}'
            f'</div>'

            f'</div>'

            f'<div class="shap-background">'

            f'<div class="shap-bar" '
            f'style="width:{width}%;">'
            f'</div>'

            f'</div>'

            f'</div>',
            unsafe_allow_html=True
        )


    st.markdown(
        '<div class="shap-note">'
        'Feature ranking is based on the global SHAP '
        'analysis performed during model explainability.'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '<div class="footer">'
    '⚙️ Machine Failure Prediction '
    '&nbsp;•&nbsp; '
    'XGBoost '
    '&nbsp;•&nbsp; '
    'Scikit-learn '
    '&nbsp;•&nbsp; '
    'Pandas '
    '&nbsp;•&nbsp; '
    'Streamlit'
    '</div>',
    unsafe_allow_html=True
)