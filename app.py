import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import base64
from pathlib import Path


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Diabetes Risk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# LOAD BACKGROUND IMAGE
# =========================================================

background_path = Path("assets/diabetes.png")

if background_path.exists():

    with open(background_path, "rb") as image_file:
        background_base64 = base64.b64encode(
            image_file.read()
        ).decode()

else:
    background_base64 = ""


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    f"""
    <style>

    /* =====================================================
       BACKGROUND
       ===================================================== */

    .stApp {{

        background-color: #071b2a;

        background-image:
            linear-gradient(
                rgba(5, 25, 39, 0.82),
                rgba(5, 25, 39, 0.82)
            ),
            url("data:image/png;base64,{background_base64}");

        background-size: cover;

        background-position: center;

        background-attachment: fixed;

        background-repeat: no-repeat;
    }}


    /* =====================================================
       MAIN CONTAINER
       ===================================================== */

    .block-container {{

        max-width: 1250px;

        padding-top: 1.5rem;

        padding-bottom: 4rem;
    }}


    /* =====================================================
       WIDGET LABELS
       ===================================================== */

    [data-testid="stWidgetLabel"] p {{

        color: #eaf7fb !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #eaf7fb !important;
    }}


    /* =====================================================
       SELECT BOX
       ===================================================== */

    div[data-baseweb="select"] > div {{

        background-color: #ffffff !important;

        border: 1px solid #cbdde7 !important;

        border-radius: 10px !important;

        min-height: 42px !important;
    }}


    div[data-baseweb="select"] * {{

        color: #183b52 !important;
    }}


    /* =====================================================
       DROPDOWN
       ===================================================== */

    div[data-baseweb="popover"] {{

        background-color: #ffffff !important;
    }}


    div[data-baseweb="popover"] * {{

        color: #183b52 !important;
    }}


    div[role="option"] {{

        background-color: #ffffff !important;
    }}


    div[role="option"]:hover {{

        background-color: #edf7fb !important;
    }}


    /* =====================================================
       NUMBER INPUT
       ===================================================== */

    div[data-testid="stNumberInput"] input {{

        background-color: #ffffff !important;

        color: #183b52 !important;

        border: 1px solid #cbdde7 !important;

        border-radius: 10px !important;
    }}


    /* =====================================================
       TABS
       ===================================================== */

    div[data-baseweb="tab-list"] {{

        background: rgba(255,255,255,0.88);

        border-radius: 12px;

        padding: 4px;

        gap: 4px;
    }}


    button[data-baseweb="tab"] {{

        color: #123f62 !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #123f62 !important;
    }}

    button[data-baseweb="tab"] *,
    button[data-baseweb="tab"] p,
    button[data-baseweb="tab"] span,
    button[data-baseweb="tab"] div {{

        color: #123f62 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #123f62 !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{

        color: #0b6794 !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #0b6794 !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] *,
    button[data-baseweb="tab"][aria-selected="true"] p,
    button[data-baseweb="tab"][aria-selected="true"] span,
    button[data-baseweb="tab"][aria-selected="true"] div {{

        color: #0b6794 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #0b6794 !important;
    }}


    /* =====================================================
       BUTTON
       ===================================================== */

    div.stButton > button {{

        width: 100%;

        min-height: 54px;

        border-radius: 13px;

        font-size: 17px;

        font-weight: 750;

        background: linear-gradient(
            135deg,
            #0d6f9e,
            #1687ad
        );

        color: white;

        border: none;

        box-shadow:
            0 7px 18px rgba(13,95,134,0.20);

        transition: all 0.2s ease;
    }}


    div.stButton > button:hover {{

        transform: translateY(-2px);

        box-shadow:
            0 10px 24px rgba(13,95,134,0.25);
    }}


    /* =====================================================
       METRICS
       ===================================================== */

    [data-testid="stMetric"] {{

        background: rgba(255,255,255,0.99) !important;
        border: 1px solid #d8e7ef !important;
        border-radius: 15px;
        padding: 15px;
        opacity: 1 !important;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] p {{

        color: #607786 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #607786 !important;
    }}

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] p {{

        color: #123f62 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #123f62 !important;
        font-weight: 800 !important;
    }}

    /* Keep the full "Balanced XGBoost" model name visible. */
    div[data-testid="stHorizontalBlock"] > div:nth-child(1) [data-testid="stMetricValue"] {{
        font-size: 26px !important;
        white-space: nowrap !important;
        letter-spacing: -0.4px !important;
    }}

    /* =====================================================
       DIVIDER
       ===================================================== */

    hr {{

        margin-top: 30px;

        margin-bottom: 30px;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return joblib.load(
        "models/diabetes_xgb_model.pkl"
    )


# =========================================================
# LOAD SHAP EXPLAINER
# =========================================================

@st.cache_resource
def load_explainer():

    model = load_model()

    return shap.TreeExplainer(model)


model = load_model()

explainer = load_explainer()


# =========================================================
# HERO
# =========================================================

st.html(
    """
    <div style="
        background:
            linear-gradient(
                135deg,
                #0b4268,
                #176f96
            );

        padding:38px 42px;

        border-radius:25px;

        margin-bottom:25px;

        box-shadow:
            0 15px 40px rgba(13,66,104,0.20);
    ">

        <div style="
            color:white;
            font-size:44px;
            font-weight:800;
            line-height:1.15;
        ">
            🩺 Diabetes Risk AI
        </div>

        <div style="
            color:#e2f3fa;
            font-size:18px;
            margin-top:10px;
            line-height:1.5;
        ">
            Explainable machine learning for
            diabetes/prediabetes risk screening
        </div>

        <div style="
            display:inline-block;
            margin-top:18px;
            padding:9px 17px;
            border-radius:20px;
            background:rgba(255,255,255,0.15);
            color:white;
            font-size:13px;
            font-weight:600;
        ">
            🤖 Balanced XGBoost
            &nbsp; • &nbsp;
            🔎 SHAP Explainability
        </div>

    </div>
    """
)


# =========================================================
# INFORMATION CARDS
# =========================================================

info1, info2, info3 = st.columns(3)


with info1:

    st.html(
        """
        <div style="
            background:rgba(255,255,255,0.96);
            border:1px solid #d8e7ef;
            border-radius:18px;
            padding:23px;
            min-height:135px;
            box-shadow:0 6px 20px rgba(24,55,75,0.07);
        ">

            <div style="
                color:#123f62;
                font-size:18px;
                font-weight:750;
                margin-bottom:8px;
            ">
                📊 Risk Screening
            </div>

            <div style="
                color:#607786;
                font-size:14px;
                line-height:1.65;
            ">
                Estimates diabetes/prediabetes risk using
                health, lifestyle and demographic indicators.
            </div>

        </div>
        """
    )


with info2:

    st.html(
        """
        <div style="
            background:rgba(255,255,255,0.96);
            border:1px solid #d8e7ef;
            border-radius:18px;
            padding:23px;
            min-height:135px;
            box-shadow:0 6px 20px rgba(24,55,75,0.07);
        ">

            <div style="
                color:#123f62;
                font-size:18px;
                font-weight:750;
                margin-bottom:8px;
            ">
                🤖 Machine Learning
            </div>

            <div style="
                color:#607786;
                font-size:14px;
                line-height:1.65;
            ">
                Uses a Balanced XGBoost classification model
                trained using CDC health indicators.
            </div>

        </div>
        """
    )


with info3:

    st.html(
        """
        <div style="
            background:rgba(255,255,255,0.96);
            border:1px solid #d8e7ef;
            border-radius:18px;
            padding:23px;
            min-height:135px;
            box-shadow:0 6px 20px rgba(24,55,75,0.07);
        ">

            <div style="
                color:#123f62;
                font-size:18px;
                font-weight:750;
                margin-bottom:8px;
            ">
                🔎 Explainable AI
            </div>

            <div style="
                color:#607786;
                font-size:14px;
                line-height:1.65;
            ">
                SHAP explains which features influenced
                each individual model prediction.
            </div>

        </div>
        """
    )


# =========================================================
# HEALTH PROFILE
# =========================================================

st.html(
    """
    <div style="
        color:#eaf7fb;
        font-size:27px;
        font-weight:750;
        margin-top:30px;
        margin-bottom:18px;
    ">
        Your Health Profile
    </div>
    """
)


# =========================================================
# TABS
# =========================================================

tab_personal, tab_health, tab_lifestyle, tab_healthcare = st.tabs(
    [
        "👤 Personal",
        "❤️ Health",
        "🏃 Lifestyle",
        "🏥 Healthcare"
    ]
)


# =========================================================
# PERSONAL
# =========================================================

with tab_personal:

    col1, col2 = st.columns(2)

    with col1:

        sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        age = st.selectbox(
            "Age Group",
            [
                "18-24",
                "25-29",
                "30-34",
                "35-39",
                "40-44",
                "45-49",
                "50-54",
                "55-59",
                "60-64",
                "65-69",
                "70-74",
                "75-79",
                "80+"
            ]
        )

    with col2:

        education = st.selectbox(
            "Education Level",
            [
                "Never attended school",
                "Elementary",
                "Some high school",
                "High school graduate",
                "Some college / technical school",
                "College graduate"
            ]
        )

        income = st.selectbox(
            "Income Category",
            [
                "Less than $10,000",
                "$10,000-$14,999",
                "$15,000-$19,999",
                "$20,000-$24,999",
                "$25,000-$34,999",
                "$35,000-$49,999",
                "$50,000-$74,999",
                "$75,000 or more"
            ]
        )


# =========================================================
# HEALTH
# =========================================================

with tab_health:

    col1, col2 = st.columns(2)

    with col1:

        high_bp = st.selectbox(
            "High Blood Pressure",
            ["No", "Yes"]
        )

        high_chol = st.selectbox(
            "High Cholesterol",
            ["No", "Yes"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10,
            max_value=80,
            value=25,
            step=1
        )

        stroke = st.selectbox(
            "History of Stroke",
            ["No", "Yes"]
        )

        heart_disease = st.selectbox(
            "Heart Disease or Heart Attack",
            ["No", "Yes"]
        )

    with col2:

        general_health = st.selectbox(
            "General Health",
            [
                "Excellent",
                "Very Good",
                "Good",
                "Fair",
                "Poor"
            ]
        )

        difficulty_walking = st.selectbox(
            "Difficulty Walking or Climbing Stairs",
            ["No", "Yes"]
        )

        physical_health = st.slider(
            "Poor Physical Health Days",
            0,
            30,
            0
        )

        mental_health = st.slider(
            "Poor Mental Health Days",
            0,
            30,
            0
        )


# =========================================================
# LIFESTYLE
# =========================================================

with tab_lifestyle:

    col1, col2 = st.columns(2)

    with col1:

        smoker = st.selectbox(
            "Smoking History",
            ["No", "Yes"]
        )

        physical_activity = st.selectbox(
            "Physical Activity in Past 30 Days",
            ["No", "Yes"]
        )

        fruits = st.selectbox(
            "Consume Fruit Daily",
            ["No", "Yes"]
        )

    with col2:

        veggies = st.selectbox(
            "Consume Vegetables Daily",
            ["No", "Yes"]
        )

        heavy_alcohol = st.selectbox(
            "Heavy Alcohol Consumption",
            ["No", "Yes"]
        )


# =========================================================
# HEALTHCARE
# =========================================================

with tab_healthcare:

    col1, col2 = st.columns(2)

    with col1:

        chol_check = st.selectbox(
            "Cholesterol Check in Last 5 Years",
            ["No", "Yes"]
        )

        healthcare = st.selectbox(
            "Have Healthcare Coverage",
            ["No", "Yes"]
        )

    with col2:

        no_doc_cost = st.selectbox(
            "Could Not See Doctor Due to Cost",
            ["No", "Yes"]
        )


# =========================================================
# PREDICT BUTTON
# =========================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

predict_button = st.button(
    "🔍 Calculate My Risk",
    type="primary"
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    # =====================================================
    # ENCODING
    # =====================================================

    binary_map = {
        "No": 0,
        "Yes": 1
    }

    age_map = {
        "18-24": 1,
        "25-29": 2,
        "30-34": 3,
        "35-39": 4,
        "40-44": 5,
        "45-49": 6,
        "50-54": 7,
        "55-59": 8,
        "60-64": 9,
        "65-69": 10,
        "70-74": 11,
        "75-79": 12,
        "80+": 13
    }

    education_map = {
        "Never attended school": 1,
        "Elementary": 2,
        "Some high school": 3,
        "High school graduate": 4,
        "Some college / technical school": 5,
        "College graduate": 6
    }

    income_map = {
        "Less than $10,000": 1,
        "$10,000-$14,999": 2,
        "$15,000-$19,999": 3,
        "$20,000-$24,999": 4,
        "$25,000-$34,999": 5,
        "$35,000-$49,999": 6,
        "$50,000-$74,999": 7,
        "$75,000 or more": 8
    }

    general_health_map = {
        "Excellent": 1,
        "Very Good": 2,
        "Good": 3,
        "Fair": 4,
        "Poor": 5
    }


    # =====================================================
    # INPUT DATA
    # =====================================================

    input_data = pd.DataFrame(
        [{
            "HighBP": binary_map[high_bp],
            "HighChol": binary_map[high_chol],
            "CholCheck": binary_map[chol_check],
            "BMI": bmi,
            "Smoker": binary_map[smoker],
            "Stroke": binary_map[stroke],
            "HeartDiseaseorAttack": binary_map[heart_disease],
            "PhysActivity": binary_map[physical_activity],
            "Fruits": binary_map[fruits],
            "Veggies": binary_map[veggies],
            "HvyAlcoholConsump": binary_map[heavy_alcohol],
            "AnyHealthcare": binary_map[healthcare],
            "NoDocbcCost": binary_map[no_doc_cost],
            "GenHlth": general_health_map[general_health],
            "MentHlth": mental_health,
            "PhysHlth": physical_health,
            "DiffWalk": binary_map[difficulty_walking],
            "Sex": 1 if sex == "Male" else 0,
            "Age": age_map[age],
            "Education": education_map[education],
            "Income": income_map[income]
        }]
    )


    # =====================================================
    # FEATURE ORDER
    # =====================================================

    feature_order = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "BMI",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "GenHlth",
        "MentHlth",
        "PhysHlth",
        "DiffWalk",
        "Sex",
        "Age",
        "Education",
        "Income"
    ]

    input_data = input_data[
        feature_order
    ]


    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    probability = model.predict_proba(
        input_data
    )[0, 1]

    threshold = 0.65

    prediction = int(
        probability >= threshold
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.markdown("---")

    st.html(
        """
        <div style="
            color:#eaf7fb;
            font-size:27px;
            font-weight:750;
            margin-top:30px;
            margin-bottom:18px;
        ">
            Your Result
        </div>
        """
    )


    percentage = probability * 100


    st.html(
        f"""
        <div style="
            background:rgba(255,255,255,0.98);

            border-radius:24px;

            padding:35px;

            text-align:center;

            border:1px solid #dce8ef;

            box-shadow:
                0 10px 32px rgba(24,55,75,0.09);

            margin:20px 0;
        ">

            <div style="
                color:#718391;
                font-size:13px;
                font-weight:750;
                letter-spacing:1.5px;
            ">
                MODEL-ESTIMATED RISK
            </div>

            <div style="
                color:#0d4268;
                font-size:58px;
                font-weight:850;
                margin:5px 0;
            ">
                {percentage:.2f}%
            </div>

            <div style="
                color:#667985;
                font-size:15px;
            ">
                Estimated probability produced by the
                machine-learning model
            </div>

            <div style="
                height:12px;
                background:#e7eef3;
                border-radius:20px;
                margin:22px auto 10px auto;
                max-width:600px;
                overflow:hidden;
            ">

                <div style="
                    height:100%;
                    width:{min(percentage,100):.2f}%;
                    border-radius:20px;
                    background:linear-gradient(
                        90deg,
                        #54a9c7,
                        #176b91
                    );
                ">
                </div>

            </div>

        </div>
        """
    )


    # =====================================================
    # RISK CATEGORY
    # =====================================================

    if prediction == 1:

        st.error(
            "⚠️ Higher-risk screening category"
        )

        st.write(
            "The model-estimated probability is at or "
            "above the selected 65% operating threshold."
        )

    else:

        st.success(
            "✓ Lower-risk screening category"
        )

        st.write(
            "The model-estimated probability is below "
            "the selected 65% operating threshold."
        )


    st.caption(
        "The 65% threshold is an evaluated model operating "
        "point for this project and is not a medically "
        "validated diagnostic cutoff."
    )


    # =====================================================
    # SHAP
    # =====================================================

    st.html(
        """
        <div style="
            color:#eaf7fb;
            font-size:27px;
            font-weight:750;
            margin-top:30px;
            margin-bottom:18px;
        ">
            🔎 Why did the model make this prediction?
        </div>
        """
    )


    shap_values = explainer.shap_values(
        input_data
    )

    expected_value = explainer.expected_value

    if hasattr(expected_value, "__len__"):

        expected_value = expected_value[0]


    shap_explanation = shap.Explanation(
        values=shap_values[0],
        base_values=expected_value,
        data=input_data.iloc[0].values,
        feature_names=input_data.columns.tolist()
    )


    st.html(
        """
        <div style="
            background:rgba(255,255,255,0.98);
            border:1px solid #dfe9ef;
            border-radius:20px;
            padding:25px;
            margin-top:20px;
            box-shadow:0 6px 22px rgba(24,55,75,0.06);
        ">
        """
    )


    shap.plots.waterfall(
        shap_explanation,
        max_display=10,
        show=False
    )


    fig = plt.gcf()

    st.pyplot(
        fig,
        clear_figure=True
    )

    st.html("</div>")


    # =====================================================
    # SHAP FEATURE LABELS
    # =====================================================

    feature_labels = {

        "HighBP":
            "High blood pressure",

        "HighChol":
            "High cholesterol",

        "CholCheck":
            "Cholesterol check",

        "BMI":
            "BMI",

        "Smoker":
            "Smoking history",

        "Stroke":
            "History of stroke",

        "HeartDiseaseorAttack":
            "Heart disease or heart attack",

        "PhysActivity":
            "Physical activity",

        "Fruits":
            "Daily fruit consumption",

        "Veggies":
            "Daily vegetable consumption",

        "HvyAlcoholConsump":
            "Heavy alcohol consumption",

        "AnyHealthcare":
            "Healthcare coverage",

        "NoDocbcCost":
            "Could not see doctor due to cost",

        "GenHlth":
            "General health",

        "MentHlth":
            "Poor mental health days",

        "PhysHlth":
            "Poor physical health days",

        "DiffWalk":
            "Difficulty walking",

        "Sex":
            "Sex",

        "Age":
            "Age group",

        "Education":
            "Education level",

        "Income":
            "Income category"
    }


    explanation_df = pd.DataFrame(
        {
            "Feature": input_data.columns,
            "SHAP": shap_values[0]
        }
    )


    explanation_df["Feature"] = (
        explanation_df["Feature"]
        .map(feature_labels)
    )


    higher_risk = (
        explanation_df[
            explanation_df["SHAP"] > 0
        ]
        .sort_values(
            "SHAP",
            ascending=False
        )
        .head(3)
    )


    lower_risk = (
        explanation_df[
            explanation_df["SHAP"] < 0
        ]
        .sort_values(
            "SHAP",
            ascending=True
        )
        .head(3)
    )


    # =====================================================
    # SHAP SUMMARY
    # =====================================================

    col1, col2 = st.columns(2)


    with col1:

        st.html(
            """
            <div style="
                background:rgba(255,255,255,0.97);
                border:1px solid #d8e7ef;
                border-radius:18px;
                padding:23px;
                min-height:150px;
                box-shadow:0 6px 20px rgba(24,55,75,0.07);
            ">

                <div style="
                    color:#123f62;
                    font-size:18px;
                    font-weight:750;
                    margin-bottom:10px;
                ">
                    🔴 Higher model contribution
                </div>
            """
        )


        if higher_risk.empty:

            st.write(
                "No major positive contributions."
            )

        else:

            for _, row in higher_risk.iterrows():

                st.write(
                    f"• **{row['Feature']}**"
                )


        st.html("</div>")


    with col2:

        st.html(
            """
            <div style="
                background:rgba(255,255,255,0.97);
                border:1px solid #d8e7ef;
                border-radius:18px;
                padding:23px;
                min-height:150px;
                box-shadow:0 6px 20px rgba(24,55,75,0.07);
            ">

                <div style="
                    color:#123f62;
                    font-size:18px;
                    font-weight:750;
                    margin-bottom:10px;
                ">
                    🔵 Lower model contribution
                </div>
            """
        )


        if lower_risk.empty:

            st.write(
                "No major negative contributions."
            )

        else:

            for _, row in lower_risk.iterrows():

                st.write(
                    f"• **{row['Feature']}**"
                )


        st.html("</div>")


# =========================================================
# HOW IT WORKS
# =========================================================

st.markdown("---")


st.html(
    """
    <div style="
        color:#eaf7fb;
        font-size:27px;
        font-weight:750;
        margin-top:30px;
        margin-bottom:18px;
    ">
        ⚙️ How It Works
    </div>
    """
)


step1, step2, step3, step4 = st.columns(4)


def process_card(
    number,
    title,
    description
):

    st.html(
        f"""
        <div style="
            background:rgba(255,255,255,0.97);
            border:1px solid #d8e7ef;
            border-radius:18px;
            padding:23px;
            min-height:150px;
            box-shadow:0 6px 20px rgba(24,55,75,0.07);
        ">

            <div style="
                color:#123f62;
                font-size:18px;
                font-weight:750;
                margin-bottom:8px;
            ">
                {number} · {title}
            </div>

            <div style="
                color:#607786;
                font-size:14px;
                line-height:1.65;
            ">
                {description}
            </div>

        </div>
        """
    )


with step1:

    process_card(
        "01",
        "Input",
        "Enter health, lifestyle and demographic information."
    )


with step2:

    process_card(
        "02",
        "Predict",
        "Balanced XGBoost estimates the model probability."
    )


with step3:

    process_card(
        "03",
        "Explain",
        "SHAP identifies features influencing the prediction."
    )


with step4:

    process_card(
        "04",
        "Understand",
        "Review the model estimate and contributing factors."
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.html(
    """
    <div style="
        color:#eaf7fb;
        font-size:27px;
        font-weight:750;
        margin-top:30px;
        margin-bottom:18px;
    ">
        📊 Model Information
    </div>
    """
)


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "Model",
        "Balanced XGBoost"
    )


with m2:

    st.metric(
        "Features",
        "21"
    )


with m3:

    st.metric(
        "Explainability",
        "SHAP"
    )


with m4:

    st.metric(
        "Threshold",
        "65%"
    )


# =========================================================
# DISCLAIMER
# =========================================================

st.html(
    """
    <div style="
        background:rgba(255,248,231,0.97);
        border-left:5px solid #e1aa28;
        padding:16px 19px;
        border-radius:10px;
        color:#634f13;
        font-size:13px;
        line-height:1.6;
        margin-top:25px;
    ">

        <b>Medical disclaimer:</b>

        This application is an educational machine-learning
        project designed for risk screening research and
        demonstration. It is not a diagnostic tool and
        should not be used to make medical decisions.

        <br><br>

        <b>Explainability disclaimer:</b>

        SHAP values describe how the machine-learning model
        used the input features for a prediction. They do
        not establish medical causation.

    </div>
    """
)


# =========================================================
# FOOTER
# =========================================================

st.html(
    """
    <div style="
        text-align:center;
        color:#b8ccd6;
        font-size:13px;
        margin-top:45px;
        padding-top:20px;
        border-top:1px solid rgba(220,230,236,0.25);
    ">

        <b>Diabetes Risk AI</b>

        <br>

        Explainable Machine Learning Project

        <br>

        Python · XGBoost · SHAP · Streamlit

    </div>
    """
)