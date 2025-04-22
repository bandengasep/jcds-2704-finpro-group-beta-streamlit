import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
import plotly.graph_objects as go
from PIL import Image
from models import AdjustedThresholdModel

# Set page config
st.set_page_config(page_title="Job Intention Predictor", page_icon="🏃‍♀️💨", layout="wide")

# CSS Styling for attractive UI
st.markdown("""
    <style>
    body {
        background-color: #f7f9fb;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #002e5d, #005b96);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5em 2em;
        transition: 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(90deg, #001f3f, #003e66);
        color: #f0f0f0;
        transform: scale(1.05);
    }
    div.stButton > button:hover {
        background: background-color(#1f77b4);
    }        
    .metric-container {
        text-align: center;
        padding: 1em;
        background-color: #e9f5ff;
        border-radius: 12px;
        margin-top: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('final_model.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Gauge chart
def create_gauge_chart(probability):
    # Force background to be black
    bg_color = "#0e1117"
    text_color = "white"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Candidate Leave Probability", 'font': {'size': 24, 'color': text_color}},
        number={'font': {'size': 20, 'color': text_color}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "#ffffff"},
            'bar': {'color': "#4CAF50"},
            'bgcolor': bg_color,
            'borderwidth': 2,
            'bordercolor': text_color,
            'steps': [
                {'range': [0, 0.41], 'color': "#90EE90"},
                {'range': [0.41, 1], 'color': "#FF6347"}
            ],
            'threshold': {
                'line': {'color': "#0e1117", 'width': 4},
                'thickness': 1,
                'value': 0.41
            }
        }))

    fig.update_layout(
        paper_bgcolor=bg_color,
        font={'color': text_color, 'family': "Arial"},
        height=300,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


# Header
header_image = Image.open("contoso_header.png")
st.image(header_image, use_container_width=True)

# Navigation Buttons
col1, col2, col3 = st.columns([1, 2, 0.3])
with col1:
    st.write("")
with col3:
    predictor_button = st.button("Predictor")
    about_button = st.button("About Us")

# About Page
if about_button:
    st.title("About Us")
    st.markdown(
        """
        <p style="text-align: center; font-size: 28px; font-weight: bold; margin-top: 20px;">
            This Job Intention Predictor was created by <span style="color: #1DA1F2;">Beta Consulting</span>:
        </p><br>
        """,
        unsafe_allow_html=True
    )

    team_members = [
        {
            "name": "Kerin Mulianto",
            "title": "Data Scientist",
            "img": "kerin.png",
            "linkedin": "https://www.linkedin.com/in/kerin-m/",
            "portfolio": "https://kerin-web-porto.netlify.app/" 
        },
        {
            "name": "Timothy Hartanto",
            "title": "Data Scientist",
            "img": "timothy.png",
            "linkedin": "https://www.linkedin.com/in/timothy-hartanto/",
            "portfolio": "https://bandengasep.github.io/personal-portfolio/" 
        },
        {
            "name": "Wafa Nabila",
            "title": "Data Scientist",
            "img": "wafa.png",
            "linkedin": "https://www.linkedin.com/in/wafanabilas/",
            "portfolio": "https://wafana.github.io/Portofolio_Website_0/#about"  
        },
    ]

    for member in team_members:
        col1, col2 = st.columns([0.5, 3])  
        with col1:
            st.image(member["img"], width=180)

        with col2:
            st.markdown(
                f"""
                <div style="margin-top: 25px;">
                    <p style="font-size: 22px; margin-bottom: 5px;"><strong>{member["name"]}</strong></p>
                    <p style="font-size: 17px; color: gray; margin-top: 0;">{member["title"]}</p>
                    <p style="margin-top: 8px;">
                        <a href="{member["linkedin"]}" target="_blank" style="text-decoration: none; color: #1DA1F2;">LinkedIn</a> | 
                        <a href="{member["portfolio"]}" target="_blank" style="text-decoration: none; color: #1DA1F2;">Portfolio</a>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        st.markdown("<hr>", unsafe_allow_html=True)

else:
    st.title("📊 Job Intention Predictor")
    st.subheader("🖋️ Enter Candidate Details")

    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", options=['city_1', 'city_2', 'city_7', 'city_8', 'city_9', 'city_10', 'city_11', 'city_12', 'city_13', 'city_14', 
 'city_16', 'city_18', 'city_19', 'city_20', 'city_21', 'city_23', 'city_24', 'city_25', 'city_26', 'city_27',
 'city_28', 'city_30', 'city_31', 'city_33', 'city_36', 'city_37', 'city_39', 'city_40', 'city_41', 'city_42',
 'city_43', 'city_44', 'city_45', 'city_46', 'city_48', 'city_50', 'city_53', 'city_54', 'city_55', 'city_57',
 'city_59', 'city_62', 'city_64', 'city_65', 'city_67', 'city_69', 'city_70', 'city_71', 'city_72', 'city_73',
 'city_74', 'city_75', 'city_76', 'city_77', 'city_78', 'city_79', 'city_80', 'city_81', 'city_82', 'city_83',
 'city_84', 'city_89', 'city_90', 'city_91', 'city_93', 'city_94', 'city_97', 'city_98', 'city_99', 'city_100',
 'city_101', 'city_102', 'city_104', 'city_105', 'city_106', 'city_107', 'city_109', 'city_111', 'city_114',
 'city_115', 'city_116', 'city_117', 'city_118', 'city_120', 'city_123', 'city_127', 'city_128', 'city_129',
 'city_131', 'city_133', 'city_134', 'city_136', 'city_138', 'city_139', 'city_140', 'city_141', 'city_142',
 'city_143', 'city_144', 'city_145', 'city_146', 'city_149', 'city_150', 'city_152', 'city_157', 'city_158',
 'city_159', 'city_160', 'city_162', 'city_165', 'city_166', 'city_167', 'city_171', 'city_173', 'city_175',
 'city_176', 'city_179']
)
        relevent_experience = st.selectbox("Relevant Experience", options=['Has relevent experience', 'No relevent experience'])
        enrolled_university = st.selectbox("Type of Current University Enrollment", options=['no_enrollment', 'Full time course', 'Part time course'])
        education_level = st.selectbox("Highest Education Level", options=['Graduate', 'Masters', 'High School', 'Phd', 'Primary School'])
        major_discipline = st.selectbox("Major Discipline", options=['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major', 'Other'])

    with col2:
        experience_bin = st.selectbox("Total Years of Experience", options=['<1', '1-5', '6-10', '11-15', '16-20', '>20'])
        company_size = st.selectbox("Previous Company Size", options=['missing', '<10','10-49','50-99','100-500','500-999','1000-4999','5000-9999','>10000'])
        company_type = st.selectbox("Previous Company Type", options=['Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Public Sector', 'NGO', 'Other'])
        last_new_job_bin = st.selectbox("Years Spent in Previous Job", options=['<=1', '2-3', '>=4'])
        city_development_index = st.slider("City Development Index (CDI):", min_value=0.44, max_value=0.95, value=100.0, step=0.01)

    input_data = pd.DataFrame({
        'city': [city],
        'relevent_experience': [relevent_experience],
        'enrolled_university': [enrolled_university],
        'education_level': [education_level],
        'major_discipline': [major_discipline],
        'experience_bin': [experience_bin],
        'company_size': [company_size],
        'company_type': [company_type],
        'last_new_job_bin': [last_new_job_bin],
        'city_development_index': [float(city_development_index)],
    })

    if st.button("Predict 🔍"):
        try:
            prediction = model.predict_proba(input_data)[0][1]
            fig = create_gauge_chart(prediction)
            st.plotly_chart(fig, use_container_width=True)

            if prediction > 0.41:
                leave_risk = "🚫 Likely to Leave"
                bg_color = "rgba(255, 99, 71, 0.1)"
            else:
                leave_risk = "✅ Likely to Stay"
                bg_color = "rgba(144, 238, 144, 0.1)"

            st.markdown(
                f"""
                <div class="metric-container" style="background-color: {bg_color};">
                    <h3>{leave_risk}</h3>
                    <h4>Probability: {prediction:.2%}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Prediction error: {e}")
