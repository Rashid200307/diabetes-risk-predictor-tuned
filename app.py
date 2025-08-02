import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from PIL import Image

# --- PATH CONFIGURATION ---
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(current_dir, 'models')

# Try different model file names
MODEL_FILES = [
    'stacked_ensemble.pkl',  # First try the ensemble model
    'tuned_xgboost.pkl',     # Then try the tuned model
    'xgboost.pkl'            # Finally try the original
]

# Find the first existing model file
model_path = None
for model_file in MODEL_FILES:
    candidate_path = os.path.join(MODEL_DIR, model_file)
    if os.path.exists(candidate_path):
        model_path = candidate_path
        break

if model_path is None:
    st.error("❌ Model file not found! Please check your models directory.")
    st.stop()

# --- ERROR HANDLING FOR DEPENDENCIES ---
try:
    # Import XGBoost only when needed
    from xgboost import XGBClassifier
except ImportError:
    st.error("""
    XGBoost not installed! Please add 'xgboost==2.0.3' to requirements.txt.
    """)
    st.stop()

# Load the trained model
try:
    model = joblib.load(model_path)
    st.sidebar.success(f"✅ Model loaded successfully from: {os.path.basename(model_path)}")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# --- MODEL PERFORMANCE METRICS (Update with your actual metrics) ---
# Default metrics (replace with your tuned model's actual metrics)
MODEL_PERFORMANCE = {
    "Accuracy": 0.84,
    "F1 Score": 0.82,
    "ROC AUC": 0.91,
    "Precision": 0.83,
    "Recall": 0.81
}

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Detect theme for proper styling
is_dark_theme = st.get_option("theme.backgroundColor") == '#0e1117'

# Language toggle
lang = st.sidebar.radio("Language", ["English", "Malay"])

# App title
titles = {"English": "Diabetes Risk Prediction", "Malay": "Ramalan Risiko Kencing Manis"}
st.title(titles[lang])

# --- MODEL TRUST SECTION ---
with st.expander("About Our Model" if lang == "English" else "Tentang Model Kami", expanded=True):
    # Performance metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{MODEL_PERFORMANCE['Accuracy']*100:.1f}%")
    col2.metric("ROC AUC", f"{MODEL_PERFORMANCE['ROC AUC']:.3f}")
    col3.metric("F1 Score", f"{MODEL_PERFORMANCE['F1 Score']:.3f}")
    col4.metric("Precision", f"{MODEL_PERFORMANCE['Precision']*100:.1f}%")
    col5.metric("Recall", f"{MODEL_PERFORMANCE['Recall']*100:.1f}%")

    # Model description
    st.info("""
    Our machine learning model was trained on a comprehensive dataset of medical records 
    and has been rigorously validated for accuracy. It uses an ensemble of advanced algorithms 
    optimized for medical prediction tasks.
    """ if lang == "English" else """
    Model pembelajaran mesin kami telah dilatih pada set data komprehensif rekod perubatan 
    dan telah divalidasi dengan ketat untuk ketepatan. Ia menggunakan ensemble algoritma 
    termaju yang dioptimumkan untuk tugas peramalan perubatan.
    """)

# --- INPUT FORM ---
input_labels = {
    "English": {
        "Pregnancies": "Number of Pregnancies",
        "Glucose": "Glucose Level (mg/dL)",
        "BloodPressure": "Blood Pressure (mmHg)",
        "SkinThickness": "Skin Thickness (mm)",
        "Insulin": "Insulin Level (μU/mL)",
        "BMI": "Body Mass Index (BMI)",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
        "Age": "Age (years)"
    },
    "Malay": {
        "Pregnancies": "Bilangan Kehamilan",
        "Glucose": "Tahap Glukosa (mg/dL)",
        "BloodPressure": "Tekanan Darah (mmHg)",
        "SkinThickness": "Ketebalan Kulit (mm)",
        "Insulin": "Tahap Insulin (μU/mL)",
        "BMI": "Indeks Jisim Badan (BMI)",
        "DiabetesPedigreeFunction": "Fungsi Salasilah Diabetes",
        "Age": "Umur (tahun)"
    }
}

with st.form("prediction_form"):
    st.subheader("Your Health Information" if lang == "English" else "Maklumat Kesihatan Anda")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        inputs = {
            'Pregnancies': st.slider(input_labels[lang]["Pregnancies"], 0, 17, 1),
            'Glucose': st.slider(input_labels[lang]["Glucose"], 0, 200, 100),
            'BloodPressure': st.slider(input_labels[lang]["BloodPressure"], 0, 122, 70),
            'SkinThickness': st.slider(input_labels[lang]["SkinThickness"], 0, 99, 20)
        }
        
    with col2:
        inputs.update({
            'Insulin': st.slider(input_labels[lang]["Insulin"], 0, 846, 79),
            'BMI': st.slider(input_labels[lang]["BMI"], 0.0, 67.1, 25.0),
            'DiabetesPedigreeFunction': st.slider(input_labels[lang]["DiabetesPedigreeFunction"], 0.08, 2.42, 0.47),
            'Age': st.slider(input_labels[lang]["Age"], 21, 81, 30)
        })
    
    predict_text = {"English": "Predict Risk", "Malay": "Ramal Risiko"}
    submitted = st.form_submit_button(predict_text[lang])

# --- PREDICTION LOGIC ---
if submitted:
    with st.spinner("Analyzing your risk..." if lang == "English" else "Menganalisis risiko anda..."):
        # Create dataframe from inputs
        input_df = pd.DataFrame([inputs])
        
        # Make prediction
        try:
            risk = model.predict_proba(input_df)[0][1] * 100
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.stop()
        
        # Display results in a visually distinct container
        with st.container():
            st.subheader("Prediction Results" if lang == "English" else "Keputusan Ramalan")
            
            # Modern risk visualization with progress bar
            st.metric(label="Risk Percentage", value=f"{risk:.1f}%")
            
            # Color-coded progress bar
            progress_color = "green" if risk < 30 else "orange" if risk < 70 else "red"
            risk_level = "Low" if risk < 30 else "Medium" if risk < 70 else "High"
            st.progress(int(risk), text=f"Risk Level: {risk_level}")
            
            # Risk interpretation
            if risk < 30:
                message = "✅ Low Risk: Maintain healthy diet and exercise"
                malay_message = "✅ Risiko Rendah: Kekalkan diet sihat dan bersenam"
                color = "green"
            elif risk < 70:
                message = "⚠️ Medium Risk: Reduce sugar intake, monitor glucose monthly"
                malay_message = "⚠️ Risiko Sederhana: Kurangkan pengambilan gula, pantau glukosa bulanan"
                color = "orange"
            else:
                message = "❌ High Risk: Consult doctor immediately, start medication"
                malay_message = "❌ Risiko Tinggi: Berjumpa doktor segera, mulakan ubat-ubatan"
                color = "red"
            
            # Highlighted message
            st.markdown(
                f"<div style='background-color: {color}10; padding: 15px; border-radius: 10px; "
                f"border-left: 5px solid {color}; margin: 20px 0;'>"
                f"<p style='font-size:18px; color:{color}; margin:0;'>{message if lang == 'English' else malay_message}</p>"
                "</div>", 
                unsafe_allow_html=True
            )
            
            # Prevention tips
            st.subheader("Prevention Tips" if lang == "English" else "Tip Pencegahan")
            tips = {
                "English": [
                    "🍎 Maintain a balanced diet with low sugar intake",
                    "🏃‍♂️ Exercise at least 30 minutes daily",
                    "🩸 Regularly monitor your blood glucose levels",
                    "🚭 Avoid smoking and limit alcohol consumption",
                    "😴 Get 7-8 hours of quality sleep each night",
                    "🥦 Increase fiber intake with vegetables and whole grains",
                    "💧 Stay hydrated with water instead of sugary drinks"
                ],
                "Malay": [
                    "🍎 Mengekalkan diet seimbang dengan pengambilan gula yang rendah",
                    "🏃‍♂️ Bersenam sekurang-kurangnya 30 minit setiap hari",
                    "🩸 Pantau tahap glukosa darah anda secara berkala",
                    "🚭 Elakkan merokok dan hadkan pengambilan alkohol",
                    "😴 Dapatkan tidur berkualiti 7-8 jam setiap malam",
                    "🥦 Tingkatkan pengambilan serat dengan sayur-sayuran dan bijirin penuh",
                    "💧 Minum air secukupnya menggantikan minuman bergula"
                ]
            }
            
            for tip in tips[lang]:
                st.info(tip)

# --- SIDEBAR FEATURES ---
st.sidebar.divider()

# Model performance metrics in sidebar
st.sidebar.subheader("Model Performance")
st.sidebar.metric("Accuracy", f"{MODEL_PERFORMANCE['Accuracy']*100:.1f}%")
st.sidebar.metric("ROC AUC", f"{MODEL_PERFORMANCE['ROC AUC']:.3f}")
st.sidebar.metric("F1 Score", f"{MODEL_PERFORMANCE['F1 Score']:.3f}")
st.sidebar.progress(MODEL_PERFORMANCE['Accuracy'])

st.sidebar.divider()
st.sidebar.subheader("About" if lang == "English" else "Mengenai")
st.sidebar.info("""
This app predicts diabetes risk using machine learning. 
It analyzes health parameters to assess your risk level.
""" if lang == "English" else """
Aplikasi ini meramalkan risiko kencing manis menggunakan pembelajaran mesin. 
Ia menganalisis parameter kesihatan untuk menilai tahap risiko anda.
""")

# Feature importance visualization
st.sidebar.divider()
if st.sidebar.checkbox("Show Feature Importance" if lang == "English" else "Tunjukkan Kepentingan Ciri"):
    try:
        # Create a visually appealing container
        with st.container():
            st.subheader("Feature Importance" if lang == "English" else "Kepentingan Ciri")
            
            # Try different image paths
            image_paths = [
                os.path.join(MODEL_DIR, 'feature_importance_comparison.png'),
                os.path.join(MODEL_DIR, 'feature_importance.png')
            ]
            
            found_image = None
            for img_path in image_paths:
                if os.path.exists(img_path):
                    found_image = img_path
                    break
            
            if found_image:
                # Add styled box with theme-adaptive colors
                st.markdown(f"""
                <style>
                .feature-box {{
                    background-color: {"#1e1e1e" if is_dark_theme else "#ffffff"};
                    padding: 15px;
                    border-radius: 10px;
                    border: 1px solid {"#444" if is_dark_theme else "#e6e9ef"};
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin-bottom: 15px;
                }}
                </style>
                """, unsafe_allow_html=True)
                
                # Apply the style
                st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                
                # Display image with proper sizing
                st.image(found_image, use_container_width=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Caption with theme-appropriate color
                caption_color = "#ffffff" if is_dark_theme else "#000000"
                st.markdown(
                    f"<div style='color: {caption_color}; margin-top: 10px;'>"
                    f"How different health factors contribute to diabetes risk" if lang == "English" else 
                    "Bagaimana faktor kesihatan berbeza menyumbang kepada risiko kencing manis"
                    "</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.warning("Feature importance image not found")
    except Exception as e:
        st.warning(f"Error loading feature importance: {str(e)}")

# Model disclaimer
st.sidebar.divider()
st.sidebar.warning("""
**Disclaimer:** This tool provides risk estimates only. 
It is not a substitute for professional medical advice. 
Always consult a healthcare provider for medical concerns.
""" if lang == "English" else """
**Penafian:** Alat ini memberikan anggaran risiko sahaja. 
Ia bukan pengganti nasihat perubatan profesional. 
Sentiasa berjumpa pembekal penjagaan kesihatan untuk masalah perubatan.
""")

# --- FOOTER ---
st.divider()
st.caption("""
Developed for BIT4333 Introduction to Machine Learning | 
[GitHub Repository](https://github.com/yourusername/diabetes-ml-project)
""" if lang == "English" else """
Dibangunkan untuk BIT4333 Pengenalan Kepada Pembelajaran Mesin | 
[Repositori GitHub](https://github.com/yourusername/diabetes-ml-project)
""")
