import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Prévention Prématurité Pro",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLE PERSONNALISÉ (CSS CORRIGÉ) ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 10px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
    }
    div[data-testid="stSidebar"] {
        background-color: #1e3d59;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff6e40;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True) # Correction ici : html au lieu de stdio

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.title("🏥 Navigation")
    page = st.radio("Sélectionnez une section :", 
                    ["Tableau de Bord", "Diagnostic Patient", "Analyse de Groupe"])
    st.markdown("---")
    st.write("**Statut du modèle :**")
    if model:
        st.success("✅ Modèle chargé")
    else:
        st.error("❌ Modèle non trouvé")

# --- PAGE 1 : TABLEAU DE BORD ---
if page == "Tableau de Bord":
    st.title("📊 Tableau de Bord Clinique")
    
    # Métriques principales
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Précision (Test)", "81%")
    m2.metric("Sensibilité", "85%")
    m3.metric("Échantillon Étude", "390")
    m4.metric("Variables", "12")

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Importance des Facteurs")
        # Données basées sur votre analyse précédente
        feat_data = pd.DataFrame({
            'Facteur': ['AGE', 'GEST', 'EFFACE', 'DILATE', 'MEMBRAN'],
            'Importance': [0.18, 0.16, 0.15, 0.10, 0.08]
        }).sort_values(by='Importance', ascending=True)
        fig = px.bar(feat_data, x='Importance', y='Facteur', orientation='h', 
                     color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribution des Risques")
        fig_pie = px.pie(names=['Accouchement à Terme', 'Prématurité'], 
                         values=[23, 55], hole=0.4,
                         color_discrete_sequence=['#1e3d59', '#ff6e40'])
        st.plotly_chart(fig_pie, use_container_width=True)

# --- PAGE 2 : DIAGNOSTIC INDIVIDUEL ---
elif page == "Diagnostic Patient":
    st.title("🩺 Assistant de Diagnostic Rapide")
    
    if not model:
        st.warning("Veuillez uploader le fichier 'prematurite_model.pkl' à la racine.")
    else:
        with st.form("patient_form"):
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Âge de la patiente", 15, 55, 28)
            gest = c2.number_input("Âge Gestationnel (semaines)", 20, 42, 32)
            dilate = c3.number_input("Dilatation du col (cm)", 0.0, 10.0, 0.0)
            
            c4, c5, c6 = st.columns(3)
            efface = c4.slider("Effacement (%)", 0, 100, 20)
            consis = c5.selectbox("Consistance", [1, 2, 3], format_func=lambda x: ["Mou", "Moyen", "Ferme"][x-1])
            membran = c6.selectbox("Membranes", [1, 2, 3], format_func=lambda x: ["Rompues", "Intactes", "Incertain"][x-1])
            
            # Variables secondaires (initialisées par défaut)
            contr = 1
            gravid = 1
            parit = 0
            diab = 2
            transf = 2
            gemel = 1

            submitted = st.form_submit_button("Calculer le Risque")
            
            if submitted:
                # Préparation des données (Ordre des 12 variables)
                input_array = np.array([[gest, dilate, efface, consis, contr, membran, 
                                         age, gravid, parit, diab, transf, gemel]])
                
                prediction = model.predict(input_array)[0]
                proba = model.predict_proba(input_array)[0][1]
                
                st.markdown("---")
                if prediction == 1:
                    st.error(f"### ⚠️ Risque de Prématurité Détecté : {proba:.1%}")
                    st.write("Le profil clinique présente des signes de travail prématuré imminent.")
                else:
                    st.success(f"### ✅ Risque de Prématurité Faible : {proba:.1%}")
                    st.write("Le profil clinique semble stable pour le moment.")

# --- PAGE 3 : ANALYSE DE GROUPE ---
elif page == "Analyse de Groupe":
    st.title("📂 Analyse Batch (CSV)")
    uploaded_file = st.file_uploader("Importer le fichier des patientes", type="csv")
    
    if uploaded_file and model:
        df = pd.read_csv(uploaded_file)
        # Supposons que le CSV contient les colonnes dans le bon ordre
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]
        
        df['Verdict'] = ["🚨 Risque" if p == 1 else "✅ Stable" for p in preds]
        df['Probabilité (%)'] = (probs * 100).round(2)
        
        st.write("### Résultats de l'analyse")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger les résultats", csv, "resultats_diagnostics.csv", "text/csv")
