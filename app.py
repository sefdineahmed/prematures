import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Prédiction Accouchement Prématuré",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLE PERSONNALISÉ (CORRECTION DU TYPEERROR) ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
        border: 1px solid #e9ecef;
    }
    div[data-testid="stSidebar"] {
        background-color: #1a3a5a;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #e63946;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True) # Paramètre corrigé ici

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
    page = st.radio("Menu", ["Tableau de Bord", "Diagnostic Patient", "Analyse de Groupe"])
    st.markdown("---")
    if model:
        st.success("✅ Modèle chargé")
    else:
        st.error("❌ Modèle non détecté")
    st.info("Outil basé sur l'analyse de 390 dossiers cliniques.")

# --- PAGE 1 : TABLEAU DE BORD ---
if page == "Tableau de Bord":
    st.title("📊 Statistiques et Facteurs de Risque")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Précision Globale", "81%")
    col2.metric("Sensibilité (Recall)", "85%")
    col3.metric("Patients Étudiés", "390")
    col4.metric("Variables Clés", "12")

    st.markdown("---")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Importance des Variables Cliniques")
        # Données de synthèse de l'étude
        feat_df = pd.DataFrame({
            'Variable': ['AGE', 'GEST', 'EFFACE', 'DILATE', 'MEMBRAN', 'GRAVID'],
            'Importance': [0.18, 0.16, 0.15, 0.10, 0.08, 0.07]
        }).sort_values('Importance')
        fig = px.bar(feat_df, x='Importance', y='Variable', orientation='h', 
                     color='Importance', color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        st.subheader("Répartition de la Population")
        fig_pie = px.pie(names=['Accouchement à Terme', 'Accouchement Prématuré'], 
                         values=[23, 55], hole=0.4,
                         color_discrete_sequence=['#457b9d', '#e63946'])
        st.plotly_chart(fig_pie, use_container_width=True)

# --- PAGE 2 : DIAGNOSTIC INDIVIDUEL ---
elif page == "Diagnostic Patient":
    st.title("🩺 Assistant de Diagnostic Individuel")
    
    if not model:
        st.error("Erreur : Le modèle prédictif est manquant.")
    else:
        with st.form("medical_form"):
            st.markdown("##### 1. Informations de Base")
            c1, c2, c3 = st.columns(3)
            age = c1.number_input("Âge de la patiente", 15, 55, 28)
            gest = c2.number_input("Âge Gestationnel (semaines)", 20, 45, 32)
            gemel = c3.selectbox("Grossesse Multiple", [1, 2], format_func=lambda x: "Simple" if x==1 else "Multiple")

            st.markdown("##### 2. Examen Clinique")
            c4, c5, c6 = st.columns(3)
            dilate = c4.number_input("Dilatation du col (cm)", 0.0, 10.0, 0.0)
            efface = c5.slider("Effacement du col (%)", 0, 100, 20)
            consis = c6.selectbox("Consistance du col", [1, 2, 3], format_func=lambda x: ["Mou", "Moyen", "Ferme"][x-1])

            st.markdown("##### 3. Symptômes et Antécédents")
            c7, c8, c9 = st.columns(3)
            contr = c7.selectbox("Contractions", [1, 2], format_func=lambda x: "Présentes" if x==1 else "Absentes")
            membran = c8.selectbox("État des membranes", [1, 2, 3], format_func=lambda x: ["Rompues", "Intactes", "Incertain"][x-1])
            diab = c9.selectbox("Diabète", [1, 2, 9], format_func=lambda x: {1:"Oui", 2:"Non", 9:"Inconnu"}[x])

            c10, c11, c12 = st.columns(3)
            gravid = c10.number_input("Gestité (Total grossesses)", 1, 15, 1)
            parit = c11.number_input("Parité (Accouchements à terme)", 0, 15, 0)
            transf = c12.selectbox("Transfert Spécialisé", [1, 2], format_func=lambda x: "Oui" if x==1 else "Non")

            submitted = st.form_submit_button("Analyser le Risque")

        if submitted:
            # Reconstruction du vecteur d'entrée selon l'ordre du modèle
            input_data = np.array([[gest, dilate, efface, consis, contr, membran, 
                                    age, gravid, parit, diab, transf, gemel]])
            
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            st.divider()
            if prediction == 1:
                st.error(f"### 🚨 Alerte : Risque de Prématurité Élevé ({probability:.1%})")
                st.warning("Action recommandée : Surveillance clinique immédiate.")
            else:
                st.success(f"### ✅ Risque de Prématurité Faible ({probability:.1%})")
                st.info("Le patient ne présente pas de signes de travail prématuré imminent selon les critères saisis.")

 # --- PAGE 3 : ANALYSE DE GROUPE (RÉALIGNEMENT AUTOMATIQUE) ---
elif page == "Analyse de Groupe":
    st.title("📂 Analyse par Lot (CSV)")
    
    up_file = st.file_uploader("Importer le fichier des patientes", type="csv")
    
    if up_file and model:
        try:
            # 1. Lecture du fichier
            df_input = pd.read_csv(up_file, sep=None, engine='python')
            
            # 2. Nettoyage des noms de colonnes (Majuscules et sans espaces)
            df_input.columns = df_input.columns.str.strip().upper()
            
            # 3. RÉALIGNEMENT DYNAMIQUE
            # On récupère l'ordre exact que le modèle a appris lors du "fit"
            if hasattr(model, 'feature_names_in_'):
                expected_order = list(model.feature_names_in_)
            else:
                # Si le modèle n'a pas enregistré les noms, on utilise l'ordre manuel
                expected_order = ['GEST', 'DILATE', 'EFFACE', 'CONSIS', 'CONTR', 'MEMBRAN', 
                                  'AGE', 'GRAVID', 'PARIT', 'DIAB', 'TRANSF', 'GEMEL']
            
            # On vérifie si toutes les colonnes requises sont là
            missing = [c for c in expected_order if c not in df_input.columns]
            
            if missing:
                st.error(f"Il manque des colonnes dans votre fichier : {missing}")
            else:
                # --- LA MAGIE EST ICI ---
                # On crée une copie avec l'ordre EXACT attendu par le modèle
                df_for_model = df_input[expected_order]
                
                # 4. Prédiction
                preds = model.predict(df_for_model)
                probs = model.predict_proba(df_for_model)[:, 1]
                
                # 5. Affichage des résultats originaux avec le verdict
                df_input['DIAGNOSTIC'] = ["🚨 RISQUE" if p == 1 else "✅ STABLE" for p in preds]
                df_input['PROBABILITÉ (%)'] = (probs * 100).round(2)
                
                st.success("✅ Analyse terminée avec succès !")
                st.dataframe(df_input.style.background_gradient(subset=['PROBABILITÉ (%)'], cmap='YlOrRd'))
                
                # Téléchargement
                csv = df_input.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Télécharger le rapport", csv, "resultats.csv", "text/csv")
                
        except Exception as e:
            st.error(f"Erreur lors de l'analyse : {e}")
