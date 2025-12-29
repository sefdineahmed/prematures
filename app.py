import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Prévention Prématurité Pro",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- STYLE PERSONNALISÉ (CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .sidebar .sidebar-content { background-image: linear-gradient(#2e7d32, #1b5e20); color: white; }
    </style>
    """, unsafe_allow_stdio=True)

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model():
    with open("prematurite_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except:
    st.error("⚠️ Fichier 'prematurite_model.pkl' introuvable. Veuillez d'abord entraîner le modèle.")
    st.stop()

# --- BARRE LATÉRALE (NAVIGATION) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=100)
    st.title("Menu Principal")
    page = st.radio("Aller vers :", ["Tableau de Bord", "Diagnostic Patient", "Analyse de Groupe (Batch)"])
    st.markdown("---")
    st.info("Outil d'aide à la décision clinique basé sur 390 dossiers d'études.")

# --- PAGE 1 : TABLEAU DE BORD (VISUALISATION) ---
if page == "Tableau de Bord":
    st.title("📊 Tableau de Bord Analytique")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Précision Modèle", "81%", "+2% (Optimisé)")
    col2.metric("Sensibilité (Recall)", "85%", "Majeure")
    col3.metric("Patients Étudiés", "390", "Total")
    col4.metric("Facteur Clé", "ÂGE", "N°1")

    st.markdown("### Répartition des Risques par Facteurs")
    c1, c2 = st.columns(2)
    
    # Graphique fictif pour illustrer (à remplacer par vos données réelles si chargées)
    df_illus = pd.DataFrame({
        'Facteurs': ['Âge', 'Gest. Semaines', 'Effacement %', 'Dilatation cm', 'Membranes'],
        'Importance': [0.18, 0.16, 0.15, 0.10, 0.08]
    })
    
    fig_imp = px.bar(df_illus, x='Importance', y='Facteurs', orientation='h', 
                     title="Top 5 des prédicteurs cliniques", color='Importance', color_continuous_scale='Viridis')
    c1.plotly_chart(fig_imp, use_container_width=True)

    fig_pie = px.pie(values=[77, 23], names=['Succès Prédit', 'Erreur'], hole=0.5, title="Fiabilité globale")
    c2.plotly_chart(fig_pie, use_container_width=True)

# --- PAGE 2 : DIAGNOSTIC INDIVIDUEL ---
elif page == "Diagnostic Patient":
    st.title("🩺 Diagnostic Individuel")
    st.subheader("Saisir les paramètres cliniques")

    with st.form("diag_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Âge de la patiente", 15, 50, 28)
        gest = c2.number_input("Âge Gestationnel (semaines)", 20.0, 42.0, 32.0)
        dilate = c3.number_input("Dilatation (cm)", 0.0, 10.0, 1.0)
        
        c4, c5, c6 = st.columns(3)
        efface = c4.slider("Effacement (%)", 0, 100, 30)
        consis = c5.selectbox("Consistance du col", [1, 2, 3], format_func=lambda x: ["Mou", "Moyen", "Ferme"][x-1])
        membran = c6.selectbox("Membranes", [1, 2, 3], format_func=lambda x: ["Rompues", "Intactes", "Incertain"][x-1])
        
        # Autres variables masquées par défaut ou simplifiées
        submit = st.form_submit_button("Lancer l'Analyse")

    if submit:
        # Préparation du vecteur (respecter l'ordre des 12 variables)
        # GEST, DILATE, EFFACE, CONSIS, CONTR, MEMBRAN, AGE, GRAVID, PARIT, DIAB, TRANSF, GEMEL
        input_data = np.array([[gest, dilate, efface, consis, 1, membran, age, 1, 0, 2, 2, 1]])
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(f"⚠️ RISQUE ÉLEVÉ DÉTECTÉ (Probabilité : {prob:.2%})")
            st.warning("Recommandation : Surveillance accrue et hospitalisation possible.")
        else:
            st.success(f"✅ RISQUE FAIBLE (Probabilité : {prob:.2%})")
            st.balloons()

# --- PAGE 3 : ANALYSE DE GROUPE (BATCH) ---
elif page == "Analyse de Groupe (Batch)":
    st.title("📂 Analyse Multi-Patients")
    st.write("Téléchargez un fichier CSV contenant les données de plusieurs patientes pour un diagnostic groupé.")

    uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
    
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write("Aperçu des données :", df_batch.head())
        
        if st.button("Analyser le fichier"):
            # Supposons que le CSV a les colonnes dans le bon ordre
            preds = model.predict(df_batch)
            probs = model.predict_proba(df_batch)[:, 1]
            
            df_batch['RESULTAT'] = ["Positif" if p == 1 else "Négatif" for p in preds]
            df_batch['SCORE_RISQUE'] = probs
            
            st.divider()
            st.subheader("Résultats de l'analyse")
            st.dataframe(df_batch.style.background_gradient(subset=['SCORE_RISQUE'], cmap='Reds'))
            
            # Graphique récapitulatif
            fig_res = px.histogram(df_batch, x="RESULTAT", color="RESULTAT", title="Distribution des diagnostics")
            st.plotly_chart(fig_res)
