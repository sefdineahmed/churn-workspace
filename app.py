import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="ChurnPredict Pro | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk { background-color: #ffebee; border-left: 5px solid #d32f2f; }
    .medium-risk { background-color: #fff3e0; border-left: 5px solid #f57c00; }
    .low-risk { background-color: #e8f5e9; border-left: 5px solid #388e3c; }
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DES ARTEFACTS DU MODÈLE ---
@st.cache_resource
def load_ml_assets():
    """
    Charge le modèle, le scaler et la liste des features.
    Utilise st.cache_resource pour éviter de recharger à chaque interaction.
    """
    try:
        model = joblib.load("rf_churn_model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")
        return model, scaler, features
    except Exception as e:
        st.error(f"❌ Erreur de chargement des modèles : {e}")
        return None, None, None

model, scaler, features = load_ml_assets()

# --- LOGIQUE MÉTIER : PRÉDICTIONS ET RECOMMANDATIONS ---
def get_recommendations(row, proba):
    """
    Génère des commentaires et recommandations basés sur le profil client.
    """
    recos = []
    comments = []
    
    if proba > 0.6:
        comments.append("⚠️ Risque de départ très élevé détecté.")
        if row['support_calls'] > 3:
            recos.append("📞 Action immédiate : Appel prioritaire du service fidélisation (trop d'appels support).")
        if row['payment_delay'] > 5:
            recos.append("💳 Proposer un plan de paiement ou une remise sur les arriérés.")
        recos.append("🎁 Offrir un bonus de data ou un surclassement temporaire.")
    elif proba > 0.4:
        comments.append("⚖️ Client indécis avec un risque modéré.")
        recos.append("📧 Envoyer une enquête de satisfaction personnalisée.")
        recos.append("🔄 Proposer un passage à un contrat long terme avec avantage.")
    else:
        comments.append("✅ Client stable et fidèle.")
        recos.append("⭐ Programme de parrainage : Proposer au client de parrainer un proche.")

    return " ".join(comments), recos

def process_predictions(df):
    """
    Prépare les données, effectue la prédiction et enrichit les résultats.
    """
    # Sauvegarde des colonnes originales pour le rapport final
    original_df = df.copy()
    
    # Prétraitement : One-hot encoding pour les variables catégorielles
    df_processed = pd.get_dummies(df)
    
    # Alignement avec les colonnes attendues par le modèle
    df_processed = df_processed.reindex(columns=features, fill_value=0)
    
    # Mise à l'échelle (Scaling)
    X_scaled = scaler.transform(df_processed)
    
    # Calcul des probabilités
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Construction des résultats
    results = []
    for i, proba in enumerate(probabilities):
        comment, recos = get_recommendations(original_df.iloc[i], proba)
        results.append({
            "Probabilité Churn": f"{proba:.1%}",
            "Niveau de Risque": "Élevé" if proba > 0.6 else "Modéré" if proba > 0.4 else "Faible",
            "Commentaire": comment,
            "Recommandations": " | ".join(recos),
            "raw_proba": proba
        })
    
    return pd.concat([original_df, pd.DataFrame(results)], axis=1)

# --- INTERFACE UTILISATEUR (UI) ---

# Barre latérale
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Menu Principal")
    app_mode = st.radio("Navigation", ["🏠 Accueil & Stats", "👤 Saisie Individuelle", "📂 Import Batch"])
    st.markdown("---")
    st.info("💡 **Astuce** : Importez un fichier CSV ou Excel pour analyser plusieurs clients d'un coup.")

# --- MODE ACCUEIL & STATS ---
if app_mode == "🏠 Accueil & Stats":
    st.title("📊 Tableau de Bord de Rétention Client")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Modèle Actif", "Random Forest v1.0")
    with col2:
        st.metric("Précision (Test)", "87.5%")
    with col3:
        st.metric("Dernière Mise à Jour", datetime.now().strftime("%d/%m/%Y"))

    st.markdown("""
    ### Bienvenue sur ChurnPredict Pro
    Cette application utilise l'intelligence artificielle pour identifier les clients susceptibles de quitter vos services.
    
    **Fonctionnalités clés :**
    - **Saisie Individuelle** : Testez le profil d'un client spécifique.
    - **Import Batch** : Traitez des fichiers Excel, JSON, CSV ou TXT.
    - **Analytique** : Visualisez les facteurs de risque en temps réel.
    """)
    
    # Simulation d'un petit graphique de tendance
    chart_data = pd.DataFrame(np.random.randn(20, 2), columns=['Fidélité', 'Churn'])
    st.plotly_chart(px.line(chart_data, title="Évolution des tendances de Churn (Simulation)"), use_container_width=True)

# --- MODE SAISIE INDIVIDUELLE ---
elif app_mode == "👤 Saisie Individuelle":
    st.title("📝 Analyse de Profil Individuel")
    
    with st.form("individual_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Âge du client", 18, 100, 35)
            tenure = st.number_input("Ancienneté (mois)", 0, 120, 12)
            monthly = st.number_input("Charges mensuelles (€)", 0.0, 500.0, 59.90)
            contract = st.selectbox("Type de contrat", ["Monthly", "One year", "Two year"])
        with c2:
            usage = st.number_input("Consommation Data (GB)", 0.0, 1000.0, 10.0)
            calls = st.number_input("Appels au Support", 0, 20, 1)
            delay = st.number_input("Retards de paiement (jours)", 0, 30, 0)
            network = st.slider("Qualité Réseau perçue", 1, 5, 4)
            
        submit = st.form_submit_button("🚀 Lancer l'analyse")

    if submit:
        input_data = pd.DataFrame([{
            "age": age, "tenure_months": tenure, "monthly_charges": monthly,
            "data_usage_gb": usage, "voice_minutes": 300, "support_calls": calls,
            "network_quality": network, "payment_delay": delay, "auto_payment": 1,
            "contract_type": contract
        }])
        
        with st.spinner("Analyse en cours..."):
            res = process_predictions(input_data).iloc[0]
            
            # Affichage du résultat avec style
            risk_class = "high-risk" if res['raw_proba'] > 0.6 else "medium-risk" if res['raw_proba'] > 0.4 else "low-risk"
            
            st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h3>Résultat : Risque {res['Niveau de Risque']} ({res['Probabilité Churn']})</h3>
                    <p><b>Commentaire :</b> {res['Commentaire']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.subheader("📋 Recommandations Stratégiques")
            for reco in res['Recommandations'].split(" | "):
                st.write(f"- {reco}")

# --- MODE IMPORT BATCH ---
elif app_mode == "📂 Import Batch":
    st.title("📂 Traitement de Fichiers en Masse")
    
    uploaded_file = st.file_uploader(
        "Déposez votre fichier ici", 
        type=["csv", "xlsx", "json", "txt"],
        help="Formats supportés : CSV, Excel, JSON, TXT (séparé par des virgules)"
    )
    
    if uploaded_file:
        try:
            # Lecture flexible selon l'extension
            ext = uploaded_file.name.split('.')[-1]
            if ext == 'csv': df = pd.read_csv(uploaded_file)
            elif ext == 'xlsx': df = pd.read_excel(uploaded_file)
            elif ext == 'json': df = pd.read_json(uploaded_file)
            elif ext == 'txt': df = pd.read_csv(uploaded_file) # On suppose format CSV pour TXT
            
            st.success(f"✅ Fichier '{uploaded_file.name}' chargé avec succès ({len(df)} lignes).")
            
            if st.button("🔍 Lancer les prédictions sur tout le fichier"):
                with st.spinner("Traitement batch en cours..."):
                    results_df = process_predictions(df)
                    
                    # Affichage des statistiques globales
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Clients", len(results_df))
                    c2.metric("Risques Élevés", len(results_df[results_df['raw_proba'] > 0.6]))
                    c3.metric("Moyenne Probabilité", f"{results_df['raw_proba'].mean():.1%}")
                    
                    # Tableau de bord interactif
                    st.subheader("📑 Résultats Détaillés")
                    st.dataframe(results_df.drop(columns=['raw_proba']), use_container_width=True)
                    
                    # Export
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Télécharger le rapport complet (CSV)", csv, "predictions_churn.csv", "text/csv")
                    
                    # Graphique de répartition
                    fig = px.pie(results_df, names='Niveau de Risque', title="Répartition du Risque de Churn",
                                color='Niveau de Risque',
                                color_discrete_map={'Élevé':'#d32f2f', 'Modéré':'#f57c00', 'Faible':'#388e3c'})
                    st.plotly_chart(fig)
                    
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement : {e}")

# --- PIED DE PAGE ---
st.markdown("---")
st.markdown(f"© {datetime.now().year} - ChurnPredict Pro - Déployé avec succès sur GitHub")
