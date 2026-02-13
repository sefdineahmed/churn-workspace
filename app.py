# ============================================================
# IMPORT DES LIBRAIRIES
# ============================================================
import streamlit as st                   # Streamlit pour le dashboard
import pandas as pd                      # Manipulation de DataFrame
import numpy as np                       # Calculs numériques
import joblib                             # Chargement du modèle
from io import StringIO                   # Pour lire les fichiers TXT/CSV
import os

# ============================================================
# CHARGEMENT DES ARTEFACTS
# ============================================================
model_path = "rf_churn_model.pkl"
scaler_path = "scaler.pkl"
features_path = "features.pkl"

# Vérifier que tous les fichiers existent
if not os.path.exists(model_path):
    st.error("Modèle introuvable !")
if not os.path.exists(scaler_path):
    st.error("Scaler introuvable !")
if not os.path.exists(features_path):
    st.error("Features introuvables !")

# Chargement des objets sauvegardés
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
features = joblib.load(features_path)

# Seuil optimisé
THRESHOLD = 0.40

# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================
def make_prediction(df: pd.DataFrame):
    """
    Fonction pour prédire le churn pour un DataFrame complet
    """
    # One-hot encoding des colonnes catégorielles
    df = pd.get_dummies(df)
    # Alignement avec les colonnes du modèle
    df = df.reindex(columns=features, fill_value=0)
    # Scaling
    X_scaled = scaler.transform(df)
    # Probabilités
    proba = model.predict_proba(X_scaled)[:, 1]
    # Résultats
    results = []
    for p in proba:
        prediction = int(p >= THRESHOLD)
        risk = "High" if p >= 0.6 else "Medium" if p >= 0.4 else "Low"
        results.append({
            "churn_probability": round(float(p), 3),
            "churn_prediction": prediction,
            "risk_level": risk
        })
    return results

# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(
    page_title="Telecom Churn Dashboard",
    layout="wide"
)

# Titre principal
st.title("📊 Telecom Churn Prediction Dashboard")

# Description
st.markdown("""
Bienvenue dans le tableau de bord de prédiction du churn.
Vous pouvez soit **uploader un fichier** pour des prédictions batch, soit **utiliser le formulaire** pour un client unique.
""")

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Options")
mode = st.sidebar.selectbox(
    "Sélectionnez le mode",
    ["Prédiction Individuelle", "Prédiction Batch (Fichier)"]
)

# ============================================================
# MODE FORMULAIRE INDIVIDUEL
# ============================================================
if mode == "Prédiction Individuelle":
    st.subheader("📝 Formulaire Client")

    # Création du formulaire
    with st.form(key="single_client_form"):
        age = st.number_input("Âge", min_value=18, max_value=100, value=30)
        tenure_months = st.number_input("Ancienneté (mois)", min_value=0, value=6)
        monthly_charges = st.number_input("Facture Mensuelle", min_value=0.0, value=75.0)
        data_usage_gb = st.number_input("Data Usage (GB)", min_value=0.0, value=5.0)
        voice_minutes = st.number_input("Minutes Vocales", min_value=0.0, value=300)
        support_calls = st.number_input("Appels Support", min_value=0, value=1)
        network_quality = st.slider("Qualité Réseau (1-5)", min_value=1, max_value=5, value=4)
        payment_delay = st.number_input("Retard Paiement", min_value=0, value=0)
        auto_payment = st.selectbox("Paiement Auto", [0, 1])
        contract_type = st.selectbox("Type de Contrat", ["Monthly", "One year", "Two year"])
        submit_button = st.form_submit_button(label="Prédire")

    # Si formulaire soumis
    if submit_button:
        client_df = pd.DataFrame([{
            "age": age,
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "data_usage_gb": data_usage_gb,
            "voice_minutes": voice_minutes,
            "support_calls": support_calls,
            "network_quality": network_quality,
            "payment_delay": payment_delay,
            "auto_payment": auto_payment,
            "contract_type": contract_type
        }])
        result = make_prediction(client_df)[0]
        st.success("✅ Prédiction effectuée !")
        st.json(result)

# ============================================================
# MODE BATCH (FICHIER)
# ============================================================
if mode == "Prédiction Batch (Fichier)":
    st.subheader("📂 Upload d'un fichier")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier (CSV, Excel, JSON, TXT)",
        type=["csv", "xlsx", "json", "txt"]
    )

    if uploaded_file is not None:
        try:
            # Lecture selon type de fichier
            if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".txt"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file)
            else:
                st.error("Format non supporté")
                st.stop()

            st.write("Aperçu des données uploadées :")
            st.dataframe(df.head())

            # Prédictions
            results = make_prediction(df)
            results_df = pd.DataFrame(results)
            st.success("✅ Prédictions effectuées !")
            st.dataframe(results_df)

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
