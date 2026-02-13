# ============================================================
# 1️⃣ IMPORT DES LIBRAIRIES
# ============================================================
import streamlit as st                   # Streamlit pour dashboard
import pandas as pd                      # DataFrame
import numpy as np                       # Calcul numérique
import joblib                             # Charger modèle et scaler
import os                                # Vérifier les fichiers
from typing import List

# ============================================================
# 2️⃣ CHARGEMENT DES ARTEFACTS
# ============================================================
model_path = "rf_churn_model.pkl"
scaler_path = "scaler.pkl"
features_path = "features.pkl"

# Vérification existence fichiers
for path in [model_path, scaler_path, features_path]:
    if not os.path.exists(path):
        st.error(f"Fichier introuvable : {path}")

# Chargement du modèle, scaler et features
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
features = joblib.load(features_path)

THRESHOLD = 0.50  # Seuil optimisé pour augmenter le recall

# ============================================================
# 3️⃣ FONCTION DE PRÉDICTION
# ============================================================
def make_prediction(df: pd.DataFrame):
    """
    Prédiction du churn pour un DataFrame complet
    """
    # One-hot encoding
    df = pd.get_dummies(df)
    # Alignement avec features du modèle
    df = df.reindex(columns=features, fill_value=0)
    # Scaling
    X_scaled = scaler.transform(df)
    # Probabilités
    proba = model.predict_proba(X_scaled)[:, 1]
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
# 4️⃣ CONFIGURATION STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Telecom Churn Dashboard",
    layout="wide"
)

st.title("📊 Telecom Churn Prediction Dashboard")
st.markdown("""
Bienvenue dans le tableau de bord de prédiction du churn.
Vous pouvez **uploader un fichier** pour des prédictions batch,
ou **utiliser le formulaire** pour un client unique.
""")

# ============================================================
# 5️⃣ SIDEBAR
# ============================================================
st.sidebar.header("Options")
mode = st.sidebar.selectbox(
    "Sélectionnez le mode",
    ["Prédiction Individuelle", "Prédiction Batch (Fichier)"]
)

# ============================================================
# 6️⃣ MODE FORMULAIRE INDIVIDUEL
# ============================================================
if mode == "Prédiction Individuelle":
    st.subheader("📝 Formulaire Client")

    # Formulaire Streamlit
    with st.form(key="single_client_form"):
        # ⚠️ Veille à ce que type min_value et value soient cohérents
        age = st.number_input("Âge", min_value=18, max_value=100, value=30, step=1)
        tenure_months = st.number_input("Ancienneté (mois)", min_value=0, value=6, step=1)
        monthly_charges = st.number_input("Facture Mensuelle", min_value=0.0, value=75.0, step=0.5)
        data_usage_gb = st.number_input("Data Usage (GB)", min_value=0.0, value=5.0, step=0.1)
        voice_minutes = st.number_input("Minutes Vocales", min_value=0, value=300, step=1)
        support_calls = st.number_input("Appels Support", min_value=0, value=1, step=1)
        network_quality = st.slider("Qualité Réseau (1-5)", min_value=1, max_value=5, value=4)
        payment_delay = st.number_input("Retard Paiement", min_value=0, value=0, step=1)
        auto_payment = st.selectbox("Paiement Auto", [0, 1])
        contract_type = st.selectbox("Type de Contrat", ["Monthly", "One year", "Two year"])

        # Bouton de soumission
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
# 7️⃣ MODE BATCH (UPLOAD FICHIER)
# ============================================================
if mode == "Prédiction Batch (Fichier)":
    st.subheader("📂 Upload d'un fichier")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier (CSV, Excel, JSON, TXT)",
        type=["csv", "xlsx", "json", "txt"]
    )

    if uploaded_file is not None:
        try:
            # Lecture selon type
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
