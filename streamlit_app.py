# ============================================================
# 📊 TELECOM CHURN PREDICTION DASHBOARD
# ============================================================
# Application Streamlit pour la prédiction du churn client
# Auteur: Votre Nom
# Date: 2026
# Description: Dashboard interactif permettant de prédire le 
#              churn des clients télécom via formulaire ou 
#              import de fichiers (CSV, Excel, JSON, TXT)
# ============================================================

# ============================================================
# 1️⃣ IMPORTATION DES BIBLIOTHÈQUES
# ============================================================

import streamlit as st                    # Framework pour créer l'interface web
import pandas as pd                       # Manipulation de données tabulaires
import numpy as np                        # Calculs numériques et manipulation d'arrays
import joblib                             # Chargement des modèles ML sauvegardés
import os                                 # Opérations sur le système de fichiers
import plotly.express as px               # Visualisations interactives
import plotly.graph_objects as go         # Graphiques personnalisés Plotly
from datetime import datetime             # Manipulation de dates et heures
from typing import List, Dict, Tuple      # Annotations de types pour le code
import warnings                           # Gestion des avertissements
warnings.filterwarnings('ignore')         # Suppression des warnings non-critiques

# ============================================================
# 2️⃣ CONFIGURATION INITIALE DE STREAMLIT
# ============================================================

# Configuration de la page principale
st.set_page_config(
    page_title="📡 Telecom Churn Predictor",    # Titre dans l'onglet du navigateur
    page_icon="📊",                              # Icône de la page
    layout="wide",                               # Utilisation de toute la largeur
    initial_sidebar_state="expanded"             # Sidebar ouverte par défaut
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
    <style>
    /* Style pour les métriques principales */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Style pour les cartes de résultats */
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    /* Style pour les alertes de risque */
    .high-risk {
        background: #fee;
        border-left: 4px solid #f44336;
        padding: 15px;
        border-radius: 5px;
    }
    
    .medium-risk {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 15px;
        border-radius: 5px;
    }
    
    .low-risk {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 15px;
        border-radius: 5px;
    }
    
    /* Animation pour le header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Style pour les boutons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 25px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 3️⃣ CHARGEMENT DES ARTEFACTS ML
# ============================================================

@st.cache_resource  # Cache pour éviter de recharger à chaque interaction
def load_ml_artifacts():
    """
    Charge le modèle ML, le scaler et la liste des features
    
    Returns:
        tuple: (model, scaler, features) - Les artefacts ML chargés
    
    Raises:
        FileNotFoundError: Si un des fichiers est manquant
    """
    # Définition des chemins des fichiers
    model_path = "rf_churn_model.pkl"
    scaler_path = "scaler.pkl"
    features_path = "features.pkl"
    
    # Vérification de l'existence de tous les fichiers requis
    missing_files = []
    for path in [model_path, scaler_path, features_path]:
        if not os.path.exists(path):
            missing_files.append(path)
    
    # Affichage d'erreur si des fichiers manquent
    if missing_files:
        st.error(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        st.info("ℹ️ Assurez-vous que les fichiers .pkl sont dans le même répertoire que l'application")
        st.stop()
    
    # Chargement des artefacts
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)
        return model, scaler, features
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement des modèles: {str(e)}")
        st.stop()

# Chargement des artefacts au démarrage
model, scaler, features = load_ml_artifacts()

# Seuil de probabilité optimisé pour maximiser le recall
THRESHOLD = 0.50

# ============================================================
# 4️⃣ FONCTIONS UTILITAIRES
# ============================================================

def get_risk_color(risk_level: str) -> str:
    """
    Retourne une couleur en fonction du niveau de risque
    
    Args:
        risk_level (str): Niveau de risque ('High', 'Medium', 'Low')
    
    Returns:
        str: Code couleur hexadécimal
    """
    colors = {
        'High': '#f44336',      # Rouge
        'Medium': '#ff9800',    # Orange
        'Low': '#4caf50'        # Vert
    }
    return colors.get(risk_level, '#999999')

def make_prediction(df: pd.DataFrame) -> List[Dict]:
    """
    Effectue les prédictions de churn sur un DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données clients
    
    Returns:
        List[Dict]: Liste de dictionnaires avec les prédictions
                   Chaque dict contient: churn_probability, churn_prediction, risk_level
    """
    try:
        # Création d'une copie pour ne pas modifier l'original
        df_processed = df.copy()
        
        # One-hot encoding des variables catégorielles
        df_processed = pd.get_dummies(df_processed)
        
        # Alignement avec les features du modèle entraîné
        # Les colonnes manquantes sont remplies avec 0
        df_processed = df_processed.reindex(columns=features, fill_value=0)
        
        # Standardisation des features numériques
        X_scaled = scaler.transform(df_processed)
        
        # Prédiction des probabilités (colonne 1 = probabilité de churn)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Création des résultats pour chaque ligne
        results = []
        for prob in probabilities:
            # Prédiction binaire basée sur le seuil
            prediction = int(prob >= THRESHOLD)
            
            # Détermination du niveau de risque
            if prob >= 0.6:
                risk = "High"
            elif prob >= 0.4:
                risk = "Medium"
            else:
                risk = "Low"
            
            # Ajout du résultat
            results.append({
                "churn_probability": round(float(prob), 3),
                "churn_prediction": prediction,
                "risk_level": risk
            })
        
        return results
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
        return []

def generate_recommendations(prediction_result: Dict, client_data: Dict) -> List[str]:
    """
    Génère des recommandations personnalisées basées sur la prédiction
    
    Args:
        prediction_result (Dict): Résultat de la prédiction
        client_data (Dict): Données du client
    
    Returns:
        List[str]: Liste de recommandations
    """
    recommendations = []
    prob = prediction_result['churn_probability']
    risk = prediction_result['risk_level']
    
    # Recommandations basées sur le niveau de risque
    if risk == "High":
        recommendations.append("🚨 **Action Urgente**: Contactez ce client immédiatement")
        recommendations.append("💰 Proposez une offre promotionnelle personnalisée (-20% pendant 3 mois)")
        recommendations.append("🎁 Offrez un upgrade gratuit vers un forfait supérieur")
    
    elif risk == "Medium":
        recommendations.append("⚠️ **Surveillance Active**: Planifiez un appel de satisfaction")
        recommendations.append("📧 Envoyez une campagne email avec des offres exclusives")
        recommendations.append("🎯 Proposez des services additionnels gratuits (roaming, data extra)")
    
    else:
        recommendations.append("✅ **Client Stable**: Continuez l'engagement régulier")
        recommendations.append("🌟 Programmes de fidélité et récompenses")
        recommendations.append("📱 Invitez à parrainer d'autres clients (programme référent)")
    
    # Recommandations basées sur les caractéristiques spécifiques
    if client_data.get('support_calls', 0) > 3:
        recommendations.append("📞 **Problème détecté**: Nombre élevé d'appels support → Améliorer la qualité de service")
    
    if client_data.get('payment_delay', 0) > 0:
        recommendations.append("💳 **Paiement**: Retards détectés → Proposer un plan de paiement flexible")
    
    if client_data.get('contract_type') == 'Monthly':
        recommendations.append("📝 **Contrat**: Type mensuel → Encourager passage à contrat annuel avec bonus")
    
    if client_data.get('network_quality', 5) < 3:
        recommendations.append("📡 **Réseau**: Qualité faible → Vérifier et améliorer la couverture dans sa zone")
    
    if client_data.get('auto_payment', 1) == 0:
        recommendations.append("🔄 **Paiement Auto**: Non activé → Inciter avec 5% de réduction")
    
    if client_data.get('tenure_months', 0) < 6:
        recommendations.append("🆕 **Nouveau Client**: Ancienneté faible → Programme d'onboarding renforcé")
    
    return recommendations

def create_gauge_chart(probability: float) -> go.Figure:
    """
    Crée un graphique jauge pour visualiser la probabilité de churn
    
    Args:
        probability (float): Probabilité de churn (0-1)
    
    Returns:
        go.Figure: Figure Plotly avec le graphique jauge
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Probabilité de Churn (%)", 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#4caf50'},     # Vert - Faible risque
                {'range': [40, 60], 'color': '#ff9800'},    # Orange - Risque moyen
                {'range': [60, 100], 'color': '#f44336'}    # Rouge - Risque élevé
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_feature_importance_chart(client_data: Dict) -> go.Figure:
    """
    Crée un graphique montrant les features du client
    
    Args:
        client_data (Dict): Données du client
    
    Returns:
        go.Figure: Figure Plotly avec le graphique en barres
    """
    # Sélection des features numériques importantes
    features_to_show = {
        'Ancienneté (mois)': client_data.get('tenure_months', 0),
        'Facture Mensuelle': client_data.get('monthly_charges', 0),
        'Appels Support': client_data.get('support_calls', 0),
        'Qualité Réseau': client_data.get('network_quality', 0),
        'Retard Paiement': client_data.get('payment_delay', 0),
        'Data Usage (GB)': client_data.get('data_usage_gb', 0)
    }
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(features_to_show.keys()),
            y=list(features_to_show.values()),
            marker_color=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#fa709a', '#fee140']
        )
    ])
    
    fig.update_layout(
        title="Caractéristiques du Client",
        xaxis_title="Features",
        yaxis_title="Valeurs",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    return fig

# ============================================================
# 5️⃣ EN-TÊTE DE L'APPLICATION
# ============================================================

# Header principal avec gradient
st.markdown("""
    <div class="main-header">
        <h1>📡 Tableau de Bord de Prédiction de Churn Télécom</h1>
        <p style='font-size: 18px; margin-top: 10px;'>
            Analysez et anticipez le départ de vos clients grâce à l'Intelligence Artificielle
        </p>
    </div>
""", unsafe_allow_html=True)

# Description de l'application
st.markdown("""
### 🎯 Fonctionnalités

Cette application vous permet de :
- 📝 **Prédire individuellement** le risque de churn d'un client via un formulaire
- 📊 **Analyser en masse** des milliers de clients via import de fichier (CSV, Excel, JSON, TXT)
- 💡 **Recevoir des recommandations** personnalisées pour chaque client
- 📈 **Visualiser** les résultats avec des graphiques interactifs
""")

st.divider()

# ============================================================
# 6️⃣ BARRE LATÉRALE (SIDEBAR)
# ============================================================

with st.sidebar:
    # Logo et titre
    st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=Telecom+AI", 
             use_container_width=True)
    
    st.title("⚙️ Paramètres")
    
    # Sélection du mode d'utilisation
    st.subheader("Mode d'Analyse")
    mode = st.radio(
        "Choisissez votre méthode de prédiction:",
        ["🧍 Prédiction Individuelle", "📂 Prédiction Batch (Fichier)"],
        help="Sélectionnez si vous voulez analyser un seul client ou un fichier entier"
    )
    
    st.divider()
    
    # Informations sur le modèle
    st.subheader("📊 Informations Modèle")
    st.metric("Seuil de Décision", f"{THRESHOLD * 100}%")
    st.metric("Features Utilisées", len(features))
    
    st.divider()
    
    # Guide utilisateur
    st.subheader("📖 Guide Rapide")
    if "Individuelle" in mode:
        st.info("""
        **Prédiction Individuelle:**
        1. Remplissez le formulaire
        2. Cliquez sur 'Prédire'
        3. Consultez les résultats et recommandations
        """)
    else:
        st.info("""
        **Prédiction Batch:**
        1. Préparez votre fichier avec les colonnes requises
        2. Uploadez le fichier (CSV, Excel, JSON, TXT)
        3. Visualisez les prédictions
        4. Téléchargez les résultats
        """)
    
    st.divider()
    
    # Informations système
    st.subheader("ℹ️ À propos")
    st.caption(f"""
    **Version**: 1.0.0  
    **Dernière MAJ**: {datetime.now().strftime('%d/%m/%Y')}  
    **Modèle**: Random Forest  
    **Précision**: ~85%
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <small>Développé avec ❤️ par votre équipe Data Science</small>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# 7️⃣ MODE PRÉDICTION INDIVIDUELLE
# ============================================================

if "Individuelle" in mode:
    st.header("🧍 Prédiction pour un Client Unique")
    
    # Information sur les champs obligatoires
    st.info("ℹ️ Remplissez tous les champs ci-dessous pour obtenir une prédiction précise")
    
    # Création du formulaire dans des colonnes pour une meilleure présentation
    with st.form(key="single_client_form"):
        
        # Section 1: Informations Démographiques
        st.subheader("👤 Informations Démographiques")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input(
                "Âge du client",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
                help="Âge du client (18-100 ans)"
            )
        
        with col2:
            tenure_months = st.number_input(
                "Ancienneté (mois)",
                min_value=0,
                max_value=120,
                value=12,
                step=1,
                help="Nombre de mois depuis l'inscription du client"
            )
        
        st.divider()
        
        # Section 2: Informations Financières
        st.subheader("💰 Informations Financières")
        col3, col4 = st.columns(2)
        
        with col3:
            monthly_charges = st.number_input(
                "Facture Mensuelle (€)",
                min_value=0.0,
                max_value=500.0,
                value=75.0,
                step=0.5,
                help="Montant moyen de la facture mensuelle"
            )
        
        with col4:
            payment_delay = st.number_input(
                "Retards de Paiement",
                min_value=0,
                max_value=12,
                value=0,
                step=1,
                help="Nombre de retards de paiement dans les 12 derniers mois"
            )
        
        col5, col6 = st.columns(2)
        
        with col5:
            auto_payment = st.selectbox(
                "Paiement Automatique",
                options=[0, 1],
                format_func=lambda x: "✅ Activé" if x == 1 else "❌ Désactivé",
                help="Le client a-t-il activé le prélèvement automatique?"
            )
        
        with col6:
            contract_type = st.selectbox(
                "Type de Contrat",
                options=["Monthly", "One year", "Two year"],
                help="Durée d'engagement du contrat"
            )
        
        st.divider()
        
        # Section 3: Utilisation des Services
        st.subheader("📱 Utilisation des Services")
        col7, col8 = st.columns(2)
        
        with col7:
            data_usage_gb = st.number_input(
                "Consommation Data (GB/mois)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.1,
                help="Consommation moyenne de données mobiles par mois"
            )
        
        with col8:
            voice_minutes = st.number_input(
                "Minutes Vocales (min/mois)",
                min_value=0,
                max_value=5000,
                value=300,
                step=10,
                help="Nombre de minutes d'appels vocaux par mois"
            )
        
        st.divider()
        
        # Section 4: Qualité de Service
        st.subheader("🔧 Qualité de Service")
        col9, col10 = st.columns(2)
        
        with col9:
            support_calls = st.number_input(
                "Appels au Support",
                min_value=0,
                max_value=20,
                value=1,
                step=1,
                help="Nombre d'appels au service client dans les 3 derniers mois"
            )
        
        with col10:
            network_quality = st.slider(
                "Qualité du Réseau",
                min_value=1,
                max_value=5,
                value=4,
                help="Note de satisfaction de la qualité du réseau (1=Très mauvais, 5=Excellent)"
            )
        
        st.divider()
        
        # Bouton de soumission centré et stylisé
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            submit_button = st.form_submit_button(
                label="🔮 Prédire le Risque de Churn",
                use_container_width=True
            )
    
    # Traitement du formulaire après soumission
    if submit_button:
        # Préparation des données du client
        client_data = {
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
        }
        
        # Conversion en DataFrame
        client_df = pd.DataFrame([client_data])
        
        # Prédiction
        with st.spinner("🔄 Analyse en cours..."):
            result = make_prediction(client_df)[0]
        
        # Affichage des résultats
        st.success("✅ Analyse Terminée!")
        
        # Création de 3 colonnes pour les métriques principales
        col_met1, col_met2, col_met3 = st.columns(3)
        
        with col_met1:
            st.metric(
                "Probabilité de Churn",
                f"{result['churn_probability'] * 100:.1f}%",
                delta=f"{(result['churn_probability'] - THRESHOLD) * 100:.1f}% vs seuil",
                delta_color="inverse"
            )
        
        with col_met2:
            prediction_text = "🔴 VA PARTIR" if result['churn_prediction'] == 1 else "🟢 VA RESTER"
            st.metric(
                "Prédiction",
                prediction_text,
                delta=None
            )
        
        with col_met3:
            risk_emoji = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}
            st.metric(
                "Niveau de Risque",
                f"{risk_emoji[result['risk_level']]} {result['risk_level']}",
                delta=None
            )
        
        st.divider()
        
        # Visualisations
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            # Graphique jauge
            gauge_fig = create_gauge_chart(result['churn_probability'])
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col_viz2:
            # Graphique des features
            features_fig = create_feature_importance_chart(client_data)
            st.plotly_chart(features_fig, use_container_width=True)
        
        st.divider()
        
        # Recommandations personnalisées
        st.subheader("💡 Recommandations Personnalisées")
        
        # Génération des recommandations
        recommendations = generate_recommendations(result, client_data)
        
        # Affichage avec style en fonction du risque
        risk_class = f"{result['risk_level'].lower()}-risk"
        
        st.markdown(f"""
            <div class="{risk_class}">
                <h4>Actions Recommandées pour ce Client:</h4>
            </div>
        """, unsafe_allow_html=True)
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
        
        # Détails techniques (expander)
        with st.expander("🔍 Détails Techniques de la Prédiction"):
            st.json({
                "Probabilité Brute": result['churn_probability'],
                "Seuil de Décision": THRESHOLD,
                "Prédiction Binaire": result['churn_prediction'],
                "Niveau de Risque": result['risk_level'],
                "Données Client": client_data
            })

# ============================================================
# 8️⃣ MODE PRÉDICTION BATCH (FICHIER)
# ============================================================

if "Batch" in mode:
    st.header("📂 Prédictions en Masse (Batch)")
    
    # Instructions pour l'utilisateur
    st.markdown("""
    ### 📋 Instructions
    
    1. **Préparez votre fichier** avec les colonnes suivantes (obligatoires):
       - `age`, `tenure_months`, `monthly_charges`, `data_usage_gb`
       - `voice_minutes`, `support_calls`, `network_quality`
       - `payment_delay`, `auto_payment`, `contract_type`
    
    2. **Formats acceptés**: CSV, Excel (.xlsx), JSON, TXT (séparateur: virgule ou tabulation)
    
    3. **Uploadez le fichier** ci-dessous
    """)
    
    # Zone d'upload
    uploaded_file = st.file_uploader(
        "Choisissez un fichier contenant vos données clients",
        type=["csv", "xlsx", "json", "txt"],
        help="Le fichier doit contenir les colonnes requises listées ci-dessus"
    )
    
    # Traitement du fichier uploadé
    if uploaded_file is not None:
        try:
            # Affichage du nom et de la taille du fichier
            file_details = {
                "Nom du fichier": uploaded_file.name,
                "Type": uploaded_file.type,
                "Taille": f"{uploaded_file.size / 1024:.2f} KB"
            }
            
            st.info(f"📄 Fichier chargé: **{uploaded_file.name}** ({file_details['Taille']})")
            
            # Lecture du fichier selon son extension
            with st.spinner("📖 Lecture du fichier en cours..."):
                if uploaded_file.name.endswith(".csv") or uploaded_file.name.endswith(".txt"):
                    # Tentative de détection automatique du séparateur
                    try:
                        df = pd.read_csv(uploaded_file)
                    except:
                        df = pd.read_csv(uploaded_file, sep='\t')
                
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                
                elif uploaded_file.name.endswith(".json"):
                    df = pd.read_json(uploaded_file)
                
                else:
                    st.error("❌ Format de fichier non supporté")
                    st.stop()
            
            # Vérification des données
            st.success(f"✅ Fichier lu avec succès! **{len(df)} lignes** détectées")
            
            # Affichage d'un aperçu des données
            st.subheader("👁️ Aperçu des Données")
            st.dataframe(
                df.head(10),
                use_container_width=True,
                height=300
            )
            
            # Statistiques descriptives
            with st.expander("📊 Statistiques Descriptives"):
                st.write(df.describe())
            
            st.divider()
            
            # Bouton pour lancer les prédictions
            if st.button("🚀 Lancer les Prédictions", use_container_width=True):
                
                # Barre de progression
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔄 Préparation des données...")
                progress_bar.progress(25)
                
                # Prédictions
                status_text.text("🤖 Prédictions en cours...")
                progress_bar.progress(50)
                
                results = make_prediction(df)
                
                status_text.text("📊 Création des résultats...")
                progress_bar.progress(75)
                
                # Conversion en DataFrame
                results_df = pd.DataFrame(results)
                
                # Ajout des index pour correspondance
                results_df.index = df.index
                
                # Fusion avec les données originales
                final_df = pd.concat([df, results_df], axis=1)
                
                progress_bar.progress(100)
                status_text.text("✅ Prédictions terminées!")
                
                # Pause pour l'effet visuel
                import time
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"🎉 **{len(results_df)} prédictions** effectuées avec succès!")
                
                # Métriques globales
                st.subheader("📈 Vue d'Ensemble des Résultats")
                
                col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
                
                with col_stats1:
                    total_clients = len(results_df)
                    st.metric("Total Clients", f"{total_clients:,}")
                
                with col_stats2:
                    churn_count = results_df['churn_prediction'].sum()
                    churn_rate = (churn_count / total_clients) * 100
                    st.metric(
                        "Clients à Risque",
                        f"{churn_count:,}",
                        delta=f"{churn_rate:.1f}%",
                        delta_color="inverse"
                    )
                
                with col_stats3:
                    high_risk = (results_df['risk_level'] == 'High').sum()
                    st.metric("Risque Élevé 🔴", f"{high_risk:,}")
                
                with col_stats4:
                    avg_prob = results_df['churn_probability'].mean()
                    st.metric("Prob. Moyenne", f"{avg_prob * 100:.1f}%")
                
                st.divider()
                
                # Visualisations des résultats batch
                st.subheader("📊 Visualisations")
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    # Distribution des niveaux de risque
                    risk_counts = results_df['risk_level'].value_counts()
                    fig_pie = px.pie(
                        values=risk_counts.values,
                        names=risk_counts.index,
                        title="Distribution des Niveaux de Risque",
                        color=risk_counts.index,
                        color_discrete_map={'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_chart2:
                    # Distribution des probabilités
                    fig_hist = px.histogram(
                        results_df,
                        x='churn_probability',
                        nbins=30,
                        title="Distribution des Probabilités de Churn",
                        labels={'churn_probability': 'Probabilité de Churn'},
                        color_discrete_sequence=['#667eea']
                    )
                    fig_hist.add_vline(
                        x=THRESHOLD,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Seuil ({THRESHOLD})"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # Top clients à risque
                st.subheader("🚨 Top 10 Clients à Plus Haut Risque")
                top_risk = final_df.nlargest(10, 'churn_probability')
                st.dataframe(
                    top_risk,
                    use_container_width=True,
                    height=350
                )
                
                st.divider()
                
                # Téléchargement des résultats
                st.subheader("💾 Télécharger les Résultats")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    # CSV
                    csv = final_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f'predictions_churn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Excel
                    from io import BytesIO
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        final_df.to_excel(writer, sheet_name='Predictions', index=False)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Télécharger Excel",
                        data=buffer,
                        file_name=f'predictions_churn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                        mime='application/vnd.ms-excel',
                        use_container_width=True
                    )
                
                with col_dl3:
                    # JSON
                    json_str = final_df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Télécharger JSON",
                        data=json_str,
                        file_name=f'predictions_churn_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
                        mime='application/json',
                        use_container_width=True
                    )
                
                # Affichage complet des résultats
                with st.expander("📋 Voir Tous les Résultats"):
                    st.dataframe(
                        final_df,
                        use_container_width=True,
                        height=500
                    )
        
        except Exception as e:
            st.error(f"❌ **Erreur lors du traitement du fichier**: {str(e)}")
            st.info("ℹ️ Vérifiez que votre fichier contient toutes les colonnes requises et qu'il est bien formaté")
            
            # Affichage de l'erreur détaillée pour le debug
            with st.expander("🔍 Détails de l'erreur (Debug)"):
                st.code(str(e))

# ============================================================
# 9️⃣ FOOTER
# ============================================================

st.divider()

st.markdown("""
<div style='text-align: center; padding: 20px; background: #f5f5f5; border-radius: 10px; margin-top: 30px;'>
    <h3 style='color: #667eea;'>📞 Besoin d'Aide ?</h3>
    <p>Contactez l'équipe Data Science: <a href='mailto:sefdinecollab@gmail.com'>sefdinecollab@gmail.com</a></p>
    <p style='color: #666; margin-top: 10px;'>
        <small>
            © 2026 Telecom Analytics | Version 1.0.0<br>
            Propulsé par Ahmed Sefdine
        </small>
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 🔚 FIN DU CODE
# ============================================================
