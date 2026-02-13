# 📊 ChurnPredict Pro : Dashboard de Prédiction du Churn

Une application **Streamlit** professionnelle conçue pour aider les entreprises de télécommunications à identifier et fidéliser les clients à risque.

## 🚀 Fonctionnalités

- **Dashboard Interactif** : Visualisation des tendances globales de churn.
- **Importation Multi-format** : Support des fichiers **Excel, CSV, JSON et TXT**.
- **Analyse Individuelle** : Formulaire de saisie pour tester des profils clients spécifiques.
- **Intelligence Artificielle** : Utilise un modèle *Random Forest* pour calculer les probabilités de départ.
- **Recommandations Stratégiques** : Génération automatique de commentaires et d'actions correctives personnalisées pour chaque client.
- **Export de Données** : Téléchargement des résultats de prédiction au format CSV.

## 🛠️ Installation

1. Clonez le dépôt :
   ```bash
   git clone https://github.com/votre-utilisateur/churn-predict-pro.git
   cd churn-predict-pro
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Lancez l'application :
   ```bash
   streamlit run app.py
   ```

## 📂 Structure du Projet

- `app.py` : Le code principal de l'application Streamlit.
- `rf_churn_model.pkl` : Le modèle Random Forest entraîné.
- `scaler.pkl` : Le scaler pour la normalisation des données.
- `features.pkl` : La liste des colonnes attendues par le modèle.
- `requirements.txt` : Liste des bibliothèques Python nécessaires.

## 🧪 Utilisation

Pour tester l'application avec un fichier, assurez-vous que votre fichier contient les colonnes suivantes :
`age`, `tenure_months`, `monthly_charges`, `data_usage_gb`, `voice_minutes`, `support_calls`, `network_quality`, `payment_delay`, `auto_payment`, `contract_type`.

---
*Développé avec ❤️ par Manus AI.*
