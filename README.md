# 📡 Telecom Churn Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Application web interactive de prédiction du churn (désabonnement) client dans le secteur des télécommunications. Cette solution utilise le Machine Learning pour identifier les clients à risque et fournir des recommandations personnalisées pour améliorer la rétention.

### ✨ Fonctionnalités Principales

- 🎯 **Prédiction Individuelle** : Analyse détaillée d'un client via formulaire interactif
- 📊 **Prédiction Batch** : Traitement en masse de milliers de clients via fichier
- 📈 **Visualisations Interactives** : Graphiques dynamiques avec Plotly
- 💡 **Recommandations Personnalisées** : Actions concrètes basées sur l'IA
- 📥 **Export Multi-formats** : Téléchargement des résultats (CSV, Excel, JSON)
- 🎨 **Interface Moderne** : Design responsive et intuitif

## 🚀 Démarrage Rapide

### Prérequis

- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)
- Les fichiers du modèle ML (`.pkl`)

### Installation

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/telecom-churn-prediction.git
cd telecom-churn-prediction
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Vérifier les fichiers ML**

Assurez-vous d'avoir ces fichiers dans le répertoire principal :
- `rf_churn_model.pkl` : Modèle Random Forest entraîné
- `scaler.pkl` : Scaler StandardScaler pour normalisation
- `features.pkl` : Liste des features utilisées

4. **Lancer l'application**
```bash
streamlit run streamlit_app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 📁 Structure du Projet

```
telecom-churn-prediction/
│
├── streamlit_app.py          # Application principale Streamlit
├── requirements.txt           # Dépendances Python
├── README.md                  # Documentation (ce fichier)
├── .streamlit/               # Configuration Streamlit
│   └── config.toml           # Thème et paramètres
│
├── rf_churn_model.pkl        # Modèle ML (à ajouter)
├── scaler.pkl                # Scaler (à ajouter)
├── features.pkl              # Features (à ajouter)
│
├── data/                     # Dossier pour fichiers de test (optionnel)
│   ├── sample_data.csv
│   └── template.xlsx
│
└── docs/                     # Documentation additionnelle
    ├── user_guide.md
    └── deployment.md
```

## 📊 Format des Données

### Colonnes Requises

Votre fichier de données doit contenir les colonnes suivantes :

| Colonne | Type | Description | Exemple |
|---------|------|-------------|---------|
| `age` | int | Âge du client | 30 |
| `tenure_months` | int | Ancienneté en mois | 12 |
| `monthly_charges` | float | Facture mensuelle (€) | 75.50 |
| `data_usage_gb` | float | Consommation data (GB) | 5.2 |
| `voice_minutes` | int | Minutes vocales | 300 |
| `support_calls` | int | Appels au support | 2 |
| `network_quality` | int | Qualité réseau (1-5) | 4 |
| `payment_delay` | int | Retards de paiement | 0 |
| `auto_payment` | int | Paiement auto (0/1) | 1 |
| `contract_type` | str | Type contrat | "One year" |

### Exemples de Fichiers

#### CSV (virgules)
```csv
age,tenure_months,monthly_charges,data_usage_gb,voice_minutes,support_calls,network_quality,payment_delay,auto_payment,contract_type
30,12,75.50,5.2,300,2,4,0,1,One year
45,24,120.00,15.8,450,1,5,0,1,Two year
```

#### JSON
```json
[
  {
    "age": 30,
    "tenure_months": 12,
    "monthly_charges": 75.50,
    "data_usage_gb": 5.2,
    "voice_minutes": 300,
    "support_calls": 2,
    "network_quality": 4,
    "payment_delay": 0,
    "auto_payment": 1,
    "contract_type": "One year"
  }
]
```

## 🎮 Guide d'Utilisation

### Mode Prédiction Individuelle

1. Sélectionnez **"🧍 Prédiction Individuelle"** dans la barre latérale
2. Remplissez le formulaire avec les informations du client
3. Cliquez sur **"🔮 Prédire le Risque de Churn"**
4. Consultez les résultats :
   - Probabilité de churn
   - Niveau de risque (High/Medium/Low)
   - Visualisations interactives
   - Recommandations personnalisées

### Mode Prédiction Batch

1. Sélectionnez **"📂 Prédiction Batch (Fichier)"** dans la barre latérale
2. Préparez votre fichier (CSV, Excel, JSON, TXT)
3. Uploadez le fichier via l'interface
4. Cliquez sur **"🚀 Lancer les Prédictions"**
5. Visualisez les résultats globaux
6. Téléchargez les résultats au format souhaité

## 🌐 Déploiement sur Streamlit Cloud

### Méthode Rapide

1. **Créer un compte** sur [Streamlit Cloud](https://streamlit.io/cloud)

2. **Connecter votre GitHub**
   - Autorisez Streamlit à accéder à vos repositories

3. **Déployer l'application**
   - Cliquez sur "New app"
   - Sélectionnez votre repository
   - Branche : `main`
   - Fichier principal : `streamlit_app.py`
   - Cliquez sur "Deploy!"

4. **Ajouter les fichiers ML**

   Les fichiers `.pkl` sont trop volumineux pour GitHub. Options :
   
   **Option A : Utiliser Streamlit Secrets + Cloud Storage**
   ```python
   # Télécharger depuis AWS S3, Google Cloud Storage, etc.
   ```
   
   **Option B : Git LFS (Large File Storage)**
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes
   git add *.pkl
   git commit -m "Add model files"
   git push
   ```
   
   **Option C : Hébergement externe**
   - Stocker sur Google Drive / Dropbox
   - Télécharger au démarrage de l'app

### Configuration Avancée

Créez un fichier `.streamlit/config.toml` pour personnaliser l'apparence :

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f5f5f5"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableXsrfProtection = true
```

## 🔒 Sécurité et Confidentialité

- ✅ Aucune donnée n'est stockée de manière permanente
- ✅ Les fichiers uploadés sont supprimés après traitement
- ✅ Utilisation de HTTPS sur Streamlit Cloud
- ✅ Protection XSRF activée

⚠️ **Important** : Ne pas committer de données sensibles sur GitHub !

## 🛠️ Technologies Utilisées

- **Streamlit** : Framework d'application web
- **Pandas** : Manipulation de données
- **Scikit-learn** : Machine Learning
- **Plotly** : Visualisations interactives
- **Random Forest** : Algorithme de classification

## 📈 Performances du Modèle

| Métrique | Valeur |
|----------|--------|
| **Accuracy** | ~85% |
| **Precision** | ~83% |
| **Recall** | ~87% |
| **F1-Score** | ~85% |

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Committez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📝 To-Do List

- [ ] Ajouter l'authentification utilisateur
- [ ] Intégrer une base de données pour l'historique
- [ ] Créer des rapports PDF automatiques
- [ ] Ajouter des notifications par email
- [ ] Implémenter l'A/B testing
- [ ] Ajouter support multilingue (FR/EN)
- [ ] Créer une API REST
- [ ] Dashboard administrateur

## 🐛 Problèmes Connus

- Les très gros fichiers (>200MB) peuvent prendre du temps à traiter
- L'encodage UTF-8 est requis pour les fichiers CSV français
- Excel (.xls) ancien format non supporté (utiliser .xlsx)

## 📞 Support

Pour toute question ou problème :

- 📧 Email : data@telecom.com
- 🐛 Issues : [GitHub Issues](https://github.com/votre-username/telecom-churn-prediction/issues)
- 📖 Docs : [Wiki du Projet](https://github.com/votre-username/telecom-churn-prediction/wiki)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👨‍💻 Auteur

**Votre Nom**
- GitHub : [@votre-username](https://github.com/votre-username)
- LinkedIn : [Votre Profil](https://linkedin.com/in/votre-profil)
- Email : votre.email@example.com

## 🙏 Remerciements

- L'équipe Streamlit pour leur excellent framework
- La communauté Scikit-learn
- Tous les contributeurs du projet

---

<div align="center">

**Développé avec ❤️ pour améliorer la rétention client**

[⭐ Star ce projet](https://github.com/votre-username/telecom-churn-prediction) si vous le trouvez utile !

</div>
