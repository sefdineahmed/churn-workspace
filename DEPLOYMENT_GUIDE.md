# 🚀 Guide de Déploiement sur GitHub et Streamlit Cloud

Ce guide détaille toutes les étapes pour déployer votre application de prédiction de churn sur GitHub et Streamlit Cloud.

## 📋 Table des Matières

1. [Préparation des Fichiers](#préparation-des-fichiers)
2. [Initialisation Git](#initialisation-git)
3. [Création du Repository GitHub](#création-du-repository-github)
4. [Push sur GitHub](#push-sur-github)
5. [Déploiement sur Streamlit Cloud](#déploiement-sur-streamlit-cloud)
6. [Gestion des Fichiers ML (.pkl)](#gestion-des-fichiers-ml-pkl)
7. [Dépannage](#dépannage)

---

## 1️⃣ Préparation des Fichiers

### Structure Recommandée

```
telecom-churn-prediction/
│
├── streamlit_app.py          ✅ Fichier principal
├── requirements.txt           ✅ Dépendances
├── README.md                  ✅ Documentation
├── .gitignore                ✅ Exclusions Git
│
├── .streamlit/
│   └── config.toml           ✅ Configuration
│
├── rf_churn_model.pkl        ⚠️ Modèle (voir section 6)
├── scaler.pkl                ⚠️ Scaler (voir section 6)
└── features.pkl              ⚠️ Features (voir section 6)
```

### Vérification des Fichiers

```bash
# Vérifiez que tous les fichiers nécessaires sont présents
ls -la

# Résultat attendu:
# streamlit_app.py
# requirements.txt
# README.md
# .gitignore
# .streamlit/config.toml
```

---

## 2️⃣ Initialisation Git

### Première Initialisation

```bash
# Naviguer dans le dossier du projet
cd /chemin/vers/telecom-churn-prediction

# Initialiser Git (si pas encore fait)
git init

# Vérifier le statut
git status
```

### Configuration Git (Première fois)

```bash
# Configurer votre nom (remplacez par le vôtre)
git config --global user.name "Votre Nom"

# Configurer votre email (remplacez par le vôtre)
git config --global user.email "votre.email@example.com"

# Vérifier la configuration
git config --list
```

---

## 3️⃣ Création du Repository GitHub

### Option A : Via l'Interface Web GitHub

1. **Connectez-vous** à [GitHub](https://github.com)

2. **Créez un nouveau repository** :
   - Cliquez sur le bouton `+` en haut à droite
   - Sélectionnez `New repository`

3. **Configurez le repository** :
   - **Repository name** : `telecom-churn-prediction`
   - **Description** : "Application de prédiction de churn télécom avec ML"
   - **Visibility** : Public ou Private (votre choix)
   - ❌ **NE PAS** cocher "Add a README file" (vous en avez déjà un)
   - ❌ **NE PAS** ajouter de .gitignore (vous en avez déjà un)
   - Cliquez sur `Create repository`

4. **Notez l'URL** de votre repository :
   ```
   https://github.com/votre-username/telecom-churn-prediction.git
   ```

### Option B : Via GitHub CLI

```bash
# Installer GitHub CLI (si pas déjà installé)
# macOS: brew install gh
# Windows: winget install GitHub.cli
# Linux: voir https://cli.github.com/

# S'authentifier
gh auth login

# Créer le repository
gh repo create telecom-churn-prediction --public --source=. --remote=origin
```

---

## 4️⃣ Push sur GitHub

### Méthode Standard

```bash
# 1. Ajouter tous les fichiers
git add .

# 2. Vérifier ce qui sera commité
git status

# 3. Premier commit
git commit -m "Initial commit: Application de prédiction de churn télécom"

# 4. Ajouter le remote GitHub (remplacez par votre URL)
git remote add origin https://github.com/votre-username/telecom-churn-prediction.git

# 5. Vérifier le remote
git remote -v

# 6. Push vers GitHub
git push -u origin main
```

### Si vous obtenez une erreur "branch main doesn't exist"

```bash
# Renommer la branche en main
git branch -M main

# Puis refaire le push
git push -u origin main
```

### Authentification GitHub

Si GitHub demande une authentification :

**Option 1 : Personal Access Token (Recommandé)**

1. Allez sur GitHub → Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Cochez : `repo` (full control)
4. Générez et copiez le token
5. Utilisez-le comme mot de passe lors du `git push`

**Option 2 : SSH**

```bash
# Générer une clé SSH
ssh-keygen -t ed25519 -C "votre.email@example.com"

# Copier la clé publique
cat ~/.ssh/id_ed25519.pub

# Ajouter sur GitHub → Settings → SSH Keys
```

---

## 5️⃣ Déploiement sur Streamlit Cloud

### Étape 1 : Créer un Compte

1. Allez sur [Streamlit Cloud](https://streamlit.io/cloud)
2. Cliquez sur `Sign up` ou `Get started`
3. **Connectez-vous avec GitHub** (recommandé)

### Étape 2 : Déployer l'Application

1. **Cliquez sur** `New app`

2. **Configurez le déploiement** :
   - **Repository** : `votre-username/telecom-churn-prediction`
   - **Branch** : `main`
   - **Main file path** : `streamlit_app.py`

3. **Advanced settings** (optionnel) :
   - Python version : `3.11` (ou votre version)
   - Secrets : voir section 6 pour les fichiers .pkl

4. **Cliquez sur** `Deploy!`

### Étape 3 : Attendre le Déploiement

- Le déploiement prend généralement 2-5 minutes
- Vous verrez les logs en temps réel
- Une fois terminé, l'app sera accessible via une URL : 
  ```
  https://votre-app-name.streamlit.app
  ```

---

## 6️⃣ Gestion des Fichiers ML (.pkl)

Les fichiers `.pkl` sont souvent **trop volumineux** pour GitHub (limite : 100 MB).

### Solution 1 : Git LFS (Large File Storage)

**Meilleure option pour les fichiers <2GB**

```bash
# Installer Git LFS
git lfs install

# Tracker les fichiers .pkl
git lfs track "*.pkl"

# Ajouter .gitattributes
git add .gitattributes

# Ajouter les fichiers .pkl
git add *.pkl

# Commit et push
git commit -m "Add ML model files with Git LFS"
git push origin main
```

**⚠️ Limitations** :
- GitHub Free : 1 GB storage, 1 GB bandwidth/month
- Si dépassé, il faudra payer ou utiliser une autre solution

### Solution 2 : Hébergement Externe + Téléchargement

**Option A : Google Drive**

```python
# Ajouter au début de streamlit_app.py
import gdown

@st.cache_resource
def download_model_from_drive():
    """Télécharge le modèle depuis Google Drive"""
    if not os.path.exists("rf_churn_model.pkl"):
        # Remplacez FILE_ID par l'ID de votre fichier Google Drive
        url = "https://drive.google.com/uc?id=FILE_ID"
        gdown.download(url, "rf_churn_model.pkl", quiet=False)
    
    # Idem pour les autres fichiers
    # ...
    
    return joblib.load("rf_churn_model.pkl")

# Utiliser la fonction
model = download_model_from_drive()
```

**Ajouter dans requirements.txt** :
```
gdown==4.7.1
```

**Option B : AWS S3, Google Cloud Storage, Dropbox, etc.**

Similaire à Google Drive, mais avec les SDK respectifs.

### Solution 3 : Streamlit Secrets

Pour des fichiers **vraiment petits** (<5MB), vous pouvez les encoder en base64 :

```python
import base64
import streamlit as st

# Décoder depuis secrets
model_bytes = base64.b64decode(st.secrets["model_base64"])
with open("model.pkl", "wb") as f:
    f.write(model_bytes)
```

**Dans Streamlit Cloud** :
- App → Settings → Secrets
- Ajouter :
  ```toml
  model_base64 = "..." # votre modèle encodé en base64
  ```

---

## 7️⃣ Dépannage

### Problème : "Application error" sur Streamlit Cloud

**Solutions** :

1. **Vérifier les logs** :
   - Streamlit Cloud → Your app → Manage app → Logs
   - Cherchez les erreurs Python

2. **Problèmes courants** :

   a) **Module manquant**
   ```
   ModuleNotFoundError: No module named 'plotly'
   ```
   → Ajoutez dans `requirements.txt`

   b) **Fichier .pkl manquant**
   ```
   FileNotFoundError: rf_churn_model.pkl
   ```
   → Voir section 6

   c) **Mauvaise version Python**
   ```
   SyntaxError: ...
   ```
   → Streamlit Cloud → Settings → Python version

### Problème : Git push rejected

```bash
# Si le push est rejeté, pull d'abord
git pull origin main --rebase

# Résoudre les conflits si nécessaire
# Puis push à nouveau
git push origin main
```

### Problème : Fichier trop volumineux

```
remote: error: File xxx.pkl is 150 MB; this exceeds GitHub's file size limit of 100 MB
```

**Solutions** :
1. Utiliser Git LFS (voir section 6)
2. Héberger ailleurs (Google Drive, S3, etc.)
3. Compresser le modèle si possible

### Problème : Application lente

**Optimisations** :

```python
# Utiliser le cache Streamlit
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
```

---

## 📞 Ressources et Support

### Documentation Officielle

- [Streamlit Docs](https://docs.streamlit.io)
- [GitHub Docs](https://docs.github.com)
- [Git LFS](https://git-lfs.github.com/)

### Communauté

- [Forum Streamlit](https://discuss.streamlit.io)
- [Streamlit Discord](https://discord.gg/streamlit)

### Contact Projet

- Issues GitHub : `https://github.com/votre-username/telecom-churn-prediction/issues`
- Email : data@telecom.com

---

## ✅ Checklist Finale

Avant de déployer, vérifiez :

- [ ] Tous les fichiers sont sur GitHub
- [ ] `requirements.txt` est à jour
- [ ] `.gitignore` exclut les fichiers sensibles
- [ ] Les fichiers .pkl sont gérés (LFS ou cloud)
- [ ] L'app fonctionne en local : `streamlit run streamlit_app.py`
- [ ] README.md est complet et à jour
- [ ] Secrets configurés (si nécessaire)
- [ ] URL de l'app partagée avec l'équipe

---

**🎉 Félicitations !** Votre application est maintenant déployée et accessible au monde entier !
