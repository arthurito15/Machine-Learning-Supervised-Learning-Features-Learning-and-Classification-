# 📊 Credit Scoring – Machine Learning & Data Processing

Ce projet explore différentes techniques d’**apprentissage supervisé**, de **feature engineering**, de **sélection de variables**, et d’**orchestration de pipelines** appliquées à un dataset de *credit scoring*.  
Il inclut également un traitement avancé de données hétérogènes et la création d’une **API FastAPI**.

---

## 📁 Contenu du projet

Le projet couvre les étapes suivantes :

### 1. Chargement et préparation des données
- Import du dataset `credit_scoring.csv`
- Séparation des features et de la variable cible `Status`
- Analyse du déséquilibre des classes  
  > *Exemple du document : 72% positifs, 28% négatifs.*

---

## 🧠 Apprentissage supervisé

### 2. Entraînement et évaluation de modèles
Modèles testés :
- Decision Tree (DT)
- KNN
- MLP (réseau de neurones)

Résultats initiaux (accuracy) :
- **DT : 0.773**
- **KNN : 0.748**
- **MLP : 0.747**

Les courbes ROC montrent des performances comparables, avec un léger avantage pour l’arbre de décision.

---

## 🔧 Feature Engineering

### 3. Normalisation des variables continues
Après normalisation :
- **MLP devient le meilleur modèle (0.808)**  
- KNN et DT progressent également

### 4. Création de nouvelles variables (combinaisons linéaires)
Ajout de composantes PCA :
- Légère amélioration globale
- Le MLP reste le plus performant sur données normalisées + PCA

---

## 🏆 Sélection de variables

### 5. Importance des variables & sélection optimale
Méthode utilisée : élimination itérative + MLP

Variables les plus importantes :
- Income  
- Seniority  
- pca2  
- pca3  
- Price  
- pca1  

> *Les 6 premières variables donnent la meilleure accuracy.*

Visualisations :
- Graphique d’importance des variables
- Courbe accuracy vs nombre de variables
- Analyse SHAP (interactions entre Seniority et Home)

---

## ⚙️ Optimisation des modèles

### 6. Recherche d’hyperparamètres
Exemple :  
```
MLP → hidden_layer_sizes = [46, 26]
```

---

## 🧵 Pipelines & Orchestration

### 7. Création d’un pipeline complet
Pipeline :
- StandardScaler  
- PCA  
- MLPClassifier  

### 8. Orchestration automatique
Fonction : `pipeline_generation_train_test_split`  
Permet d’automatiser :
- Sélection de variables  
- Normalisation  
- PCA  
- Entraînement du modèle optimal  

---

## 🌐 API FastAPI

### 9. Déploiement d’une API
Le fichier `api.py` expose un endpoint permettant :
- De charger un modèle entraîné
- De prédire le statut d’un client à partir de nouvelles données

---

# 🔬 Comparaison avancée d’algorithmes

### 10. Tests sur plusieurs modèles
Modèles inclus :
- Naive Bayes
- CART / ID3 / Decision Stump
- MLP (20-10)
- KNN
- Bagging
- AdaBoost
- Random Forest
- Gradient Boosting

Tests réalisés sur :
- Données brutes  
- Données normalisées  
- Données normalisées + nouvelles variables  

### Résultats marquants :
- **AdaBoost, Random Forest et Gradient Boosting** sont les plus performants (≈ 0.79–0.80)
- Le MLP progresse fortement après normalisation
- Les méthodes d’ensemble (Bagging, Boosting) dominent globalement

---

# 🧩 Données hétérogènes

## 1. Variables continues
- Nettoyage des données manquantes
- Normalisation
- Comparaison des modèles

## 2. Traitement des données manquantes
Techniques utilisées :
- Imputation moyenne (numérique)
- Imputation mode (catégorielle)
- One-Hot Encoding
- Standardisation

Résultats :
- Les modèles d’ensemble restent les plus performants  
- **Bagging atteint 0.876**, meilleur score global

---

# 🚀 Installation & Exécution

### Prérequis
```
Python 3.9+
pip install -r requirements.txt
```

### Lancer les notebooks
```
jupyter notebook
```

### Lancer l’API FastAPI
```
uvicorn api:app --reload
```

---

# 📌 Auteurs

- **Konkobo Ulrich Arthur**
- **Pellois Guillaume**
- **Issoumaila Fomba**

