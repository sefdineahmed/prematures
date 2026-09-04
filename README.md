# Prédiction du Risque d'Accouchement Prématuré

## Description du Projet

Ce projet est une étude clinique basée sur les facteurs prénataux (médicaux et personnels) liés à un accouchement prématuré chez les femmes enceintes déjà en travail. L'objectif est de fournir un outil d'aide à la décision pour les professionnels de santé afin d'identifier les patientes à haut risque et d'optimiser la prise en charge médicale.

L'étude porte sur un échantillon de **390 femmes** et utilise **12 variables prédictives** pour classer l'issue de la grossesse (Prématuré ou À terme).

## Fonctionnalités

* **Tableau de Bord Analytique :** Visualisation des statistiques clés du modèle et de l'importance des facteurs de risque.
* **Diagnostic Individuel :** Interface permettant de saisir les données d'une patiente pour obtenir un score de risque immédiat et une probabilité.
* **Analyse de Groupe (Batch) :** Importation de fichiers CSV pour traiter plusieurs dossiers de patientes simultanément.
* **Modèle Optimisé :** Utilisation d'un algorithme de type Forêt Aléatoire (Random Forest) avec gestion du déséquilibre des classes (`class_weight='balanced'`).

## Variables du Modèle

Les facteurs clés utilisés pour la prédiction incluent :

1. **GEST :** Âge gestationnel en semaines à l'entrée dans l'étude.
2. **DILATE :** Dilatation du col en cm.
3. **EFFACE :** Effacement du col en %.
4. **CONSIS :** Consistance du col (1=mou, 2=moyen, 3=ferme).
5. **MEMBRAN :** Rupture des membranes (1=oui, 2=non, 3=incertain).
6. **AGE :** Âge de la patiente.
7. **DIAB :** Présence de diabète (1=oui, 2=non, 9=manquant).
8. **GEMEL :** Grossesse simple ou multiple.
*(Note : La variable `BEBAGE` a été exclue pour éviter les fuites de données).*

## Performances du Modèle

Après optimisation des hyperparamètres, le modèle présente les résultats suivants :

* **Exactitude (Accuracy) :** 76 %
* **Sensibilité (Recall Classe 1) :** 85 % (Capacité à détecter les cas de prématurité réels)
* **Précision (Classe 1) :** 81 %

## Installation et Utilisation

### Prérequis

* Python 3.8 ou supérieur
* Un gestionnaire de paquets (`pip`)

### Installation

1. **Cloner le dépôt :**
```bash
git clone https://github.com/votre-u/projet-prematures.git
cd prematures

```


2. **Installer les dépendances :**
```bash
pip install -r requirements.txt

```

### Lancement de l'application

Pour exécuter le tableau de bord Streamlit localement :

```bash
streamlit run app.py

```

## Structure des fichiers

* `app.py` : Code de l'application Streamlit.
* `model.pkl` : Modèle de Machine Learning entraîné et sérialisé.
* `requirements.txt` : Liste des bibliothèques nécessaires (Streamlit, Pandas, Scikit-learn, Plotly).
* `naccouchement-premature.ipynb` : Notebook contenant l'exploration des données et l'entraînement du modèle.

## Avertissement Médical

Cet outil est un prototype basé sur des données statistiques et est destiné à des fins de recherche et d'éducation. Il ne remplace en aucun cas le diagnostic d'un professionnel de santé qualifié.
