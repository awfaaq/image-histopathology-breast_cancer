# Classification d'images médicales par deep learning : Détection de carcinome canalaire infiltrant

Ce projet a été réalisé dans le cadre du **Master 2 Mathématiques, Modélisation et apprentissage statistique (M2 MMAS)** de l'**Université Paris Cité**, pour le cours d'introduction au Deep Learning (Année 2025-2026).

---

##  Description du projet

Le cancer du sein est la forme de cancer la plus courante chez la femme. Le Carcinome Canalaire Infiltrant (Invasive Ductal Carcinoma - IDC) est le sous-type le plus fréquent. L'objectif de ce projet est de développer un pipeline de Deep Learning capable de classer automatiquement des images histologiques de tissus mammaires afin de déterminer la présence ou l'absence de cellules cancéreuses (IDC).

Une attention particulière a été portée à la gestion du déséquilibre des classes et à l'interprétabilité des résultats via des cartes de chaleur (Grad-CAM).

##  Dataset

Les données proviennent du jeu de données **Breast Histopathology Images** (disponible sur Kaggle).
*   **Contenu :** Environ 277 524 patches d'images (50x50 pixels).
*   **Classes :**
    *   `0` (IDC négatif) : Tissu sain.
    *   `1` (IDC positif) : Présence de cancer.
*   **Problématique :** Le jeu de données est fortement déséquilibré (ratio ~2.6:1 en faveur de la classe négative), ce qui biaise les modèles standards.

##  Méthodologie et pipeline

Notre approche s'est décomposée en plusieurs étapes clés :

1.  **Exploration et Prétraitement :**
    *   Analyse statistique et visualisation des données.
    *   Redimensionnement des images (224x224 pour le Transfer Learning).
    *   Normalisation (moyenne et écart-type d'ImageNet ou calculés sur le dataset).

2.  **Modélisation :**
    *   **CNN "From Scratch" :** Développement d'architectures personnalisées (avec et sans Batch Normalization).
    *   **Transfer Learning :** Utilisation de modèles pré-entrainés sur ImageNet :
        *   ResNet18
        *   ResNet50
        *   EfficientNet-B0

3.  **Optimisation et gestion du déséquilibre :**
    *   **Data augmentation :** Rotations, flips horizontaux/verticaux, ajustement de la luminosité/contraste.
    *   **Pondération de la perte (Class weights) :** Pénalisation accrue des erreurs sur la classe minoritaire.
    *   **Sur-échantillonnage (oversampling) :** Utilisation de `WeightedRandomSampler` pour équilibrer les batches.
    *   **Recherche d'hyperparamètres :** Grid Search sur le taux d'apprentissage (LR), l'optimiseur (Adam, SGD), la taille de batch et la patience.

4.  **Interprétabilité :**
    *   Implémentation de **Grad-CAM** pour visualiser les zones de l'image influençant la décision du modèle (mise en évidence des zones pathologiques).

## Résultats Principaux

*   **Impact du déséquilibre :** Les modèles initiaux favorisaient la classe majoritaire (haute précision, faible rappel sur la classe IDC+).
*   **Améliorations :** L'ajout de la Batch normalization et la combinaison "pondération + data augmentation" ont permis d'améliorer le F1-score et le rappel (réduction des Faux Négatifs).
*   **Comparaison des modèles :**
    *   **EfficientNet-B0** a offert le meilleur compromis global (stabilité et performance).
    *   **ResNet18** s'est montré particulièrement efficace pour minimiser les faux négatifs, un critère critique en diagnostic médical.

## Installation et utilisation

Le projet a été développé sous forme de Notebook (Jupyter/Colab). Pour reproduire les résultats, les librairies suivantes sont nécessaires :

```bash
pip install torch torchvision torchcam opencv-python matplotlib seaborn scikit-learn pandas numpy kagglehub
```

**Structure du code :**
*   Chargement des données via l'API Kaggle.
*   Définition des classes `CustomDataset` et des `DataLoaders`.
*   Fonctions d'entraînement (`train_validate`) avec Early Stopping.
*   Évaluation (`evaluate_model`) : Matrices de confusion, Courbes ROC.
*   Visualisation Grad-CAM.

## Auteurs

*   **Daniel ASHRAFUL**
*   **Noaga Ferdinand Arthur DAHANI**
*   **Ahmed Salem HABIBI**

---
*Note : Ce projet a été réalisé à des fins pédagogiques. Les modèles développés ne constituent pas un outil de diagnostic médical certifié.*
