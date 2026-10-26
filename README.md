# Détection des émotions à partir de texte — GoEmotions

Mini-projet de classification de commentaires en anglais avec TF‑IDF et modèles de Machine Learning classiques. Le dépôt réunit les données, le notebook reproductible et la présentation.

## Résumé des résultats

Les scores de validation ci-dessous ont été relevés dans les sorties de l’exécution source du notebook. Les sorties en cache ont été retirées du notebook de revue pour éviter des résultats périmés. Les métriques de F1 sont **pondérées**. Les expériences ne partagent pas toutes la même cible; les scores décrivent donc des tâches de difficulté différente.

| Expérience | Meilleur modèle sur dev | Accuracy dev | F1 pondéré dev |
|---|---|---:|---:|
| 28 labels (27 émotions + neutral) | Random Forest | 0,508 | 0,447 |
| 3 sentiments, train non équilibré | Logistic Regression | 0,690 | 0,693 |
| 3 sentiments, undersampling du train | Random Forest | 0,674 | 0,679 |
| Positive / negative, undersampling du train | Linear SVM | 0,829 | 0,831 |

Le modèle binaire Linear SVM atteint **0,823 d’accuracy** et **0,825 de F1 pondéré** sur le jeu de test. Sa matrice de confusion (ordre des classes : négatif, positif) est : `[[998, 196], [379, 1675]]`.

## Organisation

```text
.
├── data/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── emotions.txt
│   ├── sentiment_mapping.json
│   └── README.md
├── docs/
│   └── SOURCES.md
├── notebooks/
│   └── goemotions_progression.ipynb
├── presentation/
│   └── Go_Emotions.pptx
├── requirements.txt
└── README.md
```

## Exécution

1. Installer Python 3.10 ou plus récent.
2. Depuis la racine du dépôt, installer les dépendances :

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Démarrer Jupyter depuis la racine du dépôt (ou depuis `notebooks/`) et ouvrir `notebooks/goemotions_progression.ipynb`.
4. Exécuter les cellules dans l’ordre. Les chemins des données sont résolus automatiquement depuis `data/`.

Les entraînements peuvent prendre plusieurs minutes selon le matériel, en particulier Random Forest. Les cinq modèles comparés sont Logistic Regression, Multinomial Naive Bayes, Linear SVM, Decision Tree et Random Forest.

## Méthode

- GoEmotions fournit à l’origine des exemples annotés avec une ou plusieurs étiquettes.
- Pour cette étude, le premier label de chaque exemple est conservé afin de transformer la cible en classification mono-label. Les exemples multi-label ne sont pas filtrés.
- La première expérience distingue les 27 émotions et la classe `neutral`, soit 28 classes.
- Le fichier `data/sentiment_mapping.json` regroupe les labels en `positive`, `negative` et `neutral`. Dans ce projet, `neutral` comprend le label `neutral` et les émotions ambiguës `realization`, `surprise`, `curiosity` et `confusion`.
- L’undersampling est appliqué uniquement aux données d’entraînement. Les ensembles dev et test gardent leur distribution d’origine.
- TF‑IDF est ajusté sur le train, puis appliqué à dev et test. Le meilleur modèle est choisi sur dev; le test est réservé à l’évaluation finale.

## Données et attribution

Les fichiers TSV sont les splits officiels de GoEmotions. Le jeu contient 43 410 exemples d’entraînement, 5 426 exemples de validation et 5 427 exemples de test. Consultez [data/README.md](data/README.md) et [docs/SOURCES.md](docs/SOURCES.md) pour les informations de provenance, de licence et de citation.

## Limites

Les commentaires sont en anglais et proviennent de Reddit. La règle du premier label perd l’information secondaire des exemples multi-label. Le regroupement en sentiments est une décision de projet, notamment pour les quatre émotions ambiguës. Les résultats ne doivent pas être interprétés comme une mesure de performance sur d’autres domaines ou langues.
