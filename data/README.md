# Données GoEmotions

`train.tsv`, `dev.tsv` et `test.tsv` sont les splits GoEmotions publiés par Google Research, sans ligne d’en-tête. Les colonnes sont : texte, identifiants de labels séparés par des virgules et identifiant du commentaire. `emotions.txt` donne l’ordre des labels utilisé par les identifiants numériques.

`sentiment_mapping.json` est le regroupement retenu pour ce mini-projet. Les douze labels positifs, onze négatifs, et cinq labels regroupés sous `neutral` couvrent les 27 émotions et le label `neutral` du jeu original. Ce fichier est spécifique à l’analyse présentée ici.

Les données GoEmotions sont attribuées à leurs auteurs et distribuées sous CC BY 4.0 dans le dépôt Google Research. Le dépôt original avertit que les commentaires peuvent contenir du contenu problématique et que les données ne représentent pas la diversité mondiale.

Source et détails : https://github.com/google-research/google-research/tree/master/goemotions
