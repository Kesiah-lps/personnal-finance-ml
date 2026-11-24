# Personal finance & ML

Projet d'analyse de données sur un dataset synthétique de finance personnelle (32 424 personnes, 20 variables, relevés de 2021 à 2025) : éligibilité au prêt, score de crédit, épargne et segmentation selon le ratio dette/revenu.

## Contenu

- `personal_finance_ml.ipynb` : le notebook principal (nettoyage, analyse exploratoire, modèles, segmentation, dashboard)
- `data/synthetic_personal_finance_dataset.csv` : les données
- `requirements.txt`

## Démarche

1. Nettoyage et contrôles de cohérence entre colonnes
2. Analyse exploratoire, corrélations et tests d'indépendance
3. Feature engineering (reste à vivre, taux d'épargne, mois de dépenses couverts par l'épargne...)
4. Split temporel : train avant juillet 2024, test sur la dernière année
5. Modèles pour le score de crédit et pour `has_loan`, comparés à une baseline
6. Grille d'éligibilité avec des règles (score ≥ 620, DTI ≤ 0,43, reste à vivre positif)
7. Recommandations d'épargne (règle 50/30/20, épargne de précaution de 6 mois)
8. Segmentation des emprunteurs avec K-Means

## Résultats

- Trois colonnes sont calculées à partir des autres : la mensualité (formule de prêt classique), le DTI (mensualité / revenu) et le ratio épargne/revenu. Le reste est généré indépendamment.
- Le score de crédit n'est pas prédictible : R² ≈ 0, AUC ≈ 0,50, test de permutation non significatif (p = 0,69).
- `has_loan` ne dépend pas du profil (AUC 0,50). En ajoutant le taux d'intérêt, connu seulement après l'octroi, l'AUC passe à 1,00 : c'est une fuite de données.
- Seulement 6,6 % des prêts existants respectent la grille d'éligibilité. La mensualité médiane représente 135 % du revenu.
- Épargne : 52 % des personnes peuvent investir leur surplus, 13 % doivent réduire leurs dépenses, 33 % sont en déficit (83 % des emprunteurs).
- K-Means donne 4 groupes d'emprunteurs (silhouette 0,27), utiles comme niveaux de risque même s'il n'y a pas de vrais groupes dans les données.

## Lancer le projet

```bash
pip install -r requirements.txt
jupyter lab personal_finance_ml.ipynb
```

## Limites

Les données sont synthétiques et la plupart des variables sont tirées au hasard, donc les modèles ne peuvent pas apprendre grand-chose. Le projet sert surtout à appliquer une méthode propre : vérifier les données, comparer à une baseline, éviter les fuites.

## Pistes

- Simuler un score de crédit qui dépend du profil pour avoir un vrai modèle de scoring
- Étudier l'équité et la calibration des décisions
- Tester sur des données réelles (Lending Club, Home Credit)
- Dashboard interactif avec Streamlit
