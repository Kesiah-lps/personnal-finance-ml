# Finance personnelle & ML : éligibilité, scoring, épargne et segmentation DTI

Projet pédagogique de data science sur `synthetic_personal_finance_dataset.csv` (32 424 individus × 20 colonnes, relevés du 23/07/2021 au 22/07/2025).

## Objectifs
1. Modéliser l'éligibilité au prêt
2. Prédire le score de crédit
3. Analyser les dépenses et recommander un niveau d'épargne
4. Segmenter les emprunteurs selon leur ratio dette/revenu (DTI)
5. Construire un tableau de bord
6. Servir de support pour enseigner le ML appliqué à la finance

## Contenu du notebook (`personal_finance_ml.ipynb`)
| § | Étape |
|---|---|
| 1-2 | Contexte, chargement |
| 3 | Audit qualité, nettoyage, **rétro-ingénierie des colonnes dérivées** |
| 4 | EDA : distributions, segments, corrélations, tests d'indépendance, tendance temporelle |
| 5 | Feature engineering (reste à vivre, taux d'épargne net, mois de coussin, tranches FICO…) |
| 6 | Protocole : **split temporel** (train < 01/07/2024, test ≥ 01/07/2024) |
| 7 | Scoring de crédit (régression + classification + test de permutation) ; `has_loan` avec **démonstration de fuite de cible** |
| 8 | Grille d'éligibilité fondée sur des règles (score ≥ 620, DTI ≤ 0,43, reste à vivre > 0) + capacité d'emprunt |
| 9 | Dépenses et recommandations d'épargne (règle 50/30/20, fonds d'urgence de 6 mois) |
| 10 | Clustering K-Means des emprunteurs (silhouette, profils) |
| 11 | Tableau de bord de synthèse |
| 12-14 | Métriques, conclusions, pistes |

## Résultats clés
- **Structure du générateur** : `monthly_emi` = formule d'annuité exacte ; `debt_to_income_ratio` = EMI / revenu mensuel ; `savings_to_income_ratio` = épargne / revenu annuel ; dépenses = 30 à 90 % du revenu. Toutes les autres variables sont tirées indépendamment (`job_title` est même incohérent avec `employment_status`).
- **Score de crédit non prédictible** : R² = −0,001, MAE = baseline (137,3), AUC (score ≥ 670) = 0,50, p de permutation = 0,69.
- **`has_loan` n'est pas une décision d'octroi** : AUC ex ante = 0,505 ; en ajoutant le taux d'intérêt (variable post-octroi), l'AUC monte à 1,00 : c'est un exemple de **fuite de cible**.
- **Seulement 6,6 % des prêts existants** respecteraient une grille bancaire standard ; le DTI médian des emprunteurs est de 1,35 (mensualité = 135 % du revenu).
- **Épargne** : 52 % « investir l'excédent », 33 % en déficit (à 83 % des emprunteurs), 13 % doivent réduire leurs dépenses (≈ 210 $/mois en médiane).
- **Clustering** (k = 4, silhouette 0,27) : frugaux, dépensiers, sans coussin, surendettement sévère. C'est une grille de lecture utile, mais les données ne contiennent pas de groupes naturels.

## Lancer le projet
```bash
pip install -r requirements.txt
# placer le CSV dans data/synthetic_personal_finance_dataset.csv
jupyter lab personal_finance_ml.ipynb
```

## Limites
Les données sont synthétiques et générées par tirages indépendants. Les modèles prédictifs ne peuvent donc rien apprendre au-delà des formules. Le projet vise surtout à enseigner une méthode : détecter les colonnes dérivées, tester l'indépendance, comparer à une baseline, faire des tests de permutation, repérer les fuites et préférer des règles explicables.

## Pistes
Simuler une cible réaliste (score dépendant du DTI et de l'épargne) puis expliquer le modèle avec SHAP ; simuler des décisions d'octroi et auditer l'équité ; comparer avec un GMM ou des données réelles (Lending Club, Home Credit) ; construire un dashboard Streamlit ; adapter la grille au contexte français (taux d'effort de 35 % selon le HCSF).
