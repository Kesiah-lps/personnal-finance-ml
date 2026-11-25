"""Fonctions utilisées dans le projet : chargement, features, règles d'éligibilité, recommandations d'épargne."""
import numpy as np
import pandas as pd

DTI_MAX = 0.43
SCORE_MIN = 620
SAVINGS_TARGET = 0.20
EMERGENCY_MONTHS = 6


def load_data(path="data/synthetic_personal_finance_dataset.csv"):
    df = pd.read_csv(path, parse_dates=["record_date"])
    df["has_loan"] = df["has_loan"].map({"Yes": 1, "No": 0}).astype(int)
    df["loan_type"] = df["loan_type"].fillna("No loan")
    return df


def monthly_payment(amount, annual_rate_pct, term_months):
    """Mensualité d'un prêt à mensualités constantes."""
    amount = np.asarray(amount, dtype=float)
    r = np.asarray(annual_rate_pct, dtype=float) / 1200
    n = np.asarray(term_months, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pay = np.where(r > 0, amount * r * (1 + r) ** n / ((1 + r) ** n - 1), amount / n)
    return np.where(n > 0, pay, 0.0)


def max_loan(max_payment, annual_rate_pct=5.0, term_months=240):
    """Montant empruntable pour une mensualité donnée."""
    r = annual_rate_pct / 1200
    return np.asarray(max_payment, dtype=float) * (1 - (1 + r) ** -term_months) / r


def add_features(df):
    df = df.copy()
    df["expense_ratio"] = df.monthly_expenses_usd / df.monthly_income_usd
    df["disposable_income"] = df.monthly_income_usd - df.monthly_expenses_usd
    df["net_cashflow"] = df.disposable_income - df.monthly_emi_usd
    df["net_savings_rate"] = df.net_cashflow / df.monthly_income_usd
    df["emergency_months"] = df.savings_usd / (df.monthly_expenses_usd + df.monthly_emi_usd)
    df["log_income"] = np.log(df.monthly_income_usd)
    df["log_savings"] = np.log(df.savings_usd)
    df["max_emi"] = np.minimum(DTI_MAX * df.monthly_income_usd, df.disposable_income).clip(lower=0)
    df["max_loan_20y_5pct"] = max_loan(df.max_emi)
    return df


def is_eligible(credit_score, dti, net_cashflow, score_min=SCORE_MIN, dti_max=DTI_MAX):
    """Grille d'éligibilité : score minimum, DTI maximum et reste à vivre positif."""
    return (np.asarray(credit_score) >= score_min) & (np.asarray(dti) <= dti_max) & (np.asarray(net_cashflow) > 0)


def savings_recommendation(net_cashflow, emergency_months, net_savings_rate):
    if net_cashflow < 0:
        return "Déficit : restructurer la dette"
    if emergency_months < EMERGENCY_MONTHS:
        return "Constituer l'épargne de précaution"
    if net_savings_rate < SAVINGS_TARGET:
        return "Réduire les dépenses"
    return "Investir le surplus"


def add_recommendations(df):
    df = df.copy()
    df["recommendation"] = [savings_recommendation(c, e, s) for c, e, s in
                            zip(df.net_cashflow, df.emergency_months, df.net_savings_rate)]
    return df
