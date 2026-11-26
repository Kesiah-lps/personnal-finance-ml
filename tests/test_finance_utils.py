import numpy as np
import pandas as pd
import pytest

from src.finance_utils import (monthly_payment, max_loan, is_eligible, savings_recommendation,
                               add_features, load_data)


def test_monthly_payment_known_value():
    # 100 000 sur 20 ans à 5 % -> environ 659,96 par mois
    assert monthly_payment(100_000, 5, 240) == pytest.approx(659.96, abs=0.01)


def test_monthly_payment_zero_rate():
    assert monthly_payment(12_000, 0, 12) == pytest.approx(1000)


def test_monthly_payment_no_loan():
    assert monthly_payment(0, 0, 0) == 0


def test_max_loan_inverse_of_payment():
    pay = monthly_payment(150_000, 5, 240)
    assert max_loan(pay, 5, 240) == pytest.approx(150_000, rel=1e-6)


def test_eligibility_rules():
    assert is_eligible(700, 0.30, 500)
    assert not is_eligible(600, 0.30, 500)   # score trop bas
    assert not is_eligible(700, 0.50, 500)   # DTI trop haut
    assert not is_eligible(700, 0.30, -10)   # reste à vivre négatif


def test_savings_recommendation_order():
    assert savings_recommendation(-100, 20, 0.5) == "Déficit : restructurer la dette"
    assert savings_recommendation(100, 3, 0.5) == "Constituer l'épargne de précaution"
    assert savings_recommendation(100, 12, 0.1) == "Réduire les dépenses"
    assert savings_recommendation(100, 12, 0.3) == "Investir le surplus"


def test_add_features_on_small_frame():
    df = pd.DataFrame({"monthly_income_usd": [4000.0], "monthly_expenses_usd": [2000.0],
                       "monthly_emi_usd": [500.0], "savings_usd": [30000.0]})
    out = add_features(df)
    assert out.expense_ratio.iloc[0] == pytest.approx(0.5)
    assert out.net_cashflow.iloc[0] == pytest.approx(1500)
    assert out.emergency_months.iloc[0] == pytest.approx(12)
    assert out.max_emi.iloc[0] == pytest.approx(1720)   # min(0.43 * 4000, 2000)


def test_dataset_consistency():
    df = load_data()
    assert len(df) == 32424
    assert df.user_id.is_unique
    borrowers = df[df.has_loan == 1]
    pay = monthly_payment(borrowers.loan_amount_usd, borrowers.loan_interest_rate_pct, borrowers.loan_term_months)
    assert np.allclose(pay, borrowers.monthly_emi_usd, rtol=1e-3)
    assert (df.loc[df.has_loan == 0, "loan_amount_usd"] == 0).all()
