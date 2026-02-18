# src/models/__init__.py
"""
Model-related modules for the Loan Approval System.
Includes training, evaluation, explainability, and fairness components.
"""

from .explainer import LoanExplainer, generate_explanation

__all__ = ['LoanExplainer', 'generate_explanation']
