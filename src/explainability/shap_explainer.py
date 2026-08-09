"""SHAP-based explainability helpers."""


def explain_prediction(model, X):
    """Return a placeholder explanation object."""
    return {"model": model, "input_shape": X.shape}
