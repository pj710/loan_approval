"""
Loan Approval Explainability Module

This module provides comprehensive model explainability for the loan approval system,
including SHAP-based explanations, feature importance analysis, and human-readable
explanation generation for underwriters.

Author: Josiah Gordor
Date: February 2026
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
warnings.filterwarnings('ignore')


class LoanExplainer:
    """
    A comprehensive explainability class for loan approval models.
    
    Provides:
    - Global interpretability (SHAP summary plots, feature importance)
    - Local interpretability (individual prediction explanations)
    - Human-readable explanation generation
    - Partial dependence plots
    
    Attributes:
        model: The trained model to explain
        feature_names: List of feature names
        explainer: SHAP explainer object
        background_data: Background data for SHAP calculations
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None,
        model_type: str = 'tree'
    ):
        """
        Initialize the LoanExplainer.
        
        Args:
            model: Trained model (XGBoost, RandomForest, LogisticRegression, etc.)
            feature_names: List of feature names in order
            background_data: Background data for SHAP calculations (sample of training data)
            model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
        """
        self.model = model
        self.feature_names = feature_names
        self.background_data = background_data
        self.model_type = model_type
        self.explainer = None
        self.shap_values = None
        
        # Feature category mappings for human-readable explanations
        self.feature_categories = {
            'income': ['income', 'income_bracket', 'loan_to_income_ratio'],
            'debt': ['debt_to_income_ratio', 'dti_risk_flag', 'dti_hashed'],
            'loan': ['loan_amount', 'loan_term', 'loan_size_category', 'loan_rate_interaction'],
            'property': ['property_value', 'loan_to_value_ratio', 'ltv_risk_flag'],
            'credit': ['interest_rate', 'rate_spread', 'combined_risk_score'],
            'demographics': ['applicant_age', 'has_coborrower'],
            'location': ['state_code', 'county_code', 'county_frequency']
        }
        
        # Risk thresholds for explanations
        self.risk_thresholds = {
            'dti_low': 36,
            'dti_high': 43,
            'ltv_low': 80,
            'ltv_high': 95,
            'income_low': 50000,
            'income_high': 100000
        }
        
        self._shap_cache = {}
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer based on model type."""
        try:
            if self.model_type == 'tree':
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == 'linear':
                if self.background_data is not None:
                    # LinearExplainer requires masker/background data
                    self.explainer = shap.LinearExplainer(
                        self.model, 
                        masker=self.background_data
                    )
                else:
                    raise ValueError("LinearExplainer requires background_data")
            elif self.model_type == 'deep':
                if self.background_data is not None:
                    self.explainer = shap.DeepExplainer(
                        self.model, 
                        self.background_data[:100]
                    )
            elif self.model_type == 'kernel':
                if self.background_data is not None:
                    # Use a small sample for kernel explainer (computationally expensive)
                    background_sample = shap.sample(self.background_data, 100)
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba if hasattr(self.model, 'predict_proba') 
                        else self.model.predict,
                        background_sample
                    )
            else:
                # Fallback to TreeExplainer for ensemble models
                self.explainer = shap.TreeExplainer(self.model)
                
            print(f"✅ Initialized {self.model_type} SHAP explainer")
            
        except Exception as e:
            print(f"⚠️ Could not initialize {self.model_type} explainer: {e}")
            print("   Falling back to KernelExplainer...")
            if self.background_data is not None:
                background_sample = shap.sample(self.background_data, 50)
                self.explainer = shap.KernelExplainer(
                    lambda x: self.model.predict_proba(x)[:, 1] if hasattr(self.model, 'predict_proba')
                    else self.model.predict(x),
                    background_sample
                )
    
    def compute_shap_values(
        self, 
        X: np.ndarray,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Compute SHAP values for given data. Uses caching to avoid recomputation.
        Args:
            X: Feature matrix (n_samples, n_features)
            check_additivity: Whether to check SHAP additivity (if supported)
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Provide background data.")
        key = hash(X.tobytes())
        if key in self._shap_cache:
            self.shap_values = self._shap_cache[key]
            print(f"✅ SHAP values loaded from cache: shape {self.shap_values.shape}")
            return self.shap_values
        print(f"Computing SHAP values for {X.shape[0]} samples...")
        try:
            if self.model_type == 'tree':
                try:
                    self.shap_values = self.explainer.shap_values(
                        X, check_additivity=check_additivity
                    )
                except TypeError:
                    self.shap_values = self.explainer.shap_values(X)
            else:
                self.shap_values = self.explainer.shap_values(X)
            if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                self.shap_values = self.shap_values[1]
            self._shap_cache[key] = self.shap_values
            if isinstance(self.shap_values, np.ndarray):
                print(f"✅ SHAP values computed: shape {self.shap_values.shape}")
            else:
                print(f"✅ SHAP values computed")
            return self.shap_values
        except Exception as e:
            print(f"⚠️ Error computing SHAP values: {e}")
            raise
    
    def get_feature_importance(
        self, 
        method: str = 'shap'
    ) -> pd.DataFrame:
        """
        Get feature importance rankings.
        
        Args:
            method: 'shap' for SHAP-based importance, 'model' for model's built-in importance
            
        Returns:
            DataFrame with feature importance rankings
        """
        importance_df = pd.DataFrame()
        importance_df['feature'] = self.feature_names
        
        if method == 'shap' and self.shap_values is not None:
            # Mean absolute SHAP value
            importance_df['importance'] = np.abs(self.shap_values).mean(axis=0)
        elif method == 'model':
            if hasattr(self.model, 'feature_importances_'):
                importance_df['importance'] = self.model.feature_importances_
            elif hasattr(self.model, 'coef_'):
                importance_df['importance'] = np.abs(self.model.coef_).flatten()
            else:
                raise ValueError("Model does not have feature importance attribute")
        else:
            raise ValueError(f"Invalid method: {method}. Use 'shap' or 'model'.")
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        return importance_df.reset_index(drop=True)
    
    def plot_summary(
        self, 
        X: np.ndarray,
        max_display: int = 20,
        plot_type: str = 'bar',
        save_path: Optional[str] = None
    ):
        """
        Generate SHAP summary plot.
        
        Args:
            X: Feature matrix for plotting
            max_display: Maximum number of features to display
            plot_type: 'bar' for bar plot, 'dot' for beeswarm plot
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'bar':
            shap.summary_plot(
                self.shap_values, 
                X,
                feature_names=self.feature_names,
                plot_type='bar',
                max_display=max_display,
                show=False
            )
        else:
            shap.summary_plot(
                self.shap_values, 
                X,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
        
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_force(
        self, 
        X: np.ndarray,
        index: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Generate SHAP force plot for a single prediction.
        
        Args:
            X: Feature matrix
            index: Index of the sample to explain
            save_path: Path to save the plot (HTML format)
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get expected value (base value)
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]  # Positive class
        
        # Create force plot
        force_plot = shap.force_plot(
            expected_value,
            self.shap_values[index],
            X[index],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f'SHAP Force Plot - Sample {index}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            if save_path.endswith('.html'):
                shap.save_html(save_path, force_plot)
            else:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Force plot saved to {save_path}")
        
        plt.show()
    
    def plot_waterfall(
        self,
        X: np.ndarray,
        index: int = 0,
        max_display: int = 15,
        save_path: Optional[str] = None
    ):
        """
        Generate SHAP waterfall plot for a single prediction.
        
        Args:
            X: Feature matrix
            index: Index of the sample to explain
            max_display: Maximum features to display
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        # Get expected value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, list):
            expected_value = expected_value[1]
        
        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=self.shap_values[index],
            base_values=expected_value,
            data=X[index],
            feature_names=self.feature_names
        )
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(f'SHAP Waterfall Plot - Sample {index}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Waterfall plot saved to {save_path}")
        
        plt.show()
    
    def plot_dependence(
        self,
        X: np.ndarray,
        feature: str,
        interaction_feature: Optional[str] = 'auto',
        save_path: Optional[str] = None
    ):
        """
        Generate SHAP dependence plot.
        
        Args:
            X: Feature matrix
            feature: Feature name to analyze
            interaction_feature: Feature for interaction coloring ('auto' to auto-detect)
            save_path: Path to save the plot
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        if feature not in self.feature_names:
            raise ValueError(f"Feature '{feature}' not found in feature names")
        
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            feature_names=self.feature_names,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Dependence plot saved to {save_path}")
        
        plt.show()
    
    def explain_prediction(
        self,
        X: np.ndarray,
        index: int = 0,
        top_k: int = 5,
        include_values: bool = True
    ) -> Dict[str, Any]:
        """
        Generate detailed explanation for a single prediction. Vectorized for performance.
        Args:
            X: Feature matrix
            index: Index of the sample to explain
            top_k: Number of top contributing features
            include_values: Whether to include feature values in explanation
        Returns:
            Dictionary with prediction explanation
        """
        if self.shap_values is None:
            self.compute_shap_values(X)
        if hasattr(self.model, 'predict_proba'):
            prob = self.model.predict_proba(X[index:index+1])[0, 1]
            pred = 1 if prob > 0.5 else 0
        else:
            pred = self.model.predict(X[index:index+1])[0]
            prob = float(pred)
        shap_vals = self.shap_values[index]
        feature_vals = X[index]
        # Vectorized top contributors
        pos_mask = shap_vals > 0
        neg_mask = shap_vals < 0
        pos_indices = np.argsort(shap_vals)[-top_k:][::-1]
        neg_indices = np.argsort(shap_vals)[:top_k]
        explanation = {
            'prediction': 'Approved' if pred == 1 else 'Denied',
            'probability': float(prob),
            'confidence': float(abs(prob - 0.5) * 200),
            'top_positive_factors': [],
            'top_negative_factors': [],
            'expected_value': float(self.explainer.expected_value[1] 
                                   if isinstance(self.explainer.expected_value, list)
                                   else self.explainer.expected_value)
        }
        for idx in pos_indices:
            if shap_vals[idx] > 0:
                factor = {
                    'feature': self.feature_names[idx],
                    'shap_value': float(shap_vals[idx]),
                    'direction': 'positive'
                }
                if include_values:
                    factor['value'] = float(feature_vals[idx])
                explanation['top_positive_factors'].append(factor)
        for idx in neg_indices:
            if shap_vals[idx] < 0:
                factor = {
                    'feature': self.feature_names[idx],
                    'shap_value': float(shap_vals[idx]),
                    'direction': 'negative'
                }
                if include_values:
                    factor['value'] = float(feature_vals[idx])
                explanation['top_negative_factors'].append(factor)
        return explanation
    
    def generate_human_explanation(
        self,
        X: np.ndarray,
        index: int = 0,
        feature_value_map: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate human-readable explanation for underwriters.
        
        Args:
            X: Feature matrix
            index: Index of the sample
            feature_value_map: Optional dictionary mapping features to readable values
            
        Returns:
            Human-readable explanation string
        """
        explanation = self.explain_prediction(X, index, top_k=5, include_values=True)
        
        # Build explanation string
        decision = explanation['prediction']
        prob = explanation['probability'] * 100
        confidence = explanation['confidence']
        
        lines = []
        lines.append(f"📋 LOAN DECISION: {decision.upper()}")
        lines.append(f"   Approval Probability: {prob:.1f}%")
        lines.append(f"   Confidence Level: {confidence:.0f}%")
        lines.append("")
        
        if explanation['top_positive_factors']:
            lines.append("✅ FACTORS SUPPORTING APPROVAL:")
            for factor in explanation['top_positive_factors'][:3]:
                feature = factor['feature'].replace('_', ' ').title()
                impact = abs(factor['shap_value'])
                lines.append(f"   • {feature} (impact: +{impact:.3f})")
            lines.append("")
        
        if explanation['top_negative_factors']:
            lines.append("⚠️ RISK FACTORS / CONCERNS:")
            for factor in explanation['top_negative_factors'][:3]:
                feature = factor['feature'].replace('_', ' ').title()
                impact = abs(factor['shap_value'])
                lines.append(f"   • {feature} (impact: -{impact:.3f})")
            lines.append("")
        
        # Add recommendation
        if prob >= 75:
            lines.append("💡 RECOMMENDATION: Strong candidate for approval")
        elif prob >= 50:
            lines.append("💡 RECOMMENDATION: Review required - borderline case")
        elif prob >= 25:
            lines.append("💡 RECOMMENDATION: High risk - consider additional documentation")
        else:
            lines.append("💡 RECOMMENDATION: Likely denial - significant risk factors present")
        
        return "\n".join(lines)


def generate_explanation(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    index: int = 0,
    background_data: Optional[np.ndarray] = None,
    model_type: str = 'tree'
) -> Dict[str, Any]:
    """
    Convenience function to generate explanation for a single prediction.
    
    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        index: Index of sample to explain
        background_data: Background data for SHAP
        model_type: Type of model
        
    Returns:
        Explanation dictionary
    """
    explainer = LoanExplainer(
        model=model,
        feature_names=feature_names,
        background_data=background_data,
        model_type=model_type
    )
    
    explainer.compute_shap_values(X)
    return explainer.explain_prediction(X, index)


def create_explanation_report(
    explainer: LoanExplainer,
    X: np.ndarray,
    y: np.ndarray,
    sample_size: int = 3,
    output_dir: str = 'reports/explanations'
) -> Dict[str, Any]:
    """
    Create a comprehensive explanation report with example cases.
    
    Args:
        explainer: Initialized LoanExplainer
        X: Feature matrix
        y: True labels
        sample_size: Number of examples per category
        output_dir: Directory to save report artifacts
        
    Returns:
        Report summary dictionary
    """
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get predictions
    if hasattr(explainer.model, 'predict_proba'):
        probs = explainer.model.predict_proba(X)[:, 1]
    else:
        probs = explainer.model.predict(X)
    
    preds = (probs > 0.5).astype(int)
    
    # Find example cases
    approved_correct = np.where((preds == 1) & (y == 1))[0]
    denied_correct = np.where((preds == 0) & (y == 0))[0]
    borderline = np.where((probs > 0.4) & (probs < 0.6))[0]
    false_positives = np.where((preds == 1) & (y == 0))[0]
    false_negatives = np.where((preds == 0) & (y == 1))[0]
    
    report = {
        'summary': {
            'total_samples': len(X),
            'approved': int(preds.sum()),
            'denied': int(len(preds) - preds.sum()),
            'accuracy': float((preds == y).mean())
        },
        'examples': {}
    }
    
    # Generate example explanations
    categories = {
        'approved_correct': approved_correct[:sample_size],
        'denied_correct': denied_correct[:sample_size],
        'borderline': borderline[:sample_size],
        'false_positives': false_positives[:sample_size],
        'false_negatives': false_negatives[:sample_size]
    }
    
    for category, indices in categories.items():
        report['examples'][category] = []
        for idx in indices:
            explanation = explainer.generate_human_explanation(X, idx)
            report['examples'][category].append({
                'index': int(idx),
                'explanation': explanation,
                'probability': float(probs[idx]),
                'actual': int(y[idx])
            })
    
    return report


# Explanation templates for underwriters
EXPLANATION_TEMPLATES = {
    'approved_high_confidence': """
APPROVAL RECOMMENDED (High Confidence: {confidence}%)

This application demonstrates strong creditworthiness based on 
{positive_factors}

While the following factors were noted, they do not significantly impact the recommendation:
{minor_concerns}

Suggested Action: Proceed to final documentation review.
""",
    'approved_moderate_confidence': """
APPROVAL RECOMMENDED (Moderate Confidence: {confidence}%)

Positive factors supporting approval:
{positive_factors}

Areas requiring attention:
{concerns}

Suggested Action: Review the noted concerns and request additional documentation if needed.
""",

    'denied_recommendation': """
DENIAL RECOMMENDED (Confidence: {confidence}%)

Primary risk factors:
{risk_factors}

Mitigating factors present:
{mitigating_factors}

Suggested Action: Consider denial or request substantial additional documentation.
Possible remediation paths for applicant: {remediation_suggestions}
""",

    'borderline_case': """
MANUAL REVIEW REQUIRED (Confidence: {confidence}%)

The model is unable to make a clear recommendation for this application.

Factors supporting approval:
{positive_factors}

Factors suggesting denial:
{negative_factors}

Suggested Action: Senior underwriter review recommended. Consider:
- Requesting additional income verification
- Reviewing comparable approved applications
- Assessing current market conditions
"""
}


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Loan Explainability Module")
    print("=" * 60)
    print("\nThis module provides:")
    print("  - LoanExplainer class for model interpretability")
    print("  - SHAP-based global and local explanations")
    print("  - Human-readable explanation generation")
    print("  - Partial dependence and interaction analysis")
    print("\nUsage:")
    print("  from src.models.explainer import LoanExplainer, generate_explanation")
    print("  explainer = LoanExplainer(model, feature_names, background_data)")
    print("  explainer.plot_summary(X_test)")
    print("  explanation = explainer.generate_human_explanation(X_test, index=0)")
