"""Feature engineering sub-package.

Four layers of features for hallucination detection:
  Layer 1: Cross-model disagreement features (≥16)
  Layer 2: Confidence features (≥8)
  Layer 3: NLI + AST verification features (≥4)
  Layer 4: Task context features (≥6)
"""
from .extractor import FeatureExtractor

__all__ = ["FeatureExtractor"]
