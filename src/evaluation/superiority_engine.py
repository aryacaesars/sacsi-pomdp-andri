"""Evidence-based superiority classification."""

def classify_comparison(delta, p_value, effect_size, alpha=0.05):
    if delta > 0 and p_value < alpha and effect_size >= 0.3:
        return "STATISTICALLY_SUPPORTED_IMPROVEMENT"
    if delta > 0:
        return "DESCRIPTIVE_IMPROVEMENT"
    return "NO_SUPPORTED_IMPROVEMENT"
