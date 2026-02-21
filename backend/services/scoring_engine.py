def calculate_final_score(ela_score: float, layout_score: float, dl_score: float = 0.5):
    """
    Combines ELA, Layout, and Deep Learning scores into a final weighted fraud confidence score.
    Returns: (final_score, classification)
    """
    # Weights: 
    # DL (Vision Transformer) is very powerful for localized forgery
    # ELA (Pixel level) is definitive for compression anomalies
    # Layout (Structural) for global consistency
    W_ELA = 0.3
    W_DL = 0.5
    W_LAYOUT = 0.2
    
    final_score = (ela_score * W_ELA) + (dl_score * W_DL) + (layout_score * W_LAYOUT)
    final_score_pct = float(final_score * 100)
    
    classification = "Authentic"
    if final_score_pct > 70:
        classification = "Highly Forged"
    elif final_score_pct > 30:
        classification = "Suspicious"
        
    return round(final_score_pct, 2), classification
