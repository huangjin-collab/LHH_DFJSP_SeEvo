import numpy as np
def get_combined_expression_v2(pt: np.ndarray, wkr: np.ndarray, rm: np.ndarray, so: np.ndarray, twk: np.ndarray, ema: np.ndarray) -> np.ndarray: 
    # Adaptive current processing time: amplify pt for jobs with positive EMA (running late)
    adjusted_pt = pt * (1 + ema)
    
    # Enhanced look-ahead urgency: include both remaining work and successor operation impact
    # Weighted sum to emphasize bottleneck traversal; high so increases priority of current job
    lookahead_urgency = rm + 0.6 * so
    
    # Uncertainty-aware workload pressure: penalize jobs with high deviation trends
    uncertain_workload = wkr * (1 + np.abs(ema))
    
    # Normalized pressure to balance across jobs of different scales
    normalized_pressure = uncertain_workload / (twk + 1e-8)
    
    # Composite expression combines adaptive local cost, global lookahead, and scaled penalty
    # Subtractive term for normalized_pressure increases priority contrast (higher pressure ¡ú lower score)
    combined_expression_data = adjusted_pt + lookahead_urgency - 0.4 * normalized_pressure
    
    # Optional: apply mild non-linearity only if EMA magnitude is high (adaptive response under high uncertainty)
    # This preserves gradient distinctions while enhancing sensitivity when needed
    mask = np.abs(ema) > 0.3  # Only amplify in high-deviation scenarios
    combined_expression_data = np.where(
        mask,
        np.tanh(combined_expression_data),
        combined_expression_data
    )
    
    return combined_expression_data
