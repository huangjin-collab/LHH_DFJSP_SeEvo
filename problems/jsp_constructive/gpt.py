import numpy as np  

def get_combined_expression_v2(pt: np.ndarray, wkr: np.ndarray, rm: np.ndarray, so: np.ndarray, twk: np.ndarray, ema: np.ndarray) -> np.ndarray:
    # EMA-adjusted processing time: inflates pt if operations are trending long (positive ema)
    adjusted_pt = pt * (1 + ema)
    
    # Look-ahead urgency: considers both remaining current task and next operation delay
    lookahead = rm + 0.5 * so
    
    # EMA-penalized remaining work: jobs with high positive ema (delayed trend) are penalized more
    penalized_wkr = wkr * (1 + np.abs(ema))
    
    # Normalized total workload contribution to balance job length awareness
    norm_twk = twk / (np.mean(twk) + 1e-8)
    
    # Combined priority: emphasizes urgency (adjusted_pt, lookahead), penalizes risky jobs (ema),
    # and de-prioritizes jobs with high overall workload unless already in progress
    combined_expression_data = (
        (adjusted_pt + 1) * 
        (lookahead + 1) * 
        (penalized_wkr + 1) / 
        (norm_twk + 1)
    )
    
    return combined_expression_data
