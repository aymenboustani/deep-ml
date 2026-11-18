import numpy as np

def detect_attention_sinks(attn_weights: np.ndarray, threshold: float) -> dict:
    """
    Detect attention sink tokens from multi-head attention weight matrices.
    
    Args:
        attn_weights: Attention weights of shape (num_heads, seq_len, seq_len)
        threshold: Minimum average received attention to qualify as a sink
        
    Returns:
        Dictionary with 'sink_positions', 'avg_attention_received', and 'sink_scores'
    """
    # Your code here
    A_ = attn_weights.mean(axis=0)
    r = A_.mean(axis=0)
    sink = r >= threshold
    sink_pos = np.where(sink)[0]
    return {
        'sink_positions': sink_pos,
        'avg_attention_received': r,
        'sink_scores': r[sink]
    }