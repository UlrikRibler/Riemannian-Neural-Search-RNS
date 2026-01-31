import jax
import jax.numpy as jnp

def hyperbolic_info_nce_loss(query, positive, negatives, manifold, c, temperature=0.1):
    """
    Hyperbolic InfoNCE Loss (Lorentz Version).
    
    Args:
        query: (B, D+1) Query embeddings in Lorentz space.
        positive: (B, D+1) Positive document embeddings.
        negatives: (B, N, D+1) Negative document embeddings.
        manifold: Instance of LorentzModel.
        c: Curvature scalar.
        temperature: Softmax temperature.
        
    Returns:
        Scalar loss.
    """
    # 1. Calculate distance to positive
    # d_pos: (B, 1)
    d_pos = manifold.dist(query, positive, c)
    
    # 2. Calculate distances to negatives
    # Need to broadcast query to (B, N, D+1)
    query_expanded = jnp.expand_dims(query, axis=1) # (B, 1, D+1)
    
    d_neg = manifold.dist(query_expanded, negatives, c) # (B, N)
    
    # Ensure shapes
    if d_pos.ndim == 3:
        d_pos = jnp.squeeze(d_pos, axis=-1)
    if d_neg.ndim == 3:
        d_neg = jnp.squeeze(d_neg, axis=-1)
        
    # 3. Logits
    logits_pos = -d_pos / temperature # (B, 1)
    logits_neg = -d_neg / temperature # (B, N)
    
    # Concatenate: [pos, negs] -> (B, 1+N)
    logits = jnp.concatenate([logits_pos, logits_neg], axis=1)
    
    # 4. Cross Entropy
    log_prob = logits_pos - jax.scipy.special.logsumexp(logits, axis=1, keepdims=True)
    
    return -jnp.mean(log_prob)