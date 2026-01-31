import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
from src.geometry.manifold import PoincareBall
from src.models.encoder import HyperbolicGCN

def test_hgcn():
    print("Initializing Hyperbolic GCN...")
    
    # 1. Setup
    manifold = PoincareBall(c=1.0)
    key = jax.random.PRNGKey(0)
    
    # 2. Data (Graph)
    num_nodes = 10
    in_dim = 8
    hidden_dim = 16
    out_dim = 4
    
    # Random Adjacency (Self-loops included)
    adj = jnp.eye(num_nodes) + 0.5 * jax.random.bernoulli(key, 0.3, (num_nodes, num_nodes)).astype(jnp.float32)
    # Row normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    adj = adj / row_sums
    
    # Random Features (Euclidean initially)
    feat_key, key = jax.random.split(key)
    x_euc = jax.random.normal(feat_key, (num_nodes, in_dim))
    
    # Map to Manifold (Input to HGCN)
    x_hyp = manifold.exp_map0(x_euc)
    
    print(f"Input Shape: {x_hyp.shape}")
    print(f"Input Norms: {jnp.linalg.norm(x_hyp, axis=-1)}") # Should be < 1
    
    # 3. Model
    hgcn = HyperbolicGCN([in_dim, hidden_dim, out_dim], manifold)
    params = hgcn.init_params(key)
    
    # 4. Forward Pass
    h_out = hgcn.forward(params, x_hyp, adj)
    
    print(f"Output Shape: {h_out.shape}")
    print(f"Output Norms: {jnp.linalg.norm(h_out, axis=-1)}")
    
    # Verify outputs are in ball
    is_valid = jnp.all(jnp.linalg.norm(h_out, axis=-1) < 1.0/jnp.sqrt(manifold.c))
    print(f"All outputs inside Poincaré Ball? {is_valid}")
    
    if is_valid:
        print("HGCN Verification Successful.")
    else:
        print("HGCN Verification Failed: Points escaped the manifold.")

if __name__ == "__main__":
    test_hgcn()
