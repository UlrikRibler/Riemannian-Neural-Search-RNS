import sys
import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
from functools import partial

# Add project root to path
sys.path.append(os.getcwd())

from src.geometry.lorentz import LorentzModel
from src.geometry.manifold import PoincareBall # For visualization/HMC conversion
from src.models.encoder import HyperbolicGCN
from src.models.losses import hyperbolic_info_nce_loss
from src.dynamics.hmc_sampler import RiemannianHMC
from src.data.synthetic import generate_sbm_graph

def run_experiment():
    print("=== Riemannian Neural Search: Upgrade (Lorentz + Learnable c + Density Metric) ===")
    
    # --- 1. Data Generation ---
    print("\n[Phase 1] Generating Synthetic Web Graph...")
    key = jax.random.PRNGKey(42)
    
    NUM_NODES = 100
    NUM_CLUSTERS = 3 
    FEAT_DIM = 16
    
    key, subkey = jax.random.split(key)
    adj, features, labels = generate_sbm_graph(subkey, NUM_NODES, NUM_CLUSTERS, feature_dim=FEAT_DIM)
    
    print(f"Graph Created: {NUM_NODES} nodes, {NUM_CLUSTERS} clusters.")
    
    # --- 2. Model Training (Lorentz + Learnable c) ---
    print("\n[Phase 2] Training Hyperbolic Index (Lorentz Backend)...")
    
    manifold = LorentzModel()
    # Encoder: Input(16) -> Hidden(32) -> Output(2)
    # Output is 2 dim in Tangent space -> 3 dim on Hyperboloid
    model = HyperbolicGCN([FEAT_DIM, 32, 2], manifold)
    
    key, subkey = jax.random.split(key)
    params = model.init_params(subkey)
    
    # Optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, x, adj):
        def loss_fn(p):
            c = jax.nn.softplus(p["c"])
            embeddings = model.forward(p, x, adj) # Embeddings are (N, 3) Lorentz
            
            # Pairwise distance matrix for InfoNCE
            # We sample a batch for efficiency
            idxs = jax.random.randint(jax.random.PRNGKey(0), (32,), 0, NUM_NODES)
            batch_emb = embeddings[idxs]
            batch_labels = labels[idxs]
            
            dists = manifold.dist(jnp.expand_dims(batch_emb, 1), jnp.expand_dims(batch_emb, 0), c)
            
            label_match = batch_labels[:, None] == batch_labels[None, :]
            mask_pos = label_match.astype(jnp.float32) - jnp.eye(32)
            
            logits = -dists / 0.1
            
            exp_logits = jnp.exp(logits) * (1 - jnp.eye(32))
            log_prob = logits - jnp.log(exp_logits.sum(1, keepdims=True) + 1e-9)
            
            loss = -(mask_pos * log_prob).sum() / (mask_pos.sum() + 1e-9)
            
            # Regularizer for c to prevent explosion/collapse
            loss += 0.01 * (c - 1.0)**2
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Train Loop
    print("Training...")
    for epoch in range(201):
        params, opt_state, loss = train_step(params, opt_state, features, adj)
        if epoch % 50 == 0:
            c_val = jax.nn.softplus(params["c"])
            print(f"Epoch {epoch}: Loss = {loss:.4f} | Curvature c = {c_val:.4f}")

    # Indexing: Get final embeddings in Lorentz
    final_emb_lorentz = model.forward(params, features, adj)
    final_c = jax.nn.softplus(params["c"])
    
    # Convert to Poincaré for Visualization/HMC
    final_emb_poincare = manifold.to_poincare(final_emb_lorentz, final_c)
    
    print(f"\nFinal Curvature Learned: {final_c:.4f}")
    
    # --- 3. Retrieval (Data-Driven HMC) ---
    print("\n[Phase 3] The 'Physicist' Search (Data-Driven)...")
    
    # Create a Poincaré Manifold instance with the LEARNED curvature
    poincare_manifold = PoincareBall(c=final_c)
    
    # Targets (Centroids of Cluster 0 and 1) in Poincaré
    mask0 = labels == 0
    mask1 = labels == 1
    # Simple arithmetic mean is not geodesic center, but close enough for demo in Poincare
    c0 = jnp.mean(final_emb_poincare[mask0], axis=0)
    c1 = jnp.mean(final_emb_poincare[mask1], axis=0)
    
    # DATA-DRIVEN POTENTIAL
    # We add a term that "pulls" the particle towards high-density regions of documents.
    # U_density(q) = -alpha * log( sum_i exp( -d(q, doc_i)^2 / sigma ) )
    # This is Kernel Density Estimation.
    
    # docs_subset = final_emb_poincare[::10] # Still too slow
    # APPROXIMATION: Use Cluster Centers as proxy for density peaks
    # This simulates G(x) being high near known topics.
    density_proxies = jnp.stack([c0, c1]) # Use the relevant centroids
    
    def search_potential(q):
        # 1. Query Relevance (Ambiguous: close to c0 AND c1)
        d0 = poincare_manifold.dist(q, c0)
        d1 = poincare_manifold.dist(q, c1)
        u_query = -jnp.log(jnp.exp(-2.0 * d0**2) + jnp.exp(-2.0 * d1**2) + 1e-9)
        
        # 2. Data Density (The "Data Driven" Part)
        # Pull towards actual documents to avoid "voids"
        d_docs = poincare_manifold.dist(jnp.expand_dims(q, 0), density_proxies)
        # KDE energy
        u_density = -0.5 * jax.scipy.special.logsumexp(-d_docs**2 / 0.2)
        
        return jnp.squeeze(u_query + u_density)
        
    def metric_fn(q):
        # Base metric: Conformal factor
        lam = poincare_manifold.lambda_x(q)
        # We can also modulate mass by density here, but Potential modification is often more stable 
        # for shaping the landscape than changing the Kinetic energy metric G directly 
        # (which affects step size adaptation).
        # We stick to the manifold metric for G.
        return 1.0 / (lam**2), q.shape[-1] * jnp.log(lam**2)

    hmc = RiemannianHMC(search_potential, metric_fn, step_size=0.05, n_steps=5)
    
    # Run Search
    q_curr = jnp.zeros(2) 
    trajectory = []
    
    key = jax.random.PRNGKey(202)
    for _ in range(50):
        key, subkey = jax.random.split(key)
        q_curr, _ = hmc.step(q_curr, subkey)
        trajectory.append(q_curr)
        
    trajectory = jnp.array(trajectory)
    
    # --- 4. Visualization ---
    print("\n[Phase 4] Visualizing Enhanced Search...")
    plt.figure(figsize=(10, 10))
    
    # Boundary (Poincaré radius = 1/sqrt(c))
    radius = 1.0 / jnp.sqrt(final_c)
    circle = plt.Circle((0, 0), radius, color='black', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    
    # Documents
    colors = ['r', 'g', 'b']
    for i in range(NUM_CLUSTERS):
        cluster_pts = final_emb_poincare[labels == i]
        plt.scatter(cluster_pts[:, 0], cluster_pts[:, 1], c=colors[i], alpha=0.3, label=f'Topic {i}')
        
    # Search
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'k.-', alpha=0.6, linewidth=1.0, label='HMC Searcher')
    plt.plot(c0[0], c0[1], 'rx', markersize=12, label='Target 0')
    plt.plot(c1[0], c1[1], 'gx', markersize=12, label='Target 1')
    
    plt.xlim(-radius*1.1, radius*1.1)
    plt.ylim(-radius*1.1, radius*1.1)
    plt.legend()
    plt.title(f"RNS Upgrade: Lorentz Training (c={final_c:.2f}) + Density-Aware HMC")
    plt.savefig("rns_upgrade_result.png")
    print("Result saved to 'rns_upgrade_result.png'")
    
if __name__ == "__main__":
    run_experiment()