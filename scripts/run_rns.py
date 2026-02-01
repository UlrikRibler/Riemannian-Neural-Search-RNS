import sys
import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial

# Add project root to path
sys.path.append(os.getcwd())

from src.geometry.lorentz import LorentzModel
from src.geometry.manifold import PoincareBall
from src.models.encoder import HyperbolicGCN
from src.models.density import RiemannianEBM
from src.dynamics.hmc_sampler import RiemannianHMC
from src.data.wordnet import WordNetLoader
from src.evaluation.metrics import Evaluator

def run_experiment():
    print("=== Riemannian Neural Search: Upgrade (World-Class Standard) ===")
    
    # --- 1. Data Generation (WordNet) ---
    print("\n[Phase 1] Loading WordNet Mammals Subtree...")
    loader = WordNetLoader()
    adj, features, node_list, closure = loader.load_graph()
    
    num_nodes = adj.shape[0]
    feat_dim = features.shape[1]
    print(f"Graph Loaded: {num_nodes} nodes, {feat_dim} features.")
    
    # Convert to JAX arrays
    adj = jnp.array(adj)
    features = jnp.array(features)
    # closure is used for evaluation later (numpy is fine)
    
    # --- 2. Model Training (Hyperbolic GCN) ---
    print("\n[Phase 2] Training Hyperbolic Encoder...")
    
    manifold = LorentzModel()
    # Encoder: Input -> Hidden(32) -> Output(2 dim tangent -> 3 dim Lorentz)
    # We use 2D hyperbolic space for visualization ease
    EMB_DIM = 2
    model = HyperbolicGCN([feat_dim, 32, EMB_DIM], manifold)
    
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    params = model.init_params(subkey)
    
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, x, adj):
        def loss_fn(p):
            c = jax.nn.softplus(p["c"])
            embeddings = model.forward(p, x, adj)
            
            # Sample batch for InfoNCE
            idxs = jax.random.randint(jax.random.PRNGKey(0), (64,), 0, num_nodes)
            batch_emb = embeddings[idxs]
            
            # Ground truth positives from Adjacency
            # mask_pos[i, j] = 1 if i and j are connected
            batch_adj = adj[idxs][:, idxs]
            mask_pos = batch_adj
            
            # Distances
            dists = manifold.dist(jnp.expand_dims(batch_emb, 1), jnp.expand_dims(batch_emb, 0), c)
            
            logits = -dists / 0.1
            # Remove self-contrast for softmax denominator if needed, or keep it (standard InfoNCE usually excludes self from positive set if it's the anchor)
            # Here we treat neighbors as positives. Self is typically a positive in adj (self-loop).
            
            # LogSoftmax: log( exp(pos) / sum(exp(all)) )
            exp_logits = jnp.exp(logits)
            # Avoid numerical instability
            log_prob = logits - jnp.log(exp_logits.sum(1, keepdims=True) + 1e-9)
            
            # Loss = - sum(mask * log_prob) / sum(mask)
            loss = -(mask_pos * log_prob).sum() / (mask_pos.sum() + 1e-9)
            
            loss += 0.01 * (c - 1.0)**2
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    print("Training Encoder...")
    for epoch in range(201):
        params, opt_state, loss = train_step(params, opt_state, features, adj)
        if epoch % 50 == 0:
            c_val = jax.nn.softplus(params["c"])
            print(f"Epoch {epoch}: Loss = {loss:.4f} | c = {c_val:.4f}")
            
    final_c = jax.nn.softplus(params["c"])
    final_emb_lorentz = model.forward(params, features, adj)
    final_emb_poincare = manifold.to_poincare(final_emb_lorentz, final_c)
    
    # --- 3. Learnable Density (EBM) ---
    print("\n[Phase 3] Training Riemannian Energy-Based Model...")
    
    ebm = RiemannianEBM(input_dim=EMB_DIM, hidden_dims=[32, 32], manifold=manifold)
    key, subkey = jax.random.split(key)
    ebm_params = ebm.init_params(subkey)
    
    ebm_opt = optax.adam(learning_rate=0.005)
    ebm_opt_state = ebm_opt.init(ebm_params)
    
    @jax.jit
    def ebm_train_step(ebm_params, opt_state, data_emb, key, c):
        # Generate Noise: Uniform in Poincare Ball -> Mapped to Lorentz
        # Or wrap Gaussian in Tangent Space
        key, subkey = jax.random.split(key)
        # Sample noise in tangent space of origin
        noise_tan = jax.random.normal(subkey, (data_emb.shape[0], EMB_DIM)) * 2.0 # Wide variance
        noise_emb = manifold.exp_map0(noise_tan, c)
        
        def loss_fn(p):
            return ebm.loss_fn(p, data_emb, noise_emb, c)
            
        loss, grads = jax.value_and_grad(loss_fn)(ebm_params)
        updates, opt_state = ebm_opt.update(grads, opt_state)
        ebm_params = optax.apply_updates(ebm_params, updates)
        return ebm_params, opt_state, loss

    print("Training EBM...")
    for epoch in range(201):
        key, subkey = jax.random.split(key)
        ebm_params, ebm_opt_state, loss = ebm_train_step(ebm_params, ebm_opt_state, final_emb_lorentz, subkey, final_c)
        if epoch % 50 == 0:
            print(f"EBM Epoch {epoch}: Loss = {loss:.4f}")

    # --- 4. Evaluation (Metrics) ---
    print("\n[Phase 4] Rigorous Evaluation...")
    evaluator = Evaluator(np.array(final_emb_poincare), closure)
    
    # Select test queries (random subset of nodes)
    # We select nodes that have at least one child/descendant
    valid_queries = [i for i in range(num_nodes) if np.sum(closure[i]) > 1] # >1 because self is included
    np.random.seed(42)
    test_queries = np.random.choice(valid_queries, size=min(20, len(valid_queries)), replace=False)
    
    print(f"Evaluating on {len(test_queries)} queries...")
    
    # Baseline: Nearest Neighbor
    results_nn = evaluator.baseline_search(test_queries, k=10)
    metrics_nn = evaluator.compute_metrics(test_queries, results_nn)
    print("Baseline (Nearest Neighbor) Metrics:")
    print(metrics_nn)
    
    # HMC Search Evaluation? 
    # Usually HMC is for finding ONE mode or sampling. 
    # For Retrieval, we might just define the score as EBM energy?
    # Or we can run HMC from origin and see where it lands (Generative)?
    # The prompt asks to "Compare HMC-based search against... Nearest Neighbor".
    # This implies HMC is used to FIND items.
    # Task: Given Query Q, find relevant items.
    # Potential U(z) = Dist(z, Q) + lambda * EBM(z).
    # We sample K particles.
    
    print("Evaluating HMC-based Search...")
    
    # Create HMC sampler
    # We define a JIT-compiled function for the loop to avoid recompilation
    
    @jax.jit
    def single_query_search(key, start_q, target_emb, ebm_params, c_val):
        # Re-create manifold with the scalar/tracer c_val
        pm = PoincareBall(c=c_val)
        
        def potential(q):
            # 1. Query Relevance
            d = pm.dist(q, target_emb)
            u_query = 2.0 * d 
            
            # 2. Density Support
            q_lorentz = pm.to_lorentz(q)
            e_val = ebm.forward(ebm_params, jnp.expand_dims(q_lorentz, 0), c_val)
            u_density = jnp.squeeze(e_val)
            
            return u_query + 0.5 * u_density
            
        def metric(q):
            lam = pm.lambda_x(q)
            return 1.0 / (lam**2), q.shape[-1] * jnp.log(lam**2)
            
        # Initialize HMC inside JIT
        hmc = RiemannianHMC(potential, metric, step_size=0.1, n_steps=3)
        
        def scan_step(carry, key):
            q_curr = carry
            q_next, info = hmc.step(q_curr, key)
            return q_next, q_next
            
        # Burn-in + Samples
        total_steps = 15 # 10 burn + 5 samples
        keys = jax.random.split(key, total_steps)
        final_q, samples = jax.lax.scan(scan_step, start_q, keys)
        
        return samples[-5:] # Return last 5
    
    results_hmc = []
    print(f"Running optimized HMC search for {len(test_queries)} queries...")
    
    for i, q_idx in enumerate(test_queries):
        target_emb = final_emb_poincare[q_idx]
        start_q = jnp.zeros(EMB_DIM)
        key_search = jax.random.PRNGKey(q_idx)
        
        samples = single_query_search(key_search, start_q, target_emb, ebm_params, final_c)
        
        # Convert samples to retrieved indices
        # This part is fast enough in python or can be jitted too
        retrieved_indices = []
        for sample in samples:
            dists = jnp.linalg.norm(final_emb_poincare - sample, axis=1)
            idx = jnp.argmin(dists)
            retrieved_indices.append(int(idx))
            
        results_hmc.append(list(set(retrieved_indices)))
        
    metrics_hmc = evaluator.compute_metrics(test_queries, results_hmc)
    print("HMC Search Metrics:")
    print(metrics_hmc)
    # --- 5. Visualization (Optional) ---
    print("\n[Phase 5] Saving Visualization...")
    plt.figure(figsize=(10, 10))
    radius = 1.0 / jnp.sqrt(final_c)
    circle = plt.Circle((0, 0), radius, color='k', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    
    # Plot all nodes
    plt.scatter(final_emb_poincare[:, 0], final_emb_poincare[:, 1], alpha=0.5, s=20, c='b', label='WordNet Nodes')
    
    # Highlight a query and its targets
    q_demo = test_queries[0]
    targets = np.where(closure[q_demo] > 0)[0]
    
    plt.scatter(final_emb_poincare[q_demo, 0], final_emb_poincare[q_demo, 1], c='r', s=100, marker='*', label='Query')
    plt.scatter(final_emb_poincare[targets, 0], final_emb_poincare[targets, 1], c='g', s=50, alpha=0.5, label='Entailment')
    
    plt.xlim(-radius*1.1, radius*1.1)
    plt.ylim(-radius*1.1, radius*1.1)
    plt.legend()
    plt.title(f"RNS Upgrade: WordNet Mammals (MAP={metrics_hmc['mAP']:.3f})")
    plt.savefig("rns_wordnet_result.png")
    print("Result saved.")

if __name__ == "__main__":
    run_experiment()
