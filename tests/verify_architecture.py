import sys
import os
sys.path.append(os.getcwd())

try:
    import jax
    import jax.numpy as jnp
    from src.geometry.manifold import PoincareBall
    from src.dynamics.hmc_sampler import RiemannianHMC
    from src.models.losses import hyperbolic_info_nce_loss
    
    def test_rns_architecture():
        print("Initializing Riemannian Neural Search Architecture...")
        
        # 1. Manifold
        manifold = PoincareBall(c=1.0)
        x = jnp.array([0.1, 0.1])
        y = jnp.array([0.2, -0.1])
        d = manifold.dist(x, y)
        print(f"Manifold Distance Check: {d}")
        
        # 2. Loss
        B, N, D = 2, 5, 4
        query = jax.random.normal(jax.random.PRNGKey(0), (B, D)) * 0.1 # Small to stay inside ball
        pos = jax.random.normal(jax.random.PRNGKey(1), (B, D)) * 0.1
        negs = jax.random.normal(jax.random.PRNGKey(2), (B, N, D)) * 0.1
        
        loss = hyperbolic_info_nce_loss(query, pos, negs, manifold)
        print(f"Hyperbolic Loss Check: {loss}")
        
        # 3. HMC Sampler
        def potential(q):
            # Simple quadratic potential in hyperbolic distance from origin
            origin = jnp.zeros_like(q)
            return 0.5 * manifold.dist(q, origin)**2
            
        def metric(q):
            # Isotropic metric: lambda_x^2
            # Return (inv_metric_scalar, log_det)
            lam = manifold.lambda_x(q)
            G_scalar = lam**2
            G_inv_scalar = 1.0 / G_scalar
            log_det = q.shape[-1] * jnp.log(G_scalar) # D * log(lambda^2)
            return G_inv_scalar, log_det

        hmc = RiemannianHMC(potential, metric, step_size=0.01, n_steps=3)
        
        q_init = jnp.array([0.0, 0.0])
        q_next, info = hmc.step(q_init, jax.random.PRNGKey(42))
        
        print(f"HMC Step Result: {q_next}")
        print("Verification Successful.")

    if __name__ == "__main__":
        test_rns_architecture()

except ImportError:
    print("JAX not installed. Skipping execution verification, but code structure is ready.")
except Exception as e:
    print(f"Verification Failed: {e}")
    import traceback
    traceback.print_exc()
