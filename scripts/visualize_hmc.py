import sys
import os
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial

# Add project root to path
sys.path.append(os.getcwd())

from src.geometry.manifold import PoincareBall
from src.dynamics.hmc_sampler import RiemannianHMC

def visualize_trajectory():
    print("Setting up Riemannian Simulation...")
    
    # 1. Setup Manifold
    manifold = PoincareBall(c=1.0)
    
    # 2. Define Target (Relevant Document)
    # Let's put a target at (0.5, 0.5)
    target = jnp.array([0.5, 0.5])
    
    # 3. Define Potential (The "Gravity")
    # U(q) = 0.5 * dist(q, target)^2
    def potential_fn(q):
        d = manifold.dist(q, target)
        # Squeeze to ensure scalar output for grad
        return jnp.squeeze(0.5 * d**2)
        
    # 4. Define Metric (The "Geometry")
    def metric_fn(q):
        # Isotropic metric: lambda_x^2
        lam = manifold.lambda_x(q)
        G_scalar = lam**2
        G_inv_scalar = 1.0 / G_scalar
        log_det = q.shape[-1] * jnp.log(G_scalar)
        return G_inv_scalar, log_det

    # 5. Initialize Sampler
    # Step size needs to be small for stability in curved space
    hmc = RiemannianHMC(potential_fn, metric_fn, step_size=0.05, n_steps=10)
    
    # 6. Run Simulation
    print("Running HMC chains...")
    n_samples = 200
    q_current = jnp.array([0.0, 0.0]) # Start at origin
    key = jax.random.PRNGKey(42)
    
    trajectory = [q_current]
    accept_count = 0
    
    for i in range(n_samples):
        key, subkey = jax.random.split(key)
        q_next, info = hmc.step(q_current, subkey)
        
        # Simple acceptance check (if q changed, it was accepted)
        if not jnp.allclose(q_next, q_current):
            accept_count += 1
            
        trajectory.append(q_next)
        q_current = q_next
        
        if i % 20 == 0:
            print(f"Step {i}/{n_samples} | Pos: {q_current} | Accepted: {accept_count}")

    trajectory = jnp.array(trajectory)
    
    # 7. Visualization
    print("Generating plot...")
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    
    # Draw Unit Circle (Boundary)
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linestyle='--', linewidth=2, label='Boundary')
    ax.add_artist(circle)
    
    # Draw Trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b.-', alpha=0.5, linewidth=1, label='Particle Path')
    plt.plot(trajectory[0, 0], trajectory[0, 1], 'go', label='Start (Query)')
    plt.plot(target[0], target[1], 'r*', markersize=15, label='Target (Doc)')
    
    # Styling
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("Riemannian Neural Search: HMC Particle Trajectory")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = "hmc_trajectory.png"
    plt.savefig(output_path)
    print(f"Simulation complete. Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_trajectory()
