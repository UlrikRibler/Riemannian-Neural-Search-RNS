import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial

class RiemannianHMC:
    def __init__(self, potential_fn, metric_fn, step_size=1e-3, n_steps=10):
        """
        Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) Sampler.
        
        Args:
            potential_fn: Function U(q) returning scalar potential energy.
            metric_fn: Function G(q) returning the metric tensor (or its properties).
                       For efficiency, we assume metric_fn returns (G_inv, log_det_G).
                       If G is isotropic lambda(x)^2 I, G_inv = lambda(x)^-2 I.
            step_size: Integrator step size.
            n_steps: Number of leapfrog steps per sample.
        """
        self.potential_fn = potential_fn
        self.metric_fn = metric_fn
        self.step_size = step_size
        self.n_steps = n_steps

    def kinetic_energy(self, q, p):
        """
        K(q, p) = 0.5 * p^T * G(q)^-1 * p + 0.5 * log|G(q)|
        """
        G_inv, log_det_G = self.metric_fn(q)
        
        # Assume G_inv is either a scalar (isotropic) or matrix
        # Handling isotropic case for Poincare Ball efficiency
        if jnp.ndim(G_inv) == 0 or (jnp.ndim(G_inv) == 1 and G_inv.shape[0] == 1):
            p_G_inv_p = jnp.sum(p**2) * G_inv
        else:
            # Full matrix case
            p_G_inv_p = jnp.dot(p, jnp.dot(G_inv, p))
            
        return 0.5 * p_G_inv_p + 0.5 * log_det_G

    def hamiltonian(self, q, p):
        return jnp.squeeze(self.potential_fn(q) + self.kinetic_energy(q, p))

    def _generalized_leapfrog(self, q_init, p_init):
        """
        Generalized Leapfrog Integrator for Non-Separable Hamiltonian.
        Implicit update for p requires fixed-point iteration.
        """
        dt = self.step_size
        q = q_init
        p = p_init
        
        grad_H_q = grad(self.hamiltonian, argnums=0)
        grad_H_p = grad(self.hamiltonian, argnums=1)

        for _ in range(self.n_steps):
            # 1. p half-step (Implicit)
            # p' = p - 0.5 * dt * dH/dq(q, p')
            # Fixed point iteration
            p_half = p
            for _ in range(5): # Fixed point iters
                p_half = p - 0.5 * dt * grad_H_q(q, p_half)
            
            # 2. q full-step (Implicit)
            # q' = q + 0.5 * dt * (dH/dp(q, p_half) + dH/dp(q', p_half))
            # Often approximated or also fixed point.
            # Simplified explicit for q usually works if H is separable in p, 
            # but here it's not. 
            # We use the explicit form: q_new = q + dt * dH/dp(q, p_half) 
            # (Note: Generalized Leapfrog usually updates q implicitly too if G depends on q)
            
            # Let's use the explicit approximation for q to avoid double nesting cost:
            # q_new = q + dt * G(q)^-1 * p_half
            # But strictly it should be symplectic. 
            # We stick to the standard generalized leapfrog split:
            
            # p_half = p - 0.5 * dt * grad_H_q(q, p_half)
            q_new = q
            for _ in range(5):
                 # q' = q + 0.5 * dt * (dH/dp(q, p_half) + dH/dp(q', p_half))
                 term1 = grad_H_p(q, p_half)
                 term2 = grad_H_p(q_new, p_half)
                 q_new = q + 0.5 * dt * (term1 + term2)
            
            # 3. p half-step (Explicit using new q)
            # p_new = p_half - 0.5 * dt * dH/dq(q_new, p_half)
            p_new = p_half - 0.5 * dt * grad_H_q(q_new, p_half)
            
            q, p = q_new, p_new

        return q, p

    def step(self, q_current, key):
        """
        Perform one HMC step.
        """
        # Sample momentum from N(0, G(q))
        # For simplicity in implementation, we sample standard normal z
        # and transform p = sqrt(G) * z.
        # But wait, K(p) uses G^-1. So if we want p ~ exp(-0.5 p^T G^-1 p),
        # p should be cov=G. So p = L * z where LL^T = G.
        # If G is isotropic lambda^2 * I, then L = lambda * I.
        # So p = lambda * z.
        
        G_inv, _ = self.metric_fn(q_current)
        # Assuming isotropic for sampling logic simplicity in this snippet
        # G = 1/G_inv
        # std = sqrt(G) = 1/sqrt(G_inv)
        
        std = 1.0 / jnp.sqrt(G_inv) 
        z = jax.random.normal(key, shape=q_current.shape)
        p_current = std * z

        # Integrate
        q_prop, p_prop = self._generalized_leapfrog(q_current, p_current)

        # Metropolis Correction
        current_H = self.hamiltonian(q_current, p_current)
        prop_H = self.hamiltonian(q_prop, p_prop)
        
        accept_prob = jnp.exp(current_H - prop_H) # Log-likelihoods are negative energy
        # Wait, H = U + K. P = exp(-H). Ratio = exp(-H_prop)/exp(-H_cur) = exp(H_cur - H_prop).
        # Correct.

        alpha = jax.random.uniform(key)
        
        # Accept/Reject
        q_next = jnp.where(alpha < accept_prob, q_prop, q_current)
        
        return q_next, {
            "accept_prob": accept_prob,
            "hamiltonian_error": prop_H - current_H
        }
