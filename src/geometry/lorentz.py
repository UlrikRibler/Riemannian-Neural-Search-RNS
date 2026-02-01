import jax
import jax.numpy as jnp

class LorentzModel:
    def __init__(self):
        """
        Lorentz (Hyperboloid) Manifold Model.
        
        This class serves as the primary geometric backend for the Riemannian Neural Search system.
        It implements the Hyperboloid model of hyperbolic space, defined as:
        H^n = {x \in R^{n+1} : <x, x>_L = -1/c, x_0 > 0}
        
        The Lorentz model is preferred for optimization and numerical stability compared 
        to the Poincaré ball, especially for deep learning (Hyperbolic GCNs).
        
        Metric signature: (-1, 1, 1, ... 1).
        """
        pass

    def minkowski_dot(self, x, y):
        """
        Minkowski inner product: -x0*y0 + x1*y1 + ... + xn*yn
        """
        # x: (..., D+1)
        res = -x[..., 0:1] * y[..., 0:1] + jnp.sum(x[..., 1:] * y[..., 1:], axis=-1, keepdims=True)
        return res

    def dist(self, x, y, c=1.0):
        """
        Squared Lorentzian distance is often used, but here geodesic distance:
        d(x, y) = 1/sqrt(c) * acosh( -c * <x, y>_L )
        """
        prod = self.minkowski_dot(x, y)
        # Numerical stability clip. prod should be <= -1/c (for c=1, <= -1)
        # -c * prod >= 1
        theta = jnp.clip(-c * prod, a_min=1.0 + 1e-7)
        dist = (1.0 / jnp.sqrt(c)) * jnp.arccosh(theta)
        return dist

    def exp_map0(self, v, c=1.0):
        """
        Exponential map at the origin (1/sqrt(c), 0, ..., 0).
        Input v is in Tangent space at origin (which is just Euclidean R^n ~ {0}xR^n).
        We assume v is (..., D) and we effectively pad it to (0, v).
        """
        # v: (..., D)
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        
        # We need to map v -> (sqrt(c)*cosh, sinh*v/norm) roughly
        # Origin o = (1/sqrt(c), 0, ...)
        
        sqrt_c = jnp.sqrt(c)
        
        # x = (cosh(sqrt(c)*|v|)/sqrt(c), sinh(sqrt(c)*|v|) * v / (sqrt(c)*|v|))
        
        # Avoid div by zero
        # sinh(x)/x -> 1 as x->0
        
        theta = sqrt_c * v_norm
        
        # Time component (x0)
        res_0 = jnp.cosh(theta) / sqrt_c
        
        # Space components (x1...xn)
        # sinh(theta) * v / (theta/sqrt_c * sqrt_c) ???
        # factor = sinh(theta) / (sqrt_c * v_norm) = sinh(theta)/theta
        
        factor = jnp.sinh(theta) / (theta + 1e-15)
        res_rest = factor * v
        
        # Concatenate
        res = jnp.concatenate([res_0, res_rest], axis=-1)
        return res

    def log_map0(self, x, c=1.0):
        """
        Log map at origin O=(1/sqrt(c), 0...).
        Maps x (D+1) -> v (D) in tangent space.
        """
        # x: (..., D+1)
        # x0 is time, x1..n is space
        
        # Inner prod with origin. 
        # O = (1/sqrt(c), 0...)
        # <x, O>_L = -x0 * (1/sqrt(c))
        
        x0 = x[..., 0:1]
        x_rest = x[..., 1:]
        
        sqrt_c = jnp.sqrt(c)
        
        # dist d = 1/sqrt(c) * acosh(sqrt(c) * x0)
        # But we can compute directly.
        
        # formula: u = (d / sinh(sqrt(c)*d)) * (x - <x,o>o) ?? 
        # Simpler:
        # Direction is x_rest.
        # Norm of x_rest is sinh(sqrt(c)*d)/sqrt(c).
        
        x_rest_norm = jnp.linalg.norm(x_rest, axis=-1, keepdims=True)
        
        # d = 1/sqrt(c) * acosh(sqrt(c)*x0)
        theta = jnp.clip(sqrt_c * x0, a_min=1.0 + 1e-7)
        d = (1.0 / sqrt_c) * jnp.arccosh(theta)
        
        # v = d * (x_rest / x_rest_norm)
        
        factor = d / (x_rest_norm + 1e-15)
        return factor * x_rest

    def to_poincare(self, x, c=1.0):
        """
        Project Lorentz (D+1) -> Poincare Ball (D).
        Formula: x_P = x_{1..n} / (1/sqrt(c) + x_0)
        """
        x0 = x[..., 0:1]
        x_rest = x[..., 1:]
        
        denom = (1.0 / jnp.sqrt(c)) + x0
        return x_rest / (denom + 1e-15)
