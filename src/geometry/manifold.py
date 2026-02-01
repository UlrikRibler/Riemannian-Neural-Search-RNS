import jax
import jax.numpy as jnp
from functools import partial

class PoincareBall:
    def __init__(self, c=1.0):
        """
        Poincaré Ball Model of Hyperbolic Space.
        
        Used primarily for:
        1. Visualization (projection to the unit disk).
        2. Riemannian HMC (explicit metric tensor G(x) is simpler in Poincaré coordinates).
        
        Args:
            c: Curvature parameter (c = -K). A higher c implies higher curvature 
               and a smaller 'radius' (1/sqrt(c)) for the boundary.
        """
        self.c = c

    def lambda_x(self, x):
        """Calculates the conformal factor lambda_x = 2 / (1 - c * ||x||^2)."""
        x_norm_sq = jnp.sum(x**2, axis=-1, keepdims=True)
        return 2.0 / (1.0 - self.c * x_norm_sq)

    def mobius_add(self, x, y):
        """
        Möbius addition: x (+) y
        """
        x2 = jnp.sum(x * x, axis=-1, keepdims=True)
        y2 = jnp.sum(y * y, axis=-1, keepdims=True)
        xy = jnp.sum(x * y, axis=-1, keepdims=True)
        
        num = (1 + 2 * self.c * xy + self.c * y2) * x + (1 - self.c * x2) * y
        denom = 1 + 2 * self.c * xy + self.c**2 * x2 * y2
        
        return num / (denom + 1e-15)

    def dist(self, x, y):
        """
        Geodesic distance.
        d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||-x (+) y||)
        """
        diff = self.mobius_add(-x, y)
        diff_norm = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        # Clip to avoid NaNs at boundary
        diff_norm = jnp.clip(diff_norm, a_min=None, a_max=1.0/jnp.sqrt(self.c) - 1e-5)
        return (2.0 / jnp.sqrt(self.c)) * jnp.arctanh(jnp.sqrt(self.c) * diff_norm)

    def to_lorentz(self, x):
        """
        Convert Poincaré Ball point x to Lorentz Model (Hyperboloid).
        """
        x_norm_sq = jnp.sum(x**2, axis=-1, keepdims=True)
        denom = 1.0 - self.c * x_norm_sq
        denom = jnp.maximum(denom, 1e-9) # Avoid div zero
        
        sqrt_c = jnp.sqrt(self.c)
        
        # x0 = (1/sqrt(c)) * (1 + c*|x|^2) / (1 - c*|x|^2)
        x0 = (1.0 / sqrt_c) * (1.0 + self.c * x_norm_sq) / denom
        
        # x_rest = 2x / (1 - c*|x|^2)
        x_rest = 2.0 * x / denom
        
        return jnp.concatenate([x0, x_rest], axis=-1)

    def exp_map(self, x, v):
        """
        Exponential map at x: Exp_x(v)
        Maps tangent vector v at x to the manifold.
        """
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        lambda_x = self.lambda_x(x)
        
        # Avoid division by zero for zero vector
        res = jnp.tanh(jnp.sqrt(self.c) * lambda_x * v_norm / 2.0) * (v / (jnp.sqrt(self.c) * v_norm + 1e-15))
        
        return self.mobius_add(x, res)

    def log_map(self, x, y):
        """
        Logarithmic map at x: Log_x(y)
        Maps y on manifold to tangent space at x.
        """
        diff = self.mobius_add(-x, y)
        diff_norm = jnp.linalg.norm(diff, axis=-1, keepdims=True)
        lambda_x = self.lambda_x(x)
        
        res = (2.0 / (jnp.sqrt(self.c) * lambda_x)) * jnp.arctanh(jnp.sqrt(self.c) * diff_norm) * (diff / (diff_norm + 1e-15))
        return res
    
    def metric_tensor(self, x):
        """
        Riemannian metric tensor G(x) = lambda_x^2 * I
        Returns diagonal elements or full matrix depending on need.
        Here we return the scalar factor lambda_x^2.
        """
        return self.lambda_x(x)**2

    def exp_map0(self, v):
        """
        Exponential map at origin: Exp_0(v)
        """
        v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
        # lambda_0 = 2.0
        # arg = sqrt(c) * 2.0 * v_norm / 2.0 = sqrt(c) * v_norm
        
        res = jnp.tanh(jnp.sqrt(self.c) * v_norm) * (v / (jnp.sqrt(self.c) * v_norm + 1e-15))
        return res

    def log_map0(self, y):
        """
        Logarithmic map at origin: Log_0(y)
        """
        y_norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
        # lambda_0 = 2.0
        # factor = 2.0 / (sqrt(c) * 2.0) = 1.0 / sqrt(c)
        
        res = (1.0 / jnp.sqrt(self.c)) * jnp.arctanh(jnp.sqrt(self.c) * y_norm) * (y / (y_norm + 1e-15))
        return res
