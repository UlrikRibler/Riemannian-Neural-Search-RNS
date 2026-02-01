import jax
import jax.numpy as jnp
from jax import random

class RiemannianEBM:
    def __init__(self, input_dim, hidden_dims=[64, 32], manifold=None):
        """
        Energy-Based Model defined on the Manifold.
        Approximates the negative log-density of the data distribution.
        
        Args:
            input_dim: Dimension of the manifold (embedded dim - 1 usually, or embedding dim). 
                       For Lorentz, input is (N, D+1), tangent is D.
            hidden_dims: List of hidden layer sizes.
            manifold: Manifold instance (LorentzModel).
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.manifold = manifold
        
    def init_params(self, key):
        """
        Initializes MLP parameters.
        Returns a list of (W, b) tuples.
        """
        params = []
        dims = [self.input_dim] + self.hidden_dims + [1] # Output is scalar energy
        
        for i in range(len(dims) - 1):
            key, subkey = random.split(key)
            din, dout = dims[i], dims[i+1]
            scale = jnp.sqrt(2.0 / (din + dout))
            W = random.normal(subkey, (din, dout)) * scale
            b = jnp.zeros((dout,))
            params.append((W, b))
            
        return params

    def forward(self, params, x, c):
        """
        Computes the Energy E(x) for points x on the manifold.
        High Energy = Low Density (usually).
        We want to learn Density D(x) ~ exp(-E(x)).
        
        Args:
            params: MLP parameters.
            x: Points on the manifold (N, D+1) (Lorentz model).
            c: Curvature.
        Returns:
            energy: (N, 1) scalar values.
        """
        # 1. Map to Tangent Space at Origin
        # This is a safe way to define a function on the manifold: f(x) = MLP(Log0(x))
        # This preserves radial symmetry if MLP is radial, but MLP is general.
        x_tan = self.manifold.log_map0(x, c) # (N, D)
        
        h = x_tan
        for i, (W, b) in enumerate(params):
            h = jnp.matmul(h, W) + b
            if i < len(params) - 1:
                h = jax.nn.swish(h) # Swish or ReLU
                
        return h

    def loss_fn(self, params, data_emb, noise_emb, c):
        """
        Training loss for EBM using Noise Contrastive Estimation (NCE) 
        or simply Energy minimization for data + maximization for noise.
        
        Loss = E(data) - E(noise) (Simple energy gap) 
        or
        Log-Likelihood approximation.
        
        We use a simple margin loss or BCE:
        Prob(real) = sigmoid(-E(x))
        """
        e_data = self.forward(params, data_emb, c)
        e_noise = self.forward(params, noise_emb, c)
        
        # We want E(data) to be LOW, E(noise) to be HIGH.
        # BCE with logits = -Energy
        # label 1 for data, 0 for noise.
        
        # logits_data = -e_data
        # logits_noise = -e_noise
        
        # loss = - (log(sigmoid(-e_data)) + log(1 - sigmoid(-e_noise)))
        #      = log(1 + exp(e_data)) + log(1 + exp(-e_noise))
        
        loss_val = jnp.mean(jnp.logaddexp(0.0, e_data)) + jnp.mean(jnp.logaddexp(0.0, -e_noise))
        return loss_val
