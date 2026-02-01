import jax
import jax.numpy as jnp
from jax import random
from src.geometry.lorentz import LorentzModel

class HyperbolicGCNLayer:
    def __init__(self, in_features, out_features, manifold):
        self.in_features = in_features
        self.out_features = out_features
        self.manifold = manifold # LorentzModel
        
    def init_params(self, key):
        # Initialize weights and bias in Tangent Space (Euclidean)
        w_key, b_key = random.split(key)
        # Xavier/Glorot initialization
        scale = jnp.sqrt(2.0 / (self.in_features + self.out_features))
        W = random.normal(w_key, (self.in_features, self.out_features)) * scale
        b = jnp.zeros((self.out_features,))
        return W, b

    def forward(self, params, h, adj, c):
        """
        Args:
            params: (W, b) tuple
            h: Node features on Hyperboloid (N, D+1) (or D for first layer if lifted)
               Wait, typically we keep h in Lorentz form between layers.
            adj: Adjacency matrix (N, N)
            c: Curvature scalar
            
        Returns:
            h_out: Node features on Hyperboloid (N, out_dim + 1)
        """
        W, b = params
        
        # 1. Lift to Tangent Space at Origin
        # h is (N, D+1)
        h_tan = self.manifold.log_map0(h, c) # (N, in_dim)
        
        # 2. Graph Convolution in Tangent Space
        h_agg = jnp.matmul(adj, h_tan)
        
        # 3. Linear Transformation
        h_trans = jnp.matmul(h_agg, W) + b
        
        # 4. Activation (ReLU in tangent space)
        h_act = jax.nn.relu(h_trans)
        
        # 5. Project back to Hyperboloid
        h_out = self.manifold.exp_map0(h_act, c) # (N, out_dim+1)
        
        return h_out

class HyperbolicGCN:
    def __init__(self, layer_sizes, manifold=None):
        """
        Hyperbolic Graph Convolutional Network (HGCN).
        
        This network learns to embed hierarchical graph data into Hyperbolic space (Lorentz model).
        It performs graph convolution operations in the tangent space of the manifold,
        leveraging the logarithmic and exponential maps for 'lifting' and 'projecting'.
        
        Args:
            layer_sizes: List of integers [input_dim, hidden_dim, ..., output_dim]
            manifold: Instance of LorentzModel.
        """
        if manifold is None:
            self.manifold = LorentzModel()
        else:
            self.manifold = manifold
            
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                HyperbolicGCNLayer(layer_sizes[i], layer_sizes[i+1], self.manifold)
            )

    def init_params(self, key):
        """
        Returns a dict containing layer parameters AND the learnable curvature 'c'.
        """
        layer_params = []
        for layer in self.layers:
            key, subkey = random.split(key)
            layer_params.append(layer.init_params(subkey))
            
        return {
            "layers": layer_params,
            "c": jnp.array(1.0) # Learnable curvature initialized to 1.0
        }

    def forward(self, params, x, adj):
        """
        x: Initial features.
           If x is Euclidean (N, D), we map it to Lorentz first using exp_map0.
           Input x is expected to be Euclidean here for convenience.
        """
        c = jax.nn.softplus(params["c"]) # Ensure c > 0
        
        # Lift input to Manifold (if it's not already, assuming input is Euclidean features)
        # x: (N, in_dim) -> (N, in_dim + 1)
        h = self.manifold.exp_map0(x, c)
        
        for layer_params, layer in zip(params["layers"], self.layers):
            h = layer.forward(layer_params, h, adj, c)
            
        return h