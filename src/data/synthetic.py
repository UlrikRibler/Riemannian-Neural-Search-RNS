import jax
import jax.numpy as jnp
from jax import random

def generate_sbm_graph(key, num_nodes, num_clusters, in_prob=0.5, out_prob=0.05, feature_dim=16):
    """
    Generates a Stochastic Block Model graph to simulate semantic clusters (topics).
    Returns:
        adj: Adjacency matrix (normalized)
        features: Euclidean node features
        labels: Cluster assignments
    """
    # 1. Assign nodes to clusters
    nodes_per_cluster = num_nodes // num_clusters
    labels = jnp.repeat(jnp.arange(num_clusters), nodes_per_cluster)
    
    # Handle remainder
    if len(labels) < num_nodes:
        labels = jnp.pad(labels, (0, num_nodes - len(labels)), constant_values=num_clusters-1)

    # 2. Generate Adjacency
    key, subkey = random.split(key)
    # Block matrix probability
    # If same label, prob = in_prob, else out_prob
    
    # Expand labels to (N, N) grid
    label_grid_i = jnp.expand_dims(labels, 1) # (N, 1)
    label_grid_j = jnp.expand_dims(labels, 0) # (1, N)
    
    prob_matrix = jnp.where(label_grid_i == label_grid_j, in_prob, out_prob)
    
    adj = random.bernoulli(subkey, prob_matrix).astype(jnp.float32)
    
    # Remove self-loops (optional, but GCN usually adds them back)
    adj = adj * (1 - jnp.eye(num_nodes))
    
    # Add self-loops
    adj = adj + jnp.eye(num_nodes)
    
    # Symmetric normalization: D^-0.5 A D^-0.5
    row_sum = adj.sum(1)
    d_inv_sqrt = jnp.power(row_sum, -0.5)
    d_mat_inv_sqrt = jnp.diag(d_inv_sqrt)
    norm_adj = jnp.matmul(jnp.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    
    # 3. Generate Features
    # Cluster centers
    key, subkey = random.split(key)
    centers = random.normal(subkey, (num_clusters, feature_dim))
    
    # Node features = center + noise
    key, subkey = random.split(key)
    noise = random.normal(subkey, (num_nodes, feature_dim)) * 0.1
    features = centers[labels] + noise
    
    return norm_adj, features, labels

def generate_ambiguous_query(key, features, labels, cluster_a, cluster_b):
    """
    Simulate an ambiguous query relevant to two different clusters.
    Returns:
        query_vector: A vector between the two cluster centers.
        targets: Indices of nodes in those clusters.
    """
    # Find centers of the two clusters
    mask_a = labels == cluster_a
    mask_b = labels == cluster_b
    
    center_a = jnp.mean(features[mask_a], axis=0)
    center_b = jnp.mean(features[mask_b], axis=0)
    
    # Query is average of centers (ambiguous)
    query_vector = (center_a + center_b) / 2.0
    
    # Targets are all nodes in these clusters
    # But for contrastive learning, we usually pick 1 positive.
    # For HMC search, we want to see it find both.
    
    return query_vector, (center_a, center_b)
