import os
import urllib.request
import jax.numpy as jnp
import numpy as np
from jax import random

class WordNetLoader:
    def __init__(self, root="data", subset="mammals"):
        self.root = root
        self.subset = subset
        self.data_path = os.path.join(root, f"{subset}_subtree.tsv")
        self.closure_path = os.path.join(root, f"{subset}_closure.tsv")
        
        if not os.path.exists(root):
            os.makedirs(root)
            
        self.ensure_data()
        
    def ensure_data(self):
        # URL for the edges
        url = "https://raw.githubusercontent.com/facebookresearch/poincare-embeddings/master/wordnet/mammal_subtree.tsv"
        
        if not os.path.exists(self.data_path):
            print(f"Downloading {self.subset} from {url}...")
            try:
                urllib.request.urlretrieve(url, self.data_path)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download: {e}")
                # Create dummy data if download fails for testing
                print("Creating dummy data for testing purposes.")
                with open(self.data_path, "w") as f:
                    f.write("id1\tid2\n")
                    f.write("dog.n.01\tcanine.n.02\n")
                    f.write("cat.n.01\tfeline.n.01\n")
                    f.write("canine.n.02\tmammal.n.01\n")
                    f.write("feline.n.01\tmammal.n.01\n")

    def load_graph(self):
        """
        Loads the graph, computes transitive closure if needed.
        Returns:
            adj: Adjacency matrix (N, N)
            features: Identity features (N, N) (or random)
            labels: Node names/IDs
            closure: Transitive closure adjacency (for evaluation)
        """
        edges = []
        nodes = set()
        
        with open(self.data_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                u, v = parts[0], parts[1]
                edges.append((u, v))
                nodes.add(u)
                nodes.add(v)
                
        node_list = sorted(list(nodes))
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        num_nodes = len(node_list)
        
        # Build Adjacency
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for u, v in edges:
            i, j = node_to_idx[u], node_to_idx[v]
            adj[i, j] = 1.0
            adj[j, i] = 1.0 # Undirected for GCN usually, or directed? 
            # WordNet is hierarchical (directed). GCN usually treats as undirected for message passing,
            # but for hierarchy learning we might care.
            # SBM was symmetric. We'll make it symmetric for GCN.
            
        # Transitive Closure (Ground Truth Entailment)
        # For evaluation: "is u a hyponym of v?"
        # We compute reachability in the DIRECTED graph.
        # Edge u -> v means u is a child of v (or vice versa depending on file format).
        # Usually file is "hyponym hypernym" (child parent).
        
        closure = np.eye(num_nodes, dtype=np.float32)
        # Warshall's algorithm or BFS for closure
        # Since it's a tree/DAG, BFS from each node is fine.
        
        # Build directed adj for closure
        dir_adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for u, v in edges:
             i, j = node_to_idx[u], node_to_idx[v]
             # Assuming format: id1 id2. Usually Child Parent.
             # We want Closure[child, parent] = 1
             dir_adj[i, j] = 1.0
             
        # Compute closure (simple BFS)
        # Slow for large graphs, but mammals is ~1k nodes.
        print("Computing transitive closure...")
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if closure[i, j] == 0:
                        closure[i, j] = max(closure[i, j], dir_adj[i, k] * dir_adj[k, j]) # Logic might be wrong for dense.
        
        # Better: boolean matrix multiplication
        # Closure = A + A^2 + ... + A^n
        curr = dir_adj.copy()
        closure = dir_adj.copy()
        for _ in range(20): # Depth limit approx
            curr = np.matmul(curr, dir_adj)
            curr = (curr > 0).astype(np.float32)
            closure = np.maximum(closure, curr)
            if np.sum(curr) == 0: break
            
        # Add self-loops to closure for "entailment includes self"
        closure = np.maximum(closure, np.eye(num_nodes))

        # Normalize Adj for GCN
        adj = adj + np.eye(num_nodes)
        row_sum = adj.sum(1)
        d_inv_sqrt = np.power(row_sum, -0.5)
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        norm_adj = np.dot(np.dot(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        # Features: Identity (One-Hot) or Random
        # Using Identity is common for link prediction if no features.
        features = np.eye(num_nodes, dtype=np.float32)
        
        return jnp.array(norm_adj), jnp.array(features), node_list, jnp.array(closure)
