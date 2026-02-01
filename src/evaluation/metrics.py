import jax.numpy as jnp
import numpy as np

class Evaluator:
    def __init__(self, embeddings, closure_matrix):
        """
        Args:
            embeddings: (N, D) - typically Poincaré embeddings.
            closure_matrix: (N, N) binary matrix. closure[i, j] = 1 if i implies j (or vice versa).
                            Ground truth for "Relevance".
                            Row i = Targets for Query i.
        """
        self.embeddings = np.array(embeddings)
        self.closure = np.array(closure_matrix)
        self.num_nodes = self.embeddings.shape[0]

    def compute_metrics(self, queries, retrieved_indices):
        """
        Computes MAP and Recall@K for a batch of queries.
        
        Args:
            queries: List of query node indices.
            retrieved_indices: List of lists (or matrix) of retrieved node indices for each query.
                               Shape: (num_queries, K)
        Returns:
            metrics: Dict with 'mAP', 'Recall@5', 'Recall@10'.
        """
        aps = []
        recalls_5 = []
        recalls_10 = []
        
        for i, q_idx in enumerate(queries):
            targets = np.where(self.closure[q_idx] > 0)[0]
            # Exclude self from targets if present (retrieval task usually excludes query)
            targets = set(t for t in targets if t != q_idx)
            
            if len(targets) == 0:
                continue
                
            retrieved = retrieved_indices[i]
            
            # AP Calculation
            hits = 0
            sum_precs = 0
            for rank, node_idx in enumerate(retrieved):
                if node_idx in targets:
                    hits += 1
                    sum_precs += hits / (rank + 1)
            
            ap = sum_precs / min(len(targets), len(retrieved)) if hits > 0 else 0
            aps.append(ap)
            
            # Recall@K
            # We assume retrieved is at least length 10 or we truncate
            r5_set = set(retrieved[:5])
            r10_set = set(retrieved[:10])
            
            hits5 = len(r5_set.intersection(targets))
            hits10 = len(r10_set.intersection(targets))
            
            recalls_5.append(hits5 / len(targets))
            recalls_10.append(hits10 / len(targets))
            
        return {
            "mAP": np.mean(aps) if aps else 0.0,
            "Recall@5": np.mean(recalls_5) if recalls_5 else 0.0,
            "Recall@10": np.mean(recalls_10) if recalls_10 else 0.0
        }

    def baseline_search(self, query_indices, k=10, manifold=None):
        """
        Standard Nearest Neighbor Search in Hyperbolic Space.
        """
        results = []
        # Precompute all distances is expensive O(N^2), but N~1000 is fine.
        # Use batching if needed.
        
        # We need the dist function from manifold.
        # Assuming embeddings are in Poincaré.
        
        for q_idx in query_indices:
            q_emb = self.embeddings[q_idx]
            
            # Simple Euclidean distance is WRONG for Poincaré, but
            # rank order is preserved if they are close to origin? No.
            # We must use Poincaré distance.
            
            # dist = arccosh(1 + 2 * |u-v|^2 / ((1-|u|^2)(1-|v|^2)))
            # For ranking, 1 + ... is monotonic with the fraction part.
            # So we just minimize the fraction part.
            
            u = q_emb
            v = self.embeddings
            sq_u = np.sum(u**2)
            sq_v = np.sum(v**2, axis=1)
            sq_dist = np.sum((v - u)**2, axis=1)
            
            # Mobius addition based distance or the formula above?
            # Formula above is for Poincare Ball.
            
            denom = (1 - sq_u) * (1 - sq_v)
            # Clip denom to avoid division by zero
            denom = np.maximum(denom, 1e-7)
            
            hyper_dist_proxy = sq_dist / denom
            
            # Sort
            sorted_indices = np.argsort(hyper_dist_proxy)
            
            # Filter self
            sorted_indices = [idx for idx in sorted_indices if idx != q_idx]
            
            results.append(sorted_indices[:k])
            
        return results
