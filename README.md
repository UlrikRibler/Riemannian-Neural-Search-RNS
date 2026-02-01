# Riemannian Neural Search: Probabilistic Retrieval via Geometric Energy Landscapes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX-blue.svg)](https://github.com/google/jax)
[![Geometry: Hyperbolic](https://img.shields.io/badge/Geometry-Lorentz%20Manifold-purple.svg)](https://en.wikipedia.org/wiki/Hyperboloid_model)

## Abstract

**Riemannian Neural Search (RNS)** represents a paradigm shift in Information Retrieval (IR) by abandoning the static, Euclidean vector space model in favor of a dynamic, probabilistic framework operating on curved manifolds. We address the fundamental mismatch between flat embedding spaces and the intrinsic hierarchical structure of semantic knowledge by modeling data in constant-negative-curvature space (the **Lorentz model**).

Crucially, this work introduces a novel retrieval mechanism: instead of simple Nearest Neighbor Search (NNS), we formulate query processing as **Bayesian inference** on the manifold. We train a **Riemannian Energy-Based Model (EBM)** to learn the latent density of the corpus and utilize **Riemannian Manifold Hamiltonian Monte Carlo (RMHMC)** to sample relevant documents from the posterior distribution. This approach naturally handles query ambiguity, polysemy, and data scarcity, offering a theoretically grounded path toward "General Geometric Intelligence."

## 1. Theoretical Foundation

### 1.1. The Geometry of Hierarchy
Complex symbolic data (e.g., taxonomies, entailment graphs) exhibits exponential volume growth, which Euclidean space cannot embed without significant distortion. We utilize the **Lorentz Hyperboloid model** (\(\mathbb{L}^d, g_x\)), defined as the Riemannian manifold:

$$ 
\mathcal{M} = \{ \mathbf{x} \in \mathbb{R}^{d+1} : \langle \mathbf{x}, \mathbf{x} \rangle_{\mathcal{L}} = -1/c, \quad x_0 > 0 \} 
$$ 

where $\langle \mathbf{x}, \mathbf{y} \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{i=1}^d x_i y_i$ is the Minkowski inner product and $c$ is the learnable curvature.

### 1.2. Riemannian Energy-Based Models (EBM)
Standard IR assumes relevance is a deterministic distance. We argue that relevance is a density. We learn a scalar energy function $E_\phi: \mathcal{M} \to \mathbb{R}$ such that the probability density of valid documents on the manifold is given by the Boltzmann distribution:

$$ 
p(\mathbf{x}) = \frac{e^{-E_\phi(\mathbf{x})}}{Z(\phi)} 
$$ 

This EBM is trained via Noise Contrastive Estimation (NCE) directly on the curved surface, capturing the "gravitational field" of the semantic space.

### 1.3. Symplectic Hamiltonian Dynamics
To retrieve documents for a query $q$, we sample from the posterior $p(d|q) \propto e^{-U(d; q)}$. The potential energy $U$ combines the query's pull and the corpus density:

$$ 
U(\mathbf{z}) = \underbrace{\alpha \cdot d_{\mathcal{M}}(\mathbf{z}, q)^2}_{ \text{Query Likelihood}} + \underbrace{\beta \cdot E_\phi(\mathbf{z})}_{ \text{Learned Prior}} 
$$ 

We simulate a particle evolving under Hamiltonian dynamics:

$$ 
\frac{d\mathbf{q}}{dt} = \nabla_{\mathbf{p}} H, \quad \frac{d\mathbf{p}}{dt} = -\nabla_{\mathbf{q}} H 
$$ 

This symplectic integration allows the search agent to traverse geodesic paths, utilizing momentum to escape local minima and explore multimodal relevance distributions.

## 2. Methodology & Architecture

The system is implemented in **JAX** for end-to-end differentiability and hardware acceleration.

### Module Overview
*   **`src.geometry.lorentz`**: Numerical backend implementing the pseudo-Riemannian metric tensors, exponential maps (\(\text{Exp}_x\)), and logarithmic maps (\(\text{Log}_x\)).
*   **`src.models.encoder` (HyperbolicGCN)**: A Graph Convolutional Network that operates in the tangent space $T_x \mathcal{M}$, performing message passing using parallel transport approximations.
*   **`src.models.density` (RiemannianEBM)**: A neural network defined on the manifold that estimates the gradient field of the data distribution.
*   **`src.dynamics.hmc_sampler`**: A symplectic integrator (Generalized Leapfrog) solving the equations of motion on $\mathbb{L}^d$.

## 3. Experimental Validation

The system is benchmarked against the **WordNet Mammals** subtree, a canonical dataset for hierarchical representation learning.

### Metrics
We evaluate the retrieval quality using standard TREC protocols:
*   **Mean Average Precision (mAP)**: Measures the quality of the ranking order.
*   **Recall@K**: Measures the proportion of relevant (entailed) nodes retrieved in the top K samples.

### Results
*   **Curvature Adaptation**: The model successfully learns a curvature $c \approx 1.0$, confirming the hyperbolic nature of the data.
*   **Density Awareness**: The EBM effectively learns to penalize "void" regions of the manifold, guiding the HMC sampler toward valid semantic clusters.
*   **Trajectory Analysis**: Visualization confirms that the search particle oscillates between relevant concepts (e.g., *canine* and *feline* for a generic *mammal* query), demonstrating uncertainty quantification.

## 4. Usage

To reproduce the experiments:

```bash
# 1. Clone the repository
git clone https://github.com/ulrikil/riemannian-neural-search.git
cd riemannian-neural-search

# 2. Install dependencies
pip install jax jaxlib optax matplotlib numpy scipy

# 3. Run the research pipeline
python scripts/run_rns.py
```

## 5. Citation

If you use this code in your research, please cite:

```bibtex
@software{rns2026,
  author = {Ulrik},
  title = {Riemannian Neural Search: Learning Energy-Based Potentials on Hyperbolic Manifolds},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/ulrikil/riemannian-neural-search}
}
```

## 6. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.