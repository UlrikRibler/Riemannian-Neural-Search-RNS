# Riemannian Neural Search (RNS)

## Abstract

**Riemannian Neural Search (RNS)** is a next-generation Information Retrieval (IR) system that supersedes traditional Euclidean vector space models by leveraging **Hyperbolic Geometry** and **Hamiltonian Dynamics**.

Standard search engines model semantic relationships in flat Euclidean space (\(\mathbb{R}^n\)), which introduces significant distortion when embedding hierarchical data (scale-free networks). RNS addresses this by embedding data into a constant-negative-curvature manifold (Lorentz Hyperboloid \(\mathbb{L}^n\)) and performing retrieval not via nearest-neighbor search, but via **Symplectic Integration** of a particle in a potential energy landscape defined by the query.

This repository implements a fully differentiable pipeline including a learnable-curvature Hyperbolic Graph Convolutional Network (HGCN) and a Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) sampler for probabilistic ranking.

## Theoretical Framework

### 1. The Manifold Hypothesis
Complex datasets, such as the World Wide Web or lexical databases (WordNet), exhibit a latent hierarchical structure. Embedding these into Euclidean space requires exponential dimensionality to preserve distances. Hyperbolic space, with its exponential volume growth, accommodates tree-like structures with arbitrary low distortion in low dimensions.

### 2. Hamiltonian Ranking
Instead of collapsing query-document relevance to a static scalar score (point estimate), RNS treats the query as a potential energy field \(U(q)\) on the manifold. The relevance distribution is approximated by the trajectory of a particle evolving under Hamiltonian dynamics:
$$ H(q, p) = U(q) + \frac{1}{2} \log((2\pi)^D |G(q)|) + \frac{1}{2} p^T G(q)^{-1} p $$
where \(G(q)\) is the Riemannian metric tensor. This allows the system to capture uncertainty and multimodal ambiguity (e.g., polysemous queries).

## System Architecture

The project is modularized into three core physical components:

### Module 1: The Geometry (`src.geometry`)
*   **Lorentz Model (`lorentz.py`):** Primary backend for numerical stability. Implements Minkowski inner products and exponential maps on the hyperboloid.
*   **Poincaré Ball (`manifold.py`):** Conformal model used for visualization and projection.
*   **Learnable Curvature:** The curvature parameter \(c\) is treated as a trainable parameter, allowing the model to adapt the geometry to the dataset's intrinsic complexity.

### Module 2: The Encoder (`src.models`)
*   **HyperbolicGCN (`encoder.py`):** A Graph Convolutional Network operating in the tangent space of the manifold.
    *   *Lifting:* \(\text{Log}_0(x)\) projects features to the tangent space.
    *   *Aggregation:* Weighted message passing in Tangent Space.
    *   *Projection:* \(\text{Exp}_0(x)\) maps aggregated features back to the manifold.
*   **Hyperbolic InfoNCE (`losses.py`):** Contrastive loss function optimizing negative hyperbolic distances.

### Module 3: The Dynamics (`src.dynamics`)
*   **Riemannian HMC (`hmc_sampler.py`):** A symplectic integrator using the Generalized Leapfrog algorithm. It samples from the posterior distribution of relevant documents given a query.

## Tech Stack

*   **JAX:** For high-performance autodifferentiation and JIT compilation.
*   **Optax:** Gradient transformation and optimization.
*   **Matplotlib:** Phase space visualization.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/ulrikribler/riemannian-neural-search.git
    cd riemannian-neural-search
    ```

2.  Install dependencies:
    ```bash
    pip install jax jaxlib optax matplotlib
    ```

## Usage

### 1. End-to-End Demonstration
To run a full simulation (Synthetic Graph Generation $\to$ Hyperbolic Training $\to$ Hamiltonian Search):

```bash
python scripts/run_rns.py
```

**Expected Output:**
*   Adaptation of curvature \(c\) (e.g., $1.0 \to 0.97$).
*   Convergence of InfoNCE loss.
*   Generation of `rns_upgrade_result.png`: A visualization of the Poincaré disk showing document clusters and the HMC particle trajectory oscillating between ambiguous targets.

### 2. Visualization of Dynamics
To observe the properties of the Hamiltonian particle in isolation:

```bash
python scripts/visualize_hmc.py
```

## References

1.  **Nickel, M., & Kiela, D.** (2017). *Poincaré Embeddings for Learning Hierarchical Representations*. NeurIPS.
2.  **Chami, I., et al.** (2019). *Hyperbolic Graph Convolutional Neural Networks*. NeurIPS.
3.  **Girolami, M., & Calderhead, B.** (2011). *Riemannian Manifold Hamiltonian Monte Carlo*. JRSS-B.
