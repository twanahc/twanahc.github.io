---
layout: post
title: "Tensor Algebra: Beyond Matrices to the Mathematics of Multidimensional Data"
date: 2026-01-10
category: math
---

When you hear "tensor" in the context of deep learning, you probably think of a multidimensional array --- a generalization of matrices to higher dimensions. PyTorch tensors. TensorFlow tensors. Blocks of floating-point numbers with a shape attribute. This is not wrong, but it misses the mathematical substance entirely. A tensor is not defined by how its components are stored. A tensor is defined by how it *transforms*.

This post builds tensor algebra from the ground up: what tensors actually are as mathematical objects, how to manipulate them, why their decompositions matter for data compression and model efficiency, and how the operations we perform daily in deep learning --- attention, convolution, einsum --- are tensor operations in disguise.

---

## Table of Contents

1. [Scalars, Vectors, Matrices: The Low-Rank Zoo](#scalars-vectors-matrices-the-low-rank-zoo)
2. [What a Tensor Actually Is](#what-a-tensor-actually-is)
3. [The Tensor Product](#the-tensor-product)
4. [Index Notation and Einstein Summation](#index-notation-and-einstein-summation)
5. [Covariant and Contravariant Indices](#covariant-and-contravariant-indices)
6. [Tensor Contractions](#tensor-contractions)
7. [The Metric Tensor](#the-metric-tensor)
8. [Tensor Decompositions](#tensor-decompositions)
9. [Tensors in Deep Learning](#tensors-in-deep-learning)
10. [Python: Tensor Operations in Practice](#python-tensor-operations-in-practice)

---

## Scalars, Vectors, Matrices: The Low-Rank Zoo

Before we can understand tensors, we need to be precise about what we already know.

A **scalar** is a single number. It has no direction, no components. Temperature at a point, mass of a particle, energy of a system. Scalars are **rank-0 tensors** --- they have zero indices and are unchanged by any coordinate transformation.

A **vector** is an ordered list of numbers *that transforms in a specific way under change of basis*. This is critical. Not every ordered list of numbers is a vector. A vector $v$ lives in a vector space $V$ and can be expressed in terms of a basis $\{e_1, e_2, \ldots, e_n\}$ as:

$$v = v^1 e_1 + v^2 e_2 + \cdots + v^n e_n = \sum_{i=1}^{n} v^i e_i$$

The numbers $v^i$ are the **components** of the vector in this basis. Change the basis, and the components change --- but the vector itself is the same geometric object. A vector is a **rank-1 tensor**: one index, one direction.

A **matrix** is a rank-2 tensor: it has two indices and represents a **linear map** between vector spaces. A matrix $A$ with components $A^i{}_j$ takes a vector $v^j$ and produces a new vector $w^i = A^i{}_j v^j$. The matrix is the coordinate representation of this linear map. Different bases give different matrices for the same underlying map.

The pattern is clear: a rank-$k$ tensor has $k$ indices. But the crucial insight is that tensors are not *defined* by their components. They are defined by their **transformation behavior** and their **multilinear** nature. The components are just one way to represent the tensor after choosing a basis.

---

## What a Tensor Actually Is

Here is the formal definition. A **tensor of type $(p, q)$** is a **multilinear map**:

$$T: \underbrace{V^* \times \cdots \times V^*}_{p \text{ copies}} \times \underbrace{V \times \cdots \times V}_{q \text{ copies}} \longrightarrow \mathbb{R}$$

where $V$ is a vector space and $V^*$ is its **dual space** (the space of all linear maps from $V$ to $\mathbb{R}$, also called linear functionals or covectors).

Let us unpack this piece by piece.

**Multilinear** means linear in each argument separately. If $T$ is a $(1,1)$ tensor, then for vectors $u, v \in V$, covectors $\alpha, \beta \in V^*$, and scalars $a, b \in \mathbb{R}$:

$$T(a\alpha + b\beta, \, v) = a \, T(\alpha, v) + b \, T(\beta, v)$$

$$T(\alpha, \, au + bv) = a \, T(\alpha, u) + b \, T(\alpha, v)$$

Linearity in each slot independently. This is weaker than full linearity (a bilinear map is not the same as a linear map on the product space).

**The dual space $V^*$** consists of all linear functions $\varphi: V \rightarrow \mathbb{R}$. If $V$ has basis $\{e_1, \ldots, e_n\}$, then $V^*$ has a dual basis $\{e^1, \ldots, e^n\}$ defined by:

$$e^i(e_j) = \delta^i_j = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

This $\delta^i_j$ is the **Kronecker delta** --- it picks out matching indices. The dual basis element $e^i$ "extracts" the $i$-th component of any vector: $e^i(v) = v^i$.

Now, why do we need both $V$ and $V^*$? Because vectors and covectors transform *differently* under change of basis. This distinction is invisible when you work in an orthonormal basis (where $V$ and $V^*$ can be identified), which is why most programmers never encounter it. But it is fundamental to the mathematics.

Let us see some examples:

- A **scalar** is a $(0,0)$ tensor: it takes no inputs and returns a real number.
- A **vector** is a $(1,0)$ tensor: it takes one covector and returns a real number. (Technically, vectors are elements of $V$, and we identify them with $(1,0)$ tensors via the natural isomorphism.)
- A **covector** (or one-form) is a $(0,1)$ tensor: it takes one vector and returns a real number.
- A **linear map** $A: V \rightarrow V$ is a $(1,1)$ tensor: it takes one covector and one vector and returns a real number. In components: $A(\alpha, v) = \alpha_i A^i{}_j v^j$.
- A **bilinear form** (like the dot product) is a $(0,2)$ tensor: it takes two vectors and returns a real number.

The **rank** of a tensor is $p + q$, the total number of indices. A matrix is rank 2. A three-dimensional array of numbers *might* be a rank-3 tensor, but only if it transforms correctly.

---

## The Tensor Product

The **tensor product** is the fundamental operation that builds higher-rank tensors from lower-rank ones. If $u \in V$ and $w \in V$ are vectors, their tensor product $u \otimes w$ is a rank-2 tensor defined by its action on a pair of covectors:

$$(u \otimes w)(\alpha, \beta) = \alpha(u) \cdot \beta(w)$$

for all $\alpha, \beta \in V^*$. In components, if $u$ has components $u^i$ and $w$ has components $w^j$, then:

$$(u \otimes w)^{ij} = u^i w^j$$

This is exactly the **outer product**. The tensor product of two vectors with $n$ components each gives an $n \times n$ matrix. But --- and this is important --- not every $n \times n$ matrix can be written as an outer product of two vectors. A matrix $M^{ij}$ that equals $u^i w^j$ for some $u, w$ is called a **rank-1 matrix** (rank in the linear algebra sense, not the tensor sense). A general rank-2 tensor (matrix) is a *sum* of such outer products:

$$M^{ij} = \sum_{r=1}^{R} u_r^i w_r^j$$

The minimum number of terms $R$ needed in this sum is the **matrix rank**. This idea generalizes to higher-order tensors and is the foundation of tensor decomposition.

More generally, if $S$ is a tensor of type $(p_1, q_1)$ and $T$ is a tensor of type $(p_2, q_2)$, then $S \otimes T$ is a tensor of type $(p_1 + p_2, q_1 + q_2)$. The tensor product of vector spaces $V \otimes W$ is a new vector space whose dimension is $\dim(V) \times \dim(W)$.

<svg viewBox="0 0 700 300" xmlns="http://www.w3.org/2000/svg" style="background: white; max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#333">Tensor Product: From Vectors to Matrices (Outer Product)</text>
  <!-- Vector u -->
  <rect x="50" y="60" width="40" height="120" fill="none" stroke="#2196F3" stroke-width="2"/>
  <text x="70" y="55" text-anchor="middle" font-size="12" fill="#2196F3" font-weight="bold">u</text>
  <text x="70" y="85" text-anchor="middle" font-size="11" fill="#333">u¹</text>
  <text x="70" y="115" text-anchor="middle" font-size="11" fill="#333">u²</text>
  <text x="70" y="145" text-anchor="middle" font-size="11" fill="#333">u³</text>
  <!-- multiply sign -->
  <text x="130" y="125" text-anchor="middle" font-size="20" fill="#333">⊗</text>
  <!-- Vector w transpose -->
  <rect x="170" y="100" width="120" height="40" fill="none" stroke="#F44336" stroke-width="2"/>
  <text x="230" y="95" text-anchor="middle" font-size="12" fill="#F44336" font-weight="bold">w</text>
  <text x="200" y="125" text-anchor="middle" font-size="11" fill="#333">w¹</text>
  <text x="230" y="125" text-anchor="middle" font-size="11" fill="#333">w²</text>
  <text x="260" y="125" text-anchor="middle" font-size="11" fill="#333">w³</text>
  <!-- equals -->
  <text x="330" y="125" text-anchor="middle" font-size="20" fill="#333">=</text>
  <!-- Result matrix -->
  <rect x="370" y="60" width="120" height="120" fill="none" stroke="#9C27B0" stroke-width="2"/>
  <text x="430" y="55" text-anchor="middle" font-size="12" fill="#9C27B0" font-weight="bold">u ⊗ w</text>
  <text x="400" y="90" text-anchor="middle" font-size="10" fill="#333">u¹w¹</text>
  <text x="430" y="90" text-anchor="middle" font-size="10" fill="#333">u¹w²</text>
  <text x="460" y="90" text-anchor="middle" font-size="10" fill="#333">u¹w³</text>
  <text x="400" y="120" text-anchor="middle" font-size="10" fill="#333">u²w¹</text>
  <text x="430" y="120" text-anchor="middle" font-size="10" fill="#333">u²w²</text>
  <text x="460" y="120" text-anchor="middle" font-size="10" fill="#333">u²w³</text>
  <text x="400" y="150" text-anchor="middle" font-size="10" fill="#333">u³w¹</text>
  <text x="430" y="150" text-anchor="middle" font-size="10" fill="#333">u³w²</text>
  <text x="460" y="150" text-anchor="middle" font-size="10" fill="#333">u³w³</text>
  <!-- Note -->
  <text x="350" y="220" text-anchor="middle" font-size="12" fill="#666">The outer product u ⊗ w produces a rank-1 matrix.</text>
  <text x="350" y="240" text-anchor="middle" font-size="12" fill="#666">Every matrix is a sum of rank-1 outer products (this is the SVD).</text>
  <text x="350" y="260" text-anchor="middle" font-size="12" fill="#666">Tensor decompositions generalize this idea to higher dimensions.</text>
</svg>

---

## Index Notation and Einstein Summation

When working with tensors beyond rank 2, writing explicit summation signs becomes unbearable. Einstein summation convention is a notational shortcut: **whenever an index appears once as a superscript and once as a subscript in the same term, it is implicitly summed over**.

For example, the matrix-vector product $w = Av$ in index notation:

$$w^i = A^i{}_j v^j$$

The index $j$ appears once up (on $v^j$) and once down (on $A^i{}_j$), so it is summed over: $w^i = \sum_{j=1}^{n} A^i{}_j v^j$. The index $i$ appears only once (as a superscript on $w$ and on $A$), so it is a **free index** --- it labels which component of the result we are computing.

Rules:
- **Free indices** appear on both sides of the equation, once per term. They label the components.
- **Contracted (dummy) indices** appear exactly twice in a single term --- once up, once down --- and are summed over. Their letter is arbitrary: $A^i{}_j v^j = A^i{}_k v^k$.
- An index appearing twice in the same position (both up or both down) typically indicates a mistake.

Some key examples:

**Dot product:** $u \cdot v = u_i v^i$ (contract the one index --- result is a scalar, no free indices).

**Matrix multiplication:** $(AB)^i{}_k = A^i{}_j B^j{}_k$ (contract over $j$).

**Trace:** $\text{tr}(A) = A^i{}_i$ (contract the two indices of a single matrix --- result is a scalar).

**Rank-3 contraction:** $T^i{}_{jk} v^k = S^i{}_j$ (contract over $k$ --- a rank-3 tensor acting on a vector gives a rank-2 tensor).

This notation scales beautifully. An expression like $T^{abc}{}_{de} S^{d}{}_{fg} v^{e} w^{g}$ has contracted indices $d$, $e$, $g$ (each appears once up and once down) and free indices $a, b, c, f$. The result is a rank-4 tensor with components labeled by $a, b, c, f$.

NumPy's `einsum` function is a direct implementation of this notation, and we will use it extensively in the Python section.

---

## Covariant and Contravariant Indices

Why do we distinguish between upper and lower indices? This is where the transformation law comes in.

Consider a vector space $V$ with basis $\{e_1, \ldots, e_n\}$. Now change to a new basis $\{e'_1, \ldots, e'_n\}$ related by:

$$e'_i = M^j{}_i \, e_j$$

where $M$ is an invertible matrix. How do the components of a vector $v$ change?

The vector itself is unchanged: $v = v^i e_i = v'^i e'_i$. Substituting the basis change:

$$v'^i e'_i = v'^i M^j{}_i e_j$$

Comparing with $v = v^j e_j$, we get $v^j = M^j{}_i v'^i$, which means:

$$v'^i = (M^{-1})^i{}_j \, v^j$$

The components transform with the **inverse** of the basis change matrix. Components that transform this way are called **contravariant** and are written with **upper indices**: $v^i$.

Now consider a covector $\varphi \in V^*$. Its components $\varphi_i$ in the dual basis transform as:

$$\varphi'_i = M^j{}_i \, \varphi_j$$

The components transform with the **same** matrix as the basis (not the inverse). Components that transform this way are called **covariant** and are written with **lower indices**: $\varphi_i$.

The naming is counterintuitive --- "contravariant" means varying *contrary* to the basis, and "covariant" means varying *with* the basis. But the key point is operational: **an index repeated once up and once down contracts correctly under any change of basis**, producing a basis-independent scalar. This is why Einstein summation only contracts between an upper and lower index.

For a general tensor of type $(p,q)$, the transformation law is:

$$T'^{i_1 \cdots i_p}{}_{j_1 \cdots j_q} = (M^{-1})^{i_1}{}_{a_1} \cdots (M^{-1})^{i_p}{}_{a_p} \, M^{b_1}{}_{j_1} \cdots M^{b_q}{}_{j_q} \, T^{a_1 \cdots a_p}{}_{b_1 \cdots b_q}$$

Each upper index gets a factor of $M^{-1}$, each lower index gets a factor of $M$. This is **the** defining property of a tensor.

---

## Tensor Contractions

A **contraction** is the operation of summing over one upper and one lower index of a tensor, reducing its rank by 2. This is the generalization of several familiar operations:

**Matrix trace** is a contraction. The matrix $A^i{}_j$ contracted over $i$ and $j$:

$$\text{tr}(A) = A^i{}_i = \sum_{i} A^i{}_i$$

This takes a rank-2 tensor to a rank-0 tensor (scalar).

**Matrix-vector multiplication** is a contraction of the tensor product. First form $A^i{}_j \otimes v^k$, which is a rank-3 object with components $A^i{}_j v^k$. Then contract $j$ with $k$ (identifying them, which requires $j$ to be a lower index and $k$ to be an upper index on the vector):

$$A^i{}_j v^j = w^i$$

**Matrix multiplication** is two contractions (or equivalently, one contraction of the tensor product of two matrices).

$$(AB)^i{}_k = A^i{}_j B^j{}_k$$

**Inner product / dot product** contracts a vector with a covector:

$$\varphi_i v^i = \text{scalar}$$

The general pattern: every contraction removes one upper and one lower index, reduces the rank by 2, and produces a quantity that is invariant under the particular change of basis along the contracted indices. Contraction is the fundamental way tensors interact.

---

## The Metric Tensor

The **metric tensor** $g$ is a rank-$(0,2)$ tensor that defines an inner product on a vector space. In components:

$$g_{ij} = g(e_i, e_j)$$

It must be symmetric ($g_{ij} = g_{ji}$) and non-degenerate ($\det(g_{ij}) \neq 0$). The metric allows you to:

1. **Measure distances.** The squared length of a vector $v$ is $\|v\|^2 = g_{ij} v^i v^j$.

2. **Measure angles.** The inner product of $u$ and $v$ is $\langle u, v \rangle = g_{ij} u^i v^j$.

3. **Lower indices.** Given a contravariant vector $v^i$, define the covariant version $v_i = g_{ij} v^j$. This converts a vector into a covector.

4. **Raise indices.** The inverse metric $g^{ij}$ (defined by $g^{ik} g_{kj} = \delta^i_j$) raises indices: $v^i = g^{ij} v_j$.

In flat Euclidean space with an orthonormal basis, the metric is just the identity matrix: $g_{ij} = \delta_{ij}$. In this case, raising and lowering indices does nothing, which is why the distinction between upper and lower indices is invisible in most machine learning code. The components $v^i$ and $v_i$ are numerically identical.

But in curved spaces (general relativity), in non-orthogonal coordinate systems, or when working with non-Euclidean metrics (as in information geometry, which underlies natural gradient methods), the metric is non-trivial and the distinction matters.

The metric tensor is perhaps the most important tensor in all of physics. It encodes the geometry of space itself. In general relativity, gravity *is* the curvature of the metric tensor.

---

## Tensor Decompositions

Just as matrices can be decomposed (SVD, eigendecomposition, LU, QR), higher-order tensors have decompositions that reveal their structure. The two most important are **CP decomposition** and **Tucker decomposition**.

### CP Decomposition (CANDECOMP/PARAFAC)

The **CP decomposition** expresses a tensor as a sum of rank-1 tensors. A rank-1 tensor of order $N$ is an outer product of $N$ vectors. For a third-order tensor $\mathcal{T} \in \mathbb{R}^{I \times J \times K}$:

$$\mathcal{T} \approx \sum_{r=1}^{R} \lambda_r \, a_r \otimes b_r \otimes c_r$$

In component form:

$$T_{ijk} \approx \sum_{r=1}^{R} \lambda_r \, a_{r,i} \, b_{r,j} \, c_{r,k}$$

where $\lambda_r$ are scalar weights and $a_r, b_r, c_r$ are vectors. The minimum $R$ for which this decomposition is exact is the **tensor rank** (or CP rank).

This directly generalizes the matrix SVD. A matrix $M = U \Sigma V^T$ can be written as $M_{ij} = \sum_r \sigma_r u_{r,i} v_{r,j}$, which is exactly a CP decomposition of a rank-2 tensor.

**Why it matters:** A tensor $\mathcal{T} \in \mathbb{R}^{n \times n \times n}$ has $n^3$ components. A CP decomposition with rank $R$ requires only $R(3n + 1)$ parameters. If $R \ll n^2/3$, this is a massive compression. In practice, many real-world tensors have low CP rank, making this decomposition useful for data compression and analysis.

### Tucker Decomposition

The **Tucker decomposition** is more flexible. It expresses a tensor as a core tensor multiplied by a matrix along each mode:

$$\mathcal{T} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$$

In components:

$$T_{ijk} \approx \sum_{p=1}^{P} \sum_{q=1}^{Q} \sum_{r=1}^{R} G_{pqr} \, A_{ip} \, B_{jq} \, C_{kr}$$

Here $\mathcal{G} \in \mathbb{R}^{P \times Q \times R}$ is a smaller **core tensor**, and $A \in \mathbb{R}^{I \times P}$, $B \in \mathbb{R}^{J \times Q}$, $C \in \mathbb{R}^{K \times R}$ are factor matrices. The $\times_n$ denotes the **mode-$n$ product** --- contracting the core tensor with a matrix along the $n$-th mode.

Tucker decomposition is a generalization of CP decomposition (if the core is superdiagonal, you get CP). It is also a generalization of PCA to higher dimensions --- the factor matrices play the role of principal directions, and the core tensor captures the interactions.

**Storage comparison.** Original tensor: $I \times J \times K$ numbers. Tucker: $PQR + IP + JQ + KR$ numbers. With $P, Q, R \ll I, J, K$, the compression is enormous.

---

## Tensors in Deep Learning

The deep learning community uses "tensor" to mean "multidimensional array." This is a simplification that drops the transformation behavior entirely. A PyTorch tensor does not know what basis it is expressed in, and it does not transform covariantly or contravariantly under anything. It is just a block of numbers with a shape.

This simplification is mostly fine for implementation. But understanding the mathematical structure reveals deeper patterns.

### Attention as a Tensor Contraction

The attention mechanism in transformers is fundamentally a tensor contraction. Given queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{n \times d_k}$, and values $V \in \mathbb{R}^{n \times d_v}$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The product $QK^T$ is a tensor contraction: $S_{ij} = \sum_k Q_{ik} K_{jk}$. In Einstein notation: $S_{ij} = Q_{ik} K_{jk}$, contracting over the shared dimension $k$. The subsequent multiplication with $V$ is another contraction: $O_{id} = A_{ij} V_{jd}$, contracting over sequence positions $j$.

Multi-head attention adds another index: the head dimension. With $h$ heads, we have $Q^{(h)}$, $K^{(h)}$, $V^{(h)}$ for each head, making the full attention a rank-4 tensor operation.

### Convolutions as Tensor Operations

A 2D convolution with input $X \in \mathbb{R}^{C_{in} \times H \times W}$ and kernel $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$ produces output:

$$Y_{o,i,j} = \sum_{c=1}^{C_{in}} \sum_{p=0}^{k-1} \sum_{q=0}^{k-1} W_{o,c,p,q} \, X_{c, \, i+p, \, j+q}$$

This is a tensor contraction with a twist --- the shifting indices $i+p, j+q$ make it a contraction with translation (technically, a cross-correlation). The kernel $W$ is a rank-4 tensor, and the convolution contracts it with the input along the channel, height-offset, and width-offset dimensions.

### What Deep Learning Loses

By treating tensors as mere arrays, deep learning loses several things:

1. **Basis independence.** Mathematical tensors are the same object regardless of basis. Neural network tensors are tied to their coordinate representation.

2. **Covariance/contravariance.** There is no distinction between inputs and outputs at the type level. A weight matrix that maps from one space to another is just a 2D array.

3. **Geometric meaning.** The metric tensor defines distances and angles. Neural networks define their own implicit geometries through learned representations, but this is not formalized in the tensor operations.

These are not just academic concerns. The Fisher information matrix (used in natural gradient methods) is a metric tensor on the space of probability distributions. Understanding it *as* a metric tensor leads to better optimization algorithms.

---

## Python: Tensor Operations in Practice

### Outer Products and Tensor Products

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Outer product: rank-1 tensors to rank-2
u = np.array([1, 2, 3])
w = np.array([4, 5, 6])

outer = np.outer(u, w)
print("u:", u)
print("w:", w)
print("u ⊗ w:")
print(outer)
# [[4, 5, 6],
#  [8, 10, 12],
#  [12, 15, 18]]
# Note: this matrix has rank 1 (all rows are multiples of w)

# Visualize the outer product as a heatmap
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# The two vectors
axes[0].barh(range(3), u, color='#2196F3')
axes[0].set_yticks(range(3))
axes[0].set_yticklabels([r'$u^1$', r'$u^2$', r'$u^3$'])
axes[0].set_title(r'Vector $\mathbf{u}$')
axes[0].invert_yaxis()

axes[1].bar(range(3), w, color='#F44336')
axes[1].set_xticks(range(3))
axes[1].set_xticklabels([r'$w^1$', r'$w^2$', r'$w^3$'])
axes[1].set_title(r'Vector $\mathbf{w}$')

# The outer product
im = axes[2].imshow(outer, cmap='Purples', aspect='equal')
axes[2].set_title(r'Outer Product $\mathbf{u} \otimes \mathbf{w}$')
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f'{outer[i,j]}', ha='center', va='center', fontsize=12)
plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig('tensor_outer_product.png', dpi=150, bbox_inches='tight')
plt.show()

# Higher-order outer product: three vectors -> rank-3 tensor
v = np.array([7, 8])
rank1_tensor = np.einsum('i,j,k->ijk', u, w, v)
print(f"\nRank-3 tensor shape: {rank1_tensor.shape}")  # (3, 3, 2)
print("This rank-3 tensor has rank 1 (single outer product of 3 vectors)")
```

### Einstein Summation with NumPy

```python
import numpy as np

# Einstein summation convention in action
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
v = np.random.randn(4)

# Matrix-vector product: w^i = A^i_j v^j
w = np.einsum('ij,j->i', A, v)
assert np.allclose(w, A @ v)

# Matrix multiplication: C^i_k = A^i_j B^j_k
C = np.einsum('ij,jk->ik', A, B)
assert np.allclose(C, A @ B)

# Trace: tr(A) = A^i_i
M = np.random.randn(4, 4)
trace = np.einsum('ii->', M)
assert np.allclose(trace, np.trace(M))

# Outer product: T^ij = u^i v^j
u = np.random.randn(3)
v = np.random.randn(5)
T = np.einsum('i,j->ij', u, v)
assert np.allclose(T, np.outer(u, v))

# Batch matrix multiply: C_bij = A_bik B_bkj
A_batch = np.random.randn(10, 3, 4)
B_batch = np.random.randn(10, 4, 5)
C_batch = np.einsum('bik,bkj->bij', A_batch, B_batch)
assert np.allclose(C_batch, A_batch @ B_batch)

# Attention scores: S_ij = Q_ik K_jk (note: K is transposed via index order)
n, d_k = 8, 64
Q = np.random.randn(n, d_k)
K = np.random.randn(n, d_k)
S = np.einsum('ik,jk->ij', Q, K) / np.sqrt(d_k)
assert np.allclose(S, Q @ K.T / np.sqrt(d_k))

print("All einsum assertions passed.")
print("\nThe power of einsum: one notation for all linear algebra operations.")
print("It is literally Einstein summation convention, implemented in code.")
```

### CP Decomposition

```python
import numpy as np
import matplotlib.pyplot as plt

def cp_als(tensor, rank, max_iter=100, tol=1e-6):
    """
    CP decomposition via Alternating Least Squares (ALS).

    The idea: fix all factor matrices except one, solve for that one
    (which is a linear least squares problem), then rotate to the next.
    Repeat until convergence.
    """
    I, J, K = tensor.shape

    # Initialize factor matrices randomly
    A = np.random.randn(I, rank)
    B = np.random.randn(J, rank)
    C = np.random.randn(K, rank)

    for iteration in range(max_iter):
        # Mode-1: Fix B, C, solve for A
        # Unfold tensor along mode 1: T_(1) is (I x JK)
        # T_(1) ≈ A * (C ⊙ B)^T  where ⊙ is Khatri-Rao product
        kr_BC = np.einsum('jr,kr->jkr', B, C).reshape(-1, rank)  # (JK x R)
        T1 = tensor.reshape(I, -1)  # (I x JK)
        A = T1 @ kr_BC @ np.linalg.pinv(kr_BC.T @ kr_BC)

        # Mode-2: Fix A, C, solve for B
        kr_AC = np.einsum('ir,kr->ikr', A, C).reshape(-1, rank)
        T2 = tensor.transpose(1, 0, 2).reshape(J, -1)
        B = T2 @ kr_AC @ np.linalg.pinv(kr_AC.T @ kr_AC)

        # Mode-3: Fix A, B, solve for C
        kr_AB = np.einsum('ir,jr->ijr', A, B).reshape(-1, rank)
        T3 = tensor.transpose(2, 0, 1).reshape(K, -1)
        C = T3 @ kr_AB @ np.linalg.pinv(kr_AB.T @ kr_AB)

        # Reconstruct and compute error
        recon = np.einsum('ir,jr,kr->ijk', A, B, C)
        error = np.linalg.norm(tensor - recon) / np.linalg.norm(tensor)

        if error < tol:
            print(f"  Converged at iteration {iteration+1}, error = {error:.6f}")
            break

    return A, B, C, error

# Create a low-rank tensor (known rank 3)
np.random.seed(42)
I, J, K = 20, 15, 10
true_rank = 3

A_true = np.random.randn(I, true_rank)
B_true = np.random.randn(J, true_rank)
C_true = np.random.randn(K, true_rank)
T_clean = np.einsum('ir,jr,kr->ijk', A_true, B_true, C_true)

# Add noise
noise_level = 0.1
T_noisy = T_clean + noise_level * np.random.randn(I, J, K)

# Decompose at various ranks
ranks = [1, 2, 3, 5, 8]
errors = []

print("CP Decomposition results:")
for r in ranks:
    print(f"\nRank {r}:")
    _, _, _, err = cp_als(T_noisy, r, max_iter=200)
    errors.append(err)
    print(f"  Relative error: {err:.6f}")

# Plot reconstruction error vs rank
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ranks, errors, 'o-', color='#9C27B0', linewidth=2, markersize=8)
ax.axvline(x=true_rank, color='red', linestyle='--', alpha=0.7, label=r'True rank $R = $' + f'{true_rank}')
ax.set_xlabel(r'CP Rank $R$', fontsize=12)
ax.set_ylabel(r'Relative Reconstruction Error', fontsize=12)
ax.set_title(r'CP Decomposition: Error vs Rank', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Storage comparison annotation
original_storage = I * J * K
for r in ranks:
    cp_storage = r * (I + J + K)
    ratio = cp_storage / original_storage
    ax.annotate(f'{ratio:.1%} storage', (r, errors[ranks.index(r)]),
                textcoords="offset points", xytext=(10, 10), fontsize=9, color='gray')

plt.tight_layout()
plt.savefig('cp_decomposition_error.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nOriginal tensor: {I*J*K} numbers")
print(f"CP rank-3: {3*(I+J+K)} numbers ({3*(I+J+K)/(I*J*K):.1%} of original)")
```

### Low-Rank Tensor Approximation

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate tensor approximation on a real-ish example:
# a 3D dataset of temperature measurements (lat x lon x time)
np.random.seed(123)

nlat, nlon, ntime = 30, 40, 24

# Create structured data: smooth spatial patterns evolving in time
lat = np.linspace(-90, 90, nlat)
lon = np.linspace(-180, 180, nlon)
time = np.linspace(0, 2*np.pi, ntime)

# Three underlying patterns (true rank ~ 3)
# Pattern 1: latitude-dependent temperature with daily cycle
pattern1 = np.cos(np.radians(lat))[:, None, None] * np.ones((1, nlon, 1)) * np.cos(time)[None, None, :]
# Pattern 2: longitudinal wave
pattern2 = np.ones((nlat, 1, 1)) * np.sin(2*np.pi*lon/360)[None, :, None] * np.sin(time + 1)[None, None, :]
# Pattern 3: localized feature
lat_bump = np.exp(-((lat - 30)**2) / 200)
lon_bump = np.exp(-((lon - 60)**2) / 800)
pattern3 = lat_bump[:, None, None] * lon_bump[None, :, None] * np.cos(2*time + 0.5)[None, None, :]

data = 3*pattern1 + 2*pattern2 + 1.5*pattern3 + 0.1*np.random.randn(nlat, nlon, ntime)

# CP decomposition at different ranks
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, rank in enumerate([1, 2, 3]):
    # Simple ALS
    A = np.random.randn(nlat, rank)
    B = np.random.randn(nlon, rank)
    C = np.random.randn(ntime, rank)

    for _ in range(200):
        kr = np.einsum('jr,kr->jkr', B, C).reshape(-1, rank)
        A = data.reshape(nlat, -1) @ kr @ np.linalg.pinv(kr.T @ kr)
        kr = np.einsum('ir,kr->ikr', A, C).reshape(-1, rank)
        B = data.transpose(1,0,2).reshape(nlon, -1) @ kr @ np.linalg.pinv(kr.T @ kr)
        kr = np.einsum('ir,jr->ijr', A, B).reshape(-1, rank)
        C = data.transpose(2,0,1).reshape(ntime, -1) @ kr @ np.linalg.pinv(kr.T @ kr)

    recon = np.einsum('ir,jr,kr->ijk', A, B, C)
    error = np.linalg.norm(data - recon) / np.linalg.norm(data)

    # Plot original slice and reconstruction at t=0
    axes[0, idx].imshow(data[:, :, 0], cmap='RdBu_r', aspect='auto')
    axes[0, idx].set_title(r'Original ($t=0$)', fontsize=11)
    axes[1, idx].imshow(recon[:, :, 0], cmap='RdBu_r', aspect='auto')
    axes[1, idx].set_title(r'CP Rank-' + f'{rank} (error={error:.3f})', fontsize=11)

for ax in axes.flat:
    ax.set_xlabel(r'Longitude $\lambda$')
    ax.set_ylabel(r'Latitude $\varphi$')

plt.suptitle(r'Low-Rank Tensor Approximation of Spatiotemporal Data', fontsize=14)
plt.tight_layout()
plt.savefig('tensor_approximation.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Conclusion

Tensors are not multidimensional arrays. They are multilinear maps that transform predictably under change of basis. This distinction matters because it tells us which operations are geometrically meaningful and which are coordinate artifacts.

The key ideas to carry forward:

1. **Tensors are defined by transformation behavior**, not by the number of indices. A multidimensional array becomes a tensor only when equipped with a transformation law.

2. **The tensor product builds higher-rank objects from lower-rank ones**, and every tensor can be decomposed as a sum of rank-1 tensor products --- this is the foundation of tensor decomposition.

3. **Einstein summation** is not just notation. It enforces the rule that only contractions between upper and lower indices (contravariant and covariant) produce basis-independent results.

4. **CP and Tucker decompositions** generalize the SVD to higher orders, enabling massive compression of structured multidimensional data.

5. **Deep learning operations are tensor operations** --- attention is contraction, convolution is contraction with translation --- but the framework drops the geometric structure that makes tensors powerful in physics.

The next time you see `torch.einsum` in a model's source code, you will know exactly what mathematical operation it represents: a tensor contraction, the same operation that Einstein used to write the field equations of general relativity in a single line.
