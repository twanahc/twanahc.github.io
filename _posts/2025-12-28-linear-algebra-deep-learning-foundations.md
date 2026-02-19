---
layout: post
title: "Linear Algebra from Scratch: The Language Every Neural Network Speaks"
date: 2025-12-28
category: math
---

Every neural network, from the simplest perceptron to the largest video diffusion transformer, is doing one thing at its core: multiplying matrices. The weights are matrices. The activations are vectors. The forward pass is a sequence of matrix multiplications interleaved with nonlinearities. The backward pass computes matrix derivatives. If you don't understand linear algebra at a deep, geometric level, you will forever be pattern-matching on API calls without understanding what's happening underneath.

This post builds linear algebra from the ground up, with an emphasis on the geometric intuitions that matter most for deep learning. We will start with vectors, build up to matrices as transformations, derive eigendecomposition and SVD, and end with the matrix calculus you need for backpropagation. Every concept is defined before it is used, every formula is derived rather than stated, and every abstraction is grounded in geometry.

---

## Table of Contents

1. [Vectors: Geometry and Data](#vectors-geometry-and-data)
2. [The Dot Product: Projection and Similarity](#the-dot-product-projection-and-similarity)
3. [Matrices as Linear Transformations](#matrices-as-linear-transformations)
4. [Matrix Multiplication as Composition](#matrix-multiplication-as-composition)
5. [Determinants: How Transformations Scale Area](#determinants-how-transformations-scale-area)
6. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
7. [Eigendecomposition](#eigendecomposition)
8. [Singular Value Decomposition](#singular-value-decomposition)
9. [SVD in Practice: Image Compression](#svd-in-practice-image-compression)
10. [Matrix Calculus for Deep Learning](#matrix-calculus-for-deep-learning)
11. [Putting It Together](#putting-it-together)

---

## Vectors: Geometry and Data

A **vector** is an ordered list of numbers. In \(\mathbb{R}^n\), a vector \(\mathbf{v}\) has \(n\) components:

$$\mathbf{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}$$

There are two ways to think about vectors, and both matter.

**The geometric view:** A vector is an arrow in space. In \(\mathbb{R}^2\), the vector \(\mathbf{v} = (3, 2)\) is an arrow from the origin to the point \((3, 2)\). It has a **magnitude** (length) and a **direction**. The magnitude is:

$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\sum_{i=1}^n v_i^2}$$

This is the Euclidean norm, which comes directly from the Pythagorean theorem generalized to \(n\) dimensions.

**The data view:** A vector is a data point. Each component represents a feature. An image with 784 pixels is a vector in \(\mathbb{R}^{784}\). A word embedding is a vector in \(\mathbb{R}^{512}\). A hidden state in a neural network is a vector in \(\mathbb{R}^{d}\), where \(d\) is the hidden dimension.

The power of linear algebra is that these two views are the same mathematical object. Every operation we define geometrically (rotation, projection, scaling) applies identically to data vectors in high dimensions, even when we can't visualize them directly.

**Vector addition** is component-wise: \(\mathbf{u} + \mathbf{v} = (u_1 + v_1, u_2 + v_2, \ldots, u_n + v_n)\). Geometrically, you place the tail of \(\mathbf{v}\) at the head of \(\mathbf{u}\) and draw the arrow from the origin to the new tip.

**Scalar multiplication** scales every component: \(c\mathbf{v} = (cv_1, cv_2, \ldots, cv_n)\). Geometrically, this stretches or shrinks the vector by factor \(c\). If \(c < 0\), it also flips the direction.

<svg viewBox="-20 -120 300 240" xmlns="http://www.w3.org/2000/svg" style="max-width:400px; display:block; margin:auto;">
  <!-- Axes -->
  <line x1="0" y1="-110" x2="0" y2="110" stroke="#555" stroke-width="0.5"/>
  <line x1="-10" y1="0" x2="280" y2="0" stroke="#555" stroke-width="0.5"/>
  <!-- Vector v = (3,2) scaled to (90, -60) in SVG coords (y is flipped) -->
  <line x1="0" y1="0" x2="90" y2="-60" stroke="#e74c3c" stroke-width="2.5" marker-end="url(#arrowR)"/>
  <text x="95" y="-62" fill="#e74c3c" font-size="14">v = (3, 2)</text>
  <!-- Vector u = (1,3) scaled to (30, -90) -->
  <line x1="0" y1="0" x2="30" y2="-90" stroke="#3498db" stroke-width="2.5" marker-end="url(#arrowB)"/>
  <text x="35" y="-92" fill="#3498db" font-size="14">u = (1, 3)</text>
  <!-- u + v -->
  <line x1="0" y1="0" x2="120" y2="-150" stroke="#2ecc71" stroke-width="2" stroke-dasharray="6,3"/>
  <line x1="90" y1="-60" x2="120" y2="-150" stroke="#3498db" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.5"/>
  <line x1="30" y1="-90" x2="120" y2="-150" stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.5"/>
  <text x="125" y="-148" fill="#2ecc71" font-size="14">u+v = (4, 5)</text>
  <!-- 2v -->
  <line x1="0" y1="0" x2="180" y2="-120" stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.6"/>
  <text x="185" y="-118" fill="#e74c3c" font-size="12" opacity="0.7">2v</text>
  <!-- Arrow markers -->
  <defs>
    <marker id="arrowR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#e74c3c"/></marker>
    <marker id="arrowB" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#3498db"/></marker>
  </defs>
</svg>

A **linear combination** of vectors \(\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\) is any expression \(c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k\) where the \(c_i\) are scalars. This is the most important concept in linear algebra. Neural network outputs are linear combinations of input features (before the nonlinearity). Attention scores produce a linear combination of value vectors. The entire output of a linear layer \(\mathbf{y} = W\mathbf{x}\) is a linear combination of the columns of \(W\), weighted by the entries of \(\mathbf{x}\).

---

## The Dot Product: Projection and Similarity

The **dot product** (or inner product) of two vectors \(\mathbf{u}, \mathbf{v} \in \mathbb{R}^n\) is:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$

This is an algebraic definition. It takes two vectors and returns a single number (a scalar). But the geometric meaning is where the insight lives.

**The geometric interpretation.** The dot product satisfies:

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \, \|\mathbf{v}\| \cos\theta$$

where \(\theta\) is the angle between the two vectors. This is not obvious from the component formula, so let's derive it.

Start with the law of cosines. For a triangle with sides \(a\), \(b\), and \(c\), where \(\theta\) is the angle between sides \(a\) and \(b\):

$$c^2 = a^2 + b^2 - 2ab\cos\theta$$

Now consider the triangle formed by vectors \(\mathbf{u}\), \(\mathbf{v}\), and \(\mathbf{u} - \mathbf{v}\). Here \(a = \|\mathbf{u}\|\), \(b = \|\mathbf{v}\|\), and \(c = \|\mathbf{u} - \mathbf{v}\|\). So:

$$\|\mathbf{u} - \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

Expand the left side using components:

$$\|\mathbf{u} - \mathbf{v}\|^2 = \sum_i (u_i - v_i)^2 = \sum_i u_i^2 - 2\sum_i u_i v_i + \sum_i v_i^2 = \|\mathbf{u}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2$$

Setting the two expressions equal:

$$\|\mathbf{u}\|^2 - 2(\mathbf{u} \cdot \mathbf{v}) + \|\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

Cancel \(\|\mathbf{u}\|^2 + \|\mathbf{v}\|^2\) from both sides:

$$-2(\mathbf{u} \cdot \mathbf{v}) = -2\|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$

This tells us three things:

1. **If \(\theta = 0\) (vectors point the same direction):** \(\cos 0 = 1\), so \(\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\). Maximum positive value.
2. **If \(\theta = 90°\) (vectors are perpendicular/orthogonal):** \(\cos 90° = 0\), so \(\mathbf{u} \cdot \mathbf{v} = 0\). Orthogonal vectors have zero dot product.
3. **If \(\theta = 180°\) (vectors point in opposite directions):** \(\cos 180° = -1\), so \(\mathbf{u} \cdot \mathbf{v} = -\|\mathbf{u}\|\|\mathbf{v}\|\). Maximum negative value.

**Cosine similarity** normalizes this: \(\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|} = \cos\theta\). This is exactly what attention mechanisms in transformers compute between query and key vectors (before the softmax). When you compute \(QK^T\), each entry is a dot product measuring how aligned a query is with a key. High alignment means high attention weight.

**Projection.** The scalar projection of \(\mathbf{u}\) onto \(\mathbf{v}\) is:

$$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$$

This gives the component of \(\mathbf{u}\) that lies along the direction of \(\mathbf{v}\). In deep learning, projection appears everywhere: residual connections project information onto subspaces, layer normalization projects onto the unit sphere, and PCA projects data onto principal components.

---

## Matrices as Linear Transformations

A **matrix** is a rectangular array of numbers. An \(m \times n\) matrix \(A\) has \(m\) rows and \(n\) columns:

$$A = \begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{pmatrix}$$

The algebraic view is that matrix-vector multiplication \(A\mathbf{x}\) produces a new vector by taking dot products of each row of \(A\) with \(\mathbf{x}\):

$$(A\mathbf{x})_i = \sum_{j=1}^n a_{ij} x_j$$

But the geometric view is far more powerful: **a matrix is a linear transformation**. It takes vectors from one space and maps them to another, preserving the operations of addition and scalar multiplication. That is, for any matrix \(A\), vectors \(\mathbf{u}, \mathbf{v}\), and scalar \(c\):

$$A(\mathbf{u} + \mathbf{v}) = A\mathbf{u} + A\mathbf{v}$$
$$A(c\mathbf{u}) = cA\mathbf{u}$$

These two properties define linearity. And here is the key theorem: **every linear transformation from \(\mathbb{R}^n\) to \(\mathbb{R}^m\) can be represented as an \(m \times n\) matrix, and every \(m \times n\) matrix represents such a transformation.** Matrices and linear transformations are the same thing.

To understand what a matrix does geometrically, look at what it does to the standard basis vectors \(\mathbf{e}_1 = (1, 0)\) and \(\mathbf{e}_2 = (0, 1)\). The first column of the matrix is where \(\mathbf{e}_1\) lands; the second column is where \(\mathbf{e}_2\) lands. Everything else follows by linearity.

Let's see the fundamental 2D transformations.

**Scaling** by factors \(s_x\) and \(s_y\):

$$S = \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix}$$

This stretches the \(x\)-axis by \(s_x\) and the \(y\)-axis by \(s_y\).

**Rotation** by angle \(\theta\) counterclockwise:

$$R = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

Why? Because \(\mathbf{e}_1 = (1, 0)\) rotated by \(\theta\) lands at \((\cos\theta, \sin\theta)\), and \(\mathbf{e}_2 = (0, 1)\) rotated by \(\theta\) lands at \((-\sin\theta, \cos\theta)\). These become the columns of the matrix.

**Shearing** in the \(x\)-direction by factor \(k\):

$$H = \begin{pmatrix} 1 & k \\ 0 & 1 \end{pmatrix}$$

This slides each point horizontally by an amount proportional to its \(y\)-coordinate, while keeping \(y\) unchanged.

<svg viewBox="-130 -130 520 260" xmlns="http://www.w3.org/2000/svg" style="max-width:600px; display:block; margin:auto;">
  <!-- Original square -->
  <g transform="translate(0,0)">
    <text x="-10" y="-115" font-size="12" fill="#d4d4d4" text-anchor="middle">Original</text>
    <line x1="-100" y1="0" x2="100" y2="0" stroke="#444" stroke-width="0.5"/>
    <line x1="0" y1="-100" x2="0" y2="100" stroke="#444" stroke-width="0.5"/>
    <rect x="-40" y="-40" width="80" height="80" fill="rgba(52,152,219,0.15)" stroke="#3498db" stroke-width="1.5"/>
    <circle cx="40" cy="-40" r="3" fill="#e74c3c"/>
    <text x="50" y="-42" font-size="10" fill="#e74c3c">(1,1)</text>
  </g>
  <!-- Rotated (45 degrees) -->
  <g transform="translate(260,0)">
    <text x="-10" y="-115" font-size="12" fill="#d4d4d4" text-anchor="middle">Rotation (45°)</text>
    <line x1="-100" y1="0" x2="100" y2="0" stroke="#444" stroke-width="0.5"/>
    <line x1="0" y1="-100" x2="0" y2="100" stroke="#444" stroke-width="0.5"/>
    <polygon points="0,-56.6 56.6,0 0,56.6 -56.6,0" fill="rgba(46,204,113,0.15)" stroke="#2ecc71" stroke-width="1.5"/>
    <circle cx="0" cy="-56.6" r="3" fill="#e74c3c"/>
    <text x="8" y="-58" font-size="10" fill="#e74c3c">(0, &#x221A;2)</text>
  </g>
</svg>

Here is a Python simulation that shows how a 2x2 matrix transforms the unit circle. Every linear transformation maps the unit circle to an ellipse --- this is a fundamental geometric fact that connects directly to the singular value decomposition.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the transformation matrix
A = np.array([[2, 1],
              [0.5, 1.5]])

# Generate points on the unit circle
theta = np.linspace(0, 2 * np.pi, 200)
circle = np.array([np.cos(theta), np.sin(theta)])  # shape (2, 200)

# Apply the transformation
ellipse = A @ circle  # shape (2, 200)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original unit circle
axes[0].plot(circle[0], circle[1], 'b-', linewidth=2)
axes[0].set_xlim(-3.5, 3.5)
axes[0].set_ylim(-3.5, 3.5)
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)
axes[0].set_title(r'Unit Circle', fontsize=14)
axes[0].axhline(y=0, color='k', linewidth=0.5)
axes[0].axvline(x=0, color='k', linewidth=0.5)
# Show basis vectors
axes[0].annotate('', xy=(1, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
axes[0].annotate('', xy=(0, 1), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
axes[0].text(1.1, 0.1, r'$\mathbf{e}_1$', fontsize=12, color='red')
axes[0].text(0.1, 1.1, r'$\mathbf{e}_2$', fontsize=12, color='green')

# Transformed ellipse
axes[1].plot(ellipse[0], ellipse[1], 'b-', linewidth=2)
axes[1].set_xlim(-3.5, 3.5)
axes[1].set_ylim(-3.5, 3.5)
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)
axes[1].set_title(rf'After $A = [[{A[0,0]}, {A[0,1]}], [{A[1,0]}, {A[1,1]}]]$', fontsize=14)
axes[1].axhline(y=0, color='k', linewidth=0.5)
axes[1].axvline(x=0, color='k', linewidth=0.5)
# Show transformed basis vectors
axes[1].annotate('', xy=(A[0,0], A[1,0]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2))
axes[1].annotate('', xy=(A[0,1], A[1,1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2))
axes[1].text(A[0,0]+0.1, A[1,0]+0.1, r'$A\mathbf{e}_1$', fontsize=12, color='red')
axes[1].text(A[0,1]+0.1, A[1,1]+0.1, r'$A\mathbf{e}_2$', fontsize=12, color='green')

plt.tight_layout()
plt.savefig('matrix_transform_circle.png', dpi=150, bbox_inches='tight')
plt.show()
```

The key observation: the circle becomes an ellipse. The lengths of the semi-axes of this ellipse are the **singular values** of the matrix. The directions of those axes are the **singular vectors**. We will make this precise shortly.

---

## Matrix Multiplication as Composition

If matrix \(A\) represents transformation \(T_A\) and matrix \(B\) represents transformation \(T_B\), then the matrix product \(AB\) represents applying \(T_B\) first, then \(T_A\). This is **composition of transformations**.

The formula for matrix multiplication: if \(A\) is \(m \times p\) and \(B\) is \(p \times n\), then \(C = AB\) is \(m \times n\) with:

$$(AB)_{ij} = \sum_{k=1}^p a_{ik} b_{kj}$$

Each entry of the result is a dot product of a row of \(A\) with a column of \(B\). The inner dimensions must match (\(p\)), and the result has the outer dimensions (\(m \times n\)).

Why does this represent composition? Consider applying \(B\) to a vector \(\mathbf{x}\) to get \(B\mathbf{x}\), then applying \(A\) to the result: \(A(B\mathbf{x})\). By the associativity of matrix multiplication, this equals \((AB)\mathbf{x}\). So the single matrix \(AB\) encodes "do \(B\), then do \(A\)."

In a neural network, a two-layer linear network computes \(\mathbf{y} = W_2(W_1\mathbf{x}) = (W_2 W_1)\mathbf{x}\). Without nonlinearities, stacking linear layers is pointless --- the composition of two linear transformations is just another linear transformation. This is the fundamental reason why neural networks need nonlinear activation functions.

**Important:** Matrix multiplication is **not commutative**. In general \(AB \neq BA\). Rotating then scaling gives a different result than scaling then rotating. Order matters. This is why the order of operations in a neural network matters, and why attention mechanisms carefully define whether query-key products come before or after value projections.

---

## Determinants: How Transformations Scale Area

The **determinant** of a square matrix tells you how the transformation scales area (in 2D) or volume (in 3D and higher). For a \(2 \times 2\) matrix:

$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

If \(\det(A) = 2\), the transformation doubles all areas. If \(\det(A) = -1\), areas are preserved but orientation is flipped (like a reflection). If \(\det(A) = 0\), the transformation collapses space to a lower dimension --- it squashes the plane into a line, or a line into a point. This means the matrix is **singular** (non-invertible), and some information is irretrievably lost.

In deep learning, a layer with a singular weight matrix destroys information. Gradients become zero (the vanishing gradient problem in its most extreme form). This is one reason weight initialization schemes are designed to keep determinants away from zero.

---

## Eigenvalues and Eigenvectors

Here is one of the most important ideas in all of mathematics.

An **eigenvector** of a matrix \(A\) is a nonzero vector \(\mathbf{v}\) that, when transformed by \(A\), only gets scaled --- its direction does not change:

$$A\mathbf{v} = \lambda\mathbf{v}$$

The scalar \(\lambda\) is the corresponding **eigenvalue**. "Eigen" is German for "own" or "characteristic" --- these are the matrix's own, characteristic directions.

Think about what this means geometrically. Most vectors get rotated and stretched when you apply a matrix. They come out pointing in a completely different direction. But eigenvectors are special: they only get stretched (or compressed, or flipped, but never rotated). The eigenvalue tells you the stretching factor.

**Finding eigenvalues.** Rearrange \(A\mathbf{v} = \lambda\mathbf{v}\):

$$A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

For this to have a nonzero solution \(\mathbf{v}\), the matrix \((A - \lambda I)\) must be singular --- it must collapse some direction. This means:

$$\det(A - \lambda I) = 0$$

This is the **characteristic equation**. For a \(2 \times 2\) matrix \(A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}\):

$$\det\begin{pmatrix} a - \lambda & b \\ c & d - \lambda \end{pmatrix} = (a - \lambda)(d - \lambda) - bc = \lambda^2 - (a+d)\lambda + (ad - bc) = 0$$

This is a quadratic in \(\lambda\). The sum of eigenvalues equals the trace \(a + d\), and the product equals the determinant \(ad - bc\). For an \(n \times n\) matrix, the characteristic equation is a degree-\(n\) polynomial, giving (counting multiplicity) exactly \(n\) eigenvalues.

Here is a Python visualization of eigenvectors. The matrix transforms all vectors, but eigenvectors maintain their direction:

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[2, 1],
              [1, 2]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{eigenvectors}")

# Generate random vectors and transform them
np.random.seed(42)
n_vectors = 12
angles = np.linspace(0, 2 * np.pi, n_vectors, endpoint=False)
vectors = np.array([np.cos(angles), np.sin(angles)])
transformed = A @ vectors

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Draw original and transformed vectors
for i in range(n_vectors):
    ax.annotate('', xy=(vectors[0, i], vectors[1, i]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1, alpha=0.5))
    ax.annotate('', xy=(transformed[0, i], transformed[1, i]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='coral', lw=1, alpha=0.5))

# Highlight eigenvectors
for j in range(2):
    ev = eigenvectors[:, j]
    lam = eigenvalues[j]
    # Original eigenvector
    ax.annotate('', xy=(ev[0], ev[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=3))
    # Transformed eigenvector (should be lambda * ev)
    ax.annotate('', xy=(lam * ev[0], lam * ev[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=3))
    ax.text(lam * ev[0] + 0.1, lam * ev[1] + 0.1,
            rf'$\lambda={lam:.1f}$', fontsize=14, color='red', fontweight='bold')

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_title('Blue = original vectors, Red = transformed by $A$\n'
             'Thick arrows: eigenvectors (direction preserved)', fontsize=13)
plt.tight_layout()
plt.savefig('eigenvectors_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

For the matrix \(\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}\), the eigenvalues are \(\lambda_1 = 3\) and \(\lambda_2 = 1\). The eigenvector for \(\lambda_1 = 3\) is along the direction \((1, 1)\) --- the matrix stretches this direction by a factor of 3. The eigenvector for \(\lambda_2 = 1\) is along \((1, -1)\) --- this direction is unchanged. Every other vector is a linear combination of these two eigenvectors, so it gets a mixture of stretching.

---

## Eigendecomposition

If an \(n \times n\) matrix \(A\) has \(n\) linearly independent eigenvectors, we can write:

$$A = V \Lambda V^{-1}$$

where \(V\) is the matrix whose columns are the eigenvectors \(\mathbf{v}_1, \ldots, \mathbf{v}_n\), and \(\Lambda\) is the diagonal matrix of eigenvalues:

$$\Lambda = \begin{pmatrix} \lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{pmatrix}$$

Why is this true? Start from \(A\mathbf{v}_i = \lambda_i \mathbf{v}_i\) for each eigenvector. Write all \(n\) equations simultaneously:

$$A \begin{pmatrix} | & | & & | \\ \mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_n \\ | & | & & | \end{pmatrix} = \begin{pmatrix} | & | & & | \\ \lambda_1\mathbf{v}_1 & \lambda_2\mathbf{v}_2 & \cdots & \lambda_n\mathbf{v}_n \\ | & | & & | \end{pmatrix}$$

The left side is \(AV\). The right side is \(V\Lambda\) (each column of \(V\) gets scaled by the corresponding diagonal entry of \(\Lambda\)). So \(AV = V\Lambda\), and multiplying both sides on the right by \(V^{-1}\) gives \(A = V\Lambda V^{-1}\).

**What this means geometrically:** To apply \(A\) to a vector, you can: (1) change to the eigenvector coordinate system (\(V^{-1}\)), (2) scale each axis by its eigenvalue (\(\Lambda\)), (3) change back to the original coordinate system (\(V\)). The matrix is "diagonal" in the right coordinate system.

**Special case: symmetric matrices.** If \(A = A^T\) (the matrix equals its transpose), then a wonderful thing happens. All eigenvalues are real, the eigenvectors are orthogonal, and we can choose them to be orthonormal. This means \(V\) is an orthogonal matrix (\(V^{-1} = V^T\)), and the decomposition simplifies to:

$$A = V\Lambda V^T$$

Symmetric matrices appear constantly in ML: covariance matrices, kernel matrices (Gram matrices), and the Hessian of the loss function are all symmetric. This is why PCA, which relies on eigendecomposition of the covariance matrix, always works cleanly.

---

## Singular Value Decomposition

Eigendecomposition requires a square matrix and doesn't always exist. The **Singular Value Decomposition** (SVD) is more general: it works for any \(m \times n\) matrix.

Every matrix \(A\) can be written as:

$$A = U \Sigma V^T$$

where:
- \(U\) is an \(m \times m\) orthogonal matrix (columns are the **left singular vectors**)
- \(\Sigma\) is an \(m \times n\) diagonal matrix (entries are the **singular values** \(\sigma_1 \geq \sigma_2 \geq \cdots \geq 0\))
- \(V\) is an \(n \times n\) orthogonal matrix (columns are the **right singular vectors**)

**The geometric meaning:** Every linear transformation can be decomposed into three steps:
1. **Rotate** (by \(V^T\)) --- align the input with the natural axes of the transformation
2. **Scale** (by \(\Sigma\)) --- stretch or compress along each axis by the singular values
3. **Rotate** (by \(U\)) --- align the output in the output space

This is why the unit circle maps to an ellipse. \(V^T\) rotates the circle (still a circle), \(\Sigma\) scales it into an ellipse aligned with the axes, and \(U\) rotates the ellipse to its final orientation. The singular values are the semi-axis lengths of the ellipse.

<svg viewBox="-20 -100 560 200" xmlns="http://www.w3.org/2000/svg" style="max-width:650px; display:block; margin:auto;">
  <!-- Step 1: Unit circle -->
  <g transform="translate(60,50)">
    <ellipse cx="0" cy="0" rx="40" ry="40" fill="rgba(52,152,219,0.15)" stroke="#3498db" stroke-width="1.5"/>
    <line x1="-50" y1="0" x2="50" y2="0" stroke="#444" stroke-width="0.5"/>
    <line x1="0" y1="-50" x2="0" y2="50" stroke="#444" stroke-width="0.5"/>
    <text x="-15" y="65" font-size="11" fill="#d4d4d4">Unit circle</text>
  </g>
  <text x="120" y="55" font-size="18" fill="#999">&#x2192;</text>
  <text x="115" y="40" font-size="10" fill="#999">V&#x1d40;</text>
  <!-- Step 2: Rotated circle (still a circle) -->
  <g transform="translate(190,50)">
    <ellipse cx="0" cy="0" rx="40" ry="40" fill="rgba(46,204,113,0.15)" stroke="#2ecc71" stroke-width="1.5"/>
    <line x1="-50" y1="0" x2="50" y2="0" stroke="#444" stroke-width="0.5"/>
    <line x1="0" y1="-50" x2="0" y2="50" stroke="#444" stroke-width="0.5"/>
    <text x="-10" y="65" font-size="11" fill="#d4d4d4">Rotated</text>
  </g>
  <text x="250" y="55" font-size="18" fill="#999">&#x2192;</text>
  <text x="248" y="40" font-size="10" fill="#999">&#x3A3;</text>
  <!-- Step 3: Axis-aligned ellipse -->
  <g transform="translate(330,50)">
    <ellipse cx="0" cy="0" rx="55" ry="25" fill="rgba(231,76,60,0.15)" stroke="#e74c3c" stroke-width="1.5"/>
    <line x1="-65" y1="0" x2="65" y2="0" stroke="#444" stroke-width="0.5"/>
    <line x1="0" y1="-50" x2="0" y2="50" stroke="#444" stroke-width="0.5"/>
    <text x="-10" y="65" font-size="11" fill="#d4d4d4">Scaled</text>
    <text x="58" y="-4" font-size="10" fill="#e74c3c">σ₁</text>
    <text x="4" y="-27" font-size="10" fill="#e74c3c">σ₂</text>
  </g>
  <text x="400" y="55" font-size="18" fill="#999">&#x2192;</text>
  <text x="398" y="40" font-size="10" fill="#999">U</text>
  <!-- Step 4: Rotated ellipse -->
  <g transform="translate(470,50)">
    <ellipse cx="0" cy="0" rx="55" ry="25" fill="rgba(155,89,182,0.15)" stroke="#9b59b6" stroke-width="1.5" transform="rotate(-30)"/>
    <line x1="-65" y1="0" x2="65" y2="0" stroke="#444" stroke-width="0.5"/>
    <line x1="0" y1="-50" x2="0" y2="50" stroke="#444" stroke-width="0.5"/>
    <text x="-10" y="65" font-size="11" fill="#d4d4d4">Final</text>
  </g>
</svg>

**Low-rank approximation.** The SVD immediately gives you the best rank-\(k\) approximation to any matrix. Write the SVD as a sum of rank-1 matrices:

$$A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

where \(r\) is the rank of \(A\). The best rank-\(k\) approximation (in the Frobenius norm or the operator norm) is:

$$A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

Just keep the \(k\) largest singular values and discard the rest. This is the **Eckart-Young theorem**, and it is the theoretical foundation for dimensionality reduction, PCA, and low-rank weight compression in neural networks.

**Connection to PCA.** Principal Component Analysis finds the directions of maximum variance in data. If your data matrix is \(X\) (centered to have zero mean, with each row as a data point), then the covariance matrix is \(\frac{1}{n-1}X^TX\). The principal components are the eigenvectors of this covariance matrix --- which are exactly the right singular vectors of \(X\). PCA and SVD are two views of the same computation.

---

## SVD in Practice: Image Compression

Nothing builds intuition for SVD like watching it compress an image. Each rank-1 component adds detail, and you can see the singular values at work:

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Create a synthetic test image (gradient + shapes)
# (Replace with any grayscale image: np.array(Image.open('photo.jpg').convert('L'), dtype=float))
x = np.linspace(0, 1, 256)
y = np.linspace(0, 1, 256)
X, Y = np.meshgrid(x, y)
img = (50 * np.sin(6 * np.pi * X) * np.cos(4 * np.pi * Y)
       + 100 * np.exp(-((X - 0.3)**2 + (Y - 0.7)**2) / 0.02)
       + 80 * ((X - 0.7)**2 + (Y - 0.3)**2 < 0.04).astype(float)
       + 30 * X + 20 * Y)

# Compute SVD
U, S, Vt = np.linalg.svd(img, full_matrices=False)

# Reconstruct at various ranks
ranks = [1, 5, 10, 20, 50, 256]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, k in zip(axes.flatten(), ranks):
    # Rank-k approximation
    img_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    compression = k * (256 + 256 + 1) / (256 * 256) * 100
    ax.imshow(img_k, cmap='gray')
    ax.set_title(rf'Rank $k={k}$ ({compression:.1f}% of original)', fontsize=13)
    ax.axis('off')

plt.suptitle(r'SVD Image Compression at Different Ranks', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('svd_compression.png', dpi=150, bbox_inches='tight')
plt.show()

# Plot singular value spectrum
fig, ax = plt.subplots(figsize=(10, 5))
ax.semilogy(S, 'b-', linewidth=2)
ax.set_xlabel(r'Index $i$', fontsize=13)
ax.set_ylabel(r'Singular Value $\sigma_i$ (log scale)', fontsize=13)
ax.set_title(r'Singular Value Spectrum', fontsize=14)
ax.grid(True, alpha=0.3)
# Mark the ranks we used
for k in ranks[:-1]:
    ax.axvline(x=k, color='red', linestyle='--', alpha=0.5)
    ax.text(k + 1, S[0] * 0.5, rf'$k={k}$', fontsize=10, color='red')
plt.tight_layout()
plt.savefig('singular_value_spectrum.png', dpi=150, bbox_inches='tight')
plt.show()
```

The singular values typically decay rapidly. The first few capture the large-scale structure (overall brightness, major shapes), and the tail captures fine details and noise. This is why low-rank approximations work so well: most of the "information" is concentrated in a small number of singular values.

In modern deep learning, this principle is exploited directly. **LoRA** (Low-Rank Adaptation) fine-tunes large models by adding low-rank updates \(\Delta W = BA\) where \(B\) is \(d \times r\) and \(A\) is \(r \times d\), with \(r \ll d\). This works because the weight changes needed for fine-tuning typically lie in a low-dimensional subspace --- exactly the situation where SVD tells us low-rank approximation is effective.

---

## Matrix Calculus for Deep Learning

Training a neural network requires computing gradients of a scalar loss \(L\) with respect to matrix and vector parameters. This requires matrix calculus.

**Gradient of a scalar with respect to a vector.** If \(f: \mathbb{R}^n \to \mathbb{R}\) is a scalar-valued function of a vector \(\mathbf{x}\), its gradient is:

$$\nabla_{\mathbf{x}} f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

The gradient is a vector that points in the direction of steepest ascent of \(f\). Gradient descent moves in the opposite direction: \(\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla_{\mathbf{x}} f\).

**Key gradient identities.** These appear in nearly every backpropagation derivation:

For \(f(\mathbf{x}) = \mathbf{a}^T\mathbf{x} = \mathbf{a} \cdot \mathbf{x}\):

$$\nabla_{\mathbf{x}} (\mathbf{a}^T\mathbf{x}) = \mathbf{a}$$

This makes sense: \(\mathbf{a}^T\mathbf{x}\) is a linear function, and its rate of change in any direction is constant.

For \(f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}\) (a quadratic form):

$$\nabla_{\mathbf{x}} (\mathbf{x}^T A \mathbf{x}) = (A + A^T)\mathbf{x}$$

If \(A\) is symmetric, this simplifies to \(2A\mathbf{x}\).

**The Jacobian.** If \(\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m\) is a vector-valued function, the Jacobian is the \(m \times n\) matrix of all partial derivatives:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{pmatrix}$$

Row \(i\) of the Jacobian is the gradient of \(f_i\). The Jacobian is the correct generalization of the derivative to vector-valued functions.

**The chain rule in matrix form.** If \(\mathbf{y} = \mathbf{f}(\mathbf{x})\) and \(L = g(\mathbf{y})\), then:

$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \cdot J_f$$

In backpropagation, you propagate the gradient backward by multiplying by Jacobians at each layer. For a linear layer \(\mathbf{y} = W\mathbf{x}\), the Jacobian with respect to \(\mathbf{x}\) is \(W\), so:

$$\frac{\partial L}{\partial \mathbf{x}} = W^T \frac{\partial L}{\partial \mathbf{y}}$$

The transpose of the weight matrix appears naturally in the backward pass. The gradient with respect to the weight matrix itself is:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{y}} \mathbf{x}^T$$

This is an outer product of the upstream gradient with the input. This formula is the heart of backpropagation for linear layers, and understanding it requires understanding everything we have built: vectors, dot products, matrix multiplication, and the chain rule.

---

## Putting It Together

Here is the roadmap of how these concepts connect to deep learning:

| Linear Algebra Concept | Deep Learning Application |
|---|---|
| Vector | Activations, embeddings, gradients |
| Dot product | Attention scores, cosine similarity |
| Matrix-vector multiply | Linear layer forward pass |
| Matrix composition | Multi-layer networks (without nonlinearities) |
| Eigenvalues | Learning rate analysis, spectral normalization |
| SVD | Low-rank approximation, LoRA, PCA |
| Gradient / Jacobian | Backpropagation |
| Determinant | Normalizing flows (change of variables) |

Linear algebra is not a prerequisite you check off and forget. It is the language the computation speaks at every layer, every timestep, every gradient update. The geometric intuitions --- transformations as shape-changing operations, eigenvalues as invariant stretching factors, singular values as the natural "resolution" of a matrix --- are the intuitions that let you reason about what a neural network is actually doing inside.

The next post in this series covers probability theory, where we shift from the deterministic world of linear transformations to the mathematical framework for reasoning under uncertainty. The two threads will merge when we reach information theory and understand why loss functions take the forms they do.
