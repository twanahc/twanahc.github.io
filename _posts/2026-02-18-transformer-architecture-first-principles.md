---
layout: post
title: "The Transformer from First Principles: Attention, Multi-Head Projections, and Why It All Works"
date: 2026-02-18
category: math
---

The transformer is the engine behind every major language model, image generator, and video synthesis system built in the last four years. GPT, DALL-E, Sora, Gemini, Stable Diffusion --- all transformers. Yet most explanations either stay at the hand-wavy "attention lets the model focus on relevant parts" level or dump the full equations without explaining where they come from or why they take the form they do.

This post builds the transformer from the ground up. We start with the raw problem --- processing sequences --- and construct every component from necessity. Each design choice exists because the alternative fails in a specific, demonstrable way. By the end, you will understand not just what the transformer computes, but why each piece is shaped exactly the way it is.

---

## Table of Contents

1. [The Sequence-to-Sequence Problem](#the-sequence-to-sequence-problem)
2. [Why Not RNNs?](#why-not-rnns)
3. [The Attention Mechanism from Scratch](#the-attention-mechanism-from-scratch)
4. [Geometric Interpretation of Attention](#geometric-interpretation-of-attention)
5. [Multi-Head Attention](#multi-head-attention)
6. [Positional Encoding](#positional-encoding)
7. [The Full Transformer Block](#the-full-transformer-block)
8. [KV Cache: Why Autoregressive Generation is Memory-Bound](#kv-cache)
9. [DiT: Diffusion Transformers for Video](#dit-diffusion-transformers-for-video)
10. [Python Demonstration](#python-demonstration)

---

## The Sequence-to-Sequence Problem

Nearly everything interesting in machine learning involves sequences. A sentence is a sequence of words. An image is a sequence of patches. A video is a sequence of frames, each of which is a sequence of patches. Audio is a sequence of spectral snapshots.

Let us formalize the problem. Suppose we have a sequence of \(n\) **tokens**. A token is the atomic unit of our sequence --- a word, a subword piece, an image patch, a video frame tile. Each token is represented as a vector in \(\mathbb{R}^d\), meaning each token is described by \(d\) numbers. We call \(d\) the **embedding dimension** or **model dimension**.

Stack all \(n\) token vectors as rows of a matrix:

$$
X \in \mathbb{R}^{n \times d}
$$

Row \(i\) of \(X\), written \(x_i \in \mathbb{R}^d\), is the representation of the \(i\)-th token. Initially, these vectors come from an embedding table (for text) or a patchification layer (for images/video). They encode what each token is, but not how it relates to the other tokens in the sequence.

The fundamental problem: **transform \(X\) into a new representation \(Y \in \mathbb{R}^{n \times d}\) where each token's representation incorporates information from the entire sequence.**

Token 5 in the sentence "The cat sat on the mat" needs to know about token 2 ("cat") and token 3 ("sat") to properly represent "the" (which "the"? the one near the mat, not the one near the cat). Token representations must be **context-dependent**.

The question is: what function \(f\) should map \(X\) to \(Y\)?

---

## Why Not RNNs?

The first generation solution was the **recurrent neural network** (RNN). An RNN processes the sequence one token at a time, maintaining a hidden state \(h_t\) that accumulates information:

$$
h_t = \sigma(W_h h_{t-1} + W_x x_t + b)
$$

Here \(\sigma\) is a nonlinear activation function (like tanh), \(W_h\) and \(W_x\) are weight matrices, and \(b\) is a bias vector. The hidden state \(h_t\) depends on the previous hidden state \(h_{t-1}\) and the current input \(x_t\).

This has three serious problems.

**Problem 1: Sequential computation.** To compute \(h_{100}\), you need \(h_{99}\), which needs \(h_{98}\), all the way back to \(h_1\). You cannot parallelize this. Processing a sequence of length \(n\) takes \(O(n)\) sequential steps. On modern GPUs with thousands of cores, this is catastrophic --- the hardware sits idle while you wait for each step to finish.

**Problem 2: Vanishing gradients.** During backpropagation, the gradient of the loss with respect to early tokens must flow through every intermediate hidden state. Each step multiplies by \(W_h\). If the spectral norm of \(W_h\) (its largest singular value) is less than 1, the gradient shrinks exponentially. After 100 steps, the gradient from the loss to token 1 is scaled by roughly \(\|W_h\|^{100}\), which is astronomically small. The model cannot learn long-range dependencies. LSTMs and GRUs mitigate this with gating mechanisms, but they do not eliminate it.

**Problem 3: Information bottleneck.** All information about the past must fit into the fixed-size hidden state \(h_t \in \mathbb{R}^{d_h}\). The model compresses the entire history of the sequence into a single vector. For long sequences, this is an impossible compression task. Information from early tokens gets overwritten.

What we want: a mechanism where every token can directly attend to every other token, in parallel, with no sequential bottleneck. That mechanism is **attention**.

---

## The Attention Mechanism from Scratch

### The Core Idea

Start with the simplest version of the problem. We have \(n\) tokens, each represented by a vector \(x_i \in \mathbb{R}^d\). We want to produce a new representation \(y_i\) for token \(i\) that incorporates information from all tokens in the sequence. The most natural approach: **compute \(y_i\) as a weighted average of all token representations**, where the weights reflect how relevant each token is to token \(i\).

$$
y_i = \sum_{j=1}^{n} \alpha_{ij} \, x_j
$$

Here \(\alpha_{ij}\) is the **attention weight** --- how much token \(i\) should attend to token \(j\). The weights must be non-negative and sum to 1 (so they form a probability distribution): \(\alpha_{ij} \geq 0\) and \(\sum_j \alpha_{ij} = 1\).

The question is: how do we compute \(\alpha_{ij}\)?

### Queries, Keys, and Values

The key insight --- and the entire intellectual contribution of the attention mechanism --- is to separate three roles that each token plays:

- **Query (\(q_i\)):** "What am I looking for?" --- the representation of token \(i\) when it is asking for information.
- **Key (\(k_j\)):** "What do I contain?" --- the representation of token \(j\) when it is advertising its content.
- **Value (\(v_j\)):** "What information do I provide?" --- the actual content that token \(j\) contributes when attended to.

These are computed by multiplying the input by three learned weight matrices:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

where \(W_Q, W_K \in \mathbb{R}^{d \times d_k}\) and \(W_V \in \mathbb{R}^{d \times d_v}\). Here \(d_k\) is the dimension of the query/key space, and \(d_v\) is the dimension of the value space. Often \(d_k = d_v = d\) for single-head attention, but we will see later that multi-head attention uses smaller dimensions.

Why separate Q, K, and V? Because the notion of "relevance" (what should I attend to?) is fundamentally different from "content" (what information should I extract?). The word "bank" in "river bank" and "bank account" has different relevance to the word "water" (high vs. low) but might have similar raw content. By using separate projections, the model can learn to decouple these roles.

### The Attention Score

The relevance of token \(j\) to token \(i\) is measured by the **dot product** of the query of \(i\) with the key of \(j\):

$$
e_{ij} = q_i \cdot k_j = q_i^T k_j
$$

Why the dot product? The dot product of two vectors measures their alignment. If \(q_i\) and \(k_j\) point in similar directions in \(\mathbb{R}^{d_k}\), the dot product is large and positive. If they are orthogonal, it is zero. If they point in opposite directions, it is large and negative. So the dot product naturally measures "how well does key \(j\) match query \(i\)?" in the learned subspace.

In matrix form, computing all pairwise dot products at once:

$$
E = QK^T \in \mathbb{R}^{n \times n}
$$

Entry \((i, j)\) of this matrix is \(q_i^T k_j\) --- the raw attention score from token \(i\) to token \(j\).

### The Scaling Factor: Why \(\sqrt{d_k}\)?

Here is where things get interesting. We do not use \(E\) directly. We divide by \(\sqrt{d_k}\):

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Why? This is not a heuristic. It comes from a precise variance calculation.

Assume the entries of \(q_i\) and \(k_j\) are independent random variables with mean 0 and variance 1 (which is approximately true after standard initialization and layer normalization). The dot product is:

$$
e_{ij} = q_i^T k_j = \sum_{m=1}^{d_k} q_{i,m} \cdot k_{j,m}
$$

Each term \(q_{i,m} \cdot k_{j,m}\) has mean 0 (product of two zero-mean independent variables) and variance 1 (since \(\text{Var}(q_{i,m} \cdot k_{j,m}) = \text{Var}(q_{i,m})\text{Var}(k_{j,m}) = 1 \cdot 1 = 1\) for independent zero-mean variables).

Since we sum \(d_k\) such independent terms:

$$
\text{Var}(e_{ij}) = d_k
$$

So the standard deviation of \(e_{ij}\) is \(\sqrt{d_k}\). For \(d_k = 64\) (a typical value), the dot products have standard deviation 8. For \(d_k = 512\), it is about 22.6.

Now consider what happens when we feed these into the **softmax** function. Softmax is defined as:

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

When the input values \(z_i\) have large magnitude, softmax **saturates**: almost all the probability mass concentrates on the single largest entry, and the output approaches a one-hot vector. This is a problem because:

1. The gradients of softmax vanish when it saturates (the derivative of a near-constant function is near zero).
2. The model cannot express "attend moderately to several tokens" --- it is forced into hard, winner-take-all attention.

By dividing by \(\sqrt{d_k}\), we normalize the dot products to have unit variance regardless of \(d_k\):

$$
\text{Var}\!\left(\frac{e_{ij}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

This keeps the softmax in its well-behaved regime where gradients flow and the model can learn soft, distributed attention patterns.

### The Full Attention Equation

Putting it all together:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

The computation proceeds in three stages:

1. **Compute scores:** \(S = QK^T / \sqrt{d_k} \in \mathbb{R}^{n \times n}\)
2. **Normalize:** \(A = \text{softmax}(S) \in \mathbb{R}^{n \times n}\) (softmax applied row-wise, so each row sums to 1)
3. **Aggregate:** \(Y = AV \in \mathbb{R}^{n \times d_v}\)

The output \(Y\) has the same number of tokens as the input, but each token's representation is now a weighted combination of all value vectors, with weights determined by query-key similarity.

**Computational cost:** The dominant term is the matrix multiplication \(QK^T\), which requires \(O(n^2 d_k)\) operations. This is the quadratic cost of attention that has driven years of research into efficient alternatives. For a sequence of 8192 tokens with \(d_k = 64\), this is about \(8192^2 \times 64 \approx 4.3 \times 10^9\) multiply-add operations per attention layer.

---

## Geometric Interpretation of Attention

The attention mechanism has a beautiful geometric interpretation that makes the whole construction feel inevitable rather than arbitrary.

<svg viewBox="0 0 700 380" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; width:100%; height:auto; display:block; margin:2em auto;">
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#999"/>
    </marker>
    <marker id="arrowblue" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#2563eb"/>
    </marker>
    <marker id="arrowred" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#dc2626"/>
    </marker>
    <marker id="arrowgreen" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#16a34a"/>
    </marker>
  </defs>

  <!-- Title -->
  <text x="350" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#d4d4d4">Attention as Soft Dictionary Lookup</text>

  <!-- Left panel: Query-Key space -->
  <rect x="20" y="40" width="300" height="300" rx="8" fill="#1e1e1e" stroke="#444" stroke-width="1.5"/>
  <text x="170" y="65" text-anchor="middle" font-size="13" font-weight="bold" fill="#b0b8c4">Query-Key Space (d_k dimensions)</text>

  <!-- Query vector -->
  <line x1="170" y1="200" x2="280" y2="110" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrowblue)"/>
  <text x="285" y="105" font-size="12" fill="#2563eb" font-weight="bold">q_i</text>

  <!-- Key vectors -->
  <line x1="170" y1="200" x2="270" y2="140" stroke="#dc2626" stroke-width="2" marker-end="url(#arrowred)"/>
  <text x="275" y="138" font-size="11" fill="#dc2626">k_1 (similar)</text>

  <line x1="170" y1="200" x2="100" y2="280" stroke="#dc2626" stroke-width="2" marker-end="url(#arrowred)" stroke-dasharray="6,3"/>
  <text x="55" y="295" font-size="11" fill="#dc2626">k_2 (opposite)</text>

  <line x1="170" y1="200" x2="90" y2="150" stroke="#dc2626" stroke-width="2" marker-end="url(#arrowred)" stroke-dasharray="3,3"/>
  <text x="40" y="145" font-size="11" fill="#dc2626">k_3 (orthogonal)</text>

  <!-- Angle arc for q-k1 -->
  <path d="M 215 168 A 30 30 0 0 0 222 160" stroke="#9333ea" stroke-width="1.5" fill="none"/>
  <text x="230" y="168" font-size="10" fill="#9333ea">small angle</text>

  <!-- Origin dot -->
  <circle cx="170" cy="200" r="3" fill="#d4d4d4"/>

  <!-- Right panel: Output construction -->
  <rect x="380" y="40" width="300" height="300" rx="8" fill="#1e1e1e" stroke="#444" stroke-width="1.5"/>
  <text x="530" y="65" text-anchor="middle" font-size="13" font-weight="bold" fill="#b0b8c4">Value Aggregation</text>

  <!-- Softmax weights -->
  <rect x="405" y="85" width="250" height="70" rx="5" fill="#1a2744" stroke="#2563eb" stroke-width="1"/>
  <text x="530" y="105" text-anchor="middle" font-size="12" fill="#60a5fa">softmax(q_i · k_j / sqrt(d_k))</text>
  <text x="530" y="125" text-anchor="middle" font-size="14" fill="#60a5fa" font-weight="bold">alpha_1 = 0.72   alpha_2 = 0.03   alpha_3 = 0.25</text>
  <text x="530" y="145" text-anchor="middle" font-size="10" fill="#999">weights sum to 1 (probability distribution)</text>

  <!-- Value vectors -->
  <rect x="410" y="175" width="70" height="40" rx="4" fill="#14332a" stroke="#16a34a" stroke-width="1.5"/>
  <text x="445" y="200" text-anchor="middle" font-size="12" fill="#16a34a" font-weight="bold">v_1</text>

  <rect x="495" y="175" width="70" height="40" rx="4" fill="#14332a" stroke="#16a34a" stroke-width="1.5"/>
  <text x="530" y="200" text-anchor="middle" font-size="12" fill="#16a34a" font-weight="bold">v_2</text>

  <rect x="580" y="175" width="70" height="40" rx="4" fill="#14332a" stroke="#16a34a" stroke-width="1.5"/>
  <text x="615" y="200" text-anchor="middle" font-size="12" fill="#16a34a" font-weight="bold">v_3</text>

  <!-- Weighted sum arrows -->
  <line x1="445" y1="215" x2="505" y2="268" stroke="#16a34a" stroke-width="2" marker-end="url(#arrowgreen)"/>
  <text x="455" y="245" font-size="10" fill="#16a34a">x 0.72</text>

  <line x1="530" y1="215" x2="515" y2="268" stroke="#16a34a" stroke-width="1" marker-end="url(#arrowgreen)" stroke-dasharray="4,3"/>
  <text x="535" y="248" font-size="10" fill="#888">x 0.03</text>

  <line x1="615" y1="215" x2="530" y2="268" stroke="#16a34a" stroke-width="1.5" marker-end="url(#arrowgreen)"/>
  <text x="585" y="248" font-size="10" fill="#16a34a">x 0.25</text>

  <!-- Output -->
  <rect x="470" y="270" width="120" height="45" rx="6" fill="#2563eb" stroke="#1d4ed8" stroke-width="2"/>
  <text x="530" y="290" text-anchor="middle" font-size="12" fill="white" font-weight="bold">y_i = sum</text>
  <text x="530" y="306" text-anchor="middle" font-size="11" fill="#bfdbfe">context-aware output</text>

  <!-- Arrow connecting panels -->
  <line x1="325" y1="200" x2="375" y2="200" stroke="#888" stroke-width="2" marker-end="url(#arrowhead)" stroke-dasharray="6,3"/>
  <text x="350" y="190" text-anchor="middle" font-size="10" fill="#888">scores</text>
</svg>

**Think of attention as a soft dictionary lookup.** In a regular dictionary (hash map), you provide a query key and get back exactly one value. In attention, the query \(q_i\) is compared against all keys \(k_1, \ldots, k_n\). Instead of returning the value of the single best match, attention returns a **weighted combination** of all values, with weights proportional to how well each key matches the query.

The Q and K projections define a learned similarity space. Two tokens might be dissimilar in the raw embedding space but highly similar in the query-key subspace, because the model has learned that the relationship between them matters for the task. The V projection defines what content to extract --- the model can learn to extract different information from a token than what it uses to determine relevance.

This factorization into Q, K, V is what gives attention its expressive power. A simple dot product between raw token embeddings \(x_i^T x_j\) would conflate relevance with content. The three separate projections let the model independently learn: what to search for, what to advertise, and what to transmit.

---

## Multi-Head Attention

### Why One Head Is Not Enough

A single attention head computes one set of attention weights \(A \in \mathbb{R}^{n \times n}\). This means each token computes one probability distribution over all other tokens and extracts one weighted combination of values.

But tokens need to attend to multiple things simultaneously. Consider the sentence: "The animal didn't cross the street because it was too wide." The word "it" needs to simultaneously:

- Attend to "street" (syntactic co-reference --- what does "it" refer to?)
- Attend to "wide" (semantic agreement --- what property is being described?)
- Attend to "cross" (verb relationship --- what action was attempted?)

A single softmax distribution must concentrate its probability mass. It cannot give high weight to three different tokens for three different reasons. It can attend to "street" or "wide" or "cross" but not all three with distinct purpose.

### The Multi-Head Construction

The solution: run \(h\) attention heads in parallel, each with its own Q, K, V projections, each learning to attend to different aspects of the input.

For head \(i \in \{1, \ldots, h\}\):

$$
Q_i = XW_Q^{(i)}, \quad K_i = XW_K^{(i)}, \quad V_i = XW_V^{(i)}
$$

where \(W_Q^{(i)}, W_K^{(i)} \in \mathbb{R}^{d \times d_k}\) and \(W_V^{(i)} \in \mathbb{R}^{d \times d_v}\), with \(d_k = d_v = d / h\).

Each head computes attention independently:

$$
\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i \in \mathbb{R}^{n \times d_k}
$$

The outputs of all heads are concatenated and projected back to the model dimension:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

where \(W_O \in \mathbb{R}^{d \times d}\) is the output projection matrix. Since each head produces output in \(\mathbb{R}^{n \times d_k}\) and \(d_k = d/h\), the concatenation gives \(\mathbb{R}^{n \times d}\), and after multiplying by \(W_O\) we get \(\mathbb{R}^{n \times d}\) --- the same shape as the input.

### The Computational Cost Trick

Here is the remarkable fact: **multi-head attention costs the same as single-head attention** with the full dimension \(d\).

Single-head attention with dimension \(d\):
- \(Q, K, V\) computation: \(3 \times (n \times d \times d) = 3nd^2\)
- \(QK^T\): \(n \times n \times d = n^2 d\)
- \(AV\): \(n \times n \times d = n^2 d\)
- Total: \(3nd^2 + 2n^2 d\)

Multi-head attention with \(h\) heads, each of dimension \(d_k = d/h\):
- Per head: \(Q_i, K_i, V_i\) computation: \(3 \times (n \times d \times d/h) = 3nd^2/h\)
- Per head: \(Q_i K_i^T\): \(n^2 \times d/h\)
- Per head: \(A_i V_i\): \(n^2 \times d/h\)
- Sum over \(h\) heads: \(h \times (3nd^2/h + 2n^2 d/h) = 3nd^2 + 2n^2 d\)
- Output projection \(W_O\): \(nd^2\)
- Total: \(4nd^2 + 2n^2 d\)

The difference is only the output projection, which is a minor addition. The \(n^2 d\) terms --- the expensive part --- are identical. We get \(h\) independent attention patterns for essentially the same computational budget as one. This is why multi-head attention is used universally: it is strictly more expressive at roughly the same cost.

In practice, the multi-head computation is implemented as a single large matrix multiplication followed by a reshape, not as \(h\) separate operations. You compute \(Q = XW_Q\) where \(W_Q \in \mathbb{R}^{d \times d}\), then reshape the result from \((n, d)\) to \((n, h, d_k)\) and transpose to \((h, n, d_k)\). This allows batched matrix multiplication across heads, fully utilizing GPU parallelism.

---

## Positional Encoding

### Attention Is Permutation-Equivariant

There is a critical property of the attention mechanism that we need to address. Consider what happens if we permute the input sequence --- shuffle the tokens in \(X\) using some permutation \(\pi\).

Let \(P_\pi\) be the permutation matrix corresponding to \(\pi\). The permuted input is \(P_\pi X\). Let us trace through the attention computation:

$$
Q' = P_\pi X W_Q = P_\pi Q
$$

$$
K' = P_\pi X W_K = P_\pi K
$$

$$
V' = P_\pi X W_V = P_\pi V
$$

The attention scores become:

$$
Q' K'^T = (P_\pi Q)(P_\pi K)^T = P_\pi Q K^T P_\pi^T
$$

After softmax (which is applied row-wise and is thus equivariant to row permutations):

$$
A' = P_\pi A P_\pi^T
$$

The output:

$$
Y' = A' V' = P_\pi A P_\pi^T P_\pi V = P_\pi A V = P_\pi Y
$$

So if you permute the input by \(\pi\), the output is permuted by exactly the same \(\pi\). This is **permutation equivariance**: the attention mechanism treats the input as a **set**, not a sequence. It has no notion of order.

This is a problem. "The cat sat on the mat" and "mat the on sat cat the" contain the same tokens, but the first is English and the second is not. We need to explicitly inject position information.

### Sinusoidal Positional Encoding

The original transformer paper (Vaswani et al., 2017) proposed adding a **positional encoding** vector \(p_{\text{pos}} \in \mathbb{R}^d\) to each token embedding before attention:

$$
\tilde{x}_i = x_i + p_i
$$

The sinusoidal encoding defines each component of the positional vector as follows. For position \(\text{pos}\) (which integer position in the sequence, starting from 0) and dimension index \(i\) (from 0 to \(d-1\)):

$$
p_{\text{pos}, 2i} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

$$
p_{\text{pos}, 2i+1} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d}}\right)
$$

Each pair of dimensions \((2i, 2i+1)\) is a sinusoid at a different frequency. The frequency decreases exponentially with the dimension index: low-indexed dimensions oscillate rapidly (capturing fine-grained position differences), while high-indexed dimensions oscillate slowly (capturing coarse position).

### Why Sinusoids? The Rotation Matrix Argument

This is not an arbitrary choice. Sinusoids have a special property: **the encoding of a relative position offset can be expressed as a linear transformation of the encoding at any absolute position.**

Define the frequency for dimension pair \(i\): \(\omega_i = 1 / 10000^{2i/d}\). The encoding for position \(\text{pos}\) in dimensions \((2i, 2i+1)\) is the 2D vector:

$$
\begin{pmatrix} \sin(\omega_i \cdot \text{pos}) \\ \cos(\omega_i \cdot \text{pos}) \end{pmatrix}
$$

The encoding for position \(\text{pos} + k\) (offset by \(k\)) is:

$$
\begin{pmatrix} \sin(\omega_i (\text{pos} + k)) \\ \cos(\omega_i (\text{pos} + k)) \end{pmatrix}
$$

Using the angle addition formulas:

$$
\sin(\omega_i \text{pos} + \omega_i k) = \sin(\omega_i \text{pos})\cos(\omega_i k) + \cos(\omega_i \text{pos})\sin(\omega_i k)
$$

$$
\cos(\omega_i \text{pos} + \omega_i k) = \cos(\omega_i \text{pos})\cos(\omega_i k) - \sin(\omega_i \text{pos})\sin(\omega_i k)
$$

In matrix form:

$$
\begin{pmatrix} \sin(\omega_i (\text{pos}+k)) \\ \cos(\omega_i (\text{pos}+k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{pmatrix} \begin{pmatrix} \sin(\omega_i \text{pos}) \\ \cos(\omega_i \text{pos}) \end{pmatrix}
$$

The matrix on the right is a **rotation matrix** \(R_k\) that depends only on the offset \(k\), not on the absolute position \(\text{pos}\). This means: to get from the encoding at position \(\text{pos}\) to the encoding at position \(\text{pos} + k\), you apply a fixed linear transformation. The dot product \(p_{\text{pos}}^T p_{\text{pos}+k}\) therefore depends only on \(k\), the relative distance, not on the absolute positions. This is exactly what we want --- attention should be able to detect "these two tokens are 3 positions apart" without needing to memorize absolute positions.

### RoPE: The Modern Alternative

Most modern transformers (LLaMA, Mistral, GPT-NeoX, and others) use **Rotary Position Embeddings** (RoPE), which applies the rotation directly to the query and key vectors rather than adding to the input:

$$
q_i' = R_i q_i, \quad k_j' = R_j k_j
$$

where \(R_i\) is a block-diagonal rotation matrix that rotates pairs of dimensions by angles proportional to position \(i\). The attention score becomes:

$$
q_i'^T k_j' = q_i^T R_i^T R_j k_j = q_i^T R_{j-i} k_j
$$

This naturally encodes relative position \(j - i\) into the attention score. The advantage over additive sinusoidal encoding: position information is injected directly into the similarity computation rather than being mixed into the token representation, which empirically gives better performance on long-context tasks.

---

## The Full Transformer Block

We now have all the ingredients to build a complete transformer block. Each block contains two sub-layers: multi-head self-attention and a position-wise feed-forward network, each wrapped with a residual connection and layer normalization.

<svg viewBox="0 0 480 680" xmlns="http://www.w3.org/2000/svg" style="max-width:480px; width:100%; height:auto; display:block; margin:2em auto;">
  <!-- Background removed for dark theme -->

  <!-- Input -->
  <rect x="160" y="10" width="160" height="36" rx="6" fill="#0c2d4a" stroke="#0284c7" stroke-width="1.5"/>
  <text x="240" y="33" text-anchor="middle" font-size="13" fill="#7dd3fc" font-weight="bold">Input X (n x d)</text>

  <!-- Arrow down -->
  <line x1="240" y1="46" x2="240" y2="70" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Add positional encoding -->
  <rect x="145" y="72" width="190" height="32" rx="6" fill="#3d2e0a" stroke="#d97706" stroke-width="1.5"/>
  <text x="240" y="93" text-anchor="middle" font-size="12" fill="#fbbf24">+ Positional Encoding</text>

  <!-- Arrow down -->
  <line x1="240" y1="104" x2="240" y2="128" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Block outline -->
  <rect x="40" y="130" width="400" height="510" rx="12" fill="none" stroke="#888" stroke-width="2" stroke-dasharray="8,4"/>
  <text x="60" y="155" font-size="12" fill="#888" font-style="italic">Transformer Block (repeat N times)</text>

  <!-- Layer Norm 1 -->
  <rect x="140" y="168" width="200" height="36" rx="6" fill="#2d1a47" stroke="#9333ea" stroke-width="1.5"/>
  <text x="240" y="191" text-anchor="middle" font-size="12" fill="#c084fc" font-weight="bold">Layer Norm</text>

  <!-- Arrow -->
  <line x1="240" y1="204" x2="240" y2="228" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Multi-Head Attention -->
  <rect x="110" y="230" width="260" height="44" rx="6" fill="#172554" stroke="#2563eb" stroke-width="2"/>
  <text x="240" y="257" text-anchor="middle" font-size="13" fill="#60a5fa" font-weight="bold">Multi-Head Self-Attention</text>

  <!-- Arrow -->
  <line x1="240" y1="274" x2="240" y2="300" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Residual connection 1 -->
  <line x1="100" y1="165" x2="100" y2="310" stroke="#f97316" stroke-width="2" stroke-dasharray="5,3"/>
  <line x1="100" y1="310" x2="200" y2="310" stroke="#f97316" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrowhead)"/>
  <text x="82" y="240" font-size="10" fill="#f97316" transform="rotate(-90, 82, 240)" text-anchor="middle">residual</text>

  <!-- Add node 1 -->
  <circle cx="218" cy="310" r="14" fill="#3d1f00" stroke="#f97316" stroke-width="2"/>
  <text x="218" y="315" text-anchor="middle" font-size="16" fill="#f97316" font-weight="bold">+</text>

  <!-- Arrow -->
  <line x1="240" y1="324" x2="240" y2="360" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Layer Norm 2 -->
  <rect x="140" y="362" width="200" height="36" rx="6" fill="#2d1a47" stroke="#9333ea" stroke-width="1.5"/>
  <text x="240" y="385" text-anchor="middle" font-size="12" fill="#c084fc" font-weight="bold">Layer Norm</text>

  <!-- Arrow -->
  <line x1="240" y1="398" x2="240" y2="422" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- FFN -->
  <rect x="110" y="424" width="260" height="44" rx="6" fill="#14332a" stroke="#16a34a" stroke-width="2"/>
  <text x="240" y="451" text-anchor="middle" font-size="13" fill="#4ade80" font-weight="bold">Feed-Forward Network</text>

  <!-- Arrow -->
  <line x1="240" y1="468" x2="240" y2="494" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Residual connection 2 -->
  <line x1="100" y1="358" x2="100" y2="504" stroke="#f97316" stroke-width="2" stroke-dasharray="5,3"/>
  <line x1="100" y1="504" x2="200" y2="504" stroke="#f97316" stroke-width="2" stroke-dasharray="5,3" marker-end="url(#arrowhead)"/>
  <text x="82" y="434" font-size="10" fill="#f97316" transform="rotate(-90, 82, 434)" text-anchor="middle">residual</text>

  <!-- Add node 2 -->
  <circle cx="218" cy="504" r="14" fill="#3d1f00" stroke="#f97316" stroke-width="2"/>
  <text x="218" y="509" text-anchor="middle" font-size="16" fill="#f97316" font-weight="bold">+</text>

  <!-- Arrow out -->
  <line x1="240" y1="518" x2="240" y2="560" stroke="#999" stroke-width="1.5" marker-end="url(#arrowhead)"/>

  <!-- Output -->
  <rect x="150" y="562" width="180" height="36" rx="6" fill="#0c2d4a" stroke="#0284c7" stroke-width="1.5"/>
  <text x="240" y="585" text-anchor="middle" font-size="13" fill="#7dd3fc" font-weight="bold">Output Y (n x d)</text>

  <!-- FFN detail box -->
  <rect x="310" y="570" width="155" height="55" rx="6" fill="#14332a" stroke="#86efac" stroke-width="1"/>
  <text x="387" y="588" text-anchor="middle" font-size="10" fill="#4ade80" font-weight="bold">FFN detail:</text>
  <text x="387" y="603" text-anchor="middle" font-size="10" fill="#4ade80">Linear(d, 4d)</text>
  <text x="387" y="616" text-anchor="middle" font-size="10" fill="#4ade80">GELU / Linear(4d, d)</text>
</svg>

### Residual Connections: Gradient Highways

The residual (skip) connection adds the input of a sub-layer to its output:

$$
\text{output} = x + f(x)
$$

where \(f\) is the sub-layer (attention or FFN). Why is this essential?

Consider the gradient flow during backpropagation. If \(y = x + f(x)\), then:

$$
\frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}
$$

The identity matrix \(I\) ensures that gradients can flow directly from later layers to earlier layers without passing through any nonlinearities or weight matrices. Even if \(\partial f / \partial x\) is small (vanishing gradient in the sub-layer), the gradient still flows through the identity path. This is the same principle that made ResNets work for deep convolutional networks --- it creates "gradient highways" that allow training networks with hundreds of layers.

Without residual connections, a 96-layer transformer would be essentially untrainable. With them, gradients from the loss reach the first layer with magnitude comparable to the last layer.

### Layer Normalization

**Layer normalization** normalizes the activations across the feature dimension for each token independently. Given a vector \(x \in \mathbb{R}^d\):

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta
$$

where \(\mu = \frac{1}{d}\sum_{i=1}^d x_i\) is the mean, \(\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2}\) is the standard deviation, \(\gamma\) and \(\beta\) are learned scale and shift parameters (vectors in \(\mathbb{R}^d\)), \(\odot\) is element-wise multiplication, and \(\epsilon\) is a small constant for numerical stability (typically \(10^{-5}\)).

Why is this needed? Without normalization, the scale of activations can drift as they pass through many layers --- growing exponentially (exploding activations) or shrinking toward zero. Layer norm resets the scale at each sub-layer, keeping activations in a numerically stable range. This is critical because transformers are deep (GPT-3 has 96 layers) and each layer's output feeds into the next.

Modern transformers typically use **Pre-Norm** ordering (normalize before the sub-layer) rather than the original Post-Norm (normalize after):

$$
y = x + \text{Attention}(\text{LayerNorm}(x))
$$

Pre-Norm is more stable to train because the residual path carries un-normalized values, preserving the gradient highway. This is what the diagram above shows.

### The Feed-Forward Network: Per-Token Memory

The feed-forward network (FFN) applies the same transformation independently to each token:

$$
\text{FFN}(x) = W_2 \, \sigma(W_1 x + b_1) + b_2
$$

where \(W_1 \in \mathbb{R}^{d \times d_{ff}}\), \(W_2 \in \mathbb{R}^{d_{ff} \times d}\), and typically \(d_{ff} = 4d\). The activation \(\sigma\) is GELU (Gaussian Error Linear Unit) in most modern transformers.

The FFN is where the transformer stores factual knowledge. Attention moves information between tokens (inter-token communication), but the FFN transforms each token independently (intra-token computation). Research has shown that individual neurons in the FFN activate for specific semantic concepts --- "Paris is the capital of France" is stored in specific weight patterns of the FFN layers.

The expansion to $4d$ and back means the FFN has a wide hidden layer, giving it capacity to represent complex nonlinear functions. Think of attention as the routing network (deciding what information goes where) and the FFN as the processing network (deciding what to do with that information).

---

## KV Cache: Why Autoregressive Generation is Memory-Bound {#kv-cache}

### The Autoregressive Generation Problem

Large language models generate text one token at a time. To generate token \(t+1\), the model runs a forward pass over the entire sequence \([x_1, x_2, \ldots, x_t]\), computes the probability distribution over the vocabulary for position \(t+1\), samples a token, appends it, and repeats.

Naively, generating the \((t+1)\)-th token requires recomputing attention over all \(t\) previous tokens. Generating a sequence of length \(n\) requires \(O(n)\) forward passes, each with \(O(n)\) attention, giving \(O(n^2)\) total work. But most of this is redundant: when generating token \(t+1\), the keys and values for tokens $1$ through \(t\) have not changed since the last step.

### The KV Cache

The solution is to **cache the key and value matrices** from all previous tokens. At step \(t+1\):

1. Compute Q, K, V only for the **new** token \(x_{t+1}\): \(q_{t+1} = x_{t+1} W_Q\), \(k_{t+1} = x_{t+1} W_K\), \(v_{t+1} = x_{t+1} W_V\)
2. **Append** \(k_{t+1}\) and \(v_{t+1}\) to the cached K and V matrices
3. Compute attention: \(q_{t+1}\) attends over all cached keys \([k_1, \ldots, k_{t+1}]\) to produce the output

This reduces the per-step computation from \(O(t \cdot d)\) for the full Q, K, V computation to \(O(d)\) for just the new token (the attention score computation is still \(O(t \cdot d_k)\) per head, but this is unavoidable).

### Memory Cost Derivation

The KV cache stores key and value vectors for every token, every head, and every layer. Let us compute the exact memory footprint.

**Parameters:**
- \(n\) = sequence length (number of cached tokens)
- \(L\) = number of transformer layers
- \(h\) = number of attention heads
- \(d_k = d / h\) = dimension per head

For each layer, each head stores:
- One key vector per token: \(d_k\) values
- One value vector per token: \(d_k\) values

Total cached values per layer:

$$
n \times h \times 2 \times d_k = n \times h \times 2 \times \frac{d}{h} = 2nd
$$

Total across all \(L\) layers:

$$
\text{KV cache size} = 2 \times n \times d \times L
$$

In bytes, with 16-bit (FP16/BF16) precision (2 bytes per value):

$$
\text{Bytes} = 2 \times n \times d \times L \times 2 = 4ndL
$$

**Concrete example: LLaMA-2 70B**
- \(d = 8192\), \(L = 80\), context length \(n = 4096\)
- KV cache: \(4 \times 4096 \times 8192 \times 80 = 10.7 \times 10^9\) bytes \(\approx 10\) GB

For a 128K context window (as in newer models): \(10 \times (128000 / 4096) \approx 312\) GB just for the KV cache of a single sequence. This is why long-context inference requires multiple GPUs even when the model weights themselves fit on one GPU. The KV cache grows linearly with sequence length, and it is the dominant memory consumer during generation.

This is the fundamental reason why autoregressive generation is **memory-bound rather than compute-bound**. The arithmetic is fast (one token's worth of matrix multiplications), but reading and writing the KV cache from GPU memory is slow. The ratio of compute to memory access --- the **arithmetic intensity** --- is very low during generation, which is why specialized hardware and memory optimization techniques (quantized KV cache, sliding window attention, paged attention) are active areas of research.

---

## DiT: Diffusion Transformers for Video

### From U-Nets to Transformers

Early diffusion models (Stable Diffusion 1.x, DALL-E 2) used U-Net architectures --- convolutional networks with skip connections --- as the denoising backbone. U-Nets process spatial data naturally but have limitations: their inductive biases toward local spatial structure make it hard to capture long-range dependencies, and they do not scale as cleanly as transformers.

**Diffusion Transformers (DiT)**, introduced by Peebles and Xie (2023), replace the U-Net entirely with a transformer. The noisy latent (image or video, encoded by a VAE) is split into patches, flattened into a sequence of tokens, and processed by a standard transformer with one critical modification: **conditioning on the diffusion timestep**.

### Adaptive Layer Norm (adaLN)

In a standard diffusion model, the denoising network must know what timestep \(t\) it is operating at --- the noise level determines what the network should do (at high \(t\), denoise aggressively; at low \(t\), refine details). In U-Nets, timestep conditioning is typically injected via additive or multiplicative conditioning.

DiT uses **adaptive layer normalization** (adaLN). Instead of using fixed learned \(\gamma\) and \(\beta\) in layer norm, these parameters are predicted from the timestep (and optionally the class label or text embedding):

$$
\gamma, \beta = \text{MLP}(t_{\text{emb}})
$$

where \(t_{\text{emb}}\) is an embedding of the diffusion timestep. The layer norm then becomes:

$$
\text{adaLN}(x, t) = \gamma(t) \odot \frac{x - \mu}{\sigma + \epsilon} + \beta(t)
$$

This is elegant: every layer norm operation in the transformer becomes conditioned on the timestep, giving the network fine-grained control over how it processes features at each noise level, without adding attention layers or cross-attention for conditioning.

### DiT for Video Generation

For video generation (as in Sora, CogVideoX, and other video models), the DiT processes a 3D grid of tokens: spatial patches across multiple frames. The self-attention operates over all spatiotemporal tokens, allowing the model to capture both spatial coherence within frames and temporal coherence across frames in a single unified mechanism.

The token count for video is large. A 16-frame video at 32x32 latent resolution with 2x2 patches gives \(16 \times 16 \times 16 = 4096\) tokens. Modern video models use factored attention (spatial attention within frames, then temporal attention across frames) or windowed attention to manage this cost, but the core architecture remains a transformer with adaLN conditioning.

---

## Python Demonstration

Let us implement self-attention from scratch and visualize what it actually computes.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    """
    Single-head self-attention from scratch.

    X: (n, d) input sequence
    W_Q, W_K: (d, d_k) query and key projections
    W_V: (d, d_v) value projection

    Returns: output (n, d_v), attention weights (n, n)
    """
    Q = X @ W_Q        # (n, d_k)
    K = X @ W_K        # (n, d_k)
    V = X @ W_V        # (n, d_v)

    d_k = Q.shape[-1]

    # Scaled dot-product attention
    scores = Q @ K.T / np.sqrt(d_k)   # (n, n)
    weights = softmax(scores, axis=-1) # (n, n), each row sums to 1
    output = weights @ V               # (n, d_v)

    return output, weights, scores

# --- Example: 6 tokens, 8 dimensions ---
n, d, d_k, d_v = 6, 8, 4, 4
tokens = ["The", "cat", "sat", "on", "the", "mat"]

# Random input embeddings (in practice, these come from an embedding layer)
X = np.random.randn(n, d)

# Random projection matrices (in practice, these are learned)
W_Q = np.random.randn(d, d_k) * 0.5
W_K = np.random.randn(d, d_k) * 0.5
W_V = np.random.randn(d, d_v) * 0.5

output, weights, raw_scores = self_attention(X, W_Q, W_K, W_V)

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Raw attention scores (before softmax)
im0 = axes[0].imshow(raw_scores, cmap="RdBu_r", aspect="equal")
axes[0].set_title(r"Raw Scores $QK^T / \sqrt{d_k}$", fontsize=13, fontweight="bold")
axes[0].set_xticks(range(n))
axes[0].set_yticks(range(n))
axes[0].set_xticklabels(tokens, fontsize=11)
axes[0].set_yticklabels(tokens, fontsize=11)
axes[0].set_xlabel(r"Key (attending to)")
axes[0].set_ylabel(r"Query (attending from)")
plt.colorbar(im0, ax=axes[0], fraction=0.046)

# 2. Attention weights (after softmax)
im1 = axes[1].imshow(weights, cmap="Blues", aspect="equal", vmin=0, vmax=1)
axes[1].set_title(r"Attention Weights (after softmax)", fontsize=13, fontweight="bold")
axes[1].set_xticks(range(n))
axes[1].set_yticks(range(n))
axes[1].set_xticklabels(tokens, fontsize=11)
axes[1].set_yticklabels(tokens, fontsize=11)
axes[1].set_xlabel(r"Key (attending to)")
axes[1].set_ylabel(r"Query (attending from)")
for i in range(n):
    for j in range(n):
        axes[1].text(j, i, f"{weights[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="white" if weights[i,j] > 0.4 else "black")
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# 3. Demonstrate the scaling effect
d_k_values = [4, 16, 64, 256, 1024]
score_stds_unscaled = []
score_stds_scaled = []

for dk in d_k_values:
    q = np.random.randn(1000, dk)
    k = np.random.randn(1000, dk)
    dots = np.sum(q * k, axis=1)  # dot products
    score_stds_unscaled.append(np.std(dots))
    score_stds_scaled.append(np.std(dots / np.sqrt(dk)))

axes[2].plot(d_k_values, score_stds_unscaled, "o-", color="#dc2626",
             linewidth=2, markersize=8, label=r"Unscaled: $\mathrm{std}(q \cdot k)$")
axes[2].plot(d_k_values, score_stds_scaled, "s-", color="#2563eb",
             linewidth=2, markersize=8, label=r"Scaled: $\mathrm{std}(q \cdot k / \sqrt{d_k})$")
axes[2].axhline(y=1.0, color="#16a34a", linestyle="--", linewidth=1.5, label=r"Target $\mathrm{std} = 1$")
axes[2].set_xlabel(r"$d_k$ (key dimension)", fontsize=12)
axes[2].set_ylabel(r"Standard Deviation of Scores", fontsize=12)
axes[2].set_title(r"Why $\sqrt{d_k}$ Scaling Matters", fontsize=13, fontweight="bold")
axes[2].legend(fontsize=11)
axes[2].set_xscale("log")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("attention_demo.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Input shape:  X = {X.shape}")
print(f"Output shape: Y = {output.shape}")
print(f"\nAttention weights (each row sums to 1):")
print(f"Row sums: {weights.sum(axis=1)}")
```

The leftmost plot shows the raw scores \(QK^T / \sqrt{d_k}\) --- these can be positive or negative, and their magnitude reflects how strongly each query-key pair aligns. The center plot shows these scores after softmax: now each row is a probability distribution summing to 1, and we can read off how much each token attends to each other token. The rightmost plot verifies our variance derivation: without \(\sqrt{d_k}\) scaling, the standard deviation of dot products grows as \(\sqrt{d_k}\), but after scaling it stays at 1 regardless of dimension.

### Multi-Head Attention Implementation

```python
def multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O):
    """
    Multi-head self-attention.

    X: (n, d)
    W_Qs, W_Ks, W_Vs: lists of h projection matrices, each (d, d_k)
    W_O: (d, d) output projection
    """
    head_outputs = []
    all_weights = []

    for W_Q, W_K, W_V in zip(W_Qs, W_Ks, W_Vs):
        out, w, _ = self_attention(X, W_Q, W_K, W_V)
        head_outputs.append(out)
        all_weights.append(w)

    # Concatenate heads: (n, h * d_k) = (n, d)
    concat = np.concatenate(head_outputs, axis=-1)

    # Output projection
    output = concat @ W_O

    return output, all_weights

# --- 4-head attention ---
h = 4
d_k_per_head = d // h  # 8 // 4 = 2

W_Qs = [np.random.randn(d, d_k_per_head) * 0.5 for _ in range(h)]
W_Ks = [np.random.randn(d, d_k_per_head) * 0.5 for _ in range(h)]
W_Vs = [np.random.randn(d, d_k_per_head) * 0.5 for _ in range(h)]
W_O = np.random.randn(d, d) * 0.5

mha_output, head_weights = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O)

# Visualize all 4 heads
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
for head_idx in range(h):
    im = axes[head_idx].imshow(head_weights[head_idx], cmap="Blues",
                                aspect="equal", vmin=0, vmax=1)
    axes[head_idx].set_title(f"Head {head_idx + 1}", fontsize=13, fontweight="bold")
    axes[head_idx].set_xticks(range(n))
    axes[head_idx].set_yticks(range(n))
    axes[head_idx].set_xticklabels(tokens, fontsize=10)
    axes[head_idx].set_yticklabels(tokens, fontsize=10)
    for i in range(n):
        for j in range(n):
            w = head_weights[head_idx][i, j]
            axes[head_idx].text(j, i, f"{w:.2f}", ha="center", va="center",
                                fontsize=8, color="white" if w > 0.4 else "black")

plt.suptitle(r"Multi-Head Attention: Each Head Learns Different Patterns",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("multihead_demo.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Multi-head output shape: {mha_output.shape}")
print(f"Number of heads: {h}")
print(f"Dimension per head: {d_k_per_head}")
```

With random weights, the four heads already show distinct attention patterns --- different rows concentrate their probability mass differently. In a trained model, these patterns become meaningful: one head might attend to the previous token (local context), another to syntactically related tokens, another to semantically similar tokens, and so on. The multi-head mechanism gives the model multiple independent "channels" to route information through the sequence.

### Verifying Permutation Equivariance

```python
# Verify that attention is permutation-equivariant
perm = np.array([3, 0, 5, 1, 4, 2])  # a random permutation
X_perm = X[perm]  # permute input rows

# Run attention on original and permuted input
Y_orig, _, _ = self_attention(X, W_Q, W_K, W_V)
Y_perm, _, _ = self_attention(X_perm, W_Q, W_K, W_V)

# Check: Y_perm should equal Y_orig[perm]
diff = np.max(np.abs(Y_perm - Y_orig[perm]))
print(f"Max difference between Y_perm and Y_orig[perm]: {diff:.2e}")
print(f"Permutation equivariance verified: {diff < 1e-12}")
```

This confirms the mathematical proof from the positional encoding section: permuting the input simply permutes the output. The attention mechanism is blind to order, which is why positional encoding is essential.

---

## Summary

Let us collect the key ideas:

1. **The problem:** Transform a sequence of token vectors \(X \in \mathbb{R}^{n \times d}\) so each token's representation incorporates context from the entire sequence.

2. **Attention** solves this by computing weighted averages: each token's output is a softmax-weighted combination of all value vectors, with weights determined by query-key dot products. The \(\sqrt{d_k}\) scaling keeps gradients healthy by maintaining unit variance in the attention scores.

3. **Multi-head attention** runs \(h\) parallel attention mechanisms with separate learned projections, allowing the model to capture multiple types of relationships simultaneously, at essentially the same computational cost as single-head attention.

4. **Positional encoding** is necessary because attention is permutation-equivariant. Sinusoidal encodings enable the model to detect relative positions via rotation matrices. RoPE injects position directly into the attention score computation.

5. **The transformer block** wraps attention and a feed-forward network with residual connections (gradient highways) and layer normalization (activation stability). Attention handles inter-token communication; the FFN handles per-token nonlinear transformation.

6. **KV caching** exploits the fact that keys and values for previous tokens do not change during autoregressive generation, reducing redundant computation but creating a memory bottleneck that scales as \(O(ndL)\).

7. **DiT** adapts the transformer for diffusion models by using adaptive layer normalization for timestep conditioning, enabling transformers to replace U-Nets as the denoising backbone for image and video generation.

Every component exists because removing it causes a specific, demonstrable failure. The \(\sqrt{d_k}\) scaling prevents gradient death. Multi-head attention prevents representational bottlenecks. Positional encoding prevents order-blindness. Residual connections prevent training collapse. Layer norm prevents activation drift. The feed-forward network provides per-token processing capacity. The KV cache prevents redundant computation.

The transformer is not a single clever idea. It is a carefully engineered stack of solutions to specific problems, each mathematically motivated and empirically validated. Understanding it at this level --- not just the equations, but the failure modes they prevent --- is what separates reading about transformers from actually understanding them.
