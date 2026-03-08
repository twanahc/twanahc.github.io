---
layout: post
title: "The Quantum Threat and Post-Quantum Cryptography: Preparing for the Next Era"
date: 2026-03-16
category: math
---

*This is Part 5 of a 5-part series on cryptology. [Part 1: Number Theory & Classical Ciphers](/2026/03/12/number-theory-classical-cryptography.html) | [Part 2: Symmetric Cryptography](/2026/03/13/symmetric-cryptography-aes-block-ciphers.html) | [Part 3: Asymmetric Cryptography](/2026/03/14/asymmetric-cryptography-rsa-elliptic-curves.html) | [Part 4: Cryptographic Protocols](/2026/03/15/cryptographic-protocols-tls-signal-national-security.html) | **Part 5: Post-Quantum Cryptography***

Harvest now, decrypt later.

That is not a theoretical scenario. It is current operational practice. The Swedish Military Intelligence and Security Service (MUST) identifies Russia, China, and Iran as state actors conducting aggressive intelligence operations against Sweden. The 2025 MUST report highlights competition between great powers for access to key technologies --- quantum computing explicitly included --- and notes that China is positioning itself as a global leader in technologies it considers crucial for the future. Russia purchases sanctioned Western technology through intelligence services. Chinese investments target Swedish dual-use high technology companies. Illegal technology acquisition and improper purchases of Swedish companies and innovation capabilities are ongoing.

Now connect these facts. Every classified Swedish military communication encrypted with RSA or ECDH has a countdown timer. State intelligence agencies are intercepting encrypted traffic today, storing it in data centers, and waiting. When a cryptographically relevant quantum computer is built, they will replay those stored ciphertexts through Shor's algorithm and read them in plaintext. The 2048-bit RSA key that protects a diplomatic cable today offers zero protection against a machine that factors integers in polynomial time.

The question is not whether this will happen. The question is when. And since MUST is responsible for protecting total defence communications using cryptographic methods, the question of *what mathematics survives quantum computing* is not academic. It is an operational requirement.

This article explains what a quantum computer can do, why it breaks the cryptography we built in Parts 3 and 4, and what mathematical structures might replace it.

---

## Table of Contents

1. [The Quantum Computing Basics You Actually Need](#the-quantum-computing-basics-you-actually-need)
2. [Shor's Algorithm --- Why RSA and ECC Break](#shors-algorithm--why-rsa-and-ecc-break)
3. [Grover's Algorithm --- The Symmetric Impact](#grovers-algorithm--the-symmetric-impact)
4. [The Quantum Timeline --- When Should We Worry?](#the-quantum-timeline--when-should-we-worry)
5. [Lattice-Based Cryptography --- The Leading Post-Quantum Approach](#lattice-based-cryptography--the-leading-post-quantum-approach)
6. [Other Post-Quantum Approaches](#other-post-quantum-approaches)
7. [The Migration --- From Theory to Practice](#the-migration--from-theory-to-practice)
8. [The Strategic Landscape](#the-strategic-landscape)

---

## The Quantum Computing Basics You Actually Need

Classical computers store information in bits. Each bit is either 0 or 1, with no in-between. A register of \\(n\\) bits is in exactly one of \\(2^n\\) possible states at any given time. A classical algorithm manipulates that single state through a sequence of logic gates.

Quantum computers store information in **qubits**. A qubit is a two-level quantum system --- physically, it might be a superconducting circuit, a trapped ion, or a photon polarization state. Mathematically, the state of a single qubit is a vector in a two-dimensional complex Hilbert space:

$$|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$$

where \\(\alpha, \beta \in \mathbb{C}\\) are **probability amplitudes** satisfying the normalization constraint:

$$|\alpha|^2 + |\beta|^2 = 1$$

The notation \\(|0\rangle\\) and \\(|1\rangle\\) refers to the **computational basis states**, which correspond to the standard basis vectors:

$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

A qubit in state \\(|\psi\rangle = \alpha|0\rangle + \beta|1\rangle\\) is in a **superposition** of both basis states simultaneously. This is not a statement about our ignorance of the qubit's true state --- the qubit genuinely has amplitude in both branches. The distinction matters because interference between these amplitudes is the mechanism that makes quantum algorithms work.

### Multiple Qubits and Tensor Products

Two qubits live in a four-dimensional Hilbert space, constructed via the **tensor product** \\(\mathcal{H}_2 \otimes \mathcal{H}_2\\). The computational basis is:

$$|00\rangle, \quad |01\rangle, \quad |10\rangle, \quad |11\rangle$$

A general two-qubit state is:

$$|\psi\rangle = \alpha_{00}|00\rangle + \alpha_{01}|01\rangle + \alpha_{10}|10\rangle + \alpha_{11}|11\rangle$$

with \\(\sum_{ij}|\alpha_{ij}|^2 = 1\\). For \\(n\\) qubits, the state space has dimension \\(2^n\\), and a general state requires \\(2^n\\) complex amplitudes to specify. This exponential scaling is the source of quantum computational power --- and also the reason simulating quantum computers classically is so expensive.

### Entanglement

Some multi-qubit states cannot be written as a tensor product of individual qubit states. Consider the **Bell state**:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

Try to factor this as \\((\alpha|0\rangle + \beta|1\rangle) \otimes (\gamma|0\rangle + \delta|1\rangle)\\). Expanding, you get \\(\alpha\gamma|00\rangle + \alpha\delta|01\rangle + \beta\gamma|10\rangle + \beta\delta|11\rangle\\). For this to equal \\(|\Phi^+\rangle\\), you need \\(\alpha\gamma = 1/\sqrt{2}\\), \\(\alpha\delta = 0\\), \\(\beta\gamma = 0\\), and \\(\beta\delta = 1/\sqrt{2}\\). The conditions \\(\alpha\delta = 0\\) and \\(\beta\gamma = 0\\) force either \\(\alpha = 0\\) or \\(\delta = 0\\), but both contradict the non-zero diagonal terms. The state is **entangled** --- it has correlations that no classical probability distribution can reproduce.

The four Bell states form a complete basis for two-qubit entangled states:

$$|\Phi^{\pm}\rangle = \frac{1}{\sqrt{2}}(|00\rangle \pm |11\rangle), \quad |\Psi^{\pm}\rangle = \frac{1}{\sqrt{2}}(|01\rangle \pm |10\rangle)$$

### Quantum Gates

Quantum computation proceeds by applying **unitary operators** to qubits. Unitarity (\\(U^\dagger U = I\\)) guarantees that the normalization condition is preserved and that the evolution is reversible.

The **Hadamard gate** creates superposition from a basis state:

$$H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$$

It maps \\(|0\rangle \to \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)\\) and \\(|1\rangle \to \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)\\). Notice the *minus sign* in the second case --- this relative phase is what enables interference.

The **CNOT gate** (controlled-NOT) operates on two qubits. It flips the target qubit if and only if the control qubit is \\(|1\rangle\\):

$$\text{CNOT} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{pmatrix}$$

To create the Bell state \\(|\Phi^+\rangle\\), apply Hadamard to the first qubit, then CNOT:

$$|00\rangle \xrightarrow{H \otimes I} \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle)|0\rangle \xrightarrow{\text{CNOT}} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

**Phase gates** add relative phases without changing probabilities. The \\(T\\) gate applies a phase of \\(e^{i\pi/4}\\) to \\(|1\rangle\\), and the \\(S\\) gate applies \\(e^{i\pi/2} = i\\). These are essential for universal quantum computation: any unitary can be approximated to arbitrary precision using \\(\{H, T, \text{CNOT}\}\\).

### Measurement and the Born Rule

When you **measure** a qubit in state \\(|\psi\rangle = \alpha|0\rangle + \beta|1\rangle\\), you get outcome 0 with probability \\(|\alpha|^2\\) and outcome 1 with probability \\(|\beta|^2\\). After measurement, the qubit collapses to the observed basis state. The superposition is destroyed.

This is the **Born rule**, and it constrains everything a quantum algorithm can do. You cannot read out the \\(2^n\\) amplitudes of an \\(n\\)-qubit state --- each measurement gives you only a single classical bit string, sampled according to the probability distribution \\(|\alpha_x|^2\\). The art of quantum algorithm design is arranging interference so that the probability concentrates on the correct answer.

### Why Quantum Computers Are Not Just Fast Classical Computers

A common misconception is that quantum computers try all \\(2^n\\) possibilities simultaneously and just "pick the right one." This is wrong. Measurement gives you a *random sample* from the amplitude distribution, not the maximum. If all \\(2^n\\) amplitudes are equal (as after applying Hadamard to all qubits), you just get a random bit string --- no better than a coin flip.

The power comes from **interference**. A quantum algorithm structures its computation so that amplitudes for wrong answers cancel (destructive interference) and amplitudes for correct answers reinforce (constructive interference). This is a subtle and constrained process. Only specific problems admit quantum speedups, and designing a quantum algorithm means finding the right interference pattern.

This is why we have only a handful of important quantum algorithms --- Shor's for period-finding, Grover's for search, quantum simulation for physics --- despite decades of research. The constraint of extracting information through measurement limits what quantum advantage is possible.

---

## Shor's Algorithm --- Why RSA and ECC Break

In Part 3, we built RSA on the assumption that factoring large integers is computationally hard. We built elliptic curve cryptography on the assumption that the discrete logarithm problem on elliptic curves is hard. Both assumptions are true for classical computers. Both are false for quantum computers.

Peter Shor showed in 1994 that a quantum computer can factor an \\(n\\)-bit integer in time \\(O(n^3)\\), compared to the best known classical algorithm (the general number field sieve) which runs in time \\(\exp(O(n^{1/3}(\log n)^{2/3}))\\). This is an exponential-to-polynomial collapse.

### The Reduction: Factoring to Period-Finding

Shor's algorithm does not factor integers directly. It reduces factoring to **period-finding**, which is the problem a quantum computer solves efficiently.

**Claim:** If you can efficiently find the **order** of a random element modulo \\(N\\) (the smallest positive \\(r\\) such that \\(a^r \equiv 1 \pmod{N}\\)), then you can efficiently factor \\(N\\).

**Proof:** Let \\(N = pq\\) be the product of two distinct odd primes (the RSA case). Pick a random \\(a\\) with \\(1 < a < N\\) and \\(\gcd(a, N) = 1\\). Let \\(r\\) be the order of \\(a\\) modulo \\(N\\), meaning \\(r\\) is the smallest positive integer with \\(a^r \equiv 1 \pmod{N}\\).

If \\(r\\) is even, we can write:

$$a^r - 1 \equiv 0 \pmod{N}$$

$$(a^{r/2} - 1)(a^{r/2} + 1) \equiv 0 \pmod{N}$$

This means \\(N\\) divides the product \\((a^{r/2} - 1)(a^{r/2} + 1)\\). If neither factor is divisible by \\(N\\) (which happens when \\(a^{r/2} \not\equiv \pm 1 \pmod{N}\\)), then \\(N\\) must share a non-trivial factor with each:

$$1 < \gcd(a^{r/2} - 1, N) < N$$

and this GCD gives us a factor of \\(N\\). Computing \\(\gcd\\) takes \\(O(n^2)\\) time using the Euclidean algorithm.

How often does this work? For a random \\(a\\), the probability that \\(r\\) is even and \\(a^{r/2} \not\equiv -1 \pmod{N}\\) is at least \\(1/2\\). So within a few random trials, you find a useful \\(a\\) with overwhelming probability.

The entire algorithm becomes: pick random \\(a\\), find the order \\(r\\), check if \\(r\\) is even and \\(a^{r/2} \not\equiv -1\\), compute the GCD. Everything is efficient *if* you can find \\(r\\) efficiently.

### Quantum Period-Finding

The classical difficulty of period-finding is that the period \\(r\\) can be as large as \\(N\\), so you cannot just compute \\(a^x \mod N\\) for all \\(x\\) from 1 to \\(N\\) --- that takes exponential time in the number of bits \\(n = \lceil \log_2 N \rceil\\).

The quantum approach uses the **Quantum Fourier Transform (QFT)** to extract the period from a superposition. The QFT over \\(\mathbb{Z}_{2^m}\\) maps:

$$\text{QFT}|x\rangle = \frac{1}{2^{m/2}} \sum_{k=0}^{2^m - 1} e^{2\pi i x k / 2^m} |k\rangle$$

This is the quantum analogue of the discrete Fourier transform. It can be implemented efficiently with \\(O(m^2)\\) gates using Hadamard gates and controlled phase rotations.

**Shor's algorithm proceeds as follows:**

**Step 1.** Choose \\(m\\) such that \\(N^2 \leq 2^m < 2N^2\\). Prepare two registers. Initialize the first register (\\(m\\) qubits) and second register (\\(n\\) qubits) to \\(|0\rangle\\).

**Step 2.** Apply Hadamard gates to every qubit in the first register, creating a uniform superposition:

$$\frac{1}{2^{m/2}} \sum_{x=0}^{2^m - 1} |x\rangle|0\rangle$$

**Step 3.** Compute \\(f(x) = a^x \mod N\\) into the second register (this is done with reversible modular exponentiation circuits):

$$\frac{1}{2^{m/2}} \sum_{x=0}^{2^m - 1} |x\rangle|a^x \bmod N\rangle$$

**Step 4.** Apply the QFT to the first register:

$$\frac{1}{2^m} \sum_{x=0}^{2^m - 1} \sum_{k=0}^{2^m - 1} e^{2\pi i x k / 2^m} |k\rangle|a^x \bmod N\rangle$$

**Step 5.** Measure the first register, obtaining some value \\(k\\).

Why does this give us the period? The function \\(f(x) = a^x \mod N\\) is periodic with period \\(r\\). After step 3, the amplitudes in the first register that correspond to the same value of \\(f\\) are spaced exactly \\(r\\) apart: they occur at positions \\(x_0, x_0 + r, x_0 + 2r, \ldots\\) The QFT converts this periodic spacing into a peak structure. Specifically, the measurement probability \\(|\alpha_k|^2\\) is concentrated at values of \\(k\\) that are close to multiples of \\(2^m / r\\).

Formally, the amplitude for measuring \\(k\\) includes a sum of the form:

$$\sum_{j=0}^{\lfloor (2^m - x_0)/r \rfloor} e^{2\pi i (x_0 + jr)k / 2^m}$$

This is a geometric series in \\(e^{2\pi i rk/2^m}\\). It has constructive interference when \\(rk/2^m\\) is close to an integer --- that is, when \\(k \approx j \cdot 2^m / r\\) for some integer \\(j\\). At these values, all terms add in phase.

**Step 6.** From the measured \\(k\\), compute the fraction \\(k/2^m\\), which is close to \\(j/r\\) for some integer \\(j\\). Use the **continued fractions algorithm** to find the best rational approximation with denominator less than \\(N\\). The denominator of this fraction (or a small multiple of it) is \\(r\\).

### Complexity Analysis

Each step runs efficiently. The Hadamard gates take \\(O(m)\\) operations. The modular exponentiation takes \\(O(m^3)\\) gates (using repeated squaring with reversible arithmetic). The QFT takes \\(O(m^2)\\) gates. The continued fractions algorithm takes \\(O(m)\\) classical operations. Since \\(m = O(\log N) = O(n)\\), the total gate count is \\(O(n^3)\\).

Compare this to the general number field sieve, which runs in time \\(\exp(O(n^{1/3}(\log n)^{2/3}))\\). For RSA-2048 (\\(n = 2048\\)), the classical algorithm requires roughly \\(2^{112}\\) operations. Shor's algorithm requires roughly \\(2048^3 \approx 8.6 \times 10^9\\) quantum gates --- a feasible number if you have a quantum computer with enough error-corrected qubits.

### Impact on RSA

To factor RSA-2048, Shor's algorithm requires approximately 4099 *logical* qubits. But logical qubits must be encoded in many *physical* qubits to enable quantum error correction. Current estimates suggest that each logical qubit requires roughly 1000 to 10000 physical qubits, depending on the error rate and the error-correcting code. So factoring RSA-2048 might require between 4 million and 40 million physical qubits.

For comparison, the largest quantum processors in 2025-2026 have around 1000-1200 physical qubits with error rates still far too high for running Shor's algorithm. The gap is enormous. But the gap is closing, and the mathematics is settled: when the hardware arrives, RSA is finished.

### Impact on Elliptic Curve Cryptography

Shor's algorithm also solves the **elliptic curve discrete logarithm problem** (ECDLP). Recall from Part 3 that ECDLP asks: given points \\(P\\) and \\(Q = kP\\) on an elliptic curve, find \\(k\\). This can be reduced to period-finding over the group \\(\mathbb{Z}_n\\) where \\(n\\) is the order of \\(P\\).

The quantum algorithm for ECDLP requires roughly \\(2n\\) logical qubits for a curve over an \\(n\\)-bit prime field. For P-256, this is about 512 logical qubits --- far fewer than RSA-2048. ECC's smaller key sizes, which are an advantage classically, become a liability quantumly: ECC is *easier* to break with a quantum computer than RSA.

### Impact on Diffie-Hellman

The discrete logarithm problem in \\(\mathbb{Z}_p^*\\) that underlies classical Diffie-Hellman is also solvable via Shor's algorithm. The approach is essentially identical: reduce DLP to period-finding over \\(\mathbb{Z}_{p-1}\\).

### What Survives: Symmetric Cryptography

Shor's algorithm exploits the algebraic structure of factoring and discrete logarithms --- the periodicity of modular exponentiation. Symmetric ciphers like AES have no such algebraic structure for Shor to exploit. The relevant quantum attack on symmetric ciphers is Grover's algorithm, which provides only a *quadratic* speedup. We analyze this next.

### Python: Simulating the Classical Part of Shor's Algorithm

The quantum parts of Shor's algorithm require a quantum computer (or an exponentially slow classical simulation). But the classical reduction --- from factoring to period-finding --- can be demonstrated directly.

```python
import numpy as np
from math import gcd
from fractions import Fraction

def find_order_classically(a: int, N: int) -> int:
    """Find the multiplicative order of a modulo N by brute force.

    This is the step that a quantum computer performs exponentially faster.
    Classically, this can take up to N steps in the worst case.
    """
    r = 1
    current = a % N
    while current != 1:
        current = (current * a) % N
        r += 1
        if r > N:
            raise ValueError(f"Order not found (a={a} may share factor with N={N})")
    return r

def shor_classical_reduction(N: int, max_attempts: int = 20) -> tuple[int, int]:
    """Factor N using order-finding (classical brute-force version).

    This demonstrates the reduction: factoring -> period-finding.
    A quantum computer replaces find_order_classically with QFT-based
    period detection, achieving exponential speedup.
    """
    if N % 2 == 0:
        return 2, N // 2

    for attempt in range(max_attempts):
        a = np.random.randint(2, N)
        d = gcd(a, N)
        if d > 1:
            # Lucky: a shares a factor with N
            return d, N // d

        # Find the order of a mod N (quantum computer does this fast)
        r = find_order_classically(a, N)

        if r % 2 != 0:
            continue  # Need even order, try again

        x = pow(a, r // 2, N)
        if x == N - 1:  # x ≡ -1 (mod N)
            continue  # This a doesn't work, try again

        p = gcd(x - 1, N)
        q = gcd(x + 1, N)

        if 1 < p < N:
            return p, N // p
        if 1 < q < N:
            return q, N // q

    raise ValueError(f"Failed to factor {N} in {max_attempts} attempts")

# Demonstrate on small semiprimes
test_cases = [15, 21, 33, 35, 77, 91, 143, 221, 323, 437]
print("Factoring semiprimes using Shor's classical reduction:")
print(f"{'N':>6} {'p':>5} {'q':>5}")
print("-" * 18)
for N in test_cases:
    p, q = shor_classical_reduction(N)
    assert p * q == N
    print(f"{N:>6} {min(p,q):>5} {max(p,q):>5}")

# Show the period-finding step explicitly for N=15, a=7
N, a = 15, 7
print(f"\nPeriod-finding for a={a}, N={N}:")
print(f"{'x':>3} {'a^x mod N':>10}")
for x in range(20):
    print(f"{x:>3} {pow(a, x, N):>10}")
print(f"\nPeriod r = {find_order_classically(a, N)}")
print(f"a^(r/2) mod N = {pow(a, find_order_classically(a,N)//2, N)}")
print(f"gcd(a^(r/2)-1, N) = {gcd(pow(a, find_order_classically(a,N)//2, N) - 1, N)}")
print(f"gcd(a^(r/2)+1, N) = {gcd(pow(a, find_order_classically(a,N)//2, N) + 1, N)}")
```

The output shows the periodicity clearly: \\(7^x \mod 15\\) cycles through \\(1, 7, 4, 13, 1, 7, 4, 13, \ldots\\) with period \\(r = 4\\). Then \\(7^2 = 49 \equiv 4 \pmod{15}\\), so \\(\gcd(4-1, 15) = 3\\) and \\(\gcd(4+1, 15) = 5\\). The factors are 3 and 5.

---

## Grover's Algorithm --- The Symmetric Impact

Shor's algorithm destroys public-key cryptography by exploiting algebraic structure. Grover's algorithm attacks *unstructured* search problems, which means it affects symmetric ciphers and hash functions.

### The Unstructured Search Problem

Given a function \\(f: \{0, 1\}^n \to \{0, 1\}\\) that outputs 1 for exactly one input \\(x^*\\) (the "marked item") and 0 for all others, find \\(x^*\\). Classically, this requires \\(O(2^n)\\) function evaluations in the worst case --- you just have to try inputs until you find the right one.

Grover's algorithm solves this with \\(O(\sqrt{2^n}) = O(2^{n/2})\\) evaluations. This is provably optimal for quantum computers --- no quantum algorithm can do better for unstructured search.

### The Algorithm

Grover's algorithm uses two operators applied repeatedly:

**The oracle** \\(O_f\\) flips the amplitude of the marked item:

$$O_f |x\rangle = (-1)^{f(x)}|x\rangle$$

This maps \\(|x^*\rangle \to -|x^*\rangle\\) and leaves all other basis states unchanged.

**The diffusion operator** \\(D\\) reflects the state about the uniform superposition \\(|s\rangle = \frac{1}{\sqrt{N}}\sum_x |x\rangle\\):

$$D = 2|s\rangle\langle s| - I$$

This operator amplifies states that differ from the average amplitude. After the oracle marks \\(|x^*\rangle\\) with a negative amplitude, the diffusion operator boosts it above the mean.

### Why It Achieves \\(O(\sqrt{N})\\)

The beautiful insight is that the entire algorithm lives in a **two-dimensional subspace** spanned by:

$$|x^*\rangle \quad \text{(the marked state)}$$
$$|x^{\perp}\rangle = \frac{1}{\sqrt{N-1}} \sum_{x \neq x^*} |x\rangle \quad \text{(uniform superposition of unmarked states)}$$

The initial state \\(|s\rangle\\) can be decomposed as:

$$|s\rangle = \frac{1}{\sqrt{N}}|x^*\rangle + \sqrt{\frac{N-1}{N}}|x^{\perp}\rangle = \sin\theta |x^*\rangle + \cos\theta |x^{\perp}\rangle$$

where \\(\sin\theta = 1/\sqrt{N}\\), so \\(\theta \approx 1/\sqrt{N}\\) for large \\(N\\).

Each Grover iteration (oracle followed by diffusion) is a **rotation by \\(2\theta\\)** in this 2D plane. After \\(k\\) iterations, the state has been rotated to angle \\((2k+1)\theta\\) from the \\(|x^{\perp}\rangle\\) axis. To maximize the probability of measuring \\(|x^*\rangle\\), we want \\((2k+1)\theta \approx \pi/2\\), which gives:

$$k \approx \frac{\pi}{4\theta} \approx \frac{\pi}{4}\sqrt{N}$$

After \\(O(\sqrt{N})\\) iterations, the amplitude is concentrated on \\(|x^*\rangle\\), and measurement yields the answer with high probability.

### Impact on Symmetric Cryptography

Searching for an AES key is an unstructured search problem: given a known plaintext-ciphertext pair \\((P, C)\\), find key \\(K\\) such that \\(\text{AES}_K(P) = C\\). Grover's algorithm means:

- **AES-128:** Security drops from \\(2^{128}\\) to \\(2^{64}\\) --- this is *breakable*
- **AES-192:** Security drops from \\(2^{192}\\) to \\(2^{96}\\) --- still secure
- **AES-256:** Security drops from \\(2^{256}\\) to \\(2^{128}\\) --- comfortable margin

The practical implication: **double your symmetric key sizes** in a post-quantum world. AES-256 remains secure. AES-128 does not.

### Impact on Hash Functions

For **preimage resistance** (given \\(h\\), find \\(m\\) with \\(H(m) = h\\)), Grover gives a quadratic speedup from \\(O(2^n)\\) to \\(O(2^{n/2})\\). SHA-256 preimage resistance drops from \\(2^{256}\\) to \\(2^{128}\\) --- still secure.

For **collision resistance**, the classical birthday attack already finds collisions in \\(O(2^{n/2})\\). The best known quantum collision-finding algorithm (BHT algorithm) achieves \\(O(2^{n/3})\\) using quantum walks, which is only a modest improvement over the classical birthday bound. SHA-256 collision resistance drops from \\(2^{128}\\) to roughly \\(2^{85}\\) --- still a comfortable margin.

The takeaway: Grover's algorithm is a nuisance for symmetric cryptography, not a catastrophe. The fix is straightforward (use larger keys). This is fundamentally different from Shor's devastating impact on public-key cryptography, where there is no simple parameter increase that restores security.

---

## The Quantum Timeline --- When Should We Worry?

### Current State (2025-2026)

The largest quantum processors have around 1000-1200 physical qubits (IBM's Condor, Google's Willow-class devices). Error rates per gate are around \\(10^{-3}\\) for two-qubit gates --- roughly one error per 1000 operations. These machines can run circuits with at most a few hundred gates before noise overwhelms the signal.

For context, Shor's algorithm on RSA-2048 requires billions of gates on thousands of logical qubits. The gap between current capability and what is needed is enormous.

### Error Correction Overhead

Quantum error correction encodes each logical qubit into many physical qubits using codes like the surface code. The overhead depends on the physical error rate \\(p\\) and the desired logical error rate \\(p_L\\):

$$\text{Physical qubits per logical qubit} \approx O\left(\log(1/p_L) / \log(p_{\text{threshold}}/p)\right)^2$$

For surface codes with current error rates, this works out to roughly 1000-10000 physical qubits per logical qubit. To run Shor's algorithm on RSA-2048 (~4099 logical qubits), you might need 4 to 40 million physical qubits.

### Estimates for Cryptographically Relevant Quantum Computers

Predictions vary widely:

- **Optimistic:** 2030-2035 (assumes rapid progress in qubit quality and count)
- **Moderate:** 2035-2045 (assumes steady but not revolutionary progress)
- **Pessimistic:** 2050+ (assumes fundamental engineering barriers persist)

The honest answer is that nobody knows. But the honest answer is also *irrelevant to the policy decision*.

### Why Migration Is Urgent Now

The "harvest now, decrypt later" threat inverts the timeline question. The relevant time horizon is not "when will quantum computers exist?" but rather:

$$T_{\text{risk}} = T_{\text{shelf life}} + T_{\text{migration}}$$

where \\(T_{\text{shelf life}}\\) is how long your data must remain secret, and \\(T_{\text{migration}}\\) is how long it takes to deploy post-quantum cryptography across your infrastructure.

For classified military communications, \\(T_{\text{shelf life}}\\) might be 25-50 years. If \\(T_{\text{migration}}\\) is 5-10 years (a realistic estimate for large government systems), then migration needed to start *years ago* for data with 30+ year secrecy requirements.

### NIST's Timeline

NIST has set concrete deprecation dates:
- **2030:** RSA-2048 and 112-bit security classical algorithms deprecated
- **2035:** RSA-2048 disallowed entirely
- **Post-quantum standards finalized:** FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA), all published in 2024

These are not suggestions. For any organization that follows NIST guidelines (which includes most NATO-aligned defence establishments), migration is a compliance requirement with a deadline.

---

## Lattice-Based Cryptography --- The Leading Post-Quantum Approach

We need cryptographic systems whose security does not rely on factoring or discrete logarithms. The leading candidate is **lattice-based cryptography**, which relies on the hardness of geometric problems in high-dimensional spaces. No quantum algorithm provides more than modest improvements for these problems.

### Lattices in \\(\mathbb{R}^n\\)

A **lattice** in \\(\mathbb{R}^n\\) is the set of all integer linear combinations of a set of linearly independent vectors \\(\mathbf{b}_1, \ldots, \mathbf{b}_n \in \mathbb{R}^n\\):

$$\mathcal{L}(\mathbf{B}) = \left\{ \sum_{i=1}^n z_i \mathbf{b}_i : z_i \in \mathbb{Z} \right\}$$

The matrix \\(\mathbf{B} = [\mathbf{b}_1 | \cdots | \mathbf{b}_n]\\) is called a **basis** for the lattice. A lattice has infinitely many bases --- any unimodular transformation (multiplication by an integer matrix with determinant \\(\pm 1\\)) yields a new basis for the same lattice.

In two dimensions, a lattice is just a grid of points. In high dimensions, lattice geometry becomes wild: the shortest vector can be exponentially hard to find, and the number of lattice points within a given distance grows in unpredictable ways.

<svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'Georgia', serif; font-size: 13px; }
    .title { font-size: 16px; font-weight: bold; }
    .label { font-size: 12px; fill: #333; }
    .small { font-size: 11px; }
  </style>
  <rect width="500" height="400" fill="#fafafa" stroke="#ccc"/>
  <text x="250" y="25" text-anchor="middle" class="title">2D Lattice: SVP and CVP Problems</text>

  <!-- Grid of lattice points using basis b1=(3,1), b2=(1,2.5) -->
  <!-- Origin at (250, 220) with scale 40 -->
  <!-- Lattice points: i*b1 + j*b2 for various i,j -->

  <!-- Basis vectors from origin -->
  <line x1="250" y1="220" x2="370" y2="180" stroke="#2563eb" stroke-width="2.5" marker-end="url(#arrowBlue)"/>
  <line x1="250" y1="220" x2="290" y2="120" stroke="#dc2626" stroke-width="2.5" marker-end="url(#arrowRed)"/>

  <defs>
    <marker id="arrowBlue" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#2563eb"/>
    </marker>
    <marker id="arrowRed" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#dc2626"/>
    </marker>
  </defs>

  <!-- Lattice points (shown as dots) -->
  <!-- Row -2 to 3, col -2 to 3 of i*b1 + j*b2 -->
  <!-- b1 = (3,1)*40 = (120,-40), b2 = (1,2.5)*40 = (40,-100) -->

  <!-- i=-1, j=-1 --> <circle cx="90" cy="360" r="4" fill="#333"/>
  <!-- i=0, j=-1 --> <circle cx="210" cy="320" r="4" fill="#333"/>
  <!-- i=1, j=-1 --> <circle cx="330" cy="280" r="4" fill="#333"/>
  <!-- i=2, j=-1 --> <circle cx="450" cy="240" r="4" fill="#333"/>
  <!-- i=-2, j=0 --> <circle cx="10" cy="300" r="4" fill="#333"/>
  <!-- i=-1, j=0 --> <circle cx="130" cy="260" r="4" fill="#333"/>
  <!-- i=0, j=0 --> <circle cx="250" cy="220" r="5" fill="#000"/>
  <!-- i=1, j=0 --> <circle cx="370" cy="180" r="4" fill="#333"/>
  <!-- i=2, j=0 --> <circle cx="490" cy="140" r="4" fill="#333"/>
  <!-- i=-2, j=1 --> <circle cx="50" cy="200" r="4" fill="#333"/>
  <!-- i=-1, j=1 --> <circle cx="170" cy="160" r="4" fill="#333"/>
  <!-- i=0, j=1 --> <circle cx="290" cy="120" r="4" fill="#333"/>
  <!-- i=1, j=1 --> <circle cx="410" cy="80" r="4" fill="#333"/>
  <!-- i=-2, j=2 --> <circle cx="90" cy="100" r="4" fill="#333"/>
  <!-- i=-1, j=2 --> <circle cx="210" cy="60" r="4" fill="#333"/>
  <!-- i=0, j=2 --> <circle cx="330" cy="20" r="4" fill="#333"/>
  <!-- i=1, j=2 --> <circle cx="450" cy="-20" r="4" fill="#333"/>
  <!-- i=-1, j=-2 --> <circle cx="50" cy="460" r="4" fill="#333"/>
  <!-- i=0, j=-2 --> <circle cx="170" cy="420" r="4" fill="#333"/>

  <!-- Shortest vector (SVP solution): b2 - b1 = (-2, 1.5) -> from origin -->
  <line x1="250" y1="220" x2="170" y2="160" stroke="#16a34a" stroke-width="3" stroke-dasharray="6,3" marker-end="url(#arrowGreen)"/>
  <defs>
    <marker id="arrowGreen" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#16a34a"/>
    </marker>
  </defs>

  <!-- CVP target point -->
  <circle cx="310" cy="175" r="6" fill="none" stroke="#9333ea" stroke-width="2"/>
  <circle cx="310" cy="175" r="2" fill="#9333ea"/>
  <line x1="310" y1="175" x2="290" y2="120" stroke="#9333ea" stroke-width="1.5" stroke-dasharray="4,3"/>
  <line x1="310" y1="175" x2="370" y2="180" stroke="#9333ea" stroke-width="1.5" stroke-dasharray="4,3"/>

  <!-- Labels -->
  <text x="320" y="195" fill="#2563eb" class="label" font-weight="bold">b₁ = (3, 1)</text>
  <text x="295" y="105" fill="#dc2626" class="label" font-weight="bold">b₂ = (1, 2.5)</text>
  <text x="90" y="155" fill="#16a34a" class="label" font-weight="bold">SVP solution</text>
  <text x="315" y="170" fill="#9333ea" class="label" font-weight="bold">CVP target t</text>
  <text x="255" y="237" class="label">Origin</text>

  <!-- Legend -->
  <rect x="15" y="345" width="220" height="50" fill="white" stroke="#ccc" rx="4"/>
  <text x="25" y="363" class="small" fill="#16a34a">● SVP: Find shortest non-zero lattice vector</text>
  <text x="25" y="383" class="small" fill="#9333ea">● CVP: Find lattice point closest to target t</text>
</svg>

### Hard Lattice Problems

The security of lattice-based cryptography rests on the (believed) hardness of several related problems.

**Shortest Vector Problem (SVP):** Given a lattice basis \\(\mathbf{B}\\), find the shortest nonzero lattice vector (under the Euclidean norm). In 2D, this is easy --- you can see the shortest vector. In dimension \\(n \geq 300\\), the best known algorithms (classical or quantum) require time \\(2^{\Omega(n)}\\).

**Closest Vector Problem (CVP):** Given a lattice basis \\(\mathbf{B}\\) and a target vector \\(\mathbf{t} \notin \mathcal{L}\\), find the lattice point closest to \\(\mathbf{t}\\). CVP is at least as hard as SVP.

**Learning With Errors (LWE):** This is the problem that actually underlies modern lattice-based cryptosystems. It was introduced by Oded Regev in 2005 with a proof that solving LWE is at least as hard as solving worst-case lattice problems.

The LWE problem is: given a matrix \\(\mathbf{A} \in \mathbb{Z}_q^{m \times n}\\) and a vector:

$$\mathbf{b} = \mathbf{A}\mathbf{s} + \mathbf{e} \pmod{q}$$

where \\(\mathbf{s} \in \mathbb{Z}_q^n\\) is a secret vector and \\(\mathbf{e} \in \mathbb{Z}_q^m\\) is a "small" error vector (each entry drawn from a discrete Gaussian distribution with small standard deviation), find \\(\mathbf{s}\\).

Without the error term, this is just a system of linear equations --- solvable in polynomial time by Gaussian elimination. The noise \\(\mathbf{e}\\) is what makes it hard. The noise is small enough that decryption works (with the secret key, you can remove it), but large enough that an eavesdropper cannot distinguish \\(\mathbf{b}\\) from a uniformly random vector.

**Why is LWE hard?** The error \\(\mathbf{e}\\) prevents direct linear algebra. Lattice reduction algorithms like LLL and BKZ can attempt to solve the underlying shortest vector problem, but their running time grows exponentially with the dimension \\(n\\). No quantum algorithm provides more than polynomial improvements over the best classical lattice algorithms.

**Ring-LWE** and **Module-LWE** are structured variants that replace \\(\mathbb{Z}_q^n\\) with polynomial rings \\(\mathbb{Z}_q[x]/(x^n + 1)\\). This structure enables:
- Smaller keys (polynomial multiplication replaces matrix-vector multiplication)
- Faster operations (using Number Theoretic Transform, the modular analogue of FFT)
- No known reduction in security (the structure does not help known attacks)

Module-LWE, used in ML-KEM, provides a middle ground: it works over *vectors* of ring elements, combining the efficiency of Ring-LWE with a more conservative security assumption.

### ML-KEM (formerly CRYSTALS-Kyber): NIST's Chosen Key Encapsulation

ML-KEM is the post-quantum key encapsulation mechanism standardized by NIST in FIPS 203. It replaces RSA and ECDH for key exchange. The core construction is an IND-CCA2 secure KEM based on Module-LWE.

**Key Generation:**
1. Sample a random matrix \\(\mathbf{A} \in R_q^{k \times k}\\), where \\(R_q = \mathbb{Z}_q[x]/(x^{256}+1)\\) and \\(k \in \{2, 3, 4\}\\) determines the security level
2. Sample secret vector \\(\mathbf{s} \in R_q^k\\) and error vector \\(\mathbf{e} \in R_q^k\\) from centered binomial distribution
3. Compute public key \\(\mathbf{t} = \mathbf{A}\mathbf{s} + \mathbf{e}\\)
4. Secret key is \\(\mathbf{s}\\), public key is \\((\mathbf{A}, \mathbf{t})\\)

**Encapsulation (to send a shared secret to the public key holder):**
1. Sample random vectors \\(\mathbf{r}, \mathbf{e}_1, e_2\\) from noise distribution
2. Compute \\(\mathbf{u} = \mathbf{A}^T \mathbf{r} + \mathbf{e}_1\\) and \\(v = \mathbf{t}^T \mathbf{r} + e_2 + \lceil q/2 \rfloor \cdot m\\), where \\(m\\) is the encoded message
3. Ciphertext is \\((\mathbf{u}, v)\\)

**Decapsulation (using the secret key):**
1. Compute \\(v - \mathbf{s}^T \mathbf{u} = \mathbf{t}^T\mathbf{r} + e_2 + \lceil q/2 \rfloor \cdot m - \mathbf{s}^T(\mathbf{A}^T\mathbf{r} + \mathbf{e}_1)\\)
2. Since \\(\mathbf{t} = \mathbf{A}\mathbf{s} + \mathbf{e}\\), this simplifies to \\(\mathbf{e}^T\mathbf{r} - \mathbf{s}^T\mathbf{e}_1 + e_2 + \lceil q/2 \rfloor \cdot m\\)
3. The noise terms \\(\mathbf{e}^T\mathbf{r} - \mathbf{s}^T\mathbf{e}_1 + e_2\\) are small (by design), so rounding recovers \\(m\\)

The parameter sets define the security level:

| Parameter Set | \\(k\\) | Security Level | Public Key Size | Ciphertext Size |
|---|---|---|---|---|
| ML-KEM-512 | 2 | NIST Level 1 (~AES-128) | 800 bytes | 768 bytes |
| ML-KEM-768 | 3 | NIST Level 3 (~AES-192) | 1184 bytes | 1088 bytes |
| ML-KEM-1024 | 4 | NIST Level 5 (~AES-256) | 1568 bytes | 1568 bytes |

Compare these to classical key sizes:

<svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: 'Georgia', serif; font-size: 12px; }
    .title { font-size: 15px; font-weight: bold; }
    .bar-label { font-size: 11px; fill: white; font-weight: bold; }
    .size-label { font-size: 11px; fill: #333; }
    .cat-label { font-size: 12px; fill: #333; font-weight: bold; }
  </style>
  <rect width="600" height="320" fill="#fafafa" stroke="#ccc"/>
  <text x="300" y="25" text-anchor="middle" class="title">Public Key Sizes Comparison (bytes, log scale)</text>

  <!-- Y axis labels -->
  <text x="120" y="70" text-anchor="end" class="cat-label">RSA-2048</text>
  <text x="120" y="110" text-anchor="end" class="cat-label">RSA-4096</text>
  <text x="120" y="150" text-anchor="end" class="cat-label">ECC P-256</text>
  <text x="120" y="190" text-anchor="end" class="cat-label">ML-KEM-512</text>
  <text x="120" y="230" text-anchor="end" class="cat-label">ML-KEM-768</text>
  <text x="120" y="270" text-anchor="end" class="cat-label">ML-KEM-1024</text>

  <!-- Bars (log scale: max width 440px = log2(1568) ≈ 10.6 bits, 1px = 41.5 per log2 unit) -->
  <!-- RSA-2048: 256 bytes -> log2(256) = 8 -> 332px -->
  <rect x="130" y="55" width="332" height="22" fill="#2563eb" rx="3"/>
  <text x="140" y="70" class="bar-label">256 B</text>

  <!-- RSA-4096: 512 bytes -> log2(512) = 9 -> 373px -->
  <rect x="130" y="95" width="373" height="22" fill="#1d4ed8" rx="3"/>
  <text x="140" y="110" class="bar-label">512 B</text>

  <!-- ECC P-256: 32 bytes -> log2(32) = 5 -> 207px -->
  <rect x="130" y="135" width="207" height="22" fill="#16a34a" rx="3"/>
  <text x="140" y="150" class="bar-label">32 B (compressed)</text>

  <!-- ML-KEM-512: 800 bytes -> log2(800) ≈ 9.64 -> 400px -->
  <rect x="130" y="175" width="400" height="22" fill="#dc2626" rx="3"/>
  <text x="140" y="190" class="bar-label">800 B</text>

  <!-- ML-KEM-768: 1184 bytes -> log2(1184) ≈ 10.21 -> 424px -->
  <rect x="130" y="215" width="424" height="22" fill="#b91c1c" rx="3"/>
  <text x="140" y="230" class="bar-label">1184 B</text>

  <!-- ML-KEM-1024: 1568 bytes -> log2(1568) ≈ 10.62 -> 441px -->
  <rect x="130" y="255" width="441" height="22" fill="#991b1b" rx="3"/>
  <text x="140" y="270" class="bar-label">1568 B</text>

  <text x="300" y="305" text-anchor="middle" class="size-label">Note: Logarithmic scale. ML-KEM keys are ~5x larger than RSA-2048 but ~25x larger than ECC.</text>
</svg>

The key sizes are larger than ECC but manageable. For TLS handshakes, the extra kilobyte is negligible compared to the typical web page payload.

### ML-DSA (formerly CRYSTALS-Dilithium): NIST's Chosen Signature Scheme

ML-DSA provides digital signatures based on Module-LWE, standardized in FIPS 204. The construction uses the **Fiat-Shamir with Aborts** paradigm.

The basic idea: to sign a message \\(\mu\\), the signer:
1. Samples a random masking vector \\(\mathbf{y}\\)
2. Computes a commitment \\(\mathbf{w} = \mathbf{A}\mathbf{y}\\)
3. Hashes the commitment with the message to get a challenge \\(c = H(\mathbf{w}, \mu)\\)
4. Computes the response \\(\mathbf{z} = \mathbf{y} + c\mathbf{s}\\), where \\(\mathbf{s}\\) is the secret key

The critical detail is **rejection sampling**: the signer checks whether \\(\mathbf{z}\\) leaks information about \\(\mathbf{s}\\). If the norm of \\(\mathbf{z}\\) is too large, or if certain coefficients reveal the secret, the signer **aborts** and restarts with a new \\(\mathbf{y}\\). This rejection step ensures that the distribution of valid signatures is independent of the secret key, which is essential for zero-knowledge.

On average, the signer needs about 4-7 attempts before producing a valid signature. This makes signing somewhat slower than verification, but both operations are fast in absolute terms (microseconds on modern hardware).

ML-DSA signature sizes: 2420 bytes (ML-DSA-44), 3293 bytes (ML-DSA-65), and 4595 bytes (ML-DSA-87) for security levels 2, 3, and 5 respectively. These are larger than ECDSA signatures (64 bytes) but much smaller than RSA-4096 signatures (512 bytes) adjusted for comparable post-quantum security.

### Python: Demonstrating LWE Hardness

```python
import numpy as np

def generate_lwe_instance(n: int, m: int, q: int, sigma: float):
    """Generate an LWE instance: (A, b = As + e mod q).

    Args:
        n: dimension of secret vector (security parameter)
        m: number of equations (samples)
        q: modulus
        sigma: standard deviation of error distribution

    Returns:
        A: random matrix (m x n)
        b: noisy product (m,)
        s: secret vector (n,) -- in practice, this is unknown
        e: error vector (m,) -- in practice, this is unknown
    """
    A = np.random.randint(0, q, size=(m, n))
    s = np.random.randint(0, q, size=n)
    e = np.round(np.random.normal(0, sigma, size=m)).astype(int) % q
    b = (A @ s + e) % q
    return A, b, s, e

def try_solve_without_noise(A, b, q):
    """Try to recover s using least-squares (ignoring noise).

    Without noise, this would recover s exactly.
    With noise, the result is garbage.
    """
    # Use pseudoinverse (works perfectly without noise)
    A_pinv = np.linalg.pinv(A)
    s_approx = A_pinv @ b
    s_recovered = np.round(s_approx) % q
    return s_recovered.astype(int)

# Demonstrate: noise makes linear algebra fail
print("=== LWE Hardness Demonstration ===\n")

q = 97  # small prime modulus
n_values = [4, 8, 16, 32]

for n in n_values:
    m = 2 * n  # number of samples

    # Case 1: No noise (just linear system)
    A, _, s_true, _ = generate_lwe_instance(n, m, q, sigma=0.0)
    b_clean = (A @ s_true) % q
    s_recovered = try_solve_without_noise(A, b_clean, q)
    match_clean = np.all(s_recovered % q == s_true % q)

    # Case 2: With noise (LWE)
    sigma = 3.0
    A, b_noisy, s_true, e = generate_lwe_instance(n, m, q, sigma)
    s_recovered_noisy = try_solve_without_noise(A, b_noisy, q)
    match_noisy = np.all(s_recovered_noisy % q == s_true % q)

    print(f"n={n:>3}, m={m:>3}, q={q}")
    print(f"  Without noise: recovery {'SUCCESS' if match_clean else 'FAILED'}")
    print(f"  With noise (σ={sigma}): recovery {'SUCCESS' if match_noisy else 'FAILED'}")
    print(f"  Error norm: ||e|| = {np.linalg.norm(e):.1f}")
    print()

# Show why lattice problems get harder with dimension
print("=== SVP Hardness vs Dimension ===\n")
print("LLL algorithm: finds a vector within factor 2^(n/2) of shortest.")
print("As dimension grows, this approximation factor explodes:\n")
print(f"{'Dimension n':>12} {'Approx factor 2^(n/2)':>25} {'Bits of hardness':>20}")
print("-" * 60)
for n in [50, 100, 200, 300, 500, 768, 1024]:
    approx = 2 ** (n / 2)
    bits = n // 2
    print(f"{n:>12} {'2^' + str(bits):>25} {bits:>20}")
```

The output makes the point vividly: without noise, Gaussian elimination recovers the secret perfectly regardless of dimension. With noise, even for \\(n = 4\\), the linear algebra approach fails completely. The small perturbation \\(\mathbf{e}\\) destroys the structure that makes linear systems solvable.

### Python: Simple Lattice and SVP Difficulty

```python
import numpy as np
from itertools import product

def shortest_vector_brute_force(basis: np.ndarray, search_range: int = 5):
    """Find shortest non-zero lattice vector by brute force.

    Enumerates all lattice vectors with coefficients in [-search_range, search_range].
    This is exponential in the dimension -- O((2*search_range+1)^n).
    """
    n = basis.shape[0]
    best_norm = float('inf')
    best_vec = None

    # Generate all integer coefficient combinations
    ranges = [range(-search_range, search_range + 1)] * n
    for coeffs in product(*ranges):
        if all(c == 0 for c in coeffs):
            continue
        vec = sum(c * basis[i] for i, c in enumerate(coeffs))
        norm = np.linalg.norm(vec)
        if norm < best_norm:
            best_norm = norm
            best_vec = vec

    return best_vec, best_norm

def random_lattice_basis(n: int, max_entry: int = 10):
    """Generate a random n-dimensional lattice basis."""
    while True:
        B = np.random.randint(-max_entry, max_entry + 1, size=(n, n)).astype(float)
        if abs(np.linalg.det(B)) > 0.5:  # Non-degenerate
            return B

# Show how SVP difficulty grows with dimension
print("=== Brute-Force SVP: Effort vs Dimension ===\n")
print(f"{'Dim':>4} {'Search range':>13} {'Candidates':>12} {'Shortest norm':>14}")
print("-" * 47)

for n in [2, 3, 4, 5, 6]:
    basis = random_lattice_basis(n)
    search_range = 3
    num_candidates = (2 * search_range + 1) ** n - 1  # exclude zero
    vec, norm = shortest_vector_brute_force(basis, search_range)
    print(f"{n:>4} {search_range:>13} {num_candidates:>12,} {norm:>14.4f}")

print(f"\nFor n=256 (ML-KEM dimension): candidates ≈ 7^256 ≈ 10^216")
print("This is why brute-force SVP is utterly infeasible in cryptographic dimensions.")
print("Even the best algorithms (BKZ) require 2^Ω(n) time.")
```

---

## Other Post-Quantum Approaches

Lattice-based cryptography is the NIST frontrunner, but it is not the only candidate. Diversifying post-quantum approaches provides resilience against the possibility that lattice problems turn out to be easier than believed.

### Code-Based Cryptography

The **McEliece cryptosystem**, proposed in 1978 --- predating RSA --- is based on the hardness of decoding random linear codes.

A **linear code** \\(C\\) over \\(\mathbb{F}_2\\) is a \\(k\\)-dimensional subspace of \\(\mathbb{F}_2^n\\), specified by a generator matrix \\(\mathbf{G} \in \mathbb{F}_2^{k \times n}\\). Encoding maps a \\(k\\)-bit message \\(\mathbf{m}\\) to a codeword \\(\mathbf{c} = \mathbf{m}\mathbf{G}\\). Some codes (like binary Goppa codes) have efficient decoding algorithms that can correct up to \\(t\\) errors.

The McEliece system works as follows:
1. **Key generation:** Choose a binary Goppa code with efficient decoding. Scramble the generator matrix with a random invertible matrix \\(\mathbf{S}\\) and a permutation matrix \\(\mathbf{P}\\): \\(\hat{\mathbf{G}} = \mathbf{S}\mathbf{G}\mathbf{P}\\). The public key is \\(\hat{\mathbf{G}}\\). The secret key is \\((\mathbf{S}, \mathbf{G}, \mathbf{P})\\).
2. **Encryption:** To encrypt message \\(\mathbf{m}\\), compute \\(\mathbf{c} = \mathbf{m}\hat{\mathbf{G}} + \mathbf{e}\\), where \\(\mathbf{e}\\) is a random error vector of weight \\(t\\).
3. **Decryption:** Using the secret key, un-permute \\(\mathbf{c}\\), apply the efficient decoder to remove \\(\mathbf{e}\\), and un-scramble to recover \\(\mathbf{m}\\).

An attacker who sees \\(\hat{\mathbf{G}}\\) sees what appears to be a random linear code with no known efficient decoding algorithm. Decoding a random linear code is an NP-hard problem, and no quantum algorithm provides more than a Grover-style quadratic speedup.

The catch: McEliece public keys are enormous. For 128-bit post-quantum security, the public key is roughly **260 kilobytes** --- orders of magnitude larger than RSA or lattice-based keys. This makes McEliece impractical for many protocols but attractive for scenarios where key size is not a constraint (e.g., long-lived keys stored on disk).

Despite nearly 50 years of cryptanalysis, no fundamental attack on McEliece has been found. This longevity gives it a strong confidence advantage over newer lattice-based schemes.

### Hash-Based Signatures

**Hash-based signatures** derive their security entirely from the properties of the underlying hash function: preimage resistance, second preimage resistance, and collision resistance. Since hash functions are not broken by Shor's algorithm (only mildly weakened by Grover's), hash-based signatures are among the most conservative post-quantum candidates.

**One-time signatures (Lamport/Winternitz):** The simplest construction. To sign one bit, the signer generates two random values \\((sk_0, sk_1)\\) and publishes their hashes \\((pk_0, pk_1) = (H(sk_0), H(sk_1))\\). To sign bit \\(b\\), reveal \\(sk_b\\). The verifier checks \\(H(sk_b) = pk_b\\). This is provably secure if \\(H\\) is preimage-resistant, but each key pair can sign only one message.

**Merkle trees** aggregate many one-time signature key pairs into a single public key. A binary tree of depth \\(d\\) holds \\(2^d\\) one-time key pairs at its leaves. The root hash is the public key. To sign the \\(i\\)-th message, use the \\(i\\)-th leaf key pair, and include the authentication path (\\(d\\) sibling hashes) so the verifier can reconstruct the root.

**XMSS (Extended Merkle Signature Scheme)** is a **stateful** hash-based signature: the signer must track which leaf has been used and never reuse one. This statefulness is operationally dangerous --- if state is lost (crash, backup restore), key reuse can break security.

**SLH-DSA (formerly SPHINCS+)**, standardized in FIPS 205, is a **stateless** hash-based signature. It avoids the state management problem by using a pseudorandom selection of which one-time key to use, based on the message and a secret seed. This means the same key pair can sign many messages without tracking state, at the cost of larger signatures (up to ~50 KB for high security levels).

### Isogeny-Based Cryptography: A Cautionary Tale

**SIKE (Supersingular Isogeny Key Encapsulation)** was a NIST PQC finalist based on the difficulty of computing isogenies between supersingular elliptic curves. Its key advantage was extremely small key sizes --- comparable to classical ECC.

In July 2022, Wouter Castryck and Thomas Decru published an attack that **completely broke SIKE** in a single hour on a laptop.

The attack exploited auxiliary torsion point information that SIKE provided as part of its public key. By using the theory of **higher-dimensional isogenies** (specifically, translating the problem to genus-2 curves and using Richelot isogenies), Castryck and Decru could recover the secret isogeny from the public information. The mathematical machinery had existed for decades, but nobody had connected it to the SIKE construction until 2022.

The lessons are sobering:

1. **New mathematical structures need decades of cryptanalysis.** SIKE was based on a problem studied seriously for only about 10 years. The lattice problems underlying ML-KEM have been studied for 30+ years. The coding theory problems behind McEliece have been studied for nearly 50.

2. **Auxiliary information is dangerous.** SIKE's vulnerability came from the torsion point information it revealed --- data that seemed harmless but provided enough structure for a devastating attack.

3. **Confidence in cryptographic assumptions should be proportional to the duration and intensity of cryptanalysis.** This is why NIST selected multiple standards based on different assumptions, and why hybrid modes (combining classical + PQC) provide defense in depth.

---

## The Migration --- From Theory to Practice

### NIST PQC Standardization

The NIST Post-Quantum Cryptography standardization process began in 2016 and reached its first milestone in August 2024 with the publication of three standards:

- **FIPS 203: ML-KEM** (Module-Lattice Key Encapsulation Mechanism, formerly CRYSTALS-Kyber)
- **FIPS 204: ML-DSA** (Module-Lattice Digital Signature Algorithm, formerly CRYSTALS-Dilithium)
- **FIPS 205: SLH-DSA** (Stateless Hash-Based Digital Signature Algorithm, formerly SPHINCS+)

Additional standards are expected, including HQC (a code-based KEM) as an alternative to ML-KEM, providing cryptographic diversity.

### Hybrid Mode

The recommended deployment strategy during the transition period is **hybrid mode**: combine a classical algorithm with a post-quantum algorithm, so security holds if *either* is secure.

For key exchange, this means performing both X25519 (classical ECDH) and ML-KEM-768, then combining the shared secrets:

$$K_{\text{final}} = \text{KDF}(K_{\text{X25519}} \| K_{\text{ML-KEM}})$$

This protects against two failure modes:
- If the quantum computer arrives and X25519 breaks, ML-KEM provides security
- If a classical attack is found against ML-KEM (like SIKE was broken), X25519 provides security

### TLS 1.3 with Post-Quantum Cryptography

Google and Cloudflare have already deployed hybrid post-quantum key exchange in TLS 1.3, using X25519+ML-KEM-768. The combined key share is about 1.2 KB larger than a pure X25519 handshake --- negligible for most applications.

The deployment revealed practical challenges:
- **Middlebox interference:** Some network middleboxes (firewalls, proxies) choke on the larger ClientHello messages that contain PQC key shares. These devices assumed TLS messages would never exceed certain sizes.
- **Certificate chains:** PQC signatures are much larger than ECDSA signatures. A certificate chain with ML-DSA signatures can exceed 10 KB, causing fragmentation and performance issues.
- **Embedded systems:** IoT devices with limited RAM and bandwidth struggle with the larger keys and ciphertexts.

### Crypto-Agility

The SIKE break teaches us that any specific post-quantum algorithm might fail. **Crypto-agility** is the design principle that cryptographic algorithms should be swappable without redesigning the system.

In practice, this means:
- Abstract cryptographic operations behind interfaces (encrypt/decrypt, sign/verify, KDF)
- Negotiate algorithms dynamically (TLS already does this)
- Avoid hard-coding key sizes, OIDs, or algorithm-specific parameters
- Maintain an inventory of all cryptographic dependencies in your codebase

Systems designed with crypto-agility can swap from ML-KEM to HQC (or whatever comes next) with a configuration change, not a rewrite.

### Connection to MUST and Swedish Total Defence

MUST is responsible for protecting total defence communications using cryptographic methods. The quantum threat creates several urgent requirements:

**Military communications systems need PQC.** Every encrypted link in the Swedish armed forces that uses RSA, ECDH, or ECDSA key exchange is vulnerable to harvest-now-decrypt-later attacks. Migration to hybrid PQC should be underway for all new systems, with retrofit plans for existing ones.

**NATO interoperability requires coordinated PQC adoption.** Sweden's 2024 NATO membership means its cryptographic systems must interoperate with allies. NATO has its own PQC migration timeline. Swedish systems that lag behind risk being incompatible with allied secure communications.

**The Swedish intelligence reform should incorporate PQC from day one.** The MUST report describes reforms aimed at providing better and faster intelligence in response to rapid technological developments. Any new intelligence infrastructure built today that does not incorporate PQC is being built obsolete.

**Signal intelligence (FRA) must also adapt.** As adversaries adopt PQC, signals intelligence becomes harder. Understanding post-quantum cryptography is necessary not just for defence but for intelligence collection --- knowing what can and cannot be broken, and where implementation weaknesses might exist.

---

## The Strategic Landscape

### The Quantum Computing Race

The MUST report identifies the competition between great powers for access to key technologies, with quantum computing explicitly listed. The strategic implications are profound.

**China** has invested heavily in quantum computing and quantum communication. Chinese researchers have demonstrated quantum key distribution over satellite links and claimed quantum computational advantages (though these demonstrations have not yet approached cryptographic relevance). The MUST report's concern about Chinese technology acquisition includes quantum computing expertise and components.

**The United States** leads in superconducting qubit technology through companies like Google, IBM, and Microsoft, and through government programs funded by DARPA and the NSF. The CHIPS and Science Act includes provisions for quantum technology development.

**The race creates a security paradox:** the country that achieves a cryptographically relevant quantum computer first gains a temporary but devastating intelligence advantage --- the ability to decrypt years of stored intercepts. This creates a strong incentive to develop quantum computers *and* to hide progress from adversaries.

### Harvest Now, Decrypt Later --- The Concrete Threat

Consider the lifecycle of a classified Swedish military communication:

1. **Today (2026):** The message is encrypted with RSA-2048 or ECDH and transmitted. A foreign intelligence service intercepts and stores the ciphertext.
2. **2035-2045 (estimated):** A cryptographically relevant quantum computer is operational somewhere in the world. The stored ciphertext is decrypted using Shor's algorithm.
3. **The message's content** --- troop positions, diplomatic positions, intelligence assessments, technical capabilities --- is read in the clear.

If the message's content has a secrecy requirement of 30 years (standard for many classified materials), and the quantum computer arrives in 15 years, the message is compromised 15 years before its classification expires.

This is not speculative. The MUST report documents that Russia and China are actively conducting intelligence operations against Sweden. The storage capacity to hold encrypted intercepts is trivially cheap. The only missing piece is the quantum computer itself.

### The Advantage of Strong COMSEC Traditions

Sweden, through MUST and FRA, has a strong tradition of communications security. Countries with mature cryptographic institutions have a significant advantage in the PQC transition:

- They understand their cryptographic inventory (which systems use which algorithms)
- They have the technical expertise to evaluate post-quantum candidates
- They have established processes for cryptographic transitions
- They have relationships with academic cryptographers who can assess new attacks

The transition to PQC is not just a technology upgrade. It is a test of institutional cryptographic maturity.

### The Role of Mathematical Research

Every section of this article relied on deep mathematics: number theory for RSA, group theory for ECC, Hilbert spaces for quantum mechanics, lattice theory for post-quantum constructions, coding theory for McEliece. Cryptography is one of the few domains where abstract mathematical research has direct national security implications.

The countries that invest in mathematical research --- particularly in lattice algorithms, quantum information theory, and algebraic geometry --- will be best positioned to both build and resist the cryptographic tools of the next era.

### What the Next Decade Looks Like

**2026-2028:** Hybrid PQC deployment becomes widespread in commercial internet infrastructure. Government systems begin migration planning. NIST publishes additional PQC standards (HQC, additional signature schemes).

**2028-2030:** NIST deprecation of 112-bit classical security forces enterprises to upgrade. PQC becomes mandatory for new government contracts. Middlebox compatibility issues are resolved through firmware updates.

**2030-2035:** RSA-2048 is disallowed for new deployments. Quantum computers reach thousands of logical qubits but likely remain below the threshold for breaking RSA-2048. The harvest-now-decrypt-later window begins to close for newly encrypted data --- but everything sent before migration remains vulnerable.

**2035+:** Cryptographically relevant quantum computers may exist. All data encrypted before the PQC transition is at risk. The security of the post-quantum world depends entirely on decisions made today.

---

## Conclusion

This series has traced the arc of cryptography from Euclid's algorithm through RSA, elliptic curves, TLS, and the Signal protocol, arriving at the quantum threshold. The mathematical structure that made public-key cryptography possible --- the one-way functions built from factoring and discrete logarithms --- is the same structure that makes it vulnerable. Shor's algorithm does not break cryptography in general. It breaks a specific mathematical assumption. The response is to find new assumptions.

Lattice-based problems, coding theory problems, and hash function properties offer hardness that quantum computers cannot efficiently exploit (as far as we know, and after decades of trying). The NIST standardization of ML-KEM, ML-DSA, and SLH-DSA provides concrete, implementable replacements. The hybrid deployment strategy provides a safety net during the transition.

But mathematics alone does not provide security. The MUST report reminds us that security exists in a context of state adversaries, technology races, and institutional decisions. The mathematics tells us *what is possible*. The strategic environment tells us *what is necessary*. And the migration timeline tells us that the answer to "when should we start?" was several years ago.

Every day that passes with RSA-2048 protecting classified communications is a day that the harvest-now-decrypt-later stockpile grows. The quantum clock is ticking. The mathematics to survive it exists. The question is whether institutions will deploy it in time.

*End of the 5-part cryptology series.*
