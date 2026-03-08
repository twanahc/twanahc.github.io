---
layout: post
title: "Symmetric Cryptography: Block Ciphers, AES, and the Mathematics of Confusion"
date: 2026-03-13
category: math
---

*This is Part 2 of a 5-part series on cryptology. [Part 1: Number Theory & Classical Ciphers](/2026/03/12/number-theory-classical-cryptography.html) | **Part 2: Symmetric Cryptography** | [Part 3: Asymmetric Cryptography](/2026/03/14/asymmetric-cryptography-rsa-elliptic-curves.html) | [Part 4: Cryptographic Protocols](/2026/03/15/cryptographic-protocols-tls-signal-national-security.html) | [Part 5: Post-Quantum Cryptography](/2026/03/16/quantum-threat-post-quantum-cryptography.html)*

In January 2025, Sweden's Military Intelligence and Security Service (MUST) — responsible for communications security (COMSEC) across the total defence — published its annual threat assessment. The report is blunt. State-sponsored actors from Russia, China, and Iran are conducting sustained cyber operations aimed at collecting intelligence and achieving destructive effects against Swedish critical infrastructure. Sweden, as one of the world's most digitally connected countries, is particularly vulnerable: the energy supply, the electricity grid, mobile communications — all are targets. The report describes how the "gig economy" is being exploited by foreign intelligence services to carry out cyber attacks anonymously, laundering offensive operations through layers of contractors and freelancers.

Consider the scenario. An advanced persistent threat group — backed by a nation-state — compromises a node in the Swedish military communications network. They intercept encrypted traffic flowing between command centres. They have the ciphertext. They have sophisticated cryptanalytic capabilities and state-level compute budgets. The question is simple: does the encryption hold?

This question is what symmetric cryptography answers. Not in the abstract, not in theory, but in the concrete mathematical structures that make it computationally infeasible to recover plaintext from ciphertext without the key. In Part 1, we built the number-theoretic foundations — modular arithmetic, Euler's theorem, the discrete logarithm problem. Now we use those foundations to understand how modern symmetric ciphers actually work, why they resist attack, and what mathematical properties guarantee their strength.

---

## Table of Contents

1. [Shannon's Principles: Confusion and Diffusion](#shannons-principles-confusion-and-diffusion)
2. [Feistel Networks — The Architecture of Block Ciphers](#feistel-networks--the-architecture-of-block-ciphers)
3. [Substitution-Permutation Networks](#substitution-permutation-networks)
4. [AES: The Advanced Encryption Standard from First Principles](#aes-the-advanced-encryption-standard-from-first-principles)
5. [Modes of Operation](#modes-of-operation--turning-a-block-cipher-into-an-encryption-scheme)
6. [Stream Ciphers](#stream-ciphers)
7. [Measuring Cipher Strength](#measuring-cipher-strength)

---

## Shannon's Principles: Confusion and Diffusion

In 1949, Claude Shannon published *Communication Theory of Secrecy Systems*, the paper that transformed cryptography from an art into a science. Shannon identified two fundamental properties that any secure cipher must possess: **confusion** and **diffusion**.

**Confusion** means that the relationship between the ciphertext and the key should be as complex as possible. Each bit of ciphertext should depend on several parts of the key in a way that obscures the statistical relationship between them. If confusion is weak, an attacker can deduce information about the key by studying the ciphertext.

**Diffusion** means that the statistical structure of the plaintext should be dissipated across the ciphertext. Concretely, changing a single bit of plaintext should change approximately 50% of the ciphertext bits. If diffusion is weak, patterns in the plaintext leak through to the ciphertext — the attacker sees structure where there should be none.

Why are both necessary? Consider a cipher with good confusion but no diffusion. Each byte of ciphertext depends complexly on the key, but byte 1 of ciphertext depends only on byte 1 of plaintext. An attacker can break the cipher one byte at a time — a divide-and-conquer attack. The key space for each byte is tiny.

Now consider a cipher with good diffusion but no confusion. Changing one plaintext bit scrambles the entire ciphertext, but the relationship between key and ciphertext is simple — perhaps linear. The attacker sets up a system of linear equations and solves for the key directly.

You need both. Confusion defeats statistical attacks on the key. Diffusion defeats statistical attacks on the plaintext.

### The Avalanche Criterion

Shannon's intuition about diffusion was formalized decades later. The **avalanche criterion** states that for a cryptographic function \\(f\\), if a single input bit is flipped, each output bit should change with probability \\(\frac{1}{2}\\). Formally, let \\(x\\) and \\(x'\\) differ in exactly one bit. Then:

$$\Pr\left[\text{bit } j \text{ of } f(x) \neq \text{bit } j \text{ of } f(x')\right] = \frac{1}{2}$$

for every output bit \\(j\\).

The **strict avalanche criterion (SAC)** strengthens this. A function \\(f: \{0,1\}^n \to \{0,1\}^m\\) satisfies SAC if, whenever a single input bit \\(i\\) is complemented, each output bit \\(j\\) changes with probability exactly \\(\frac{1}{2}\\), averaged uniformly over all possible inputs. This means flipping input bit \\(i\\) causes every output bit to behave like a fair coin flip — maximum uncertainty.

SAC is the gold standard. It means the cipher provides no information about which input bit changed by observing the output change. AES satisfies SAC after just two rounds.

---

## Feistel Networks — The Architecture of Block Ciphers

A **block cipher** encrypts data in fixed-size blocks — typically 64 or 128 bits. The fundamental challenge in designing a block cipher is: how do you build a complex, invertible transformation from simpler components? Decryption must undo encryption, so the encryption function must be a bijection (one-to-one and onto). Designing bijections that are also cryptographically strong is hard.

Horst Feistel, working at IBM in the early 1970s, found an elegant solution. The **Feistel network** is an architecture that constructs an invertible transformation from *any* function — the function itself does not need to be invertible.

### The Feistel Structure

Take a plaintext block of \\(2n\\) bits. Split it into two halves: a left half \\(L_0\\) of \\(n\\) bits and a right half \\(R_0\\) of \\(n\\) bits. Let \\(F\\) be any function (called the **round function**) that takes an \\(n\\)-bit input and a round key \\(K_i\\), and produces an \\(n\\)-bit output. One round of the Feistel network computes:

$$L_{i+1} = R_i$$

$$R_{i+1} = L_i \oplus F(R_i, K_i)$$

where \\(\oplus\\) denotes the bitwise XOR operation. After \\(r\\) rounds, the ciphertext is \\((L_r, R_r)\\).

The beauty is in the decryption. To invert round \\(i+1\\), we need to recover \\(L_i\\) and \\(R_i\\) from \\(L_{i+1}\\) and \\(R_{i+1}\\).

<svg viewBox="0 0 500 520" xmlns="http://www.w3.org/2000/svg" style="max-width:460px; display:block; margin:2em auto;">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#ccc"/></marker>
  </defs>
  <rect width="500" height="520" rx="12" fill="#181818"/>
  <text x="250" y="32" text-anchor="middle" fill="#e0e0e0" font-size="16" font-weight="bold">Feistel Network — One Round</text>
  <!-- Input labels -->
  <text x="130" y="72" text-anchor="middle" fill="#7ec8e3" font-size="15">L_i</text>
  <text x="370" y="72" text-anchor="middle" fill="#f7c948" font-size="15">R_i</text>
  <!-- Input lines down -->
  <line x1="130" y1="80" x2="130" y2="340" stroke="#7ec8e3" stroke-width="2" marker-end="url(#arrow)"/>
  <line x1="370" y1="80" x2="370" y2="180" stroke="#f7c948" stroke-width="2"/>
  <!-- R_i splits: one goes to F, one goes down -->
  <line x1="370" y1="180" x2="370" y2="340" stroke="#f7c948" stroke-width="2"/>
  <!-- R_i branch to F box -->
  <line x1="370" y1="180" x2="290" y2="180" stroke="#f7c948" stroke-width="2"/>
  <line x1="290" y1="180" x2="290" y2="200" stroke="#f7c948" stroke-width="2" marker-end="url(#arrow)"/>
  <!-- F box -->
  <rect x="240" y="205" width="100" height="45" rx="8" fill="#2d2d2d" stroke="#e0e0e0" stroke-width="1.5"/>
  <text x="290" y="233" text-anchor="middle" fill="#e0e0e0" font-size="15" font-weight="bold">F</text>
  <!-- Key input to F -->
  <text x="400" y="228" fill="#f77" font-size="13">K_i</text>
  <line x1="395" y1="228" x2="342" y2="228" stroke="#f77" stroke-width="1.5" marker-end="url(#arrow)"/>
  <!-- F output goes left to XOR -->
  <line x1="290" y1="250" x2="290" y2="300" stroke="#ccc" stroke-width="2"/>
  <line x1="290" y1="300" x2="170" y2="300" stroke="#ccc" stroke-width="2" marker-end="url(#arrow)"/>
  <!-- XOR circle on L line -->
  <circle cx="130" cy="300" r="18" fill="none" stroke="#7ec8e3" stroke-width="2"/>
  <text x="130" y="306" text-anchor="middle" fill="#7ec8e3" font-size="18">⊕</text>
  <!-- Crossed wires to outputs -->
  <line x1="130" y1="340" x2="130" y2="380" stroke="#7ec8e3" stroke-width="2"/>
  <line x1="370" y1="340" x2="370" y2="380" stroke="#f7c948" stroke-width="2"/>
  <!-- Cross wires -->
  <line x1="130" y1="380" x2="370" y2="440" stroke="#7ec8e3" stroke-width="2"/>
  <line x1="370" y1="380" x2="130" y2="440" stroke="#f7c948" stroke-width="2"/>
  <!-- Output labels -->
  <text x="130" y="475" text-anchor="middle" fill="#f7c948" font-size="15">L_{i+1} = R_i</text>
  <text x="370" y="475" text-anchor="middle" fill="#7ec8e3" font-size="15">R_{i+1} = L_i ⊕ F(R_i,K_i)</text>
</svg>

### Proof: Feistel Networks Are Always Invertible

**Theorem.** The Feistel round is invertible for any function \\(F\\), regardless of whether \\(F\\) itself is invertible.

**Proof.** Given \\(L_{i+1}\\) and \\(R_{i+1}\\), we want to recover \\(L_i\\) and \\(R_i\\).

From the first equation: \\(L_{i+1} = R_i\\), so immediately:

$$R_i = L_{i+1}$$

Now we know \\(R_i\\), and we know the round key \\(K_i\\), so we can compute \\(F(R_i, K_i) = F(L_{i+1}, K_i)\\). From the second equation:

$$R_{i+1} = L_i \oplus F(R_i, K_i)$$

XOR both sides with \\(F(R_i, K_i)\\):

$$R_{i+1} \oplus F(R_i, K_i) = L_i \oplus F(R_i, K_i) \oplus F(R_i, K_i) = L_i$$

because \\(A \oplus A = 0\\) and \\(A \oplus 0 = A\\) for any bitstring \\(A\\). Therefore:

$$L_i = R_{i+1} \oplus F(L_{i+1}, K_i)$$

The decryption formulas are:

$$R_i = L_{i+1}$$

$$L_i = R_{i+1} \oplus F(L_{i+1}, K_i)$$

Notice: the decryption uses the *same function* \\(F\\) — not its inverse. This is why \\(F\\) need not be invertible. The XOR operation is self-inverse, and that is doing all the heavy lifting. To decrypt a full Feistel cipher with \\(r\\) rounds, apply these formulas from round \\(r\\) back to round 1, using the round keys in reverse order. ∎

### Key Schedule

The **key schedule** is the algorithm that derives round keys \\(K_1, K_2, \ldots, K_r\\) from the master key \\(K\\). A good key schedule ensures that the round keys appear independent — knowing one round key should not help predict another. A weak key schedule can catastrophically undermine an otherwise strong cipher.

### The Luby-Rackoff Theorem

How many Feistel rounds are enough? In 1988, Michael Luby and Charles Rackoff proved a remarkable theorem: if the round function \\(F\\) is a truly random function, then a **3-round Feistel network is a pseudorandom permutation** (PRP), and a **4-round Feistel network is a strong PRP** (secure even against chosen-ciphertext attacks).

This is a theoretical lower bound. In practice, cipher designers use many more rounds to provide margin against attacks that exploit the non-randomness of the actual round function. But the Luby-Rackoff theorem tells us that the Feistel structure itself is sound — given good enough round functions, three rounds suffice for semantic security.

### DES: A Feistel Cipher

The Data Encryption Standard (DES), adopted by NIST in 1977, is a 16-round Feistel cipher with a 64-bit block size and a 56-bit key. Each round uses a 48-bit round key derived from the 56-bit master key. The round function \\(F\\) consists of:

1. **Expansion:** Expand the 32-bit right half to 48 bits by duplicating certain bits
2. **Key mixing:** XOR with the 48-bit round key
3. **S-boxes:** Eight 6-to-4-bit substitution boxes provide nonlinearity
4. **Permutation:** A fixed bit permutation for diffusion

DES was carefully designed — the S-boxes were engineered by IBM's cryptography team (with classified input from the NSA) to resist differential cryptanalysis, an attack that would not be publicly discovered for another 15 years.

### Why DES Fell

DES did not fall because of a flaw in its structure. It fell because its key is too short. A 56-bit key means \\(2^{56} \approx 7.2 \times 10^{16}\\) possible keys. In 1998, the Electronic Frontier Foundation built "Deep Crack," a machine that brute-forced a DES key in 56 hours for under \$250,000. Today, a cloud computing cluster can do it in hours for a few thousand dollars.

The natural idea — encrypt twice with two independent keys (2DES) — fails because of the **meet-in-the-middle attack**. An attacker with a known plaintext-ciphertext pair \\((P, C)\\) computes all \\(2^{56}\\) possible encryptions \\(E_{K_1}(P)\\) and all \\(2^{56}\\) possible decryptions \\(D_{K_2}(C)\\), then looks for a match. This requires \\(2^{57}\\) operations and \\(2^{56}\\) storage — far less than the \\(2^{112}\\) brute force on the combined key.

**3DES** (Triple DES) uses the encrypt-decrypt-encrypt pattern \\(C = E_{K_1}(D_{K_2}(E_{K_1}(P)))\\) with two or three independent keys. It provides an effective security level of approximately 112 bits, which was adequate as a stopgap. But 3DES is three times slower than DES, and the 64-bit block size creates problems when encrypting large amounts of data (birthday-bound collisions after \\(2^{32}\\) blocks). A replacement was needed.

---

## Substitution-Permutation Networks

The **substitution-permutation network (SPN)** is an alternative block cipher architecture. Where a Feistel network processes half the block per round, an SPN transforms the *entire block* in every round.

An SPN round consists of three layers:

1. **Substitution layer (S-boxes):** The block is divided into small chunks (typically bytes), and each chunk is passed through a substitution box — a nonlinear bijective mapping. This provides confusion.

2. **Permutation layer (P-box):** The bits of the output are rearranged according to a fixed permutation. This provides diffusion by spreading the influence of each S-box output across multiple S-boxes in the next round.

3. **Key mixing:** The round key is XORed into the block.

### Why Nonlinearity Is Critical

If the S-boxes were linear functions — that is, if \\(S(x \oplus y) = S(x) \oplus S(y)\\) — then the entire cipher would be a linear function of the plaintext and key. An attacker could collect a few plaintext-ciphertext pairs, set up a system of linear equations over GF(2), and solve for the key using Gaussian elimination. The system would be \\(n\\) equations in \\(k\\) unknowns (where \\(n\\) is the block size and \\(k\\) is the key size), and a handful of known plaintext-ciphertext pairs would suffice.

This is exactly the idea behind **linear cryptanalysis**, discovered by Mitsuru Matsui in 1993. Matsui's insight was that even if the S-boxes are not perfectly linear, any statistical bias — any linear approximation that holds with probability different from \\(\frac{1}{2}\\) — can be exploited. The further the S-boxes are from any linear function, the more known plaintexts the attacker needs, and the more secure the cipher is.

### Why AES Chose SPN Over Feistel

When NIST launched the AES competition in 1997, the winning design — Rijndael by Joan Daemen and Vincent Rijmen — used an SPN rather than a Feistel network. The reasons were pragmatic:

- **Full block diffusion per round.** An SPN transforms every bit of the block in every round. A Feistel network only modifies one half. This means an SPN achieves full diffusion in fewer rounds.
- **Parallelism.** The S-box layer in an SPN applies 16 independent byte substitutions simultaneously. This maps naturally to hardware and to instruction-level parallelism.
- **Provable diffusion bounds.** Rijndael's design allows rigorous proofs about the minimum number of active S-boxes in any differential or linear trail — the so-called **wide trail strategy**.

---

## AES: The Advanced Encryption Standard from First Principles

In 1997, NIST announced an open competition to select a successor to DES. The requirements: 128-bit block size, support for 128, 192, and 256-bit keys, efficient in both hardware and software, and resistant to all known attacks. Fifteen algorithms were submitted. After three years of public cryptanalysis, Rijndael was selected in October 2000.

AES operates on a 128-bit (16-byte) block, arranged as a \\(4 \times 4\\) matrix of bytes called the **state**. Each byte is an element of the finite field \\(\text{GF}(2^8)\\). The number of rounds depends on the key size: 10 rounds for 128-bit keys, 12 for 192-bit, and 14 for 256-bit.

To understand AES, we first need to understand the field it operates over.

### The Finite Field GF(2^8)

A **finite field** (or Galois field) is a set with a finite number of elements where addition, subtraction, multiplication, and division (by nonzero elements) are all defined and satisfy the usual field axioms: associativity, commutativity, distributivity, and the existence of additive and multiplicative identities and inverses.

The field \\(\text{GF}(2)\\) is simply \\(\{0, 1\}\\) with addition and multiplication modulo 2. Addition in \\(\text{GF}(2)\\) is XOR, and multiplication is AND.

To build \\(\text{GF}(2^8)\\), we consider **polynomials over \\(\text{GF}(2)\\)** of degree less than 8. Each such polynomial has the form:

$$a_7 x^7 + a_6 x^6 + a_5 x^5 + a_4 x^4 + a_3 x^3 + a_2 x^2 + a_1 x + a_0$$

where each coefficient \\(a_i \in \{0, 1\}\\). Since there are 8 binary coefficients, there are \\(2^8 = 256\\) such polynomials — conveniently, each polynomial can be represented as a single byte, where bit \\(i\\) is the coefficient \\(a_i\\).

**Addition** in \\(\text{GF}(2^8)\\) is polynomial addition with coefficients reduced modulo 2. Since \\(1 + 1 = 0\\) in \\(\text{GF}(2)\\), addition is simply bitwise XOR. For example:

$$(x^6 + x^4 + x^2 + x + 1) + (x^7 + x + 1) = x^7 + x^6 + x^4 + x^2$$

In hex: `0x57 XOR 0x83 = 0xD4`.

**Multiplication** is polynomial multiplication followed by reduction modulo an **irreducible polynomial**. An irreducible polynomial over \\(\text{GF}(2)\\) cannot be factored into polynomials of lower degree with coefficients in \\(\text{GF}(2)\\) — it is the polynomial analogue of a prime number. AES uses the irreducible polynomial:

$$m(x) = x^8 + x^4 + x^3 + x + 1$$

which is `0x11B` in hex. To multiply two elements \\(a(x)\\) and \\(b(x)\\), compute their polynomial product and then take the remainder when dividing by \\(m(x)\\).

**Why does every nonzero element have a multiplicative inverse?** Because \\(m(x)\\) is irreducible. The set of polynomials modulo an irreducible polynomial of degree \\(n\\) over \\(\text{GF}(2)\\) forms a field with \\(2^n\\) elements. In a field, every nonzero element has a multiplicative inverse — this follows from the extended Euclidean algorithm applied to polynomials. Given any nonzero \\(a(x)\\), since \\(\gcd(a(x), m(x)) = 1\\) (because \\(m(x)\\) is irreducible and \\(a(x)\\) has degree less than 8), there exist polynomials \\(s(x)\\) and \\(t(x)\\) such that:

$$a(x) \cdot s(x) + m(x) \cdot t(x) = 1$$

Reducing modulo \\(m(x)\\): \\(a(x) \cdot s(x) \equiv 1 \pmod{m(x)}\\), so \\(s(x)\\) is the multiplicative inverse of \\(a(x)\\).

### Python Implementation of GF(2^8)

```python
import numpy as np

def gf_mul(a: int, b: int, mod: int = 0x11B) -> int:
    """Multiply two elements in GF(2^8) using the AES irreducible polynomial."""
    result = 0
    for _ in range(8):
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0x100:
            a ^= mod
        b >>= 1
    return result

def gf_pow(a: int, n: int) -> int:
    """Compute a^n in GF(2^8)."""
    result = 1
    base = a
    while n > 0:
        if n & 1:
            result = gf_mul(result, base)
        base = gf_mul(base, base)
        n >>= 1
    return result

def gf_inv(a: int) -> int:
    """Compute multiplicative inverse in GF(2^8). 0 maps to 0."""
    if a == 0:
        return 0
    # a^254 = a^(-1) in GF(2^8) since the multiplicative group has order 255
    return gf_pow(a, 254)

# Verify: a * a^(-1) should equal 1 for all nonzero a
for a in range(1, 256):
    assert gf_mul(a, gf_inv(a)) == 1, f"Inverse failed for {a}"
print("All 255 multiplicative inverses verified.")
```

### SubBytes: The S-Box

The AES S-box is the heart of the cipher's nonlinearity. It maps each byte to another byte via two steps:

**Step 1: Multiplicative inverse in \\(\text{GF}(2^8)\\).** The input byte \\(a\\) is mapped to its multiplicative inverse \\(a^{-1}\\) in \\(\text{GF}(2^8)\\). The element 0 is mapped to 0 (by convention). The multiplicative inverse provides strong nonlinearity — it has the highest possible algebraic degree (7) for a function on \\(\text{GF}(2^8)\\), and it has optimal resistance to both linear and differential cryptanalysis.

**Step 2: Affine transformation over \\(\text{GF}(2)\\).** The 8-bit result is transformed by a fixed affine map. Let \\(b = a^{-1}\\) and write \\(b\\) as a column vector of bits \\((b_0, b_1, \ldots, b_7)^T\\). The affine transformation computes:

$$b' = M \cdot b \oplus c$$

where \\(M\\) is the circulant matrix:

$$M = \begin{pmatrix} 1 & 0 & 0 & 0 & 1 & 1 & 1 & 1 \\ 1 & 1 & 0 & 0 & 0 & 1 & 1 & 1 \\ 1 & 1 & 1 & 0 & 0 & 0 & 1 & 1 \\ 1 & 1 & 1 & 1 & 0 & 0 & 0 & 1 \\ 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0 \\ 0 & 1 & 1 & 1 & 1 & 1 & 0 & 0 \\ 0 & 0 & 1 & 1 & 1 & 1 & 1 & 0 \\ 0 & 0 & 0 & 1 & 1 & 1 & 1 & 1 \end{pmatrix}$$

and \\(c = (1, 1, 0, 0, 0, 1, 1, 0)^T\\) (which is `0x63` in hex).

Why the affine transformation? The multiplicative inverse alone has an algebraic structure that could be exploited — it is a permutation polynomial over \\(\text{GF}(2^8)\\) with a compact algebraic expression. The affine transformation breaks this structure, destroying any algebraic relationships that a clever attacker might exploit while preserving the nonlinearity properties.

```python
def aes_sbox(byte_in: int) -> int:
    """Compute the AES S-box for a single byte."""
    # Step 1: multiplicative inverse
    inv = gf_inv(byte_in)
    # Step 2: affine transformation
    result = 0
    for i in range(8):
        # Each output bit is XOR of specific input bits plus constant
        bit = 0
        for j in range(8):
            bit ^= (inv >> ((j + i) % 8)) & 1
        bit ^= (0x63 >> i) & 1
        result |= (bit & 1) << i
    return result

# Build the full S-box lookup table
SBOX = [aes_sbox(i) for i in range(256)]
print("First 16 S-box values:", [hex(s) for s in SBOX[:16]])
# [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
#  0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76]
```

### ShiftRows: Inter-Column Diffusion

The AES state is a \\(4 \times 4\\) matrix of bytes:

$$\begin{pmatrix} s_{0,0} & s_{0,1} & s_{0,2} & s_{0,3} \\ s_{1,0} & s_{1,1} & s_{1,2} & s_{1,3} \\ s_{2,0} & s_{2,1} & s_{2,2} & s_{2,3} \\ s_{3,0} & s_{3,1} & s_{3,2} & s_{3,3} \end{pmatrix}$$

**ShiftRows** cyclically shifts each row to the left by its row index:
- Row 0: no shift
- Row 1: shift left by 1
- Row 2: shift left by 2
- Row 3: shift left by 3

After ShiftRows:

$$\begin{pmatrix} s_{0,0} & s_{0,1} & s_{0,2} & s_{0,3} \\ s_{1,1} & s_{1,2} & s_{1,3} & s_{1,0} \\ s_{2,2} & s_{2,3} & s_{2,0} & s_{2,1} \\ s_{3,3} & s_{3,0} & s_{3,1} & s_{3,2} \end{pmatrix}$$

This operation ensures that each column of the state after ShiftRows contains bytes that came from four *different* columns before ShiftRows. Without ShiftRows, MixColumns (which operates on columns) would only mix bytes within the same column — diffusion would be confined to columns and would never spread across the full block.

### MixColumns: Optimal Diffusion via MDS Matrices

**MixColumns** treats each column of the state as a 4-element vector over \\(\text{GF}(2^8)\\) and multiplies it by a fixed \\(4 \times 4\\) matrix:

$$\begin{pmatrix} 2 & 3 & 1 & 1 \\ 1 & 2 & 3 & 1 \\ 1 & 1 & 2 & 3 \\ 3 & 1 & 1 & 2 \end{pmatrix} \begin{pmatrix} s_{0,j} \\ s_{1,j} \\ s_{2,j} \\ s_{3,j} \end{pmatrix} = \begin{pmatrix} s'_{0,j} \\ s'_{1,j} \\ s'_{2,j} \\ s'_{3,j} \end{pmatrix}$$

All arithmetic is in \\(\text{GF}(2^8)\\) — addition is XOR, and multiplication uses the AES irreducible polynomial. For example, the first output byte of a column is:

$$s'_{0,j} = (2 \cdot s_{0,j}) \oplus (3 \cdot s_{1,j}) \oplus s_{2,j} \oplus s_{3,j}$$

where \\(2 \cdot s\\) means multiplication by 2 in \\(\text{GF}(2^8)\\) (shift left by 1, XOR with `0x1B` if the high bit was set) and \\(3 \cdot s = (2 \cdot s) \oplus s\\).

This matrix is a **maximum distance separable (MDS) matrix**. An MDS matrix has the property that any \\(r\\) rows (or columns) of the corresponding \\(4 \times 4\\) matrix are linearly independent. The cryptographic consequence: if you change \\(k\\) input bytes of a column (\\(1 \leq k \leq 4\\)), at least \\(5 - k\\) output bytes will change. The minimum number of input plus output changes is always at least 5 — the maximum possible for a \\(4 \times 4\\) matrix. This is **optimal diffusion**: the transformation spreads changes as widely as mathematically possible.

```python
def mix_column(col: list[int]) -> list[int]:
    """Apply AES MixColumns to a single column (4 bytes)."""
    def xtime(a):
        """Multiply by 2 in GF(2^8)."""
        return ((a << 1) ^ 0x1B) & 0xFF if a & 0x80 else (a << 1) & 0xFF

    def gf_mul_simple(a, b):
        """Multiply in GF(2^8) for small constants (1, 2, 3)."""
        if b == 1:
            return a
        if b == 2:
            return xtime(a)
        if b == 3:
            return xtime(a) ^ a
        raise ValueError(f"Unexpected multiplier: {b}")

    matrix = [[2, 3, 1, 1],
              [1, 2, 3, 1],
              [1, 1, 2, 3],
              [3, 1, 1, 2]]

    result = [0] * 4
    for i in range(4):
        for j in range(4):
            result[i] ^= gf_mul_simple(col[j], matrix[i][j])
    return result

# Example
col = [0xDB, 0x13, 0x53, 0x45]
print("Input column: ", [hex(b) for b in col])
print("After MixColumns:", [hex(b) for b in mix_column(col)])
# [0x8e, 0x4d, 0xa1, 0xbc]
```

### AddRoundKey

The simplest step: XOR the 128-bit state with the 128-bit round key. This is where the secret key enters the computation. Without AddRoundKey, encryption would be a fixed, key-independent permutation — useless.

### Key Schedule

The AES key schedule expands the master key into \\(N_r + 1\\) round keys (one per round, plus one for the initial AddRoundKey before round 1). For AES-128, this means expanding 16 bytes into \\(11 \times 16 = 176\\) bytes.

The expansion works word-by-word (a word is 4 bytes). Let \\(W[i]\\) denote the \\(i\\)-th word. For \\(i < 4\\) (the first four words), \\(W[i]\\) is simply the \\(i\\)-th word of the master key. For \\(i \geq 4\\):

- If \\(i \equiv 0 \pmod{4}\\): \\(W[i] = W[i-4] \oplus \text{SubWord}(\text{RotWord}(W[i-1])) \oplus \text{Rcon}[i/4]\\)
- Otherwise: \\(W[i] = W[i-4] \oplus W[i-1]\\)

Here, RotWord rotates a 4-byte word left by one byte, SubWord applies the S-box to each byte, and Rcon is a round constant: \\(\text{Rcon}[j] = (r_j, 0, 0, 0)\\) where \\(r_1 = 1\\) and \\(r_j = 2 \cdot r_{j-1}\\) in \\(\text{GF}(2^8)\\).

### A Complete AES Round

<svg viewBox="0 0 560 600" xmlns="http://www.w3.org/2000/svg" style="max-width:520px; display:block; margin:2em auto;">
  <rect width="560" height="600" rx="12" fill="#181818"/>
  <text x="280" y="32" text-anchor="middle" fill="#e0e0e0" font-size="16" font-weight="bold">AES Round Operations</text>
  <!-- Input state -->
  <rect x="180" y="55" width="200" height="40" rx="6" fill="#2a4a6a" stroke="#7ec8e3" stroke-width="1.5"/>
  <text x="280" y="80" text-anchor="middle" fill="#e0e0e0" font-size="14">Input State (16 bytes)</text>
  <!-- Arrow -->
  <line x1="280" y1="95" x2="280" y2="130" stroke="#ccc" stroke-width="2" marker-end="url(#arrow2)"/>
  <defs><marker id="arrow2" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#ccc"/></marker></defs>
  <!-- SubBytes -->
  <rect x="180" y="135" width="200" height="50" rx="8" fill="#3a2a5a" stroke="#b28ae6" stroke-width="2"/>
  <text x="280" y="157" text-anchor="middle" fill="#e0e0e0" font-size="14" font-weight="bold">SubBytes</text>
  <text x="280" y="175" text-anchor="middle" fill="#aaa" font-size="11">S-box on each byte (confusion)</text>
  <line x1="280" y1="185" x2="280" y2="220" stroke="#ccc" stroke-width="2" marker-end="url(#arrow2)"/>
  <!-- ShiftRows -->
  <rect x="180" y="225" width="200" height="50" rx="8" fill="#2a4a3a" stroke="#6bcf8e" stroke-width="2"/>
  <text x="280" y="247" text-anchor="middle" fill="#e0e0e0" font-size="14" font-weight="bold">ShiftRows</text>
  <text x="280" y="265" text-anchor="middle" fill="#aaa" font-size="11">Cyclic row shifts (diffusion)</text>
  <line x1="280" y1="275" x2="280" y2="310" stroke="#ccc" stroke-width="2" marker-end="url(#arrow2)"/>
  <!-- MixColumns -->
  <rect x="180" y="315" width="200" height="50" rx="8" fill="#4a3a1a" stroke="#f7c948" stroke-width="2"/>
  <text x="280" y="337" text-anchor="middle" fill="#e0e0e0" font-size="14" font-weight="bold">MixColumns</text>
  <text x="280" y="355" text-anchor="middle" fill="#aaa" font-size="11">MDS matrix multiply (diffusion)</text>
  <line x1="280" y1="365" x2="280" y2="400" stroke="#ccc" stroke-width="2" marker-end="url(#arrow2)"/>
  <!-- AddRoundKey -->
  <rect x="180" y="405" width="200" height="50" rx="8" fill="#4a1a1a" stroke="#f77" stroke-width="2"/>
  <text x="280" y="427" text-anchor="middle" fill="#e0e0e0" font-size="14" font-weight="bold">AddRoundKey</text>
  <text x="280" y="445" text-anchor="middle" fill="#aaa" font-size="11">XOR with round key (key mixing)</text>
  <!-- Key input -->
  <text x="460" y="432" fill="#f77" font-size="13">Round Key K_i</text>
  <line x1="453" y1="430" x2="382" y2="430" stroke="#f77" stroke-width="1.5" marker-end="url(#arrow2)"/>
  <line x1="280" y1="455" x2="280" y2="490" stroke="#ccc" stroke-width="2" marker-end="url(#arrow2)"/>
  <!-- Output -->
  <rect x="180" y="495" width="200" height="40" rx="6" fill="#2a4a6a" stroke="#7ec8e3" stroke-width="1.5"/>
  <text x="280" y="520" text-anchor="middle" fill="#e0e0e0" font-size="14">Output State</text>
  <!-- Note -->
  <text x="280" y="565" text-anchor="middle" fill="#888" font-size="11" font-style="italic">Note: MixColumns is omitted in the final round</text>
</svg>

Let us trace one AES-128 round with concrete values. Suppose the state entering the round is:

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| Row 0 | `0x19` | `0xa0` | `0x9a` | `0xe9` |
| Row 1 | `0x3d` | `0xf4` | `0xc6` | `0xf8` |
| Row 2 | `0xe3` | `0xe2` | `0x8d` | `0x48` |
| Row 3 | `0xbe` | `0x2b` | `0x2a` | `0x08` |

**After SubBytes** (applying the S-box to each byte):

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| Row 0 | `0xd4` | `0xe0` | `0xb8` | `0x1e` |
| Row 1 | `0x27` | `0xbf` | `0xb4` | `0x41` |
| Row 2 | `0x11` | `0x98` | `0x5d` | `0x52` |
| Row 3 | `0xae` | `0xf1` | `0xe5` | `0x30` |

**After ShiftRows:**

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| Row 0 | `0xd4` | `0xe0` | `0xb8` | `0x1e` |
| Row 1 | `0xbf` | `0xb4` | `0x41` | `0x27` |
| Row 2 | `0x5d` | `0x52` | `0x11` | `0x98` |
| Row 3 | `0x30` | `0xae` | `0xf1` | `0xe5` |

**After MixColumns:** Each column is multiplied by the MDS matrix in \\(\text{GF}(2^8)\\).

**After AddRoundKey:** XOR with the round key to produce the output state.

### Python: Simplified AES Round

```python
import numpy as np

# Build the full AES S-box
def build_sbox():
    sbox = []
    for i in range(256):
        inv = gf_inv(i)
        # Affine transformation
        result = 0
        for bit in range(8):
            b = 0
            for j in range(8):
                b ^= (inv >> ((j + bit) % 8)) & 1
            b ^= (0x63 >> bit) & 1
            result |= (b & 1) << bit
        sbox.append(result)
    return sbox

SBOX = build_sbox()

def sub_bytes(state: np.ndarray) -> np.ndarray:
    """Apply S-box substitution to every byte."""
    return np.array([[SBOX[state[r][c]] for c in range(4)] for r in range(4)], dtype=np.uint8)

def shift_rows(state: np.ndarray) -> np.ndarray:
    """Cyclically shift each row left by its row index."""
    result = state.copy()
    for r in range(1, 4):
        result[r] = np.roll(state[r], -r)
    return result

def mix_columns_state(state: np.ndarray) -> np.ndarray:
    """Apply MixColumns to each column of the state."""
    result = state.copy()
    for j in range(4):
        col = [int(state[i][j]) for i in range(4)]
        mixed = mix_column(col)
        for i in range(4):
            result[i][j] = mixed[i]
    return result

def add_round_key(state: np.ndarray, round_key: np.ndarray) -> np.ndarray:
    """XOR state with round key."""
    return state ^ round_key

def aes_round(state: np.ndarray, round_key: np.ndarray) -> np.ndarray:
    """One complete AES round."""
    state = sub_bytes(state)
    state = shift_rows(state)
    state = mix_columns_state(state)
    state = add_round_key(state, round_key)
    return state

# Example: run one round
state = np.array([
    [0x19, 0xa0, 0x9a, 0xe9],
    [0x3d, 0xf4, 0xc6, 0xf8],
    [0xe3, 0xe2, 0x8d, 0x48],
    [0xbe, 0x2b, 0x2a, 0x08]
], dtype=np.uint8)

# Use a dummy round key for illustration
round_key = np.zeros((4, 4), dtype=np.uint8)

result = aes_round(state, round_key)
print("State after one AES round:")
for row in result:
    print(" ".join(f"0x{b:02x}" for b in row))
```

---

## Modes of Operation — Turning a Block Cipher into an Encryption Scheme

A block cipher like AES encrypts exactly one block — 128 bits, 16 bytes. Real messages are longer. A **mode of operation** defines how to use a block cipher to encrypt arbitrarily long messages. The choice of mode is at least as important as the choice of cipher. A strong cipher in a weak mode is a weak cipher.

### ECB Mode: The Textbook Mistake

**Electronic Codebook (ECB)** mode is the simplest: divide the message into blocks and encrypt each block independently.

$$C_i = E_K(P_i)$$

ECB is catastrophically broken for most uses. Because identical plaintext blocks produce identical ciphertext blocks, patterns in the plaintext are preserved in the ciphertext. The classic demonstration is the "ECB penguin" — encrypt a bitmap image using ECB, and the penguin is still visible in the ciphertext because regions of the same color produce the same ciphertext blocks.

ECB violates diffusion at the message level. Each block is independent. An attacker can reorder, duplicate, or delete ciphertext blocks without detection. ECB must never be used for encrypting data longer than one block.

### CBC Mode: Chaining for Diffusion

**Cipher Block Chaining (CBC)** mode XORs each plaintext block with the previous ciphertext block before encryption:

$$C_0 = E_K(P_0 \oplus IV)$$
$$C_i = E_K(P_i \oplus C_{i-1}) \quad \text{for } i \geq 1$$

The **initialization vector (IV)** must be unpredictable (not merely unique). If the IV is predictable, an attacker can mount a chosen-plaintext attack: they can test whether a guessed plaintext block \\(P^*\\) equals a previously encrypted block by crafting a new plaintext that XORs away the IV difference.

CBC provides diffusion across blocks — changing one plaintext bit affects all subsequent ciphertext blocks. However, CBC encryption is inherently sequential (each block depends on the previous ciphertext), which limits parallelism.

### CTR Mode: Block Cipher as Stream Cipher

**Counter (CTR)** mode turns a block cipher into a stream cipher. Instead of encrypting the plaintext directly, it encrypts a counter value and XORs the result with the plaintext:

$$C_i = P_i \oplus E_K(\text{Nonce} \| \text{Counter}_i)$$

CTR mode is fully parallelizable — all counter values are known in advance, so all blocks can be encrypted simultaneously. It requires only the encryption function (never decryption), which simplifies hardware. The nonce must never be reused with the same key; nonce reuse is catastrophic because it reveals \\(P_i \oplus P_j\\).

<svg viewBox="0 0 700 680" xmlns="http://www.w3.org/2000/svg" style="max-width:660px; display:block; margin:2em auto;">
  <defs>
    <marker id="arrow3" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#ccc"/></marker>
  </defs>
  <rect width="700" height="680" rx="12" fill="#181818"/>

  <!-- ECB Section -->
  <text x="350" y="30" text-anchor="middle" fill="#e0e0e0" font-size="16" font-weight="bold">Modes of Operation Comparison</text>
  <text x="175" y="62" text-anchor="middle" fill="#f77" font-size="14" font-weight="bold">ECB Mode (Insecure)</text>

  <!-- ECB Block 1 -->
  <rect x="60" y="75" width="70" height="30" rx="4" fill="#2a4a6a" stroke="#7ec8e3" stroke-width="1"/>
  <text x="95" y="95" text-anchor="middle" fill="#e0e0e0" font-size="11">P₁</text>
  <line x1="95" y1="105" x2="95" y2="125" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="65" y="128" width="60" height="28" rx="4" fill="#3a2a5a" stroke="#b28ae6" stroke-width="1.5"/>
  <text x="95" y="147" text-anchor="middle" fill="#e0e0e0" font-size="10">E_K</text>
  <line x1="95" y1="156" x2="95" y2="178" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="60" y="180" width="70" height="30" rx="4" fill="#4a1a1a" stroke="#f77" stroke-width="1"/>
  <text x="95" y="200" text-anchor="middle" fill="#e0e0e0" font-size="11">C₁</text>

  <!-- ECB Block 2 -->
  <rect x="180" y="75" width="70" height="30" rx="4" fill="#2a4a6a" stroke="#7ec8e3" stroke-width="1"/>
  <text x="215" y="95" text-anchor="middle" fill="#e0e0e0" font-size="11">P₂</text>
  <line x1="215" y1="105" x2="215" y2="125" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="185" y="128" width="60" height="28" rx="4" fill="#3a2a5a" stroke="#b28ae6" stroke-width="1.5"/>
  <text x="215" y="147" text-anchor="middle" fill="#e0e0e0" font-size="10">E_K</text>
  <line x1="215" y1="156" x2="215" y2="178" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="180" y="180" width="70" height="30" rx="4" fill="#4a1a1a" stroke="#f77" stroke-width="1"/>
  <text x="215" y="200" text-anchor="middle" fill="#e0e0e0" font-size="11">C₂</text>

  <text x="273" y="145" fill="#888" font-size="12">...</text>

  <!-- ECB Note -->
  <text x="175" y="232" text-anchor="middle" fill="#f77" font-size="10" font-style="italic">P₁ = P₂ ⟹ C₁ = C₂ (patterns leak!)</text>

  <!-- CBC Section -->
  <text x="350" y="275" text-anchor="middle" fill="#6bcf8e" font-size="14" font-weight="bold">CBC Mode</text>

  <rect x="30" y="290" width="50" height="25" rx="4" fill="#2a3a2a" stroke="#6bcf8e" stroke-width="1"/>
  <text x="55" y="307" text-anchor="middle" fill="#e0e0e0" font-size="10">IV</text>

  <rect x="110" y="290" width="70" height="25" rx="4" fill="#2a4a6a" stroke="#7ec8e3" stroke-width="1"/>
  <text x="145" y="307" text-anchor="middle" fill="#e0e0e0" font-size="11">P₁</text>

  <!-- XOR -->
  <line x1="80" y1="302" x2="100" y2="330" stroke="#6bcf8e" stroke-width="1.5"/>
  <line x1="145" y1="315" x2="145" y2="325" stroke="#ccc" stroke-width="1.5"/>
  <circle cx="120" cy="335" r="12" fill="none" stroke="#6bcf8e" stroke-width="1.5"/>
  <text x="120" y="340" text-anchor="middle" fill="#6bcf8e" font-size="13">⊕</text>

  <line x1="120" y1="347" x2="120" y2="365" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="90" y="368" width="60" height="28" rx="4" fill="#3a2a5a" stroke="#b28ae6" stroke-width="1.5"/>
  <text x="120" y="387" text-anchor="middle" fill="#e0e0e0" font-size="10">E_K</text>
  <line x1="120" y1="396" x2="120" y2="418" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="85" y="420" width="70" height="25" rx="4" fill="#4a1a1a" stroke="#f77" stroke-width="1"/>
  <text x="120" y="437" text-anchor="middle" fill="#e0e0e0" font-size="11">C₁</text>

  <!-- CBC Block 2 -->
  <rect x="280" y="290" width="70" height="25" rx="4" fill="#2a4a6a" stroke="#7ec8e3" stroke-width="1"/>
  <text x="315" y="307" text-anchor="middle" fill="#e0e0e0" font-size="11">P₂</text>

  <line x1="155" y1="433" x2="270" y2="335" stroke="#6bcf8e" stroke-width="1.5" stroke-dasharray="4,3"/>
  <circle cx="290" cy="335" r="12" fill="none" stroke="#6bcf8e" stroke-width="1.5"/>
  <text x="290" y="340" text-anchor="middle" fill="#6bcf8e" font-size="13">⊕</text>

  <line x1="290" y1="347" x2="290" y2="365" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="260" y="368" width="60" height="28" rx="4" fill="#3a2a5a" stroke="#b28ae6" stroke-width="1.5"/>
  <text x="290" y="387" text-anchor="middle" fill="#e0e0e0" font-size="10">E_K</text>
  <line x1="290" y1="396" x2="290" y2="418" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="255" y="420" width="70" height="25" rx="4" fill="#4a1a1a" stroke="#f77" stroke-width="1"/>
  <text x="290" y="437" text-anchor="middle" fill="#e0e0e0" font-size="11">C₂</text>

  <!-- CTR Section -->
  <text x="350" y="485" text-anchor="middle" fill="#f7c948" font-size="14" font-weight="bold">CTR Mode</text>

  <!-- CTR Block 1 -->
  <rect x="60" y="500" width="80" height="25" rx="4" fill="#2a3a2a" stroke="#f7c948" stroke-width="1"/>
  <text x="100" y="517" text-anchor="middle" fill="#e0e0e0" font-size="9">Nonce ‖ Ctr₁</text>
  <line x1="100" y1="525" x2="100" y2="545" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="70" y="548" width="60" height="28" rx="4" fill="#3a2a5a" stroke="#b28ae6" stroke-width="1.5"/>
  <text x="100" y="567" text-anchor="middle" fill="#e0e0e0" font-size="10">E_K</text>
  <line x1="100" y1="576" x2="100" y2="596" stroke="#ccc" stroke-width="1.5"/>
  <circle cx="100" cy="610" r="12" fill="none" stroke="#f7c948" stroke-width="1.5"/>
  <text x="100" y="615" text-anchor="middle" fill="#f7c948" font-size="13">⊕</text>

  <rect x="155" y="598" width="50" height="25" rx="4" fill="#2a4a6a" stroke="#7ec8e3" stroke-width="1"/>
  <text x="180" y="615" text-anchor="middle" fill="#e0e0e0" font-size="10">P₁</text>
  <line x1="155" y1="610" x2="112" y2="610" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>

  <line x1="100" y1="622" x2="100" y2="645" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="65" y="647" width="70" height="22" rx="4" fill="#4a1a1a" stroke="#f77" stroke-width="1"/>
  <text x="100" y="663" text-anchor="middle" fill="#e0e0e0" font-size="11">C₁</text>

  <!-- CTR Block 2 -->
  <rect x="290" y="500" width="80" height="25" rx="4" fill="#2a3a2a" stroke="#f7c948" stroke-width="1"/>
  <text x="330" y="517" text-anchor="middle" fill="#e0e0e0" font-size="9">Nonce ‖ Ctr₂</text>
  <line x1="330" y1="525" x2="330" y2="545" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="300" y="548" width="60" height="28" rx="4" fill="#3a2a5a" stroke="#b28ae6" stroke-width="1.5"/>
  <text x="330" y="567" text-anchor="middle" fill="#e0e0e0" font-size="10">E_K</text>
  <line x1="330" y1="576" x2="330" y2="596" stroke="#ccc" stroke-width="1.5"/>
  <circle cx="330" cy="610" r="12" fill="none" stroke="#f7c948" stroke-width="1.5"/>
  <text x="330" y="615" text-anchor="middle" fill="#f7c948" font-size="13">⊕</text>

  <rect x="385" y="598" width="50" height="25" rx="4" fill="#2a4a6a" stroke="#7ec8e3" stroke-width="1"/>
  <text x="410" y="615" text-anchor="middle" fill="#e0e0e0" font-size="10">P₂</text>
  <line x1="385" y1="610" x2="342" y2="610" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>

  <line x1="330" y1="622" x2="330" y2="645" stroke="#ccc" stroke-width="1.5" marker-end="url(#arrow3)"/>
  <rect x="295" y="647" width="70" height="22" rx="4" fill="#4a1a1a" stroke="#f77" stroke-width="1"/>
  <text x="330" y="663" text-anchor="middle" fill="#e0e0e0" font-size="11">C₂</text>

  <!-- CTR note -->
  <text x="530" y="565" fill="#888" font-size="10" font-style="italic">Fully</text>
  <text x="530" y="580" fill="#888" font-size="10" font-style="italic">parallelizable</text>
</svg>

### GCM Mode: Authenticated Encryption

Encryption alone is not enough. An attacker who cannot read the plaintext can still *modify* the ciphertext. Without authentication, the receiver has no way to detect tampering. This leads to devastating **chosen-ciphertext attacks** — the attacker modifies ciphertext blocks and observes how the system responds to the decrypted (now corrupted) plaintext.

**Galois/Counter Mode (GCM)** combines CTR mode encryption with a polynomial authentication tag. GCM provides **authenticated encryption with associated data (AEAD)**: it guarantees both confidentiality and integrity.

GCM encrypts using CTR mode and simultaneously computes an authentication tag using the **GHASH** function. GHASH is a universal hash function based on multiplication in \\(\text{GF}(2^{128})\\). Let \\(H = E_K(0^{128})\\) be the hash key (the encryption of the zero block). GHASH processes the ciphertext blocks \\(C_1, C_2, \ldots, C_n\\) as:

$$X_0 = 0$$
$$X_i = (X_{i-1} \oplus C_i) \cdot H \quad \text{in } \text{GF}(2^{128})$$

The final tag is \\(T = X_n \oplus E_K(\text{Nonce} \| 0^{31} \| 1)\\).

The GHASH construction is a polynomial evaluation: the tag is essentially the polynomial \\(C_1 H^n + C_2 H^{n-1} + \cdots + C_n H\\) evaluated in \\(\text{GF}(2^{128})\\). If any ciphertext block is modified, the tag changes unpredictably (assuming the attacker does not know \\(H\\)), and the receiver rejects the message.

GCM is the standard mode for TLS 1.3, IPsec, and most modern encrypted communications. AES-256-GCM is what protects the encrypted military communications in our opening scenario.

---

## Stream Ciphers

Block ciphers encrypt fixed-size blocks. **Stream ciphers** encrypt one bit (or byte) at a time, XORing each plaintext unit with a corresponding unit from a pseudorandom **keystream**:

$$C_i = P_i \oplus S_i$$

where \\(S_1, S_2, S_3, \ldots\\) is the keystream generated from the key (and typically a nonce). The security of the stream cipher depends entirely on the quality of the keystream generator.

### Linear Feedback Shift Registers (LFSRs)

An **LFSR** is a shift register whose input bit is a linear function (XOR) of its current state. An LFSR of length \\(n\\) has \\(n\\) bit positions and produces one output bit per clock cycle. The feedback function is defined by a set of **tap positions**. If the state at time \\(t\\) is \\((s_t, s_{t+1}, \ldots, s_{t+n-1})\\), the next bit is:

$$s_{t+n} = c_1 s_{t+n-1} \oplus c_2 s_{t+n-2} \oplus \cdots \oplus c_n s_t$$

where \\(c_1, \ldots, c_n \in \{0, 1\}\\) are the feedback coefficients.

The **characteristic polynomial** of the LFSR is:

$$p(x) = x^n + c_1 x^{n-1} + c_2 x^{n-2} + \cdots + c_n$$

If \\(p(x)\\) is a **primitive polynomial** over \\(\text{GF}(2)\\), the LFSR has maximal period \\(2^n - 1\\) — it cycles through every nonzero \\(n\\)-bit state before repeating. This produces a sequence with excellent statistical properties: balanced 0s and 1s, good autocorrelation, long period.

The evolution of the LFSR can be expressed as matrix multiplication. Let \\(\mathbf{s}_t = (s_t, s_{t+1}, \ldots, s_{t+n-1})^T\\). Then:

$$\mathbf{s}_{t+1} = A \cdot \mathbf{s}_t$$

where \\(A\\) is the **companion matrix** of \\(p(x)\\). This linear algebraic structure is both the LFSR's strength (easy to analyze and implement) and its fatal weakness.

### Why LFSRs Alone Are Insecure: The Berlekamp-Massey Algorithm

The **Berlekamp-Massey algorithm** can reconstruct the feedback polynomial of an LFSR of length \\(n\\) from just \\(2n\\) consecutive output bits. The algorithm runs in \\(O(n^2)\\) time. This means an attacker who observes \\(2n\\) bits of keystream can predict all future (and past) keystream bits.

An LFSR with 128 taps might seem to have a huge state space, but 256 known keystream bits are enough to break it. This is why LFSRs are never used alone in modern stream ciphers. They are combined using nonlinear operations — clock control, nonlinear filters, irregular clocking — to defeat Berlekamp-Massey.

### ChaCha20: Modern Stream Cipher Design

**ChaCha20**, designed by Daniel Bernstein, takes a completely different approach from LFSR-based designs. It operates on a \\(4 \times 4\\) matrix of 32-bit words — a 512-bit state. The initial state contains:

- 4 words of a fixed constant ("expand 32-byte k")
- 8 words of the 256-bit key
- 1 word counter
- 3 words of nonce

The core operation is the **quarter-round**, which applies to four 32-bit words \\((a, b, c, d)\\):

$$a = a + b; \quad d = (d \oplus a) \lll 16$$
$$c = c + d; \quad b = (b \oplus c) \lll 12$$
$$a = a + b; \quad d = (d \oplus a) \lll 8$$
$$c = c + d; \quad b = (b \oplus c) \lll 7$$

where \\(\lll\\) denotes left rotation. The full ChaCha20 function applies 20 rounds (alternating column rounds and diagonal rounds) and adds the initial state to the result. Each 512-bit output block serves as 64 bytes of keystream.

ChaCha20's design philosophy is: use simple operations (addition, XOR, rotation — collectively called **ARX**) applied many times. No S-boxes, no finite field arithmetic, no lookup tables. This makes ChaCha20 resistant to cache-timing attacks that can leak information from table-based designs like AES when implemented in software without hardware acceleration.

### ChaCha20-Poly1305: AEAD for the Modern Web

**Poly1305** is a one-time authenticator that, paired with ChaCha20, gives an AEAD construction analogous to AES-GCM. Poly1305 computes a 128-bit tag by evaluating the message as a polynomial modulo the prime \\(2^{130} - 5\\).

Given a one-time key \\(r\\) and a message split into 128-bit blocks \\(m_1, m_2, \ldots, m_n\\), the tag is:

$$\text{tag} = \left(\sum_{i=1}^{n} m_i \cdot r^{n+1-i}\right) \mod (2^{130} - 5)$$

plus a one-time pad derived from ChaCha20. The arithmetic is fast because \\(2^{130} - 5\\) is a Mersenne-like prime that allows efficient modular reduction.

ChaCha20-Poly1305 is the default cipher suite in Google Chrome, Android, and WireGuard VPN. It is the primary alternative to AES-GCM in TLS 1.3.

---

## Measuring Cipher Strength

How do we know a cipher is secure? We cannot prove that AES is unbreakable — that would require proving \\(P \neq NP\\) and much more. What we can do is show that a cipher resists all known classes of attacks and has provable lower bounds on the cost of the most efficient attacks.

### Linear Cryptanalysis

**Linear cryptanalysis**, introduced by Matsui (1993), exploits statistical biases in the cipher's linear approximations. A **linear approximation** is an equation of the form:

$$P_{i_1} \oplus P_{i_2} \oplus \cdots \oplus C_{j_1} \oplus C_{j_2} \oplus \cdots = K_{k_1} \oplus K_{k_2} \oplus \cdots$$

that holds with probability \\(p \neq \frac{1}{2}\\). The **bias** is \\(\epsilon = |p - \frac{1}{2}|\\). The number of known plaintext-ciphertext pairs needed to exploit this bias is approximately:

$$N \approx \frac{1}{\epsilon^2}$$

For the linear approximation to span the full cipher, biases from individual rounds are combined using the **piling-up lemma**. If \\(n\\) independent biases \\(\epsilon_1, \ldots, \epsilon_n\\) are stacked, the overall bias is:

$$\epsilon = 2^{n-1} \prod_{i=1}^{n} \epsilon_i$$

This shrinks exponentially with the number of rounds. If each S-box has maximum bias \\(\epsilon_s\\), and the cipher forces at least \\(d\\) active S-boxes in any linear trail, then the overall bias is bounded by:

$$\epsilon \leq 2^{n-1} \cdot \epsilon_s^d$$

AES is designed so that any linear trail across 4 rounds must pass through at least 25 active S-boxes. With the AES S-box's maximum bias of \\(2^{-3}\\), the overall bias over 4 rounds is at most \\(2^3 \cdot (2^{-3})^{25} = 2^{-72}\\). Exploiting this would require more than \\(2^{144}\\) known plaintexts — far exceeding the \\(2^{128}\\) possible distinct plaintext blocks. The attack is information-theoretically impossible.

### Differential Cryptanalysis

**Differential cryptanalysis**, discovered by Biham and Shamir (1990), studies how input differences propagate through the cipher. Given a pair of plaintexts with a known XOR difference \\(\Delta P = P \oplus P'\\), the attacker studies the distribution of output differences \\(\Delta C = C \oplus C'\\).

The key object is the **difference distribution table (DDT)** of each S-box. For an S-box \\(S\\), the DDT entry at \\((\Delta_{\text{in}}, \Delta_{\text{out}})\\) counts the number of input pairs with difference \\(\Delta_{\text{in}}\\) that produce output difference \\(\Delta_{\text{out}}\\):

$$\text{DDT}(\Delta_{\text{in}}, \Delta_{\text{out}}) = \#\{x : S(x) \oplus S(x \oplus \Delta_{\text{in}}) = \Delta_{\text{out}}\}$$

For a random 8-bit permutation, the expected DDT entry is 1 (out of 256). The AES S-box has a maximum DDT entry of 4 — very close to ideal. The **differential probability** through one S-box is at most \\(4/256 = 2^{-6}\\).

With at least 25 active S-boxes over 4 rounds, the probability of any differential characteristic spanning 4 rounds is at most \\((2^{-6})^{25} = 2^{-150}\\). This is far below \\(2^{-128}\\), so the attack requires more chosen-plaintext pairs than exist.

### The Wide Trail Strategy

The **wide trail strategy** is Daemen and Rijmen's design principle that guarantees these bounds. It works by ensuring that the diffusion layer (ShiftRows + MixColumns) forces any non-trivial differential or linear trail to activate many S-boxes. The mathematical tool is the concept of a **branch number**.

The **branch number** of a linear transformation \\(L\\) is:

$$\mathcal{B}(L) = \min_{a \neq 0} \left( w(a) + w(L(a)) \right)$$

where \\(w(\cdot)\\) counts the number of nonzero bytes. For AES's MixColumns (an MDS matrix), the branch number is 5 — the maximum possible for a \\(4 \times 4\\) matrix. This guarantees that any two consecutive rounds have at least 5 active S-boxes, and four consecutive rounds have at least 25.

### Python: Demonstrating the Avalanche Effect

```python
import numpy as np
import matplotlib.pyplot as plt

def simple_spn_round(block: int, key: int, sbox: list[int]) -> int:
    """One round of a simplified 16-bit SPN."""
    # Key mixing
    block ^= key
    # Substitution: four 4-bit S-boxes
    result = 0
    for i in range(4):
        nibble = (block >> (4 * i)) & 0xF
        result |= sbox[nibble] << (4 * i)
    # Permutation: bit permutation
    perm = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    permuted = 0
    for i in range(16):
        if result & (1 << i):
            permuted |= 1 << perm[i]
    return permuted

def simple_spn(plaintext: int, key: int, rounds: int = 4) -> int:
    """Simplified SPN cipher for demonstration."""
    # Simple 4-bit S-box with good nonlinearity
    sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
            0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
    block = plaintext
    for r in range(rounds):
        round_key = (key >> (r * 4)) & 0xFFFF  # Simple key schedule
        block = simple_spn_round(block, round_key, sbox)
    return block

def count_bit_changes(a: int, b: int, bits: int = 16) -> int:
    """Count the number of differing bits."""
    return bin(a ^ b).count('1')

# Measure avalanche: flip each input bit, count output bit changes
np.random.seed(42)
num_trials = 2000
rounds_to_test = range(1, 6)

avg_changes = []
for num_rounds in rounds_to_test:
    changes = []
    for _ in range(num_trials):
        plaintext = np.random.randint(0, 0x10000)
        key = np.random.randint(0, 0x100000)
        ct_original = simple_spn(plaintext, key, num_rounds)
        for bit in range(16):
            pt_flipped = plaintext ^ (1 << bit)
            ct_flipped = simple_spn(pt_flipped, key, num_rounds)
            changes.append(count_bit_changes(ct_original, ct_flipped))
    avg_changes.append(np.mean(changes))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(list(rounds_to_test), avg_changes, color='#7ec8e3', edgecolor='#2a4a6a', linewidth=1.2)
ax.axhline(y=8.0, color='#f77', linestyle='--', linewidth=1.5, label=r'Ideal: $n/2 = 8$ bits')
ax.set_xlabel(r'Number of Rounds', fontsize=13)
ax.set_ylabel(r'Average Bit Changes (out of 16)', fontsize=13)
ax.set_title(r'Avalanche Effect: Bit Diffusion vs. Number of Rounds', fontsize=14)
ax.set_ylim(0, 16)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('avalanche_effect.png', dpi=150, bbox_inches='tight')
plt.show()
```

The plot shows the avalanche effect building over successive rounds. After one round, a single bit flip changes only a few output bits — the diffusion has not yet spread the influence across the full block. By round 3 or 4, the average number of changed bits approaches the ideal of \\(n/2 = 8\\) out of 16. This is exactly Shannon's diffusion principle made quantitative.

---

## Conclusion

Symmetric cryptography is the workhorse of secure communications. When MUST warns about state-sponsored cyber attacks against Swedish critical infrastructure, the mathematical structures in this article are the front line of defence. AES-256-GCM, protecting military and government communications, rests on:

1. **Shannon's principles** — confusion from the S-box, diffusion from ShiftRows and MixColumns
2. **The finite field \\(\text{GF}(2^8)\\)** — providing the algebraic structure for nonlinear operations and optimal diffusion matrices
3. **The wide trail strategy** — guaranteeing that every attack path activates enough S-boxes to make the attack computationally infeasible
4. **Authenticated encryption** — ensuring that an attacker who cannot break the cipher also cannot modify the ciphertext undetected

The 256-bit key space means \\(2^{256}\\) possible keys — a number larger than the estimated number of atoms in the observable universe. Even a nation-state with access to every computer on Earth, running for the age of the universe, could not brute-force a single AES-256 key. The known cryptanalytic attacks on full AES (biclique attacks) reduce the effective security from \\(2^{256}\\) to \\(2^{254.4}\\) — a speedup so marginal it is entirely irrelevant in practice.

The intercepted military communications from our opening scenario are safe. The encryption holds. But symmetric cryptography has a fundamental limitation: both parties must share the same secret key. How do two parties who have never met establish a shared secret over an insecure channel? That is the problem of **asymmetric cryptography**, and the subject of Part 3.

*Next: [Part 3 — Asymmetric Cryptography: RSA, Elliptic Curves, and the Key Distribution Problem](/2026/03/14/asymmetric-cryptography-rsa-elliptic-curves.html)*
