---
layout: post
title: "Number Theory and Classical Cryptography: The Mathematics of Secrecy"
date: 2026-03-12
category: math
---

*This is Part 1 of a 5-part series on cryptology. **Part 1: Number Theory & Classical Ciphers** | [Part 2: Symmetric Cryptography](/2026/03/13/symmetric-cryptography-aes-block-ciphers.html) | [Part 3: Asymmetric Cryptography](/2026/03/14/asymmetric-cryptography-rsa-elliptic-curves.html) | [Part 4: Cryptographic Protocols](/2026/03/15/cryptographic-protocols-tls-signal-national-security.html) | [Part 5: Post-Quantum Cryptography](/2026/03/16/quantum-threat-post-quantum-cryptography.html)*

In January 2024, several undersea fiber-optic cables in the Baltic Sea were severed in quick succession. Swedish and Finnish authorities launched investigations. The cables --- carrying internet traffic, financial data, and government communications between Nordic states --- had been physically cut. The Swedish Military Intelligence and Security Service (MUST) noted that this fit a broader pattern: Russia's hybrid warfare operations in the Baltic region, which also include systematic GNSS spoofing near Gotland, manipulation of AIS signals used by maritime traffic, and sophisticated cyber intrusions targeting Sweden's total defence infrastructure. MUST's 2025 annual report describes a landscape where adversaries deploy AI-generated influence operations, run a "gig economy" for cyber attacks, and probe communications infrastructure with increasing frequency.

MUST's mandate includes Communications Security (COMSEC) for Sweden's total defence --- protecting the confidentiality and integrity of military, governmental, and critical civilian communications. Working alongside FRA (the National Defence Radio Establishment, Sweden's signals intelligence agency) and Säkerhetspolisen (the Security Service), MUST is responsible for ensuring that even when cables are tapped, signals are intercepted, or networks are compromised, the *content* of communications remains unintelligible to the adversary.

This is not a networking problem. It is not an engineering problem. It is a mathematical problem. The question at the heart of COMSEC is: **can we transform a message so that anyone who intercepts it learns nothing about its content, while the intended recipient can recover the original message perfectly?** The answer involves number theory, abstract algebra, information theory, and computational complexity. This series builds that mathematics from the ground up.

This first part lays the algebraic foundations. We start with modular arithmetic, build the group-theoretic structures that underpin every modern cipher, examine the classical ciphers that secured (and failed to secure) communications for centuries, and arrive at Shannon's rigorous definition of what "perfect secrecy" even means.

---

## Table of Contents

1. [The Arithmetic of Secrecy](#the-arithmetic-of-secrecy)
2. [Modular Arithmetic from First Principles](#modular-arithmetic-from-first-principles)
3. [The Extended Euclidean Algorithm](#the-extended-euclidean-algorithm)
4. [Euler's Totient Function and Euler's Theorem](#eulers-totient-function-and-eulers-theorem)
5. [Groups, Rings, and Fields --- The Algebraic Backbone](#groups-rings-and-fields--the-algebraic-backbone)
6. [Classical Ciphers and Their Destruction](#classical-ciphers-and-their-destruction)
7. [Shannon's Perfect Secrecy](#shannons-perfect-secrecy)
8. [From Classical to Modern](#from-classical-to-modern--why-we-need-more-mathematics)

---

## The Arithmetic of Secrecy

Why is secrecy a *mathematical* problem? Because a cipher is a function. You take a message \\(m\\), a key \\(k\\), and produce a ciphertext \\(c = E(k, m)\\). The recipient, who knows \\(k\\), applies the decryption function \\(m = D(k, c)\\). The adversary sees \\(c\\) and wants to recover \\(m\\) without knowing \\(k\\).

The security of the system depends entirely on the mathematical properties of \\(E\\) and \\(D\\). If \\(E\\) has algebraic structure that leaks information about \\(m\\) through \\(c\\), the cipher is broken. If \\(E\\) destroys all statistical patterns in \\(m\\), the cipher is secure. The history of cryptography is the history of discovering which mathematical structures preserve secrecy and which ones shatter it.

Every cipher we will encounter in this series --- from Caesar's trivial shift to AES to elliptic curve Diffie-Hellman --- performs its operations in the world of **modular arithmetic**. Numbers wrap around. Addition, multiplication, and exponentiation happen not on the infinite number line but on finite cyclic structures where everything eventually returns to where it started. This is where we begin.

---

## Modular Arithmetic from First Principles

### Clock arithmetic

Take the integers and imagine wrapping them around a circle with \\(n\\) positions, labeled \\(0, 1, 2, \ldots, n-1\\). When you reach \\(n\\), you wrap back to \\(0\\). This is modular arithmetic modulo \\(n\\), and it is the single most important structure in cryptography.

**Definition.** Let \\(a\\) and \\(b\\) be integers and \\(n\\) a positive integer. We say \\(a\\) is **congruent** to \\(b\\) modulo \\(n\\), written

$$a \equiv b \pmod{n}$$

if \\(n\\) divides \\(a - b\\). Equivalently, \\(a\\) and \\(b\\) leave the same remainder when divided by \\(n\\).

This is not just notation. Congruence modulo \\(n\\) is an **equivalence relation** on the integers. Let us verify the three required properties.

**Reflexive:** \\(a \equiv a \pmod{n}\\) because \\(n \mid (a - a) = 0\\). Every integer is congruent to itself.

**Symmetric:** If \\(a \equiv b \pmod{n}\\), then \\(n \mid (a - b)\\), so \\(n \mid (b - a)\\), so \\(b \equiv a \pmod{n}\\).

**Transitive:** If \\(a \equiv b \pmod{n}\\) and \\(b \equiv c \pmod{n}\\), then \\(n \mid (a - b)\\) and \\(n \mid (b - c)\\). Since divisibility is closed under addition, \\(n \mid ((a - b) + (b - c)) = (a - c)\\), so \\(a \equiv c \pmod{n}\\).

Because congruence is an equivalence relation, it partitions the integers into \\(n\\) equivalence classes: \\([0], [1], [2], \ldots, [n-1]\\). The set of these equivalence classes is denoted \\(\mathbb{Z}/n\mathbb{Z}\\) (read "the integers modulo \\(n\\)").

### Modular operations

The power of this construction is that we can define arithmetic on equivalence classes, and it is **well-defined** --- meaning the result does not depend on which representative we choose from each class.

**Addition:** \\([a] + [b] = [a + b]\\)

**Subtraction:** \\([a] - [b] = [a - b]\\)

**Multiplication:** \\([a] \cdot [b] = [a \cdot b]\\)

Let us prove well-definedness for multiplication. Suppose \\(a \equiv a' \pmod{n}\\) and \\(b \equiv b' \pmod{n}\\). Then \\(a = a' + sn\\) and \\(b = b' + tn\\) for some integers \\(s, t\\). So:

$$ab = (a' + sn)(b' + tn) = a'b' + a'tn + snb' + stn^2 = a'b' + n(a't + sb' + stn)$$

Therefore \\(ab \equiv a'b' \pmod{n}\\), and multiplication is well-defined on equivalence classes.

### The ring \\(\mathbb{Z}/n\mathbb{Z}\\)

With addition and multiplication defined, \\(\mathbb{Z}/n\mathbb{Z}\\) forms a **commutative ring with unity**. It has an additive identity (\\([0]\\)), a multiplicative identity (\\([1]\\)), addition is commutative and associative, multiplication is commutative and associative, and multiplication distributes over addition. We will make these algebraic definitions precise in the section on groups, rings, and fields.

For now, what matters is this: we have a finite number system where we can add, subtract, and multiply. But can we divide?

### Modular inverses

Division in modular arithmetic means finding a **multiplicative inverse**. The multiplicative inverse of \\([a]\\) in \\(\mathbb{Z}/n\mathbb{Z}\\) is an element \\([b]\\) such that:

$$ab \equiv 1 \pmod{n}$$

Not every element has an inverse. Consider \\(\mathbb{Z}/6\mathbb{Z}\\). The element \\([2]\\) has no inverse: \\(2 \times 0 = 0\\), \\(2 \times 1 = 2\\), \\(2 \times 2 = 4\\), \\(2 \times 3 = 0\\), \\(2 \times 4 = 2\\), \\(2 \times 5 = 4\\). None of these are \\(1 \pmod{6}\\). But \\([5]\\) does have an inverse: \\(5 \times 5 = 25 \equiv 1 \pmod{6}\\), so \\([5]^{-1} = [5]\\).

**Theorem.** \\([a]\\) has a multiplicative inverse in \\(\mathbb{Z}/n\mathbb{Z}\\) if and only if \\(\gcd(a, n) = 1\\).

**Proof.** \\((\Rightarrow)\\) Suppose \\(ab \equiv 1 \pmod{n}\\). Then \\(ab = 1 + kn\\) for some integer \\(k\\), which means \\(ab - kn = 1\\). Any common divisor of \\(a\\) and \\(n\\) must divide the left side, hence must divide \\(1\\). Therefore \\(\gcd(a, n) = 1\\).

\\((\Leftarrow)\\) Suppose \\(\gcd(a, n) = 1\\). By Bézout's identity (which we will prove shortly), there exist integers \\(x, y\\) such that \\(ax + ny = 1\\). Reducing modulo \\(n\\): \\(ax \equiv 1 \pmod{n}\\). So \\([x]\\) is the inverse of \\([a]\\). \\(\square\\)

This theorem is the reason prime moduli are so important in cryptography. If \\(p\\) is prime, then \\(\gcd(a, p) = 1\\) for every \\(a \in \{1, 2, \ldots, p-1\}\\), so every nonzero element of \\(\mathbb{Z}/p\mathbb{Z}\\) has a multiplicative inverse. This makes \\(\mathbb{Z}/p\mathbb{Z}\\) a **field**.

---

## The Extended Euclidean Algorithm

The proof above used Bézout's identity: if \\(\gcd(a, b) = d\\), then there exist integers \\(x, y\\) such that \\(ax + by = d\\). The **Extended Euclidean Algorithm** both computes \\(\gcd(a, b)\\) and finds \\(x, y\\) simultaneously.

### The Euclidean algorithm

The basic Euclidean algorithm computes \\(\gcd(a, b)\\) by repeated division. The key insight is:

$$\gcd(a, b) = \gcd(b, a \bmod b)$$

Why? If \\(d \mid a\\) and \\(d \mid b\\), then \\(d \mid (a - qb) = a \bmod b\\). Conversely, if \\(d \mid b\\) and \\(d \mid (a \bmod b)\\), then \\(d \mid (qb + (a \bmod b)) = a\\). So the set of common divisors is the same, and in particular the greatest one is the same.

We apply this repeatedly until the remainder is zero:

$$a = q_1 b + r_1$$
$$b = q_2 r_1 + r_2$$
$$r_1 = q_3 r_2 + r_3$$
$$\vdots$$
$$r_{k-1} = q_{k+1} r_k + 0$$

The last nonzero remainder \\(r_k\\) is \\(\gcd(a, b)\\).

### Extending the algorithm

The extended version works backward through the chain of divisions to express \\(\gcd(a, b)\\) as a linear combination of \\(a\\) and \\(b\\). Let us work through a concrete example.

**Example:** Find \\(\gcd(240, 46)\\) and express it as \\(240x + 46y\\).

Forward pass (Euclidean algorithm):

$$240 = 5 \times 46 + 10$$
$$46 = 4 \times 10 + 6$$
$$10 = 1 \times 6 + 4$$
$$6 = 1 \times 4 + 2$$
$$4 = 2 \times 2 + 0$$

So \\(\gcd(240, 46) = 2\\).

Backward pass (back-substitution):

$$2 = 6 - 1 \times 4$$
$$= 6 - 1 \times (10 - 1 \times 6) = 2 \times 6 - 1 \times 10$$
$$= 2 \times (46 - 4 \times 10) - 1 \times 10 = 2 \times 46 - 9 \times 10$$
$$= 2 \times 46 - 9 \times (240 - 5 \times 46) = 47 \times 46 - 9 \times 240$$

Check: \\(47 \times 46 - 9 \times 240 = 2162 - 2160 = 2\\). Correct.

So \\(x = -9\\), \\(y = 47\\), and \\(240(-9) + 46(47) = 2\\).

### Python implementation

Here is a clean implementation that returns the gcd, and the Bézout coefficients:

```python
import numpy as np

def extended_gcd(a, b):
    """
    Returns (gcd, x, y) such that a*x + b*y = gcd(a, b).
    Uses the iterative extended Euclidean algorithm.
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t  # gcd, x, y


def mod_inverse(a, n):
    """
    Returns the modular inverse of a modulo n, if it exists.
    Raises ValueError if gcd(a, n) != 1.
    """
    g, x, _ = extended_gcd(a % n, n)
    if g != 1:
        raise ValueError(f"No inverse: gcd({a}, {n}) = {g}")
    return x % n


def mod_exp(base, exp, mod):
    """
    Computes base^exp mod mod using fast exponentiation (square-and-multiply).
    This is O(log exp) multiplications, not O(exp).
    """
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp = exp >> 1
        base = (base * base) % mod
    return result


# Demonstrate
print("Extended GCD examples:")
for a, b in [(240, 46), (35, 15), (17, 3120)]:
    g, x, y = extended_gcd(a, b)
    print(f"  gcd({a}, {b}) = {g},  {a}*({x}) + {b}*({y}) = {a*x + b*y}")

print("\nModular inverses in Z/26Z (the Caesar cipher ring):")
for a in range(1, 26):
    try:
        inv = mod_inverse(a, 26)
        print(f"  {a}^(-1) = {inv}  (check: {a}*{inv} mod 26 = {(a*inv)%26})")
    except ValueError:
        print(f"  {a} has no inverse mod 26 (gcd = {np.gcd(a, 26)})")

print("\nFast modular exponentiation:")
print(f"  7^256 mod 13 = {mod_exp(7, 256, 13)}")
print(f"  2^100 mod 101 = {mod_exp(2, 100, 101)}  (Fermat: should be 1)")
```

The `mod_exp` function deserves attention. Naively computing \\(a^e \bmod n\\) by multiplying \\(a\\) by itself \\(e\\) times requires \\(e\\) multiplications, which is catastrophic when \\(e\\) has hundreds of digits. The **square-and-multiply** method writes \\(e\\) in binary and uses the identity \\(a^{2k} = (a^k)^2\\) to reduce this to \\(O(\log e)\\) multiplications. This is the algorithm that makes RSA and Diffie-Hellman computationally feasible.

---

## Euler's Totient Function and Euler's Theorem

### Euler's totient function

**Definition.** For a positive integer \\(n\\), the **Euler totient function** \\(\phi(n)\\) counts the number of integers in \\(\{1, 2, \ldots, n\}\\) that are coprime to \\(n\\):

$$\phi(n) = \left|\{a \in \{1, 2, \ldots, n\} : \gcd(a, n) = 1\}\right|$$

Equivalently, \\(\phi(n)\\) is the number of invertible elements in \\(\mathbb{Z}/n\mathbb{Z}\\), which is the order of the multiplicative group \\((\mathbb{Z}/n\mathbb{Z})^*\\).

**For a prime \\(p\\):** Every integer from \\(1\\) to \\(p-1\\) is coprime to \\(p\\), so \\(\phi(p) = p - 1\\).

**For a prime power \\(p^k\\):** The integers in \\(\{1, \ldots, p^k\}\\) that are *not* coprime to \\(p^k\\) are exactly the multiples of \\(p\\): there are \\(p^{k-1}\\) of them (namely \\(p, 2p, 3p, \ldots, p^k\\)). So:

$$\phi(p^k) = p^k - p^{k-1} = p^{k-1}(p - 1) = p^k\left(1 - \frac{1}{p}\right)$$

**For a product of coprime integers:** If \\(\gcd(m, n) = 1\\), then \\(\phi(mn) = \phi(m)\phi(n)\\). This is a consequence of the Chinese Remainder Theorem: the ring \\(\mathbb{Z}/mn\mathbb{Z}\\) is isomorphic to \\(\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}\\) when \\(\gcd(m, n) = 1\\), and an element is invertible in the product ring if and only if each component is invertible.

Combining these, for any \\(n\\) with prime factorization \\(n = p_1^{k_1} p_2^{k_2} \cdots p_r^{k_r}\\):

$$\phi(n) = n \prod_{p \mid n} \left(1 - \frac{1}{p}\right) = n \cdot \frac{p_1 - 1}{p_1} \cdot \frac{p_2 - 1}{p_2} \cdots \frac{p_r - 1}{p_r}$$

**Example.** \\(\phi(12) = \phi(2^2 \cdot 3) = 12 \cdot (1 - 1/2)(1 - 1/3) = 12 \cdot 1/2 \cdot 2/3 = 4\\). Indeed, the integers coprime to \\(12\\) in \\(\{1, \ldots, 12\}\\) are \\(\{1, 5, 7, 11\}\\).

### Euler's theorem

**Theorem (Euler).** If \\(\gcd(a, n) = 1\\), then:

$$a^{\phi(n)} \equiv 1 \pmod{n}$$

This is one of the most consequential results in number theory. RSA encryption is essentially a direct application of this theorem.

**Proof.** Let \\((\mathbb{Z}/n\mathbb{Z})^* = \{r_1, r_2, \ldots, r_{\phi(n)}\}\\) be the set of all invertible residues modulo \\(n\\). Since \\(\gcd(a, n) = 1\\), multiplication by \\(a\\) is a bijection on \\((\mathbb{Z}/n\mathbb{Z})^*\\). (If \\(ar_i \equiv ar_j \pmod{n}\\), then since \\(a\\) is invertible, \\(r_i \equiv r_j\\), so multiplication by \\(a\\) is injective; since the set is finite, it is also surjective.)

Therefore, the set \\(\{ar_1, ar_2, \ldots, ar_{\phi(n)}\}\\) is just a rearrangement of \\(\{r_1, r_2, \ldots, r_{\phi(n)}\}\\) modulo \\(n\\). Taking the product of all elements in both sets:

$$\prod_{i=1}^{\phi(n)} (ar_i) \equiv \prod_{i=1}^{\phi(n)} r_i \pmod{n}$$

The left side is \\(a^{\phi(n)} \prod_{i=1}^{\phi(n)} r_i\\). So:

$$a^{\phi(n)} \prod_{i=1}^{\phi(n)} r_i \equiv \prod_{i=1}^{\phi(n)} r_i \pmod{n}$$

Since every \\(r_i\\) is coprime to \\(n\\), their product is also coprime to \\(n\\), so we can cancel it:

$$a^{\phi(n)} \equiv 1 \pmod{n} \qquad \square$$

### Fermat's little theorem

When \\(n = p\\) is prime, \\(\phi(p) = p - 1\\), and Euler's theorem gives **Fermat's little theorem**:

$$a^{p-1} \equiv 1 \pmod{p} \qquad \text{for } \gcd(a, p) = 1$$

Equivalently, \\(a^p \equiv a \pmod{p}\\) for all integers \\(a\\) (including \\(a\\) divisible by \\(p\\), since both sides are then \\(0 \pmod{p}\\)).

Fermat's little theorem gives a quick way to compute modular inverses when the modulus is prime: \\(a^{-1} \equiv a^{p-2} \pmod{p}\\). Combined with fast modular exponentiation, this is often more convenient than the extended Euclidean algorithm.

---

## Groups, Rings, and Fields --- The Algebraic Backbone

The structures we have been building --- \\(\mathbb{Z}/n\mathbb{Z}\\) under addition, \\((\mathbb{Z}/n\mathbb{Z})^*\\) under multiplication --- are instances of fundamental algebraic objects. Modern cryptography is built on the properties of these objects, so let us define them precisely.

### Groups

A **group** is a set \\(G\\) together with a binary operation \\(\cdot\\) satisfying four axioms:

1. **Closure:** For all \\(a, b \in G\\), \\(a \cdot b \in G\\).
2. **Associativity:** For all \\(a, b, c \in G\\), \\((a \cdot b) \cdot c = a \cdot (b \cdot c)\\).
3. **Identity:** There exists \\(e \in G\\) such that \\(e \cdot a = a \cdot e = a\\) for all \\(a \in G\\).
4. **Inverse:** For every \\(a \in G\\), there exists \\(a^{-1} \in G\\) such that \\(a \cdot a^{-1} = a^{-1} \cdot a = e\\).

If additionally \\(a \cdot b = b \cdot a\\) for all \\(a, b\\), the group is **abelian** (or commutative).

**Example: \\(\mathbb{Z}/n\mathbb{Z}\\) under addition.** The set \\(\{0, 1, \ldots, n-1\}\\) with addition modulo \\(n\\) forms an abelian group. The identity is \\(0\\). The inverse of \\(a\\) is \\(n - a\\).

**Example: \\((\mathbb{Z}/n\mathbb{Z})^*\\) under multiplication.** The set of integers in \\(\{1, \ldots, n-1\}\\) coprime to \\(n\\), with multiplication modulo \\(n\\), forms an abelian group. The identity is \\(1\\). Inverses exist by the theorem we proved earlier.

### Order and cyclic groups

The **order** of a group \\(G\\), written \\(|G|\\), is the number of elements in \\(G\\).

The **order** of an element \\(g \in G\\) is the smallest positive integer \\(k\\) such that \\(g^k = e\\). (For additive groups, this means \\(kg = 0\\).)

A group \\(G\\) is **cyclic** if there exists an element \\(g \in G\\) such that every element of \\(G\\) can be written as \\(g^k\\) for some integer \\(k\\). Such a \\(g\\) is called a **generator** of \\(G\\).

**Example:** \\(\mathbb{Z}/n\mathbb{Z}\\) under addition is always cyclic, generated by \\(1\\) (since every element is \\(1 + 1 + \cdots + 1\\) some number of times).

**Key fact:** \\((\mathbb{Z}/p\mathbb{Z})^*\\) is cyclic for every prime \\(p\\). This is a nontrivial theorem (it follows from the fact that \\(\mathbb{Z}/p\mathbb{Z}\\) is a field, and the multiplicative group of any finite field is cyclic). The existence of generators for \\((\mathbb{Z}/p\mathbb{Z})^*\\) is what makes Diffie-Hellman key exchange work.

### Lagrange's theorem

**Theorem (Lagrange).** If \\(H\\) is a subgroup of a finite group \\(G\\), then \\(|H|\\) divides \\(|G|\\).

**Corollary.** The order of every element of \\(G\\) divides \\(|G|\\).

This immediately gives Euler's theorem as a corollary: the element \\(a \in (\mathbb{Z}/n\mathbb{Z})^*\\) has some order \\(k\\) dividing \\(|(\mathbb{Z}/n\mathbb{Z})^*| = \phi(n)\\), so \\(a^{\phi(n)} = (a^k)^{\phi(n)/k} = e^{\phi(n)/k} = 1\\).

### Rings and fields

A **ring** \\((R, +, \cdot)\\) is a set with two operations --- addition and multiplication --- such that \\((R, +)\\) is an abelian group, \\((R, \cdot)\\) is associative with identity \\(1\\), and multiplication distributes over addition.

A **field** is a commutative ring where every nonzero element has a multiplicative inverse. Equivalently, \\((F \setminus \{0\}, \cdot)\\) is an abelian group.

**\\(\mathbb{Z}/n\mathbb{Z}\\) is a ring** for every \\(n\\). **It is a field if and only if \\(n\\) is prime.** This is because every nonzero element has a multiplicative inverse precisely when every element in \\(\{1, \ldots, n-1\}\\) is coprime to \\(n\\), which happens if and only if \\(n\\) is prime.

### Why finite fields matter for cryptography

Finite fields give us the perfect arithmetic environment for cryptographic operations:

- **Every nonzero element is invertible**, so we can "divide" freely. This means encryption operations can always be reversed.
- **The multiplicative group is cyclic**, so exponentiation has a rich, well-understood structure. The **discrete logarithm problem** --- given \\(g\\), \\(h\\), and \\(p\\), find \\(x\\) such that \\(g^x \equiv h \pmod{p}\\) --- is believed to be computationally hard in general, and this hardness is the foundation of Diffie-Hellman and ElGamal.
- **Polynomial arithmetic works cleanly**, which is essential for AES (which operates in \\(\mathrm{GF}(2^8)\\), the field with 256 elements).

---

## Classical Ciphers and Their Destruction

With the algebraic machinery in place, we can now describe the classical ciphers, understand their mathematical structure, and see exactly why they fail.

### The Caesar cipher

The simplest cipher maps each letter to the letter \\(k\\) positions later in the alphabet. Encoding each letter as a number (A=0, B=1, ..., Z=25), encryption is:

$$E(k, m) = (m + k) \bmod 26$$

Decryption is:

$$D(k, c) = (c - k) \bmod 26$$

This is addition in \\(\mathbb{Z}/26\mathbb{Z}\\). The key space has exactly 26 elements (or 25 nontrivial ones, since \\(k = 0\\) is the identity). An attacker can simply try all 25 keys and see which one produces readable text. The Caesar cipher is not a cipher in any meaningful security sense; it is an encoding.

### The affine cipher

A slight generalization: encrypt by applying an affine transformation in \\(\mathbb{Z}/26\mathbb{Z}\\):

$$E((a, b), m) = (am + b) \bmod 26$$

Decryption requires inverting \\(a\\) modulo 26:

$$D((a, b), c) = a^{-1}(c - b) \bmod 26$$

For this to work, \\(a\\) must be coprime to 26. Since \\(26 = 2 \times 13\\), the values of \\(a\\) with \\(\gcd(a, 26) = 1\\) are \\(\{1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25\}\\) --- that is 12 choices. With 26 choices for \\(b\\), the key space has \\(12 \times 26 = 312\\) elements. Still trivially searchable.

### Substitution cipher

A **general substitution cipher** replaces each letter with another letter according to a permutation \\(\pi\\) of the alphabet. The key is the permutation itself, and the key space has \\(26! = 403{,}291{,}461{,}126{,}605{,}635{,}584{,}000{,}000\\) elements --- approximately \\(2^{88}\\).

This is a colossal number. Even checking one billion keys per second, brute force would take about \\(1.3 \times 10^{10}\\) years. So why is the substitution cipher insecure?

### Frequency analysis: why key space size is not security

**The attack does not search the key space. It exploits the structure of the ciphertext.**

In English text, letters do not appear with equal frequency. The letter 'E' appears about 12.7% of the time, 'T' about 9.1%, 'A' about 8.2%, while 'Z' appears only 0.07%. A substitution cipher preserves these frequencies perfectly --- it just relabels them. The most common letter in the ciphertext is likely 'E', the second most common is likely 'T', and so on.

This is the fundamental lesson: **a cipher's security depends not on the size of its key space, but on how much structure the ciphertext reveals about the plaintext.**

### Index of coincidence

To make frequency analysis precise, we define the **index of coincidence** (IC) of a text. Given a text of length \\(N\\) with \\(n_i\\) occurrences of the \\(i\\)-th letter (\\(i = 0, \ldots, 25\\)):

$$\text{IC} = \frac{\sum_{i=0}^{25} n_i(n_i - 1)}{N(N - 1)}$$

This is the probability that two randomly chosen letters from the text are the same. For random text (uniform distribution), \\(\text{IC} \approx 1/26 \approx 0.0385\\). For English text, the IC is approximately \\(0.0667\\), because the non-uniform frequency distribution makes coincidences more likely.

**Derivation.** There are \\(\binom{N}{2} = N(N-1)/2\\) ways to choose two letter positions. The number of pairs where both positions hold the \\(i\\)-th letter is \\(\binom{n_i}{2} = n_i(n_i - 1)/2\\). The total number of matching pairs is \\(\sum_i n_i(n_i - 1)/2\\). Dividing by the total number of pairs:

$$\text{IC} = \frac{\sum_{i=0}^{25} n_i(n_i - 1)/2}{N(N - 1)/2} = \frac{\sum_{i=0}^{25} n_i(n_i - 1)}{N(N - 1)}$$

The IC is invariant under substitution (relabeling letters does not change how many times any particular symbol appears), so it can distinguish monoalphabetic substitution (IC \\(\approx 0.067\\)) from polyalphabetic substitution (IC closer to \\(0.038\\)).

### The Vigenère cipher

The **Vigenère cipher** uses a keyword of length \\(d\\) to create \\(d\\) interleaved Caesar ciphers. If the keyword is \\((k_0, k_1, \ldots, k_{d-1})\\), the \\(i\\)-th character of plaintext is encrypted as:

$$c_i = (m_i + k_{i \bmod d}) \bmod 26$$

This is a **polyalphabetic** cipher: different positions in the text use different substitution alphabets. If the key length \\(d\\) is unknown and the key is long enough, frequency analysis on the full ciphertext fails because the frequencies are smeared out.

For centuries, the Vigenère cipher was called "le chiffre indéchiffrable" (the unbreakable cipher). It is not.

### Kasiski examination

In 1863, Friedrich Kasiski published a method to break the Vigenère cipher. The key observation: if a repeated sequence of plaintext letters happens to be encrypted with the same portion of the keyword, the resulting ciphertext will also repeat. The distance between these repetitions is a multiple of the key length \\(d\\).

**Method:**
1. Find repeated sequences in the ciphertext (trigrams or longer).
2. Record the distances between repetitions.
3. The key length \\(d\\) likely divides the GCD of these distances.
4. Once \\(d\\) is known, split the ciphertext into \\(d\\) groups (positions \\(0, d, 2d, \ldots\\) form one group, positions \\(1, d+1, 2d+1, \ldots\\) form another, etc.).
5. Each group is a simple Caesar cipher. Apply frequency analysis to each group independently.

The IC provides an alternative route to finding \\(d\\): compute the IC for each candidate key length by interleaving. The correct \\(d\\) will produce groups with IC \\(\approx 0.067\\) (English), while incorrect \\(d\\) will produce groups with IC closer to \\(0.038\\) (random).

### SVG: The Vigenère tableau

<svg viewBox="0 0 580 440" xmlns="http://www.w3.org/2000/svg" style="max-width:600px; display:block; margin:auto; background:#1a1a2e; border-radius:8px; padding:10px;">
  <style>
    .label { font: bold 13px monospace; fill: #e0e0e0; }
    .header { font: bold 13px monospace; fill: #e8b84b; }
    .cell { font: 11px monospace; fill: #a0c4ff; }
    .highlight { fill: #ff6b6b; font: bold 11px monospace; }
    .dim { fill: #555; font: 11px monospace; }
    .title { font: bold 15px sans-serif; fill: #e0e0e0; }
    .note { font: 12px sans-serif; fill: #888; }
  </style>

  <text x="290" y="25" text-anchor="middle" class="title">Vigenère Tableau (first 10 rows)</text>

  <!-- Column headers: plaintext letters -->
  <text x="35" y="55" class="note">Plain→</text>
  <text x="100" y="55" class="header">A</text>
  <text x="118" y="55" class="header">B</text>
  <text x="136" y="55" class="header">C</text>
  <text x="154" y="55" class="header">D</text>
  <text x="172" y="55" class="header">E</text>
  <text x="190" y="55" class="header">F</text>
  <text x="208" y="55" class="header">G</text>
  <text x="226" y="55" class="header">H</text>
  <text x="244" y="55" class="header">I</text>
  <text x="262" y="55" class="header">J</text>
  <text x="280" y="55" class="header">K</text>
  <text x="298" y="55" class="header">L</text>
  <text x="316" y="55" class="header">M</text>
  <text x="334" y="55" class="header">N</text>
  <text x="352" y="55" class="header">O</text>
  <text x="370" y="55" class="header">P</text>
  <text x="388" y="55" class="header">Q</text>
  <text x="406" y="55" class="header">R</text>
  <text x="424" y="55" class="header">S</text>
  <text x="442" y="55" class="header">T</text>
  <text x="460" y="55" class="header">U</text>
  <text x="478" y="55" class="header">V</text>
  <text x="496" y="55" class="header">W</text>
  <text x="514" y="55" class="header">X</text>
  <text x="532" y="55" class="header">Y</text>
  <text x="550" y="55" class="header">Z</text>

  <line x1="90" y1="60" x2="560" y2="60" stroke="#444" stroke-width="1"/>

  <!-- Row A (shift 0) -->
  <text x="70" y="78" text-anchor="end" class="header">A</text>
  <text x="100" y="78" class="cell">A</text><text x="118" y="78" class="cell">B</text><text x="136" y="78" class="cell">C</text><text x="154" y="78" class="cell">D</text><text x="172" y="78" class="cell">E</text><text x="190" y="78" class="cell">F</text><text x="208" y="78" class="cell">G</text><text x="226" y="78" class="cell">H</text><text x="244" y="78" class="cell">I</text><text x="262" y="78" class="cell">J</text><text x="280" y="78" class="cell">K</text><text x="298" y="78" class="cell">L</text><text x="316" y="78" class="cell">M</text><text x="334" y="78" class="cell">N</text><text x="352" y="78" class="cell">O</text><text x="370" y="78" class="cell">P</text><text x="388" y="78" class="cell">Q</text><text x="406" y="78" class="cell">R</text><text x="424" y="78" class="cell">S</text><text x="442" y="78" class="cell">T</text><text x="460" y="78" class="cell">U</text><text x="478" y="78" class="cell">V</text><text x="496" y="78" class="cell">W</text><text x="514" y="78" class="cell">X</text><text x="532" y="78" class="cell">Y</text><text x="550" y="78" class="cell">Z</text>

  <!-- Row B (shift 1) -->
  <text x="70" y="96" text-anchor="end" class="header">B</text>
  <text x="100" y="96" class="cell">B</text><text x="118" y="96" class="cell">C</text><text x="136" y="96" class="cell">D</text><text x="154" y="96" class="cell">E</text><text x="172" y="96" class="cell">F</text><text x="190" y="96" class="cell">G</text><text x="208" y="96" class="cell">H</text><text x="226" y="96" class="cell">I</text><text x="244" y="96" class="cell">J</text><text x="262" y="96" class="cell">K</text><text x="280" y="96" class="cell">L</text><text x="298" y="96" class="cell">M</text><text x="316" y="96" class="cell">N</text><text x="334" y="96" class="cell">O</text><text x="352" y="96" class="cell">P</text><text x="370" y="96" class="cell">Q</text><text x="388" y="96" class="cell">R</text><text x="406" y="96" class="cell">S</text><text x="424" y="96" class="cell">T</text><text x="442" y="96" class="cell">U</text><text x="460" y="96" class="cell">V</text><text x="478" y="96" class="cell">W</text><text x="496" y="96" class="cell">X</text><text x="514" y="96" class="cell">Y</text><text x="532" y="96" class="cell">Z</text><text x="550" y="96" class="cell">A</text>

  <!-- Row C (shift 2) -->
  <text x="70" y="114" text-anchor="end" class="header">C</text>
  <text x="100" y="114" class="cell">C</text><text x="118" y="114" class="cell">D</text><text x="136" y="114" class="cell">E</text><text x="154" y="114" class="cell">F</text><text x="172" y="114" class="cell">G</text><text x="190" y="114" class="cell">H</text><text x="208" y="114" class="cell">I</text><text x="226" y="114" class="cell">J</text><text x="244" y="114" class="cell">K</text><text x="262" y="114" class="cell">L</text><text x="280" y="114" class="cell">M</text><text x="298" y="114" class="cell">N</text><text x="316" y="114" class="cell">O</text><text x="334" y="114" class="cell">P</text><text x="352" y="114" class="cell">Q</text><text x="370" y="114" class="cell">R</text><text x="388" y="114" class="cell">S</text><text x="406" y="114" class="cell">T</text><text x="424" y="114" class="cell">U</text><text x="442" y="114" class="cell">V</text><text x="460" y="114" class="cell">W</text><text x="478" y="114" class="cell">X</text><text x="496" y="114" class="cell">Y</text><text x="514" y="114" class="cell">Z</text><text x="532" y="114" class="cell">A</text><text x="550" y="114" class="cell">B</text>

  <!-- Row D (shift 3) -->
  <text x="70" y="132" text-anchor="end" class="header">D</text>
  <text x="100" y="132" class="cell">D</text><text x="118" y="132" class="cell">E</text><text x="136" y="132" class="cell">F</text><text x="154" y="132" class="cell">G</text><text x="172" y="132" class="cell">H</text><text x="190" y="132" class="cell">I</text><text x="208" y="132" class="cell">J</text><text x="226" y="132" class="cell">K</text><text x="244" y="132" class="cell">L</text><text x="262" y="132" class="cell">M</text><text x="280" y="132" class="cell">N</text><text x="298" y="132" class="cell">O</text><text x="316" y="132" class="cell">P</text><text x="334" y="132" class="cell">Q</text><text x="352" y="132" class="cell">R</text><text x="370" y="132" class="cell">S</text><text x="388" y="132" class="cell">T</text><text x="406" y="132" class="cell">U</text><text x="424" y="132" class="cell">V</text><text x="442" y="132" class="cell">W</text><text x="460" y="132" class="cell">X</text><text x="478" y="132" class="cell">Y</text><text x="496" y="132" class="cell">Z</text><text x="514" y="132" class="cell">A</text><text x="532" y="132" class="cell">B</text><text x="550" y="132" class="cell">C</text>

  <!-- Row E (shift 4) -->
  <text x="70" y="150" text-anchor="end" class="header">E</text>
  <text x="100" y="150" class="cell">E</text><text x="118" y="150" class="cell">F</text><text x="136" y="150" class="cell">G</text><text x="154" y="150" class="cell">H</text><text x="172" y="150" class="highlight">I</text><text x="190" y="150" class="cell">J</text><text x="208" y="150" class="cell">K</text><text x="226" y="150" class="cell">L</text><text x="244" y="150" class="cell">M</text><text x="262" y="150" class="cell">N</text><text x="280" y="150" class="cell">O</text><text x="298" y="150" class="cell">P</text><text x="316" y="150" class="cell">Q</text><text x="334" y="150" class="cell">R</text><text x="352" y="150" class="cell">S</text><text x="370" y="150" class="cell">T</text><text x="388" y="150" class="cell">U</text><text x="406" y="150" class="cell">V</text><text x="424" y="150" class="cell">W</text><text x="442" y="150" class="cell">X</text><text x="460" y="150" class="cell">Y</text><text x="478" y="150" class="cell">Z</text><text x="496" y="150" class="cell">A</text><text x="514" y="150" class="cell">B</text><text x="532" y="150" class="cell">C</text><text x="550" y="150" class="cell">D</text>

  <!-- Row F through J (abbreviated with dots pattern) -->
  <text x="70" y="168" text-anchor="end" class="header">F</text>
  <text x="100" y="168" class="dim">F</text><text x="118" y="168" class="dim">G</text><text x="136" y="168" class="dim">H</text><text x="154" y="168" class="dim">I</text><text x="172" y="168" class="dim">J</text><text x="190" y="168" class="dim">K</text><text x="208" y="168" class="dim">L</text><text x="226" y="168" class="dim">M</text><text x="244" y="168" class="dim">N</text><text x="262" y="168" class="dim">O</text><text x="280" y="168" class="dim">P</text><text x="298" y="168" class="dim">Q</text><text x="316" y="168" class="dim">R</text><text x="334" y="168" class="dim">S</text><text x="352" y="168" class="dim">T</text><text x="370" y="168" class="dim">U</text><text x="388" y="168" class="dim">V</text><text x="406" y="168" class="dim">W</text><text x="424" y="168" class="dim">X</text><text x="442" y="168" class="dim">Y</text><text x="460" y="168" class="dim">Z</text><text x="478" y="168" class="dim">A</text><text x="496" y="168" class="dim">B</text><text x="514" y="168" class="dim">C</text><text x="532" y="168" class="dim">D</text><text x="550" y="168" class="dim">E</text>

  <!-- Ellipsis rows -->
  <text x="70" y="195" text-anchor="end" class="dim">⋮</text>
  <text x="316" y="195" class="dim">⋮</text>

  <text x="70" y="218" text-anchor="end" class="header">Z</text>
  <text x="100" y="218" class="cell">Z</text><text x="118" y="218" class="cell">A</text><text x="136" y="218" class="cell">B</text><text x="154" y="218" class="cell">C</text><text x="172" y="218" class="cell">D</text><text x="190" y="218" class="cell">E</text><text x="208" y="218" class="cell">F</text><text x="226" y="218" class="cell">G</text><text x="244" y="218" class="cell">H</text><text x="262" y="218" class="cell">I</text><text x="280" y="218" class="cell">J</text><text x="298" y="218" class="cell">K</text><text x="316" y="218" class="cell">L</text><text x="334" y="218" class="cell">M</text><text x="352" y="218" class="cell">N</text><text x="370" y="218" class="cell">O</text><text x="388" y="218" class="cell">P</text><text x="406" y="218" class="cell">Q</text><text x="424" y="218" class="cell">R</text><text x="442" y="218" class="cell">S</text><text x="460" y="218" class="cell">T</text><text x="478" y="218" class="cell">U</text><text x="496" y="218" class="cell">V</text><text x="514" y="218" class="cell">W</text><text x="532" y="218" class="cell">X</text><text x="550" y="218" class="cell">Y</text>

  <!-- Key arrow and annotation -->
  <text x="30" y="250" class="note">Key↓</text>
  <line x1="80" y1="60" x2="80" y2="225" stroke="#e8b84b" stroke-width="1.5" stroke-dasharray="4,3"/>

  <!-- Highlighted cell annotation -->
  <rect x="165" y="139" width="16" height="16" fill="none" stroke="#ff6b6b" stroke-width="2" rx="2"/>
  <line x1="182" y1="147" x2="250" y2="260" stroke="#ff6b6b" stroke-width="1" stroke-dasharray="3,2"/>
  <text x="255" y="265" class="highlight">Key=E, Plain=E → Cipher=I</text>

  <!-- Description -->
  <text x="100" y="310" class="note">Each row is the alphabet shifted by the key letter's position.</text>
  <text x="100" y="328" class="note">Row A = shift 0, Row B = shift 1, ..., Row Z = shift 25.</text>
  <text x="100" y="346" class="note">The keyword cycles through rows for successive plaintext letters.</text>

  <!-- Example -->
  <text x="100" y="380" class="label">Example: key = "KEY", plaintext = "HELLO"</text>
  <text x="100" y="400" class="cell">H+K=R,  E+E=I,  L+Y=J,  L+K=V,  O+E=S</text>
  <text x="100" y="420" class="cell">Ciphertext: RIJVS</text>
</svg>

### Python: frequency analysis and IC computation

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Standard English letter frequencies (from large corpora)
ENGLISH_FREQ = np.array([
    0.0817, 0.0150, 0.0278, 0.0425, 0.1270, 0.0223,  # A-F
    0.0202, 0.0609, 0.0697, 0.0015, 0.0077, 0.0403,  # G-L
    0.0241, 0.0675, 0.0751, 0.0193, 0.0010, 0.0599,  # M-R
    0.0633, 0.0906, 0.0276, 0.0098, 0.0236, 0.0015,  # S-X
    0.0197, 0.0007                                      # Y-Z
])

def letter_frequencies(text):
    """Compute letter frequency distribution of a text (letters only, case-insensitive)."""
    text = ''.join(c.upper() for c in text if c.isalpha())
    counts = np.zeros(26)
    for c in text:
        counts[ord(c) - ord('A')] += 1
    total = counts.sum()
    if total == 0:
        return counts
    return counts / total

def index_of_coincidence(text):
    """Compute the index of coincidence of a text."""
    text = ''.join(c.upper() for c in text if c.isalpha())
    N = len(text)
    if N < 2:
        return 0.0
    counts = np.zeros(26)
    for c in text:
        counts[ord(c) - ord('A')] += 1
    return np.sum(counts * (counts - 1)) / (N * (N - 1))

def vigenere_encrypt(plaintext, key):
    """Encrypt plaintext with Vigenère cipher."""
    plaintext = ''.join(c.upper() for c in plaintext if c.isalpha())
    key = key.upper()
    cipher = []
    for i, ch in enumerate(plaintext):
        shift = ord(key[i % len(key)]) - ord('A')
        cipher.append(chr((ord(ch) - ord('A') + shift) % 26 + ord('A')))
    return ''.join(cipher)

def estimate_key_length(ciphertext, max_len=20):
    """Estimate Vigenère key length using IC for each candidate."""
    ciphertext = ''.join(c.upper() for c in ciphertext if c.isalpha())
    ic_values = []
    for d in range(1, max_len + 1):
        groups = ['' for _ in range(d)]
        for i, ch in enumerate(ciphertext):
            groups[i % d] += ch
        avg_ic = np.mean([index_of_coincidence(g) for g in groups])
        ic_values.append(avg_ic)
    return ic_values

# --- Demonstration ---
# A sample English text (opening of the Declaration of Independence)
sample_text = (
    "When in the Course of human events it becomes necessary for one people "
    "to dissolve the political bands which have connected them with another "
    "and to assume among the powers of the earth the separate and equal "
    "station to which the Laws of Nature and of Natures God entitle them "
    "a decent respect to the opinions of mankind requires that they should "
    "declare the causes which impel them to the separation"
)

# Encrypt with Vigenère, key = "SWEDEN"
key = "SWEDEN"
ciphertext = vigenere_encrypt(sample_text, key)
print(f"Plaintext IC:   {index_of_coincidence(sample_text):.4f}  (English ≈ 0.0667)")
print(f"Ciphertext IC:  {index_of_coincidence(ciphertext):.4f}  (closer to random ≈ 0.0385)")

# Estimate key length
ic_values = estimate_key_length(ciphertext, max_len=15)
print(f"\nIC by candidate key length:")
for d, ic in enumerate(ic_values, 1):
    marker = " ◄── peak" if d == len(key) else ""
    print(f"  d={d:2d}: IC = {ic:.4f}{marker}")

# --- Plot: English frequencies vs ciphertext frequencies ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

letters = [chr(i + ord('A')) for i in range(26)]

# English frequencies
axes[0].bar(letters, ENGLISH_FREQ, color='#3498db', alpha=0.8, edgecolor='#2c3e50')
axes[0].set_xlabel(r'Letter', fontsize=12)
axes[0].set_ylabel(r'Frequency', fontsize=12)
axes[0].set_title(r'English Letter Frequencies', fontsize=13)
axes[0].set_ylim(0, 0.15)
axes[0].axhline(y=1/26, color='#e74c3c', linestyle='--', linewidth=1, label=r'Uniform $1/26$')
axes[0].legend(fontsize=10)

# Ciphertext frequencies
cipher_freq = letter_frequencies(ciphertext)
axes[1].bar(letters, cipher_freq, color='#e74c3c', alpha=0.8, edgecolor='#2c3e50')
axes[1].set_xlabel(r'Letter', fontsize=12)
axes[1].set_ylabel(r'Frequency', fontsize=12)
axes[1].set_title(r'Vigenère Ciphertext Frequencies (key="SWEDEN")', fontsize=13)
axes[1].set_ylim(0, 0.15)
axes[1].axhline(y=1/26, color='#3498db', linestyle='--', linewidth=1, label=r'Uniform $1/26$')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('frequency_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# --- Plot: IC vs candidate key length ---
fig, ax = plt.subplots(figsize=(10, 5))
d_values = np.arange(1, len(ic_values) + 1)
ax.plot(d_values, ic_values, 'o-', color='#2ecc71', linewidth=2, markersize=8)
ax.axhline(y=0.0667, color='#3498db', linestyle='--', linewidth=1.5,
           label=r'English IC $\approx 0.0667$')
ax.axhline(y=1/26, color='#e74c3c', linestyle='--', linewidth=1.5,
           label=r'Random IC $\approx 1/26$')
ax.set_xlabel(r'Candidate key length $d$', fontsize=13)
ax.set_ylabel(r'Average IC of $d$ groups', fontsize=13)
ax.set_title(r'Key Length Estimation via Index of Coincidence', fontsize=14)
ax.set_xticks(d_values)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Annotate the peak
peak_d = np.argmax(ic_values) + 1
peak_ic = ic_values[peak_d - 1]
ax.annotate(rf'$d = {peak_d}$ (correct key length)',
            xy=(peak_d, peak_ic), xytext=(peak_d + 2, peak_ic + 0.005),
            arrowprops=dict(arrowstyle='->', color='#e8b84b', lw=2),
            fontsize=12, color='#e8b84b', fontweight='bold')

plt.tight_layout()
plt.savefig('ic_key_length.png', dpi=150, bbox_inches='tight')
plt.show()
```

### The Enigma machine: permutations as ciphers

The most famous cipher machine in history, the German Enigma, is mathematically a product of permutations. Each rotor implements a permutation \\(\sigma_i\\) of the 26-letter alphabet. The reflector implements a fixed-point-free involution \\(R\\) (a permutation with \\(R^2 = \text{id}\\) and no fixed points). The plugboard implements another involution \\(P\\).

The encryption of a single character, with rotor positions determining the specific permutations, is:

$$E = P \circ \sigma_3 \circ \sigma_2 \circ \sigma_1 \circ R \circ \sigma_1^{-1} \circ \sigma_2^{-1} \circ \sigma_3^{-1} \circ P$$

The key mathematical features:

- **The reflector makes Enigma self-reciprocal:** encrypting twice with the same settings gives back the plaintext. This was a design feature (the same machine settings could encrypt and decrypt), but it was also a fatal cryptographic weakness: no letter can ever encrypt to itself. This property --- that \\(E(x) \neq x\\) for all \\(x\\) --- eliminated a large fraction of possible rotor positions during cryptanalysis.

- **The rotors step mechanically**, so each keypress changes the permutation. This makes Enigma a polyalphabetic cipher with a period determined by the rotor stepping mechanism (\\(26^3 = 17{,}576\\) for the basic three-rotor machine, though the double-stepping anomaly complicates this slightly).

The breaking of Enigma --- by Polish mathematicians Marian Rejewski, Jerzy Różycki, and Henryk Zygalski, and later by Alan Turing and the team at Bletchley Park --- exploited these algebraic properties together with known-plaintext attacks (the "cribs"), particularly the no-self-encryption property. The mathematical analysis of permutation groups was literally the tool that broke Enigma.

---

## Shannon's Perfect Secrecy

In 1949, Claude Shannon published "Communication Theory of Secrecy Systems," which transformed cryptography from an art into a science. He asked: what does it mean, mathematically, for a cipher to be perfectly secure? And what does perfect security cost?

### Formal definition of a cryptosystem

A **cryptosystem** is a tuple \\((M, K, C, E, D)\\) where:
- \\(M\\) is the **message space** (set of possible plaintexts)
- \\(K\\) is the **key space** (set of possible keys)
- \\(C\\) is the **ciphertext space** (set of possible ciphertexts)
- \\(E: K \times M \to C\\) is the **encryption function**
- \\(D: K \times C \to M\\) is the **decryption function**

with the correctness requirement: for every key \\(k \in K\\) and message \\(m \in M\\):

$$D(k, E(k, m)) = m$$

We also need probability distributions. The plaintext \\(m\\) is drawn from some distribution over \\(M\\), and the key \\(k\\) is drawn independently from some distribution over \\(K\\).

### Perfect secrecy

**Definition.** A cryptosystem has **perfect secrecy** if for every message \\(m \in M\\) and every ciphertext \\(c \in C\\) with \\(P(C = c) > 0\\):

$$P(M = m \mid C = c) = P(M = m)$$

In words: observing the ciphertext gives the adversary *zero information* about which message was sent. The posterior probability of any message, given the ciphertext, equals the prior probability. The ciphertext is statistically independent of the plaintext.

This is the strongest possible security notion. An adversary with unlimited computational power, unlimited time, and access to the ciphertext still cannot do better than random guessing.

### Shannon's theorem: the cost of perfect secrecy

**Theorem (Shannon).** If a cryptosystem has perfect secrecy, then \\(|K| \geq |M|\\).

The key space must be at least as large as the message space. This is a devastating result: it means that for perfect secrecy, you need a key at least as long as the message.

**Proof.** We prove the contrapositive: if \\(|K| < |M|\\), there exists a ciphertext \\(c\\) and a message \\(m_0\\) such that \\(P(M = m_0 \mid C = c) \neq P(M = m_0)\\).

Fix a ciphertext \\(c\\) with \\(P(C = c) > 0\\). For each key \\(k \in K\\), the decryption \\(D(k, c)\\) produces at most one message. So the set of messages that can produce \\(c\\) (under any key) is:

$$M(c) = \{D(k, c) : k \in K\}$$

Since \\(|M(c)| \leq |K| < |M|\\), there exists a message \\(m_0 \in M \setminus M(c)\\). This message \\(m_0\\) cannot possibly produce ciphertext \\(c\\) under any key. Therefore:

$$P(C = c \mid M = m_0) = 0$$

By Bayes' theorem:

$$P(M = m_0 \mid C = c) = \frac{P(C = c \mid M = m_0) \cdot P(M = m_0)}{P(C = c)} = \frac{0 \cdot P(M = m_0)}{P(C = c)} = 0$$

But if \\(P(M = m_0) > 0\\) (which it is, since \\(m_0\\) is a valid message), then:

$$P(M = m_0 \mid C = c) = 0 \neq P(M = m_0)$$

This violates the perfect secrecy condition. \\(\square\\)

### The one-time pad

The **one-time pad** (OTP), also known as the Vernam cipher, achieves perfect secrecy. The construction is simple.

Let \\(M = K = C = \{0, 1\}^n\\) (binary strings of length \\(n\\)). The key \\(k\\) is chosen uniformly at random from \\(\{0, 1\}^n\\). Encryption and decryption are both the XOR operation:

$$E(k, m) = m \oplus k$$
$$D(k, c) = c \oplus k$$

Correctness: \\(D(k, E(k, m)) = (m \oplus k) \oplus k = m \oplus (k \oplus k) = m \oplus \mathbf{0} = m\\). (XOR is associative, and \\(k \oplus k = \mathbf{0}\\).)

**Theorem.** The one-time pad has perfect secrecy.

**Proof.** For any message \\(m\\) and ciphertext \\(c\\), there is exactly one key that maps \\(m\\) to \\(c\\), namely \\(k = m \oplus c\\). Since \\(k\\) is chosen uniformly:

$$P(C = c \mid M = m) = P(K = m \oplus c) = \frac{1}{2^n}$$

This does not depend on \\(m\\). So by Bayes' theorem:

$$P(M = m \mid C = c) = \frac{P(C = c \mid M = m) \cdot P(M = m)}{P(C = c)}$$

We compute \\(P(C = c)\\) by marginalizing over all messages:

$$P(C = c) = \sum_{m' \in M} P(C = c \mid M = m') \cdot P(M = m') = \frac{1}{2^n} \sum_{m' \in M} P(M = m') = \frac{1}{2^n}$$

Therefore:

$$P(M = m \mid C = c) = \frac{(1/2^n) \cdot P(M = m)}{1/2^n} = P(M = m) \qquad \square$$

The one-time pad is perfectly secure regardless of the adversary's computational power. The ciphertext is literally independent of the plaintext.

### Why the one-time pad is impractical

Perfect secrecy comes at a price:

1. **The key must be as long as the message.** Shannon's theorem tells us this is unavoidable for perfect secrecy.
2. **The key must never be reused.** If two messages \\(m_1\\) and \\(m_2\\) are encrypted with the same key \\(k\\), then \\(c_1 \oplus c_2 = (m_1 \oplus k) \oplus (m_2 \oplus k) = m_1 \oplus m_2\\). The adversary obtains the XOR of the two plaintexts, which leaks substantial information (for instance, common words can be identified through statistical analysis of \\(m_1 \oplus m_2\\)).
3. **Key distribution is as hard as message distribution.** If you have a secure channel to transmit the key, you could have used that channel to transmit the message directly.

### The VENONA project: the cost of reuse

The most dramatic real-world demonstration of why one-time pads must never be reused is the VENONA project. During World War II, Soviet intelligence used one-time pads for communications between Moscow and its embassies and agents abroad. Due to wartime key distribution difficulties, the Soviet cryptographic agency (the "Fifth Directorate") duplicated some one-time pad pages and sent them to different offices.

American cryptanalysts at the U.S. Army's Signal Intelligence Service (later NSA) discovered these reuses. By XORing pairs of ciphertexts encrypted with the same pad pages, they obtained the XOR of the plaintexts. Combined with known plaintext attacks (standard message headers, predictable phrases in diplomatic traffic), they were able to decrypt thousands of Soviet intelligence messages between 1943 and 1980.

The VENONA decrypts revealed, among other things, the existence of Soviet espionage rings in the Manhattan Project and the British government. A mathematically perfect cipher was defeated because the implementation violated its one critical assumption: each key is used exactly once.

### Unicity distance

If a cipher does not achieve perfect secrecy (which, by Shannon's theorem, means any cipher with \\(|K| < |M|\\)), how much ciphertext does an adversary need before the key can be uniquely determined?

**Definition.** The **unicity distance** \\(U\\) of a cipher is the minimum amount of ciphertext needed so that, on average, there is only one key consistent with the ciphertext and plausible plaintext.

To derive this, we need the concept of **spurious keys**. Given a ciphertext \\(c\\) of length \\(n\\), a key \\(k\\) is **spurious** if \\(D(k, c)\\) produces a valid (meaningful) plaintext but \\(k\\) is not the actual key used. As \\(n\\) grows, the number of spurious keys decreases because it becomes increasingly unlikely that a wrong key produces meaningful text.

Let \\(H(K) = \log_2 |K|\\) be the entropy of the key space (assuming uniform key distribution), and let \\(D\\) be the **redundancy** of the plaintext language, defined as:

$$D = \log_2 |\Sigma| - H_L$$

where \\(|\Sigma|\\) is the alphabet size and \\(H_L\\) is the per-character entropy of the language. For English, \\(|\Sigma| = 26\\), \\(\log_2 26 \approx 4.7\\) bits, and the per-character entropy is approximately \\(H_L \approx 1.0\\) to \\(1.5\\) bits (depending on the model of English used). So the redundancy is roughly \\(D \approx 3.2\\) to \\(3.7\\) bits per character.

Shannon showed that the expected number of spurious keys for ciphertext of length \\(n\\) is approximately:

$$\bar{s}(n) \approx 2^{H(K) - nD} - 1$$

Setting \\(\bar{s}(n) = 0\\) (one unique key remains), we get the unicity distance:

$$U = \frac{H(K)}{D}$$

**Example:** For a simple substitution cipher on English text, \\(H(K) = \log_2(26!) \approx 88\\) bits. With \\(D \approx 3.5\\) bits per character:

$$U \approx \frac{88}{3.5} \approx 25 \text{ characters}$$

So with about 25 characters of ciphertext, there is typically only one key that produces meaningful English. This is remarkably little --- it means that even a cipher with \\(2^{88}\\) keys can be uniquely broken from a short paragraph.

For the one-time pad, \\(H(K) = n \log_2 |\Sigma|\\), which grows with the message length, so \\(U \to \infty\\). The unicity distance is infinite: no amount of ciphertext suffices to uniquely determine the key. This is another way of stating perfect secrecy.

### Connection to information theory

The unicity distance connects cryptography to Shannon's information theory in a deep way. The redundancy \\(D\\) measures how much "extra" information the plaintext language carries beyond the raw content. English is highly redundant --- you can remove many letters from a sentence and still read it. This redundancy is what allows the adversary to distinguish correct decryptions from incorrect ones.

If the plaintext had zero redundancy (every bit string was equally likely as a message), then every key would produce a "valid" plaintext, and no amount of ciphertext would help. This is exactly the one-time pad scenario: when \\(|K| = |M|\\) and the key is uniform, the plaintext distribution is maximally entropic from the adversary's perspective.

---

## From Classical to Modern --- Why We Need More Mathematics

We have arrived at a fundamental tension in cryptography:

**Perfect secrecy requires keys as long as messages.** This is Shannon's theorem, and it is a mathematical fact, not a technological limitation. No future computer, no clever algorithm, no quantum breakthrough will change it.

**Practical communication requires short keys and long messages.** A 256-bit key must secure gigabytes of data. The key must be transmitted once over a secure channel and then used to protect months or years of communication.

This means we must abandon perfect secrecy and settle for something weaker. But "weaker" does not mean "weak." The brilliant insight of modern cryptography is to replace **information-theoretic security** (security against adversaries with unlimited computation) with **computational security** (security against adversaries with bounded computation).

The distinction is this:

- **Information-theoretic security:** The adversary *cannot* break the cipher, no matter how much computation they perform. The one-time pad achieves this.
- **Computational security:** The adversary *could* break the cipher in principle (the information is there in the ciphertext), but doing so requires more computation than is physically feasible. Every modern cipher --- AES, RSA, elliptic curves --- operates in this regime.

Computational security rests on the existence of **one-way functions**: functions that are easy to compute but hard to invert. Multiplying two large primes is easy; factoring the product is (believed to be) hard. Computing \\(g^x \bmod p\\) is easy; finding \\(x\\) given \\(g^x\\) is (believed to be) hard. The mathematical structures that give rise to one-way functions --- prime factorization, the discrete logarithm problem, elliptic curve groups, lattice problems --- are the subject of the rest of this series.

In [Part 2](/2026/03/13/symmetric-cryptography-aes-block-ciphers.html), we will see how AES uses the finite field \\(\mathrm{GF}(2^8)\\) and carefully designed permutations to build a cipher that is computationally secure with a 128-bit key. In [Part 3](/2026/03/14/asymmetric-cryptography-rsa-elliptic-curves.html), we will see how RSA converts Euler's theorem into a public-key cryptosystem, and how elliptic curves provide the same security with smaller keys. In [Part 4](/2026/03/15/cryptographic-protocols-tls-signal-national-security.html), we will see how these primitives compose into protocols like TLS and Signal that protect real-world communications --- the kind of communications that MUST is mandated to secure. And in [Part 5](/2026/03/16/quantum-threat-post-quantum-cryptography.html), we will confront the quantum threat: Shor's algorithm breaks RSA and Diffie-Hellman, and the post-quantum replacements are built on entirely different mathematical structures (lattices, codes, isogenies) that we will develop from scratch.

The mathematics of secrecy is deep, and it starts here, with the arithmetic that makes clocks tick and letters shift.
