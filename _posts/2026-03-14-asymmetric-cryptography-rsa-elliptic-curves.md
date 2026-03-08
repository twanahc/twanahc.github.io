---
layout: post
title: "Asymmetric Cryptography: RSA, Diffie-Hellman, and Elliptic Curves"
date: 2026-03-14
category: math
---

*This is Part 3 of a 5-part series on cryptology. [Part 1: Number Theory & Classical Ciphers](/2026/03/12/number-theory-classical-cryptography.html) | [Part 2: Symmetric Cryptography](/2026/03/13/symmetric-cryptography-aes-block-ciphers.html) | **Part 3: Asymmetric Cryptography** | [Part 4: Cryptographic Protocols](/2026/03/15/cryptographic-protocols-tls-signal-national-security.html) | [Part 5: Post-Quantum Cryptography](/2026/03/16/quantum-threat-post-quantum-cryptography.html)*

Here is the problem. Sweden joined NATO in 2024. Swedish military planners need to exchange classified operational data with allied counterparts in Oslo, Tallinn, and Washington. Every packet crosses the Baltic Sea --- a body of water where Russian intelligence operates sophisticated SIGINT capabilities, tapping undersea cables and intercepting satellite uplinks. MUST (Sweden's military intelligence agency) coordinates with FRA and with NATO partners to ensure secure communications. The data must be encrypted.

In [Part 2](/2026/03/13/symmetric-cryptography-aes-block-ciphers.html), we built AES --- a cipher that can encrypt anything, provided both sides share a secret key. But that proviso is the entire problem. How do Stockholm and Tallinn agree on a 256-bit AES key when the channel between them is compromised? You cannot send the key over the same channel you are trying to secure. That is circular. You cannot fly a courier to Tallinn every time you need a new key --- not at the speed modern operations demand, and not when you need to establish secure channels with dozens of partners simultaneously.

This is the **key distribution problem**, and solving it required one of the most beautiful ideas in the history of mathematics: public-key cryptography.

---

## Table of Contents

1. [The Key Distribution Problem](#the-key-distribution-problem)
2. [One-Way Functions and Trapdoors](#one-way-functions-and-trapdoors)
3. [Diffie-Hellman Key Exchange](#diffie-hellman-key-exchange)
4. [RSA — From Euler's Theorem to Public-Key Encryption](#rsa--from-eulers-theorem-to-public-key-encryption)
5. [ElGamal Encryption](#elgamal-encryption)
6. [Elliptic Curve Cryptography — Shorter Keys, Same Security](#elliptic-curve-cryptography--shorter-keys-same-security)
7. [Key Size Comparison and the March of Computation](#key-size-comparison-and-the-march-of-computation)

---

## The Key Distribution Problem

Symmetric cryptography has a fatal logistics problem. If Alice wants to communicate securely with Bob, they need a shared secret key. If there are \\(n\\) parties in a network (say, \\(n\\) NATO member states), and every pair needs a unique shared key, the total number of keys required is:

$$
\binom{n}{2} = \frac{n(n-1)}{2}
$$

For NATO's 32 member states, that is \\(\frac{32 \cdot 31}{2} = 496\\) unique keys. Add in bilateral intelligence partnerships, joint task forces, and interoperable command systems, and the number explodes. Each key must be generated, transported securely, stored, rotated, and revoked. This is a logistical nightmare.

Historically, the solution was **key couriers** --- trusted individuals who physically carried keys in tamper-evident containers. During the Cold War, embassy staff would transport codebooks in diplomatic pouches. This works, but it is slow, expensive, and scales terribly. If a courier is compromised, every key they carried is burned.

The alternative was **pre-shared keys** --- distributing keys in advance and storing them for future use. Military key management systems like the US KDN (Key Distribution Network) pre-loaded encryption devices with key material. But pre-shared keys have a fixed lifetime, cannot handle ad-hoc communications, and create an enormous target for adversaries.

The conceptual breakthrough came in the 1970s. What if two parties could establish a shared secret over a completely public channel? What if eavesdropping on every single message exchanged between them still left the adversary unable to determine the shared key?

This sounds impossible. It is not. It requires a mathematical structure called a **one-way function**.

---

## One-Way Functions and Trapdoors

A **one-way function** is a function \\(f\\) that is easy to compute but hard to invert. More precisely:

- **Easy to compute:** Given \\(x\\), computing \\(f(x)\\) takes polynomial time.
- **Hard to invert:** Given \\(y = f(x)\\), finding any \\(x'\\) such that \\(f(x') = y\\) takes superpolynomial time (no efficient algorithm exists).

"Easy" and "hard" here refer to computational complexity. A polynomial-time algorithm runs in \\(O(n^k)\\) time for some fixed \\(k\\), where \\(n\\) is the input size. A superpolynomial algorithm --- like one running in \\(O(2^n)\\) --- becomes infeasible as \\(n\\) grows. A function you can evaluate in microseconds but that requires centuries to invert is exactly what cryptography needs.

A **trapdoor one-way function** adds one more property: there exists a piece of secret information (the "trapdoor") that makes inversion easy. Without the trapdoor, inversion is hard. With it, inversion is efficient. The trapdoor is the private key.

The two most important candidate one-way functions in cryptography are:

1. **Integer factorization:** Given two large primes \\(p\\) and \\(q\\), computing \\(n = pq\\) is trivial (one multiplication). Given \\(n\\), finding \\(p\\) and \\(q\\) is believed to be hard. The best known classical algorithm (the General Number Field Sieve) runs in subexponential time \\(L_n[1/3, c]\\), which for 2048-bit \\(n\\) means roughly \\(2^{112}\\) operations.

2. **Discrete logarithm:** Given a prime \\(p\\), a generator \\(g\\) of \\(\mathbb{Z}/p\mathbb{Z}^*\\), and \\(h = g^x \bmod p\\), finding \\(x\\) is believed to be hard. The best classical algorithm (Index Calculus) also runs in subexponential time.

A note on complexity classes. The class **P** contains problems solvable in polynomial time. The class **NP** contains problems whose solutions can be *verified* in polynomial time. Whether \\(\mathbf{P} = \mathbf{NP}\\) remains the most important open question in computer science. Neither factoring nor discrete logarithm is known to be NP-complete --- they sit in a gray zone, believed to be hard but not provably so. All of modern public-key cryptography rests on this *assumption* of hardness. If someone proved \\(\mathbf{P} = \mathbf{NP}\\), the entire edifice would collapse.

This is the assumption landscape of cryptography: we do not have proofs of security, only confidence built from decades of failed attacks.

---

## Diffie-Hellman Key Exchange

In 1976, Whitfield Diffie and Martin Hellman published "New Directions in Cryptography," arguably the most consequential paper in the history of information security. They described a protocol that allows two parties to establish a shared secret over a public channel.

### The Discrete Logarithm Problem

Let \\(p\\) be a large prime and \\(g\\) a **generator** (or primitive root) of the multiplicative group \\(\mathbb{Z}/p\mathbb{Z}^*\\). This means every element of \\(\{1, 2, \ldots, p-1\}\\) can be written as \\(g^k \bmod p\\) for some integer \\(k\\). The group has order \\(p - 1\\).

The **Discrete Logarithm Problem (DLP)** is: given \\(p\\), \\(g\\), and \\(h = g^x \bmod p\\), find \\(x\\).

Computing \\(g^x \bmod p\\) is easy --- **modular exponentiation** via repeated squaring runs in \\(O(\log x)\\) multiplications modulo \\(p\\). But inverting this --- computing the discrete logarithm --- has no known efficient classical algorithm for general primes.

### Protocol Description

Alice and Bob want to establish a shared secret. They publicly agree on a prime \\(p\\) and generator \\(g\\). Then:

1. **Alice** picks a random secret \\(a \in \{2, \ldots, p-2\}\\) and computes \\(A = g^a \bmod p\\). She sends \\(A\\) to Bob.
2. **Bob** picks a random secret \\(b \in \{2, \ldots, p-2\}\\) and computes \\(B = g^b \bmod p\\). He sends \\(B\\) to Alice.
3. **Alice** computes the shared secret: \\(s = B^a \bmod p\\).
4. **Bob** computes the shared secret: \\(s = A^b \bmod p\\).

<svg viewBox="0 0 600 320" xmlns="http://www.w3.org/2000/svg" style="max-width:600px; width:100%; background:#1a1a2e; border-radius:8px; margin:1.5em auto; display:block;">
  <text x="100" y="30" fill="#6db3f2" font-size="16" font-weight="bold" text-anchor="middle">Alice</text>
  <text x="500" y="30" fill="#6db3f2" font-size="16" font-weight="bold" text-anchor="middle">Bob</text>
  <line x1="100" y1="40" x2="100" y2="300" stroke="#6db3f2" stroke-width="2"/>
  <line x1="500" y1="40" x2="500" y2="300" stroke="#6db3f2" stroke-width="2"/>
  <text x="60" y="70" fill="#d4d4d4" font-size="12" text-anchor="end">pick secret a</text>
  <text x="540" y="70" fill="#d4d4d4" font-size="12" text-anchor="start">pick secret b</text>
  <text x="60" y="95" fill="#aaa" font-size="11" text-anchor="end">A = gᵃ mod p</text>
  <text x="540" y="95" fill="#aaa" font-size="11" text-anchor="start">B = gᵇ mod p</text>
  <line x1="105" y1="120" x2="495" y2="150" stroke="#e6c07b" stroke-width="2" marker-end="url(#arrowY)"/>
  <text x="300" y="125" fill="#e6c07b" font-size="13" text-anchor="middle">A = gᵃ mod p (public)</text>
  <line x1="495" y1="170" x2="105" y2="200" stroke="#98c379" stroke-width="2" marker-end="url(#arrowG)"/>
  <text x="300" y="185" fill="#98c379" font-size="13" text-anchor="middle">B = gᵇ mod p (public)</text>
  <text x="60" y="240" fill="#d4d4d4" font-size="12" text-anchor="end">s = Bᵃ mod p</text>
  <text x="540" y="240" fill="#d4d4d4" font-size="12" text-anchor="start">s = Aᵇ mod p</text>
  <text x="60" y="260" fill="#c678dd" font-size="12" text-anchor="end">= gᵃᵇ mod p</text>
  <text x="540" y="260" fill="#c678dd" font-size="12" text-anchor="start">= gᵃᵇ mod p</text>
  <text x="300" y="295" fill="#c678dd" font-size="14" font-weight="bold" text-anchor="middle">Shared secret: s = gᵃᵇ mod p</text>
  <defs>
    <marker id="arrowY" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#e6c07b"/></marker>
    <marker id="arrowG" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#98c379"/></marker>
  </defs>
</svg>

### Correctness Proof

Both parties compute the same value because modular exponentiation respects the group structure:

$$
s_{\text{Alice}} = B^a \bmod p = (g^b)^a \bmod p = g^{ba} \bmod p
$$

$$
s_{\text{Bob}} = A^b \bmod p = (g^a)^b \bmod p = g^{ab} \bmod p
$$

Since multiplication of integers is commutative, \\(ab = ba\\), so \\(g^{ab} \equiv g^{ba} \pmod{p}\\). Both parties arrive at the same shared secret \\(s = g^{ab} \bmod p\\). \\(\blacksquare\\)

### Security: CDH and DDH Assumptions

An eavesdropper sees \\(p\\), \\(g\\), \\(A = g^a \bmod p\\), and \\(B = g^b \bmod p\\). To recover the shared secret \\(s = g^{ab} \bmod p\\), they would need to solve one of two problems:

**Computational Diffie-Hellman (CDH) Assumption:** Given \\(g\\), \\(g^a\\), and \\(g^b\\), it is computationally infeasible to compute \\(g^{ab}\\).

**Decisional Diffie-Hellman (DDH) Assumption:** Given \\(g\\), \\(g^a\\), \\(g^b\\), and a value \\(z\\), it is computationally infeasible to determine whether \\(z = g^{ab}\\) or \\(z\\) is a random group element.

The DDH assumption is strictly stronger than CDH (DDH implies CDH but not vice versa). The DDH assumption is what we need for many cryptographic protocols --- it says the shared secret is not merely hard to compute, but indistinguishable from random.

### Man-in-the-Middle Attack

Diffie-Hellman by itself has a critical vulnerability. If an active adversary Mallory sits between Alice and Bob, she can:

1. Intercept Alice's \\(A = g^a \bmod p\\) and send her own \\(M_1 = g^{m_1} \bmod p\\) to Bob.
2. Intercept Bob's \\(B = g^b \bmod p\\) and send her own \\(M_2 = g^{m_2} \bmod p\\) to Alice.

Now Alice and Mallory share key \\(g^{am_2}\\), and Bob and Mallory share key \\(g^{bm_1}\\). Mallory decrypts everything from Alice, re-encrypts it for Bob, and vice versa. Neither Alice nor Bob detects anything.

This is why Diffie-Hellman must be combined with **authentication** --- digital signatures, certificates, or pre-shared identity information. The TLS handshake (which we will explore in Part 4) uses authenticated Diffie-Hellman to prevent this attack.

### Choosing Safe Parameters

Not all primes are equal. The prime \\(p\\) should be a **safe prime**: \\(p = 2q + 1\\) where \\(q\\) is also prime (a Sophie Germain prime). This ensures that the subgroup of quadratic residues has prime order \\(q\\), preventing attacks that exploit small subgroup structure.

The generator \\(g\\) should generate the subgroup of order \\(q\\) (not the full group of order \\(p - 1\\)), so that the discrete logarithm must be computed in a group with no small factors.

Standard parameter sets are specified in RFCs. For example, RFC 3526 defines groups for IKE (Internet Key Exchange) with primes of 2048, 3072, and 4096 bits.

### Python Implementation

```python
import numpy as np
from sympy import isprime, primitive_root, randprime

def generate_dh_params(bits=32):
    """Generate Diffie-Hellman parameters (small for demonstration)."""
    # Find a prime p
    lower = 2**(bits - 1)
    upper = 2**bits
    p = randprime(lower, upper)
    # Find a generator of Z/pZ*
    g = primitive_root(p)
    return p, g

def dh_key_exchange():
    """Demonstrate Diffie-Hellman key exchange."""
    # Public parameters
    p, g = generate_dh_params(bits=32)
    print(f"Public parameters: p = {p}, g = {g}")

    # Alice generates her secret and public value
    a = np.random.randint(2, p - 2)  # Alice's secret
    A = pow(g, int(a), p)            # Alice's public value
    print(f"\nAlice's secret a = {a}")
    print(f"Alice's public A = g^a mod p = {A}")

    # Bob generates his secret and public value
    b = np.random.randint(2, p - 2)  # Bob's secret
    B = pow(g, int(b), p)            # Bob's public value
    print(f"\nBob's secret b = {b}")
    print(f"Bob's public B = g^b mod p = {B}")

    # Both compute the shared secret
    s_alice = pow(B, int(a), p)  # Alice computes B^a mod p
    s_bob = pow(A, int(b), p)    # Bob computes A^b mod p
    print(f"\nAlice's shared secret: B^a mod p = {s_alice}")
    print(f"Bob's shared secret:   A^b mod p = {s_bob}")
    print(f"Secrets match: {s_alice == s_bob}")

    return s_alice

shared_secret = dh_key_exchange()
```

---

## RSA — From Euler's Theorem to Public-Key Encryption

In 1977, Ron Rivest, Adi Shamir, and Leonard Adleman published a system that realized the dream of public-key encryption: anyone can encrypt a message to you using your public key, but only you can decrypt it using your private key. The system is called **RSA**, and its security rests on the hardness of integer factorization.

### Key Generation

1. Choose two large distinct primes \\(p\\) and \\(q\\).
2. Compute \\(n = pq\\). This is the **modulus**.
3. Compute **Euler's totient**: \\(\phi(n) = (p-1)(q-1)\\).
4. Choose an integer \\(e\\) with \\(1 < e < \phi(n)\\) and \\(\gcd(e, \phi(n)) = 1\\). This is the **public exponent**. The standard choice is \\(e = 65537 = 2^{16} + 1\\), which is prime and has a convenient binary representation (fast exponentiation).
5. Compute the **private exponent** \\(d = e^{-1} \bmod \phi(n)\\), meaning \\(ed \equiv 1 \pmod{\phi(n)}\\). This is found using the extended Euclidean algorithm (which we derived in Part 1).

The **public key** is \\((n, e)\\). The **private key** is \\(d\\) (or equivalently, \\((p, q, d)\\)).

### Encryption and Decryption

To encrypt a message \\(m \in \{0, 1, \ldots, n-1\}\\):

$$
c = m^e \bmod n
$$

To decrypt:

$$
m = c^d \bmod n
$$

### Full Correctness Proof

We need to prove that \\(c^d \equiv m \pmod{n}\\), i.e., that \\(m^{ed} \equiv m \pmod{n}\\).

**Case 1: \\(\gcd(m, n) = 1\\).**

By construction, \\(ed \equiv 1 \pmod{\phi(n)}\\), so \\(ed = 1 + k\phi(n)\\) for some non-negative integer \\(k\\). By Euler's theorem (proved in Part 1), if \\(\gcd(m, n) = 1\\):

$$
m^{\phi(n)} \equiv 1 \pmod{n}
$$

Therefore:

$$
m^{ed} = m^{1 + k\phi(n)} = m \cdot (m^{\phi(n)})^k \equiv m \cdot 1^k = m \pmod{n}
$$

**Case 2: \\(\gcd(m, n) \neq 1\\).**

Since \\(n = pq\\) and \\(m \in \{0, \ldots, n-1\}\\), if \\(\gcd(m, n) \neq 1\\), then \\(m\\) is divisible by \\(p\\), by \\(q\\), or by both.

If \\(m \equiv 0 \pmod{n}\\), then \\(m^{ed} \equiv 0 \equiv m \pmod{n}\\). Done.

Suppose \\(p \mid m\\) but \\(q \nmid m\\) (the other case is symmetric). We use the **Chinese Remainder Theorem** (CRT): to show \\(m^{ed} \equiv m \pmod{n}\\), it suffices to show it modulo \\(p\\) and modulo \\(q\\) separately.

**Modulo \\(p\\):** Since \\(p \mid m\\), we have \\(m \equiv 0 \pmod{p}\\), so \\(m^{ed} \equiv 0 \equiv m \pmod{p}\\). \\(\checkmark\\)

**Modulo \\(q\\):** Since \\(q \nmid m\\), we have \\(\gcd(m, q) = 1\\). By Fermat's little theorem:

$$
m^{q-1} \equiv 1 \pmod{q}
$$

Now \\(ed = 1 + k(p-1)(q-1)\\), so:

$$
m^{ed} = m \cdot m^{k(p-1)(q-1)} = m \cdot \left(m^{q-1}\right)^{k(p-1)} \equiv m \cdot 1^{k(p-1)} = m \pmod{q}
$$

By CRT, since \\(m^{ed} \equiv m \pmod{p}\\) and \\(m^{ed} \equiv m \pmod{q}\\), and \\(\gcd(p, q) = 1\\):

$$
m^{ed} \equiv m \pmod{n}
$$

\\(\blacksquare\\)

### Why Factoring Breaks RSA

If an attacker can factor \\(n\\) into \\(p\\) and \\(q\\), they can compute \\(\phi(n) = (p-1)(q-1)\\), then compute \\(d = e^{-1} \bmod \phi(n)\\) using the extended Euclidean algorithm. The private key is recovered.

The converse --- whether breaking RSA is *equivalent* to factoring --- is an open question. It is possible that there exists a way to compute \\(m\\) from \\(c = m^e \bmod n\\) without factoring \\(n\\). However, no such method is known, and the assumption that RSA is at least as hard as factoring is standard.

### RSA Key Sizes and Security Levels

The security of RSA depends on the difficulty of factoring \\(n\\). As factoring algorithms improve and computation gets cheaper, key sizes must grow:

| RSA Key Size | Approximate Security | Status |
|:---:|:---:|:---:|
| 1024 bits | ~80 bits | **Broken** — factorable with sufficient resources |
| 2048 bits | ~112 bits | Minimum for current use |
| 3072 bits | ~128 bits | Recommended for medium-term security |
| 4096 bits | ~152 bits | Conservative choice |

The "security bits" indicate the equivalent symmetric key strength. A 2048-bit RSA key provides roughly the same security as a 112-bit symmetric key. Note how inefficient RSA is compared to symmetric cryptography: 2048 bits of public key for 112 bits of security.

### Textbook RSA Is Insecure

The encryption \\(c = m^e \bmod n\\) described above is called **textbook RSA**, and it is deterministic. The same plaintext always produces the same ciphertext. This violates **semantic security** --- an adversary who suspects the message is either "ATTACK" or "RETREAT" can encrypt both and compare. Textbook RSA also has algebraic structure: \\(E(m_1) \cdot E(m_2) = m_1^e \cdot m_2^e = (m_1 m_2)^e = E(m_1 m_2) \pmod{n}\\). This **homomorphic** property enables chosen-ciphertext attacks.

The fix is **padding**. The standard is **OAEP** (Optimal Asymmetric Encryption Padding), which adds randomness and structure to the plaintext before encryption. RSA-OAEP is provably secure under the RSA assumption in the random oracle model.

In practice, RSA is rarely used to encrypt messages directly. Instead, it encrypts a randomly generated symmetric key (a "key encapsulation"), and the actual message is encrypted with AES using that key. This is called a **hybrid cryptosystem**.

### Python: RSA Implementation

```python
import numpy as np
from sympy import isprime, gcd, mod_inverse, randprime

def generate_rsa_keys(bits=32):
    """Generate RSA key pair (small primes for demonstration)."""
    # Step 1: Choose two distinct primes
    lower = 2**(bits//2 - 1)
    upper = 2**(bits//2)
    p = randprime(lower, upper)
    q = randprime(lower, upper)
    while q == p:
        q = randprime(lower, upper)

    # Step 2: Compute n and phi(n)
    n = p * q
    phi_n = (p - 1) * (q - 1)

    # Step 3: Choose public exponent e
    e = 65537
    while gcd(e, phi_n) != 1:
        e += 2  # Ensure coprimality

    # Step 4: Compute private exponent d
    d = mod_inverse(e, phi_n)

    print(f"Primes: p = {p}, q = {q}")
    print(f"Modulus: n = {n}")
    print(f"Euler's totient: phi(n) = {phi_n}")
    print(f"Public exponent: e = {e}")
    print(f"Private exponent: d = {d}")
    print(f"Verification: e*d mod phi(n) = {(e * d) % phi_n}")

    return (n, e), (n, d), (p, q)

def rsa_encrypt(m, public_key):
    """Encrypt message m using RSA public key."""
    n, e = public_key
    if m >= n:
        raise ValueError(f"Message {m} must be less than modulus {n}")
    return pow(m, e, n)

def rsa_decrypt(c, private_key):
    """Decrypt ciphertext c using RSA private key."""
    n, d = private_key
    return pow(c, d, n)

# Demonstration
public_key, private_key, (p, q) = generate_rsa_keys(bits=32)

message = 42
print(f"\nOriginal message: {message}")

ciphertext = rsa_encrypt(message, public_key)
print(f"Encrypted: {ciphertext}")

decrypted = rsa_decrypt(ciphertext, private_key)
print(f"Decrypted: {decrypted}")
print(f"Correct: {decrypted == message}")
```

---

## ElGamal Encryption

The **ElGamal** encryption scheme, published by Taher ElGamal in 1985, provides an alternative to RSA based on the Diffie-Hellman problem rather than factoring. Its significance is both practical and theoretical: ElGamal achieves **semantic security** under the DDH assumption without padding.

### Key Generation

1. Choose a large prime \\(p\\) and a generator \\(g\\) of \\(\mathbb{Z}/p\mathbb{Z}^*\\).
2. Choose a random private key \\(x \in \{1, \ldots, p-2\}\\).
3. Compute the public key \\(h = g^x \bmod p\\).
4. The public key is \\((p, g, h)\\); the private key is \\(x\\).

### Encryption

To encrypt a message \\(m \in \{1, \ldots, p-1\}\\):

1. Choose a random ephemeral key \\(k \in \{1, \ldots, p-2\}\\).
2. Compute \\(c_1 = g^k \bmod p\\).
3. Compute \\(c_2 = m \cdot h^k \bmod p\\).
4. The ciphertext is \\((c_1, c_2)\\).

The randomness \\(k\\) is critical --- it makes the encryption probabilistic. Each encryption of the same message produces a different ciphertext.

### Decryption

Given ciphertext \\((c_1, c_2)\\) and private key \\(x\\):

1. Compute the shared secret \\(s = c_1^x \bmod p\\).
2. Compute \\(s^{-1} \bmod p\\) (the modular inverse).
3. Recover \\(m = c_2 \cdot s^{-1} \bmod p\\).

### Correctness Proof

We need \\(c_2 \cdot s^{-1} \equiv m \pmod{p}\\):

$$
s = c_1^x = (g^k)^x = g^{kx} \pmod{p}
$$

$$
c_2 \cdot s^{-1} = m \cdot h^k \cdot (g^{kx})^{-1} = m \cdot (g^x)^k \cdot g^{-kx} = m \cdot g^{kx} \cdot g^{-kx} = m \pmod{p}
$$

\\(\blacksquare\\)

### Semantic Security Under DDH

ElGamal is **semantically secure** under the DDH assumption. Intuitively, the ciphertext \\((g^k, m \cdot h^k)\\) looks like \\((g^k, m \cdot g^{xk})\\). If DDH holds, then \\(g^{xk}\\) is indistinguishable from a random group element, so \\(m \cdot g^{xk}\\) perfectly masks \\(m\\). An adversary who cannot distinguish \\(g^{xk}\\) from random cannot learn anything about \\(m\\) from the ciphertext.

This is a significant advantage over textbook RSA, which is deterministic and therefore cannot be semantically secure without padding.

---

## Elliptic Curve Cryptography — Shorter Keys, Same Security

The systems above --- RSA, Diffie-Hellman, ElGamal --- all work in the multiplicative group \\(\mathbb{Z}/n\mathbb{Z}^*\\). They are effective, but they require enormous key sizes. A 3072-bit RSA modulus provides about 128 bits of security. Can we find a mathematical structure where the discrete logarithm problem is *harder*, so that smaller keys achieve the same security?

The answer is **elliptic curves**.

### Elliptic Curves Over the Reals

An **elliptic curve** over \\(\mathbb{R}\\) is the set of points \\((x, y)\\) satisfying the **Weierstrass equation**:

$$
y^2 = x^3 + ax + b
$$

together with a special **point at infinity** \\(\mathcal{O}\\), which serves as the identity element.

The constants \\(a\\) and \\(b\\) must satisfy the **discriminant condition**:

$$
\Delta = 4a^3 + 27b^2 \neq 0
$$

This condition ensures the curve is **non-singular** --- it has no cusps or self-intersections. A singular curve would not have the group structure we need.

For example, the curve \\(y^2 = x^3 - x + 1\\) has \\(a = -1\\), \\(b = 1\\), and \\(\Delta = 4(-1)^3 + 27(1)^2 = -4 + 27 = 23 \neq 0\\). It is a valid elliptic curve.

### The Group Law: Chord and Tangent

The remarkable fact about elliptic curves is that the set of points forms an **abelian group** under a geometrically natural addition operation.

<svg viewBox="0 0 500 400" xmlns="http://www.w3.org/2000/svg" style="max-width:500px; width:100%; background:#1a1a2e; border-radius:8px; margin:1.5em auto; display:block;">
  <!-- Axes -->
  <line x1="50" y1="200" x2="450" y2="200" stroke="#555" stroke-width="1"/>
  <line x1="180" y1="30" x2="180" y2="370" stroke="#555" stroke-width="1"/>
  <!-- Elliptic curve y^2 = x^3 - x + 1 (stylized) -->
  <path d="M 80,200 C 80,140 120,60 160,55 C 200,50 220,100 240,140 C 260,170 280,185 320,170 C 360,150 400,80 440,40" stroke="#6db3f2" stroke-width="2.5" fill="none"/>
  <path d="M 80,200 C 80,260 120,340 160,345 C 200,350 220,300 240,260 C 260,230 280,215 320,230 C 360,250 400,320 440,360" stroke="#6db3f2" stroke-width="2.5" fill="none"/>
  <!-- Point P -->
  <circle cx="160" cy="55" r="5" fill="#e6c07b"/>
  <text x="145" y="45" fill="#e6c07b" font-size="14" font-weight="bold">P</text>
  <!-- Point Q -->
  <circle cx="280" cy="185" r="5" fill="#98c379"/>
  <text x="288" y="180" fill="#98c379" font-size="14" font-weight="bold">Q</text>
  <!-- Chord line through P and Q, extended -->
  <line x1="100" y1="19" x2="420" y2="270" stroke="#e06c75" stroke-width="1.5" stroke-dasharray="6,3"/>
  <!-- Intersection R' -->
  <circle cx="370" cy="240" r="5" fill="#e06c75"/>
  <text x="378" y="235" fill="#e06c75" font-size="13">R'</text>
  <!-- Vertical line from R' to R -->
  <line x1="370" y1="240" x2="370" y2="160" stroke="#c678dd" stroke-width="1.5" stroke-dasharray="4,3"/>
  <!-- Result R = P + Q -->
  <circle cx="370" cy="160" r="5" fill="#c678dd"/>
  <text x="378" y="155" fill="#c678dd" font-size="14" font-weight="bold">P + Q</text>
  <!-- Labels -->
  <text x="250" y="390" fill="#aaa" font-size="12" text-anchor="middle">Chord-and-tangent construction: draw line through P and Q,</text>
  <text x="250" y="388" fill="#aaa" font-size="12" text-anchor="middle"> </text>
  <text x="450" y="198" fill="#aaa" font-size="11">x</text>
  <text x="184" y="35" fill="#aaa" font-size="11">y</text>
</svg>

**Point addition** (\\(P + Q\\) where \\(P \neq Q\\)):

1. Draw the line through \\(P = (x_1, y_1)\\) and \\(Q = (x_2, y_2)\\).
2. This line intersects the curve at a third point \\(R' = (x_3', y_3')\\).
3. Reflect \\(R'\\) across the x-axis to get \\(R = P + Q = (x_3, y_3)\\) where \\(y_3 = -y_3'\\).

The explicit formulas: the slope of the line through \\(P\\) and \\(Q\\) is:

$$
\lambda = \frac{y_2 - y_1}{x_2 - x_1}
$$

The coordinates of \\(P + Q = (x_3, y_3)\\) are:

$$
x_3 = \lambda^2 - x_1 - x_2
$$

$$
y_3 = \lambda(x_1 - x_3) - y_1
$$

**Point doubling** (\\(P + P = 2P\\)):

When \\(P = Q\\), the "line through two points" becomes the tangent line at \\(P\\). Using implicit differentiation on \\(y^2 = x^3 + ax + b\\):

$$
2y \, dy = (3x^2 + a) \, dx \implies \frac{dy}{dx} = \frac{3x^2 + a}{2y}
$$

So the slope for doubling at \\(P = (x_1, y_1)\\) is:

$$
\lambda = \frac{3x_1^2 + a}{2y_1}
$$

The formulas for \\((x_3, y_3)\\) are the same as above. Note that doubling requires \\(y_1 \neq 0\\); if \\(y_1 = 0\\), then \\(2P = \mathcal{O}\\).

**The identity element** \\(\mathcal{O}\\) (the point at infinity) satisfies \\(P + \mathcal{O} = P\\) for all \\(P\\). The **inverse** of \\(P = (x, y)\\) is \\(-P = (x, -y)\\) (reflection across the x-axis). Indeed, the line through \\((x, y)\\) and \\((x, -y)\\) is vertical and intersects the curve "at infinity," giving \\(P + (-P) = \mathcal{O}\\).

### Elliptic Curves Over Finite Fields \\(\mathbb{F}_p\\)

For cryptography, we work over a **finite field** \\(\mathbb{F}_p = \mathbb{Z}/p\mathbb{Z}\\) where \\(p\\) is a large prime. The equation is the same:

$$
y^2 \equiv x^3 + ax + b \pmod{p}
$$

The algebraic formulas for point addition and doubling carry over exactly --- we simply replace division by modular inverse. For example, \\(\frac{y_2 - y_1}{x_2 - x_1}\\) becomes \\((y_2 - y_1)(x_2 - x_1)^{-1} \bmod p\\).

The set of points \\(E(\mathbb{F}_p)\\) --- all \\((x, y)\\) satisfying the curve equation plus the point at infinity --- forms a finite abelian group.

**Hasse's Theorem** gives us the size of this group:

$$
|N - (p + 1)| \leq 2\sqrt{p}
$$

where \\(N = |E(\mathbb{F}_p)|\\) is the number of points on the curve. So the number of points is approximately \\(p\\), bounded within an interval of width \\(4\sqrt{p}\\) centered at \\(p + 1\\).

The group \\(E(\mathbb{F}_p)\\) is isomorphic to \\(\mathbb{Z}/n_1\mathbb{Z} \times \mathbb{Z}/n_2\mathbb{Z}\\) where \\(n_2 \mid n_1\\) and \\(n_2 \mid (p - 1)\\). For cryptographic curves, we choose parameters so that \\(N\\) has a large prime factor, giving us a cyclic subgroup of prime order where the discrete logarithm is hard.

### The Elliptic Curve Discrete Logarithm Problem (ECDLP)

Given a base point \\(P\\) on the curve and a point \\(Q = nP\\) (where \\(nP\\) means adding \\(P\\) to itself \\(n\\) times), the **ECDLP** is: find \\(n\\).

The critical question: why is ECDLP harder than the ordinary DLP in \\(\mathbb{Z}/p\mathbb{Z}^*\\)?

In \\(\mathbb{Z}/p\mathbb{Z}^*\\), the **index calculus** algorithm exploits the multiplicative structure to achieve subexponential running time \\(L_p[1/3, c]\\). This is why DLP over integers requires 3072-bit primes for 128-bit security.

For elliptic curves over prime fields, *no analogue of index calculus is known*. The best generic algorithms are:

- **Baby-step giant-step:** \\(O(\sqrt{n})\\) time and space.
- **Pollard's rho:** \\(O(\sqrt{n})\\) time, \\(O(1)\\) space.

Both are fully exponential in the bit-length of the group order. A 256-bit elliptic curve group requires \\(\sim 2^{128}\\) operations to attack --- exactly 128 bits of security. Compare this to the 3072 bits required for RSA to achieve the same level.

This is the fundamental advantage of ECC: **the same security with dramatically smaller keys**.

### ECDH — Elliptic Curve Diffie-Hellman

ECDH is simply Diffie-Hellman in an elliptic curve group instead of \\(\mathbb{Z}/p\mathbb{Z}^*\\):

1. Alice and Bob agree on a curve \\(E\\) over \\(\mathbb{F}_p\\) and a base point \\(G\\) of prime order \\(n\\).
2. Alice picks secret \\(a \in \{1, \ldots, n-1\}\\), computes \\(A = aG\\), sends \\(A\\) to Bob.
3. Bob picks secret \\(b \in \{1, \ldots, n-1\}\\), computes \\(B = bG\\), sends \\(B\\) to Alice.
4. Shared secret: Alice computes \\(aB = a(bG) = abG\\). Bob computes \\(bA = b(aG) = abG\\).

The shared secret is the point \\(abG\\). In practice, the x-coordinate of this point is used as the shared key (after hashing).

Why does 256-bit ECC provide security comparable to 3072-bit RSA? Because the best attack on ECDLP is fully exponential (\\(O(2^{128})\\) for a 256-bit group), while the best attack on RSA factoring is subexponential. The asymptotic gap means ECC keys can be an order of magnitude smaller for equivalent security.

### ECDSA — Elliptic Curve Digital Signature Algorithm

ECDSA provides digital signatures using elliptic curves. The parameters are a curve \\(E\\), a base point \\(G\\) of prime order \\(n\\), a private key \\(d\\), and a public key \\(Q = dG\\).

**Signing** a message (actually, its hash \\(z\\)):

1. Choose a random nonce \\(k \in \{1, \ldots, n-1\}\\).
2. Compute \\(kG = (x_1, y_1)\\) and set \\(r = x_1 \bmod n\\). If \\(r = 0\\), choose a new \\(k\\).
3. Compute \\(s = k^{-1}(z + rd) \bmod n\\). If \\(s = 0\\), choose a new \\(k\\).
4. The signature is \\((r, s)\\).

**Verification** given message hash \\(z\\), signature \\((r, s)\\), and public key \\(Q\\):

1. Compute \\(w = s^{-1} \bmod n\\).
2. Compute \\(u_1 = zw \bmod n\\) and \\(u_2 = rw \bmod n\\).
3. Compute the point \\(u_1 G + u_2 Q = (x_1, y_1)\\).
4. Accept if \\(r \equiv x_1 \pmod{n}\\).

**The nonce reuse disaster.** If the same nonce \\(k\\) is used for two different messages, the private key \\(d\\) can be recovered. Given two signatures \\((r, s_1)\\) and \\((r, s_2)\\) on messages with hashes \\(z_1\\) and \\(z_2\\):

$$
s_1 - s_2 = k^{-1}(z_1 - z_2) \bmod n
$$

$$
k = \frac{z_1 - z_2}{s_1 - s_2} \bmod n
$$

$$
d = \frac{s_1 k - z_1}{r} \bmod n
$$

This is not a theoretical concern. In 2010, the PlayStation 3's ECDSA implementation used a **constant nonce** for every signature. Researchers extracted Sony's private signing key, enabling anyone to sign arbitrary code as Sony. The entire PS3 security model --- game DRM, firmware validation, everything --- collapsed because of a single repeated random number.

The lesson: cryptographic nonces must be generated with a cryptographically secure random number generator, never reused, and ideally derived deterministically from the message (RFC 6979) to eliminate the random number generator as a point of failure.

### Standard Curves

Several standard curves are in widespread use:

**NIST P-256 (secp256r1):** The most widely deployed curve. Defined over a 256-bit prime field with parameters chosen by NIST. Used in TLS, code signing, and government applications. Some concerns about the parameter generation process (the seed for the curve coefficients is unexplained, leading to speculation about possible NSA involvement in parameter selection).

**Curve25519:** Designed by Daniel Bernstein in 2006. Defined by \\(y^2 = x^3 + 486662x^2 + x\\) over \\(\mathbb{F}_p\\) where \\(p = 2^{255} - 19\\). Preferred by the security community for several reasons:
- The prime \\(2^{255} - 19\\) enables very fast modular arithmetic.
- The curve supports efficient constant-time implementations, resisting timing side-channel attacks.
- Parameters are "nothing-up-my-sleeve" numbers --- transparently chosen, with no unexplained seeds.
- No NSA involvement in the design.

**Ed25519:** A digital signature scheme based on a "twisted Edwards" form of Curve25519. Designed by Bernstein, Duif, Lange, Schwabe, and Yang. Used in SSH (since OpenSSH 6.5), Signal Protocol, and many modern systems. Deterministic nonce generation eliminates the nonce reuse vulnerability of ECDSA.

For any new system being designed today, Curve25519 (for key exchange) and Ed25519 (for signatures) are the default recommendations.

### Python: Elliptic Curve Over the Reals

```python
import numpy as np
import matplotlib.pyplot as plt

# Plot the elliptic curve y^2 = x^3 - x + 1 over the reals
a, b = -1, 1

x = np.linspace(-1.5, 2.5, 10000)
# Compute y^2 = x^3 + ax + b
rhs = x**3 + a * x + b

# Only plot where rhs >= 0
mask = rhs >= 0
x_pos = x[mask]
y_pos = np.sqrt(rhs[mask])
y_neg = -y_pos

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')

# Plot the curve
ax.plot(x_pos, y_pos, color='#6db3f2', linewidth=2, label=r'$y^2 = x^3 - x + 1$')
ax.plot(x_pos, y_neg, color='#6db3f2', linewidth=2)

# Demonstrate point addition: P and Q on the curve
# P = (-1, 1): check (-1)^3 + (-1)(-1) + 1 = -1 + 1 + 1 = 1, y^2 = 1. Yes.
# Q = (0, 1): check 0 + 0 + 1 = 1, y^2 = 1. Yes.
Px, Py = -1, 1
Qx, Qy = 0, 1

# Slope of secant line
lam = (Qy - Py) / (Qx - Px)
# Third intersection: x3 = lam^2 - x1 - x2
x3 = lam**2 - Px - Qx
y3_prime = lam * (Px - x3) - Py  # Point on line, then reflect
y3 = y3_prime  # This IS the result after reflection in the formula

# The raw intersection point (before reflection) is (x3, -y3)
ax.plot([Px, Qx, x3], [Py, Qy, -y3], 'o--', color='#e06c75',
        markersize=0, linewidth=1.5, alpha=0.7)
# Extend the line for visual clarity
line_x = np.linspace(Px - 0.5, x3 + 0.5, 100)
line_y = Py + lam * (line_x - Px)
ax.plot(line_x, line_y, '--', color='#e06c75', linewidth=1.2, alpha=0.6)

# Vertical reflection line
ax.plot([x3, x3], [-y3, y3], '--', color='#c678dd', linewidth=1.2, alpha=0.7)

# Plot points
ax.plot(Px, Py, 'o', color='#e6c07b', markersize=10, zorder=5)
ax.annotate(r'$P$', (Px, Py), textcoords="offset points",
            xytext=(-15, 10), color='#e6c07b', fontsize=14, fontweight='bold')

ax.plot(Qx, Qy, 'o', color='#98c379', markersize=10, zorder=5)
ax.annotate(r'$Q$', (Qx, Qy), textcoords="offset points",
            xytext=(10, 10), color='#98c379', fontsize=14, fontweight='bold')

ax.plot(x3, -y3, 'o', color='#e06c75', markersize=8, zorder=5)
ax.annotate(r"$R'$", (x3, -y3), textcoords="offset points",
            xytext=(10, -15), color='#e06c75', fontsize=12)

ax.plot(x3, y3, 'o', color='#c678dd', markersize=10, zorder=5)
ax.annotate(r'$P + Q$', (x3, y3), textcoords="offset points",
            xytext=(10, 10), color='#c678dd', fontsize=14, fontweight='bold')

ax.set_xlabel(r'$x$', fontsize=14, color='#d4d4d4')
ax.set_ylabel(r'$y$', fontsize=14, color='#d4d4d4')
ax.set_title(r'Elliptic Curve Point Addition: $y^2 = x^3 - x + 1$',
             fontsize=14, color='#d4d4d4')
ax.axhline(y=0, color='#555', linewidth=0.5)
ax.axvline(x=0, color='#555', linewidth=0.5)
ax.tick_params(colors='#999')
ax.spines['bottom'].set_color('#555')
ax.spines['left'].set_color('#555')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=12, facecolor='#1a1a2e', edgecolor='#555', labelcolor='#d4d4d4')
ax.set_xlim(-1.8, 3.0)
ax.set_ylim(-3.5, 3.5)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('ec_point_addition.png', dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()
```

### Python: Point Addition Over a Finite Field

```python
import numpy as np

def ec_add(P, Q, a, p):
    """Add two points on elliptic curve y^2 = x^3 + ax + b over F_p.

    Points are (x, y) tuples, or None for the point at infinity.
    """
    if P is None:
        return Q
    if Q is None:
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2 and (y1 + y2) % p == 0:
        return None  # P + (-P) = O (point at infinity)

    if P == Q:
        # Point doubling
        lam = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:
        # Point addition
        lam = (y2 - y1) * pow(x2 - x1, -1, p) % p

    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)

def ec_multiply(k, P, a, p):
    """Compute kP using double-and-add algorithm."""
    result = None  # Point at infinity
    addend = P
    while k > 0:
        if k & 1:
            result = ec_add(result, addend, a, p)
        addend = ec_add(addend, addend, a, p)
        k >>= 1
    return result

# Work on y^2 = x^3 + 2x + 3 over F_97
a_coeff, b_coeff, p = 2, 3, 97

# Find a point on the curve by brute force
points = []
for x in range(p):
    rhs = (x**3 + a_coeff * x + b_coeff) % p
    for y in range(p):
        if (y * y) % p == rhs:
            points.append((x, y))

print(f"Curve: y^2 = x^3 + {a_coeff}x + {b_coeff} over F_{p}")
print(f"Number of points (including O): {len(points) + 1}")

# Pick a base point and demonstrate scalar multiplication
G = points[0]
print(f"\nBase point G = {G}")

# Compute successive multiples of G
print("\nScalar multiples of G:")
for k in range(1, 11):
    kG = ec_multiply(k, G, a_coeff, p)
    print(f"  {k}G = {kG}")

# Verify point addition is consistent
P = ec_multiply(3, G, a_coeff, p)
Q = ec_multiply(5, G, a_coeff, p)
R = ec_multiply(8, G, a_coeff, p)
PQ = ec_add(P, Q, a_coeff, p)
print(f"\n3G = {P}")
print(f"5G = {Q}")
print(f"3G + 5G = {PQ}")
print(f"8G = {R}")
print(f"Verification (3G + 5G == 8G): {PQ == R}")

# Demonstrate ECDH with this small curve
print("\n--- ECDH Key Exchange (toy example) ---")
# Find the order of G
order = 1
temp = G
while ec_multiply(order, G, a_coeff, p) is not None:
    order += 1
print(f"Order of G: {order}")

alice_secret = 17 % order
bob_secret = 23 % order

alice_public = ec_multiply(alice_secret, G, a_coeff, p)
bob_public = ec_multiply(bob_secret, G, a_coeff, p)

alice_shared = ec_multiply(alice_secret, bob_public, a_coeff, p)
bob_shared = ec_multiply(bob_secret, alice_public, a_coeff, p)

print(f"Alice's secret: {alice_secret}")
print(f"Alice's public point: {alice_public}")
print(f"Bob's secret: {bob_secret}")
print(f"Bob's public point: {bob_public}")
print(f"Alice's shared secret: {alice_shared}")
print(f"Bob's shared secret:   {bob_shared}")
print(f"Shared secrets match: {alice_shared == bob_shared}")
```

---

## Key Size Comparison and the March of Computation

The table below summarizes the key sizes required for equivalent security across the three families of cryptographic primitives:

| Symmetric Key (bits) | RSA Modulus (bits) | DH Key (bits) | ECC Key (bits) |
|:---:|:---:|:---:|:---:|
| 80 | 1024 | 1024 | 160 |
| 112 | 2048 | 2048 | 224 |
| 128 | 3072 | 3072 | 256 |
| 192 | 7680 | 7680 | 384 |
| 256 | 15360 | 15360 | 512 |

The disparity is dramatic. For 128-bit security (the current standard), RSA needs a 3072-bit key while ECC needs only 256 bits --- a factor of 12 in key size. The computational cost difference is even larger: RSA key generation requires finding two 1536-bit primes, while ECC key generation is a single scalar multiplication in a 256-bit group.

### Why ECC Won for Mobile and IoT

The practical consequences of smaller keys go beyond storage:

- **Bandwidth:** A 256-bit ECC public key is 32 bytes. A 3072-bit RSA key is 384 bytes. In a TLS handshake, this difference matters for every connection.
- **Computation:** ECDH key exchange is roughly 10x faster than RSA key exchange at equivalent security levels.
- **Power consumption:** On battery-powered IoT devices, the energy cost of 3072-bit modular exponentiation is prohibitive. ECC operations are feasible even on 8-bit microcontrollers.
- **Latency:** Faster key exchange means faster TLS handshakes, which means faster page loads. When MUST and NATO partners need to establish secure channels under operational time pressure, every millisecond matters.

This is why modern protocols default to ECC. TLS 1.3 uses ECDHE (Elliptic Curve Diffie-Hellman Ephemeral) as its primary key exchange mechanism. Signal Protocol uses Curve25519. SSH defaults to Ed25519 keys. The transition from RSA to ECC is largely complete.

### The Looming Quantum Threat

There is a shadow over all of this. In 1994, Peter Shor showed that a sufficiently powerful quantum computer can factor integers and compute discrete logarithms in polynomial time. Shor's algorithm would break RSA, Diffie-Hellman, and ECC --- all three --- regardless of key size.

No cryptographically relevant quantum computer exists today. But the threat is not hypothetical: nation-states --- China in particular, with massive investments in quantum computing and strategic technologies --- may be collecting encrypted data now with the intention of decrypting it later when quantum computers arrive. This is the "harvest now, decrypt later" strategy, and it means that data with long-term confidentiality requirements (intelligence reports, diplomatic communications, strategic plans) may already be at risk.

MUST's 2024 annual report notes Chinese investments in quantum computing as part of broader strategic technology competition. For intelligence agencies and military organizations, post-quantum migration is not a future concern but a present imperative.

In [Part 5](/2026/03/16/quantum-threat-post-quantum-cryptography.html), we will examine the quantum threat in detail and explore the lattice-based, code-based, and hash-based cryptosystems that are designed to resist it.

---

## Summary

We have covered a lot of ground. The key distribution problem --- the impossibility of securely sharing symmetric keys over insecure channels --- drove the invention of public-key cryptography in the 1970s. The mathematical foundations are one-way functions: operations that are easy to compute but (we believe) hard to invert.

**Diffie-Hellman** solved key exchange by exploiting the discrete logarithm problem: two parties exchange public values and independently compute the same shared secret. It requires authentication to prevent man-in-the-middle attacks.

**RSA** provided public-key encryption and digital signatures by exploiting the difficulty of integer factorization. Its correctness follows from Euler's theorem, with a careful CRT argument for the edge case where the message shares a factor with the modulus. Textbook RSA is insecure; practical RSA uses OAEP padding.

**ElGamal** achieves semantic security under the DDH assumption, providing probabilistic encryption based on the discrete logarithm problem.

**Elliptic curve cryptography** achieves the same security as RSA and Diffie-Hellman with dramatically smaller keys, because the elliptic curve discrete logarithm problem is resistant to the index calculus attacks that work against the ordinary DLP. The group law on elliptic curves --- the chord-and-tangent construction --- gives us a finite abelian group where scalar multiplication is easy but the inverse problem (ECDLP) is hard.

Standard curves like Curve25519 and Ed25519 are the modern defaults, chosen for speed, security, and resistance to implementation pitfalls.

Everything we have built so far --- symmetric ciphers, public-key encryption, key exchange, digital signatures --- are components. In [Part 4](/2026/03/15/cryptographic-protocols-tls-signal-national-security.html), we will assemble them into complete protocols: TLS (securing the web), Signal (securing messages), and the protocol design principles that determine whether a system actually achieves the security its components promise.
