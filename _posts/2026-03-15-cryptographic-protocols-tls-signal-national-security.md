---
layout: post
title: "Cryptographic Protocols: TLS, Signal, and Securing a Nation's Communications"
date: 2026-03-15
category: math
---

*This is Part 4 of a 5-part series on cryptology. [Part 1: Number Theory & Classical Ciphers](/2026/03/12/number-theory-classical-cryptography.html) | [Part 2: Symmetric Cryptography](/2026/03/13/symmetric-cryptography-aes-block-ciphers.html) | [Part 3: Asymmetric Cryptography](/2026/03/14/asymmetric-cryptography-rsa-elliptic-curves.html) | **Part 4: Cryptographic Protocols** | [Part 5: Post-Quantum Cryptography](/2026/03/16/quantum-threat-post-quantum-cryptography.html)*

January 2024. A Russian-flagged merchant vessel in the Baltic Sea turns off its AIS transponder while transporting sanctioned oil through Swedish territorial waters. The ship vanishes from every civilian maritime tracking system. Minutes later, AIS signals reappear --- but the position data is wrong, placing the vessel 40 nautical miles from its actual location, well outside the territorial boundary. Simultaneously, GPS signals near Gotland begin drifting. Swedish Coast Guard patrol boats find their navigation systems placing them hundreds of meters from their true positions. Aircraft approaching Visby report GNSS anomalies.

These are not hypothetical scenarios. Sweden's Military Intelligence and Security Service --- Militära underrättelse- och säkerhetstjänsten, MUST --- documents exactly these kinds of hybrid threats in its annual assessments. Russia's GNSS interference disrupts navigation for ships and aircraft across the Baltic Sea. Vessels falsify AIS identification signals by entering incorrect positions, providing wrong identities and cargo information, or turning off transponders entirely. These attacks exploit a fundamental weakness: civilian navigation and identification signals were designed without cryptographic authentication.

The AIS system broadcasts ship identity, position, course, and speed in plaintext. No authentication. No integrity check. Anyone with a software-defined radio can forge an AIS message claiming to be any vessel at any position. The GPS civilian signal --- L1 C/A --- is similarly unauthenticated. A moderately resourced adversary can broadcast fake GPS signals that overpower the genuine satellite signals, feeding false position data to every receiver in range.

Every protocol failure in that Baltic Sea scenario comes down to missing or broken cryptographic primitives: hash functions that guarantee data integrity, authentication codes that prove message origin, digital signatures that enable public verification, and key exchange protocols that establish secure channels. These are the building blocks of TLS, Signal, GNSS authentication, and every secure communication system that Sweden's total defence depends on.

This article builds those building blocks from the ground up.

---

## Table of Contents

1. [Cryptographic Hash Functions](#cryptographic-hash-functions)
2. [Message Authentication Codes (MACs)](#message-authentication-codes-macs)
3. [Digital Signatures](#digital-signatures)
4. [Public Key Infrastructure (PKI)](#public-key-infrastructure-pki)
5. [TLS 1.3: Securing Every Connection](#tls-13-securing-every-connection)
6. [The Signal Protocol: End-to-End Encryption for Messaging](#the-signal-protocol-end-to-end-encryption-for-messaging)
7. [GNSS Authentication and Signal Integrity](#gnss-authentication-and-signal-integrity)
8. [The Limits of Protocols](#the-limits-of-protocols)

---

## Cryptographic Hash Functions

A **cryptographic hash function** is a function \\(H\\) that takes an input of arbitrary length and produces a fixed-length output called a **digest** or **hash**. For SHA-256, the output is always 256 bits regardless of whether the input is a single byte or an entire database.

Three properties define what makes a hash function *cryptographic* rather than merely a hash function:

**Preimage resistance.** Given a hash value \\(h\\), it is computationally infeasible to find any input \\(m\\) such that \\(H(m) = h\\). You cannot reverse the function. If someone gives you a 256-bit hash, you cannot reconstruct the message that produced it --- the only strategy is brute force, which for a 256-bit output requires on the order of \\(2^{256}\\) evaluations.

**Second preimage resistance.** Given an input \\(m_1\\), it is computationally infeasible to find a different input \\(m_2 \neq m_1\\) such that \\(H(m_1) = H(m_2)\\). If you know a message and its hash, you cannot find a different message with the same hash. This is what prevents an attacker from substituting a malicious file for a legitimate one while keeping the same checksum.

**Collision resistance.** It is computationally infeasible to find *any* pair \\((m_1, m_2)\\) with \\(m_1 \neq m_2\\) such that \\(H(m_1) = H(m_2)\\). Notice the difference from second preimage resistance: here the attacker gets to choose *both* messages freely. This is a strictly stronger requirement.

### The Birthday Paradox and Collision Bounds

The gap between preimage resistance and collision resistance is quantified by the **birthday paradox**. The classic version: how many people do you need in a room before there is a 50% chance that two share a birthday? The answer is only 23, which surprises most people because there are 365 possible birthdays.

The mathematics generalize directly to hash collisions. Suppose our hash function produces outputs uniformly distributed over \\(N = 2^n\\) possible values. We sample random inputs and compute their hashes. After \\(q\\) samples, the probability of *no* collision is:

$$P(\text{no collision}) = \prod_{i=1}^{q-1} \left(1 - \frac{i}{N}\right)$$

Using the approximation \\(1 - x \approx e^{-x}\\) for small \\(x\\):

$$P(\text{no collision}) \approx \exp\left(-\sum_{i=1}^{q-1} \frac{i}{N}\right) = \exp\left(-\frac{q(q-1)}{2N}\right)$$

Setting this equal to \\(1/2\\) and solving:

$$\exp\left(-\frac{q^2}{2N}\right) \approx \frac{1}{2}$$

$$\frac{q^2}{2N} \approx \ln 2$$

$$q \approx \sqrt{2N \ln 2} \approx 1.177\sqrt{N}$$

For a hash function with \\(n\\)-bit output, \\(N = 2^n\\), so:

$$q \approx 1.177 \cdot 2^{n/2}$$

This is why collision resistance requires \\(O(2^{n/2})\\) work, not \\(O(2^n)\\). A 256-bit hash provides 128 bits of collision resistance. A 128-bit hash provides only 64 bits of collision resistance --- completely breakable. This is one reason MD5 (128-bit output) was retired: even before cryptanalytic attacks were found, the birthday bound gave only \\(2^{64}\\) collision resistance.

### The Merkle-Damgard Construction

Most classical hash functions (MD5, SHA-1, SHA-256) follow the **Merkle-Damgard construction**. The idea is to build a hash function for arbitrary-length inputs from a fixed-size **compression function** \\(f\\).

The message \\(M\\) is padded to a multiple of the block size, then split into blocks \\(M_1, M_2, \ldots, M_L\\). An initial value \\(IV\\) (a fixed constant) serves as the starting state. The construction iterates:

$$h_0 = IV$$

$$h_i = f(h_{i-1}, M_i) \quad \text{for } i = 1, 2, \ldots, L$$

$$H(M) = h_L$$

Each call to \\(f\\) takes the previous chaining value \\(h_{i-1}\\) and the current message block \\(M_i\\) and produces the next chaining value. The final chaining value is the hash.

The critical theorem: **if \\(f\\) is collision-resistant, then the Merkle-Damgard construction is collision-resistant.** The proof is a reduction: given a collision in \\(H\\), you can find a collision in \\(f\\) by walking back through the chains.

### SHA-256

SHA-256 is a Merkle-Damgard hash with a 256-bit state and 512-bit message blocks. Its compression function operates in 64 rounds. Each round mixes the current state using bitwise operations (AND, OR, XOR, rotations, shifts) and additions modulo \\(2^{32}\\). The initial value consists of the first 32 bits of the fractional parts of the square roots of the first 8 primes. The round constants are the first 32 bits of the fractional parts of the cube roots of the first 64 primes --- nothing-up-my-sleeve numbers that prevent the designers from hiding a backdoor.

### The Sponge Construction (SHA-3 / Keccak)

SHA-3, standardized in 2015, uses a fundamentally different architecture: the **sponge construction**. Instead of the Merkle-Damgard iterate-a-compression-function approach, a sponge has a fixed permutation \\(f\\) operating on a state of \\(b = r + c\\) bits, where \\(r\\) is the **rate** (bits absorbed/squeezed per step) and \\(c\\) is the **capacity** (bits that provide security).

**Absorbing phase:** the message blocks are XORed into the first \\(r\\) bits of the state, and \\(f\\) is applied after each block.

**Squeezing phase:** the first \\(r\\) bits of the state are output, and \\(f\\) is applied between output blocks. You squeeze until you have enough output bits.

The security level is determined by \\(c\\): collision resistance is \\(O(2^{c/2})\\) and preimage resistance is \\(O(2^{c/2})\\) or \\(O(2^n)\\) (whichever is smaller, where \\(n\\) is the output length). For SHA3-256, \\(c = 512\\), giving 256-bit collision resistance.

The sponge construction avoids the length-extension vulnerability of Merkle-Damgard: knowing \\(H(M)\\) does not let you compute \\(H(M \| M')\\) without knowing \\(M\\), because the capacity bits are never directly accessible.

### Demonstrating the Avalanche Effect

A good hash function exhibits the **avalanche effect**: changing a single bit of input changes approximately half the output bits. This is what makes hash functions useful as fingerprints --- similar inputs produce completely unrelated outputs.

```python
import hashlib

def hash_binary(msg: bytes) -> str:
    """Return SHA-256 hash as a binary string."""
    digest = hashlib.sha256(msg).digest()
    return ''.join(f'{byte:08b}' for byte in digest)

def hamming_distance(s1: str, s2: str) -> int:
    """Count differing bits between two binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

# Original message
msg1 = b"AIS: MMSI=265803000, LAT=57.6882, LON=18.2848, SOG=12.4"
# Flip one bit: change last character from '4' (0x34) to '5' (0x35)
msg2 = b"AIS: MMSI=265803000, LAT=57.6882, LON=18.2848, SOG=12.5"

h1 = hash_binary(msg1)
h2 = hash_binary(msg2)

diff = hamming_distance(h1, h2)
print(f"Message 1: {msg1.decode()}")
print(f"Hash 1:    {hashlib.sha256(msg1).hexdigest()}")
print()
print(f"Message 2: {msg2.decode()}")
print(f"Hash 2:    {hashlib.sha256(msg2).hexdigest()}")
print()
print(f"Bits changed: {diff} / 256 ({diff/256*100:.1f}%)")
# Expect ~128 bits to differ (≈ 50%)
```

Running this produces approximately 128 differing bits out of 256 --- essentially a coin flip for each bit. An attacker who modifies an AIS message even slightly produces a completely different hash. If the receiver checks the hash, the tampering is immediately detected.

But hashing alone is not enough. An attacker could modify the message *and* recompute the hash. To prevent this, we need a secret key --- which brings us to message authentication codes.

---

## Message Authentication Codes (MACs)

Encryption provides **confidentiality** --- an eavesdropper cannot read the message. But encryption alone does not provide **integrity**. Many encryption schemes are **malleable**: an attacker can modify the ciphertext in ways that produce predictable changes in the plaintext, without ever decrypting it.

Consider a stream cipher (or CTR mode). The ciphertext is \\(C = M \oplus K_s\\) where \\(K_s\\) is the keystream. If an attacker flips bit \\(i\\) of \\(C\\), the decrypted plaintext has bit \\(i\\) flipped. The attacker does not know the plaintext, but they can manipulate it. If they know the structure of the message --- say, an AIS position report where the latitude field occupies bytes 12--19 --- they can flip specific bits to alter the reported position.

A **Message Authentication Code** (MAC) solves this. A MAC is a function that takes a secret key \\(K\\) and a message \\(M\\) and produces a fixed-length **tag**:

$$\text{Tag} = \text{MAC}(K, M)$$

The sender computes the tag and appends it to the message. The receiver, who shares the key \\(K\\), recomputes the tag and checks that it matches. An attacker who does not know \\(K\\) cannot forge a valid tag for a modified message.

The security requirement is **unforgeability**: an attacker who sees many valid \\((M_i, \text{Tag}_i)\\) pairs cannot produce a valid tag for any new message \\(M'\\) that was not previously authenticated. This must hold even if the attacker can adaptively choose which messages to see tags for.

### HMAC: A MAC from Hash Functions

The most widely used MAC construction is **HMAC** (Hash-based Message Authentication Code). Given a hash function \\(H\\), a key \\(K\\), and a message \\(M\\):

$$\text{HMAC}(K, M) = H\bigl((K \oplus \text{opad}) \;\|\; H((K \oplus \text{ipad}) \;\|\; M)\bigr)$$

Here \\(\text{opad}\\) is the byte `0x5c` repeated to fill a block, \\(\text{ipad}\\) is `0x36` repeated, and \\(\|\\) denotes concatenation. If \\(K\\) is shorter than the block size, it is padded with zeros; if longer, it is first hashed.

Why the double hashing? A naive approach like \\(H(K \| M)\\) is vulnerable to **length-extension attacks** in Merkle-Damgard hashes: knowing \\(H(K \| M)\\) lets you compute \\(H(K \| M \| M')\\) without knowing \\(K\\). The inner hash \\(H((K \oplus \text{ipad}) \| M)\\) produces an intermediate digest, and the outer hash \\(H((K \oplus \text{opad}) \| \text{inner})\\) prevents extension.

The security proof for HMAC reduces its unforgeability to the assumption that the compression function of \\(H\\) is a pseudorandom function (PRF). Under this assumption, HMAC is a secure PRF, which implies it is a secure MAC.

### Encrypt-then-MAC vs MAC-then-Encrypt vs Encrypt-and-MAC

When combining encryption and authentication, the order matters critically:

**Encrypt-then-MAC.** Encrypt the plaintext, then MAC the ciphertext: \\(C = \text{Enc}(K_e, M)\\), \\(\text{Tag} = \text{MAC}(K_m, C)\\). The receiver checks the tag *before* decrypting. This is provably secure: if the MAC rejects, the decryption never runs. Attacks against the decryption function (padding oracles, timing leaks) are completely blocked because tampered ciphertexts are rejected at the MAC check. This is the correct construction.

**MAC-then-Encrypt.** MAC the plaintext, then encrypt the plaintext and tag together: \\(\text{Tag} = \text{MAC}(K_m, M)\\), \\(C = \text{Enc}(K_e, M \| \text{Tag})\\). The receiver must decrypt before checking the tag. This opens the door to padding oracle attacks --- the receiver's decryption behavior (success or failure) leaks information. TLS 1.0--1.2 used this ordering with CBC mode, leading to the BEAST, Lucky 13, and POODLE attacks.

**Encrypt-and-MAC.** Encrypt the plaintext and independently MAC the plaintext: \\(C = \text{Enc}(K_e, M)\\), \\(\text{Tag} = \text{MAC}(K_m, M)\\). This leaks information about the plaintext through the tag, since the tag is computed on the plaintext. SSH originally used this construction.

### AEAD: The Modern Solution

Modern cryptography has converged on **Authenticated Encryption with Associated Data** (AEAD), which provides both confidentiality and integrity in a single primitive. The two dominant AEAD constructions are:

**AES-GCM** (Galois/Counter Mode): combines AES in CTR mode for encryption with a polynomial MAC (GHASH) over \\(\text{GF}(2^{128})\\) for authentication. It is hardware-accelerated on modern processors via AES-NI and PCLMULQDQ instructions.

**ChaCha20-Poly1305**: combines the ChaCha20 stream cipher with the Poly1305 MAC. Designed by Daniel Bernstein, it is fast in software without hardware acceleration, making it the preferred choice on mobile devices and platforms without AES-NI.

Both constructions take a key, a nonce (number used once), plaintext, and optional **associated data** (authenticated but not encrypted --- headers, metadata, routing information). They output ciphertext and an authentication tag. Reusing a nonce with the same key is catastrophic for both: AES-GCM completely breaks, leaking the authentication key; ChaCha20-Poly1305 leaks plaintext XORs.

**Connection to MUST.** The AIS spoofing attacks in the Baltic Sea exploit the complete absence of message authentication. AIS messages are broadcast in plaintext with no MAC. If AIS messages included an HMAC tag computed with a shared key known only to the vessel and authorized receivers, spoofing would require compromising that key. An attacker could still jam the signal, but they could not forge a valid position report. The International Maritime Organization has been studying authenticated AIS for years, but the installed base of millions of AIS transponders makes migration painfully slow.

---

## Digital Signatures

MACs have a limitation: they require a **shared secret key**. Both parties must know \\(K\\). This means:

1. A MAC cannot provide **non-repudiation** --- the receiver can forge messages too, since they have the same key.
2. Distributing shared keys to all parties in a large system is logistically difficult.

**Digital signatures** solve both problems by using asymmetric cryptography. The signer uses a **private key** \\(sk\\) to produce a signature; anyone with the corresponding **public key** \\(pk\\) can verify it.

### RSA Signatures

The simplest signature scheme, conceptually, uses RSA (from Part 3). Given an RSA key pair \\((n, e, d)\\):

**Signing:** Compute \\(s = m^d \bmod n\\), where \\(m\\) is the message (or, in practice, its hash).

**Verification:** Check that \\(s^e \equiv m \pmod{n}\\).

This works because \\((m^d)^e = m^{de} \equiv m \pmod{n}\\) by Euler's theorem. Only the holder of \\(d\\) can compute the signature, but anyone with \\(e\\) can verify it.

In practice, signing the raw message is insecure (it is homomorphic: \\(\text{sig}(m_1) \cdot \text{sig}(m_2) = \text{sig}(m_1 \cdot m_2)\\)). The **hash-then-sign** paradigm fixes this: compute \\(h = H(m)\\), apply a padding scheme (PSS --- Probabilistic Signature Scheme), then sign the padded hash. The padding adds randomness, preventing the multiplicative attack.

### ECDSA: Elliptic Curve Digital Signature Algorithm

ECDSA operates on an elliptic curve group of prime order \\(q\\) with generator \\(G\\) (from Part 3). The private key is an integer \\(d\\), and the public key is the point \\(Q = dG\\).

**Signing a message \\(m\\):**

1. Compute \\(e = H(m)\\), take the leftmost \\(\lceil \log_2 q \rceil\\) bits as an integer.
2. Choose a random \\(k \in [1, q-1]\\).
3. Compute the point \\(R = kG\\) and let \\(r = R_x \bmod q\\) (the x-coordinate modulo \\(q\\)).
4. Compute \\(s = k^{-1}(e + rd) \bmod q\\).
5. The signature is \\((r, s)\\).

**Verifying a signature \\((r, s)\\) on message \\(m\\):**

1. Compute \\(e = H(m)\\).
2. Compute \\(u_1 = es^{-1} \bmod q\\) and \\(u_2 = rs^{-1} \bmod q\\).
3. Compute the point \\(R' = u_1 G + u_2 Q\\).
4. Check that \\(R'_x \equiv r \pmod{q}\\).

Why does this work? Substituting \\(s = k^{-1}(e + rd)\\):

$$u_1 G + u_2 Q = es^{-1}G + rs^{-1}(dG) = s^{-1}(e + rd)G = kG = R$$

So the verification recovers the point \\(R\\) from the public information.

The critical security requirement: **the nonce \\(k\\) must be truly random and unique for every signature.** If \\(k\\) is reused for two different messages, the private key \\(d\\) can be extracted algebraically. This is not a theoretical concern --- in 2010, hackers extracted Sony's PlayStation 3 signing key because Sony used a constant \\(k\\) for every ECDSA signature.

### EdDSA (Ed25519): Deterministic Signatures

EdDSA eliminates the nonce catastrophe by deriving \\(k\\) deterministically from the private key and the message:

$$k = H(\text{prefix} \| M)$$

where \\(\text{prefix}\\) is derived from the private key during key generation. Since \\(k\\) is a deterministic function of \\(sk\\) and \\(M\\), it is never reused for different messages (unless the messages are identical, which produces the same signature --- harmless). No random number generator is needed at signing time.

Ed25519 uses the Edwards curve \\(-x^2 + y^2 = 1 + dx^2y^2\\) over \\(\mathbb{F}_p\\) where \\(p = 2^{255} - 19\\). It provides 128-bit security with 64-byte signatures and is significantly faster than ECDSA for both signing and verification.

**Connection to MUST.** Digital signatures are the foundation of software supply chain security. When the Swedish Armed Forces deploy firmware updates to communication systems, every binary must be signed with a key controlled by an authorized entity. The certificate chain that validates those signatures must trace back to a trusted root --- and that chain must work even in disconnected, contested environments where online verification is impossible. NATO's Allied Communications Publication (ACP) standards mandate specific signature algorithms for interoperability across member nations.

---

## Public Key Infrastructure (PKI)

Asymmetric cryptography gives us key pairs, and digital signatures let anyone verify a message's origin. But there is a foundational question: **how do you know that a public key belongs to the entity it claims to represent?**

If an attacker can substitute their own public key for the legitimate one, they can impersonate anyone. This is the **man-in-the-middle problem**, and Public Key Infrastructure (PKI) is the framework that solves it.

### The Certificate Chain

PKI organizes trust into a hierarchy:

**Root Certificate Authority (CA).** A root CA is a trusted entity whose public key is pre-installed in operating systems and browsers. There are roughly 150 root CAs trusted by major browsers. The root CA signs its own certificate (self-signed).

**Intermediate CA.** The root CA signs certificates for intermediate CAs, which handle day-to-day issuance. This limits the root CA's exposure --- its private key can be kept offline in a hardware security module (HSM).

**End-entity certificate.** The intermediate CA signs the certificate for a specific domain, server, or organization. This is the certificate your browser checks when you visit a website.

The **chain of trust** works by signature verification: the end-entity certificate is signed by the intermediate CA, the intermediate CA's certificate is signed by the root CA, and the root CA's certificate is pre-trusted. Verifying a certificate means verifying every signature up the chain to a trusted root.

### X.509 Certificates

An X.509 certificate contains:

- **Subject**: the entity the certificate identifies (e.g., `CN=www.forsvarsmakten.se`)
- **Issuer**: the CA that signed the certificate
- **Public key**: the subject's public key and algorithm (RSA, ECDSA, etc.)
- **Validity period**: not-before and not-after dates
- **Serial number**: unique identifier within the CA
- **Extensions**: Subject Alternative Names (SANs), key usage constraints, basic constraints (is this a CA certificate?), Certificate Transparency SCTs
- **Signature**: the issuer's digital signature over all the above

### Certificate Transparency

A rogue or compromised CA could issue a certificate for any domain. Certificate Transparency (CT) addresses this by requiring CAs to submit every certificate to public, append-only **CT logs** before issuance. These logs are cryptographically verifiable (using Merkle trees --- hash trees where each leaf is a certificate and each internal node is the hash of its children). Monitors and auditors watch these logs, and domain owners can detect unauthorized certificates for their domains.

### Revocation

When a private key is compromised, the corresponding certificate must be revoked. Two mechanisms exist:

**Certificate Revocation Lists (CRLs):** The CA publishes a signed list of revoked certificate serial numbers. Clients download the CRL and check it. Problem: CRLs can be large and slow to propagate.

**Online Certificate Status Protocol (OCSP):** The client queries the CA's OCSP responder in real time: "Is certificate serial number X still valid?" The responder returns a signed yes/no. Problem: privacy (the CA learns which sites you visit) and availability (if the OCSP responder is down, should the client fail open or fail closed?).

**OCSP stapling** is the practical solution: the server periodically fetches its own OCSP response and "staples" it to the TLS handshake. The client gets a fresh, signed validity proof without contacting the CA.

### The Web of Trust Alternative

PGP/GPG uses a decentralized **web of trust** instead of hierarchical CAs. Users sign each other's public keys, and trust propagates through the social graph. If Alice trusts Bob and Bob has signed Carol's key, Alice may choose to trust Carol's key. This model avoids the single-point-of-failure problem of CAs but scales poorly and requires active participation from users. It has largely been supplanted by the CA model for general-purpose use.

**Connection to MUST.** Military PKI systems face a unique challenge: they must work in **disconnected, intermittent, and limited-bandwidth (DIL)** environments. A Swedish naval vessel operating in the Baltic Sea may lose connectivity to certificate validation infrastructure. The vessel's systems must still validate certificates for software updates, secure communications, and authentication. This requires pre-loaded certificate chains, locally cached revocation information, and fallback policies for when validation infrastructure is unreachable. NATO's PKI interoperability standards (STANAG 4774/4778) define how member nations' certificate hierarchies cross-certify to enable coalition operations.

---

## TLS 1.3: Securing Every Connection

Transport Layer Security (TLS) is the protocol that secures nearly all internet communication. When your browser shows a padlock icon, TLS is running underneath. When MUST's systems communicate over networks, TLS (or its classified equivalents) protects the channel.

### Evolution

- **SSL 2.0 (1995):** Fundamentally broken. Deprecated.
- **SSL 3.0 (1996):** Better, but vulnerable to POODLE. Deprecated.
- **TLS 1.0 (1999):** Essentially SSL 3.1. Vulnerable to BEAST. Deprecated.
- **TLS 1.1 (2006):** Fixed BEAST. Still uses weak constructions. Deprecated.
- **TLS 1.2 (2008):** Solid when configured correctly. Still widely deployed.
- **TLS 1.3 (2018):** Major redesign. Removed legacy baggage. Current standard.

### What TLS 1.3 Removed (and Why)

TLS 1.3 is notable as much for what it eliminated as for what it added:

- **RSA key exchange**: removed because it lacks forward secrecy. If the server's long-term RSA key is later compromised, all recorded sessions can be decrypted.
- **Static Diffie-Hellman**: removed for the same reason. Only ephemeral (EC)DHE key exchange is permitted.
- **CBC mode ciphers**: removed because MAC-then-encrypt with CBC enabled padding oracle attacks (Lucky 13, POODLE).
- **RC4**: removed because of statistical biases that leak plaintext.
- **Compression**: removed because CRIME and BREACH attacks exploited compression ratios to extract secrets.
- **Renegotiation**: removed because it enabled man-in-the-middle injection attacks.

What remains is a small set of cipher suites, all using AEAD:

- `TLS_AES_256_GCM_SHA384`
- `TLS_AES_128_GCM_SHA256`
- `TLS_CHACHA20_POLY1305_SHA256`

### The TLS 1.3 Handshake

The TLS 1.3 handshake establishes a shared secret between client and server in a single round trip (1-RTT), down from two round trips in TLS 1.2.

<svg viewBox="0 0 700 620" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; width:100%; height:auto; display:block; margin:2em auto;">
  <defs>
    <marker id="arrowR" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#2563eb"/>
    </marker>
    <marker id="arrowL" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
      <polygon points="10 0, 0 3.5, 10 7" fill="#dc2626"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="700" height="620" rx="8" fill="#0f172a"/>

  <!-- Client / Server labels -->
  <text x="120" y="35" fill="#93c5fd" font-family="monospace" font-size="16" font-weight="bold" text-anchor="middle">CLIENT</text>
  <text x="580" y="35" fill="#fca5a5" font-family="monospace" font-size="16" font-weight="bold" text-anchor="middle">SERVER</text>

  <!-- Vertical lifelines -->
  <line x1="120" y1="50" x2="120" y2="590" stroke="#475569" stroke-width="2" stroke-dasharray="6,4"/>
  <line x1="580" y1="50" x2="580" y2="590" stroke="#475569" stroke-width="2" stroke-dasharray="6,4"/>

  <!-- 1. ClientHello -->
  <line x1="120" y1="80" x2="580" y2="120" stroke="#2563eb" stroke-width="2" marker-end="url(#arrowR)"/>
  <text x="350" y="75" fill="#93c5fd" font-family="monospace" font-size="12" text-anchor="middle">ClientHello</text>
  <text x="350" y="92" fill="#64748b" font-family="monospace" font-size="10" text-anchor="middle">cipher suites, key_share(s), SNI</text>

  <!-- Encryption begins label -->
  <rect x="260" y="135" width="180" height="22" rx="4" fill="#1e3a5f"/>
  <text x="350" y="151" fill="#fbbf24" font-family="monospace" font-size="10" text-anchor="middle">— encrypted from here —</text>

  <!-- 2. ServerHello -->
  <line x1="580" y1="170" x2="120" y2="210" stroke="#dc2626" stroke-width="2" marker-end="url(#arrowL)"/>
  <text x="350" y="168" fill="#fca5a5" font-family="monospace" font-size="12" text-anchor="middle">ServerHello</text>
  <text x="350" y="185" fill="#64748b" font-family="monospace" font-size="10" text-anchor="middle">selected cipher, key_share</text>

  <!-- 3. EncryptedExtensions -->
  <line x1="580" y1="240" x2="120" y2="280" stroke="#dc2626" stroke-width="2" marker-end="url(#arrowL)"/>
  <text x="350" y="238" fill="#fca5a5" font-family="monospace" font-size="12" text-anchor="middle">EncryptedExtensions</text>

  <!-- 4. Certificate -->
  <line x1="580" y1="310" x2="120" y2="350" stroke="#dc2626" stroke-width="2" marker-end="url(#arrowL)"/>
  <text x="350" y="308" fill="#fca5a5" font-family="monospace" font-size="12" text-anchor="middle">Certificate</text>

  <!-- 5. CertificateVerify -->
  <line x1="580" y1="380" x2="120" y2="420" stroke="#dc2626" stroke-width="2" marker-end="url(#arrowL)"/>
  <text x="350" y="378" fill="#fca5a5" font-family="monospace" font-size="12" text-anchor="middle">CertificateVerify</text>
  <text x="350" y="395" fill="#64748b" font-family="monospace" font-size="10" text-anchor="middle">signature over handshake transcript</text>

  <!-- 6. Server Finished -->
  <line x1="580" y1="450" x2="120" y2="490" stroke="#dc2626" stroke-width="2" marker-end="url(#arrowL)"/>
  <text x="350" y="448" fill="#fca5a5" font-family="monospace" font-size="12" text-anchor="middle">Finished</text>
  <text x="350" y="465" fill="#64748b" font-family="monospace" font-size="10" text-anchor="middle">HMAC over handshake transcript</text>

  <!-- 7. Client Finished -->
  <line x1="120" y1="520" x2="580" y2="560" stroke="#2563eb" stroke-width="2" marker-end="url(#arrowR)"/>
  <text x="350" y="518" fill="#93c5fd" font-family="monospace" font-size="12" text-anchor="middle">Finished</text>

  <!-- Application Data -->
  <rect x="200" y="578" width="300" height="24" rx="4" fill="#065f46"/>
  <text x="350" y="594" fill="#6ee7b7" font-family="monospace" font-size="12" text-anchor="middle">Application Data (encrypted)</text>

  <!-- RTT bracket -->
  <text x="670" y="335" fill="#94a3b8" font-family="monospace" font-size="11" text-anchor="middle" transform="rotate(90, 670, 335)">1-RTT handshake</text>
</svg>

**Step by step:**

**1. ClientHello.** The client sends:
- A list of supported cipher suites (e.g., `TLS_AES_256_GCM_SHA384`).
- One or more **key shares** --- the client's ephemeral public keys for key exchange. The client guesses which groups the server will accept and sends key shares for those groups (typically X25519 and P-256).
- The **Server Name Indication (SNI)** --- the hostname the client wants to connect to, so servers hosting multiple domains can select the right certificate.
- Supported signature algorithms for certificate verification.

**2. ServerHello.** The server selects:
- One cipher suite from the client's list.
- One key share matching one of the client's offered groups.

At this point, both sides can compute the **shared secret** using ECDHE: the client combines its private key with the server's public key share, and the server combines its private key with the client's public key share. Both arrive at the same shared point, from which the shared secret is derived.

**3. Key Derivation.** TLS 1.3 uses **HKDF** (HMAC-based Key Derivation Function) to derive all session keys from the shared secret. HKDF operates in two phases:
- **Extract**: \\(\text{PRK} = \text{HMAC}(\text{salt}, \text{input keying material})\\) --- concentrates entropy into a pseudorandom key.
- **Expand**: derives multiple keys of arbitrary length from the PRK using HMAC in a counter mode.

The handshake traffic keys (encrypting the remaining handshake messages) and the application traffic keys (encrypting application data) are derived through separate branches of the key schedule.

**4. Server messages (encrypted with handshake keys).** The server sends:
- **EncryptedExtensions**: extensions that are not needed for key exchange (ALPN, etc.).
- **Certificate**: the server's X.509 certificate chain.
- **CertificateVerify**: a digital signature over the entire handshake transcript (all messages so far), proving the server possesses the private key for the certificate.
- **Finished**: an HMAC over the handshake transcript, providing key confirmation.

**5. Client Finished.** The client verifies the server's certificate chain, checks the CertificateVerify signature, verifies the Finished HMAC, then sends its own Finished message.

**6. Application data flows**, encrypted with the application traffic keys.

### 0-RTT Resumption

TLS 1.3 supports **0-RTT** (zero round-trip time) resumption. If a client has previously connected to a server, it caches a **pre-shared key (PSK)** from the previous session. On reconnection, the client can send encrypted application data in the very first message, alongside the ClientHello.

The gain is latency: the first application data arrives at the server before any round trip completes.

The risk is **replay attacks**. Since 0-RTT data is sent before the server contributes any fresh randomness, an attacker who captures the 0-RTT data can replay it. The server cannot distinguish the replay from the original. For this reason, 0-RTT should only be used for idempotent requests (e.g., HTTP GET) --- never for state-changing operations (e.g., financial transactions).

### Perfect Forward Secrecy

TLS 1.3 mandates ephemeral key exchange. Every connection generates fresh ECDHE key pairs that are discarded after the session keys are derived. This provides **perfect forward secrecy (PFS)**: even if an attacker compromises the server's long-term private key (the one in the certificate), they cannot decrypt past sessions. Each session's shared secret was computed from ephemeral keys that no longer exist.

This is why TLS 1.3 removed RSA key exchange. In TLS 1.2 with RSA key exchange, the client encrypted the pre-master secret under the server's RSA public key. If the server's RSA private key was later compromised (or compelled by a court order), every recorded session could be decrypted. With ephemeral ECDHE, the long-term key is only used for *signatures* (authentication), not key exchange.

---

## The Signal Protocol: End-to-End Encryption for Messaging

TLS secures the channel between a client and a server. But in a messaging application, the *server is the threat*. If Alice sends a message to Bob through a server, TLS protects the message from eavesdroppers on the network --- but the server itself sees the plaintext. A compromised server, a malicious insider, or a government subpoena can access every message.

**End-to-end encryption (E2EE)** solves this: the message is encrypted on Alice's device and decrypted on Bob's device. The server transports ciphertext it cannot read. The Signal Protocol, designed by Moxie Marlinspike and Trevor Perrin, is the gold standard for E2EE messaging. It is used by Signal, WhatsApp (2+ billion users), Facebook Messenger, and Google Messages.

### X3DH: Extended Triple Diffie-Hellman Key Agreement

The first challenge is **key agreement for asynchronous messaging**. When Alice wants to message Bob, Bob might be offline. She cannot do an interactive Diffie-Hellman exchange. X3DH solves this using prekeys.

**Bob's published keys (uploaded to the server in advance):**

- \\(IK_B\\): Bob's long-term **identity key** (a Curve25519 key pair). This key represents Bob's cryptographic identity.
- \\(SPK_B\\): Bob's **signed prekey** --- an ephemeral public key signed by \\(IK_B\\). Rotated periodically (e.g., weekly).
- \\(OPK_B\\): Bob's **one-time prekeys** --- a set of ephemeral public keys, each used exactly once and then deleted.

**Alice initiates by computing four DH values:**

1. \\(\text{DH}_1 = \text{DH}(IK_A, SPK_B)\\) --- Alice's identity key with Bob's signed prekey. This proves Alice's identity to Bob.
2. \\(\text{DH}_2 = \text{DH}(EK_A, IK_B)\\) --- Alice's ephemeral key with Bob's identity key. This proves Bob's identity (only Bob has \\(IK_B\\)).
3. \\(\text{DH}_3 = \text{DH}(EK_A, SPK_B)\\) --- Alice's ephemeral key with Bob's signed prekey. This provides forward secrecy.
4. \\(\text{DH}_4 = \text{DH}(EK_A, OPK_B)\\) --- Alice's ephemeral key with Bob's one-time prekey. This provides replay protection (each one-time prekey is used once).

The shared secret is derived by concatenating all four DH outputs and passing them through HKDF:

$$SK = \text{HKDF}(\text{DH}_1 \| \text{DH}_2 \| \text{DH}_3 \| \text{DH}_4)$$

Each DH computation serves a specific purpose. If any one is removed, a specific attack becomes possible:

- Without \\(\text{DH}_1\\): no authentication of Alice to Bob.
- Without \\(\text{DH}_2\\): no authentication of Bob to Alice.
- Without \\(\text{DH}_3\\): no forward secrecy from the ephemeral key.
- Without \\(\text{DH}_4\\): no protection against replay of Alice's initial message.

Alice sends her initial message (encrypted with keys derived from \\(SK\\)) along with \\(IK_A\\), \\(EK_A\\), and an identifier for which of Bob's prekeys she used. When Bob comes online, he retrieves this bundle, performs the same four DH computations, derives the same \\(SK\\), and decrypts the message.

### The Double Ratchet Algorithm

X3DH establishes the initial shared secret. The **double ratchet** algorithm manages the ongoing key evolution for the entire conversation. It provides two critical properties:

**Forward secrecy:** If an attacker compromises the current keys, they cannot decrypt past messages.

**Post-compromise security (break-in recovery):** If an attacker compromises the current keys but the legitimate parties continue communicating, the attacker is *locked out* of future messages. The protocol "heals" itself.

The double ratchet combines two mechanisms:

**1. The symmetric-key ratchet (hash ratchet).** A chain key \\(CK_n\\) produces a message key and the next chain key:

$$MK_n = \text{HMAC}(CK_n, \texttt{0x01})$$

$$CK_{n+1} = \text{HMAC}(CK_n, \texttt{0x02})$$

Each message key \\(MK_n\\) encrypts one message. After use, \\(MK_n\\) is deleted. Because the chain is one-way (you cannot compute \\(CK_n\\) from \\(CK_{n+1}\\)), compromising the current chain key does not reveal past message keys. This provides forward secrecy within a single chain.

**2. The DH ratchet.** Each time the conversation "turns" (Alice sends, then Bob replies), the replying party generates a fresh ephemeral DH key pair and includes the public key in their message. The new DH exchange produces a new shared secret, which is mixed into the **root key**:

$$RK_{n+1}, CK_{n+1} = \text{HKDF}(RK_n, \text{DH}(EK_A^{(i)}, EK_B^{(j)}))$$

The new root key \\(RK_{n+1}\\) and chain key \\(CK_{n+1}\\) are derived from the old root key and the new DH output. This is what provides post-compromise security: even if an attacker has the current root key, the next DH ratchet step mixes in a new DH output that the attacker does not know (because they do not have the new ephemeral private key).

<svg viewBox="0 0 700 500" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; width:100%; height:auto; display:block; margin:2em auto;">
  <rect x="0" y="0" width="700" height="500" rx="8" fill="#0f172a"/>

  <!-- Title -->
  <text x="350" y="30" fill="#e2e8f0" font-family="monospace" font-size="14" font-weight="bold" text-anchor="middle">Double Ratchet Key Derivation</text>

  <!-- Root chain (vertical, center) -->
  <text x="350" y="65" fill="#fbbf24" font-family="monospace" font-size="12" text-anchor="middle">Root Chain</text>

  <!-- RK nodes -->
  <circle cx="350" cy="100" r="20" fill="#92400e" stroke="#fbbf24" stroke-width="2"/>
  <text x="350" y="105" fill="#fef3c7" font-family="monospace" font-size="11" text-anchor="middle">RK₀</text>

  <circle cx="350" cy="210" r="20" fill="#92400e" stroke="#fbbf24" stroke-width="2"/>
  <text x="350" y="215" fill="#fef3c7" font-family="monospace" font-size="11" text-anchor="middle">RK₁</text>

  <circle cx="350" cy="320" r="20" fill="#92400e" stroke="#fbbf24" stroke-width="2"/>
  <text x="350" y="325" fill="#fef3c7" font-family="monospace" font-size="11" text-anchor="middle">RK₂</text>

  <!-- Root chain arrows -->
  <line x1="350" y1="120" x2="350" y2="190" stroke="#fbbf24" stroke-width="2" marker-end="url(#arrowR)"/>
  <line x1="350" y1="230" x2="350" y2="300" stroke="#fbbf24" stroke-width="2" marker-end="url(#arrowR)"/>

  <!-- DH inputs to root chain -->
  <text x="265" y="150" fill="#a78bfa" font-family="monospace" font-size="10" text-anchor="end">DH(EKₐ¹,EK_b¹)</text>
  <line x1="270" y1="155" x2="340" y2="195" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="4,3"/>

  <text x="265" y="260" fill="#a78bfa" font-family="monospace" font-size="10" text-anchor="end">DH(EKₐ²,EK_b¹)</text>
  <line x1="270" y1="265" x2="340" y2="305" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="4,3"/>

  <!-- Sending chain (right side, from RK0) -->
  <text x="560" y="85" fill="#6ee7b7" font-family="monospace" font-size="12" text-anchor="middle">Sending Chain</text>

  <line x1="370" y1="100" x2="440" y2="100" stroke="#6ee7b7" stroke-width="2"/>
  <circle cx="470" cy="100" r="18" fill="#064e3b" stroke="#6ee7b7" stroke-width="2"/>
  <text x="470" y="104" fill="#d1fae5" font-family="monospace" font-size="10" text-anchor="middle">CK₀</text>

  <line x1="488" y1="100" x2="540" y2="100" stroke="#6ee7b7" stroke-width="2"/>
  <circle cx="570" cy="100" r="18" fill="#064e3b" stroke="#6ee7b7" stroke-width="2"/>
  <text x="570" y="104" fill="#d1fae5" font-family="monospace" font-size="10" text-anchor="middle">CK₁</text>

  <!-- Message keys from sending chain -->
  <line x1="470" y1="118" x2="470" y2="148" stroke="#34d399" stroke-width="1.5"/>
  <rect x="450" y="150" width="40" height="20" rx="4" fill="#065f46"/>
  <text x="470" y="164" fill="#a7f3d0" font-family="monospace" font-size="9" text-anchor="middle">MK₀</text>

  <line x1="570" y1="118" x2="570" y2="148" stroke="#34d399" stroke-width="1.5"/>
  <rect x="550" y="150" width="40" height="20" rx="4" fill="#065f46"/>
  <text x="570" y="164" fill="#a7f3d0" font-family="monospace" font-size="9" text-anchor="middle">MK₁</text>

  <!-- Receiving chain (right side, from RK1) -->
  <text x="560" y="195" fill="#93c5fd" font-family="monospace" font-size="12" text-anchor="middle">Receiving Chain</text>

  <line x1="370" y1="210" x2="440" y2="210" stroke="#93c5fd" stroke-width="2"/>
  <circle cx="470" cy="210" r="18" fill="#1e3a5f" stroke="#93c5fd" stroke-width="2"/>
  <text x="470" y="214" fill="#bfdbfe" font-family="monospace" font-size="10" text-anchor="middle">CK₂</text>

  <line x1="488" y1="210" x2="540" y2="210" stroke="#93c5fd" stroke-width="2"/>
  <circle cx="570" cy="210" r="18" fill="#1e3a5f" stroke="#93c5fd" stroke-width="2"/>
  <text x="570" y="214" fill="#bfdbfe" font-family="monospace" font-size="10" text-anchor="middle">CK₃</text>

  <!-- Message keys from receiving chain -->
  <line x1="470" y1="228" x2="470" y2="258" stroke="#60a5fa" stroke-width="1.5"/>
  <rect x="450" y="260" width="40" height="20" rx="4" fill="#1e3a5f"/>
  <text x="470" y="274" fill="#bfdbfe" font-family="monospace" font-size="9" text-anchor="middle">MK₂</text>

  <line x1="570" y1="228" x2="570" y2="258" stroke="#60a5fa" stroke-width="1.5"/>
  <rect x="550" y="260" width="40" height="20" rx="4" fill="#1e3a5f"/>
  <text x="570" y="274" fill="#bfdbfe" font-family="monospace" font-size="9" text-anchor="middle">MK₃</text>

  <!-- New sending chain (from RK2) -->
  <text x="560" y="305" fill="#6ee7b7" font-family="monospace" font-size="12" text-anchor="middle">Sending Chain</text>

  <line x1="370" y1="320" x2="440" y2="320" stroke="#6ee7b7" stroke-width="2"/>
  <circle cx="470" cy="320" r="18" fill="#064e3b" stroke="#6ee7b7" stroke-width="2"/>
  <text x="470" y="324" fill="#d1fae5" font-family="monospace" font-size="10" text-anchor="middle">CK₄</text>

  <line x1="488" y1="320" x2="540" y2="320" stroke="#6ee7b7" stroke-width="2"/>
  <circle cx="570" cy="320" r="18" fill="#064e3b" stroke="#6ee7b7" stroke-width="2"/>
  <text x="570" y="324" fill="#d1fae5" font-family="monospace" font-size="10" text-anchor="middle">CK₅</text>

  <!-- Message keys from new sending chain -->
  <line x1="470" y1="338" x2="470" y2="368" stroke="#34d399" stroke-width="1.5"/>
  <rect x="450" y="370" width="40" height="20" rx="4" fill="#065f46"/>
  <text x="470" y="384" fill="#a7f3d0" font-family="monospace" font-size="9" text-anchor="middle">MK₄</text>

  <line x1="570" y1="338" x2="570" y2="368" stroke="#34d399" stroke-width="1.5"/>
  <rect x="550" y="370" width="40" height="20" rx="4" fill="#065f46"/>
  <text x="570" y="384" fill="#a7f3d0" font-family="monospace" font-size="9" text-anchor="middle">MK₅</text>

  <!-- Legend -->
  <rect x="40" y="410" width="620" height="75" rx="6" fill="#1e293b"/>
  <circle cx="70" cy="435" r="8" fill="#92400e" stroke="#fbbf24" stroke-width="1.5"/>
  <text x="85" y="439" fill="#cbd5e1" font-family="monospace" font-size="10">Root Key — stepped by DH ratchet</text>

  <circle cx="70" cy="460" r="8" fill="#064e3b" stroke="#6ee7b7" stroke-width="1.5"/>
  <text x="85" y="464" fill="#cbd5e1" font-family="monospace" font-size="10">Sending Chain Key — stepped by hash ratchet</text>

  <rect x="395" y="427" width="30" height="16" rx="3" fill="#065f46"/>
  <text x="432" y="439" fill="#cbd5e1" font-family="monospace" font-size="10">Message Key — used once, then deleted</text>

  <line x1="395" y1="460" x2="425" y2="460" stroke="#a78bfa" stroke-width="1.5" stroke-dasharray="4,3"/>
  <text x="432" y="464" fill="#cbd5e1" font-family="monospace" font-size="10">New DH exchange input</text>
</svg>

**The combined ratchet in action:**

1. Alice sends messages to Bob. Each message advances the **sending chain** (symmetric ratchet): \\(CK_0 \to CK_1 \to CK_2 \ldots\\), producing message keys \\(MK_0, MK_1, MK_2, \ldots\\). Each \\(MK_i\\) encrypts one message and is deleted.

2. Bob replies. He generates a fresh ephemeral key pair and includes the public key. This triggers a **DH ratchet step**: Alice and Bob perform a new DH exchange, deriving a new root key and chain key. Bob now has a new receiving chain.

3. Alice replies. She generates a fresh ephemeral key pair. Another DH ratchet step. New root key, new chain key.

Every turn of conversation ratchets the DH keys forward. Within a turn, the symmetric ratchet advances per message. Old keys are deleted. The result: an attacker who compromises Alice's device at time \\(t\\) cannot read messages from before \\(t\\) (forward secrecy) and, once Alice and Bob exchange a few more messages, cannot read messages after \\(t\\) either (post-compromise security).

### Sealed Sender: Hiding Metadata

Signal goes beyond encrypting message content. The **sealed sender** feature hides the sender's identity from the Signal server. Instead of labeling each message with "from Alice to Bob," the sender encrypts the entire envelope (including the sender's identity) using the recipient's identity key. The server sees only "deliver this opaque blob to Bob." Bob decrypts the envelope and discovers who sent it.

This does not achieve perfect metadata protection --- the server still sees timing, message sizes, and that *someone* sent a message to Bob --- but it significantly raises the bar for surveillance.

**Connection to MUST.** Secure communications for intelligence operatives require more than content encryption. If an adversary can determine *who is communicating with whom* and *when* --- even without reading message content --- they can map the organizational structure of an intelligence network. The MUST report highlights Sweden as one of the world's most digitally connected countries, making metadata analysis particularly potent. MUST cooperates with FRA (Försvarets radioanstalt) on signals intelligence and with Säkerhetspolisen (the Security Service) on counter-intelligence, and all three organizations must protect their internal communications not just from content interception but from traffic analysis. The Signal Protocol's double ratchet provides the cryptographic foundation, but operational security requires additional measures: varying communication patterns, using anonymous access networks, and minimizing the metadata footprint.

---

## GNSS Authentication and Signal Integrity

The Baltic Sea scenarios from the opening are attacks on **Global Navigation Satellite Systems** (GNSS). Understanding why these attacks work --- and how cryptography can prevent them --- requires understanding how GNSS works.

### How GNSS Works: Trilateration

A GNSS receiver determines its position by measuring the time it takes for signals to arrive from multiple satellites. Each satellite continuously broadcasts its position and the precise time of transmission. The receiver computes the distance to each satellite as:

$$d_i = c \cdot (t_{\text{receive}} - t_{\text{transmit},i})$$

where \\(c\\) is the speed of light. With distances to at least four satellites, the receiver solves a system of equations to determine its three spatial coordinates \\((x, y, z)\\) and its clock offset \\(\Delta t\\):

$$(x - x_i)^2 + (y - y_i)^2 + (z - z_i)^2 = \bigl(c \cdot (t_{\text{receive}} + \Delta t - t_{\text{transmit},i})\bigr)^2$$

for \\(i = 1, 2, 3, 4\\). The clock offset is necessary because the receiver's clock is not synchronized to satellite time with nanosecond precision.

### Why Civilian GNSS Signals Are Unauthenticated

The GPS civilian signal (L1 C/A), GLONASS civilian signals, and early Galileo signals broadcast navigation messages in the clear. There is no authentication. The receiver has no way to verify that a signal genuinely came from a satellite rather than from a ground-based transmitter.

This is a design choice from the 1970s. GPS was built for military and civilian dual use. The military signals (P(Y) code, now M-code) are encrypted and authenticated. The civilian signals were intentionally left open to enable global adoption without key distribution.

### Spoofing Attacks

A **GNSS spoofing attack** broadcasts counterfeit signals that mimic legitimate satellite signals but carry false timing or position data. The attack works because:

1. The signal structure is publicly documented.
2. The receiver has no way to distinguish genuine from counterfeit signals.
3. The attacker can broadcast at higher power than the satellites (which are 20,200 km away), causing the receiver to lock onto the spoofed signals.

A sophisticated spoofing attack gradually shifts the victim's computed position, making the transition imperceptible. A less sophisticated attack can simply overpower the genuine signals, causing the receiver to report a wildly incorrect position --- which is what has been observed near Gotland.

```python
import numpy as np

def trilateration_2d(satellites, distances):
    """
    Demonstrate 2D trilateration from 3 satellites.
    satellites: array of shape (3, 2) — satellite (x, y) positions
    distances: array of shape (3,) — measured distances
    Returns estimated receiver position.
    """
    # Linearize by subtracting equations pairwise
    # (x - x1)^2 + (y - y1)^2 = d1^2  ... (i)
    # (x - x2)^2 + (y - y2)^2 = d2^2  ... (ii)
    # (ii) - (i): 2(x1-x2)x + 2(y1-y2)y = d1^2 - d2^2 - x1^2 + x2^2 - y1^2 + y2^2

    x, y = satellites[:, 0], satellites[:, 1]
    d = distances

    # Build linear system Ax = b from pairwise differences
    A = np.array([
        [2 * (x[0] - x[1]), 2 * (y[0] - y[1])],
        [2 * (x[0] - x[2]), 2 * (y[0] - y[2])]
    ])

    b = np.array([
        d[1]**2 - d[0]**2 - x[1]**2 + x[0]**2 - y[1]**2 + y[0]**2,
        d[2]**2 - d[0]**2 - x[2]**2 + x[0]**2 - y[2]**2 + y[0]**2
    ])

    position = np.linalg.solve(A, b)
    return position

# Satellites at known positions (km)
sats = np.array([
    [0.0, 20200.0],     # Satellite 1 — overhead
    [15000.0, 13000.0], # Satellite 2
    [-12000.0, 16000.0] # Satellite 3
])

# True receiver position: Gotland (approx 57.6°N, 18.3°E in local coords)
true_pos = np.array([100.0, 0.0])  # 100 km east of origin

# True distances
true_distances = np.linalg.norm(sats - true_pos, axis=1)

# Spoofed distances: attacker shifts position 5 km east
spoofed_pos = np.array([105.0, 0.0])
spoofed_distances = np.linalg.norm(sats - spoofed_pos, axis=1)

# Solve trilateration with true and spoofed distances
est_true = trilateration_2d(sats, true_distances)
est_spoofed = trilateration_2d(sats, spoofed_distances)

print(f"True position:     ({est_true[0]:.2f}, {est_true[1]:.2f}) km")
print(f"Spoofed position:  ({est_spoofed[0]:.2f}, {est_spoofed[1]:.2f}) km")
print(f"Position error:    {np.linalg.norm(est_spoofed - est_true):.2f} km")
```

### Galileo OSNMA: Navigation Message Authentication

The European Galileo system is deploying **OSNMA** (Open Service Navigation Message Authentication), the first civilian GNSS authentication system. OSNMA uses **ECDSA** signatures to authenticate navigation messages.

The challenge is bandwidth. GNSS navigation messages are transmitted at very low data rates (50--500 bps for the navigation data). A full ECDSA signature (64 bytes for P-256) would consume an enormous fraction of the available bandwidth. OSNMA addresses this through:

**TESLA protocol** (Timed Efficient Stream Loss-tolerant Authentication): a delayed-disclosure MAC scheme. The satellite broadcasts MACs computed with a key that has not yet been revealed. After a delay, the key is disclosed. The receiver buffers the data, receives the key, and retroactively verifies the MACs. The key chain is structured so that each key can be verified against a previously authenticated key, forming a one-way chain anchored by an ECDSA-signed root key.

$$K_0 \xleftarrow{H} K_1 \xleftarrow{H} K_2 \xleftarrow{H} \ldots \xleftarrow{H} K_N$$

Key \\(K_i\\) is the hash of \\(K_{i+1}\\). The root \\(K_0\\) is signed with ECDSA. To verify \\(K_j\\), compute \\(j\\) hash iterations and check against \\(K_0\\). The satellite reveals keys in reverse order: \\(K_N, K_{N-1}, \ldots\\), each verifiable against the previous disclosure.

This achieves asymmetric authentication (only the satellite can generate valid MACs, since only it knows future keys) with symmetric-key efficiency (MAC verification is fast and small).

**Connection to MUST.** Russian GNSS interference in the Baltic Sea is not speculative --- it is documented, ongoing, and affects both military and civilian navigation. Galileo OSNMA, when fully deployed, will provide civilian receivers with the ability to reject spoofed signals. However, OSNMA has limitations: it does not prevent jamming (signal denial), the authentication delay makes it less useful for safety-of-life applications requiring real-time integrity, and military-grade GPS (M-code) already provides authentication for authorized users. The Swedish total defence concept requires multi-layered positioning resilience: authenticated GNSS, inertial navigation systems, terrestrial radio navigation, and visual/radar fixes.

---

## The Limits of Protocols

Cryptographic protocols protect against specific, well-defined threat models. Understanding their limits is as important as understanding their strengths.

### Metadata: What Encryption Doesn't Hide

End-to-end encryption hides message content. It does not hide:

- **Who is communicating with whom** (unless sealed sender or anonymous routing is used)
- **When communications occur** (timing patterns)
- **How much data is exchanged** (volume)
- **Where the communicating parties are** (IP addresses, cell tower connections)
- **The frequency and pattern of communication** (burst patterns during crises)

Metadata analysis is extraordinarily powerful. Former NSA and CIA director Michael Hayden stated: "We kill people based on metadata." The MUST report documents AI-generated disinformation campaigns using automated botnets --- identifying these campaigns relies on traffic pattern analysis, which is itself a form of metadata analysis.

### Traffic Analysis

Even with encrypted content and hidden endpoints, traffic analysis can reveal critical information:

- **Volume correlation**: if a classified briefing occurs at 09:00 and encrypted traffic spikes from a specific subnet at 09:05, the correlation is informative.
- **Timing analysis**: response times can reveal whether a message was read and acted upon.
- **Network topology**: observing which nodes communicate can reconstruct organizational hierarchies.

Countermeasures include constant-rate traffic (padding all channels to a fixed bandwidth), decoy traffic, mix networks (Tor), and careful operational discipline.

### Side Channels

Cryptographic protocols prove security in mathematical models. Real implementations run on physical hardware that leaks information through channels the model does not capture:

**Timing attacks.** If a comparison function returns early on the first mismatched byte, an attacker can determine the correct value byte by byte by measuring response time. Constant-time comparison functions are essential for any security-critical comparison.

**Power analysis.** The power consumption of a processor varies depending on the operations being performed. Differential power analysis (DPA) can extract AES keys by statistically analyzing power traces across many encryptions. This is relevant for embedded GNSS receivers and military hardware that might be physically captured.

**Electromagnetic emanations.** Processors emit electromagnetic radiation correlated with their computations. TEMPEST shielding (a classified set of standards for limiting electromagnetic emanations) is a real and active concern for Swedish military and intelligence facilities.

**Cache-timing attacks.** On shared hardware (cloud servers), an attacker can measure cache access patterns to infer which memory addresses a victim process is accessing, potentially extracting cryptographic keys. This is why sensitive cryptographic operations should use constant-time, cache-oblivious implementations.

### The Human Factor

The most expensive cryptographic protocol in the world is useless if:

- A user chooses a weak password and the key derivation function is not strong enough.
- A system administrator stores private keys in an unencrypted file.
- A software update introduces a vulnerability in the random number generator (the Debian OpenSSL bug of 2008 reduced the key space to approximately 32,000 keys).
- Social engineering convinces an authorized user to reveal credentials.
- An insider with legitimate access exfiltrates data before encryption.

The MUST report emphasizes that Sweden faces hybrid threats combining military and non-military means: disinformation, election interference, cyber attacks, disruption of critical trade flows, and sabotage of infrastructure. Many of these threats target the human layer rather than the cryptographic layer. The intelligence reform currently underway --- establishing a new civilian foreign intelligence service --- is partly a response to the recognition that technical cryptographic defences must be complemented by institutional resilience.

### Preview: When the Math Breaks

Everything in this article assumes that certain mathematical problems are computationally hard: factoring large integers (RSA), computing discrete logarithms in elliptic curve groups (ECDSA, X25519, ECDHE), and finding preimages of hash functions.

A sufficiently large quantum computer breaks the first two completely. Shor's algorithm factors integers and computes discrete logarithms in polynomial time. Every RSA key, every ECDSA signature, every Diffie-Hellman key exchange, every TLS handshake, every Signal Protocol session --- all retroactively breakable if the adversary recorded the ciphertext and later builds a quantum computer.

This is not a distant theoretical concern. Intelligence agencies, including those documented in the MUST report as adversaries, are presumed to be recording encrypted traffic today for future decryption --- a strategy called "harvest now, decrypt later."

Part 5 addresses the response: post-quantum cryptography, lattice-based schemes, and the ongoing migration to quantum-resistant algorithms.

---

*Next: [Part 5 --- Post-Quantum Cryptography: Lattice Problems, CRYSTALS, and the Race to Replace Everything](/2026/03/16/quantum-threat-post-quantum-cryptography.html)*
