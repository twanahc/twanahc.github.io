---
layout: post
title: "Fourier Analysis: Decomposing Signals into Frequencies and Why Video Models Care"
date: 2026-01-02
category: math
---

Every signal you have ever encountered --- a sound wave, an image, a video frame, the loss curve of a training run --- is the superposition of simple oscillations at different frequencies. This is not a metaphor. It is a precise mathematical fact, and understanding it at a deep level unlocks an enormous amount of practical insight: why blurring is low-pass filtering, why edges in images correspond to high frequencies, why aliasing creates visual artifacts, and why the convolution theorem makes neural network operations vastly more efficient.

This post builds the entire theory of Fourier analysis from scratch. We start with pure intuition, formalize with derivations, extend to the continuous and discrete transforms, prove the convolution theorem, and then connect everything to images and video. Every step is motivated and explained.

---

## Table of Contents

1. [Periodic Functions and Why They Matter](#periodic-functions-and-why-they-matter)
2. [The Core Idea: Any Function as a Sum of Sines and Cosines](#the-core-idea-any-function-as-a-sum-of-sines-and-cosines)
3. [Fourier Series: Deriving the Coefficients](#fourier-series-deriving-the-coefficients)
4. [The Fourier Transform: From Periodic to Non-Periodic](#the-fourier-transform-from-periodic-to-non-periodic)
5. [The DFT and FFT: The Discrete Case](#the-dft-and-fft-the-discrete-case)
6. [The Convolution Theorem](#the-convolution-theorem)
7. [Power Spectral Density](#power-spectral-density)
8. [Applications to Images and Video](#applications-to-images-and-video)
9. [Python Simulations](#python-simulations)
10. [Conclusion](#conclusion)

---

## Periodic Functions and Why They Matter

A function $f(t)$ is **periodic** with period $T$ if $f(t + T) = f(t)$ for all $t$. The simplest periodic functions are sines and cosines:

$$f(t) = A\sin(2\pi f_0 t + \phi)$$

where $A$ is the **amplitude** (how tall the wave is), $f_0$ is the **frequency** (how many complete cycles per second, measured in Hertz), and $\phi$ is the **phase** (where in the cycle the wave starts).

Why do periodic functions deserve special treatment? Three reasons.

**Reason 1: Nature oscillates.** Sound is pressure oscillations in air. Light is electromagnetic field oscillations. AC electricity oscillates. Pendulums oscillate. Planetary orbits are periodic. Seasons are periodic. The heartbeat is approximately periodic. When you build mathematical tools for periodic phenomena, you get tools that apply nearly everywhere.

**Reason 2: Linearity.** Most physical systems and many computational systems are approximately **linear** --- the response to a sum of inputs equals the sum of responses to each input individually. If you decompose an arbitrary input into simple periodic components, you can analyze the system's response to each component independently and then add the responses back together. This turns hard problems into collections of trivial ones.

**Reason 3: Eigenfunction property.** Sines and cosines (or equivalently, complex exponentials $e^{i\omega t}$) are **eigenfunctions** of linear time-invariant (LTI) systems. What does this mean? If you feed a sine wave at frequency $\omega$ into an LTI system, you get out a sine wave at the same frequency $\omega$, just scaled in amplitude and shifted in phase. The system cannot create new frequencies. This property is what makes frequency-domain analysis so powerful --- each frequency component passes through the system independently.

---

## The Core Idea: Any Function as a Sum of Sines and Cosines

Here is the central claim of Fourier analysis: **any reasonable function can be written as a (possibly infinite) sum of sines and cosines.**

Let us build intuition with sound. When you hear a musical note, say middle C on a piano, you are not hearing a pure sine wave. You are hearing a fundamental frequency (261.6 Hz for middle C) plus a collection of **harmonics** --- oscillations at integer multiples of the fundamental frequency (523.2 Hz, 784.8 Hz, 1046.4 Hz, ...). What distinguishes a piano from a violin or a flute playing the same note is the relative amplitudes of these harmonics. The fundamental frequency determines the pitch; the harmonic content determines the **timbre**.

This means the waveform of any musical instrument, no matter how complex it looks in the time domain, is just a sum of pure sine waves at specific frequencies with specific amplitudes.

Now extend this idea: it is not just musical instruments. **Any** periodic function --- square waves, sawtooth waves, triangle waves, arbitrary wiggly periodic shapes --- can be decomposed into sines and cosines. And as we will see later, even non-periodic functions have a frequency decomposition, just with a continuous spectrum instead of discrete harmonics.

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#e8e8e8">Building a Square Wave from Sines</text>
  <!-- Axes -->
  <line x1="50" y1="80" x2="650" y2="80" stroke="#444" stroke-width="1"/>
  <line x1="50" y1="180" x2="650" y2="180" stroke="#444" stroke-width="1"/>
  <line x1="50" y1="280" x2="650" y2="280" stroke="#444" stroke-width="1"/>
  <!-- Labels -->
  <text x="40" y="85" text-anchor="end" font-size="11" fill="#999">sin(t)</text>
  <text x="40" y="185" text-anchor="end" font-size="11" fill="#999">+ sin(3t)/3</text>
  <text x="40" y="285" text-anchor="end" font-size="11" fill="#999">+ sin(5t)/5 + ...</text>
  <!-- Fundamental sine -->
  <path d="M50,80 Q200,40 350,80 Q500,120 650,80" stroke="#2563eb" stroke-width="2.5" fill="none"/>
  <!-- Sum of 1st and 3rd harmonic (approximation) -->
  <path d="M50,180 Q125,135 200,155 Q275,175 350,180 Q425,225 500,205 Q575,185 650,180" stroke="#dc2626" stroke-width="2.5" fill="none"/>
  <!-- More harmonics approaching square wave -->
  <path d="M50,280 L50,250 L170,250 L170,280 L170,310 L290,310 L290,280 L290,250 L410,250 L410,280 L410,310 L530,310 L530,280 L530,250 L650,250" stroke="#16a34a" stroke-width="2.5" fill="none" stroke-dasharray="6,3"/>
  <path d="M50,280 Q90,248 130,252 Q160,254 170,280 Q180,306 210,310 Q250,312 290,280 Q310,252 350,250 Q390,252 410,280 Q420,306 450,310 Q490,312 530,280 Q550,252 590,250 Q630,252 650,258" stroke="#16a34a" stroke-width="2.5" fill="none"/>
  <!-- Legend -->
  <rect x="200" y="320" width="15" height="3" fill="#2563eb"/>
  <text x="220" y="325" font-size="10" fill="#d4d4d4">Fundamental</text>
  <rect x="310" y="320" width="15" height="3" fill="#dc2626"/>
  <text x="330" y="325" font-size="10" fill="#d4d4d4">2 terms</text>
  <rect x="400" y="320" width="15" height="3" fill="#16a34a"/>
  <text x="420" y="325" font-size="10" fill="#d4d4d4">Many terms (approaching square)</text>
</svg>

---

## Fourier Series: Deriving the Coefficients

Now let us formalize this. Given a periodic function $f(t)$ with period $T$, we claim it can be written as:

$$f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty}\left[a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right)\right]$$

The question is: how do we find the coefficients $a_n$ and $b_n$? The answer comes from **orthogonality**.

### Orthogonality of Trigonometric Functions

Two functions $g(t)$ and $h(t)$ are **orthogonal** over an interval $[0, T]$ if their **inner product** is zero:

$$\langle g, h \rangle = \int_0^T g(t) \, h(t) \, dt = 0$$

This is exactly analogous to orthogonal vectors in linear algebra, where $\vec{u} \cdot \vec{v} = 0$. Just as orthogonal vectors point in independent directions, orthogonal functions represent independent "directions" in function space.

The trigonometric functions satisfy these orthogonality relations (all integrals over one full period $[0, T]$):

$$\int_0^T \cos\left(\frac{2\pi m t}{T}\right) \cos\left(\frac{2\pi n t}{T}\right) dt = \begin{cases} T & \text{if } m = n = 0 \\ T/2 & \text{if } m = n \neq 0 \\ 0 & \text{if } m \neq n \end{cases}$$

$$\int_0^T \sin\left(\frac{2\pi m t}{T}\right) \sin\left(\frac{2\pi n t}{T}\right) dt = \begin{cases} T/2 & \text{if } m = n \neq 0 \\ 0 & \text{if } m \neq n \end{cases}$$

$$\int_0^T \cos\left(\frac{2\pi m t}{T}\right) \sin\left(\frac{2\pi n t}{T}\right) dt = 0 \quad \text{for all } m, n$$

Why are these true? The product-to-sum identities. For example, $\cos(A)\cos(B) = \frac{1}{2}[\cos(A-B) + \cos(A+B)]$. When $m \neq n$, both $\cos(A-B)$ and $\cos(A+B)$ oscillate an integer number of times over the period and integrate to zero. When $m = n$, the $\cos(A-B) = \cos(0) = 1$ term survives, contributing $T/2$.

### Extracting the Coefficients

Now the derivation of the coefficients becomes elegant. To find $a_m$, multiply both sides of the Fourier series by $\cos(2\pi m t / T)$ and integrate over one period:

$$\int_0^T f(t) \cos\left(\frac{2\pi m t}{T}\right) dt = \int_0^T \left[\frac{a_0}{2} + \sum_{n=1}^{\infty} a_n \cos\left(\frac{2\pi n t}{T}\right) + b_n \sin\left(\frac{2\pi n t}{T}\right)\right] \cos\left(\frac{2\pi m t}{T}\right) dt$$

By orthogonality, every term on the right vanishes except the one where $n = m$ in the cosine sum. The sine terms all vanish because cosine is orthogonal to sine. We get:

$$\int_0^T f(t) \cos\left(\frac{2\pi m t}{T}\right) dt = a_m \cdot \frac{T}{2}$$

Solving:

$$\boxed{a_n = \frac{2}{T}\int_0^T f(t) \cos\left(\frac{2\pi n t}{T}\right) dt}$$

By the same argument, multiplying by $\sin(2\pi m t / T)$ and integrating:

$$\boxed{b_n = \frac{2}{T}\int_0^T f(t) \sin\left(\frac{2\pi n t}{T}\right) dt}$$

And for $a_0$ (the constant / DC component):

$$a_0 = \frac{2}{T}\int_0^T f(t) \, dt$$

This is the average value of $f(t)$ over one period, multiplied by 2 (but divided by 2 in the series, so the DC component is just the average).

What is happening conceptually? We are **projecting** the function $f(t)$ onto each basis function, exactly as we project a vector onto a coordinate axis using the dot product. The Fourier coefficients are the "coordinates" of $f$ in the trigonometric basis.

### Complex Exponential Form

It is often cleaner to use complex exponentials. Using Euler's formula, $e^{i\theta} = \cos\theta + i\sin\theta$, we can write:

$$f(t) = \sum_{n=-\infty}^{\infty} c_n \, e^{i 2\pi n t / T}$$

where the complex coefficients are:

$$c_n = \frac{1}{T}\int_0^T f(t) \, e^{-i 2\pi n t / T} \, dt$$

The relationship is $c_0 = a_0/2$, and for $n > 0$: $c_n = (a_n - ib_n)/2$, $c_{-n} = (a_n + ib_n)/2$. This form is compact and will lead naturally to the Fourier transform.

---

## The Fourier Transform: From Periodic to Non-Periodic

The Fourier series works for periodic functions. But most real signals are not periodic --- a spoken sentence, a single video clip, a training loss curve. How do we handle these?

The key insight: **a non-periodic function is the limit of a periodic function whose period goes to infinity.**

Let us make this precise. Start with the complex Fourier series of a function with period $T$:

$$f(t) = \sum_{n=-\infty}^{\infty} c_n \, e^{i 2\pi n t / T}, \qquad c_n = \frac{1}{T}\int_{-T/2}^{T/2} f(t) \, e^{-i 2\pi n t / T} \, dt$$

Define the frequency spacing $\Delta f = 1/T$ and the $n$-th frequency as $f_n = n \Delta f = n/T$. Substituting back:

$$f(t) = \sum_{n=-\infty}^{\infty} \left[\frac{1}{T}\int_{-T/2}^{T/2} f(t') \, e^{-i 2\pi f_n t'} \, dt'\right] e^{i 2\pi f_n t}$$

Rewrite this as:

$$f(t) = \sum_{n=-\infty}^{\infty} \underbrace{\left[\int_{-T/2}^{T/2} f(t') \, e^{-i 2\pi f_n t'} \, dt'\right]}_{\text{call this } \hat{f}(f_n)} e^{i 2\pi f_n t} \, \Delta f$$

Now take $T \to \infty$. The spacing $\Delta f \to 0$, the discrete frequencies $f_n$ become a continuous variable $\xi$, the sum becomes an integral, and the limits of the inner integral extend to $\pm\infty$:

$$\boxed{f(t) = \int_{-\infty}^{\infty} \hat{f}(\xi) \, e^{i 2\pi \xi t} \, d\xi}$$

where

$$\boxed{\hat{f}(\xi) = \int_{-\infty}^{\infty} f(t) \, e^{-i 2\pi \xi t} \, dt}$$

The first equation is the **inverse Fourier transform** (synthesis --- building the function from its frequencies). The second is the **Fourier transform** (analysis --- decomposing the function into its frequencies). Together, they form a transform pair: $f(t) \leftrightarrow \hat{f}(\xi)$.

The variable $\xi$ is a **continuous frequency**. The function $\hat{f}(\xi)$ tells you the amplitude and phase of the frequency-$\xi$ component in $f(t)$. A periodic function has a discrete spectrum (spikes at the harmonics). A non-periodic function has a continuous spectrum (energy spread across all frequencies).

---

## The DFT and FFT: The Discrete Case

In practice, we never have continuous functions. We have sampled data: $N$ values $x_0, x_1, \ldots, x_{N-1}$ taken at uniform intervals. The **Discrete Fourier Transform (DFT)** is the version of the Fourier transform for this setting.

### Definition

$$X_k = \sum_{n=0}^{N-1} x_n \, e^{-i 2\pi k n / N}, \qquad k = 0, 1, \ldots, N-1$$

The inverse:

$$x_n = \frac{1}{N}\sum_{k=0}^{N-1} X_k \, e^{i 2\pi k n / N}, \qquad n = 0, 1, \ldots, N-1$$

$X_k$ is the complex amplitude of the $k$-th frequency component. The magnitude $|X_k|$ gives the amplitude; the argument $\arg(X_k)$ gives the phase.

### Computational Complexity

Computing the DFT naively requires $N$ multiplications for each of $N$ output values, giving $O(N^2)$ complexity. For a 1-megapixel image ($N \approx 10^6$), this is $10^{12}$ operations --- impractical.

The **Fast Fourier Transform (FFT)**, discovered by Cooley and Tukey in 1965 (though Gauss had a version in 1805), reduces this to $O(N \log N)$. The idea is divide-and-conquer: split the DFT into even-indexed and odd-indexed terms.

$$X_k = \sum_{m=0}^{N/2-1} x_{2m} \, e^{-i 2\pi k (2m) / N} + \sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-i 2\pi k (2m+1) / N}$$

$$= \underbrace{\sum_{m=0}^{N/2-1} x_{2m} \, e^{-i 2\pi k m / (N/2)}}_{E_k} + e^{-i 2\pi k / N} \underbrace{\sum_{m=0}^{N/2-1} x_{2m+1} \, e^{-i 2\pi k m / (N/2)}}_{O_k}$$

Both $E_k$ and $O_k$ are DFTs of size $N/2$. So we have reduced one DFT of size $N$ to two DFTs of size $N/2$ plus $O(N)$ work for the combination. The recurrence $T(N) = 2T(N/2) + O(N)$ solves to $T(N) = O(N \log N)$.

For our 1-megapixel image, this is $10^6 \times 20 = 2 \times 10^7$ operations --- a speedup of 50,000x. This is why the FFT has been called "the most important numerical algorithm of our lifetime."

### The Nyquist-Shannon Sampling Theorem and Aliasing

When you sample a continuous signal at rate $f_s$ (samples per second), you can only faithfully represent frequencies up to $f_s / 2$, called the **Nyquist frequency**. Frequencies above $f_s / 2$ get "folded back" into the range $[0, f_s/2]$, appearing as lower-frequency imposters. This is **aliasing**.

A concrete example: if you sample a 900 Hz sine wave at 1000 Hz, it looks identical to a 100 Hz sine wave. The 900 Hz signal has aliased to 100 Hz because $900 = 1000 - 100$.

In video, aliasing shows up as the wagon-wheel effect (spoked wheels appearing to rotate backward), moire patterns on fine textures, and staircase artifacts on diagonal lines. Video models must deal with these artifacts both in their training data and in their generated outputs.

---

## The Convolution Theorem

This is one of the most powerful results in all of applied mathematics. It says:

$$\boxed{f * g \leftrightarrow \hat{f} \cdot \hat{g}}$$

**Convolution in the time/spatial domain corresponds to multiplication in the frequency domain.**

Let us define convolution and then prove this.

### Convolution

The convolution of two functions $f$ and $g$ is:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) \, g(t - \tau) \, d\tau$$

Operationally: you slide $g$ across $f$, multiply them pointwise at each position, and integrate. Convolution computes a weighted running average. In neural networks, convolutional layers compute discrete versions of exactly this operation.

### Proof of the Convolution Theorem

We want to show that $\widehat{f * g}(\xi) = \hat{f}(\xi) \cdot \hat{g}(\xi)$.

Start by taking the Fourier transform of the convolution:

$$\widehat{f * g}(\xi) = \int_{-\infty}^{\infty} \left[\int_{-\infty}^{\infty} f(\tau) \, g(t-\tau) \, d\tau\right] e^{-i2\pi\xi t} \, dt$$

Swap the order of integration (justified by Fubini's theorem when both functions are well-behaved):

$$= \int_{-\infty}^{\infty} f(\tau) \left[\int_{-\infty}^{\infty} g(t-\tau) \, e^{-i2\pi\xi t} \, dt\right] d\tau$$

In the inner integral, substitute $u = t - \tau$, so $t = u + \tau$ and $dt = du$:

$$= \int_{-\infty}^{\infty} f(\tau) \left[\int_{-\infty}^{\infty} g(u) \, e^{-i2\pi\xi(u+\tau)} \, du\right] d\tau$$

$$= \int_{-\infty}^{\infty} f(\tau) \, e^{-i2\pi\xi\tau} \left[\int_{-\infty}^{\infty} g(u) \, e^{-i2\pi\xi u} \, du\right] d\tau$$

The inner integral is $\hat{g}(\xi)$ --- it does not depend on $\tau$, so it factors out:

$$= \hat{g}(\xi) \int_{-\infty}^{\infty} f(\tau) \, e^{-i2\pi\xi\tau} \, d\tau = \hat{g}(\xi) \cdot \hat{f}(\xi)$$

$$\boxed{\widehat{f * g}(\xi) = \hat{f}(\xi) \cdot \hat{g}(\xi)} \quad \blacksquare$$

### Why This Matters Computationally

Direct convolution of two signals of length $N$ takes $O(N^2)$ operations. Using the convolution theorem: take the FFT of both signals ($O(N\log N)$ each), multiply elementwise ($O(N)$), and take the inverse FFT ($O(N\log N)$). Total: $O(N\log N)$.

For large kernels or signals, this is a dramatic speedup. In deep learning, some implementations of convolutional layers use FFT-based convolution for exactly this reason, particularly when the kernel size is large.

There is also a dual result: **multiplication in time corresponds to convolution in frequency.** This means windowing a signal (multiplying it by a finite-duration window function) spreads its spectrum --- a fundamental tradeoff between time resolution and frequency resolution known as the **uncertainty principle** of signal processing.

---

## Power Spectral Density

The **power spectral density (PSD)** of a signal tells you how power (energy per unit time) is distributed across frequencies:

$$S(\xi) = |\hat{f}(\xi)|^2$$

This is the squared magnitude of the Fourier transform. The phase information is discarded --- PSD tells you *how much* of each frequency is present, but not *when* those frequencies occur.

Several important spectral shapes appear throughout science and engineering:

- **White noise**: $S(\xi) = \text{constant}$. Equal power at all frequencies. Random pixel noise in images.
- **Pink noise** ($1/f$ noise): $S(\xi) \propto 1/|\xi|$. Equal power per octave. Natural images approximately follow this.
- **Brownian noise** ($1/f^2$ noise): $S(\xi) \propto 1/|\xi|^2$. Smoother, more correlated fluctuations.

The observation that natural images have approximately $1/f$ power spectra has deep implications. It means natural images have a specific statistical structure that encoders and decoders in generative models implicitly learn to exploit.

---

## Applications to Images and Video

### The 2D Fourier Transform

Images are 2D signals, so we need the 2D Fourier transform:

$$\hat{I}(\xi_x, \xi_y) = \int\int I(x, y) \, e^{-i2\pi(\xi_x x + \xi_y y)} \, dx \, dy$$

Now $\xi_x$ and $\xi_y$ are spatial frequencies --- oscillations per unit distance in the $x$ and $y$ directions. The 2D Fourier transform decomposes an image into sinusoidal gratings (2D sine waves) at every combination of horizontal and vertical frequency.

The 2D DFT is separable: you can compute it by doing 1D FFTs along all rows, then along all columns (or vice versa). This is efficient and is exactly what `numpy.fft.fft2` does.

### Why Blurring is Low-Pass Filtering

Blurring an image means replacing each pixel with a weighted average of its neighbors. Mathematically, this is convolution with a blur kernel (e.g., a Gaussian). By the convolution theorem, this is multiplication in the frequency domain.

A Gaussian kernel in the spatial domain is also a Gaussian in the frequency domain (the Fourier transform of a Gaussian is a Gaussian --- one of the beautiful self-duality properties). A wide Gaussian in frequency means keeping most frequencies; a narrow Gaussian means keeping only low frequencies.

Blurring = convolution with a spatially wide kernel = multiplication by a narrow Gaussian in frequency = suppressing high frequencies = **low-pass filtering**.

### Why Edges are High-Frequency

An edge in an image is a sharp transition --- pixel intensity changes abruptly over a short distance. To represent a sharp transition as a sum of sinusoids, you need high-frequency components (fast oscillations can change quickly). Conversely, smooth gradual changes require only low-frequency components.

This is precisely the Fourier series of a step function: you need infinitely many harmonics to build a perfect discontinuity. The more harmonics you include, the sharper the edge. The fewer you include, the blurrier the transition.

### Implications for Video Models

Video generation models operate in latent spaces that are typically produced by a VAE (Variational Autoencoder) with convolutional layers. The convolutional architecture inherently performs operations in a way that is connected to the frequency domain through the convolution theorem.

Several practical consequences:

1. **Training data with aliasing artifacts** introduces spurious high-frequency patterns that the model may learn to reproduce.
2. **Temporal aliasing** in video (insufficient frame rate for fast motion) creates the same problems in the time dimension.
3. **Progressive generation** (generating low-resolution first, then upsampling) is essentially generating the low-frequency content first and then adding high-frequency detail --- a coarse-to-fine strategy that mirrors the frequency decomposition.
4. **Spectral normalization** in discriminators constrains the Lipschitz constant of the network, which is directly a frequency-domain constraint.

---

## Python Simulations

### Simulation 1: Square Wave Fourier Decomposition

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 2, 1000)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for idx, N in enumerate([1, 3, 9, 49]):
    ax = axes[idx // 2, idx % 2]
    # Build partial sum of Fourier series for square wave
    # Square wave = (4/pi) * sum_{k=0}^{N} sin((2k+1)*2*pi*t) / (2k+1)
    y = np.zeros_like(t)
    for k in range(N + 1):
        n = 2 * k + 1
        y += (4 / np.pi) * np.sin(n * 2 * np.pi * t) / n

    # True square wave for reference
    square = np.sign(np.sin(2 * np.pi * t))

    ax.plot(t, square, 'k--', alpha=0.3, label='True square wave')
    ax.plot(t, y, 'b-', linewidth=1.5, label=rf'${N+1}$ terms')
    ax.set_title(rf'$N = {N+1}$ harmonic{"s" if N > 0 else ""}')
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$f(t)$')
    ax.legend(fontsize=9)
    ax.set_ylim(-1.5, 1.5)
    ax.grid(True, alpha=0.3)

plt.suptitle(r'Fourier Series Convergence to a Square Wave', fontsize=14)
plt.tight_layout()
plt.savefig('fourier_square_wave.png', dpi=150, bbox_inches='tight')
plt.show()
```

This simulation shows the Gibbs phenomenon: even with many terms, the Fourier partial sums overshoot near the discontinuity by about 9% (approximately $\frac{1}{2}\int_0^\pi \frac{\sin t}{t}dt - \frac{\pi}{4} \approx 0.0895\pi$). This overshoot never disappears as you add more terms --- it just gets narrower but stays the same height. A beautiful and counterintuitive result.

### Simulation 2: 2D FFT of an Image

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a synthetic image with known frequency content
N = 256
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# Image: sum of 2D sinusoidal gratings at different frequencies
img = (np.sin(2 * np.pi * 5 * X) +           # 5 cycles horizontally
       0.5 * np.sin(2 * np.pi * 20 * Y) +     # 20 cycles vertically
       0.3 * np.sin(2 * np.pi * (10*X + 10*Y)))  # diagonal

# Compute 2D FFT
F = np.fft.fft2(img)
F_shifted = np.fft.fftshift(F)  # Center the zero-frequency component
magnitude = np.log1p(np.abs(F_shifted))  # Log scale for visibility

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title(r'Spatial Domain (Image)')
axes[0].axis('off')

axes[1].imshow(magnitude, cmap='hot')
axes[1].set_title(r'Frequency Domain ($\log$ Magnitude)')
axes[1].axis('off')

plt.suptitle(r'2D Fourier Transform of a Synthetic Image', fontsize=14)
plt.tight_layout()
plt.savefig('fourier_2d_fft.png', dpi=150, bbox_inches='tight')
plt.show()
```

You will see bright dots in the frequency domain at exactly the frequencies of the component gratings. The horizontal grating appears as vertical dots (perpendicular --- spatial frequency is measured in the direction of variation), and vice versa. The diagonal grating appears as dots along the diagonal of the frequency plane.

### Simulation 3: Convolution Theorem Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a signal and a kernel
N = 256
t = np.linspace(0, 1, N, endpoint=False)
signal = np.zeros(N)
signal[80:180] = 1.0  # Rectangular pulse

# Gaussian blur kernel
sigma = 5
kernel = np.exp(-0.5 * (np.arange(N) - N//2)**2 / sigma**2)
kernel = np.roll(kernel, -N//2)  # Center at index 0 for FFT
kernel /= kernel.sum()

# Method 1: Direct convolution (via numpy)
conv_direct = np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(kernel)))

# Method 2: FFT-based (identical by theorem)
F_signal = np.fft.fft(signal)
F_kernel = np.fft.fft(kernel)
F_product = F_signal * F_kernel
conv_fft = np.real(np.fft.ifft(F_product))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(t, signal, 'b-', linewidth=1.5)
axes[0, 0].set_title(r'Signal $f(t)$')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, np.roll(kernel, N//2), 'r-', linewidth=1.5)
axes[0, 1].set_title(r'Kernel $g(t)$ (Gaussian)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t, np.abs(np.fft.fftshift(F_signal)), 'b-', alpha=0.7, label=r'$|\hat{f}|$')
axes[1, 0].plot(t, np.abs(np.fft.fftshift(F_kernel)) * 50, 'r-', alpha=0.7, label=r'$|\hat{g}| \times 50$')
axes[1, 0].set_title(r'Frequency Domain (Magnitudes)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, conv_fft, 'g-', linewidth=2, label=r'FFT method')
axes[1, 1].plot(t, conv_direct, 'k--', linewidth=1, label=r'Direct (verify)')
axes[1, 1].set_title(r'Convolution Result $(f * g)$')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle(r'Convolution Theorem: Spatial vs Frequency Domain', fontsize=14)
plt.tight_layout()
plt.savefig('convolution_theorem.png', dpi=150, bbox_inches='tight')
plt.show()
```

Both methods produce identical results, confirming the convolution theorem. The Gaussian kernel acts as a low-pass filter: in the frequency domain, it has a Gaussian shape that attenuates high frequencies, smoothing the sharp edges of the rectangular pulse.

### Simulation 4: Aliasing Demonstration

```python
import numpy as np
import matplotlib.pyplot as plt

# True signal: 5 Hz sine wave
f_true = 5  # Hz
t_fine = np.linspace(0, 1, 10000)
y_fine = np.sin(2 * np.pi * f_true * t_fine)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sample_rates = [50, 12, 7]
titles = [
    rf'$f_s = {sample_rates[0]}$ Hz (well above Nyquist)',
    rf'$f_s = {sample_rates[1]}$ Hz (above Nyquist)',
    rf'$f_s = {sample_rates[2]}$ Hz (BELOW Nyquist — aliasing!)'
]

for ax, fs, title in zip(axes, sample_rates, titles):
    t_sampled = np.arange(0, 1, 1/fs)
    y_sampled = np.sin(2 * np.pi * f_true * t_sampled)

    ax.plot(t_fine, y_fine, 'b-', alpha=0.3, label=rf'True ${f_true}$ Hz signal')
    ax.stem(t_sampled, y_sampled, linefmt='r-', markerfmt='ro', basefmt='k-',
            label=rf'Samples at ${fs}$ Hz')

    # Reconstruct by fitting a sine through the samples
    # (simplified: just connect the dots for visual)
    ax.plot(t_sampled, y_sampled, 'r--', alpha=0.5)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r'Time $t$ (s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle(r'Aliasing: What Happens When You Sample Below the Nyquist Rate', fontsize=13)
plt.tight_layout()
plt.savefig('aliasing_demo.png', dpi=150, bbox_inches='tight')
plt.show()
```

At $f_s = 7$ Hz, the Nyquist frequency is 3.5 Hz, which is below the signal's 5 Hz. The samples can be fit perfectly by a 2 Hz sine wave ($5 - 7 = -2$, and $|-2| = 2$). The 5 Hz signal has aliased to 2 Hz --- information is irrecoverably lost.

---

## Conclusion

Fourier analysis is not just a mathematical curiosity --- it is the lens through which we understand signals, images, and video at the most fundamental level. The key ideas:

1. **Any function decomposes into sinusoidal components**, with coefficients determined by inner products (projections onto the orthogonal trigonometric basis).
2. **The Fourier transform** extends this from periodic to non-periodic functions by taking the period to infinity.
3. **The FFT** makes this computationally feasible, reducing $O(N^2)$ to $O(N\log N)$.
4. **The convolution theorem** converts convolution (expensive) to multiplication (cheap) in the frequency domain.
5. **The Nyquist theorem** tells us the sampling rate must be at least twice the highest frequency present --- violating this causes aliasing.
6. **Images and video** are 2D and 3D signals where blurring is low-pass filtering, edges are high-frequency content, and many generation architectures implicitly work in a frequency-aware manner.

Every convolutional layer in a video generation model, every downsampling operation, every blur kernel --- they are all frequency-domain operations in disguise. Understanding them in this language gives you a principled framework for diagnosing artifacts, designing architectures, and understanding why certain techniques work.
