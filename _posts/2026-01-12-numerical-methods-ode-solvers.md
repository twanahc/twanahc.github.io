---
layout: post
title: "Numerical Methods and ODE Solvers: How Computers Solve the Equations Models Cannot"
date: 2026-01-12
category: math
---

Most differential equations cannot be solved analytically. This is not a failure of mathematics --- it is a fundamental fact about the structure of nonlinear systems. The three-body problem in gravitational mechanics, turbulent fluid flow, the dynamics of neural networks during training, the reverse diffusion process that generates images from noise --- all of these are governed by differential equations that have no closed-form solution. We must solve them numerically.

This post develops numerical methods from first principles. We start with the simplest possible approach (Euler's method, which is just a Taylor expansion truncated after the first term), build up to the workhorse of scientific computing (Runge-Kutta 4), explore stability and adaptive methods, extend to partial differential equations, and connect everything to the numerical solvers that power modern diffusion models.

---

## Table of Contents

1. [Why We Need Numerical Methods](#why-we-need-numerical-methods)
2. [The Euler Method](#the-euler-method)
3. [Error Analysis: Local and Global](#error-analysis-local-and-global)
4. [The Implicit Euler Method and Stiffness](#the-implicit-euler-method-and-stiffness)
5. [Higher-Order Methods: Midpoint and Heun](#higher-order-methods-midpoint-and-heun)
6. [Runge-Kutta Methods](#runge-kutta-methods)
7. [Adaptive Step Size Control](#adaptive-step-size-control)
8. [Stability Regions](#stability-regions)
9. [Numerical Integration](#numerical-integration)
10. [Finite Differences for PDEs](#finite-differences-for-pdes)
11. [Connections to Diffusion Models](#connections-to-diffusion-models)
12. [Python: From Lorenz Chaos to the Heat Equation](#python-from-lorenz-chaos-to-the-heat-equation)

---

## Why We Need Numerical Methods

An **ordinary differential equation (ODE)** specifies how a quantity changes as a function of one variable (usually time). The general initial value problem is:

$$\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0$$

where $y(t)$ is the unknown function, $f(t, y)$ is a known function describing the rate of change, and $y_0$ is the initial condition.

Some ODEs have closed-form solutions. The simplest: $\frac{dy}{dt} = ky$ has solution $y(t) = y_0 e^{kt}$. Separation of variables, integrating factors, and other analytical techniques can handle a range of problems.

But most real-world ODEs cannot be solved this way. Consider the **Lorenz system**:

$$\frac{dx}{dt} = \sigma(y - x), \quad \frac{dy}{dt} = x(\rho - z) - y, \quad \frac{dz}{dt} = xy - \beta z$$

This is a system of three coupled nonlinear ODEs. The $xy$ and $xz$ terms make it nonlinear, and no closed-form solution exists. Yet this system describes atmospheric convection and exhibits chaotic behavior --- small changes in initial conditions lead to exponentially diverging trajectories. The only way to understand its behavior is to solve it numerically.

The fundamental idea behind all numerical ODE solvers is the same: **discretize time into small steps, and use the differential equation to approximate how the solution changes over each step**. The differences between methods come down to how cleverly they approximate that change.

---

## The Euler Method

The **Euler method** is the simplest numerical ODE solver, and understanding it deeply is the key to understanding everything else.

Start from the Taylor expansion of $y(t + h)$ around $t$:

$$y(t + h) = y(t) + h \, y'(t) + \frac{h^2}{2} y''(t) + \frac{h^3}{6} y'''(t) + \cdots$$

Now use the ODE: $y'(t) = f(t, y(t))$. Substitute and drop everything beyond the first-order term:

$$y(t + h) \approx y(t) + h \, f(t, y(t))$$

This is the Euler method. Starting from $y_0$ at $t_0$, we march forward in steps of size $h$:

$$y_{n+1} = y_n + h \, f(t_n, y_n)$$

Geometrically, at each step we compute the slope $f(t_n, y_n)$ at the current point, draw a straight line with that slope for a distance $h$, and call the endpoint our new approximation. We are following the tangent line instead of the actual curve.

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="max-width: 700px; display: block; margin: 2em auto;">
  <defs>
    <marker id="arrowE" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#d4d4d4"/>
    </marker>
  </defs>
  <text x="350" y="25" text-anchor="middle" font-size="14" font-weight="bold" fill="#d4d4d4">Euler's Method: Following Tangent Lines</text>
  <!-- Axes -->
  <line x1="60" y1="300" x2="660" y2="300" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowE)"/>
  <line x1="60" y1="300" x2="60" y2="40" stroke="#d4d4d4" stroke-width="1.5" marker-end="url(#arrowE)"/>
  <text x="660" y="320" text-anchor="end" font-size="12" fill="#d4d4d4">t</text>
  <text x="45" y="45" text-anchor="middle" font-size="12" fill="#d4d4d4">y</text>
  <!-- True curve -->
  <path d="M 80,260 Q 200,180 320,120 Q 440,80 600,70" fill="none" stroke="#2196F3" stroke-width="2.5"/>
  <text x="610" y="65" font-size="11" fill="#2196F3">True solution y(t)</text>
  <!-- Euler steps -->
  <circle cx="80" cy="260" r="4" fill="#F44336"/>
  <line x1="80" y1="260" x2="220" y2="200" stroke="#F44336" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="220" cy="200" r="4" fill="#F44336"/>
  <line x1="220" y1="200" x2="360" y2="155" stroke="#F44336" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="360" cy="155" r="4" fill="#F44336"/>
  <line x1="360" y1="155" x2="500" y2="130" stroke="#F44336" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="500" cy="130" r="4" fill="#F44336"/>
  <line x1="500" y1="130" x2="600" y2="115" stroke="#F44336" stroke-width="2" stroke-dasharray="6,3"/>
  <circle cx="600" cy="115" r="4" fill="#F44336"/>
  <text x="610" y="120" font-size="11" fill="#F44336">Euler approx.</text>
  <!-- Step labels -->
  <text x="150" y="245" font-size="10" fill="#F44336">slope = f(t₀,y₀)</text>
  <text x="290" y="190" font-size="10" fill="#F44336">slope = f(t₁,y₁)</text>
  <!-- Error annotation -->
  <line x1="600" y1="70" x2="600" y2="115" stroke="#9C27B0" stroke-width="1.5"/>
  <text x="625" y="95" font-size="10" fill="#9C27B0">Error</text>
  <!-- h label -->
  <line x1="80" y1="310" x2="220" y2="310" stroke="#d4d4d4" stroke-width="1"/>
  <line x1="80" y1="305" x2="80" y2="315" stroke="#d4d4d4" stroke-width="1"/>
  <line x1="220" y1="305" x2="220" y2="315" stroke="#d4d4d4" stroke-width="1"/>
  <text x="150" y="328" text-anchor="middle" font-size="11" fill="#d4d4d4">h (step size)</text>
</svg>

The method is embarrassingly simple, and that is both its strength (easy to understand and implement) and its weakness (the error accumulates quickly).

---

## Error Analysis: Local and Global

Two types of error matter in numerical ODE solving:

**Local truncation error** is the error made in a single step, assuming the starting point is exact. From the Taylor expansion, the Euler method drops the $\frac{h^2}{2} y''$ term and beyond, so:

$$\text{Local error} = \frac{h^2}{2} y''(\xi) = O(h^2)$$

for some $\xi$ between $t_n$ and $t_{n+1}$. The local error is proportional to $h^2$.

**Global error** is the total accumulated error after many steps. To integrate from $t_0$ to $T$, we take $N = (T - t_0)/h$ steps. Each step contributes $O(h^2)$ error, and these errors compound:

$$\text{Global error} = N \times O(h^2) = \frac{T - t_0}{h} \times O(h^2) = O(h)$$

The global error is proportional to $h$. This is why Euler is called a **first-order method**: halving the step size halves the error. To get one more digit of accuracy, you need 10 times as many steps. This is catastrophically inefficient for problems requiring high accuracy.

The **order** of a numerical method is defined by how the global error scales with step size $h$. An order-$p$ method has global error $O(h^p)$: halving $h$ reduces the error by a factor of $2^p$. Euler is order 1. We want higher.

---

## The Implicit Euler Method and Stiffness

The standard Euler method evaluates $f$ at the *current* point $(t_n, y_n)$. The **implicit Euler method** (or backward Euler) evaluates it at the *next* point:

$$y_{n+1} = y_n + h \, f(t_{n+1}, y_{n+1})$$

This looks circular --- $y_{n+1}$ appears on both sides --- and in general requires solving a nonlinear equation at each step (using Newton's method or fixed-point iteration). This is more expensive per step, so why bother?

The answer is **stiffness**. A stiff ODE is one where the solution has components that vary on very different timescales. Consider:

$$\frac{dy}{dt} = -1000y + 1000t + 1$$

The exact solution is $y(t) = t + Ce^{-1000t}$ for some constant $C$. The exponential term $e^{-1000t}$ decays incredibly rapidly --- after $t = 0.01$, it is essentially zero, and the solution smoothly follows $y \approx t$. But the explicit Euler method does not know this.

For explicit Euler to be stable on this problem, we need $|1 + h \lambda| < 1$ where $\lambda = -1000$, giving $h < 2/1000 = 0.002$. Even though the solution is nearly constant for $t > 0.01$, explicit Euler requires tiny steps *forever* to avoid blowing up. This is stiffness: the step size is limited by stability, not accuracy.

Implicit Euler, by contrast, is **unconditionally stable** for such problems (its stability region covers the entire left half of the complex plane). It can take large steps once the transient has decayed. The cost per step is higher, but far fewer steps are needed.

**Stiffness** in the context of machine learning shows up more than you might think. Neural ODE solvers, the reverse-time SDE in diffusion models, and any system where some modes decay much faster than others --- all of these benefit from implicit or semi-implicit methods.

---

## Higher-Order Methods: Midpoint and Heun

To improve on Euler, we need to use the slope information more cleverly within each step.

### The Midpoint Method (Modified Euler)

Instead of using the slope at the start of the interval, evaluate the slope at the midpoint:

1. Take a half-step with Euler: $k_1 = f(t_n, y_n)$, then $y_{\text{mid}} = y_n + \frac{h}{2} k_1$
2. Evaluate the slope at the midpoint: $k_2 = f(t_n + \frac{h}{2}, y_{\text{mid}})$
3. Take a full step using the midpoint slope: $y_{n+1} = y_n + h \, k_2$

A Taylor expansion analysis shows the local error is $O(h^3)$, making the global error $O(h^2)$. This is a **second-order method**: halving $h$ reduces the error by a factor of 4.

### Heun's Method (Improved Euler)

An alternative second-order approach:

1. Compute the slope at the start: $k_1 = f(t_n, y_n)$
2. Use Euler to predict the endpoint: $\tilde{y}_{n+1} = y_n + h \, k_1$
3. Compute the slope at the predicted endpoint: $k_2 = f(t_{n+1}, \tilde{y}_{n+1})$
4. Average the two slopes: $y_{n+1} = y_n + \frac{h}{2}(k_1 + k_2)$

This is a **predictor-corrector** method. Euler predicts, then the average corrects. Both slopes give information about how the solution curves, producing a more accurate result than either alone.

Both methods use two function evaluations per step (versus one for Euler). They gain an order of accuracy for doubling the computational cost --- a good trade.

---

## Runge-Kutta Methods

The Runge-Kutta (RK) family generalizes the midpoint and Heun methods. The idea: evaluate the slope at several carefully chosen points within each step, then take a weighted average.

### Derivation of RK4

The classical fourth-order Runge-Kutta method (RK4) is:

$$k_1 = f(t_n, y_n)$$

$$k_2 = f\left(t_n + \frac{h}{2}, \, y_n + \frac{h}{2} k_1\right)$$

$$k_3 = f\left(t_n + \frac{h}{2}, \, y_n + \frac{h}{2} k_2\right)$$

$$k_4 = f(t_n + h, \, y_n + h \, k_3)$$

$$y_{n+1} = y_n + \frac{h}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)$$

Let us understand what each slope represents:

- **$k_1$** is the slope at the beginning of the interval. This is what Euler uses alone.
- **$k_2$** is the slope at the midpoint, using $k_1$ to estimate where the midpoint is. This is what the midpoint method uses.
- **$k_3$** is the slope at the midpoint again, but using the *corrected* midpoint estimate from $k_2$. This is a re-evaluation: $k_2$ gives a better estimate of the midpoint, so $k_3$ should be a better midpoint slope.
- **$k_4$** is the slope at the end of the interval, using $k_3$ to estimate the endpoint.

The weights $(1, 2, 2, 1)/6$ come from matching the Taylor expansion to fourth order. To see why these specific weights work, expand $y(t+h)$ in a Taylor series:

$$y(t+h) = y + hy' + \frac{h^2}{2}y'' + \frac{h^3}{6}y''' + \frac{h^4}{24}y^{(4)} + O(h^5)$$

Each $k_i$ can also be Taylor-expanded. The coefficients of $k_1$ through $k_4$ in the weighted average are chosen so that when you expand the RK4 formula in powers of $h$, it matches the Taylor series through $h^4$. The local error is $O(h^5)$, giving a global error of $O(h^4)$. Fourth-order: halving $h$ reduces the error by a factor of 16.

The weights $(1, 2, 2, 1)/6$ are reminiscent of Simpson's rule for numerical integration --- and this is not a coincidence. Simpson's rule uses the same weighting to integrate a function exactly for polynomials up to degree 3, and the RK4 method achieves analogous accuracy for ODEs.

RK4 uses four function evaluations per step. It gains two orders over Euler (which uses one) and two orders over midpoint/Heun (which use two). The cost per order of accuracy improves as we go higher --- but only up to a point. Beyond order 4, the number of slope evaluations needed per step grows faster than the order, a result known as the **Butcher barrier**.

### The Butcher Tableau

Any explicit Runge-Kutta method can be specified compactly by a **Butcher tableau**:

$$\begin{array}{c|cccc}
c_1 & & & & \\
c_2 & a_{21} & & & \\
c_3 & a_{31} & a_{32} & & \\
\vdots & \vdots & & \ddots & \\
c_s & a_{s1} & a_{s2} & \cdots & a_{s,s-1} \\
\hline
& b_1 & b_2 & \cdots & b_s
\end{array}$$

The $c_i$ are the time points within the step (as fractions of $h$), the $a_{ij}$ are the coefficients used to combine previous slopes for each evaluation, and the $b_i$ are the final weights. For RK4:

$$\begin{array}{c|cccc}
0 & & & & \\
1/2 & 1/2 & & & \\
1/2 & 0 & 1/2 & & \\
1 & 0 & 0 & 1 & \\
\hline
& 1/6 & 1/3 & 1/3 & 1/6
\end{array}$$

---

## Adaptive Step Size Control

Fixed step sizes are wasteful. When the solution is nearly linear, large steps are fine. When it curves sharply, small steps are needed. **Adaptive step size** methods estimate the error at each step and adjust $h$ automatically.

The most common approach is **embedded Runge-Kutta methods**. These compute two approximations of different orders using the *same* set of slope evaluations, and use the difference as an error estimate.

The **Dormand-Prince method** (used in `scipy.integrate.solve_ivp` with method `'RK45'` and MATLAB's `ode45`) is the gold standard. It is a 7-stage method that produces both a 4th-order and a 5th-order approximation. The 5th-order result is used as the solution, and the difference between the 4th and 5th-order results estimates the local error.

If the estimated error $\varepsilon$ exceeds a tolerance $\text{tol}$, the step is rejected and retried with a smaller $h$. If the error is well below the tolerance, $h$ is increased. The step size update formula is:

$$h_{\text{new}} = h \cdot \min\left(5, \, \max\left(0.2, \, 0.9 \left(\frac{\text{tol}}{\varepsilon}\right)^{1/5}\right)\right)$$

The factor $0.9$ is a safety factor (do not use the full step size the formula suggests), and the $\min/\max$ prevent the step from changing too drastically in one iteration. The exponent $1/5$ comes from the 5th-order method: if the error scales as $h^5$, then to achieve a target error ratio, $h$ should be scaled by the fifth root of that ratio.

---

## Stability Regions

A method's **stability region** is the set of values $h\lambda$ (where $\lambda$ is the eigenvalue of the linearized problem $y' = \lambda y$) for which the method does not blow up. This is a property of the method itself, not the problem.

For the test equation $y' = \lambda y$ (with $\lambda$ complex), the exact solution $y = e^{\lambda t}$ decays when $\text{Re}(\lambda) < 0$. A numerical method should also produce a decaying solution in this case. Apply each method to $y' = \lambda y$ and find the **amplification factor** $R(z)$ where $z = h\lambda$:

**Euler:** $y_{n+1} = y_n + h\lambda y_n = (1 + z)y_n$, so $R(z) = 1 + z$. The method is stable when $|1 + z| \leq 1$, which is a disk of radius 1 centered at $z = -1$ in the complex plane.

**Implicit Euler:** $y_{n+1} = y_n + h\lambda y_{n+1}$, so $y_{n+1} = \frac{y_n}{1 - z}$ and $R(z) = \frac{1}{1 - z}$. Stable when $|1/(1-z)| \leq 1$, i.e., $|1-z| \geq 1$. This is the *exterior* of a disk centered at $z = 1$ --- it includes the entire left half-plane. This is why implicit Euler is so good for stiff problems.

**RK4:** The amplification factor is $R(z) = 1 + z + \frac{z^2}{2} + \frac{z^3}{6} + \frac{z^4}{24}$, which is the Taylor expansion of $e^z$ through order 4. The stability region is larger than Euler's, extending further along the imaginary axis (important for oscillatory problems) and further along the negative real axis.

The stability region determines the maximum step size for a given problem. If any eigenvalue $\lambda_i$ of the Jacobian $\partial f / \partial y$ lies outside the scaled stability region (after multiplying by $h$), the method will produce growing oscillations or outright diverge.

---

## Numerical Integration

Before moving to PDEs, let us briefly cover numerical integration (quadrature), which is closely related to ODE solving.

**Trapezoidal rule:** Approximate the area under $f(x)$ as a trapezoid:

$$\int_a^b f(x) \, dx \approx \frac{b-a}{2}\left[f(a) + f(b)\right]$$

Error: $O(h^2)$ where $h = b - a$. For a composite rule with $n$ subintervals of width $h = (b-a)/n$:

$$\int_a^b f(x) \, dx \approx \frac{h}{2}\left[f(x_0) + 2f(x_1) + 2f(x_2) + \cdots + 2f(x_{n-1}) + f(x_n)\right]$$

**Simpson's rule:** Approximate $f$ by a quadratic (parabola) through three points:

$$\int_a^b f(x) \, dx \approx \frac{b-a}{6}\left[f(a) + 4f\left(\frac{a+b}{2}\right) + f(b)\right]$$

Error: $O(h^4)$. The weights $(1, 4, 1)/6$ are the same pattern as the RK4 weights $(1, 2, 2, 1)/6$, which is not surprising --- both are matching polynomial approximations.

**Gaussian quadrature:** Instead of spacing the evaluation points equally, choose them optimally. It can be shown that for $n$ evaluation points, the optimal locations are the roots of the $n$-th Legendre polynomial, and the corresponding weights can be computed analytically. Gaussian quadrature with $n$ points integrates polynomials of degree $2n-1$ exactly --- it extracts twice as many orders of accuracy as you'd expect from $n$ points.

---

## Finite Differences for PDEs

A **partial differential equation (PDE)** involves derivatives with respect to multiple independent variables (e.g., space and time). The canonical example is the **heat equation**:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

where $u(x, t)$ is temperature, $\alpha$ is thermal diffusivity, and the equation says that the rate of temperature change is proportional to the spatial curvature of the temperature profile.

**Finite differences** approximate derivatives using nearby values on a grid. The key formulas come from Taylor expansion:

$$f(x + h) = f(x) + hf'(x) + \frac{h^2}{2}f''(x) + O(h^3)$$

$$f(x - h) = f(x) - hf'(x) + \frac{h^2}{2}f''(x) + O(h^3)$$

Adding these: $f(x+h) + f(x-h) = 2f(x) + h^2 f''(x) + O(h^4)$, giving:

$$f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$$

This is the **second-order central difference** approximation for the second derivative. Applying it to the heat equation on a grid $x_j = j \Delta x$, $t_n = n \Delta t$:

$$\frac{u_j^{n+1} - u_j^n}{\Delta t} = \alpha \frac{u_{j+1}^n - 2u_j^n + u_{j-1}^n}{(\Delta x)^2}$$

This is the **forward-time centered-space (FTCS)** scheme. Solving for $u_j^{n+1}$:

$$u_j^{n+1} = u_j^n + r\left(u_{j+1}^n - 2u_j^n + u_{j-1}^n\right)$$

where $r = \alpha \Delta t / (\Delta x)^2$. This is explicit: each new time step is computed directly from the previous one. But stability requires $r \leq 1/2$ (the **CFL condition** for this equation), which constrains the time step.

---

## Connections to Diffusion Models

The numerical methods in this post are not just abstract --- they are precisely the tools used inside diffusion model samplers.

### DDPM: Euler on the Reverse SDE

The denoising diffusion probabilistic model (DDPM) sampling process is:

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

where $z \sim \mathcal{N}(0, I)$. This is the **Euler-Maruyama** method applied to the reverse-time stochastic differential equation (SDE). Euler-Maruyama is the stochastic version of Euler's method --- the simplest possible discretization. It is first-order, which is why DDPM needs 1000 steps for good generation quality.

### DDIM: An Implicit Discretization

Denoising Diffusion Implicit Models (DDIM) use a different discretization that is deterministic (no noise term $z$). The DDIM update can be interpreted as an implicit discretization of the probability flow ODE associated with the reverse-time SDE. By removing stochasticity, DDIM can take larger steps without the noise accumulation that plagues DDPM. It achieves reasonable quality in 50-100 steps.

### DPM-Solver: Higher-Order Methods

DPM-Solver applies higher-order methods (analogous to RK2 and RK3) to the probability flow ODE. It uses the structure of the ODE --- specifically, that the noise prediction $\epsilon_\theta$ changes smoothly --- to extrapolate and take larger steps. DPM-Solver++ achieves 10-20 steps for high-quality generation.

The progression from DDPM to DDIM to DPM-Solver mirrors exactly the progression from Euler to implicit methods to higher-order Runge-Kutta. The same mathematical principles --- order of accuracy, stability, adaptive stepping --- explain why each successive sampler is faster.

---

## Python: From Lorenz Chaos to the Heat Equation

### Euler vs RK4 on the Lorenz System

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def lorenz(state, t, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """The Lorenz system: dx/dt = sigma*(y-x), dy/dt = x*(rho-z)-y, dz/dt = x*y - beta*z"""
    x, y, z = state
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])

def euler_solve(f, y0, t_span, h):
    """Euler method for systems of ODEs."""
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        y[i+1] = y[i] + h * f(y[i], t[i])
    return t, y

def rk4_solve(f, y0, t_span, h):
    """Classic RK4 method for systems of ODEs."""
    t0, tf = t_span
    t = np.arange(t0, tf + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        k1 = f(y[i], t[i])
        k2 = f(y[i] + 0.5*h*k1, t[i] + 0.5*h)
        k3 = f(y[i] + 0.5*h*k2, t[i] + 0.5*h)
        k4 = f(y[i] + h*k3, t[i] + h)
        y[i+1] = y[i] + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return t, y

# Solve with both methods
y0 = np.array([1.0, 1.0, 1.0])
t_span = (0, 25)

# RK4 with small step (reference solution)
t_ref, y_ref = rk4_solve(lorenz, y0, t_span, h=0.001)

# RK4 with moderate step
t_rk4, y_rk4 = rk4_solve(lorenz, y0, t_span, h=0.01)

# Euler with same step size
t_euler, y_euler = euler_solve(lorenz, y0, t_span, h=0.01)

# Euler with smaller step
t_euler2, y_euler2 = euler_solve(lorenz, y0, t_span, h=0.001)

# 3D trajectory comparison
fig = plt.figure(figsize=(16, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(y_ref[:, 0], y_ref[:, 1], y_ref[:, 2], color='#2196F3', linewidth=0.5, alpha=0.8)
ax1.set_title(r'Reference (RK4, $h=0.001$)', fontsize=11)
ax1.set_xlabel(r'$x$'); ax1.set_ylabel(r'$y$'); ax1.set_zlabel(r'$z$')

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(y_rk4[:, 0], y_rk4[:, 1], y_rk4[:, 2], color='#4CAF50', linewidth=0.5, alpha=0.8)
ax2.set_title(r'RK4, $h=0.01$', fontsize=11)
ax2.set_xlabel(r'$x$'); ax2.set_ylabel(r'$y$'); ax2.set_zlabel(r'$z$')

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(y_euler[:, 0], y_euler[:, 1], y_euler[:, 2], color='#F44336', linewidth=0.5, alpha=0.8)
ax3.set_title(r'Euler, $h=0.01$', fontsize=11)
ax3.set_xlabel(r'$x$'); ax3.set_ylabel(r'$y$'); ax3.set_zlabel(r'$z$')

plt.suptitle(r'Lorenz System: Method Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('lorenz_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Time series comparison (x-component)
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(t_ref, y_ref[:, 0], color='#2196F3', linewidth=1, label=r'Reference (RK4 $h=0.001$)')
axes[0].plot(t_rk4, y_rk4[:, 0], color='#4CAF50', linewidth=1, linestyle='--', label=r'RK4 $h=0.01$')
axes[0].plot(t_euler, y_euler[:, 0], color='#F44336', linewidth=1, linestyle=':', label=r'Euler $h=0.01$')
axes[0].set_xlabel(r'Time $t$')
axes[0].set_ylabel(r'$x(t)$')
axes[0].set_title(r'Lorenz $x$-component: Euler diverges from true solution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Error comparison
# Interpolate to common time grid for error computation
from scipy.interpolate import interp1d
t_common = t_rk4  # Use the coarser grid
y_ref_interp = interp1d(t_ref, y_ref[:, 0])(t_common)
error_rk4 = np.abs(y_rk4[:, 0] - y_ref_interp)
error_euler = np.abs(y_euler[:, 0] - y_ref_interp)

axes[1].semilogy(t_common, error_rk4 + 1e-16, color='#4CAF50', linewidth=1, label=r'RK4 $h=0.01$ error')
axes[1].semilogy(t_common, error_euler + 1e-16, color='#F44336', linewidth=1, label=r'Euler $h=0.01$ error')
axes[1].set_xlabel(r'Time $t$')
axes[1].set_ylabel(r'$|\mathrm{Error}|$ (log scale)')
axes[1].set_title(r'Error Growth: Euler error grows orders of magnitude faster')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lorenz_error.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Stability Region Plots

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_stability_regions():
    """Plot stability regions for Euler, Implicit Euler, RK2, and RK4."""
    # Create grid in complex plane
    x = np.linspace(-5, 3, 500)
    y = np.linspace(-4, 4, 500)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Euler: R(z) = 1 + z
    R_euler = np.abs(1 + Z)
    axes[0, 0].contourf(X, Y, R_euler, levels=[0, 1], colors=['#BBDEFB'], alpha=0.7)
    axes[0, 0].contour(X, Y, R_euler, levels=[1], colors=['#2196F3'], linewidths=2)
    axes[0, 0].set_title(r'Forward Euler (Order 1)', fontsize=13)

    # Implicit Euler: R(z) = 1/(1 - z), stable when |1-z| >= 1
    R_implicit = np.abs(1.0 / (1.0 - Z))
    axes[0, 1].contourf(X, Y, R_implicit, levels=[0, 1], colors=['#C8E6C9'], alpha=0.7)
    axes[0, 1].contour(X, Y, R_implicit, levels=[1], colors=['#4CAF50'], linewidths=2)
    axes[0, 1].set_title(r'Implicit Euler (Order 1, $A$-stable)', fontsize=13)

    # RK2 (Midpoint): R(z) = 1 + z + z^2/2
    R_rk2 = np.abs(1 + Z + Z**2/2)
    axes[1, 0].contourf(X, Y, R_rk2, levels=[0, 1], colors=['#FFE0B2'], alpha=0.7)
    axes[1, 0].contour(X, Y, R_rk2, levels=[1], colors=['#FF9800'], linewidths=2)
    axes[1, 0].set_title(r'RK2 / Midpoint (Order 2)', fontsize=13)

    # RK4: R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
    R_rk4 = np.abs(1 + Z + Z**2/2 + Z**3/6 + Z**4/24)
    axes[1, 1].contourf(X, Y, R_rk4, levels=[0, 1], colors=['#E1BEE7'], alpha=0.7)
    axes[1, 1].contour(X, Y, R_rk4, levels=[1], colors=['#9C27B0'], linewidths=2)
    axes[1, 1].set_title(r'RK4 (Order 4)', fontsize=13)

    for ax in axes.flat:
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.axvline(x=0, color='gray', linewidth=0.5)
        ax.set_xlabel(r'$\mathrm{Re}(h\lambda)$', fontsize=11)
        ax.set_ylabel(r'$\mathrm{Im}(h\lambda)$', fontsize=11)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.plot(0, 0, 'ko', markersize=3)

    plt.suptitle(r'Stability Regions (shaded = stable)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig('stability_regions.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_stability_regions()
```

### Heat Equation with Finite Differences

```python
import numpy as np
import matplotlib.pyplot as plt

def solve_heat_equation(nx=100, nt=5000, L=1.0, T=0.5, alpha=0.01):
    """
    Solve the 1D heat equation using FTCS finite differences.

    du/dt = alpha * d²u/dx²

    Initial condition: Gaussian pulse
    Boundary conditions: u(0,t) = u(L,t) = 0
    """
    dx = L / (nx - 1)
    dt = T / nt
    r = alpha * dt / dx**2

    print(f"Grid: {nx} points, {nt} time steps")
    print(f"dx = {dx:.4f}, dt = {dt:.6f}")
    print(f"r = alpha*dt/dx² = {r:.4f} (must be <= 0.5 for stability)")
    assert r <= 0.5, f"Unstable! r = {r} > 0.5"

    x = np.linspace(0, L, nx)

    # Initial condition: Gaussian pulse centered at L/2
    u = np.exp(-((x - L/2)**2) / (2 * 0.01))

    # Store snapshots for plotting
    snapshots = [u.copy()]
    snapshot_times = [0.0]
    snapshot_interval = nt // 10

    for n in range(1, nt + 1):
        u_new = u.copy()
        # FTCS: u_j^{n+1} = u_j^n + r*(u_{j+1}^n - 2*u_j^n + u_{j-1}^n)
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        # Boundary conditions
        u_new[0] = 0
        u_new[-1] = 0
        u = u_new

        if n % snapshot_interval == 0:
            snapshots.append(u.copy())
            snapshot_times.append(n * dt)

    return x, snapshots, snapshot_times

# Solve and plot
x, snapshots, times = solve_heat_equation()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: overlaid snapshots
colors = plt.cm.coolwarm(np.linspace(0, 1, len(snapshots)))
for i, (snap, t) in enumerate(zip(snapshots, times)):
    axes[0].plot(x, snap, color=colors[i], linewidth=1.5, label=rf'$t = {t:.3f}$')
axes[0].set_xlabel(r'$x$', fontsize=12)
axes[0].set_ylabel(r'$u(x, t)$', fontsize=12)
axes[0].set_title(r'Heat Equation: Temperature Evolution', fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Right: spacetime heatmap
# Re-solve to get full history
nx, nt = 100, 5000
L, T_final, alpha = 1.0, 0.5, 0.01
dx = L / (nx - 1)
dt = T_final / nt
r = alpha * dt / dx**2
x = np.linspace(0, L, nx)
u = np.exp(-((x - L/2)**2) / (2 * 0.01))

# Store every 50th frame
save_every = 50
history = [u.copy()]
for n in range(1, nt + 1):
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
    u_new[0] = 0; u_new[-1] = 0
    u = u_new
    if n % save_every == 0:
        history.append(u.copy())

history = np.array(history)
t_vals = np.arange(0, len(history)) * save_every * dt

im = axes[1].imshow(history.T, aspect='auto', origin='lower',
                     extent=[0, T_final, 0, L], cmap='hot')
axes[1].set_xlabel(r'Time $t$', fontsize=12)
axes[1].set_ylabel(r'Position $x$', fontsize=12)
axes[1].set_title(r'Heat Equation: Spacetime Diagram', fontsize=13)
plt.colorbar(im, ax=axes[1], label=r'Temperature $u(x,t)$')

plt.tight_layout()
plt.savefig('heat_equation.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Step Size Comparison

```python
import numpy as np
import matplotlib.pyplot as plt

def exact_exponential(t, y0=1.0, lam=-2.0):
    """Exact solution of dy/dt = lambda*y."""
    return y0 * np.exp(lam * t)

def euler_step(y, h, lam):
    return y + h * lam * y

def rk4_step(y, h, lam):
    k1 = lam * y
    k2 = lam * (y + 0.5*h*k1)
    k3 = lam * (y + 0.5*h*k2)
    k4 = lam * (y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# Compare step sizes
step_sizes = [0.5, 0.2, 0.1, 0.05, 0.01]
lam = -2.0
T = 3.0
y0 = 1.0

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Euler: different step sizes
for h in step_sizes:
    t = np.arange(0, T + h, h)
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(len(t)-1):
        y[i+1] = euler_step(y[i], h, lam)
    axes[0].plot(t, y, 'o-', markersize=3, label=rf'$h = {h}$')

t_exact = np.linspace(0, T, 200)
axes[0].plot(t_exact, exact_exponential(t_exact), 'k-', linewidth=2, label=r'Exact')
axes[0].set_title(r"Euler Method: Effect of Step Size $h$", fontsize=13)
axes[0].set_xlabel(r'$t$', fontsize=12)
axes[0].set_ylabel(r'$y(t)$', fontsize=12)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Convergence rate comparison
step_sizes_fine = np.logspace(-3, -0.3, 20)
euler_errors = []
rk4_errors = []

for h in step_sizes_fine:
    t = np.arange(0, T + h, h)
    # Euler
    y_e = y0
    for i in range(len(t)-1):
        y_e = euler_step(y_e, h, lam)
    euler_errors.append(abs(y_e - exact_exponential(T)))

    # RK4
    y_r = y0
    for i in range(len(t)-1):
        y_r = rk4_step(y_r, h, lam)
    rk4_errors.append(abs(y_r - exact_exponential(T)))

axes[1].loglog(step_sizes_fine, euler_errors, 'o-', color='#F44336', label=r'Euler (order 1)')
axes[1].loglog(step_sizes_fine, rk4_errors, 's-', color='#4CAF50', label=r'RK4 (order 4)')

# Reference slopes
h_ref = np.array([step_sizes_fine[0], step_sizes_fine[-1]])
axes[1].loglog(h_ref, 0.5*h_ref**1, 'r--', alpha=0.5, label=r'Slope 1')
axes[1].loglog(h_ref, 5*h_ref**4, 'g--', alpha=0.5, label=r'Slope 4')

axes[1].set_xlabel(r'Step size $h$', fontsize=12)
axes[1].set_ylabel(r'Global error at $t=3$', fontsize=12)
axes[1].set_title(r'Convergence: Error vs Step Size', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('step_size_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Conclusion

Every numerical ODE solver is built on the same foundation: discretize time, approximate the derivative, step forward. The differences between methods --- Euler, implicit Euler, midpoint, Heun, RK4, Dormand-Prince --- come down to three questions:

1. **How many slope evaluations per step?** More evaluations mean more information about the solution's curvature, enabling higher-order accuracy.

2. **Where are the slopes evaluated?** At the start of the interval, the midpoint, the end, or some combination. The choice determines the order and the stability properties.

3. **How large can the step be?** Stability regions determine when a method can take large steps without blowing up. Implicit methods have larger stability regions but require solving equations at each step.

These same principles directly explain the evolution of diffusion model samplers. DDPM uses 1000 steps because it is first-order Euler. DDIM uses 50-100 steps because it is implicit and deterministic. DPM-Solver uses 10-20 steps because it is higher-order. The mathematics of 18th-century numerical analysis is the engineering toolkit of 21st-century generative AI.
