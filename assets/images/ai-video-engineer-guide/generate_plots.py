"""
Generate all plots for: The AI Video Engineer's Complete Technical Reference
Uses numpy + matplotlib with publication-quality dark theme styling.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import os

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#555',
    'axes.labelcolor': '#d4d4d4',
    'text.color': '#d4d4d4',
    'xtick.color': '#999',
    'ytick.color': '#999',
    'grid.color': '#333',
    'grid.alpha': 0.4,
    'legend.facecolor': '#1a1a2e',
    'legend.edgecolor': '#444',
    'legend.labelcolor': '#d4d4d4',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'text.usetex': False,
    'savefig.dpi': 180,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#1a1a2e',
    'savefig.pad_inches': 0.3,
})

OUT = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: YCbCr Color Space Decomposition
# ═══════════════════════════════════════════════════════════════════════════════
def plot_ycbcr_decomposition():
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # Create a synthetic color gradient image
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    X, Y = np.meshgrid(x, y)

    R = np.clip(X, 0, 1)
    G = np.clip(1 - Y, 0, 1)
    B = np.clip(Y * X, 0, 1)
    rgb = np.stack([R, G, B], axis=-1)

    # ITU-R BT.601 conversion
    Yc = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.500 * B + 0.5
    Cr = 0.500 * R - 0.419 * G - 0.081 * B + 0.5

    titles = [r'RGB Original', r'Y (Luma)', r'$C_b$ (Blue Chroma)', r'$C_r$ (Red Chroma)']
    images = [rgb, Yc, Cb, Cr]
    cmaps = [None, 'gray', 'coolwarm', 'coolwarm']

    for ax, img, title, cmap in zip(axes, images, titles, cmaps):
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#555')

    fig.suptitle(r'YCbCr Color Space Decomposition (BT.601)', fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '01_ycbcr_decomposition.png'))
    plt.close()
    print("  [1/12] YCbCr decomposition")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: 2D DCT Basis Functions
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dct_basis():
    N = 8
    fig, axes = plt.subplots(N, N, figsize=(9, 9))

    for u in range(N):
        for v in range(N):
            basis = np.zeros((N, N))
            for x in range(N):
                for y in range(N):
                    basis[x, y] = (np.cos(np.pi * (2*x+1) * u / (2*N)) *
                                   np.cos(np.pi * (2*y+1) * v / (2*N)))
            axes[u, v].imshow(basis, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            axes[u, v].set_xticks([])
            axes[u, v].set_yticks([])
            for spine in axes[u, v].spines.values():
                spine.set_color('#333')
                spine.set_linewidth(0.5)

    fig.suptitle(r'8×8 DCT-II Basis Functions — $B_{uv}[x,y] = \cos\frac{\pi(2x+1)u}{16}\cos\frac{\pi(2y+1)v}{16}$',
                 fontsize=13, y=0.98, fontweight='bold')
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(OUT, '02_dct_basis_functions.png'))
    plt.close()
    print("  [2/12] DCT basis functions")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: DCT Energy Compaction
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dct_energy_compaction():
    np.random.seed(42)

    # Simulate a "natural image" block (smooth with some edges)
    x = np.linspace(0, np.pi, 64)
    natural = np.outer(np.sin(x) + 0.3*np.sin(3*x), np.cos(x) + 0.2*np.cos(2*x))
    natural += 0.05 * np.random.randn(64, 64)
    natural = (natural - natural.min()) / (natural.max() - natural.min())

    # Random noise image
    random_img = np.random.rand(64, 64)

    # Compute 2D DCT via matrix multiplication
    def dct2(block):
        N = block.shape[0]
        n = np.arange(N)
        C = np.cos(np.pi * np.outer(n, 2*n + 1) / (2*N))
        C[0] *= 1/np.sqrt(2)
        C *= np.sqrt(2/N)
        return C @ block @ C.T

    dct_natural = dct2(natural)
    dct_random = dct2(random_img)

    # Sort coefficients by magnitude (descending)
    sorted_natural = np.sort(np.abs(dct_natural.flatten()))[::-1]
    sorted_random = np.sort(np.abs(dct_random.flatten()))[::-1]

    # Cumulative energy
    cum_natural = np.cumsum(sorted_natural**2) / np.sum(sorted_natural**2)
    cum_random = np.cumsum(sorted_random**2) / np.sum(sorted_random**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: coefficient magnitude
    k = np.arange(1, len(sorted_natural)+1)
    ax1.semilogy(k, sorted_natural, color='#4fc3f7', linewidth=2, label='Natural image')
    ax1.semilogy(k, sorted_random, color='#ef5350', linewidth=2, alpha=0.7, label='Random noise')
    ax1.set_xlabel(r'Coefficient index $k$ (sorted by magnitude)')
    ax1.set_ylabel(r'$|c_k|$ (log scale)')
    ax1.set_title(r'DCT Coefficient Magnitudes', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 500)

    # Right: cumulative energy
    ax2.plot(k / len(k) * 100, cum_natural * 100, color='#4fc3f7', linewidth=2.5, label='Natural image')
    ax2.plot(k / len(k) * 100, cum_random * 100, color='#ef5350', linewidth=2.5, alpha=0.7, label='Random noise')
    ax2.axhline(y=95, color='#66bb6a', linestyle='--', alpha=0.6, linewidth=1)
    ax2.text(55, 96.5, '95% energy threshold', color='#66bb6a', fontsize=10)

    # Find 95% point for natural
    idx_95 = np.searchsorted(cum_natural, 0.95)
    pct_95 = idx_95 / len(k) * 100
    ax2.axvline(x=pct_95, color='#4fc3f7', linestyle=':', alpha=0.5)
    ax2.annotate(f'{pct_95:.1f}% of coefficients', xy=(pct_95, 95), xytext=(pct_95+10, 80),
                 arrowprops=dict(arrowstyle='->', color='#4fc3f7'), color='#4fc3f7', fontsize=10)

    ax2.set_xlabel(r'Percentage of coefficients retained')
    ax2.set_ylabel(r'Cumulative energy (%)')
    ax2.set_title(r'Energy Compaction — Why Natural Video Compresses Well', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '03_dct_energy_compaction.png'))
    plt.close()
    print("  [3/12] DCT energy compaction")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: Optical Flow Field Visualization
# ═══════════════════════════════════════════════════════════════════════════════
def plot_optical_flow():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Synthetic rotational + translational flow
    y, x = np.mgrid[-3:3:20j, -3:3:20j]

    # Rotation + translation
    u_rot = -y * 0.5 + 0.3
    v_rot = x * 0.5 + 0.2

    magnitude = np.sqrt(u_rot**2 + v_rot**2)

    ax1.quiver(x, y, u_rot, v_rot, magnitude, cmap='plasma', scale=15, width=0.005)
    ax1.set_title(r'Optical Flow Field: Rotation + Translation', fontweight='bold')
    ax1.set_xlabel(r'$x$')
    ax1.set_ylabel(r'$y$')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # Diverging flow (zoom)
    u_div = x * 0.4
    v_div = y * 0.4
    mag_div = np.sqrt(u_div**2 + v_div**2)

    ax2.quiver(x, y, u_div, v_div, mag_div, cmap='plasma', scale=12, width=0.005)
    ax2.set_title(r'Optical Flow Field: Radial Expansion (Zoom)', fontweight='bold')
    ax2.set_xlabel(r'$x$')
    ax2.set_ylabel(r'$y$')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)

    fig.suptitle(r'Optical Flow $\mathbf{v}(x,y) = (u, v)$ — Dense Motion Fields', fontsize=14, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '04_optical_flow_fields.png'))
    plt.close()
    print("  [4/12] Optical flow fields")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5: Quality Metrics Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def plot_quality_metrics():
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # PSNR as function of MSE
    mse = np.linspace(0.001, 0.1, 500)
    psnr = 10 * np.log10(1.0 / mse)

    ax = axes[0, 0]
    ax.plot(mse, psnr, color='#4fc3f7', linewidth=2.5)
    ax.axhline(y=30, color='#66bb6a', linestyle='--', alpha=0.5, label='Acceptable threshold (30 dB)')
    ax.axhline(y=40, color='#FF9800', linestyle='--', alpha=0.5, label='High quality (40 dB)')
    ax.set_xlabel(r'MSE')
    ax.set_ylabel(r'PSNR (dB)')
    ax.set_title(r'PSNR $= 10\log_{10}\frac{1}{\mathrm{MSE}}$', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # SSIM components
    ax = axes[0, 1]
    mu_x = 0.5
    sigma_x = 0.2
    C1 = 0.01**2
    C2 = 0.03**2

    mu_y = np.linspace(0, 1, 200)
    sigma_y = 0.2
    sigma_xy = 0.15

    luminance = (2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
    contrast = (2 * sigma_x * sigma_y + C2) / (sigma_x**2 + sigma_y**2 + C2)
    ssim_vals = luminance * contrast

    ax.plot(mu_y, luminance, color='#4fc3f7', linewidth=2, label=r'Luminance $l(x,y)$')
    ax.plot(mu_y, np.full_like(mu_y, contrast), color='#66bb6a', linewidth=2, linestyle='--', label=r'Contrast $c(x,y)$')
    ax.plot(mu_y, ssim_vals, color='#FF9800', linewidth=2.5, label=r'SSIM $= l \cdot c \cdot s$')
    ax.axvline(x=0.5, color='#999', linestyle=':', alpha=0.4)
    ax.set_xlabel(r'Mean of reference patch $\mu_y$')
    ax.set_ylabel(r'Component value')
    ax.set_title(r'SSIM Components vs Luminance Shift', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # FID vs number of samples
    ax = axes[1, 0]
    n_samples = np.array([100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000])
    # FID decreases and stabilizes with more samples
    true_fid = 15.0
    fid_estimate = true_fid + 200 / np.sqrt(n_samples) + np.random.randn(len(n_samples)) * 2
    fid_std = 150 / np.sqrt(n_samples)

    ax.errorbar(n_samples, fid_estimate, yerr=fid_std, color='#CE93D8', linewidth=2,
                capsize=4, marker='o', markersize=5, label=r'FID estimate $\pm 1\sigma$')
    ax.axhline(y=true_fid, color='#66bb6a', linestyle='--', alpha=0.6, label=f'True FID = {true_fid}')
    ax.set_xscale('log')
    ax.set_xlabel(r'Number of samples $N$')
    ax.set_ylabel(r'FID')
    ax.set_title(r'FID Convergence — Bias $\propto 1/\sqrt{N}$', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Perception-Distortion tradeoff
    ax = axes[1, 1]
    d = np.linspace(0.01, 2.0, 200)
    # Blau & Michaeli bound: p >= f(d) convex decreasing
    p_bound = 1.5 / (1 + d**1.5)

    ax.plot(d, p_bound, color='#4fc3f7', linewidth=3, label='Pareto frontier')
    ax.fill_between(d, p_bound, 1.5, alpha=0.08, color='#4fc3f7')
    ax.fill_between(d, 0, p_bound, alpha=0.05, color='#ef5350')

    # Operating points
    points = {
        'MSE-optimal': (0.3, 1.2),
        'GAN': (1.0, 0.15),
        'Perceptual loss': (0.6, 0.5),
        'Diffusion': (0.5, 0.25),
    }
    colors = ['#ef5350', '#66bb6a', '#FF9800', '#CE93D8']
    for (name, (dx, py)), c in zip(points.items(), colors):
        ax.scatter(dx, py, s=80, color=c, zorder=5, edgecolors='white', linewidth=0.5)
        ax.annotate(name, (dx, py), textcoords="offset points", xytext=(8, 8), fontsize=9, color=c)

    ax.set_xlabel(r'Distortion $d(\hat{x}, x)$')
    ax.set_ylabel(r'Perceptual quality $p(\hat{x})$')
    ax.set_title(r'Perception–Distortion Tradeoff', fontweight='bold')
    ax.text(1.3, 1.0, 'Impossible\nregion', color='#ef5350', fontsize=10, alpha=0.6, ha='center')
    ax.text(0.8, 0.8, 'Achievable', color='#4fc3f7', fontsize=10, alpha=0.6, ha='center')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '05_quality_metrics.png'))
    plt.close()
    print("  [5/12] Quality metrics comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6: Diffusion Forward and Reverse Process
# ═══════════════════════════════════════════════════════════════════════════════
def plot_diffusion_process():
    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Top-left: Noise schedule comparison
    ax = axes[0, 0]
    T = 1000
    t = np.arange(T)

    # Linear schedule
    beta_linear = np.linspace(1e-4, 0.02, T)
    alpha_linear = 1 - beta_linear
    alpha_bar_linear = np.cumprod(alpha_linear)

    # Cosine schedule
    s = 0.008
    f_t = np.cos((t/T + s) / (1 + s) * np.pi/2)**2
    alpha_bar_cosine = f_t / f_t[0]

    # Squared cosine (EDM-style)
    sigma = np.exp(np.linspace(np.log(0.002), np.log(80), T))
    snr_edm = 1 / sigma**2
    alpha_bar_edm = snr_edm / (1 + snr_edm)

    ax.plot(t, alpha_bar_linear, color='#ef5350', linewidth=2, label=r'Linear: $\bar{\alpha}_t$')
    ax.plot(t, alpha_bar_cosine, color='#4fc3f7', linewidth=2, label=r'Cosine: $\bar{\alpha}_t$')
    ax.plot(t, alpha_bar_edm, color='#66bb6a', linewidth=2, label=r'EDM (log-normal $\sigma$)')
    ax.set_xlabel(r'Timestep $t$')
    ax.set_ylabel(r'$\bar{\alpha}_t = \prod_{s=1}^{t}(1-\beta_s)$')
    ax.set_title(r'Noise Schedules — Signal Retention $\bar{\alpha}_t$', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: SNR over time
    ax = axes[0, 1]
    snr_linear = alpha_bar_linear / (1 - alpha_bar_linear + 1e-10)
    snr_cosine = alpha_bar_cosine / (1 - alpha_bar_cosine + 1e-10)

    ax.semilogy(t, snr_linear, color='#ef5350', linewidth=2, label='Linear')
    ax.semilogy(t, snr_cosine, color='#4fc3f7', linewidth=2, label='Cosine')
    ax.semilogy(t, snr_edm, color='#66bb6a', linewidth=2, label='EDM')
    ax.set_xlabel(r'Timestep $t$')
    ax.set_ylabel(r'SNR $= \bar{\alpha}_t / (1 - \bar{\alpha}_t)$')
    ax.set_title(r'Signal-to-Noise Ratio Over Time', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: 1D diffusion trajectory
    ax = axes[1, 0]
    x0 = 2.0  # Start point
    n_steps = 50
    timesteps = np.linspace(0, 1, n_steps)
    n_trajectories = 8

    for i in range(n_trajectories):
        noise = np.random.randn(n_steps)
        trajectory = x0 * np.sqrt(1 - timesteps) + np.cumsum(noise) * 0.15 * np.sqrt(timesteps)
        alpha = 0.3 if i > 0 else 1.0
        lw = 1.0 if i > 0 else 2.0
        ax.plot(timesteps, trajectory, color='#4fc3f7', alpha=alpha, linewidth=lw)

    ax.axhline(y=x0, color='#66bb6a', linestyle='--', alpha=0.4, linewidth=1)
    ax.set_xlabel(r'Diffusion time $t$ ($0 \to 1$)')
    ax.set_ylabel(r'$x_t$')
    ax.set_title(r'Forward Diffusion Trajectories: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Bottom-right: Score function vector field
    ax = axes[1, 1]
    # 2D score field for a mixture of two Gaussians
    xx = np.linspace(-4, 4, 20)
    yy = np.linspace(-4, 4, 20)
    X, Y = np.meshgrid(xx, yy)

    # Mixture: N([-1.5, 0], I) + N([1.5, 0], I)
    mu1 = np.array([-1.5, 0])
    mu2 = np.array([1.5, 0])

    def gaussian_score(x, y, mu, sigma=1.0):
        dx, dy = x - mu[0], y - mu[1]
        p = np.exp(-0.5 * (dx**2 + dy**2) / sigma**2)
        sx = -dx / sigma**2 * p
        sy = -dy / sigma**2 * p
        return sx, sy, p

    sx1, sy1, p1 = gaussian_score(X, Y, mu1)
    sx2, sy2, p2 = gaussian_score(X, Y, mu2)

    # Score of mixture
    total_p = p1 + p2 + 1e-10
    Sx = (sx1 + sx2) / total_p * (p1 + p2)
    Sy = (sy1 + sy2) / total_p * (p1 + p2)

    # Normalize for visibility
    mag = np.sqrt(Sx**2 + Sy**2) + 1e-10
    Sx_norm = Sx / mag
    Sy_norm = Sy / mag

    ax.quiver(X, Y, Sx_norm, Sy_norm, mag, cmap='coolwarm', scale=30, width=0.004, alpha=0.9)
    ax.scatter(*mu1, s=120, color='#4fc3f7', zorder=5, edgecolors='white', linewidth=1.5, marker='*')
    ax.scatter(*mu2, s=120, color='#66bb6a', zorder=5, edgecolors='white', linewidth=1.5, marker='*')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_title(r'Score Field $\nabla_x \log p(x)$ — Gaussian Mixture', fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '06_diffusion_process.png'))
    plt.close()
    print("  [6/12] Diffusion process")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Rate-Distortion Theory
# ═══════════════════════════════════════════════════════════════════════════════
def plot_rate_distortion():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Gaussian R(D)
    sigma2 = 1.0
    D = np.linspace(0.01, sigma2, 500)
    R_D = 0.5 * np.log2(sigma2 / D)

    ax1.plot(D, R_D, color='#4fc3f7', linewidth=3, label=r'$R(D) = \frac{1}{2}\log_2\frac{\sigma^2}{D}$')
    ax1.fill_between(D, R_D, 5, alpha=0.06, color='#66bb6a')
    ax1.fill_between(D, 0, R_D, alpha=0.04, color='#ef5350')
    ax1.text(0.15, 0.8, 'Achievable', color='#66bb6a', fontsize=11, alpha=0.7)
    ax1.text(0.6, 2.0, 'Impossible', color='#ef5350', fontsize=11, alpha=0.7)

    # Mark practical operating points
    ax1.scatter([0.05], [0.5 * np.log2(1/0.05)], s=80, color='#FF9800', zorder=5, edgecolors='white')
    ax1.annotate('High quality\n(low D)', (0.05, 0.5*np.log2(1/0.05)),
                 xytext=(0.15, 3.8), arrowprops=dict(arrowstyle='->', color='#FF9800'), color='#FF9800', fontsize=9)

    ax1.scatter([0.5], [0.5 * np.log2(1/0.5)], s=80, color='#CE93D8', zorder=5, edgecolors='white')
    ax1.annotate('Low quality\n(high D)', (0.5, 0.5*np.log2(1/0.5)),
                 xytext=(0.65, 1.8), arrowprops=dict(arrowstyle='->', color='#CE93D8'), color='#CE93D8', fontsize=9)

    ax1.set_xlabel(r'Distortion $D$ (MSE)')
    ax1.set_ylabel(r'Rate $R$ (bits/symbol)')
    ax1.set_title(r'Rate-Distortion Function — Gaussian Source', fontweight='bold')
    ax1.set_ylim(0, 5)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Compression ratio vs quality for video
    ax2_data = {
        'Raw (uncompressed)': (1, 48),
        'ProRes 422': (6, 45),
        'H.264 (high bitrate)': (50, 42),
        'H.264 (medium)': (200, 36),
        'H.265 (medium)': (200, 39),
        'AV1 (medium)': (200, 40),
        'H.265 (low bitrate)': (1000, 30),
        '3D VAE (8× spatial)': (256, 38),
        '3D VAE (16× spatial)': (1024, 33),
    }

    names = list(ax2_data.keys())
    ratios = [v[0] for v in ax2_data.values()]
    quality = [v[1] for v in ax2_data.values()]
    colors_pts = ['#d4d4d4', '#d4d4d4', '#4fc3f7', '#4fc3f7', '#66bb6a', '#CE93D8', '#4fc3f7', '#FF9800', '#FF9800']

    ax2.scatter(ratios, quality, s=80, c=colors_pts, zorder=5, edgecolors='white', linewidth=0.5)
    for name, r, q in zip(names, ratios, quality):
        offset = (10, 5) if r < 500 else (-10, -15)
        ax2.annotate(name, (r, q), textcoords="offset points", xytext=offset, fontsize=8, color='#d4d4d4')

    ax2.set_xscale('log')
    ax2.set_xlabel(r'Compression Ratio')
    ax2.set_ylabel(r'PSNR (dB)')
    ax2.set_title(r'Codec Quality vs Compression Ratio', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '07_rate_distortion.png'))
    plt.close()
    print("  [7/12] Rate-distortion theory")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 8: Temporal Coherence Analysis
# ═══════════════════════════════════════════════════════════════════════════════
def plot_temporal_coherence():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    T = 100  # frames

    # Frame-to-frame SSIM for different models
    ax = axes[0]
    t = np.arange(T - 1)

    # Good temporal model
    ssim_good = 0.95 + 0.02 * np.random.randn(T - 1)
    ssim_good = np.clip(ssim_good, 0.88, 1.0)

    # Mediocre model (some flicker)
    ssim_med = 0.90 + 0.05 * np.random.randn(T - 1)
    ssim_med[30:35] -= 0.15  # scene transition artifact
    ssim_med = np.clip(ssim_med, 0.7, 1.0)

    # Bad model (heavy flicker)
    ssim_bad = 0.82 + 0.10 * np.random.randn(T - 1)
    ssim_bad = np.clip(ssim_bad, 0.5, 1.0)

    ax.plot(t, ssim_good, color='#66bb6a', linewidth=1.5, label='Temporally coherent', alpha=0.9)
    ax.plot(t, ssim_med, color='#FF9800', linewidth=1.5, label='Moderate flicker', alpha=0.9)
    ax.plot(t, ssim_bad, color='#ef5350', linewidth=1.5, label='Severe flicker', alpha=0.9)
    ax.set_xlabel(r'Frame pair $(t, t+1)$')
    ax.set_ylabel(r'SSIM$(f_t, f_{t+1})$')
    ax.set_title(r'Frame-to-Frame Consistency', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.05)

    # Temporal frequency spectrum
    ax = axes[1]
    fps = 24
    freq = np.fft.rfftfreq(T, d=1/fps)[1:]  # Hz

    # Pixel intensity over time for different models
    pixel_good = np.sin(2 * np.pi * 2 * np.arange(T) / fps) + 0.02 * np.random.randn(T)
    pixel_bad = np.sin(2 * np.pi * 2 * np.arange(T) / fps) + 0.3 * np.random.randn(T)
    # Add a flicker artifact at 12Hz
    pixel_bad += 0.2 * np.sin(2 * np.pi * 12 * np.arange(T) / fps)

    fft_good = np.abs(np.fft.rfft(pixel_good))[1:]
    fft_bad = np.abs(np.fft.rfft(pixel_bad))[1:]

    ax.plot(freq, fft_good / fft_good.max(), color='#66bb6a', linewidth=2, label='Clean signal')
    ax.plot(freq, fft_bad / fft_bad.max(), color='#ef5350', linewidth=2, alpha=0.8, label='Flickering signal')
    ax.axvspan(8, 12, alpha=0.1, color='#ef5350')
    ax.text(9.5, 0.85, 'Flicker\nband', color='#ef5350', fontsize=9, ha='center')
    ax.set_xlabel(r'Temporal frequency (Hz)')
    ax.set_ylabel(r'Normalized magnitude')
    ax.set_title(r'Temporal Frequency Spectrum', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Warping error
    ax = axes[2]
    n_frames = 48
    frame_idx = np.arange(n_frames)

    # Warping error (flow-warped frame vs actual frame)
    warp_err_diffusion = 0.03 + 0.008 * np.random.randn(n_frames)
    warp_err_diffusion = np.clip(warp_err_diffusion, 0.01, 0.08)

    warp_err_autoregressive = 0.02 + np.arange(n_frames) * 0.001 + 0.005 * np.random.randn(n_frames)
    warp_err_autoregressive = np.clip(warp_err_autoregressive, 0.01, 0.1)

    ax.plot(frame_idx, warp_err_diffusion, color='#4fc3f7', linewidth=2, label='Parallel diffusion')
    ax.plot(frame_idx, warp_err_autoregressive, color='#FF9800', linewidth=2, label='Autoregressive')
    ax.set_xlabel(r'Frame index')
    ax.set_ylabel(r'Warping error (MSE)')
    ax.set_title(r'Temporal Error Accumulation', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '08_temporal_coherence.png'))
    plt.close()
    print("  [8/12] Temporal coherence")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 9: VAE Latent Space Compression
# ═══════════════════════════════════════════════════════════════════════════════
def plot_latent_space():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # KL divergence vs reconstruction loss (ELBO tradeoff)
    ax = axes[0]
    beta = np.linspace(0.001, 5, 200)
    recon_loss = 10 * np.exp(-0.8 * beta) + 2
    kl_loss = 3 * (1 - np.exp(-beta))
    elbo = recon_loss + beta[:, None].flatten() * kl_loss if False else recon_loss + kl_loss

    ax.plot(beta, recon_loss, color='#4fc3f7', linewidth=2.5, label=r'Reconstruction $\mathbb{E}[\log p(x|z)]$')
    ax.plot(beta, kl_loss, color='#ef5350', linewidth=2.5, label=r'KL $D_{\mathrm{KL}}(q(z|x)\|p(z))$')
    ax.plot(beta, recon_loss + kl_loss, color='#FF9800', linewidth=2.5, linestyle='--', label=r'Total ELBO')

    # Find optimum
    total = recon_loss + kl_loss
    opt_idx = np.argmin(total)
    ax.scatter(beta[opt_idx], total[opt_idx], s=100, color='#66bb6a', zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(r'Optimal $\beta$', (beta[opt_idx], total[opt_idx]),
                xytext=(beta[opt_idx]+0.8, total[opt_idx]+1), arrowprops=dict(arrowstyle='->', color='#66bb6a'),
                color='#66bb6a', fontsize=10)

    ax.set_xlabel(r'$\beta$ (KL weight)')
    ax.set_ylabel(r'Loss')
    ax.set_title(r'$\beta$-VAE Tradeoff: Reconstruction vs Regularization', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Latent dimension vs reconstruction quality
    ax = axes[1]
    dims = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    # Reconstruction improves then plateaus
    recon_quality = 40 - 30 * np.exp(-dims / 40)
    recon_quality += np.random.randn(len(dims)) * 0.3

    ax.plot(dims, recon_quality, 'o-', color='#4fc3f7', linewidth=2, markersize=7)
    ax.axhline(y=38, color='#66bb6a', linestyle='--', alpha=0.5, label='Diminishing returns')
    ax.set_xlabel(r'Latent channels $C_z$')
    ax.set_ylabel(r'Reconstruction PSNR (dB)')
    ax.set_title(r'Latent Dimensionality vs Quality', fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Compression architecture comparison
    ax = axes[2]
    models = ['SD 1.5\nVAE', 'SDXL\nVAE', 'SVD\n3D VAE', 'CogVideo\n3D VAE', 'Sora\n(est.)']
    spatial_comp = [8, 8, 8, 8, 16]
    temporal_comp = [1, 1, 4, 4, 8]
    channels = [4, 4, 4, 16, 16]

    x_pos = np.arange(len(models))
    width = 0.25

    bars1 = ax.bar(x_pos - width, spatial_comp, width, color='#4fc3f7', alpha=0.8, label=r'Spatial $\downarrow$')
    bars2 = ax.bar(x_pos, temporal_comp, width, color='#66bb6a', alpha=0.8, label=r'Temporal $\downarrow$')
    bars3 = ax.bar(x_pos + width, channels, width, color='#FF9800', alpha=0.8, label=r'Latent $C_z$')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel(r'Factor')
    ax.set_title(r'3D VAE Architectures: Compression Factors', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '09_latent_space.png'))
    plt.close()
    print("  [9/12] Latent space compression")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 10: Inference Optimization Landscape
# ═══════════════════════════════════════════════════════════════════════════════
def plot_inference_optimization():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Steps vs quality for different samplers
    ax = axes[0]
    steps = np.arange(1, 51)

    # DDPM (needs many steps)
    quality_ddpm = 35 * (1 - np.exp(-steps / 15))
    # DDIM (fewer steps)
    quality_ddim = 35 * (1 - np.exp(-steps / 8))
    # DPM-Solver++ (even fewer)
    quality_dpm = 35 * (1 - np.exp(-steps / 4))
    # Consistency distillation (1-4 steps)
    quality_consist = np.minimum(35 * (1 - np.exp(-steps / 1.5)), 33)

    ax.plot(steps, quality_ddpm, color='#ef5350', linewidth=2, label='DDPM')
    ax.plot(steps, quality_ddim, color='#FF9800', linewidth=2, label='DDIM')
    ax.plot(steps, quality_dpm, color='#4fc3f7', linewidth=2, label='DPM-Solver++')
    ax.plot(steps, quality_consist, color='#66bb6a', linewidth=2, label='Consistency')
    ax.axhline(y=33, color='#999', linestyle=':', alpha=0.4)
    ax.set_xlabel(r'Denoising steps')
    ax.set_ylabel(r'Quality (FID $\downarrow$, inverted)')
    ax.set_title(r'Sampler Efficiency: Steps vs Quality', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Quantization: bits vs quality
    ax = axes[1]
    bits = np.array([2, 3, 4, 6, 8, 16, 32])
    # Quality degrades below 8-bit
    quality_quant = np.array([18, 28, 34.5, 36.8, 37.0, 37.1, 37.1])
    speedup = 32 / bits

    ax_twin = ax.twinx()
    line1 = ax.plot(bits, quality_quant, 'o-', color='#4fc3f7', linewidth=2.5, markersize=7, label='PSNR')
    line2 = ax_twin.plot(bits, speedup, 's--', color='#FF9800', linewidth=2, markersize=6, label='Speedup')

    ax.set_xlabel(r'Weight precision (bits)')
    ax.set_ylabel(r'PSNR (dB)', color='#4fc3f7')
    ax_twin.set_ylabel(r'Memory reduction ($\times$)', color='#FF9800')
    ax_twin.tick_params(axis='y', colors='#FF9800')
    ax.set_title(r'Quantization: Precision vs Quality', fontweight='bold')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=8, color='#66bb6a', linestyle=':', alpha=0.4)
    ax.text(8.3, 20, 'Sweet\nspot', color='#66bb6a', fontsize=9)

    # Latency breakdown
    ax = axes[2]
    components = ['VAE\nEncode', 'Denoise\n(×30)', 'Attn\nCompute', 'VAE\nDecode', 'Post-\nProcess']
    latency_base = np.array([120, 4500, 2800, 180, 50])
    latency_opt = np.array([80, 900, 600, 120, 40])

    x_pos = np.arange(len(components))
    width = 0.35

    ax.bar(x_pos - width/2, latency_base, width, color='#ef5350', alpha=0.8, label='Baseline (FP32)')
    ax.bar(x_pos + width/2, latency_opt, width, color='#66bb6a', alpha=0.8, label='Optimized (INT8 + flash attn)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(components, fontsize=9)
    ax.set_ylabel(r'Latency (ms)')
    ax.set_title(r'Inference Latency Breakdown', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Add speedup annotations
    for i, (b, o) in enumerate(zip(latency_base, latency_opt)):
        speedup = b / o
        ax.text(i, max(b, o) + 150, f'{speedup:.1f}×', ha='center', fontsize=9, color='#FF9800', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '10_inference_optimization.png'))
    plt.close()
    print("  [10/12] Inference optimization")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 11: Super-Resolution Methods Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def plot_super_resolution():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Upsampling kernels comparison
    ax = axes[0]
    x = np.linspace(-3, 3, 1000)

    # Nearest neighbor
    nearest = np.where(np.abs(x) < 0.5, 1.0, 0.0)
    # Bilinear
    bilinear = np.maximum(1 - np.abs(x), 0)
    # Bicubic (Keys kernel, a=-0.5)
    a = -0.5
    bicubic = np.where(np.abs(x) <= 1,
                       (a+2)*np.abs(x)**3 - (a+3)*np.abs(x)**2 + 1,
                       np.where(np.abs(x) <= 2,
                                a*np.abs(x)**3 - 5*a*np.abs(x)**2 + 8*a*np.abs(x) - 4*a,
                                0))
    # Lanczos-3
    lanczos = np.where(np.abs(x) < 3,
                       np.sinc(x) * np.sinc(x/3),
                       0)

    ax.plot(x, nearest, color='#ef5350', linewidth=2, label='Nearest', alpha=0.7)
    ax.plot(x, bilinear, color='#FF9800', linewidth=2, label='Bilinear')
    ax.plot(x, bicubic, color='#4fc3f7', linewidth=2.5, label='Bicubic (Keys)')
    ax.plot(x, lanczos, color='#66bb6a', linewidth=2, label='Lanczos-3')
    ax.axhline(y=0, color='#555', linewidth=0.5)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$h(x)$')
    ax.set_title(r'Interpolation Kernels', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Sub-pixel convolution (pixel shuffle) diagram as data
    ax = axes[1]
    # Show frequency response of kernels
    freq = np.fft.rfftfreq(1000, d=6/1000)
    H_nearest = np.abs(np.fft.rfft(nearest))
    H_bilinear = np.abs(np.fft.rfft(bilinear))
    H_bicubic = np.abs(np.fft.rfft(bicubic))
    H_lanczos = np.abs(np.fft.rfft(lanczos))

    # Normalize
    H_nearest /= H_nearest[0] + 1e-10
    H_bilinear /= H_bilinear[0] + 1e-10
    H_bicubic /= H_bicubic[0] + 1e-10
    H_lanczos /= H_lanczos[0] + 1e-10

    n_show = len(freq) // 3
    ax.plot(freq[:n_show], H_nearest[:n_show], color='#ef5350', linewidth=2, label='Nearest', alpha=0.7)
    ax.plot(freq[:n_show], H_bilinear[:n_show], color='#FF9800', linewidth=2, label='Bilinear')
    ax.plot(freq[:n_show], H_bicubic[:n_show], color='#4fc3f7', linewidth=2.5, label='Bicubic')
    ax.plot(freq[:n_show], H_lanczos[:n_show], color='#66bb6a', linewidth=2, label='Lanczos-3')
    ax.axhline(y=0, color='#555', linewidth=0.5)

    # Mark Nyquist
    nyquist = 1/(2*(6/1000))  # approximate
    ax.axvline(x=freq[n_show//2], color='#CE93D8', linestyle='--', alpha=0.5, label='Nyquist freq')

    ax.set_xlabel(r'Frequency')
    ax.set_ylabel(r'$|H(f)|$ (normalized)')
    ax.set_title(r'Frequency Response of Interpolation Kernels', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Scale factor vs PSNR for different methods
    ax = axes[2]
    scales = np.array([2, 3, 4, 6, 8])

    psnr_bicubic = np.array([33.7, 30.4, 28.4, 25.8, 24.0])
    psnr_srcnn = np.array([36.7, 32.8, 30.5, 27.6, 25.4])
    psnr_esrgan = np.array([37.6, 33.8, 31.4, 28.5, 26.2])
    psnr_diffusion = np.array([37.2, 33.5, 31.8, 29.2, 27.0])

    ax.plot(scales, psnr_bicubic, 'o-', color='#ef5350', linewidth=2, label='Bicubic')
    ax.plot(scales, psnr_srcnn, 's-', color='#FF9800', linewidth=2, label='SRCNN')
    ax.plot(scales, psnr_esrgan, '^-', color='#4fc3f7', linewidth=2, label='Real-ESRGAN')
    ax.plot(scales, psnr_diffusion, 'D-', color='#66bb6a', linewidth=2, label='Diffusion SR')

    ax.set_xlabel(r'Upscaling factor')
    ax.set_ylabel(r'PSNR (dB)')
    ax.set_title(r'Super-Resolution: Scale vs Quality', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(scales)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '11_super_resolution.png'))
    plt.close()
    print("  [11/12] Super-resolution")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 12: Production Pipeline Cost & Throughput
# ═══════════════════════════════════════════════════════════════════════════════
def plot_production_pipeline():
    np.random.seed(42)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # GPU utilization vs batch size
    ax = axes[0]
    batch_sizes = np.array([1, 2, 4, 8, 16, 32, 64])
    throughput_a100 = np.array([1.0, 1.9, 3.5, 6.2, 10.5, 14.8, 16.2])
    throughput_h100 = np.array([1.8, 3.4, 6.5, 11.5, 19.2, 28.0, 31.5])
    gpu_util = np.minimum(throughput_a100 / 16.2 * 100, 100)

    ax_twin = ax.twinx()
    line1 = ax.plot(batch_sizes, throughput_a100, 'o-', color='#4fc3f7', linewidth=2, label='A100 80GB')
    line2 = ax.plot(batch_sizes, throughput_h100, 's-', color='#66bb6a', linewidth=2, label='H100 80GB')
    line3 = ax_twin.plot(batch_sizes, gpu_util, '^--', color='#FF9800', linewidth=1.5, alpha=0.7, label='GPU util (A100)')

    ax.set_xlabel(r'Batch size')
    ax.set_ylabel(r'Throughput (videos/min)', color='#4fc3f7')
    ax_twin.set_ylabel(r'GPU utilization (%)', color='#FF9800')
    ax_twin.tick_params(axis='y', colors='#FF9800')
    ax_twin.set_ylim(0, 110)
    ax.set_title(r'Batch Size vs Throughput', fontweight='bold')
    ax.set_xscale('log', base=2)

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc='center left')
    ax.grid(True, alpha=0.3)

    # Cost per video at different resolutions
    ax = axes[1]
    resolutions = ['480p\n4s', '720p\n4s', '1080p\n4s', '480p\n10s', '720p\n10s', '1080p\n10s']
    cost_a100 = np.array([0.08, 0.25, 0.85, 0.20, 0.62, 2.12])
    cost_h100 = np.array([0.05, 0.15, 0.52, 0.12, 0.38, 1.30])

    x_pos = np.arange(len(resolutions))
    width = 0.35

    ax.bar(x_pos - width/2, cost_a100, width, color='#4fc3f7', alpha=0.8, label='A100 ($3.50/hr)')
    ax.bar(x_pos + width/2, cost_h100, width, color='#66bb6a', alpha=0.8, label='H100 ($5.50/hr)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(resolutions, fontsize=9)
    ax.set_ylabel(r'Cost per video (\$)')
    ax.set_title(r'Generation Cost by Resolution & Duration', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Scaling: concurrent users vs p99 latency
    ax = axes[2]
    users = np.array([1, 5, 10, 20, 50, 100, 200, 500])

    # Without queue (direct GPU)
    latency_direct = 5.0 + 5.0 * np.maximum(users / 8 - 1, 0)  # 8 GPU slots
    latency_direct = np.minimum(latency_direct, 300)

    # With async queue
    latency_queue = 5.0 + 0.5 * np.log2(np.maximum(users / 8, 1))

    # With autoscaling
    latency_autoscale = 5.0 + 2.0 * np.log2(np.maximum(users / 32, 1))
    latency_autoscale = np.maximum(latency_autoscale, 5.0)

    ax.plot(users, latency_direct, 'o-', color='#ef5350', linewidth=2, label='Direct (no queue)')
    ax.plot(users, latency_queue, 's-', color='#FF9800', linewidth=2, label='Async queue (fixed GPUs)')
    ax.plot(users, latency_autoscale, '^-', color='#66bb6a', linewidth=2, label='Queue + autoscale')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Concurrent users')
    ax.set_ylabel(r'p99 latency (seconds)')
    ax.set_title(r'Scaling Strategy: Latency vs Load', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, '12_production_pipeline.png'))
    plt.close()
    print("  [12/12] Production pipeline")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating plots for AI Video Engineer's Technical Reference...")
    print("=" * 60)
    plot_ycbcr_decomposition()
    plot_dct_basis()
    plot_dct_energy_compaction()
    plot_optical_flow()
    plot_quality_metrics()
    plot_diffusion_process()
    plot_rate_distortion()
    plot_temporal_coherence()
    plot_latent_space()
    plot_inference_optimization()
    plot_super_resolution()
    plot_production_pipeline()
    print("=" * 60)
    print(f"All 12 plots saved to: {OUT}")
