---
layout: post
title: "Audio-Driven Video Generation: When Music and Speech Control the Visuals"
date: 2025-12-31
category: models
---

Text-to-video has dominated the generative video conversation for the past two years. You type a prompt, you get a video. But text is a lossy compression of creative intent. Try describing the exact moment a bass drop should hit, the precise rhythm of camera cuts in a music video, or the way a voiceover's cadence should dictate scene pacing. Text falls apart.

Audio-driven video generation flips the paradigm. Instead of text as the primary conditioning signal, audio --- music, speech, sound effects --- provides the temporal structure, emotional arc, and rhythmic backbone that guides visual generation. This post is the deep technical dive: how audio features map to visual properties, the architectures that make it work, the math of beat alignment, and where this fits in a production video pipeline.

---

## Table of Contents

1. [Why Audio as a Conditioning Signal](#why-audio-as-a-conditioning-signal)
2. [Audio Feature Extraction](#audio-feature-extraction)
3. [The Math of Beat Detection and Tempo Estimation](#the-math-of-beat-detection-and-tempo-estimation)
4. [Architectural Approaches](#architectural-approaches)
5. [Beat-Synchronous Generation](#beat-synchronous-generation)
6. [Music Visualization with Diffusion Models](#music-visualization-with-diffusion-models)
7. [Multi-Modal Conditioning: Text + Audio + Image](#multi-modal-conditioning-text--audio--image)
8. [Use Cases for Video Platforms](#use-cases-for-video-platforms)
9. [Evaluation: Measuring Audio-Visual Sync](#evaluation-measuring-audio-visual-sync)
10. [Latency and Production Considerations](#latency-and-production-considerations)
11. [The Road Ahead](#the-road-ahead)

---

## Why Audio as a Conditioning Signal

Text conditioning works through a single mechanism: a text encoder (CLIP, T5, etc.) produces an embedding that cross-attends with the denoising network. This embedding captures semantic content --- what should appear in the video --- but encodes almost no temporal information. The word "explosion" tells the model what to show, but not when.

Audio conditioning is fundamentally different. An audio signal is a time-series with rich temporal structure at multiple scales:

- **Microsecond scale**: Waveform oscillations, frequency content
- **Millisecond scale**: Phonemes, transients, note onsets
- **Second scale**: Beats, phrases, melody contours
- **Multi-second scale**: Song sections (verse, chorus, bridge), speech paragraphs
- **Minute scale**: Overall energy arc, narrative structure

Each of these temporal scales can map to a visual property:

| Audio Feature | Temporal Scale | Visual Mapping |
|:--|:--|:--|
| Beat onsets | 100--500 ms | Camera cuts, scene transitions |
| Energy envelope | 500 ms -- 2 s | Motion intensity, zoom speed |
| Spectral centroid | Frame-level | Brightness, color temperature |
| Chromagram | 1--4 s | Color palette, mood lighting |
| Mel spectrogram | Frame-level | Texture detail, visual complexity |
| Onset strength | 50--200 ms | Particle bursts, flash effects |
| Speech phonemes | 30--100 ms | Lip sync, facial expression |
| Tempo (BPM) | Global | Editing pace, animation speed |

This is not arbitrary aesthetic mapping. There is psychophysical evidence that humans perceive audio-visual correspondence along these dimensions. Cross-modal associations between pitch and brightness, loudness and size, and tempo and movement speed are well-documented in perception research.

<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg" style="max-width:800px; margin: 2em auto; display: block;">
  <rect width="800" height="500" fill="white"/>

  <text x="400" y="30" font-family="Georgia, serif" font-size="18" fill="#333" text-anchor="middle" font-weight="bold">Audio Feature Extraction Pipeline</text>

  <!-- Audio waveform input -->
  <rect x="30" y="60" width="140" height="60" rx="8" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="2"/>
  <text x="100" y="85" font-family="Georgia, serif" font-size="13" fill="#0288d1" text-anchor="middle" font-weight="bold">Raw Audio</text>
  <text x="100" y="105" font-family="Georgia, serif" font-size="11" fill="#0288d1" text-anchor="middle">Waveform (44.1kHz)</text>

  <!-- Arrow -->
  <line x1="170" y1="90" x2="210" y2="90" stroke="#333" stroke-width="1.5" marker-end="url(#arrowBlk1)"/>

  <!-- STFT block -->
  <rect x="210" y="60" width="120" height="60" rx="8" fill="#ffa726" opacity="0.15" stroke="#ffa726" stroke-width="2"/>
  <text x="270" y="85" font-family="Georgia, serif" font-size="13" fill="#e65100" text-anchor="middle" font-weight="bold">STFT</text>
  <text x="270" y="105" font-family="Georgia, serif" font-size="11" fill="#e65100" text-anchor="middle">Time-Frequency</text>

  <!-- Branching arrows -->
  <line x1="330" y1="75" x2="410" y2="55" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <line x1="330" y1="85" x2="410" y2="135" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <line x1="330" y1="95" x2="410" y2="215" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <line x1="330" y1="105" x2="410" y2="295" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <line x1="330" y1="110" x2="410" y2="375" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>

  <!-- Feature branches -->
  <!-- Mel Spectrogram -->
  <rect x="410" y="35" width="160" height="45" rx="6" fill="#ef5350" opacity="0.12" stroke="#ef5350" stroke-width="1.5"/>
  <text x="490" y="55" font-family="Georgia, serif" font-size="12" fill="#c62828" text-anchor="middle" font-weight="bold">Mel Spectrogram</text>
  <text x="490" y="72" font-family="Georgia, serif" font-size="10" fill="#c62828" text-anchor="middle">128 mel bands x T frames</text>

  <!-- Beat / Onset -->
  <rect x="410" y="115" width="160" height="45" rx="6" fill="#8bc34a" opacity="0.12" stroke="#8bc34a" stroke-width="1.5"/>
  <text x="490" y="135" font-family="Georgia, serif" font-size="12" fill="#558b2f" text-anchor="middle" font-weight="bold">Beat / Onset Detection</text>
  <text x="490" y="152" font-family="Georgia, serif" font-size="10" fill="#558b2f" text-anchor="middle">Beat timestamps, BPM</text>

  <!-- Energy Envelope -->
  <rect x="410" y="195" width="160" height="45" rx="6" fill="#4fc3f7" opacity="0.12" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="490" y="215" font-family="Georgia, serif" font-size="12" fill="#0288d1" text-anchor="middle" font-weight="bold">Energy Envelope</text>
  <text x="490" y="232" font-family="Georgia, serif" font-size="10" fill="#0288d1" text-anchor="middle">RMS per frame</text>

  <!-- Chromagram -->
  <rect x="410" y="275" width="160" height="45" rx="6" fill="#ffa726" opacity="0.12" stroke="#ffa726" stroke-width="1.5"/>
  <text x="490" y="295" font-family="Georgia, serif" font-size="12" fill="#e65100" text-anchor="middle" font-weight="bold">Chromagram</text>
  <text x="490" y="312" font-family="Georgia, serif" font-size="10" fill="#e65100" text-anchor="middle">12 pitch classes x T</text>

  <!-- Spectral Features -->
  <rect x="410" y="355" width="160" height="45" rx="6" fill="#ef5350" opacity="0.12" stroke="#ef5350" stroke-width="1.5"/>
  <text x="490" y="375" font-family="Georgia, serif" font-size="12" fill="#c62828" text-anchor="middle" font-weight="bold">Spectral Features</text>
  <text x="490" y="392" font-family="Georgia, serif" font-size="10" fill="#c62828" text-anchor="middle">Centroid, rolloff, flux</text>

  <!-- Right side: Visual mappings -->
  <line x1="570" y1="57" x2="630" y2="57" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <text x="640" y="52" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Texture, detail level</text>
  <text x="640" y="66" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Visual complexity</text>

  <line x1="570" y1="137" x2="630" y2="137" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <text x="640" y="132" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Camera cuts, transitions</text>
  <text x="640" y="146" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Scene boundaries</text>

  <line x1="570" y1="217" x2="630" y2="217" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <text x="640" y="212" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Motion intensity</text>
  <text x="640" y="226" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Zoom speed, camera energy</text>

  <line x1="570" y1="297" x2="630" y2="297" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <text x="640" y="292" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Color palette, mood</text>
  <text x="640" y="306" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Warm/cool tones</text>

  <line x1="570" y1="377" x2="630" y2="377" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk1)"/>
  <text x="640" y="372" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">Brightness, saturation</text>
  <text x="640" y="386" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="start">High-freq detail emphasis</text>

  <!-- Title labels -->
  <text x="490" y="460" font-family="Georgia, serif" font-size="12" fill="#999" text-anchor="middle" font-style="italic">Audio Domain</text>
  <text x="700" y="460" font-family="Georgia, serif" font-size="12" fill="#999" text-anchor="middle" font-style="italic">Visual Domain</text>

  <defs>
    <marker id="arrowBlk1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
    </marker>
  </defs>
</svg>

---

## Audio Feature Extraction

Before any audio can condition a video model, it must be transformed from raw waveform into structured representations. Here are the core features, their mathematical definitions, and their computational extraction.

### Mel Spectrogram

The mel spectrogram is the workhorse representation for audio in deep learning. It converts the linear frequency scale of the Short-Time Fourier Transform (STFT) into a perceptually-motivated mel scale.

**Step 1: STFT.** Given a discrete audio signal $x[n]$ sampled at rate $f_s$, the STFT at frame $m$ and frequency bin $k$ is:

$$X[m, k] = \sum_{n=0}^{N-1} x[n + mH] \cdot w[n] \cdot e^{-j2\pi kn/N}$$

where $N$ is the FFT window size (typically 2048), $H$ is the hop size (typically 512), and $w[n]$ is a window function (Hann window).

**Step 2: Power spectrum.**

$$S[m, k] = |X[m, k]|^2$$

**Step 3: Mel filterbank.** A set of $M$ triangular filters $\{h_i[k]\}_{i=1}^{M}$ (typically $M = 128$) spaced according to the mel scale:

$$\text{mel}(f) = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

**Step 4: Apply filterbank and take log.**

$$\text{MelSpec}[m, i] = \log\left(\sum_{k} h_i[k] \cdot S[m, k] + \epsilon\right)$$

The result is a 2D representation of shape $(T, M)$ where $T$ is the number of time frames and $M$ is the number of mel bands. This is the standard input to audio encoders in cross-modal models.

### Beat Detection

Beat detection identifies the temporal positions of rhythmic pulses in music. The standard approach uses the onset strength envelope.

**Onset strength envelope.** Compute the spectral flux --- the positive half-wave rectified first-order difference of the mel spectrogram across time:

$$O[m] = \sum_{i=1}^{M} \max\left(0, \text{MelSpec}[m, i] - \text{MelSpec}[m-1, i]\right)$$

This captures sudden increases in spectral energy --- exactly what characterizes a musical beat onset.

**Peak picking.** Beats correspond to peaks in $O[m]$. A peak at frame $m$ is selected if:

$$O[m] > O[m-\delta] \quad \text{and} \quad O[m] > O[m+\delta] \quad \text{and} \quad O[m] > \mu + \lambda\sigma$$

where $\delta$ is a minimum peak distance, $\mu$ and $\sigma$ are the local mean and standard deviation of $O$, and $\lambda$ is a threshold factor. The timestamps of selected peaks form the beat grid.

### Energy Envelope (RMS)

The root-mean-square energy per frame measures overall loudness:

$$E[m] = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n + mH]^2}$$

This is the simplest audio feature but one of the most useful for video conditioning. High energy means more visual motion; low energy means calm, slow visuals.

### Chromagram

The chromagram maps spectral energy to 12 pitch classes (C, C#, D, ..., B), collapsing all octaves. Each chroma bin $c \in \{0, ..., 11\}$ at time frame $m$:

$$\text{Chroma}[m, c] = \sum_{k \in \text{bins}(c)} S[m, k]$$

where $\text{bins}(c)$ contains all frequency bins whose fundamental frequency corresponds to pitch class $c$ across all audible octaves.

The chromagram captures harmonic content --- whether the music is in a major or minor key at any moment, which chord is playing, how the harmonic progression evolves. This maps naturally to color mood: major keys to warm, bright palettes; minor keys to cool, desaturated tones.

### Practical Extraction with librosa

```python
import librosa
import numpy as np

def extract_audio_features(audio_path, sr=22050, hop_length=512):
    """Extract all relevant audio features for video conditioning."""
    y, sr = librosa.load(audio_path, sr=sr)

    # Mel spectrogram (128 bands)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)

    # Beat detection
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Onset strength envelope
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length
    )

    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Chromagram
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=hop_length
    )

    # Spectral centroid (brightness)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    return {
        'mel_spectrogram': log_mel,         # (128, T)
        'tempo_bpm': tempo,                  # scalar
        'beat_times': beat_times,            # (num_beats,)
        'onset_envelope': onset_env,         # (T,)
        'rms_energy': rms,                   # (T,)
        'chromagram': chroma,                # (12, T)
        'spectral_centroid': spectral_centroid,  # (T,)
        'duration': len(y) / sr,             # seconds
        'sr': sr,
        'hop_length': hop_length,
    }
```

---

## The Math of Beat Detection and Tempo Estimation

Beat detection is the foundation of audio-visual synchronization. Getting it right is the difference between a music video that feels locked to the rhythm and one that feels randomly assembled. Let us go deeper into the signal processing.

### Tempo Estimation via Autocorrelation

Given the onset strength envelope $O[m]$, the tempo can be estimated by finding the dominant periodicity through autocorrelation:

$$R[\tau] = \sum_{m} O[m] \cdot O[m + \tau]$$

The autocorrelation $R[\tau]$ peaks at lags $\tau$ corresponding to periodic repetitions in the onset envelope. The dominant period $\tau^*$ is:

$$\tau^* = \arg\max_{\tau \in [\tau_\min, \tau_\max]} R[\tau]$$

where $\tau_\min$ and $\tau_\max$ correspond to the plausible BPM range (typically 60--200 BPM). Converting lag to BPM:

$$\text{BPM} = \frac{60 \cdot f_s}{H \cdot \tau^*}$$

where $f_s$ is the sample rate and $H$ is the hop length.

An alternative formulation uses the median inter-beat interval:

$$\text{BPM} = \frac{60}{\text{median}(\Delta t_{\text{beats}})}$$

where $\Delta t_{\text{beats}} = \{t_{i+1} - t_i\}$ is the set of inter-beat time differences. This is more robust to occasional missed or extra beats.

### Dynamic Programming Beat Tracking

The state-of-the-art approach (Ellis 2007, used in librosa) formulates beat tracking as a dynamic programming problem. Define a cost function over candidate beat sequences $\{b_1, b_2, \ldots, b_K\}$:

$$C(\{b_k\}) = \sum_{k=1}^{K} O[b_k] + \alpha \sum_{k=2}^{K} F(b_k - b_{k-1}, \tau^*)$$

The first term rewards placing beats at frames with high onset strength. The second term penalizes deviations from the estimated tempo period $\tau^*$ via a penalty function:

$$F(\Delta, \tau^*) = -\left(\log \frac{\Delta}{\tau^*}\right)^2$$

This is a Gaussian log-penalty centered on the expected inter-beat interval. The parameter $\alpha$ controls the tradeoff between rhythmic regularity and onset fidelity. The optimal beat sequence is found by dynamic programming in $O(T \cdot W)$ time, where $W$ is the search window around the expected beat period.

### Phase and Downbeat Estimation

Beats have phase (where in the measure the first detected beat falls) and hierarchical structure (downbeats mark the start of each measure, typically every 4 beats in 4/4 time). For video generation, downbeats are particularly important --- they mark natural positions for major visual transitions (scene changes, perspective shifts).

Downbeat detection typically requires a trained classifier that operates on spectral features around each detected beat, identifying which beats carry the acoustic emphasis characteristic of measure boundaries.

---

## Architectural Approaches

There are three main architectural strategies for injecting audio conditioning into video generation models. Each has distinct strengths and limitations.

### Approach 1: Audio Embedding + Cross-Attention

This mirrors how text conditioning works in standard diffusion models. An audio encoder produces a sequence of embeddings, which then cross-attend with the visual denoising network.

**Audio encoders:**

- **CLAP (Contrastive Language-Audio Pretraining)**: Trained to align audio and text in a shared embedding space, analogous to CLIP for images. Produces a single global embedding or a sequence of frame-level embeddings from audio.
- **AudioMAE (Audio Masked Autoencoder)**: A self-supervised model trained on mel spectrograms via masked reconstruction. Produces rich, high-dimensional frame-level features.
- **BEATs**: Audio pre-training through iterative audio tokenization and acoustic model training. Strong on both music and speech.
- **Whisper encoder**: Originally designed for speech recognition, but the encoder features capture detailed temporal speech structure useful for speech-driven video.

The cross-attention mechanism is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q$ comes from the visual (noisy latent) features and $K, V$ come from the audio encoder output. This allows each spatial-temporal position in the video to attend to the most relevant parts of the audio.

**Strengths**: Leverages the well-understood cross-attention mechanism. Compatible with existing diffusion architectures. The audio encoder can be pretrained independently.

**Weaknesses**: Global audio embeddings lose temporal precision. Frame-level embeddings require careful temporal alignment between audio frames and video frames. The cross-attention mechanism does not enforce strict temporal synchronization.

<svg viewBox="0 0 800 450" xmlns="http://www.w3.org/2000/svg" style="max-width:800px; margin: 2em auto; display: block;">
  <rect width="800" height="450" fill="white"/>

  <text x="400" y="30" font-family="Georgia, serif" font-size="18" fill="#333" text-anchor="middle" font-weight="bold">Audio Embedding + Cross-Attention Architecture</text>

  <!-- Audio input -->
  <rect x="30" y="170" width="110" height="50" rx="6" fill="#ffa726" opacity="0.15" stroke="#ffa726" stroke-width="2"/>
  <text x="85" y="192" font-family="Georgia, serif" font-size="12" fill="#e65100" text-anchor="middle" font-weight="bold">Audio Input</text>
  <text x="85" y="208" font-family="Georgia, serif" font-size="10" fill="#e65100" text-anchor="middle">Mel Spectrogram</text>

  <!-- Audio encoder -->
  <rect x="175" y="160" width="120" height="70" rx="6" fill="#ffa726" opacity="0.15" stroke="#ffa726" stroke-width="2"/>
  <text x="235" y="185" font-family="Georgia, serif" font-size="12" fill="#e65100" text-anchor="middle" font-weight="bold">Audio Encoder</text>
  <text x="235" y="200" font-family="Georgia, serif" font-size="10" fill="#e65100" text-anchor="middle">(CLAP / AudioMAE</text>
  <text x="235" y="215" font-family="Georgia, serif" font-size="10" fill="#e65100" text-anchor="middle">/ BEATs)</text>

  <line x1="140" y1="195" x2="175" y2="195" stroke="#333" stroke-width="1.5" marker-end="url(#arrowBlk2)"/>

  <!-- Audio embeddings -->
  <rect x="330" y="170" width="100" height="50" rx="6" fill="#ffa726" opacity="0.25" stroke="#ffa726" stroke-width="1.5"/>
  <text x="380" y="192" font-family="Georgia, serif" font-size="11" fill="#e65100" text-anchor="middle" font-weight="bold">K, V</text>
  <text x="380" y="207" font-family="Georgia, serif" font-size="10" fill="#e65100" text-anchor="middle">Audio Embeddings</text>

  <line x1="295" y1="195" x2="330" y2="195" stroke="#333" stroke-width="1.5" marker-end="url(#arrowBlk2)"/>

  <!-- Noisy latent -->
  <rect x="30" y="330" width="110" height="50" rx="6" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="2"/>
  <text x="85" y="352" font-family="Georgia, serif" font-size="12" fill="#0288d1" text-anchor="middle" font-weight="bold">Noisy Latent</text>
  <text x="85" y="368" font-family="Georgia, serif" font-size="10" fill="#0288d1" text-anchor="middle">z_t (video frames)</text>

  <!-- Self-attention -->
  <rect x="200" y="320" width="130" height="70" rx="6" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="2"/>
  <text x="265" y="347" font-family="Georgia, serif" font-size="12" fill="#0288d1" text-anchor="middle" font-weight="bold">Self-Attention</text>
  <text x="265" y="363" font-family="Georgia, serif" font-size="10" fill="#0288d1" text-anchor="middle">Spatial + Temporal</text>
  <text x="265" y="378" font-family="Georgia, serif" font-size="10" fill="#0288d1" text-anchor="middle">Q from visual features</text>

  <line x1="140" y1="355" x2="200" y2="355" stroke="#333" stroke-width="1.5" marker-end="url(#arrowBlk2)"/>

  <!-- Cross-attention block -->
  <rect x="440" y="260" width="140" height="80" rx="8" fill="#8bc34a" opacity="0.15" stroke="#8bc34a" stroke-width="2"/>
  <text x="510" y="290" font-family="Georgia, serif" font-size="13" fill="#558b2f" text-anchor="middle" font-weight="bold">Cross-Attention</text>
  <text x="510" y="308" font-family="Georgia, serif" font-size="11" fill="#558b2f" text-anchor="middle">Q = visual features</text>
  <text x="510" y="323" font-family="Georgia, serif" font-size="11" fill="#558b2f" text-anchor="middle">K, V = audio emb.</text>

  <!-- Arrows into cross-attention -->
  <line x1="430" y1="195" x2="460" y2="260" stroke="#ffa726" stroke-width="1.5" marker-end="url(#arrowBlk2)"/>
  <line x1="330" y1="355" x2="440" y2="310" stroke="#4fc3f7" stroke-width="1.5" marker-end="url(#arrowBlk2)"/>

  <!-- Output -->
  <rect x="640" y="275" width="120" height="50" rx="6" fill="#ef5350" opacity="0.15" stroke="#ef5350" stroke-width="2"/>
  <text x="700" y="297" font-family="Georgia, serif" font-size="12" fill="#c62828" text-anchor="middle" font-weight="bold">Denoised</text>
  <text x="700" y="313" font-family="Georgia, serif" font-size="10" fill="#c62828" text-anchor="middle">Audio-conditioned z</text>

  <line x1="580" y1="300" x2="640" y2="300" stroke="#333" stroke-width="1.5" marker-end="url(#arrowBlk2)"/>

  <!-- Text conditioning (optional) -->
  <rect x="440" y="80" width="140" height="50" rx="6" fill="#ef5350" opacity="0.12" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3"/>
  <text x="510" y="100" font-family="Georgia, serif" font-size="11" fill="#c62828" text-anchor="middle">Text Embeddings</text>
  <text x="510" y="115" font-family="Georgia, serif" font-size="10" fill="#c62828" text-anchor="middle">(optional, CLIP/T5)</text>

  <line x1="510" y1="130" x2="510" y2="260" stroke="#ef5350" stroke-width="1.2" stroke-dasharray="4,3" marker-end="url(#arrowBlk2)"/>
  <text x="525" y="200" font-family="Georgia, serif" font-size="10" fill="#999" text-anchor="start">Additional K, V</text>

  <defs>
    <marker id="arrowBlk2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
    </marker>
  </defs>
</svg>

### Approach 2: Beat-Synchronous Keyframe Generation

Rather than conditioning the diffusion process directly on audio, this approach uses audio analysis to determine the temporal structure of the video, then generates keyframes at beat-aligned timestamps.

The pipeline:

1. **Beat analysis**: Extract beats, downbeats, sections from the audio.
2. **Temporal plan**: Assign scene types, camera movements, and transitions to beat intervals.
3. **Keyframe generation**: Generate one image per beat (or per downbeat) using image diffusion, conditioned on per-beat text prompts derived from the overall video description.
4. **Interpolation**: Fill frames between keyframes using video interpolation models (FILM, AMT, or latent interpolation within a video diffusion model).

This approach has several advantages: the beat-visual alignment is exact by construction (keyframes are placed at beat times), the image generation quality is higher than current video models (image models are more mature), and the pipeline is modular.

The main disadvantage is that motion between keyframes is synthesized after the fact, which can produce unnatural movement. The interpolation model has no knowledge of the audio, so subtle rhythm-motion correspondences within a beat interval are lost.

### Approach 3: Feature-Modulated Generation

This approach modulates the diffusion model's internal activations directly using audio features, without cross-attention. Audio features are projected to match the channel dimensions of intermediate activations and applied via adaptive normalization:

$$\hat{h} = \gamma(a) \cdot \frac{h - \mu_h}{\sigma_h} + \beta(a)$$

where $h$ is an intermediate feature map, $a$ is the audio feature vector at the corresponding timestamp, and $\gamma(a), \beta(a)$ are learned affine transformations of the audio. This is analogous to AdaIN (Adaptive Instance Normalization) or FiLM (Feature-wise Linear Modulation).

This gives the audio very direct control over the visual features --- not through attention (which is soft and distributed) but through explicit modulation of activations. The audio literally scales and shifts the visual features at each layer.

---

## Beat-Synchronous Generation

The core technical challenge of audio-driven video is temporal alignment: ensuring that visual events (cuts, transitions, motion peaks) align with audio events (beats, onsets, phrase boundaries). Let us formalize this.

### Beat Grid Construction

Given detected beat times $\{b_1, b_2, \ldots, b_K\}$ and a target video frame rate $f_\text{fps}$, construct a beat grid that maps each beat to a video frame:

$$f_k = \text{round}(b_k \cdot f_\text{fps})$$

where $f_k$ is the frame index of the $k$-th beat. The inter-beat interval in frames is:

$$\Delta f_k = f_{k+1} - f_k$$

For 120 BPM music at 24 fps: $\Delta f = \frac{60}{120} \times 24 = 12$ frames per beat. For 140 BPM: $\Delta f \approx 10.3$ frames --- note the non-integer value, which requires careful handling to avoid drift.

### Quantization to Beat Grid

When generating video in segments (e.g., 4-second chunks for current models), segment boundaries should align with beats. The optimal segmentation is:

$$\text{segments} = \{(b_{k_i}, b_{k_{i+1}})\}$$

where $\{k_i\}$ are chosen so that each segment duration is close to the model's optimal generation length while respecting beat boundaries. This is a bin-packing problem:

$$\min_{\{k_i\}} \sum_i \left| (b_{k_{i+1}} - b_{k_i}) - T_\text{target} \right|$$

subject to $k_0 = 1$, $k_M = K$, and $b_{k_{i+1}} - b_{k_i} \leq T_\max$.

### Visual Event Assignment

Each beat interval needs a visual "plan." This is where creative intelligence enters. A rule-based approach assigns visual events based on audio features:

```python
def plan_visual_events(audio_features, beat_times):
    """Assign visual events to beat intervals based on audio analysis."""
    events = []
    onset_env = audio_features['onset_envelope']
    rms = audio_features['rms_energy']
    sr = audio_features['sr']
    hop = audio_features['hop_length']

    for i in range(len(beat_times) - 1):
        t_start = beat_times[i]
        t_end = beat_times[i + 1]

        # Convert to frame indices
        f_start = int(t_start * sr / hop)
        f_end = int(t_end * sr / hop)

        # Compute local audio features for this beat interval
        local_energy = np.mean(rms[f_start:f_end])
        local_onset = np.max(onset_env[f_start:f_end])
        energy_change = rms[min(f_end, len(rms)-1)] - rms[f_start]

        event = {
            'start': t_start,
            'end': t_end,
            'duration': t_end - t_start,
        }

        # High onset strength -> hard cut or flash transition
        if local_onset > np.percentile(onset_env, 90):
            event['transition'] = 'hard_cut'
        elif local_onset > np.percentile(onset_env, 70):
            event['transition'] = 'quick_dissolve'
        else:
            event['transition'] = 'smooth_blend'

        # Energy level -> motion intensity
        energy_norm = local_energy / (np.max(rms) + 1e-8)
        if energy_norm > 0.7:
            event['motion'] = 'high'
            event['camera'] = 'dynamic'  # fast pan, zoom, shake
        elif energy_norm > 0.3:
            event['motion'] = 'medium'
            event['camera'] = 'slow_move'  # gentle drift
        else:
            event['motion'] = 'low'
            event['camera'] = 'static'  # locked off

        # Energy rising -> zoom in, falling -> zoom out
        if energy_change > 0.1 * np.max(rms):
            event['zoom'] = 'in'
        elif energy_change < -0.1 * np.max(rms):
            event['zoom'] = 'out'
        else:
            event['zoom'] = 'none'

        events.append(event)

    return events
```

An LLM can also be used to produce these plans. Given the audio features as structured data plus a text description of the desired video, an LLM like Gemini Flash can output a per-beat shot list with camera directions, scene descriptions, and transition types.

<svg viewBox="0 0 800 320" xmlns="http://www.w3.org/2000/svg" style="max-width:800px; margin: 2em auto; display: block;">
  <rect width="800" height="320" fill="white"/>

  <text x="400" y="25" font-family="Georgia, serif" font-size="16" fill="#333" text-anchor="middle" font-weight="bold">Beat-Aligned Video Generation</text>

  <!-- Audio waveform representation -->
  <text x="40" y="60" font-family="Georgia, serif" font-size="12" fill="#555" font-weight="bold">Audio:</text>
  <rect x="90" y="45" width="680" height="40" rx="4" fill="#ffa726" opacity="0.08" stroke="#ffa726" stroke-width="1"/>

  <!-- Beat markers -->
  <line x1="130" y1="45" x2="130" y2="85" stroke="#ef5350" stroke-width="2"/>
  <text x="130" y="100" font-family="monospace" font-size="9" fill="#ef5350" text-anchor="middle">b1</text>
  <line x1="230" y1="45" x2="230" y2="85" stroke="#ef5350" stroke-width="2"/>
  <text x="230" y="100" font-family="monospace" font-size="9" fill="#ef5350" text-anchor="middle">b2</text>
  <line x1="330" y1="45" x2="330" y2="85" stroke="#ef5350" stroke-width="2"/>
  <text x="330" y="100" font-family="monospace" font-size="9" fill="#ef5350" text-anchor="middle">b3</text>
  <line x1="430" y1="45" x2="430" y2="85" stroke="#ef5350" stroke-width="2"/>
  <text x="430" y="100" font-family="monospace" font-size="9" fill="#ef5350" text-anchor="middle">b4</text>
  <line x1="530" y1="45" x2="530" y2="85" stroke="#ef5350" stroke-width="3"/>
  <text x="530" y="100" font-family="monospace" font-size="9" fill="#ef5350" text-anchor="middle" font-weight="bold">DOWNBEAT</text>
  <line x1="630" y1="45" x2="630" y2="85" stroke="#ef5350" stroke-width="2"/>
  <text x="630" y="100" font-family="monospace" font-size="9" fill="#ef5350" text-anchor="middle">b6</text>
  <line x1="730" y1="45" x2="730" y2="85" stroke="#ef5350" stroke-width="2"/>
  <text x="730" y="100" font-family="monospace" font-size="9" fill="#ef5350" text-anchor="middle">b7</text>

  <!-- Simplified waveform between beats -->
  <polyline points="130,65 150,55 170,72 190,58 210,68 230,65" fill="none" stroke="#ffa726" stroke-width="1.5"/>
  <polyline points="230,65 260,50 290,75 310,48 330,65" fill="none" stroke="#ffa726" stroke-width="1.5"/>
  <polyline points="330,65 360,52 390,73 410,55 430,65" fill="none" stroke="#ffa726" stroke-width="1.5"/>
  <polyline points="430,65 460,48 490,78 510,45 530,65" fill="none" stroke="#ffa726" stroke-width="2.5"/>
  <polyline points="530,65 560,50 590,70 610,55 630,65" fill="none" stroke="#ffa726" stroke-width="1.5"/>
  <polyline points="630,65 660,58 690,70 710,60 730,65" fill="none" stroke="#ffa726" stroke-width="1.5"/>

  <!-- Video frames row -->
  <text x="40" y="140" font-family="Georgia, serif" font-size="12" fill="#555" font-weight="bold">Video:</text>

  <!-- Frame blocks aligned to beats -->
  <rect x="130" y="120" width="95" height="55" rx="4" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="177" y="145" font-family="Georgia, serif" font-size="10" fill="#0288d1" text-anchor="middle">Scene A</text>
  <text x="177" y="160" font-family="Georgia, serif" font-size="9" fill="#0288d1" text-anchor="middle">slow pan right</text>

  <rect x="230" y="120" width="95" height="55" rx="4" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="277" y="145" font-family="Georgia, serif" font-size="10" fill="#0288d1" text-anchor="middle">Scene A</text>
  <text x="277" y="160" font-family="Georgia, serif" font-size="9" fill="#0288d1" text-anchor="middle">zoom in</text>

  <rect x="330" y="120" width="95" height="55" rx="4" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="377" y="145" font-family="Georgia, serif" font-size="10" fill="#0288d1" text-anchor="middle">Scene B</text>
  <text x="377" y="160" font-family="Georgia, serif" font-size="9" fill="#0288d1" text-anchor="middle">fast motion</text>

  <rect x="430" y="120" width="95" height="55" rx="4" fill="#8bc34a" opacity="0.15" stroke="#8bc34a" stroke-width="1.5"/>
  <text x="477" y="145" font-family="Georgia, serif" font-size="10" fill="#558b2f" text-anchor="middle">Build-up</text>
  <text x="477" y="160" font-family="Georgia, serif" font-size="9" fill="#558b2f" text-anchor="middle">accelerating</text>

  <rect x="530" y="120" width="95" height="55" rx="4" fill="#ef5350" opacity="0.15" stroke="#ef5350" stroke-width="2"/>
  <text x="577" y="145" font-family="Georgia, serif" font-size="10" fill="#c62828" text-anchor="middle" font-weight="bold">Scene C</text>
  <text x="577" y="160" font-family="Georgia, serif" font-size="9" fill="#c62828" text-anchor="middle">HARD CUT</text>

  <rect x="630" y="120" width="95" height="55" rx="4" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="677" y="145" font-family="Georgia, serif" font-size="10" fill="#0288d1" text-anchor="middle">Scene C</text>
  <text x="677" y="160" font-family="Georgia, serif" font-size="9" fill="#0288d1" text-anchor="middle">drift</text>

  <!-- Transition labels -->
  <text x="227" y="195" font-family="Georgia, serif" font-size="9" fill="#999" text-anchor="middle">blend</text>
  <text x="327" y="195" font-family="Georgia, serif" font-size="9" fill="#999" text-anchor="middle">dissolve</text>
  <text x="427" y="195" font-family="Georgia, serif" font-size="9" fill="#999" text-anchor="middle">blend</text>
  <text x="527" y="195" font-family="Georgia, serif" font-size="9" fill="#ef5350" text-anchor="middle" font-weight="bold">hard cut</text>
  <text x="627" y="195" font-family="Georgia, serif" font-size="9" fill="#999" text-anchor="middle">blend</text>

  <!-- Energy mapping -->
  <text x="40" y="235" font-family="Georgia, serif" font-size="12" fill="#555" font-weight="bold">Energy:</text>
  <rect x="130" y="220" width="600" height="30" rx="4" fill="#eee"/>
  <!-- Energy bars -->
  <rect x="130" y="235" width="95" height="12" fill="#8bc34a" opacity="0.6"/>
  <rect x="230" y="232" width="95" height="15" fill="#8bc34a" opacity="0.7"/>
  <rect x="330" y="228" width="95" height="19" fill="#ffa726" opacity="0.7"/>
  <rect x="430" y="222" width="95" height="25" fill="#ffa726" opacity="0.8"/>
  <rect x="530" y="220" width="95" height="30" fill="#ef5350" opacity="0.7"/>
  <rect x="630" y="230" width="95" height="17" fill="#8bc34a" opacity="0.6"/>

  <text x="400" y="280" font-family="Georgia, serif" font-size="11" fill="#999" text-anchor="middle" font-style="italic">Energy level modulates motion intensity and camera dynamics per beat interval</text>
</svg>

---

## Music Visualization with Diffusion Models

Music visualization --- generating abstract or semi-abstract visuals that respond in real-time to audio --- is the oldest form of audio-driven visual generation. Winamp, MilkDrop, and their descendants have been doing this for decades with shader-based procedural graphics. Diffusion models bring this into a new era.

### The Classic Approach, Modernized

Traditional music visualizers map audio features to shader parameters:

| Audio Feature | Shader Parameter |
|:--|:--|
| Bass energy (20--200 Hz) | Background pulse, zoom |
| Mid energy (200--2000 Hz) | Particle density, color saturation |
| Treble energy (2000--16000 Hz) | Edge sharpness, sparkle effects |
| Beat onset | Transition trigger, flash |
| Spectral centroid | Color hue rotation |

A diffusion model replaces the shader. Instead of mapping audio features to handcrafted visual parameters, you map them to conditioning signals for a generative model. The visual output is no longer limited to procedural geometry --- it can be photorealistic scenes, painted landscapes, abstract art, anything the model can generate.

### Deforum and Stable Diffusion Animation

The Deforum extension for Stable Diffusion pioneered audio-reactive diffusion art. It works by:

1. Generating an initial keyframe with standard text-to-image.
2. Applying 2D or 3D transforms (zoom, rotation, translation) to the image, with transform magnitudes driven by audio features.
3. Adding noise to the transformed image and re-denoising with a low denoising strength (0.3--0.6).
4. Repeating for each frame.

The audio-driven transform equations:

$$\text{zoom}[m] = 1.0 + z_\text{base} + z_\text{scale} \cdot E_\text{bass}[m]$$

$$\theta[m] = \theta_\text{base} + \theta_\text{scale} \cdot E_\text{mid}[m]$$

$$dx[m] = d_\text{scale} \cdot \text{Chroma}[m, c_\text{dominant}]$$

where $E_\text{bass}[m]$ is the energy in the bass frequency band at frame $m$, $E_\text{mid}[m]$ is mid-band energy, and $c_\text{dominant}$ is the dominant chroma at that moment.

This approach is frame-by-frame (not temporally coherent in the diffusion model's latent space), which creates a characteristic "flowing" or "morphing" aesthetic. It is an aesthetic choice, not a limitation --- many music video creators prefer this dreamlike quality.

### Latent Space Interpolation

A more controlled approach generates keyframes at musically significant moments (downbeats, section changes) and interpolates in latent space between them:

$$z_t = \text{slerp}(z_A, z_B, \alpha(t))$$

where $z_A$ and $z_B$ are latent codes of consecutive keyframes, and $\alpha(t) \in [0, 1]$ is an interpolation parameter driven by the audio:

$$\alpha(t) = \frac{1}{2}\left(1 - \cos\left(\pi \cdot \frac{t - t_A}{t_B - t_A}\right)\right) + \delta \cdot O[m(t)]$$

The cosine term provides smooth baseline interpolation, and the onset envelope $O$ term adds audio-reactive acceleration --- the interpolation speeds up at beat onsets, creating a visual "pulse" synchronized to the rhythm.

Spherical linear interpolation (slerp) is preferred over linear interpolation for latent codes because it traverses the surface of the hypersphere in latent space, avoiding the low-probability interior:

$$\text{slerp}(z_A, z_B, \alpha) = \frac{\sin((1-\alpha)\Omega)}{\sin \Omega} z_A + \frac{\sin(\alpha\Omega)}{\sin \Omega} z_B$$

where $\Omega = \arccos\left(\frac{z_A \cdot z_B}{\|z_A\| \|z_B\|}\right)$.

---

## Multi-Modal Conditioning: Text + Audio + Image

The most powerful audio-driven video systems combine multiple conditioning signals: text for semantic content, audio for temporal structure, and reference images for visual style. But combining multiple conditions raises a fundamental question: how do you allocate the model's finite representational capacity across them?

### The Attention Budget Problem

In a cross-attention-based diffusion model, every conditioning signal competes for the same attention mechanism. With a text embedding sequence of length $L_\text{text}$ and an audio embedding sequence of length $L_\text{audio}$, the combined key/value sequence has length $L_\text{text} + L_\text{audio}$.

The attention weights for a visual query $q$ at position $(x, y, t)$ are:

$$\alpha_i = \frac{\exp(q \cdot k_i / \sqrt{d})}{\sum_{j=1}^{L_\text{text} + L_\text{audio}} \exp(q \cdot k_j / \sqrt{d})}$$

If text embeddings produce higher-magnitude key vectors (because the text encoder is more powerful or better aligned with the visual features), they will dominate the attention weights, and the audio signal will be drowned out. The reverse can also happen.

### Strategies for Balancing

**Separate cross-attention layers.** Instead of concatenating text and audio embeddings into a single K/V sequence, use separate cross-attention layers --- one for text, one for audio:

$$h' = h + \text{CrossAttn}_\text{text}(h, c_\text{text}) + \text{CrossAttn}_\text{audio}(h, c_\text{audio})$$

This gives each modality its own attention mechanism, preventing competition. The downside is increased computational cost (two cross-attention operations per layer instead of one).

**Learned gating.** Apply a learned gating mechanism that controls the contribution of each modality:

$$h' = h + g_\text{text} \cdot \text{CrossAttn}_\text{text}(h, c_\text{text}) + g_\text{audio} \cdot \text{CrossAttn}_\text{audio}(h, c_\text{audio})$$

where $g_\text{text}, g_\text{audio}$ are scalar gates (or per-channel gates) that the model learns during training. This allows the model to dynamically weight modalities depending on the content.

**Hierarchical conditioning.** Use text for global conditioning (via cross-attention at every layer) and audio for temporal modulation (via FiLM conditioning at temporal attention layers only). This respects the natural division: text provides "what to show" while audio provides "when and how to move."

<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg" style="max-width:800px; margin: 2em auto; display: block;">
  <rect width="800" height="400" fill="white"/>

  <text x="400" y="28" font-family="Georgia, serif" font-size="17" fill="#333" text-anchor="middle" font-weight="bold">Multi-Modal Conditioning Architecture</text>

  <!-- Text branch -->
  <rect x="40" y="60" width="100" height="40" rx="6" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="2"/>
  <text x="90" y="85" font-family="Georgia, serif" font-size="12" fill="#0288d1" text-anchor="middle" font-weight="bold">Text Prompt</text>

  <line x1="140" y1="80" x2="180" y2="80" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk3)"/>

  <rect x="180" y="60" width="100" height="40" rx="6" fill="#4fc3f7" opacity="0.15" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="230" y="82" font-family="Georgia, serif" font-size="11" fill="#0288d1" text-anchor="middle">CLIP / T5</text>

  <line x1="280" y1="80" x2="380" y2="150" stroke="#4fc3f7" stroke-width="1.5" marker-end="url(#arrowBlk3)"/>

  <!-- Audio branch -->
  <rect x="40" y="150" width="100" height="40" rx="6" fill="#ffa726" opacity="0.15" stroke="#ffa726" stroke-width="2"/>
  <text x="90" y="173" font-family="Georgia, serif" font-size="12" fill="#e65100" text-anchor="middle" font-weight="bold">Audio</text>

  <line x1="140" y1="170" x2="180" y2="170" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk3)"/>

  <rect x="180" y="150" width="100" height="40" rx="6" fill="#ffa726" opacity="0.15" stroke="#ffa726" stroke-width="1.5"/>
  <text x="230" y="172" font-family="Georgia, serif" font-size="11" fill="#e65100" text-anchor="middle">CLAP / BEATs</text>

  <line x1="280" y1="170" x2="380" y2="170" stroke="#ffa726" stroke-width="1.5" marker-end="url(#arrowBlk3)"/>

  <!-- Image branch -->
  <rect x="40" y="240" width="100" height="40" rx="6" fill="#8bc34a" opacity="0.15" stroke="#8bc34a" stroke-width="2"/>
  <text x="90" y="263" font-family="Georgia, serif" font-size="12" fill="#558b2f" text-anchor="middle" font-weight="bold">Ref. Image</text>

  <line x1="140" y1="260" x2="180" y2="260" stroke="#333" stroke-width="1.2" marker-end="url(#arrowBlk3)"/>

  <rect x="180" y="240" width="100" height="40" rx="6" fill="#8bc34a" opacity="0.15" stroke="#8bc34a" stroke-width="1.5"/>
  <text x="230" y="262" font-family="Georgia, serif" font-size="11" fill="#558b2f" text-anchor="middle">VAE / CLIP-I</text>

  <line x1="280" y1="260" x2="380" y2="190" stroke="#8bc34a" stroke-width="1.5" marker-end="url(#arrowBlk3)"/>

  <!-- Central fusion block -->
  <rect x="380" y="120" width="160" height="100" rx="10" fill="#ef5350" opacity="0.1" stroke="#ef5350" stroke-width="2"/>
  <text x="460" y="150" font-family="Georgia, serif" font-size="13" fill="#c62828" text-anchor="middle" font-weight="bold">Fusion Module</text>
  <text x="460" y="168" font-family="Georgia, serif" font-size="10" fill="#c62828" text-anchor="middle">Separate cross-attn</text>
  <text x="460" y="182" font-family="Georgia, serif" font-size="10" fill="#c62828" text-anchor="middle">+ learned gating</text>
  <text x="460" y="196" font-family="Georgia, serif" font-size="10" fill="#c62828" text-anchor="middle">+ temporal FiLM</text>

  <!-- To UNet -->
  <line x1="540" y1="170" x2="600" y2="170" stroke="#333" stroke-width="2" marker-end="url(#arrowBlk3)"/>

  <rect x="600" y="130" width="160" height="80" rx="8" fill="#333" opacity="0.06" stroke="#333" stroke-width="2"/>
  <text x="680" y="162" font-family="Georgia, serif" font-size="14" fill="#333" text-anchor="middle" font-weight="bold">Video DiT / UNet</text>
  <text x="680" y="180" font-family="Georgia, serif" font-size="11" fill="#555" text-anchor="middle">Denoising Network</text>
  <text x="680" y="195" font-family="Georgia, serif" font-size="10" fill="#555" text-anchor="middle">3D latent space</text>

  <!-- Labels for what each modality provides -->
  <text x="340" y="310" font-family="Georgia, serif" font-size="11" fill="#0288d1" text-anchor="start">Text: semantic content (what to show)</text>
  <text x="340" y="330" font-family="Georgia, serif" font-size="11" fill="#e65100" text-anchor="start">Audio: temporal structure (when to move)</text>
  <text x="340" y="350" font-family="Georgia, serif" font-size="11" fill="#558b2f" text-anchor="start">Image: visual style (how it looks)</text>

  <defs>
    <marker id="arrowBlk3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
    </marker>
  </defs>
</svg>

### CFG with Multiple Modalities

When applying classifier-free guidance with multiple conditions, you need to decide how to drop conditions during training and how to apply guidance at inference.

**Joint dropout**: Drop all conditions simultaneously with probability $p_\text{uncond}$. This learns a single unconditional mode and a single conditional mode. CFG is applied once:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s \cdot [\epsilon_\theta(x_t, c_\text{text}, c_\text{audio}) - \epsilon_\theta(x_t, \emptyset)]$$

**Independent dropout**: Drop each condition independently during training, allowing the model to learn conditional predictions for every subset of conditions. At inference, apply per-modality guidance:

$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + s_\text{text} \cdot [\epsilon_\theta(x_t, c_\text{text}, \emptyset_\text{audio}) - \epsilon_\theta(x_t, \emptyset)] + s_\text{audio} \cdot [\epsilon_\theta(x_t, \emptyset_\text{text}, c_\text{audio}) - \epsilon_\theta(x_t, \emptyset)]$$

This requires three forward passes instead of two but gives independent control over how strongly each modality influences the output. You can set $s_\text{audio} = 8$ and $s_\text{text} = 3$ to strongly follow the audio rhythm while loosely following the text description, or vice versa.

---

## Use Cases for Video Platforms

Audio-driven generation opens four distinct product surfaces for a video generation platform. Each has different technical requirements and different monetization profiles.

### 1. Music Video Generation

The user provides a song (MP3/WAV) and optionally a text description or style reference. The platform generates a complete music video.

**Technical flow:**
1. Extract audio features (beats, sections, energy, chroma).
2. Detect song structure: intro, verse, chorus, bridge, outro.
3. For each section, generate a visual concept (text prompt) that matches the section's energy and mood.
4. Generate keyframes at downbeat boundaries using image-to-video or text-to-video models.
5. Apply beat-synchronized transitions between sections.
6. Composite final video with the original audio track.

**Key challenge**: Song-level coherence. A 3-minute song at 120 BPM has ~360 beats and ~90 downbeats. Generating 90 coherent video segments that maintain visual continuity while varying with the music is a hard orchestration problem.

**Pricing**: Premium feature. Music video generation could be priced at $5--$20 per minute of output, justified by the multi-step pipeline and high compute cost.

### 2. Podcast Visualization

The user provides a podcast episode or audio narration. The platform generates background visuals that complement the spoken content.

**Technical flow:**
1. Transcribe audio using Whisper.
2. Segment transcript into topical chunks.
3. For each chunk, extract key topics and generate a visual prompt.
4. Generate ambient video segments (slow motion, abstract, scenic) aligned to topic boundaries.
5. Overlay text/graphics for emphasis on key quotes.

**Key insight**: Podcast visuals should be understated. They complement the audio rather than competing with it. Low-motion, aesthetically pleasing backgrounds work better than dynamic action sequences. The generation can be lower quality (fewer denoising steps, lower resolution) because the visual is secondary.

### 3. Sound Design to Video

The user provides a sound design document or collection of sound effects. The platform generates video that matches.

**Technical flow:**
1. Classify each sound effect (explosion, footstep, rain, etc.) using an audio classifier.
2. Align sound events on a timeline.
3. Generate video content that produces the correct visual source for each sound.
4. Ensure temporal alignment: the visual of a door slamming happens at the exact frame of the door slam sound.

This is an inversion of the usual pipeline --- instead of adding sound to video, you create video for existing sound.

### 4. Ad Creation: Voiceover to Video

The user provides a voiceover script (or a pre-recorded voiceover) and brand assets. The platform generates a video ad synchronized to the voiceover.

**Technical flow:**
1. Transcribe and segment the voiceover into sentences/phrases.
2. For each phrase, generate a visual scene that illustrates the spoken content.
3. Time scene changes to natural pause points in the voiceover.
4. Apply brand colors, logos, and product shots at specified points.
5. Add a music bed (AI-generated or from library) synchronized to the edit.

**Key commercial value**: This converts a $500 voiceover recording into a $5,000+ video ad at a fraction of the traditional production cost. The voiceover provides the temporal structure that makes the output feel professionally edited.

---

## Evaluation: Measuring Audio-Visual Sync

Evaluating audio-driven video quality is harder than evaluating text-to-video. In addition to standard video quality metrics (FVD, FID per frame, temporal consistency), you need metrics that measure audio-visual correspondence.

### Mean Opinion Score (MOS) for Audio-Visual Sync

The gold standard is human evaluation. Raters watch generated videos with their audio tracks and rate:

- **Overall quality** (1--5): Does the video look good?
- **Audio-visual sync** (1--5): Do the visuals match the audio?
- **Beat alignment** (1--5): Do visual transitions occur at musically appropriate moments?
- **Mood correspondence** (1--5): Does the visual mood match the audio mood?

The Mean Opinion Score (MOS) is the arithmetic mean across raters and criteria:

$$\text{MOS} = \frac{1}{N \cdot K} \sum_{n=1}^{N} \sum_{k=1}^{K} r_{n,k}$$

where $N$ is the number of raters and $K$ is the number of criteria. For statistical rigor, you need at least 20 raters per video and should report 95% confidence intervals.

### Audio-Visual Correspondence (AVC) Metric

An automated metric based on learned audio-visual embeddings. Train a classifier to predict whether an audio clip and a video clip are from the same source (positive pairs) or different sources (negative pairs). The AVC score is the classifier's accuracy on held-out test data:

$$\text{AVC} = P(\hat{y} = y \mid (a, v))$$

where $a$ is an audio clip, $v$ is a video clip, $y$ is the true label (matched or mismatched), and $\hat{y}$ is the classifier's prediction. A higher AVC indicates better audio-visual correspondence in the generated content.

### Beat Alignment Score (BAS)

A simple, interpretable metric. Given detected beats $\{b_k\}$ and detected visual transitions $\{v_j\}$ (via optical flow peaks or scene change detection):

$$\text{BAS} = \frac{1}{K} \sum_{k=1}^{K} \max_j \exp\left(-\frac{(b_k - v_j)^2}{2\tau^2}\right)$$

where $\tau$ is a tolerance window (typically 50--100 ms). This measures, for each beat, how close the nearest visual transition is. A BAS of 1.0 means every beat has a visual event within the tolerance window; a BAS of 0.0 means no beats are visually marked.

### Onset-Motion Correlation

The Pearson correlation between the audio onset envelope and the video motion magnitude (computed from optical flow):

$$\rho = \frac{\text{Cov}(O, M)}{\sigma_O \cdot \sigma_M}$$

where $O$ is the onset strength envelope (resampled to video frame rate) and $M$ is the mean optical flow magnitude per frame. Higher positive correlation indicates that visual motion tracks audio intensity.

| Metric | What It Measures | Automated? | Target Range |
|:--|:--|:-:|:--|
| MOS (sync) | Human-perceived sync quality | No | > 3.5/5.0 |
| AVC | Learned audio-visual match | Yes | > 0.75 |
| BAS | Beat-to-visual transition alignment | Yes | > 0.6 |
| Onset-Motion $\rho$ | Audio energy to video motion correlation | Yes | > 0.4 |
| FVD | Video quality (no audio) | Yes | < 300 |

---

## Latency and Production Considerations

### End-to-End Latency

For an audio-driven music video pipeline generating 30 seconds of output:

| Stage | Time | Notes |
|:--|:--|:--|
| Audio feature extraction | 1--2 s | librosa on CPU, fast |
| Beat/section analysis | 1--3 s | Includes downbeat classification |
| Per-section prompt generation | 2--5 s | LLM call (Gemini Flash) |
| Keyframe generation (8 scenes) | 40--80 s | Parallel image gen, ~5--10s each |
| Scene interpolation / video gen | 60--180 s | Per-scene video generation |
| Beat-aligned compositing | 5--10 s | FFmpeg, GPU-accelerated |
| Audio-video mux | 1--2 s | FFmpeg |
| **Total** | **~2--5 min** | **For 30s of output** |

This is production-feasible but not real-time. For music visualization (abstract art), latency can be much lower because you are generating frame-by-frame with img2img rather than full video segments.

### Batch Processing for Long-Form Content

A 3-minute music video requires generating ~30--60 video segments. Sequential generation would take 15--30 minutes. Parallel generation on multiple GPUs can bring this down to 2--5 minutes, but requires:

1. **Segment independence**: Each segment is generated independently, with only the text prompt and style reference connecting them.
2. **Edge consistency**: The last frame of segment $n$ must match the first frame of segment $n+1$. This can be achieved by using the last frame of segment $n$ as a conditioning image for segment $n+1$ (image-to-video generation), but this creates sequential dependency.
3. **Temporal overlap**: Generate segments with overlapping frames and cross-dissolve in the overlap region.

The tradeoff between parallelism and visual continuity is a core engineering challenge. In practice, a hierarchical approach works well: generate all keyframes in parallel (image generation, fast), then generate video segments sequentially using keyframes as anchors (slower, but each segment only needs the previous keyframe, not the previous segment).

### Real-Time Music Visualization

For real-time applications (live music visualization, DJ tools), the pipeline must run under 40ms per frame at 24fps. This is achievable with:

- **Pre-computed latent interpolation paths**: Before the performance, generate a library of latent codes for different visual states. During performance, interpolate between pre-computed latents based on real-time audio features.
- **Lightweight diffusion**: Use 2--4 denoising steps with consistency models or LCM (Latent Consistency Models), which can generate a single frame in 20--50ms on modern GPUs.
- **VAE-only generation**: Skip the diffusion process entirely. Use a VAE decoder to convert audio-conditioned latent codes directly to images. Quality is lower but latency is sub-10ms.

---

## The Road Ahead

Audio-driven video generation sits at an intersection that will become increasingly important as AI video matures. Text conditioning is reaching a ceiling --- there are only so many words you can use to describe temporal dynamics. Audio conditioning fills that gap with natural, high-bandwidth temporal information that humans already know how to create and manipulate.

The key technical challenges remaining:

1. **Native audio conditioning in foundation models.** Today's best video models (Veo, Sora, Kling) are text-conditioned. Adding audio as a first-class conditioning modality during pretraining, rather than as a post-hoc add-on, would dramatically improve audio-visual alignment.

2. **Joint audio-video generation.** Instead of generating video conditioned on audio, generate both simultaneously from a shared latent space. This would produce naturally synchronized audio-visual content without explicit alignment steps.

3. **Long-form temporal coherence.** Current models generate 4--10 second clips. Music videos are 3--5 minutes. Maintaining visual coherence and audio alignment across hundreds of clips is an open problem in orchestration and consistency.

4. **Real-time performance.** The gap between offline generation (minutes per second of output) and real-time needs (24+ fps) is still large. Consistency models and architectural innovations are closing it, but we are not there yet for high-quality output.

For platform builders: audio-driven generation is a differentiation opportunity. Most video platforms today are text-to-video. Adding audio conditioning creates a new product surface --- music videos, podcast visuals, audio-synced ads --- that your competitors have not built. The technical components (audio feature extraction, beat alignment, multi-modal conditioning) are well-understood individually. The engineering challenge is assembling them into a reliable, scalable pipeline.

The creator who uploads a song and gets back a perfectly beat-synced music video in 5 minutes --- that is the product. And the technical stack to build it is now within reach.

---

## References

1. Ellis, D. P. W. (2007). *Beat Tracking by Dynamic Programming.* Journal of New Music Research, 36(1), 51--60.
2. Elizalde, B. et al. (2023). *CLAP: Learning Audio Concepts From Natural Language Supervision.* ICASSP 2023.
3. Xu, Y. et al. (2022). *AudioMAE: Masked Autoencoders that Listen.* NeurIPS 2022.
4. McFee, B. et al. (2015). *librosa: Audio and Music Signal Analysis in Python.* SciPy 2015.
5. Owens, A. et al. (2016). *Visually Indicated Sounds.* CVPR 2016.
6. Chen, H. et al. (2023). *Aligned Audio-Visual Generation with Text Prompts.* arXiv:2309.11489.
7. Luo, S. et al. (2023). *Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference.* arXiv:2310.04378.
8. Zhu, Y. et al. (2022). *Quantized GAN for Complex Music Generation from Dance Videos.* ECCV 2022.
9. Perez, E. et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018.
10. Arandjelovic, R. & Zisserman, A. (2017). *Look, Listen and Learn.* ICCV 2017.
