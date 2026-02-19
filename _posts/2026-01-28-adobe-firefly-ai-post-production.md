---
layout: post
title: "Adobe Firefly and the Automation of Post-Production: Generative Extend, AI Object Masking, and Building Competitive Editing Features"
date: 2026-01-28
category: market
---

On January 20, 2026, Adobe dropped six AI features into Premiere Pro and After Effects simultaneously. Not a keynote tease. Not a "coming later this year." Shipped. Available. In the apps that 70% of professional editors already use.

This post is not a feature review. It is a complete technical breakdown of what Adobe shipped, what it means for the post-production automation market, and exactly how to build competitive post-processing features into your own video generation platform -- with code, cost analysis, and architecture diagrams.

---

## Table of Contents

1. [Everything Adobe Shipped in January 2026](#everything-adobe-shipped-in-january-2026)
2. [The Strategic Signal](#the-strategic-signal)
3. [Building Duration Extension ("Make Longer")](#building-duration-extension)
4. [Light Editing Suite with FFmpeg](#light-editing-suite-with-ffmpeg)
5. [Smart Aspect Ratio Adaptation](#smart-aspect-ratio-adaptation)
6. [Auto-Subtitles with Whisper](#auto-subtitles-with-whisper)
7. [Build vs. Buy Analysis](#build-vs-buy-analysis)
8. [The Editing UX Challenge](#the-editing-ux-challenge)
9. [Market Sizing](#market-sizing)
10. [Conclusion](#conclusion)

---

## Everything Adobe Shipped in January 2026

### 1. Generative Extend

Generative Extend adds frames to the beginning or end of any video clip using the Firefly Video Model (built on top of Adobe's proprietary diffusion transformer architecture). The user drags a clip edge beyond its duration, and Firefly synthesizes temporally-consistent continuation frames.

**Technical details:**

- **Model**: Firefly Video Model v3, a DiT-based diffusion model conditioned on the final (or initial) N frames of the source clip.
- **Context window**: The model ingests the last 16 frames (approximately 0.53 seconds at 30fps) as conditioning input. This is encoded through a 3D VAE into a latent representation, and the model generates a continuation in latent space that is then decoded.
- **Maximum extension per operation**: Up to 2 seconds per invocation. Users can chain multiple extensions, but each invocation only "sees" the most recent context frames.
- **Resolution**: Matches source clip resolution up to 4K. The model generates at a base resolution (likely 720p internally) and upscales via Firefly's super-resolution module.
- **Latency**: Approximately 10-15 seconds for a 2-second extension on Adobe's cloud infrastructure.
- **Quality characteristics**: Strong temporal consistency for simple camera motions (pans, zooms, static shots). Degradation visible on complex multi-object scenes with occluded motion after 3+ chained extensions.

**How it works (inferred architecture):**

```
Source clip:  [frame_0, frame_1, ..., frame_N]
                                        |
                Extract last K frames:  [frame_{N-K+1}, ..., frame_N]
                                        |
                3D VAE Encoder:         z_context = Encode(last_K_frames)
                                        |
                DiT Denoising Loop:     z_new = Denoise(noise, z_context, text_cond)
                                        |
                3D VAE Decoder:         [frame_{N+1}, ..., frame_{N+M}]
                                        |
                Cross-fade stitch:      Blend frames at boundary
                                        |
Output clip:  [frame_0, ..., frame_N, ..., frame_{N+M}]
```

The critical design decision is the cross-fade stitch at the boundary. Without it, there is a visible seam where generated frames begin. Adobe applies a 4-8 frame (0.13-0.27 second) blending window where the last few source frames and first few generated frames are alpha-composited.

### 2. AI Object Mask

AI Object Mask is a re-engineered object segmentation and tracking system in Premiere Pro. The user hovers over an object, sees a real-time preview of the segmentation mask, clicks to confirm, and the mask automatically propagates through the entire clip.

**Technical details:**

- **Segmentation model**: Based on a SAM-2 (Segment Anything Model 2) variant, fine-tuned on Adobe's internal video dataset. SAM-2 uses a Hiera image encoder backbone with a memory-conditioned mask decoder.
- **Tracking speed**: Adobe claims 20x faster than the previous roto-brush tool. Previous roto-brush: ~2-3 fps processing. New AI Object Mask: ~40-60 fps processing. On a clip with 150 frames, this means tracking completes in ~2.5-3.8 seconds rather than ~50-75 seconds.
- **Memory mechanism**: SAM-2 maintains a memory bank of past segmented frames. When tracking forward, it conditions on the most recent 6 memory frames plus the initial annotation frame, allowing it to handle occlusions and reappearances.
- **Multi-object**: Supports simultaneous tracking of multiple objects. Each object gets an independent mask track.
- **Edge quality**: Sub-pixel accurate boundaries using the mask decoder's upsampling head. Soft edges on hair and fur thanks to an alpha-matte output mode.

**Performance comparison:**

| Metric | Old Roto-Brush | AI Object Mask |
|--------|---------------|----------------|
| Processing speed | 2-3 fps | 40-60 fps |
| Time for 150-frame clip | 50-75 sec | 2.5-3.8 sec |
| Manual corrections needed | 15-30% of frames | 2-5% of frames |
| Occlusion recovery | Manual | Automatic |
| Multi-object | Sequential | Simultaneous |
| Edge quality (hair/fur) | Hard matte | Soft alpha |

### 3. Firefly Audio Model

Adobe's Firefly Audio Model generates three types of audio content:

- **Music generation**: Text-prompted music in various genres, tempos, and moods. Duration-matched to video clip length. Stems-based output (separate tracks for drums, bass, melody, etc.) for mixing flexibility.
- **Sound effects (SFX)**: Environmental sounds, foley, and impact effects. Context-aware generation based on video content analysis.
- **Speech synthesis**: Multi-language voiceover with emotion control (neutral, excited, somber, urgent) and pacing parameters.

**The licensing moat**: All Firefly Audio outputs are trained exclusively on Adobe Stock licensed content and public domain material. Adobe provides IP indemnification for Creative Cloud subscribers. This is not a minor detail -- it is the single most important differentiator against open-source audio generation tools. For commercial use at scale, copyright risk is existential.

**Audio specifications:**

| Parameter | Value |
|-----------|-------|
| Sample rate | 44.1 kHz |
| Bit depth | 24-bit |
| Format | WAV (lossless) |
| Maximum duration | 60 seconds per generation |
| Stem separation | Yes (drums, bass, melody, vocals) |
| Languages (speech) | 12 languages at launch |
| Latency | ~5-8 seconds for a 30-second music clip |

### 4. Firefly Video Editor (Public Beta)

A browser-based multi-track video editor built on web technologies (Canvas + WebCodecs API + WASM-based video decode). Key characteristics:

- **Timeline**: Multi-track timeline supporting video, audio, text, and effects layers.
- **Rendering**: Client-side preview rendering using WebCodecs for hardware-accelerated decode. Final export rendered server-side on Adobe's infrastructure.
- **Integration**: Direct access to Firefly generation models. Users can generate, edit, and export without leaving the browser.
- **Collaboration**: Real-time multiplayer editing (Google Docs-style) with conflict resolution.
- **Storage**: Integrated with Adobe Creative Cloud storage. Projects sync across devices.

The strategic play is obvious: capture users who start with generation and keep them through editing and delivery. The browser-based approach means no app install -- lower friction than Premiere Pro.

### 5. Camera Motion Transfer

Upload a reference video showing desired camera motion (pan, tilt, dolly, orbit, zoom). Upload a start frame. Firefly generates a new video that applies the reference motion to the start frame content.

**Technical details:**

- **Motion extraction**: The reference video is processed through a motion estimation network that extracts a camera trajectory representation -- likely a sequence of 6-DOF (rotation + translation) camera pose deltas, similar to the approach in MotionCtrl or CameraCtrl research.
- **Conditioning**: The extracted motion trajectory is injected into the diffusion transformer via cross-attention or additive conditioning on the spatial attention layers.
- **Advantages over text-based motion**: "Slow pan left while slightly zooming in" is ambiguous. A reference video showing exactly that motion is unambiguous. This is a UX insight, not a model insight -- the underlying capability (motion-conditioned generation) exists in several models.

---

## The Strategic Signal

### Post-Production Automation as a Market Category

Adobe's January release is not six independent features. It is a coordinated declaration that **post-production automation** is a product category, and Adobe intends to own it.

Consider Adobe's revenue structure:

| Segment | Annual Revenue (FY2025) | % of Total |
|---------|------------------------|------------|
| Digital Media | ~$14.5B | ~73% |
| Creative Cloud (subset) | ~$12.5B | ~63% |
| Document Cloud | ~$3.1B | ~16% |
| Digital Experience | ~$5.3B | ~27% |

Creative Cloud generates approximately $12.5 billion per year. At ~27 million paid subscribers, that is approximately $463 per subscriber per year, or roughly $39/month average revenue.

**Why Adobe can afford to give AI features away in Creative Cloud:**

The AI features are not a separate revenue line. They are subscriber retention and acquisition tools. Adobe's strategic calculus:

$$
\text{Value of AI features} = \Delta\text{Retention} \times \text{LTV} + \Delta\text{Acquisition} \times \text{LTV}
$$

If AI features reduce monthly churn by even 0.5% (from ~3.5% to 3.0% hypothetically), the revenue impact is:

$$
\Delta\text{Revenue} = N \times \text{ARPU}_{\text{monthly}} \times \frac{1}{\text{churn}_{\text{new}}} - N \times \text{ARPU}_{\text{monthly}} \times \frac{1}{\text{churn}_{\text{old}}}
$$

$$
= 27M \times \$39 \times \left(\frac{1}{0.030} - \frac{1}{0.035}\right) = 27M \times \$39 \times (33.3 - 28.6) = 27M \times \$39 \times 4.76
$$

$$
\approx \$5.01B \text{ in additional lifetime revenue across the subscriber base}
$$

Even if the actual churn reduction is a tenth of that, it justifies massive R&D investment in AI features. Adobe is playing a retention game, not a per-feature monetization game.

### What This Means for Independent Platform Builders

Adobe is not your direct competitor -- they are selling to professional editors at $55-83/month. But they are setting user expectations. When a user tries Generative Extend in Premiere Pro, they will expect a "Make Longer" button in your platform. When they use AI Object Mask, they will expect smart cropping in your platform.

The features you need to build are not copies of Adobe's features. They are the **equivalents that make sense in a generation-first workflow:**

| Adobe Feature | Your Platform Equivalent | Complexity |
|--------------|--------------------------|------------|
| Generative Extend | Duration Extension (I2V chaining) | Medium |
| AI Object Mask | Smart subject tracking for cropping | Medium-High |
| Firefly Audio | Audio API integration | Low (API) |
| Video Editor | Light editing timeline | High |
| Camera Motion Transfer | Motion presets / reference upload | Medium |

Let me now walk through the implementation of each equivalent feature in detail.

---

## Building Duration Extension

The most requested feature in AI video: "Make it longer." Every model has duration limits. Veo 3.1 caps at 8 seconds. Kling 3.0 caps at 10 seconds. Sora 2 caps at 20 seconds but quality degrades after 10. Users want 15, 30, 60 seconds. Generative Extend solves this.

### The Algorithm

Duration extension is implemented as **iterative image-to-video (I2V) chaining** with cross-fade stitching. Here is the complete algorithm:

```
ALGORITHM: Video Duration Extension
INPUT: source_video (duration D seconds), target_duration (T seconds)
OUTPUT: extended_video (duration ~T seconds)

1. Let current_video = source_video
2. While duration(current_video) < T:
   a. Extract last frame: last_frame = extract_frame(current_video, -1)
   b. Extract context frames: context = extract_frames(current_video, -16, -1)
   c. Generate continuation:
      new_segment = I2V_generate(
          start_frame = last_frame,
          context_frames = context,  // if model supports it
          prompt = original_prompt,
          duration = min(model_max_duration, T - duration(current_video))
      )
   d. Cross-fade stitch:
      extended = crossfade_stitch(current_video, new_segment, overlap=8_frames)
   e. current_video = extended
3. Trim current_video to exactly T seconds
4. Return current_video
```

### Step-by-Step Implementation

#### Step 1: Extract the Last Frame

```python
import subprocess
import os

def extract_last_frame(video_path: str, output_path: str) -> str:
    """Extract the last frame of a video as a PNG image."""
    # First, get total frame count
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets', '-show_entries',
        'stream=nb_read_packets',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    total_frames = int(result.stdout.strip())

    # Extract last frame
    extract_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'select=eq(n\\,{total_frames - 1})',
        '-vframes', '1',
        '-q:v', '1',  # highest quality JPEG, or use PNG
        output_path
    ]
    subprocess.run(extract_cmd, check=True)
    return output_path


def extract_last_n_frames(video_path: str, n: int, output_dir: str) -> list:
    """Extract the last N frames as individual images."""
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets', '-show_entries',
        'stream=nb_read_packets',
        '-of', 'csv=p=0',
        video_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    total_frames = int(result.stdout.strip())

    start_frame = max(0, total_frames - n)

    extract_cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'select=between(n\\,{start_frame}\\,{total_frames - 1})',
        '-vsync', 'vfr',
        '-q:v', '1',
        os.path.join(output_dir, 'context_%04d.png')
    ]
    subprocess.run(extract_cmd, check=True)
    return [os.path.join(output_dir, f'context_{i:04d}.png')
            for i in range(1, n + 1)]
```

#### Step 2: Generate Continuation with I2V

```python
import httpx
import base64
import time

async def generate_continuation(
    start_frame_path: str,
    prompt: str,
    duration_seconds: float = 5.0,
    model: str = "kling-v3",
    api_key: str = None
) -> str:
    """Generate a video continuation from a start frame using I2V."""

    # Read and encode the start frame
    with open(start_frame_path, 'rb') as f:
        frame_b64 = base64.b64encode(f.read()).decode()

    if model == "kling-v3":
        return await _generate_kling(frame_b64, prompt, duration_seconds, api_key)
    elif model == "veo-31":
        return await _generate_veo(frame_b64, prompt, duration_seconds, api_key)
    elif model == "wan-22":
        return await _generate_wan(frame_b64, prompt, duration_seconds)
    else:
        raise ValueError(f"Unknown model: {model}")


async def _generate_kling(
    frame_b64: str, prompt: str, duration: float, api_key: str
) -> str:
    """Kling 3.0 I2V generation."""
    async with httpx.AsyncClient(timeout=300) as client:
        # Submit generation request
        response = await client.post(
            "https://api.klingai.com/v1/videos/image2video",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model_name": "kling-v3",
                "image": frame_b64,
                "prompt": prompt,
                "duration": str(min(duration, 10.0)),  # Kling max: 10s
                "cfg_scale": 0.5,
                "mode": "pro"  # Higher quality
            }
        )
        task_id = response.json()["data"]["task_id"]

        # Poll for completion
        while True:
            status = await client.get(
                f"https://api.klingai.com/v1/videos/image2video/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            data = status.json()["data"]
            if data["task_status"] == "succeed":
                video_url = data["task_result"]["videos"][0]["url"]
                # Download video
                video_response = await client.get(video_url)
                output_path = f"/tmp/continuation_{task_id}.mp4"
                with open(output_path, 'wb') as f:
                    f.write(video_response.content)
                return output_path
            elif data["task_status"] == "failed":
                raise RuntimeError(f"Generation failed: {data}")
            await asyncio.sleep(2)
```

#### Step 3: Cross-Fade Stitch

The cross-fade is critical for seamless extensions. Without it, there is a visible jump at every extension boundary. The overlap region blends the tail of the source with the head of the continuation.

```python
def crossfade_stitch(
    video_a: str,
    video_b: str,
    output_path: str,
    overlap_frames: int = 8,
    fps: int = 30
) -> str:
    """
    Stitch two videos with a cross-fade at the boundary.

    The last `overlap_frames` of video_a are blended with the
    first `overlap_frames` of video_b.
    """
    overlap_duration = overlap_frames / fps

    # Get duration of video_a
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_a
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    duration_a = float(result.stdout.strip())

    # Calculate trim points
    # video_a plays fully, but its last overlap_duration fades out
    # video_b starts at overlap_duration offset, but its first
    #   overlap_duration fades in
    trim_a_end = duration_a
    trim_b_start = overlap_frames  # skip first N frames of continuation

    # FFmpeg complex filter for cross-fade
    # This uses xfade filter for frame-accurate blending
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-i', video_a,
        '-i', video_b,
        '-filter_complex',
        f'[0:v][1:v]xfade=transition=fade:'
        f'duration={overlap_duration}:'
        f'offset={duration_a - overlap_duration}[outv]',
        '-map', '[outv]',
        '-c:v', 'libx264',
        '-crf', '18',
        '-preset', 'fast',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return output_path
```

#### Step 4: The Complete Extension Pipeline

```python
import asyncio

async def extend_video(
    source_video: str,
    target_duration: float,
    prompt: str,
    model: str = "kling-v3",
    api_key: str = None,
    overlap_frames: int = 8,
    fps: int = 30
) -> str:
    """
    Extend a video to a target duration by iteratively generating
    continuations and stitching them together.

    Args:
        source_video: Path to the source video file
        target_duration: Desired final duration in seconds
        prompt: Text prompt for continuation generation
        model: Model identifier for I2V generation
        api_key: API key for the generation service
        overlap_frames: Number of frames for cross-fade overlap
        fps: Frame rate of the video

    Returns:
        Path to the extended video file
    """
    current_video = source_video
    iteration = 0

    while True:
        # Check current duration
        probe_cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            current_video
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        current_duration = float(result.stdout.strip())

        if current_duration >= target_duration:
            break

        # Calculate remaining duration needed
        remaining = target_duration - current_duration
        # Account for overlap: we lose overlap_frames worth of time
        generation_duration = min(
            remaining + (overlap_frames / fps),
            10.0  # Model max duration
        )

        print(f"Iteration {iteration}: {current_duration:.1f}s -> "
              f"generating {generation_duration:.1f}s continuation")

        # Extract last frame
        last_frame = f"/tmp/extend_frame_{iteration}.png"
        extract_last_frame(current_video, last_frame)

        # Generate continuation
        continuation = await generate_continuation(
            start_frame_path=last_frame,
            prompt=prompt,
            duration_seconds=generation_duration,
            model=model,
            api_key=api_key
        )

        # Stitch
        output = f"/tmp/extended_{iteration}.mp4"
        crossfade_stitch(
            current_video, continuation, output,
            overlap_frames=overlap_frames, fps=fps
        )

        current_video = output
        iteration += 1

    # Final trim to exact target duration
    final_output = f"/tmp/extended_final.mp4"
    trim_cmd = [
        'ffmpeg', '-y',
        '-i', current_video,
        '-t', str(target_duration),
        '-c:v', 'libx264',
        '-crf', '18',
        output
    ]
    subprocess.run(trim_cmd, check=True)

    return final_output
```

### Extension Quality Degradation

Each extension iteration introduces a small amount of quality drift. The generated continuation never perfectly matches the style and content of the source. Over multiple iterations, this compounds.

**Empirical quality degradation curve (approximate):**

```
Quality (subjective 1-10)
10 |*
   | *
 9 |  *
   |   *
 8 |    *
   |      *
 7 |        *
   |           *
 6 |              *
   |                  *
 5 |                       *
   |                             *
 4 |________________________________
   0  1  2  3  4  5  6  7  8  9  10
              Extension Iterations
```

**Practical limits:**

| Extensions | Total Duration (from 5s source) | Quality | Recommendation |
|-----------|-------------------------------|---------|----------------|
| 0 | 5 seconds | Excellent | -- |
| 1 | ~10 seconds | Very Good | Safe for production |
| 2 | ~15 seconds | Good | Acceptable for most uses |
| 3 | ~20 seconds | Fair | Review before delivery |
| 4+ | 25+ seconds | Declining | Use with caution |

The key insight: **after 3 extensions, consider regenerating from scratch with a longer-duration model** (e.g., Sora 2 at 20 seconds) rather than continuing to chain.

---

## Light Editing Suite with FFmpeg

You do not need to build Premiere Pro. You need to build enough editing that 80% of users never leave your platform. Here is the complete FFmpeg command reference for every editing operation your platform needs.

### Trim (Cut Start/End)

```bash
# Trim from timestamp 00:00:02 to 00:00:07 (5 seconds)
ffmpeg -y \
  -i input.mp4 \
  -ss 00:00:02 \
  -to 00:00:07 \
  -c:v libx264 -crf 18 -preset fast \
  -c:a aac -b:a 192k \
  -pix_fmt yuv420p \
  output_trimmed.mp4

# Trim using frame numbers (more precise)
# Keep frames 60-210 (at 30fps: seconds 2-7)
ffmpeg -y \
  -i input.mp4 \
  -vf "select=between(n\,60\,210),setpts=PTS-STARTPTS" \
  -af "aselect=between(t\,2\,7),asetpts=PTS-STARTPTS" \
  -c:v libx264 -crf 18 \
  output_trimmed_precise.mp4
```

### Stitch (Concatenate Multiple Clips)

```bash
# Method 1: Concat demuxer (same codec, fastest)
# Create a file list
echo "file 'clip1.mp4'" > /tmp/concat_list.txt
echo "file 'clip2.mp4'" >> /tmp/concat_list.txt
echo "file 'clip3.mp4'" >> /tmp/concat_list.txt

ffmpeg -y \
  -f concat -safe 0 \
  -i /tmp/concat_list.txt \
  -c copy \
  output_stitched.mp4

# Method 2: Filter-based concat (different codecs/resolutions)
ffmpeg -y \
  -i clip1.mp4 -i clip2.mp4 -i clip3.mp4 \
  -filter_complex \
    "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[outv][outa]" \
  -map "[outv]" -map "[outa]" \
  -c:v libx264 -crf 18 \
  -c:a aac -b:a 192k \
  output_stitched.mp4

# Method 3: Concat with cross-fade transitions between clips
ffmpeg -y \
  -i clip1.mp4 -i clip2.mp4 \
  -filter_complex \
    "[0:v][1:v]xfade=transition=fade:duration=0.5:offset=4.5[outv]; \
     [0:a][1:a]acrossfade=d=0.5[outa]" \
  -map "[outv]" -map "[outa]" \
  -c:v libx264 -crf 18 \
  output_crossfade.mp4
```

### Audio Overlay

```bash
# Add background music at 30% volume, keeping original audio
ffmpeg -y \
  -i video.mp4 \
  -i background_music.mp3 \
  -filter_complex \
    "[1:a]volume=0.3[music]; \
     [0:a][music]amix=inputs=2:duration=first:dropout_transition=3[outa]" \
  -map "0:v" -map "[outa]" \
  -c:v copy \
  -c:a aac -b:a 192k \
  output_with_music.mp4

# Replace audio entirely
ffmpeg -y \
  -i video.mp4 \
  -i narration.mp3 \
  -map 0:v -map 1:a \
  -c:v copy \
  -c:a aac -b:a 192k \
  -shortest \
  output_new_audio.mp4

# Add audio to a silent video
ffmpeg -y \
  -i silent_video.mp4 \
  -i soundtrack.mp3 \
  -map 0:v -map 1:a \
  -c:v copy \
  -c:a aac -b:a 192k \
  -shortest \
  output_with_audio.mp4
```

### Basic Color Grading

```bash
# Brightness (+0.1), Contrast (1.2x), Saturation (1.3x)
ffmpeg -y \
  -i input.mp4 \
  -vf "eq=brightness=0.1:contrast=1.2:saturation=1.3" \
  -c:v libx264 -crf 18 \
  -c:a copy \
  output_graded.mp4

# Color temperature adjustment (warm shift)
# Increase red channel, decrease blue channel
ffmpeg -y \
  -i input.mp4 \
  -vf "colorbalance=rs=0.1:gs=0:bs=-0.1:rm=0.05:gm=0:bm=-0.05" \
  -c:v libx264 -crf 18 \
  -c:a copy \
  output_warm.mp4

# Cinematic look (desaturate + add contrast + vignette)
ffmpeg -y \
  -i input.mp4 \
  -vf " \
    eq=saturation=0.8:contrast=1.15:brightness=-0.02, \
    unsharp=5:5:0.8:5:5:0, \
    vignette=PI/4 \
  " \
  -c:v libx264 -crf 18 \
  -c:a copy \
  output_cinematic.mp4

# LUT (Look-Up Table) application for professional color grades
# Apply a .cube LUT file
ffmpeg -y \
  -i input.mp4 \
  -vf "lut3d=cinematic_grade.cube" \
  -c:v libx264 -crf 18 \
  -c:a copy \
  output_lut.mp4
```

### Speed Adjustment

```bash
# 2x speed (fast forward)
ffmpeg -y \
  -i input.mp4 \
  -filter_complex \
    "[0:v]setpts=0.5*PTS[v]; \
     [0:a]atempo=2.0[a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -crf 18 \
  output_2x.mp4

# 0.5x speed (slow motion)
ffmpeg -y \
  -i input.mp4 \
  -filter_complex \
    "[0:v]setpts=2.0*PTS[v]; \
     [0:a]atempo=0.5[a]" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -crf 18 \
  output_slow.mp4

# Smooth slow motion with motion interpolation (minterpolate)
ffmpeg -y \
  -i input.mp4 \
  -vf "minterpolate=fps=60:mi_mode=mci:mc_mode=aobmc:vsbmc=1,setpts=2.0*PTS" \
  -c:v libx264 -crf 18 \
  output_smooth_slow.mp4
```

### Text Overlay

```bash
# Simple text overlay (bottom center, white with shadow)
ffmpeg -y \
  -i input.mp4 \
  -vf "drawtext= \
    text='Generated with AI': \
    fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf: \
    fontsize=36: \
    fontcolor=white: \
    shadowcolor=black: \
    shadowx=2:shadowy=2: \
    x=(w-text_w)/2: \
    y=h-text_h-40" \
  -c:v libx264 -crf 18 \
  -c:a copy \
  output_text.mp4

# Animated text (fade in at 1s, fade out at 4s)
ffmpeg -y \
  -i input.mp4 \
  -vf "drawtext= \
    text='Product Demo': \
    fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf: \
    fontsize=48: \
    fontcolor=white: \
    x=(w-text_w)/2:y=(h-text_h)/2: \
    enable='between(t,1,4)': \
    alpha='if(lt(t,1.5),((t-1)/0.5),if(gt(t,3.5),((4-t)/0.5),1))'" \
  -c:v libx264 -crf 18 \
  -c:a copy \
  output_animated_text.mp4
```

### Complete FFmpeg Post-Processing Pipeline

Here is a Python class that wraps all of these operations into a clean API:

```python
import subprocess
import json
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ColorGrade:
    brightness: float = 0.0     # -1.0 to 1.0
    contrast: float = 1.0       # 0.0 to 3.0
    saturation: float = 1.0     # 0.0 to 3.0
    temperature: float = 0.0    # -1.0 (cool) to 1.0 (warm)


class VideoEditor:
    """FFmpeg-based video editing operations for a SaaS platform."""

    def __init__(self, ffmpeg_path: str = "ffmpeg",
                 ffprobe_path: str = "ffprobe"):
        self.ffmpeg = ffmpeg_path
        self.ffprobe = ffprobe_path

    def get_info(self, video_path: str) -> dict:
        """Get video metadata (duration, resolution, fps, codec)."""
        cmd = [
            self.ffprobe, '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return json.loads(result.stdout)

    def trim(self, input_path: str, output_path: str,
             start: float, end: float) -> str:
        """Trim video between start and end timestamps (seconds)."""
        cmd = [
            self.ffmpeg, '-y',
            '-i', input_path,
            '-ss', str(start),
            '-to', str(end),
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-c:a', 'aac', '-b:a', '192k',
            '-pix_fmt', 'yuv420p',
            output_path
        ]
        subprocess.run(cmd, check=True)
        return output_path

    def stitch(self, clips: List[str], output_path: str,
               transition: str = "none",
               transition_duration: float = 0.5) -> str:
        """Concatenate multiple clips with optional transitions."""
        if transition == "none":
            # Fast concat without re-encoding
            list_file = "/tmp/concat_list.txt"
            with open(list_file, 'w') as f:
                for clip in clips:
                    f.write(f"file '{clip}'\n")
            cmd = [
                self.ffmpeg, '-y',
                '-f', 'concat', '-safe', '0',
                '-i', list_file,
                '-c', 'copy',
                output_path
            ]
        else:
            # Build xfade filter chain for N clips
            filter_parts = []
            for i in range(len(clips) - 1):
                if i == 0:
                    src_a = f"[{i}:v]"
                else:
                    src_a = f"[v{i}]"

                src_b = f"[{i+1}:v]"
                out = f"[v{i+1}]" if i < len(clips) - 2 else "[outv]"

                # Calculate offset (sum of all previous clip durations
                # minus accumulated transition overlaps)
                offset = sum(self._get_duration(clips[j])
                             for j in range(i + 1)) \
                         - transition_duration * (i + 1)

                filter_parts.append(
                    f"{src_a}{src_b}xfade=transition={transition}:"
                    f"duration={transition_duration}:"
                    f"offset={offset:.3f}{out}"
                )

            inputs = []
            for clip in clips:
                inputs.extend(['-i', clip])

            cmd = [
                self.ffmpeg, '-y',
                *inputs,
                '-filter_complex', ';'.join(filter_parts),
                '-map', '[outv]',
                '-c:v', 'libx264', '-crf', '18',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
        subprocess.run(cmd, check=True)
        return output_path

    def add_audio(self, video_path: str, audio_path: str,
                  output_path: str, volume: float = 1.0,
                  mix_with_original: bool = False) -> str:
        """Add audio track to video."""
        if mix_with_original:
            cmd = [
                self.ffmpeg, '-y',
                '-i', video_path,
                '-i', audio_path,
                '-filter_complex',
                f'[1:a]volume={volume}[new];'
                f'[0:a][new]amix=inputs=2:duration=first'
                f':dropout_transition=3[outa]',
                '-map', '0:v', '-map', '[outa]',
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k',
                output_path
            ]
        else:
            cmd = [
                self.ffmpeg, '-y',
                '-i', video_path,
                '-i', audio_path,
                '-map', '0:v', '-map', '1:a',
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k',
                '-shortest',
                output_path
            ]
        subprocess.run(cmd, check=True)
        return output_path

    def color_grade(self, input_path: str, output_path: str,
                    grade: ColorGrade) -> str:
        """Apply color grading adjustments."""
        filters = []

        # Brightness, contrast, saturation via eq filter
        filters.append(
            f"eq=brightness={grade.brightness}"
            f":contrast={grade.contrast}"
            f":saturation={grade.saturation}"
        )

        # Temperature via colorbalance
        if grade.temperature != 0.0:
            r_shift = grade.temperature * 0.15
            b_shift = -grade.temperature * 0.15
            filters.append(
                f"colorbalance=rs={r_shift}:bs={b_shift}"
                f":rm={r_shift * 0.5}:bm={b_shift * 0.5}"
            )

        cmd = [
            self.ffmpeg, '-y',
            '-i', input_path,
            '-vf', ','.join(filters),
            '-c:v', 'libx264', '-crf', '18',
            '-c:a', 'copy',
            output_path
        ]
        subprocess.run(cmd, check=True)
        return output_path

    def change_speed(self, input_path: str, output_path: str,
                     speed: float) -> str:
        """Change playback speed (0.25 to 4.0)."""
        video_pts = 1.0 / speed

        # atempo only accepts 0.5-2.0, chain for extremes
        audio_filters = []
        remaining = speed
        while remaining > 2.0:
            audio_filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            audio_filters.append("atempo=0.5")
            remaining /= 0.5
        audio_filters.append(f"atempo={remaining}")

        cmd = [
            self.ffmpeg, '-y',
            '-i', input_path,
            '-filter_complex',
            f'[0:v]setpts={video_pts}*PTS[v];'
            f'[0:a]{",".join(audio_filters)}[a]',
            '-map', '[v]', '-map', '[a]',
            '-c:v', 'libx264', '-crf', '18',
            output_path
        ]
        subprocess.run(cmd, check=True)
        return output_path

    def _get_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        cmd = [
            self.ffprobe, '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return float(result.stdout.strip())
```

---

## Smart Aspect Ratio Adaptation

Social media platforms demand different aspect ratios. A single generation might need to be delivered as:

| Platform | Aspect Ratio | Resolution |
|----------|-------------|------------|
| YouTube | 16:9 | 1920x1080 |
| TikTok / Reels / Shorts | 9:16 | 1080x1920 |
| Instagram Feed | 1:1 | 1080x1080 |
| Twitter / X | 16:9 or 1:1 | 1280x720 or 720x720 |
| LinkedIn | 1:1 or 16:9 | 1080x1080 or 1920x1080 |

Naive center-crop wastes content. Adobe's approach uses AI-powered subject tracking to intelligently position the crop window. Here is how to build this.

### Architecture

```
                     Input Video (16:9)
                            |
                    +-------v--------+
                    | Subject        |
                    | Detection      |
                    | (per-frame)    |
                    +-------+--------+
                            |
                    [bounding boxes per frame]
                            |
                    +-------v--------+
                    | Trajectory     |
                    | Smoothing      |
                    | (Kalman filter)|
                    +-------+--------+
                            |
                    [smooth crop center per frame]
                            |
                    +-------v--------+
                    | Dynamic Crop   |
                    | (FFmpeg)       |
                    +-------+--------+
                            |
                     Output Video (9:16)
```

### Step 1: Subject Detection

We use a lightweight object detection model (YOLO or MediaPipe) to find the primary subject in each frame.

```python
from ultralytics import YOLO
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class BoundingBox:
    x_center: float  # normalized 0-1
    y_center: float  # normalized 0-1
    width: float     # normalized 0-1
    height: float    # normalized 0-1
    confidence: float
    class_id: int


def detect_subjects(
    video_path: str,
    model_path: str = "yolov8n.pt",
    target_classes: Optional[List[int]] = None  # e.g., [0] for persons
) -> List[Optional[BoundingBox]]:
    """
    Detect the primary subject in each frame of a video.

    Returns a list of BoundingBox objects (one per frame),
    or None for frames where no subject is detected.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        best_detection = None
        best_confidence = 0.0

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if target_classes and class_id not in target_classes:
                continue

            if confidence > best_confidence:
                best_confidence = confidence
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                h, w = frame.shape[:2]
                best_detection = BoundingBox(
                    x_center=(x1 + x2) / 2 / w,
                    y_center=(y1 + y2) / 2 / h,
                    width=(x2 - x1) / w,
                    height=(y2 - y1) / h,
                    confidence=confidence,
                    class_id=class_id
                )

        detections.append(best_detection)

    cap.release()
    return detections
```

### Step 2: Trajectory Smoothing with Kalman Filter

Raw detection bounding boxes jitter frame-to-frame. A Kalman filter smooths the crop center trajectory, producing fluid camera-like motion instead of jerky snapping.

```python
class CropTrajectorySmoothor:
    """
    Smooth the crop center trajectory using a Kalman filter.

    State: [x, y, vx, vy] (position and velocity of crop center)
    """

    def __init__(self, process_noise: float = 0.001,
                 measurement_noise: float = 0.05):
        # State: [x, y, vx, vy]
        self.state = np.array([0.5, 0.5, 0.0, 0.0])

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # Measurement matrix (we observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        # Covariance matrix
        self.P = np.eye(4) * 0.1

        # Process noise
        self.Q = np.eye(4) * process_noise

        # Measurement noise
        self.R = np.eye(2) * measurement_noise

    def update(self, measurement: Optional[Tuple[float, float]]
               ) -> Tuple[float, float]:
        """
        Update with a new measurement (or None if no detection).
        Returns smoothed (x, y) crop center.
        """
        # Predict
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        if measurement is not None:
            # Update
            z = np.array(measurement)
            y = z - self.H @ self.state  # innovation
            S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
            K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
            self.state = self.state + K @ y
            self.P = (np.eye(4) - K @ self.H) @ self.P

        # Clamp to valid range
        x = np.clip(self.state[0], 0.0, 1.0)
        y = np.clip(self.state[1], 0.0, 1.0)

        return (x, y)


def smooth_detections(
    detections: List[Optional[BoundingBox]],
    process_noise: float = 0.001,
    measurement_noise: float = 0.05
) -> List[Tuple[float, float]]:
    """
    Convert raw detections into a smooth crop center trajectory.
    """
    smoother = CropTrajectorySmoothor(process_noise, measurement_noise)
    trajectory = []

    for det in detections:
        if det is not None:
            measurement = (det.x_center, det.y_center)
        else:
            measurement = None

        smooth_center = smoother.update(measurement)
        trajectory.append(smooth_center)

    return trajectory
```

### Step 3: Dynamic Crop with FFmpeg

```python
def apply_dynamic_crop(
    video_path: str,
    output_path: str,
    trajectory: List[Tuple[float, float]],
    source_width: int,
    source_height: int,
    target_aspect: Tuple[int, int] = (9, 16),
    output_resolution: Tuple[int, int] = (1080, 1920)
) -> str:
    """
    Apply a dynamic crop following the smoothed trajectory.

    Uses FFmpeg's crop filter with per-frame expressions, or
    generates a cropdetect script for complex trajectories.
    """
    target_w_ratio, target_h_ratio = target_aspect
    out_w, out_h = output_resolution

    # Calculate crop dimensions in source pixels
    # The crop window must maintain target aspect ratio
    # and fit within the source frame
    crop_h = source_height
    crop_w = int(crop_h * target_w_ratio / target_h_ratio)

    if crop_w > source_width:
        crop_w = source_width
        crop_h = int(crop_w * target_h_ratio / target_w_ratio)

    # Generate per-frame crop positions
    # Write as FFmpeg sendcmd script
    script_path = "/tmp/crop_script.txt"
    fps = 30  # adjust based on actual video fps

    with open(script_path, 'w') as f:
        for i, (cx, cy) in enumerate(trajectory):
            t = i / fps

            # Convert normalized center to pixel crop position
            crop_x = int(cx * source_width - crop_w / 2)
            crop_y = int(cy * source_height - crop_h / 2)

            # Clamp to valid range
            crop_x = max(0, min(crop_x, source_width - crop_w))
            crop_y = max(0, min(crop_y, source_height - crop_h))

            f.write(f"{t:.4f} crop x {crop_x};\n")
            f.write(f"{t:.4f} crop y {crop_y};\n")

    # Apply dynamic crop via sendcmd
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-filter_complex',
        f'sendcmd=f={script_path},'
        f'crop=w={crop_w}:h={crop_h}:x=0:y=0,'
        f'scale={out_w}:{out_h}:flags=lanczos',
        '-c:v', 'libx264', '-crf', '18',
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path
```

### Complete Smart Reframe Pipeline

```python
async def smart_reframe(
    video_path: str,
    output_path: str,
    target_aspect: Tuple[int, int] = (9, 16),
    output_resolution: Tuple[int, int] = (1080, 1920),
    subject_classes: Optional[List[int]] = None
) -> str:
    """
    Intelligently reframe a video for a different aspect ratio
    by tracking the primary subject.

    Usage:
        # Convert 16:9 YouTube video to 9:16 TikTok
        await smart_reframe("youtube.mp4", "tiktok.mp4", (9, 16))

        # Convert to 1:1 Instagram square
        await smart_reframe("youtube.mp4", "insta.mp4", (1, 1), (1080, 1080))
    """
    # Get source video info
    info = VideoEditor().get_info(video_path)
    stream = next(s for s in info['streams'] if s['codec_type'] == 'video')
    src_w = int(stream['width'])
    src_h = int(stream['height'])

    # Step 1: Detect subjects
    detections = detect_subjects(
        video_path,
        target_classes=subject_classes or [0]  # default: persons
    )

    # Step 2: Smooth trajectory
    trajectory = smooth_detections(detections)

    # Step 3: Apply dynamic crop
    return apply_dynamic_crop(
        video_path, output_path, trajectory,
        src_w, src_h, target_aspect, output_resolution
    )
```

---

## Auto-Subtitles with Whisper

Subtitles are one of the highest-ROI post-processing features. They increase watch time by 12-25% on social platforms (Facebook's internal data, reported in 2024), and they are required for accessibility compliance in many markets.

### Integration Architecture

```
Input Video
    |
    v
+---+---+
| FFmpeg |--- Extract audio track ---> audio.wav (16kHz, mono)
+---+---+
    |
    v
+---+---+
| Whisper|--- Transcribe with timestamps ---> segments[]
+---+---+
    |
    v
+---+---+
| Format |--- Convert to SRT/ASS ---> subtitles.srt
+---+---+
    |
    v
+---+---+
| FFmpeg |--- Burn in or attach ---> output_with_subs.mp4
+---+---+
```

### Step 1: Extract Audio

```bash
# Extract audio as 16kHz mono WAV (Whisper's expected format)
ffmpeg -y \
  -i input_video.mp4 \
  -vn \
  -acodec pcm_s16le \
  -ar 16000 \
  -ac 1 \
  /tmp/audio_for_whisper.wav
```

### Step 2: Transcribe with Whisper

```python
import whisper
from typing import List
from dataclasses import dataclass


@dataclass
class SubtitleSegment:
    index: int
    start: float  # seconds
    end: float    # seconds
    text: str


def transcribe_video(
    video_path: str,
    model_size: str = "medium",  # tiny, base, small, medium, large-v3
    language: str = None  # auto-detect if None
) -> List[SubtitleSegment]:
    """
    Transcribe video audio using OpenAI Whisper.

    Model sizes and their characteristics:
    +----------+--------+--------+-----------+
    | Model    | Params | VRAM   | Speed     |
    +----------+--------+--------+-----------+
    | tiny     | 39M    | ~1 GB  | ~32x RT   |
    | base     | 74M    | ~1 GB  | ~16x RT   |
    | small    | 244M   | ~2 GB  | ~6x RT    |
    | medium   | 769M   | ~5 GB  | ~2x RT    |
    | large-v3 | 1550M  | ~10 GB | ~1x RT    |
    +----------+--------+--------+-----------+
    RT = Real-time (e.g., 32x means 32 seconds of audio per second)
    """
    # Extract audio
    audio_path = "/tmp/whisper_audio.wav"
    subprocess.run([
        'ffmpeg', '-y',
        '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        audio_path
    ], check=True, capture_output=True)

    # Load model
    model = whisper.load_model(model_size)

    # Transcribe
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,  # Get word-level timing
        task="transcribe"
    )

    # Convert to segments
    segments = []
    for i, seg in enumerate(result["segments"]):
        segments.append(SubtitleSegment(
            index=i + 1,
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip()
        ))

    return segments
```

### Step 3: Generate SRT and ASS Files

```python
def to_srt(segments: List[SubtitleSegment]) -> str:
    """Convert segments to SRT format."""
    lines = []
    for seg in segments:
        start = _format_timestamp_srt(seg.start)
        end = _format_timestamp_srt(seg.end)
        lines.append(f"{seg.index}")
        lines.append(f"{start} --> {end}")
        lines.append(seg.text)
        lines.append("")  # blank line separator
    return "\n".join(lines)


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_ass(
    segments: List[SubtitleSegment],
    style: str = "modern",
    video_width: int = 1080,
    video_height: int = 1920
) -> str:
    """
    Convert segments to ASS (Advanced SubStation Alpha) format
    with styled subtitles.

    Styles:
    - "modern": Bold white text with rounded background box (TikTok-style)
    - "classic": White text with black outline (YouTube-style)
    - "minimal": Small white text, bottom of screen
    """
    styles = {
        "modern": (
            "Style: Default,Arial,58,&H00FFFFFF,&H00FFFFFF,"
            "&H00000000,&H80000000,1,0,0,0,100,100,0,0,3,0,8,"
            "2,30,30,60,1"
        ),
        "classic": (
            "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,"
            "&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,"
            "2,20,20,40,1"
        ),
        "minimal": (
            "Style: Default,Arial,36,&H00FFFFFF,&H000000FF,"
            "&H00000000,&H40000000,-1,0,0,0,100,100,0,0,1,2,0,"
            "2,15,15,30,1"
        )
    }

    header = f"""[Script Info]
Title: Auto-generated subtitles
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{styles.get(style, styles["modern"])}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

    events = []
    for seg in segments:
        start = _format_timestamp_ass(seg.start)
        end = _format_timestamp_ass(seg.end)
        # Word-by-word highlight effect for modern style
        text = seg.text.replace("\n", "\\N")
        events.append(
            f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
        )

    return header + "\n" + "\n".join(events) + "\n"


def _format_timestamp_ass(seconds: float) -> str:
    """Format seconds as ASS timestamp (H:MM:SS.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"
```

### Step 4: Burn Subtitles into Video

```bash
# Soft subtitles (separate track, toggleable by player)
ffmpeg -y \
  -i input.mp4 \
  -i subtitles.srt \
  -c:v copy -c:a copy \
  -c:s mov_text \
  -metadata:s:s:0 language=eng \
  output_soft_subs.mp4

# Hard subtitles (burned into video pixels, always visible)
# Using SRT
ffmpeg -y \
  -i input.mp4 \
  -vf "subtitles=subtitles.srt:force_style='FontSize=24,FontName=Arial,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1'" \
  -c:v libx264 -crf 18 \
  -c:a copy \
  output_hard_subs.mp4

# Hard subtitles using ASS (preserves all styling)
ffmpeg -y \
  -i input.mp4 \
  -vf "ass=subtitles.ass" \
  -c:v libx264 -crf 18 \
  -c:a copy \
  output_styled_subs.mp4
```

### Cost Analysis: Self-Hosted Whisper vs. API Services

| Approach | Cost per Minute of Audio | Latency | Quality (WER) |
|----------|------------------------|---------|---------------|
| Whisper large-v3 (self-hosted, A10G) | ~$0.003 | ~60s/min | 4.2% |
| Whisper medium (self-hosted, T4) | ~$0.001 | ~30s/min | 5.7% |
| Whisper large-v3 (Replicate API) | ~$0.006 | ~45s/min | 4.2% |
| Deepgram Nova-2 | $0.0043 | ~5s/min | 5.0% |
| AssemblyAI Best | $0.0065 | ~15s/min | 4.5% |
| Google Speech-to-Text v2 | $0.006 | ~10s/min | 5.5% |

**Recommendation**: For a SaaS platform processing thousands of videos per day, self-host Whisper medium on a GPU instance. At $0.001/minute, the cost for a 30-second AI-generated video is $0.0005 -- effectively free. Upgrade to large-v3 if you need to support non-English languages (large-v3's multilingual performance is significantly better).

For low volume (under 100 videos/day), use Deepgram's API. The integration is simpler, latency is lower, and the cost difference is negligible at low volume.

---

## Build vs. Buy Analysis

For every post-processing feature, here is the detailed comparison. The key variables: development time, ongoing maintenance cost, per-unit processing cost, and quality.

### The Full Comparison Matrix

| Feature | Build | Buy | Build Cost (Dev Hours) | Build Cost (Per Unit) | Buy Cost (Per Unit) | Quality Delta | Recommendation |
|---------|-------|-----|----------------------|---------------------|--------------------|--------------|----|
| **Trim/Cut** | FFmpeg | N/A | 8-16 hours | ~$0.00 | N/A | Equal | Build |
| **Stitch/Concat** | FFmpeg | N/A | 16-24 hours | ~$0.00 | N/A | Equal | Build |
| **Audio Overlay** | FFmpeg | N/A | 16-24 hours | ~$0.00 | N/A | Equal | Build |
| **Color Grading** | FFmpeg filters | Cloudinary | 24-40 hours | ~$0.00 | $0.02-0.05 | Build: good, Buy: better | Build (basic), Buy (advanced) |
| **Speed Change** | FFmpeg | N/A | 8-16 hours | ~$0.00 | N/A | Equal | Build |
| **Text Overlay** | FFmpeg drawtext | Cloudinary | 24-40 hours | ~$0.00 | $0.02-0.05 | Buy slightly better | Build |
| **Subtitles** | Whisper self-host | Deepgram/AssemblyAI | 40-60 hours | $0.001/min | $0.004-0.007/min | Comparable | Build at scale, Buy at low volume |
| **Smart Reframe** | YOLO + Kalman + FFmpeg | Cloudinary AI Crop | 80-120 hours | ~$0.01 | $0.05-0.10 | Comparable | Build |
| **Duration Extend** | I2V chaining | No pure API option | 60-80 hours | $0.08-0.40 (model API) | N/A | Build is only option | Build |
| **Upscaling (4K)** | Real-ESRGAN / RTX Video | Topaz API | 40-60 hours | ~$0.02-0.05 | $0.10-0.25 | Buy slightly better | Build |
| **Background Removal** | rembg (open source) | Remove.bg API | 24-40 hours | ~$0.001 | $0.05-0.20 | Buy noticeably better | Buy for quality, Build for cost |
| **Music Generation** | MusicGen (Meta) | ElevenLabs / Suno API | 80-120 hours | ~$0.01 | $0.05-0.15 | Buy significantly better | Buy |
| **Voice Generation** | Coqui TTS (open source) | ElevenLabs API | 60-100 hours | ~$0.005 | $0.03-0.10 | Buy significantly better | Buy |
| **Motion Transfer** | Not practical | Adobe API (when avail.) | 200+ hours | N/A | TBD | Only buy option | Wait for API |

### Cost at Scale: 10,000 Videos/Day

Let us calculate the monthly cost for a platform processing 10,000 videos per day, where each video is 10 seconds long and requires trim + subtitle + smart reframe + audio overlay.

**Build everything approach:**

| Component | Monthly Cost |
|-----------|-------------|
| FFmpeg processing (2x c6g.xlarge EC2) | $245/mo |
| Whisper GPU (1x g5.xlarge, shared) | $1,200/mo |
| YOLO inference (shared with Whisper GPU) | Included |
| Storage (R2, 5TB/mo throughput) | $75/mo |
| Development amortized (600 hours / 12 months) | $12,500/mo |
| **Total** | **$14,020/mo** |
| **Per video** | **$0.047** |

**Buy everything approach:**

| Component | Monthly Cost |
|-----------|-------------|
| Deepgram subtitles (10K * 10s * 30 days) | $215/mo |
| Cloudinary reframe | $15,000/mo |
| Cloudinary color + overlay | $6,000/mo |
| Storage | $75/mo |
| **Total** | **$21,290/mo** |
| **Per video** | **$0.071** |

**Hybrid approach (recommended):**

| Component | Monthly Cost |
|-----------|-------------|
| FFmpeg processing (build: trim, stitch, audio, color, text) | $245/mo |
| Whisper self-hosted (build: subtitles) | $1,200/mo |
| YOLO + Kalman reframe (build) | Included |
| ElevenLabs (buy: voice, music) | $330/mo |
| Development amortized (400 hours / 12 months) | $8,333/mo |
| **Total** | **$10,108/mo** |
| **Per video** | **$0.034** |

The hybrid approach saves 28% over pure build (less dev time) and 52% over pure buy (less API cost). The rule of thumb: **build for commodity operations (trim, stitch, color), buy for generative operations (voice, music, extend).**

---

## The Editing UX Challenge

The hardest part of post-production features is not the processing -- it is the user interface. A timeline editor in a web app is a significant engineering undertaking.

### Canvas-Based vs. DOM-Based Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Canvas-based** | Smooth scrubbing, pixel-perfect rendering, good performance with many tracks | Complex hit-testing, no native accessibility, custom text rendering |
| **DOM-based** | Native events, accessibility built-in, easier to build initially | Performance degrades with many elements, janky scrubbing, difficult zoom/scroll sync |
| **Hybrid** | Canvas for timeline + waveforms, DOM for controls + overlays | Most complex to build, but best UX |

**Recommendation**: Hybrid. Use HTML Canvas for the timeline visualization (where performance matters) and DOM elements for buttons, menus, property panels (where accessibility matters).

### Timeline Architecture

```
+------------------------------------------------------------------+
|  Toolbar: [Trim] [Split] [Delete] [Undo] [Redo] [Export]        |
+------------------------------------------------------------------+
|  +-------------------------------------------------------------+ |
|  | Track 1 (Video)  [====CLIP A====][==CLIP B==][===CLIP C===] | |
|  +-------------------------------------------------------------+ |
|  | Track 2 (Audio)  [========MUSIC TRACK================]      | |
|  +-------------------------------------------------------------+ |
|  | Track 3 (Text)      [TITLE]           [CAPTION]            | |
|  +-------------------------------------------------------------+ |
|  | Track 4 (Effects)          [FADE]    [COLOR GRADE]          | |
|  +-------------------------------------------------------------+ |
|  |  Playhead                                                    | |
|  |     |                                                        | |
|  |  00:00  00:02  00:04  00:06  00:08  00:10  00:12  00:14    | |
+------------------------------------------------------------------+
|  Preview Window                                                   |
|  +---------------------------+  Properties Panel                  |
|  |                           |  +-----------+                     |
|  |   [VIDEO PREVIEW]         |  | Start: 2.0|                     |
|  |                           |  | End: 7.5  |                     |
|  |       1920 x 1080         |  | Speed: 1x |                     |
|  |                           |  | Volume: 80|                     |
|  +---------------------------+  +-----------+                     |
+------------------------------------------------------------------+
```

### Open-Source Libraries

**Remotion** (React-based video rendering framework):
- Strength: Programmatic video composition with React components. Server-side rendering. Strong TypeScript support.
- Weakness: Not a traditional NLE timeline. Better for programmatic composition than interactive editing.
- Use case: If your "editing" is more about templates and automation than freeform timeline editing, Remotion is excellent.
- License: Business license required for commercial use at scale.

**editly** (Node.js, FFmpeg-based):
- Strength: Declarative video composition from JSON definitions. Handles transitions, text overlays, audio mixing.
- Weakness: No interactive UI -- it is a rendering engine, not an editor.
- Use case: Backend rendering pipeline for composing final outputs from user-defined parameters.

**Fabric.js** (Canvas library):
- Strength: Interactive canvas with object manipulation. Good for building the preview window.
- Weakness: Not timeline-specific. You build the timeline on top of it.

**wavesurfer.js** (Audio waveform visualization):
- Strength: Excellent waveform rendering for audio tracks in the timeline.
- Weakness: Audio only.

**Timeline UI approach (custom, recommended):**

```typescript
// Core data model for a multi-track timeline
interface Project {
  id: string;
  tracks: Track[];
  duration: number; // total project duration in seconds
  fps: number;
}

interface Track {
  id: string;
  type: "video" | "audio" | "text" | "effect";
  clips: Clip[];
  locked: boolean;
  muted: boolean;
  volume: number; // 0-1 for audio tracks
}

interface Clip {
  id: string;
  trackId: string;
  startTime: number;     // position on timeline (seconds)
  duration: number;      // duration on timeline (seconds)
  sourceStart: number;   // trim start within source (seconds)
  sourceDuration: number;// original source duration (seconds)
  source: ClipSource;
  effects: Effect[];
  speed: number;         // playback speed multiplier
}

interface ClipSource {
  type: "generated" | "uploaded" | "audio" | "text";
  url: string;           // R2/S3 URL
  thumbnailUrl: string;  // for timeline preview
  waveformUrl?: string;  // pre-computed waveform data
  originalDuration: number;
  width?: number;
  height?: number;
}

interface Effect {
  type: "color_grade" | "speed" | "fade_in" | "fade_out" | "text_overlay";
  params: Record<string, number | string>;
  startOffset: number;  // relative to clip start
  duration: number;
}

// Timeline rendering engine
class TimelineRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private project: Project;
  private zoom: number = 100; // pixels per second
  private scrollX: number = 0;
  private playheadPosition: number = 0;
  private trackHeight: number = 60;

  constructor(canvas: HTMLCanvasElement, project: Project) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
    this.project = project;
    this.setupEventListeners();
  }

  render() {
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw time ruler
    this.drawTimeRuler();

    // Draw tracks
    this.project.tracks.forEach((track, i) => {
      const y = 30 + i * (this.trackHeight + 4); // 30px for ruler
      this.drawTrack(track, y);
    });

    // Draw playhead
    this.drawPlayhead();
  }

  private drawTrack(track: Track, y: number) {
    const { ctx } = this;

    // Track background
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, y, this.canvas.width, this.trackHeight);

    // Draw clips
    track.clips.forEach(clip => {
      const x = this.timeToPixel(clip.startTime);
      const width = clip.duration * this.zoom;

      // Clip rectangle
      ctx.fillStyle = this.getClipColor(track.type);
      ctx.strokeStyle = "#ffffff33";
      ctx.lineWidth = 1;

      // Rounded rectangle
      this.roundRect(x - this.scrollX, y + 2,
                     width, this.trackHeight - 4, 4);

      // Clip label
      ctx.fillStyle = "#ffffff";
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText(
        clip.source.type,
        x - this.scrollX + 8,
        y + this.trackHeight / 2 + 4,
      );
    });
  }

  private timeToPixel(seconds: number): number {
    return seconds * this.zoom;
  }

  private getClipColor(type: string): string {
    const colors: Record<string, string> = {
      video: "#4a90d9",
      audio: "#50c878",
      text: "#e8a838",
      effect: "#d94a7a",
    };
    return colors[type] || "#888888";
  }

  // ... additional methods for interaction, drag, resize, etc.
}
```

### Export Pipeline

When the user clicks "Export," the project definition is serialized and sent to the backend, where FFmpeg assembles the final video from the project JSON:

```python
def render_project(project: dict, output_path: str) -> str:
    """
    Render a multi-track timeline project to a final video file.

    The project dict matches the TypeScript Project interface above.
    """
    # Build FFmpeg filter graph from project definition
    inputs = []
    filter_parts = []
    current_video = None
    current_audio = None

    for track in project["tracks"]:
        for clip in track["clips"]:
            input_idx = len(inputs)
            inputs.extend(["-i", clip["source"]["url"]])

            if track["type"] == "video":
                # Trim source
                trim_filter = (
                    f"[{input_idx}:v]"
                    f"trim=start={clip['sourceStart']}:"
                    f"duration={clip['duration']},"
                    f"setpts=PTS-STARTPTS"
                )

                # Apply effects
                for effect in clip.get("effects", []):
                    if effect["type"] == "color_grade":
                        p = effect["params"]
                        trim_filter += (
                            f",eq=brightness={p.get('brightness', 0)}"
                            f":contrast={p.get('contrast', 1)}"
                            f":saturation={p.get('saturation', 1)}"
                        )
                    elif effect["type"] == "speed":
                        speed = effect["params"]["speed"]
                        trim_filter += f",setpts={1/speed}*PTS"

                trim_filter += f"[v{input_idx}]"
                filter_parts.append(trim_filter)

    # Concatenate all video clips
    video_labels = [f"[v{i}]" for i in range(len(inputs) // 2)]
    concat_filter = (
        f"{''.join(video_labels)}"
        f"concat=n={len(video_labels)}:v=1:a=0[outv]"
    )
    filter_parts.append(concat_filter)

    # Build and execute FFmpeg command
    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path
```

---

## Market Sizing

### The Post-Production Automation Market

Post-production automation sits at the intersection of two markets:

**1. Video Editing Software Market**

| Year | Market Size | Growth |
|------|------------|--------|
| 2024 | $4.1B | -- |
| 2025 | $4.7B | 14.6% |
| 2026E | $5.5B | 17.0% |
| 2027E | $6.5B | 18.2% |
| 2030E | $10.2B | ~16% CAGR |

Source: Various industry analyst reports. The acceleration from 14.6% to 18.2% growth reflects AI feature adoption driving new user segments into video editing.

**2. AI Video Generation Market**

| Year | Market Size | Growth |
|------|------------|--------|
| 2024 | $0.5B | -- |
| 2025 | $1.8B | 260% |
| 2026E | $5.2B | 189% |
| 2027E | $10.5B | 102% |
| 2030E | $25-30B | ~50% CAGR |

### The Post-Production Automation Subset

Post-production automation is the overlap -- AI-powered tools that automate traditionally manual editing tasks. This includes:

- Auto-subtitling and captioning
- Intelligent reframing / aspect ratio adaptation
- AI-powered color grading and matching
- Automated audio mixing and normalization
- Duration extension and temporal manipulation
- Object removal and background replacement
- Automated transitions and assembly

Estimated market size for AI post-production automation specifically:

$$
\text{Post-Production AI Market} \approx 0.15 \times \text{Video Editing Market} + 0.20 \times \text{AI Video Market}
$$

For 2026:

$$
\approx 0.15 \times \$5.5B + 0.20 \times \$5.2B = \$0.825B + \$1.04B = \$1.865B
$$

### Where the Revenue Opportunity Is

For a SaaS platform builder, the revenue opportunity in post-production is structured as follows:

**Tier 1: Free / Low-Friction (retention features)**
- Trim, stitch, basic speed change
- These features cost nothing to operate (FFmpeg on CPU)
- Revenue model: increased retention = increased generation revenue

**Tier 2: Premium Features ($5-20/month add-on)**
- Auto-subtitles with style templates
- Smart reframing to multiple aspect ratios
- Audio overlay with music library
- Duration extension (1-2 extensions per video)
- Revenue: direct subscription upgrade

**Tier 3: Professional Features ($30-50/month or per-use)**
- Advanced color grading with LUT support
- Unlimited duration extension
- AI voice generation integration
- Multi-track timeline editor
- Batch processing (reframe 50 videos to 3 aspect ratios)
- Revenue: professional/enterprise tier

**Revenue projection for a platform with 10,000 MAU:**

| Tier | Conversion Rate | Users | Monthly Price | Monthly Revenue |
|------|----------------|-------|---------------|----------------|
| Free (generation only) | 100% | 10,000 | $0 (included) | $0 |
| Tier 2 (post-pro basics) | 15% | 1,500 | $10 | $15,000 |
| Tier 3 (pro editing) | 3% | 300 | $35 | $10,500 |
| **Total incremental** | | | | **$25,500/mo** |

At \(25,500/month in incremental revenue from post-production features, with operating costs of ~\)10,000/month (the hybrid approach from earlier), the net margin on post-production is approximately $15,500/month or 61%.

This is why Adobe can afford to bundle these features into Creative Cloud -- the retention and upsell value exceeds the compute cost by an order of magnitude. The same economics apply to any platform with a generation revenue base.

---

## Conclusion

Adobe's January 2026 release draws a clear line: the complete AI video workflow spans generation, post-production, audio, and delivery. No single step is sufficient on its own.

For platform builders, the strategic response is not to panic and build a Premiere Pro clone. It is to systematically add the post-processing features that keep users on-platform and increase per-user revenue:

1. **Start immediately**: Trim, stitch, speed change, audio overlay. These are pure FFmpeg -- under 100 hours of development, zero marginal cost.

2. **Next quarter**: Auto-subtitles (Whisper), smart reframing (YOLO + Kalman), basic color grading. These require a GPU instance but the cost per operation is sub-penny.

3. **Medium term**: Duration extension (I2V chaining), audio generation integration (ElevenLabs), timeline editor (major engineering investment but highest retention impact).

4. **Watch and wait**: Motion transfer, advanced object masking, generative audio effects. Let Adobe and others ship APIs, then integrate.

The build vs. buy framework is straightforward: build commodity operations on FFmpeg, buy generative operations from specialized APIs, and always calculate whether your volume justifies self-hosting.

Post-production automation is not a feature. It is a market. And it is growing faster than the tools market it is disrupting.
