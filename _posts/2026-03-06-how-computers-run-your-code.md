---
layout: post
title: "How Your Computer Actually Runs Your Code: Caches, Memory Layout, and Why O(n) Lies to You"
date: 2026-03-06
category: infra
---

# How Your Computer Actually Runs Your Code: Caches, Memory Layout, and Why O(n) Lies to You

*This is Part 5 of the series on taking vibe-coded AI projects to production. Parts 1--4 covered [performance engineering](/2026/03/02/vibe-code-to-production-performance-engineering.html), [containerization](/2026/03/03/containerizing-deploying-ai-video-platform.html), [load testing](/2026/03/04/load-testing-breaking-video-pipeline.html), and [observability](/2026/03/05/observability-failure-modes-production-ai.html). This begins a new Foundations sub-series --- the systems knowledge behind those practices.*

You have an image processing step in your AI video pipeline. Each generated frame is a 1920×1080 array of pixels. You need to normalize every pixel value, so you write a simple nested loop. The algorithm is \(O(n)\) where \(n\) is the number of pixels --- about 2 million per frame. Straightforward.

Then you notice something strange. When you iterate row-by-row, the loop takes 1.2 seconds. When you iterate column-by-column --- same pixels, same operation, same number of iterations --- it takes 12 seconds. Ten times slower. Your algorithm has not changed. Your Big-O complexity has not changed. The number of operations is identical to the last addition.

You check for bugs. There are no bugs. You swap the loops back. Fast again. Swap them. Slow again. The code is correct both ways. It produces the same output. But one version is an order of magnitude slower than the other, and your computer science education has no explanation for this, because Big-O notation --- the tool you were taught to measure performance --- does not model the machine your code runs on.

This post is about what Big-O leaves out. It is about the actual hardware beneath your abstractions: how data moves between the CPU and memory, why some access patterns are fast and others are catastrophically slow, and why production performance requires thinking about the machine, not just the algorithm.

---

## Table of Contents

1. [The Memory Hierarchy: Why Not All Memory Is Equal](#1-the-memory-hierarchy-why-not-all-memory-is-equal)
2. [Cache Lines: The Unit of Memory Transfer](#2-cache-lines-the-unit-of-memory-transfer)
3. [Memory Access Patterns: Row-Major vs Column-Major](#3-memory-access-patterns-row-major-vs-column-major)
4. [Branch Prediction: When Your CPU Guesses Wrong](#4-branch-prediction-when-your-cpu-guesses-wrong)
5. [Why HashMap Is O(1) But Slow](#5-why-hashmap-is-o1-but-slow)
6. [Data-Oriented Design: Thinking Like Your Hardware](#6-data-oriented-design-thinking-like-your-hardware)
7. [Practical Measurement: How to Profile Memory Access](#7-practical-measurement-how-to-profile-memory-access)
8. [The Performance-Aware Coding Checklist](#8-the-performance-aware-coding-checklist)
9. [Series Navigation](#9-series-navigation)

---

## 1. The Memory Hierarchy: Why Not All Memory Is Equal

Every program you write does two things: it computes (add, multiply, compare) and it moves data (load from memory, store to memory). The fundamental bottleneck in modern computing is not computation --- it is data movement. Your CPU can perform billions of arithmetic operations per second. But it can only fetch data from main memory a few hundred million times per second. That gap --- roughly 100x to 200x --- is the reason your column-major loop is slow, and understanding it is the single most important thing you can learn about real-world performance.

### The Hierarchy

Modern computers do not have a single pool of memory. They have a **hierarchy** of storage, arranged by speed, size, and cost. Each level is faster, smaller, and more expensive than the one below it.

| Level | Typical Size | Typical Latency | Analogy |
|-------|-------------|----------------|---------|
| **CPU Registers** | ~1 KB | ~0.3 ns (1 cycle) | The number you're holding in your hand |
| **L1 Cache** | 32--64 KB per core | ~1 ns (3--4 cycles) | The notebook on your desk |
| **L2 Cache** | 256 KB--1 MB per core | ~3--10 ns (10--30 cycles) | The filing cabinet in your office |
| **L3 Cache** | 4--64 MB shared | ~10--40 ns (30--120 cycles) | The storage room down the hall |
| **Main Memory (RAM)** | 8--256 GB | ~50--100 ns (150--300 cycles) | The warehouse across town |
| **SSD** | 256 GB--4 TB | ~10,000--100,000 ns | The distribution center in another city |
| **Network** | Unlimited | ~500,000+ ns | Another country |

Look at those latency numbers. L1 cache is ~1 nanosecond. Main memory is ~100 nanoseconds. That is a **100x** difference. If L1 cache access were 1 second (reaching for the notebook on your desk), then a main memory access would be **almost 2 minutes** (driving across town to the warehouse). Every time your program needs data that is not in cache, the CPU sits idle for those 2 metaphorical minutes, doing absolutely nothing, waiting for data to arrive.

This is called a **cache miss**, and it is the single largest source of unexpected performance problems in real software.

<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <text x="400" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">The Memory Hierarchy</text>
  <text x="400" y="50" text-anchor="middle" font-size="12" fill="#888">Faster and smaller at the top, slower and larger at the bottom</text>

  <!-- Pyramid layers -->
  <polygon points="400,80 340,150 460,150" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="400" y="125" text-anchor="middle" font-size="11" font-weight="bold" fill="#1976d2">Registers</text>
  <text x="400" y="140" text-anchor="middle" font-size="9" fill="#555">~1 KB | 0.3 ns</text>

  <polygon points="340,155 270,230 530,230 460,155" fill="#bbdefb" stroke="#1976d2" stroke-width="2"/>
  <text x="400" y="195" text-anchor="middle" font-size="12" font-weight="bold" fill="#1976d2">L1 Cache</text>
  <text x="400" y="212" text-anchor="middle" font-size="10" fill="#555">32-64 KB | ~1 ns</text>

  <polygon points="270,235 195,310 605,310 530,235" fill="#90caf9" stroke="#1565c0" stroke-width="2"/>
  <text x="400" y="270" text-anchor="middle" font-size="12" font-weight="bold" fill="#1565c0">L2 Cache</text>
  <text x="400" y="288" text-anchor="middle" font-size="10" fill="#555">256 KB - 1 MB | ~5 ns</text>

  <polygon points="195,315 130,385 670,385 605,315" fill="#64b5f6" stroke="#0d47a1" stroke-width="2"/>
  <text x="400" y="348" text-anchor="middle" font-size="12" font-weight="bold" fill="#fff">L3 Cache</text>
  <text x="400" y="366" text-anchor="middle" font-size="10" fill="#e3f2fd">4-64 MB shared | ~30 ns</text>

  <polygon points="130,390 60,465 740,465 670,390" fill="#42a5f5" stroke="#0d47a1" stroke-width="2"/>
  <text x="400" y="425" text-anchor="middle" font-size="13" font-weight="bold" fill="#fff">Main Memory (RAM)</text>
  <text x="400" y="445" text-anchor="middle" font-size="10" fill="#e3f2fd">8-256 GB | ~100 ns (100x slower than L1)</text>

  <!-- Speed arrow -->
  <line x1="750" y1="90" x2="750" y2="460" stroke="#ef5350" stroke-width="2" marker-end="url(#arr-mem)"/>
  <text x="770" y="200" font-size="11" fill="#ef5350" transform="rotate(90,770,200)">Slower, Larger, Cheaper →</text>

  <defs>
    <marker id="arr-mem" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#ef5350"/>
    </marker>
  </defs>
</svg>

### Why the Gap Exists

The reason for this hierarchy is physics. Fast memory is made from SRAM (static random-access memory) --- six transistors per bit, blazingly fast, but physically large and expensive. Main memory is made from DRAM (dynamic random-access memory) --- one transistor and one capacitor per bit, much denser and cheaper, but slower because the capacitor needs time to charge and discharge, and because DRAM is physically farther from the CPU die.

The cache is the CPU's attempt to bridge this gap. It keeps a small copy of recently-accessed data close to the processor, betting that you will access the same data (or nearby data) again soon. When that bet pays off --- a **cache hit** --- the CPU gets its data in 1--4 nanoseconds. When the bet fails --- a **cache miss** --- the CPU must fetch from main memory and wait 100+ nanoseconds.

The entire game of low-level performance is maximizing cache hits.

---

## 2. Cache Lines: The Unit of Memory Transfer

Here is a fact that changes how you think about memory access: when the CPU fetches data from main memory, **it never fetches just the byte you asked for**. It fetches an entire **cache line** --- a contiguous block of 64 bytes on all modern x86 and ARM processors.

If you access a single `float64` value (8 bytes) at memory address 1000, the CPU will fetch all 64 bytes from address 960 to 1023 (the cache line containing address 1000). Those 64 bytes are loaded into the cache together. If you then access the `float64` at address 1008, it is already in the cache --- free. Address 1016? Also free. All 8 of the `float64` values in that cache line are available at L1 speed after a single memory fetch.

This is called **spatial locality**: if you access one piece of data, you are likely to access nearby data soon. The cache line is the hardware's mechanism for exploiting spatial locality.

### The Consequences

This seemingly minor hardware detail has enormous consequences for how you write code.

**Sequential access is nearly free.** If you iterate through an array of `float64` values from start to end, you pay for one cache miss every 8 elements (64 bytes / 8 bytes per float64). The other 7 elements are free because they were pre-loaded with the cache line. Your effective cache miss rate is 12.5%.

**Random access is catastrophically expensive.** If you jump around in memory --- accessing element 7000, then element 42, then element 999,000, then element 3 --- every single access is a cache miss. Your effective cache miss rate is close to 100%. Each access costs 100 nanoseconds instead of 1 nanosecond.

Let us put numbers on this. Suppose you have an array of 10 million `float64` values (80 MB --- larger than any cache) and you want to sum them all.

**Sequential access:**

$$\text{Cache misses} = \frac{10{,}000{,}000}{8} = 1{,}250{,}000$$

$$\text{Time from cache misses} = 1{,}250{,}000 \times 100\text{ ns} = 125\text{ ms}$$

**Random access (shuffled indices):**

$$\text{Cache misses} \approx 10{,}000{,}000$$

$$\text{Time from cache misses} = 10{,}000{,}000 \times 100\text{ ns} = 1{,}000\text{ ms}$$

Same data. Same operation. Same Big-O complexity: \(O(n)\). But the random access version is **8x slower**, entirely because of cache misses.

<svg viewBox="0 0 860 350" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <text x="430" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Cache Line Loading: Sequential vs Random Access</text>

  <!-- Sequential access -->
  <text x="30" y="70" font-size="13" font-weight="bold" fill="#1976d2">Sequential Access</text>
  <text x="30" y="88" font-size="10" fill="#555">One cache miss loads 8 useful values</text>

  <!-- Cache line box -->
  <rect x="30" y="100" width="70" height="40" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="65" y="124" text-anchor="middle" font-size="9" fill="#1976d2">val[0]</text>
  <rect x="105" y="100" width="70" height="40" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="140" y="124" text-anchor="middle" font-size="9" fill="#1976d2">val[1]</text>
  <rect x="180" y="100" width="70" height="40" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="215" y="124" text-anchor="middle" font-size="9" fill="#1976d2">val[2]</text>
  <rect x="255" y="100" width="70" height="40" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="290" y="124" text-anchor="middle" font-size="9" fill="#1976d2">val[3]</text>
  <rect x="330" y="100" width="70" height="40" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="365" y="124" text-anchor="middle" font-size="9" fill="#1976d2">val[4]</text>
  <rect x="405" y="100" width="70" height="40" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="440" y="124" text-anchor="middle" font-size="9" fill="#1976d2">val[5]</text>
  <rect x="480" y="100" width="70" height="40" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="515" y="124" text-anchor="middle" font-size="9" fill="#1976d2">val[6]</text>
  <rect x="555" y="100" width="70" height="40" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="590" y="124" text-anchor="middle" font-size="9" fill="#1976d2">val[7]</text>

  <rect x="30" y="95" width="595" height="50" rx="6" fill="none" stroke="#81c784" stroke-width="3" stroke-dasharray="8,4"/>
  <text x="640" y="124" font-size="10" font-weight="bold" fill="#81c784">1 cache line</text>
  <text x="640" y="138" font-size="9" fill="#81c784">= 64 bytes</text>
  <text x="640" y="108" font-size="9" fill="#888">1 miss → 8 hits</text>

  <!-- Random access -->
  <text x="30" y="200" font-size="13" font-weight="bold" fill="#ef5350">Random Access</text>
  <text x="30" y="218" font-size="10" fill="#555">Each access likely touches a different cache line</text>

  <rect x="30" y="230" width="70" height="40" rx="4" fill="#ffcdd2" stroke="#ef5350" stroke-width="2"/>
  <text x="65" y="248" text-anchor="middle" font-size="8" fill="#c62828">val[7042]</text>
  <text x="65" y="260" text-anchor="middle" font-size="7" fill="#ef5350">MISS</text>

  <rect x="140" y="230" width="70" height="40" rx="4" fill="#ffcdd2" stroke="#ef5350" stroke-width="2"/>
  <text x="175" y="248" text-anchor="middle" font-size="8" fill="#c62828">val[31]</text>
  <text x="175" y="260" text-anchor="middle" font-size="7" fill="#ef5350">MISS</text>

  <rect x="250" y="230" width="70" height="40" rx="4" fill="#ffcdd2" stroke="#ef5350" stroke-width="2"/>
  <text x="285" y="248" text-anchor="middle" font-size="8" fill="#c62828">val[9999]</text>
  <text x="285" y="260" text-anchor="middle" font-size="7" fill="#ef5350">MISS</text>

  <rect x="360" y="230" width="70" height="40" rx="4" fill="#ffcdd2" stroke="#ef5350" stroke-width="2"/>
  <text x="395" y="248" text-anchor="middle" font-size="8" fill="#c62828">val[503]</text>
  <text x="395" y="260" text-anchor="middle" font-size="7" fill="#ef5350">MISS</text>

  <rect x="470" y="230" width="70" height="40" rx="4" fill="#ffcdd2" stroke="#ef5350" stroke-width="2"/>
  <text x="505" y="248" text-anchor="middle" font-size="8" fill="#c62828">val[4217]</text>
  <text x="505" y="260" text-anchor="middle" font-size="7" fill="#ef5350">MISS</text>

  <text x="580" y="255" font-size="10" font-weight="bold" fill="#ef5350">5 separate cache</text>
  <text x="580" y="269" font-size="10" font-weight="bold" fill="#ef5350">lines loaded</text>
  <text x="580" y="283" font-size="9" fill="#888">5 misses → 0 reuse</text>

  <!-- Summary -->
  <rect x="30" y="300" width="800" height="35" rx="6" fill="#f5f5f5" stroke="#ddd" stroke-width="1"/>
  <text x="430" y="322" text-anchor="middle" font-size="12" fill="#333">Same O(n) complexity. Sequential: <tspan fill="#81c784" font-weight="bold">12.5% miss rate</tspan>. Random: <tspan fill="#ef5350" font-weight="bold">~100% miss rate</tspan>. Up to <tspan font-weight="bold">8x real-world difference</tspan>.</text>
</svg>

### Temporal Locality

There is a second form of locality that caches exploit: **temporal locality**. If you accessed a piece of data recently, you are likely to access it again soon. The cache keeps recently-accessed data around, evicting the oldest unused data when it needs space for new data (typically using a least-recently-used or pseudo-LRU policy).

This is why tight loops over small datasets are fast: the data stays in cache for the entire loop. And it is why processing one large dataset in a single pass can be faster than processing it in multiple passes --- each pass might evict the data that the next pass needs.

---

## 3. Memory Access Patterns: Row-Major vs Column-Major

Now we can explain the mystery from the opening. When you create a 2D array in C, C++, or NumPy (by default), the data is stored in **row-major order**. This means the elements of each row are contiguous in memory.

Consider a 4×4 matrix:

```
Matrix (logical view):        Memory (physical layout):
[ a  b  c  d ]                [a][b][c][d][e][f][g][h][i][j][k][l][m][n][o][p]
[ e  f  g  h ]                 row 0       row 1       row 2       row 3
[ i  j  k  l ]
[ m  n  o  p ]
```

In memory, the elements of row 0 (`a, b, c, d`) are adjacent. The elements of row 1 (`e, f, g, h`) come right after. And so on.

### Row-Major Iteration (Cache-Friendly)

When you iterate row-by-row:

```python
for i in range(rows):
    for j in range(cols):
        process(A[i, j])
```

You access `a, b, c, d, e, f, g, h, ...` --- sequential memory addresses. Every cache line you load is fully utilized. This is the fast path.

### Column-Major Iteration (Cache-Hostile)

When you iterate column-by-column:

```python
for j in range(cols):
    for i in range(rows):
        process(A[i, j])
```

You access `a, e, i, m, b, f, j, n, ...` --- you jump by an entire row width on every access. If the matrix is large (wider than a cache line), every single access is a cache miss.

### The Benchmark

Let us measure this directly:

```python
import numpy as np
import time

N = 8000
A = np.random.rand(N, N)

# Row-major traversal (cache-friendly)
start = time.perf_counter()
row_sum = 0.0
for i in range(N):
    for j in range(N):
        row_sum += A[i, j]
row_time = time.perf_counter() - start

# Column-major traversal (cache-hostile)
start = time.perf_counter()
col_sum = 0.0
for i in range(N):
    for j in range(N):
        col_sum += A[j, i]  # Note: indices swapped
col_time = time.perf_counter() - start

print(f"Row-major:    {row_time:.2f}s")
print(f"Column-major: {col_time:.2f}s")
print(f"Ratio:        {col_time / row_time:.1f}x slower")
```

On a typical machine, you will see something like:

```
Row-major:    8.41s
Column-major: 56.23s
Ratio:        6.7x slower
```

Same data. Same operation. Same number of iterations. Same Big-O complexity. Nearly 7x difference, entirely because of cache behavior.

### Why NumPy's Vectorized Operations Are Fast

This is also why `np.sum(A)` is orders of magnitude faster than the Python loop. NumPy's C implementation iterates through the underlying memory buffer sequentially, respecting the row-major layout. It processes cache-line-sized chunks at a time, and it uses SIMD (Single Instruction, Multiple Data) instructions to process multiple values per clock cycle. The combination of cache-friendly access and SIMD gives you 100--1000x speedup over a Python loop.

```python
start = time.perf_counter()
vec_sum = np.sum(A)
vec_time = time.perf_counter() - start
print(f"np.sum:       {vec_time:.4f}s")  # Typically 0.03-0.05s
```

### The Math: Cache Miss Rate by Stride

For an array of `float64` (8 bytes each) with 64-byte cache lines (8 elements per line):

**Stride-1 access** (row-major iteration): every 8th access is a miss.

$$\text{Miss rate} = \frac{1}{64 / 8} = \frac{1}{8} = 12.5\%$$

**Stride-N access** (column-major iteration on an N-column matrix): if \(N \geq 8\), every access is a miss.

$$\text{Miss rate} = \frac{1}{\min(1, \lfloor 64 / (8 \times N) \rfloor + 1)} \approx 100\% \text{ for } N \geq 8$$

**Stride-K access** (general case): misses whenever the stride crosses a cache line boundary.

$$\text{Miss rate} = \min\left(1,\ \frac{K \times \text{element\_size}}{64}\right)$$

This is the formula that Big-O cannot capture. Two \(O(n)\) algorithms with different strides can have radically different performance because one produces 12.5% cache misses and the other produces 100%.

---

## 4. Branch Prediction: When Your CPU Guesses Wrong

There is a second hardware feature that can make identically-complex code run at very different speeds: the **branch predictor**.

### The Pipeline

Modern CPUs do not execute one instruction at a time. They use a **pipeline**: a series of stages (fetch, decode, execute, write-back) where multiple instructions are in-flight simultaneously. Think of it like an assembly line in a factory --- while one instruction is being executed, the next one is being decoded, and the one after that is being fetched from memory.

This pipeline is typically 15--20 stages deep on modern processors. That means 15--20 instructions are in various stages of completion at any given moment. When everything flows smoothly, the CPU completes one instruction per clock cycle despite each instruction taking 15--20 cycles to complete end-to-end.

### The Problem with Branches

A **branch** is an instruction that changes the flow of execution: an `if` statement, a `while` loop condition, a `for` loop termination check. When the CPU encounters a branch, it has a problem: it does not know which instruction comes next until the branch condition is evaluated. But by the time the condition is evaluated (deep in the pipeline), the CPU has already started fetching and decoding the next 15--20 instructions. If it fetched the wrong ones, it must **flush the pipeline** and start over with the correct path.

A pipeline flush wastes ~15--20 cycles. On a 3 GHz processor, that is about 5--7 nanoseconds. That does not sound like much, but if it happens millions of times in a tight loop, it dominates the runtime.

### The Branch Predictor

To avoid constant pipeline flushes, CPUs contain a **branch predictor** --- specialized hardware that guesses which way a branch will go before the condition is evaluated. Modern branch predictors are remarkably sophisticated. They use pattern recognition (similar to a tiny neural network) to track the history of each branch and predict its future behavior.

If a branch goes the same way 99% of the time (like a loop condition that is true for 999 iterations and false once), the predictor learns this and guesses correctly 99% of the time. The 1% misprediction causes a pipeline flush, but 99% of the time the pipeline flows smoothly.

The predictor fails when branches are **unpredictable** --- when the outcome is essentially random, like a condition that depends on unsorted data.

### The Classic Benchmark

This is the most famous branch prediction benchmark, and it directly illustrates the problem:

```python
import numpy as np
import time

N = 100_000
data = np.random.randint(0, 256, size=N)

# Unsorted: branch is unpredictable
start = time.perf_counter()
unsorted_sum = 0
for val in data:
    if val >= 128:  # ~50% true, ~50% false --- unpredictable
        unsorted_sum += val
unsorted_time = time.perf_counter() - start

# Sorted: branch becomes predictable
data_sorted = np.sort(data)
start = time.perf_counter()
sorted_sum = 0
for val in data_sorted:
    if val >= 128:  # First half: always false. Second half: always true.
        sorted_sum += val
sorted_time = time.perf_counter() - start

print(f"Unsorted: {unsorted_time:.3f}s")
print(f"Sorted:   {sorted_time:.3f}s")
print(f"Ratio:    {unsorted_time / sorted_time:.1f}x")
```

Typical results:

```
Unsorted: 0.041s
Sorted:   0.024s
Ratio:    1.7x
```

The Python overhead masks the effect somewhat (in C, the difference is 3--6x), but it is still clearly measurable. Same data, same algorithm, same result --- just sorted first so the branch predictor can do its job.

### Why This Matters for AI Video Platforms

In an ML inference pipeline, you often have conditional processing: skip frames below a quality threshold, apply different models based on resolution, filter results by confidence score. If your data is unsorted with respect to these conditions, you pay a branch prediction penalty on every decision. Sorting or partitioning your data before conditional processing can give measurable speedups, especially in C/C++ code paths inside libraries like OpenCV, PyTorch, or TensorFlow.

This is also why ML frameworks prefer to process data in sorted batches of similar lengths --- it is not just about padding efficiency, it is about branch prediction.

---

## 5. Why HashMap Is O(1) But Slow

Every data structures course teaches that hash tables provide \(O(1)\) average-case lookup. This is true in the asymptotic sense. But when you compare the actual wall-clock time of a hash table lookup to a linear scan of a small sorted array, the linear scan often wins --- sometimes dramatically.

### What O(1) Actually Means

Big-O notation describes how the number of operations scales with input size. \(O(1)\) means the number of operations does not grow as the data structure gets larger. For a hash table, this is true: you compute a hash (one operation), find a bucket (one operation), and retrieve the value (one or a few operations for collision resolution). The count of operations is constant regardless of whether the table has 10 entries or 10 million.

But \(O(1)\) says nothing about **how long each operation takes**. And each of those constant-number operations involves a memory access that may or may not hit the cache.

### The Memory Access Pattern Problem

A hash table is an array of buckets, where the bucket index is determined by the hash of the key. Hash functions are **designed to scatter keys uniformly** across the bucket array. This is good for avoiding collisions but catastrophic for cache performance.

When you look up key `"user_123"`, the hash might send you to bucket 7,042. Then you look up `"project_456"` and get sent to bucket 31. Then `"generation_789"` goes to bucket 999,003. Each lookup jumps to a completely different location in memory. If the hash table is larger than the cache (which it is for any non-trivial dataset), **every single lookup is a cache miss**.

For a hash table with chaining (linked list per bucket), it is even worse: after finding the bucket, you follow a linked list pointer to a heap-allocated node, which is at yet another random memory location. That is two cache misses per lookup.

### When Linear Scan Wins

Consider looking up a value in a small sorted array using linear scan:

```python
# Hash table lookup: O(1) but cache-hostile
config = {"model": "veo-2", "resolution": "1080p", "fps": 24, "codec": "h264"}
value = config["resolution"]  # Hash → random memory location

# Sorted array lookup: O(n) but cache-friendly
config_list = [("codec", "h264"), ("fps", 24), ("model", "veo-2"), ("resolution", "1080p")]
for key, val in config_list:  # Sequential scan through contiguous memory
    if key == "resolution":
        value = val
        break
```

For a 4-element config, the linear scan touches 4 contiguous memory locations --- all likely in the same cache line (one cache miss total). The hash table computes a hash function and does a random memory lookup (one cache miss, plus the hash computation overhead).

The crossover point --- where hash tables become faster than sorted arrays with binary search --- depends on the data size. For most platforms:

| Data Size | Winner | Why |
|-----------|--------|-----|
| < 16 elements | Linear scan | Everything fits in 1--2 cache lines |
| 16--64 elements | Binary search on sorted array | \(O(\log n)\) with sequential-ish access |
| > 64 elements | Hash table | \(O(1)\) amortizes the cache miss cost |

This is why many high-performance systems use flat, sorted arrays for small lookup tables instead of hash maps. The asymptotic complexity does not matter at small scale --- cache behavior does.

### Implications for Your Code

In an AI video platform, you have many small lookup tables: model configurations, codec settings, resolution presets, user preferences. For these, a sorted array or a small struct is often faster than a `dict` or `HashMap`. For large datasets --- your database of generations, your frame buffer, your feature store --- hash tables win.

The lesson is not "never use hash tables." The lesson is that **Big-O is a model of computation, not a model of hardware**, and at the scale where most lookups happen (small config objects, option maps, feature flags), the hardware model dominates.

---

## 6. Data-Oriented Design: Thinking Like Your Hardware

Everything in this article so far points to one architectural principle: **organize your data for how you access it, not for how you think about it.**

Object-oriented programming teaches you to group related data together. A `VideoGeneration` object has a `status`, `model`, `prompt`, `created_at`, `user_id`, `resolution`, `duration`, `output_url`, and so on. All the fields of one generation live together in one object, allocated as one chunk of memory.

This is called **Array of Structures (AoS)**: you have an array where each element is a structure containing all fields.

```python
# Array of Structures (AoS)
generations = [
    {"id": "gen_001", "status": "completed", "model": "veo-2", "prompt": "...", "user_id": "u_1", ...},
    {"id": "gen_002", "status": "pending",   "model": "kling", "prompt": "...", "user_id": "u_2", ...},
    {"id": "gen_003", "status": "failed",    "model": "veo-2", "prompt": "...", "user_id": "u_1", ...},
    # ... 100,000 more
]
```

Now suppose you want to count how many generations are in "completed" status. You iterate through all 100,000 generations, but for each one you only read the `status` field. In AoS layout, each generation object might be 500 bytes. The `status` field is 10 bytes of those 500. You are loading 500 bytes into cache to read 10 bytes --- a 98% waste of cache bandwidth.

### Structure of Arrays (SoA)

The alternative is **Structure of Arrays (SoA)**: store each field in its own contiguous array.

```python
# Structure of Arrays (SoA)
generation_ids    = ["gen_001", "gen_002", "gen_003", ...]
generation_status = ["completed", "pending", "failed", ...]
generation_models = ["veo-2", "kling", "veo-2", ...]
generation_prompts = ["...", "...", "...", ...]
generation_users  = ["u_1", "u_2", "u_1", ...]
```

Now when you count completed generations, you iterate through `generation_status` only --- a contiguous array of small strings. Every cache line is fully utilized. No wasted bandwidth loading prompt text and user IDs you do not need.

<svg viewBox="0 0 880 420" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <text x="440" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Array of Structures vs Structure of Arrays</text>

  <!-- AoS -->
  <text x="30" y="65" font-size="13" font-weight="bold" fill="#ef5350">Array of Structures (AoS)</text>
  <text x="30" y="82" font-size="10" fill="#555">Query: "count completed" → loads ALL fields per object</text>

  <rect x="30" y="95" width="60" height="30" rx="3" fill="#ffcdd2" stroke="#ef5350" stroke-width="1.5"/>
  <text x="60" y="114" text-anchor="middle" font-size="8" fill="#c62828">status</text>
  <rect x="92" y="95" width="50" height="30" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1"/>
  <text x="117" y="114" text-anchor="middle" font-size="8" fill="#888">model</text>
  <rect x="144" y="95" width="80" height="30" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1"/>
  <text x="184" y="114" text-anchor="middle" font-size="8" fill="#888">prompt</text>
  <rect x="226" y="95" width="50" height="30" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1"/>
  <text x="251" y="114" text-anchor="middle" font-size="8" fill="#888">user</text>
  <rect x="278" y="95" width="40" height="30" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1"/>
  <text x="298" y="114" text-anchor="middle" font-size="7" fill="#888">...</text>
  <text x="340" y="114" font-size="9" fill="#888">← gen_001 (all fields)</text>

  <rect x="30" y="130" width="60" height="30" rx="3" fill="#ffcdd2" stroke="#ef5350" stroke-width="1.5"/>
  <text x="60" y="149" text-anchor="middle" font-size="8" fill="#c62828">status</text>
  <rect x="92" y="130" width="50" height="30" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1"/>
  <text x="117" y="149" text-anchor="middle" font-size="8" fill="#888">model</text>
  <rect x="144" y="130" width="80" height="30" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1"/>
  <text x="184" y="149" text-anchor="middle" font-size="8" fill="#888">prompt</text>
  <rect x="226" y="130" width="50" height="30" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1"/>
  <text x="251" y="149" text-anchor="middle" font-size="8" fill="#888">user</text>
  <rect x="278" y="130" width="40" height="30" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1"/>
  <text x="298" y="149" text-anchor="middle" font-size="7" fill="#888">...</text>
  <text x="340" y="149" font-size="9" fill="#888">← gen_002 (all fields)</text>

  <text x="30" y="190" font-size="10" fill="#ef5350">Cache loads: 500 bytes × N objects. Uses: 10 bytes × N. Waste: 98%</text>

  <!-- SoA -->
  <text x="30" y="240" font-size="13" font-weight="bold" fill="#81c784">Structure of Arrays (SoA)</text>
  <text x="30" y="257" font-size="10" fill="#555">Query: "count completed" → loads ONLY status array</text>

  <rect x="30" y="270" width="80" height="30" rx="3" fill="#e8f5e9" stroke="#81c784" stroke-width="1.5"/>
  <text x="70" y="289" text-anchor="middle" font-size="8" fill="#2e7d32">completed</text>
  <rect x="115" y="270" width="80" height="30" rx="3" fill="#e8f5e9" stroke="#81c784" stroke-width="1.5"/>
  <text x="155" y="289" text-anchor="middle" font-size="8" fill="#2e7d32">pending</text>
  <rect x="200" y="270" width="80" height="30" rx="3" fill="#e8f5e9" stroke="#81c784" stroke-width="1.5"/>
  <text x="240" y="289" text-anchor="middle" font-size="8" fill="#2e7d32">failed</text>
  <rect x="285" y="270" width="80" height="30" rx="3" fill="#e8f5e9" stroke="#81c784" stroke-width="1.5"/>
  <text x="325" y="289" text-anchor="middle" font-size="8" fill="#2e7d32">completed</text>
  <rect x="370" y="270" width="40" height="30" rx="3" fill="#e8f5e9" stroke="#81c784" stroke-width="1.5"/>
  <text x="390" y="289" text-anchor="middle" font-size="7" fill="#2e7d32">...</text>
  <text x="430" y="289" font-size="9" fill="#888">← status array only</text>

  <rect x="30" y="310" width="80" height="25" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1" stroke-dasharray="4,2"/>
  <text x="70" y="327" text-anchor="middle" font-size="8" fill="#aaa">model array (not loaded)</text>
  <rect x="30" y="340" width="80" height="25" rx="3" fill="#f5f5f5" stroke="#bbb" stroke-width="1" stroke-dasharray="4,2"/>
  <text x="70" y="357" text-anchor="middle" font-size="8" fill="#aaa">prompt array (not loaded)</text>

  <text x="30" y="395" font-size="10" fill="#81c784">Cache loads: 10 bytes × N. Uses: 10 bytes × N. Waste: 0%</text>
</svg>

### Where You See SoA in Practice

This is not an academic distinction. SoA is the dominant pattern in high-performance systems:

- **Columnar databases** (BigQuery, ClickHouse, DuckDB, Parquet files): store each column contiguously. Queries that touch few columns are vastly faster.
- **Pandas DataFrames**: internally columnar. `df["status"].value_counts()` is fast because it reads one contiguous array.
- **Entity Component Systems** (Unity DOTS, Bevy): game engines use SoA to iterate over millions of entities per frame.
- **GPU computing**: CUDA and shader programming strongly prefer SoA for coalesced memory access.

In your AI video platform, if you have a dashboard query that counts generations by status, SoA (or a columnar database) will be dramatically faster than iterating through complete generation objects. This is why analytics databases are columnar --- the access pattern is almost always "read a few columns across many rows."

---

## 7. Practical Measurement: How to Profile Memory Access

Knowing the theory is necessary but not sufficient. You need to measure. Here are the tools for profiling cache and memory behavior.

### perf stat: Counting Cache Misses (Linux)

The `perf` tool on Linux gives you direct access to CPU hardware performance counters, including cache miss counts:

```bash
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
    python benchmark.py
```

Output:

```
 Performance counter stats for 'python benchmark.py':

     1,247,832,105      cache-references
       312,458,026      cache-misses              #   25.04% of all cache refs
     8,491,023,847      L1-dcache-loads
     1,623,847,219      L1-dcache-load-misses     #   19.12% of all L1-dcache loads
```

A cache miss rate above 10% is a red flag. Above 25% means your access pattern is fighting the hardware.

### NumPy Stride Inspection

You can directly examine how NumPy arrays are laid out in memory:

```python
import numpy as np

A = np.zeros((1000, 1000), dtype=np.float64)

print(f"Shape:   {A.shape}")
print(f"Strides: {A.strides}")  # (8000, 8) for row-major
print(f"Order:   {'C (row-major)' if A.flags['C_CONTIGUOUS'] else 'F (column-major)'}")
```

The `strides` tuple tells you how many bytes to jump for each dimension. `(8000, 8)` means: to move one row, jump 8000 bytes (1000 elements × 8 bytes each); to move one column, jump 8 bytes (one element). If you iterate with the large stride in the inner loop, you have a cache problem.

### Cachegrind (Valgrind)

For compiled code (C, C++, Rust), Valgrind's cachegrind tool simulates the cache hierarchy and reports exact hit/miss counts per line of code:

```bash
valgrind --tool=cachegrind ./my_program

# Then annotate the results:
cg_annotate cachegrind.out.<pid>
```

This shows you exactly which lines of code are causing cache misses --- invaluable for optimizing hot loops.

### Python memory_profiler

For Python-level memory profiling, `memory_profiler` shows per-line memory usage:

```bash
pip install memory_profiler

python -m memory_profiler my_script.py
```

This is more about allocation than cache behavior, but it helps you find unexpected memory bloat that pushes your working set out of cache.

### The Measurement Workflow

1. **Profile first.** Never optimize based on intuition. Measure where the time actually goes.
2. **Count cache misses**, not just execution time. Two functions with the same runtime might have very different cache behavior --- one might be CPU-bound (fixable with a better algorithm) and the other memory-bound (fixable with a better access pattern).
3. **Check strides** on NumPy arrays before writing loops. If the inner loop has a large stride, swap the loop order.
4. **Benchmark with realistic data sizes.** Cache effects only appear when the data exceeds the cache size. A benchmark on 100 elements will not show cache problems that appear at 1 million elements.

---

## 8. The Performance-Aware Coding Checklist

Use this checklist when writing performance-sensitive code. Not every item applies to every situation, but each is worth considering.

1. **Prefer contiguous arrays over linked structures.** Arrays are cache-friendly. Linked lists, trees, and hash maps scatter data across memory.

2. **Iterate in memory order.** For row-major arrays (C, NumPy default), iterate with the rightmost index changing fastest. For column-major arrays (Fortran, MATLAB), iterate with the leftmost index changing fastest.

3. **Keep hot data small.** If your inner loop only touches a few fields, consider SoA layout so those fields are contiguous. Smaller working sets fit in cache.

4. **Avoid pointer chasing.** Every pointer dereference is a potential cache miss. Prefer flat data structures with indices instead of pointers when performance matters.

5. **Use vectorized operations.** NumPy, pandas, and SIMD instructions process data in cache-friendly chunks. A Python for-loop over array elements is a cache-unfriendly, branch-heavy antipattern.

6. **Batch small operations.** Instead of processing one item at a time through the entire pipeline, process all items through each stage. This keeps each stage's code and data in cache.

7. **Be skeptical of Big-O for small N.** For datasets under ~64 elements, cache effects and constant factors dominate. A linear scan can beat a hash table. A flat array can beat a tree.

8. **Sort or partition before conditional processing.** If you have a branch that depends on data values, sorting the data first helps the branch predictor.

9. **Measure before optimizing.** Use `perf stat`, stride inspection, or cachegrind. Profile on realistic data sizes. Never optimize based on intuition.

10. **Understand your data access patterns.** Before writing a loop, ask: "Am I accessing memory sequentially or randomly? Is my working set smaller than the cache? Am I loading data I don't need?"

---

## 9. Series Navigation

This article is Part 5 of the series on taking vibe-coded AI projects to production.

| Part | Title | Focus |
|------|-------|-------|
| 1 | [Performance Engineering](/2026/03/02/vibe-code-to-production-performance-engineering.html) | Profiling, N+1 queries, caching, async I/O |
| 2 | [Containerizing & Deploying](/2026/03/03/containerizing-deploying-ai-video-platform.html) | Docker, Nginx, TLS, CI/CD |
| 3 | [Load Testing](/2026/03/04/load-testing-breaking-video-pipeline.html) | k6, stress/soak/spike testing, SLOs |
| 4 | [Observability](/2026/03/05/observability-failure-modes-production-ai.html) | Logging, Prometheus, Grafana, OpenTelemetry |
| **5** | **How Your Computer Runs Your Code** (this post) | **CPU caches, memory layout, branch prediction** |
| 6 | [Linux for the 2 AM Incident](/2026/03/07/linux-for-the-2am-incident.html) | Processes, file descriptors, signals, systemd |
| 7 | [Networking from Packet to Page Load](/2026/03/08/networking-from-packet-to-page-load.html) | DNS, TCP, TLS, reverse proxies, firewalls |

Parts 1--4 tell you **what to do** in production. Parts 5--7 explain **why it works** --- the foundational systems knowledge that makes the practices in Parts 1--4 make sense.

---

The column-major loop from the opening was not wrong. It computed the correct answer. It had the correct Big-O complexity. It would pass every unit test. But it was fighting the hardware, and the hardware won by a factor of 7.

Production performance is not just about algorithms. It is about understanding the machine your code runs on --- the cache hierarchy, the memory layout, the branch predictor, the gap between \(O(1)\) in theory and 100 nanoseconds in practice. The most impactful performance optimizations are often not algorithmic improvements. They are access pattern improvements: iterating in the right order, keeping data contiguous, respecting the cache.

Big-O tells you how algorithms scale. The memory hierarchy tells you how fast they actually run. You need both.
