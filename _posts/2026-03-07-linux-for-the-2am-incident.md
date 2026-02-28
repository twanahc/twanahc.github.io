---
layout: post
title: "Linux for the 2 AM Incident: Processes, File Descriptors, Signals, and Systemd"
date: 2026-03-07
category: infra
mathjax: false
---

# Linux for the 2 AM Incident: Processes, File Descriptors, Signals, and Systemd

*This is Part 6 of the series on taking vibe-coded AI projects to production. Parts 1--4 covered [performance engineering](/2026/03/02/vibe-code-to-production-performance-engineering.html), [containerization](/2026/03/03/containerizing-deploying-ai-video-platform.html), [load testing](/2026/03/04/load-testing-breaking-video-pipeline.html), and [observability](/2026/03/05/observability-failure-modes-production-ai.html). Part 5 covered [CPU caches and memory layout](/2026/03/06/how-computers-run-your-code.html). This continues the Foundations sub-series.*

It is 2:17 AM on a Saturday. Your phone buzzes with an alert: the AI video platform's WebSocket error rate has spiked to 40%. Users in the Discord are reporting that the real-time generation progress bar is stuck. You SSH into the VPS, run `htop`, and see something confusing --- CPU usage is 12%, memory usage is 38%. The machine is barely working. Nothing looks wrong.

You check your API logs. The latest entries look normal. You try `curl localhost:3000/health` from the server and it works instantly. The application is running. The server is healthy. But WebSocket connections keep dropping.

After an hour of confused Googling, you discover a command you have never run before: `ls /proc/$(pgrep node)/fd | wc -l`. It returns 63,847. You do not know what that number means, but it is close to 65,536, which you vaguely recognize as a power of two. Another 15 minutes of Googling reveals the answer: your Node.js process has 63,847 open **file descriptors**, and the system limit is 65,536. Every new WebSocket connection requires a new file descriptor. You have been leaking them for three days --- never closing WebSocket connections when clients disconnect --- and you just ran out.

The fix is six lines of code. But finding the problem took 90 minutes because you did not know what a file descriptor was, what a process limit was, or where to look. This post is the knowledge that turns that 90-minute mystery into a 5-minute diagnosis.

---

## Table of Contents

1. [What a Process Actually Is](#1-what-a-process-actually-is)
2. [PID 1 and the Init System](#2-pid-1-and-the-init-system)
3. [Signals: How Processes Communicate](#3-signals-how-processes-communicate)
4. [File Descriptors: The Invisible Resource Limit](#4-file-descriptors-the-invisible-resource-limit)
5. [The /proc Filesystem: Your Window Into the Kernel](#5-the-proc-filesystem-your-window-into-the-kernel)
6. [Systemd: Managing Long-Running Services](#6-systemd-managing-long-running-services)
7. [Journald: Where Your Logs Actually Go](#7-journald-where-your-logs-actually-go)
8. [Putting It All Together: The 2 AM Triage Runbook](#8-putting-it-all-together-the-2-am-triage-runbook)
9. [The 2 AM Incident Checklist](#9-the-2-am-incident-checklist)
10. [Series Navigation](#10-series-navigation)

---

## 1. What a Process Actually Is

When you type `node server.js` and press Enter, the shell does not simply "run your code." It triggers a precise sequence of kernel operations that creates a new **process** --- the fundamental unit of execution in Linux. Understanding what a process is (and is not) is the foundation for everything else in this article.

### The Anatomy of a Process

A **process** is a running instance of a program. It is not the program itself --- the program is a file on disk (`/opt/video-platform/dist/server.js`). The process is what happens when the kernel loads that program into memory and starts executing it. Each process has:

- **A Process ID (PID):** A unique integer assigned by the kernel. The first process on the system gets PID 1. Your Node.js process might get PID 48,271. PIDs are recycled after a process exits.

- **A memory space:** A private, isolated region of memory. Your process cannot see or modify another process's memory (without explicit shared memory mechanisms). This is enforced by the hardware --- the CPU's Memory Management Unit (MMU) translates the process's virtual addresses to physical addresses, and different processes get different translations.

- **A file descriptor table:** A list of all open "files" (regular files, sockets, pipes, devices). More on this in Section 4 --- it is the key to the opening story.

- **An environment:** A set of key-value pairs (environment variables) inherited from the parent process. `NODE_ENV=production`, `PORT=3000`, `DATABASE_URL=postgres://...`.

- **A parent:** Every process (except PID 1) has a parent process that created it. The parent's PID is stored as the PPID (Parent Process ID).

### How Processes Are Created: fork() and exec()

On Linux, new processes are created through a two-step mechanism that is elegant once you understand it, and confusing until you do.

**`fork()`** creates an exact copy of the current process. The new process (the **child**) gets a new PID but is otherwise identical to the parent: same code, same memory contents, same open files, same environment variables. Both the parent and child continue executing from the same point --- the line after the `fork()` call. The only difference is the return value: `fork()` returns 0 to the child and the child's PID to the parent.

**`exec()`** replaces the current process's code and memory with a new program. When the child calls `exec("node", ["server.js"])`, it discards everything about its previous existence and becomes a Node.js process running `server.js`.

So when you type `node server.js` in your shell:

1. The shell (itself a process, say PID 1000) calls `fork()` to create a child (PID 1001).
2. The child (PID 1001) calls `exec("node", ["server.js"])` and becomes your Node.js server.
3. The shell (PID 1000) either waits for the child to finish (foreground) or continues to accept input (background, with `&`).

This is why every process has a parent. The entire system is a tree rooted at PID 1.

### Process States

A process is always in one of several states. You can see the state in `ps aux` or `htop`:

| State | Symbol | Meaning |
|-------|--------|---------|
| **Running** | `R` | Actively executing on a CPU core, or in the queue ready to run |
| **Sleeping (Interruptible)** | `S` | Waiting for something (I/O, a timer, a signal). Will wake up when the event arrives. This is the normal state for a server waiting for incoming connections. |
| **Sleeping (Uninterruptible)** | `D` | Waiting for I/O that cannot be interrupted (typically disk). If you see many processes in `D` state, your disk is likely the bottleneck. |
| **Stopped** | `T` | Paused by a signal (SIGSTOP or SIGTSTP). Resume with `fg` or SIGCONT. |
| **Zombie** | `Z` | Finished executing, but its parent has not yet read its exit status. The process is dead --- it uses no CPU or memory --- but its PID is still reserved in the process table. |

### What Is a Zombie Process?

A zombie is not a bug in the traditional sense. It is a design consequence. When a process exits, the kernel keeps a small entry in the process table (just the PID and exit status) until the parent calls `wait()` to retrieve that exit status. This is the parent's way of learning whether the child succeeded or failed.

If the parent never calls `wait()` --- because it is busy, or because it has a bug --- the child's entry stays in the process table forever. That is a zombie. The zombie uses no CPU and no memory, but it occupies a PID. If enough zombies accumulate, you run out of PIDs and cannot create new processes.

You can spot zombies in `ps aux` by the `Z` state and `<defunct>` in the command column:

```bash
ps aux | grep Z
# USER  PID  %CPU %MEM  VSZ  RSS TTY STAT  COMMAND
# root  4521  0.0  0.0    0    0  ?   Z    [worker] <defunct>
```

The fix is almost always fixing the parent process to properly handle child termination (call `wait()` or handle SIGCHLD).

<svg viewBox="0 0 860 400" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <defs>
    <marker id="arr-ps" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="430" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">Process State Machine</text>

  <!-- Created -->
  <rect x="30" y="160" width="110" height="50" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="85" y="190" text-anchor="middle" font-size="12" font-weight="bold" fill="#1976d2">Created</text>

  <!-- Running -->
  <rect x="220" y="80" width="120" height="50" rx="8" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="280" y="110" text-anchor="middle" font-size="12" font-weight="bold" fill="#388e3c">Running (R)</text>

  <!-- Sleeping -->
  <rect x="220" y="240" width="120" height="50" rx="8" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="280" y="270" text-anchor="middle" font-size="12" font-weight="bold" fill="#f57c00">Sleeping (S)</text>

  <!-- Stopped -->
  <rect x="460" y="160" width="120" height="50" rx="8" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
  <text x="520" y="190" text-anchor="middle" font-size="12" font-weight="bold" fill="#e91e63">Stopped (T)</text>

  <!-- Zombie -->
  <rect x="660" y="80" width="120" height="50" rx="8" fill="#f5f5f5" stroke="#888" stroke-width="2"/>
  <text x="720" y="110" text-anchor="middle" font-size="12" font-weight="bold" fill="#888">Zombie (Z)</text>

  <!-- Terminated -->
  <rect x="660" y="240" width="120" height="50" rx="8" fill="#fff" stroke="#333" stroke-width="2"/>
  <text x="720" y="270" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Terminated</text>

  <!-- Arrows -->
  <line x1="140" y1="180" x2="218" y2="110" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ps)"/>
  <text x="160" y="135" font-size="9" fill="#555">fork+exec</text>

  <line x1="280" y1="130" x2="280" y2="238" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ps)"/>
  <text x="290" y="188" font-size="9" fill="#555">wait for I/O</text>

  <line x1="280" y1="240" x2="280" y2="132" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ps)" transform="translate(30,0)"/>
  <text x="320" y="188" font-size="9" fill="#555">I/O ready</text>

  <line x1="340" y1="105" x2="458" y2="175" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ps)"/>
  <text x="385" y="130" font-size="9" fill="#555">SIGSTOP</text>

  <line x1="460" y1="175" x2="342" y2="105" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ps)" transform="translate(0,15)"/>
  <text x="385" y="165" font-size="9" fill="#555">SIGCONT</text>

  <line x1="340" y1="95" x2="658" y2="95" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ps)"/>
  <text x="500" y="88" font-size="9" fill="#555">exit() (parent hasn't wait()ed)</text>

  <line x1="720" y1="130" x2="720" y2="238" stroke="#333" stroke-width="1.5" marker-end="url(#arr-ps)"/>
  <text x="730" y="188" font-size="9" fill="#555">parent wait()</text>
</svg>

### Viewing Processes

The tools you will use most often:

```bash
# List all processes with details
ps aux

# Interactive process viewer (much better than top)
htop

# Show process tree (parent-child relationships)
pstree -p

# Find a specific process
pgrep -a node     # Find all node processes with command lines
pidof node        # Find PIDs of all node processes
```

In `ps aux` output, the columns you care about:

| Column | Meaning |
|--------|---------|
| `USER` | Who owns the process |
| `PID` | Process ID |
| `%CPU` | CPU usage (100% = one full core) |
| `%MEM` | Percentage of physical memory used |
| `VSZ` | Virtual memory size (allocated, not necessarily used) |
| `RSS` | Resident Set Size (actually in physical memory --- this is the number you care about) |
| `STAT` | State (R, S, D, T, Z) |
| `COMMAND` | The command that started the process |

---

## 2. PID 1 and the Init System

PID 1 is special. It is the first process started by the kernel after boot, and it has unique responsibilities that no other process has. If PID 1 dies, the kernel panics and the system halts. Every other process is, directly or indirectly, a descendant of PID 1.

### What PID 1 Does

PID 1 has two critical responsibilities:

**1. Reaping orphaned child processes.** When a process exits, its parent is expected to call `wait()` to collect its exit status and release its PID. But what if the parent exits before the child? The child becomes an **orphan**. The kernel re-parents orphaned processes to PID 1. PID 1 is then responsible for calling `wait()` on them when they exit, preventing zombies.

**2. Forwarding signals.** When the system sends a signal to PID 1 (like SIGTERM during shutdown), PID 1 is responsible for propagating that signal to its children so they can shut down gracefully.

### Why This Matters for Docker

When you write `CMD ["node", "server.js"]` in your Dockerfile, your Node.js process becomes PID 1 inside the container. This is a problem because Node.js was not designed to be PID 1. It does not reap orphaned processes, and it does not forward signals by default.

The consequences:

- **Zombie accumulation:** If your Node.js app spawns child processes (e.g., `ffmpeg` for video processing) and they finish, their PID entries stay in the process table because Node.js does not call `wait()` on processes it did not explicitly spawn. Over time, you accumulate zombie processes.

- **Graceful shutdown fails:** When `docker stop` sends SIGTERM to PID 1, a normal init system would forward that signal to all child processes. Node.js does not, so child processes keep running until Docker gives up and sends SIGKILL after 10 seconds.

### The Fix: tini or dumb-init

The standard solution is to use a minimal init process that handles PID 1 responsibilities:

```dockerfile
# Install tini
RUN apt-get update && apt-get install -y tini

# Use tini as PID 1, which will run and manage node
ENTRYPOINT ["tini", "--"]
CMD ["node", "server.js"]
```

Now `tini` is PID 1. It reaps orphans and forwards signals. Your Node.js process runs as PID 2 (or similar) and gets SIGTERM properly forwarded when `docker stop` is called.

Alternatively, Docker has a built-in init flag:

```bash
docker run --init your-image
```

This uses Docker's built-in `tini` equivalent. For `docker-compose`:

```yaml
services:
  api:
    image: video-platform
    init: true
```

---

## 3. Signals: How Processes Communicate

A **signal** is an asynchronous notification sent to a process. It is the kernel's way of telling a process that something happened --- the user pressed Ctrl+C, the system is shutting down, a child process exited, or another process explicitly sent a signal. Signals are software interrupts: they interrupt whatever the process is currently doing and invoke a signal handler.

### The Signals That Matter

There are 31 standard signals in Linux. Most of them are obscure. These are the ones you will encounter in production:

| Signal | Number | Default Action | Can Be Caught? | When It's Sent |
|--------|--------|---------------|---------------|----------------|
| **SIGTERM** | 15 | Terminate | Yes | `kill <PID>`, `docker stop`, systemd stop, deployment |
| **SIGKILL** | 9 | Terminate | **No** | `kill -9 <PID>`, Docker after 10s timeout, OOM killer |
| **SIGINT** | 2 | Terminate | Yes | User presses Ctrl+C |
| **SIGHUP** | 1 | Terminate | Yes | Terminal closes, SSH disconnects |
| **SIGCHLD** | 17 | Ignore | Yes | A child process exits |
| **SIGUSR1** | 10 | Terminate | Yes | User-defined (often: dump debug info) |
| **SIGUSR2** | 12 | Terminate | Yes | User-defined (often: reopen log files) |
| **SIGSTOP** | 19 | Stop | **No** | `kill -STOP`, Ctrl+Z sends SIGTSTP (similar) |
| **SIGCONT** | 18 | Continue | Yes | `fg`, `kill -CONT` |

### SIGTERM vs SIGKILL: The Critical Difference

This is the single most important distinction in process management.

**SIGTERM** (signal 15) says: "Please shut down." The process can catch this signal and run cleanup code: close database connections, flush write buffers, finish in-progress requests, deregister from service discovery, save state. The process can even choose to ignore SIGTERM entirely (though this is bad practice). SIGTERM is a polite request.

**SIGKILL** (signal 9) says: "Die now." The process cannot catch it. The process cannot ignore it. The kernel terminates the process immediately, no cleanup, no final writes, no graceful anything. Open files may have unwritten data. Database transactions may be half-complete. In-progress video generations vanish.

This is why `kill -9` should be your **last resort**, not your first instinct. Using it on a database process can corrupt data. Using it on a web server drops all in-flight requests without response. Always try SIGTERM first and wait.

### The docker stop Sequence

When you run `docker stop <container>`:

1. Docker sends **SIGTERM** to PID 1 inside the container.
2. Docker waits for a grace period (default: 10 seconds).
3. If the process is still running after the grace period, Docker sends **SIGKILL**.

This is why containers sometimes take exactly 10 seconds to stop --- the application is not handling SIGTERM, so it ignores the signal and Docker has to kill it. This is also why your application must have a SIGTERM handler.

<svg viewBox="0 0 860 300" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <defs>
    <marker id="arr-sig" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="430" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">docker stop Signal Flow</text>

  <!-- Docker daemon -->
  <rect x="30" y="70" width="130" height="60" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="95" y="100" text-anchor="middle" font-size="12" font-weight="bold" fill="#1976d2">Docker</text>
  <text x="95" y="116" text-anchor="middle" font-size="10" fill="#555">docker stop</text>

  <!-- tini / PID 1 -->
  <rect x="250" y="70" width="130" height="60" rx="8" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="315" y="95" text-anchor="middle" font-size="11" font-weight="bold" fill="#388e3c">tini (PID 1)</text>
  <text x="315" y="112" text-anchor="middle" font-size="9" fill="#555">Forwards signals</text>

  <!-- Node app -->
  <rect x="470" y="70" width="150" height="60" rx="8" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="545" y="95" text-anchor="middle" font-size="11" font-weight="bold" fill="#f57c00">Node.js (PID 2)</text>
  <text x="545" y="112" text-anchor="middle" font-size="9" fill="#555">Your application</text>

  <!-- Workers -->
  <rect x="700" y="70" width="130" height="60" rx="8" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
  <text x="765" y="95" text-anchor="middle" font-size="11" font-weight="bold" fill="#e91e63">ffmpeg (PID 3)</text>
  <text x="765" y="112" text-anchor="middle" font-size="9" fill="#555">Child process</text>

  <!-- SIGTERM arrows -->
  <line x1="160" y1="90" x2="248" y2="90" stroke="#388e3c" stroke-width="2" marker-end="url(#arr-sig)"/>
  <text x="200" y="82" font-size="9" font-weight="bold" fill="#388e3c">SIGTERM</text>

  <line x1="380" y1="90" x2="468" y2="90" stroke="#388e3c" stroke-width="2" marker-end="url(#arr-sig)"/>
  <text x="420" y="82" font-size="9" font-weight="bold" fill="#388e3c">SIGTERM</text>

  <line x1="620" y1="90" x2="698" y2="90" stroke="#388e3c" stroke-width="2" marker-end="url(#arr-sig)"/>
  <text x="655" y="82" font-size="9" font-weight="bold" fill="#388e3c">SIGTERM</text>

  <!-- Timeline -->
  <line x1="30" y1="180" x2="830" y2="180" stroke="#ddd" stroke-width="2"/>

  <circle cx="100" cy="180" r="6" fill="#388e3c"/>
  <text x="100" y="205" text-anchor="middle" font-size="10" fill="#333">t=0s</text>
  <text x="100" y="220" text-anchor="middle" font-size="9" fill="#388e3c">SIGTERM sent</text>

  <rect x="200" y="170" width="300" height="20" rx="4" fill="#e8f5e9" stroke="#388e3c" stroke-width="1"/>
  <text x="350" y="185" text-anchor="middle" font-size="9" fill="#388e3c">Grace period: app performs cleanup</text>

  <circle cx="560" cy="180" r="6" fill="#81c784"/>
  <text x="560" y="205" text-anchor="middle" font-size="10" fill="#333">t=2s</text>
  <text x="560" y="220" text-anchor="middle" font-size="9" fill="#81c784">App exits cleanly</text>

  <circle cx="750" cy="180" r="6" fill="#ef5350"/>
  <text x="750" y="205" text-anchor="middle" font-size="10" fill="#333">t=10s</text>
  <text x="750" y="220" text-anchor="middle" font-size="9" fill="#ef5350">SIGKILL (if still alive)</text>

  <!-- Good/bad labels -->
  <text x="430" y="270" text-anchor="middle" font-size="11" fill="#333">
    <tspan fill="#388e3c" font-weight="bold">Good:</tspan> App handles SIGTERM, exits at t=2s.
    <tspan fill="#ef5350" font-weight="bold">Bad:</tspan> App ignores SIGTERM, gets killed at t=10s.
  </text>
</svg>

### Writing a Graceful Shutdown Handler

Every production Node.js application must handle SIGTERM:

```javascript
// Graceful shutdown handler
async function shutdown(signal) {
  console.log(`Received ${signal}. Starting graceful shutdown...`);

  // 1. Stop accepting new connections
  server.close(() => {
    console.log('HTTP server closed');
  });

  // 2. Close WebSocket connections with a proper close frame
  wss.clients.forEach((client) => {
    client.close(1001, 'Server shutting down');
  });

  // 3. Wait for in-progress requests to complete (with timeout)
  await Promise.race([
    waitForInflightRequests(),
    new Promise((resolve) => setTimeout(resolve, 5000)), // 5s max
  ]);

  // 4. Close database connections
  await db.end();

  // 5. Close Redis connection
  await redis.quit();

  console.log('Graceful shutdown complete');
  process.exit(0);
}

process.on('SIGTERM', () => shutdown('SIGTERM'));
process.on('SIGINT', () => shutdown('SIGINT'));
```

### SIGHUP: Why Your Process Dies When You Close the Terminal

When you SSH into a server and run `node server.js`, your Node.js process is a child of the SSH shell. When you close the SSH connection, the terminal sends **SIGHUP** (hangup) to all processes in the session. The default action for SIGHUP is termination. Your server dies.

This is why `nohup`, `tmux`, and `screen` exist:

```bash
# nohup: ignore SIGHUP, redirect output to nohup.out
nohup node server.js &

# tmux: create a persistent terminal session that survives SSH disconnects
tmux new -s video-platform
node server.js
# Ctrl+B, then D to detach. Reconnect with: tmux attach -t video-platform

# But really, you should use systemd (Section 6)
```

---

## 4. File Descriptors: The Invisible Resource Limit

This is the section that explains the opening story. File descriptors are the most common invisible resource limit, and running out of them is one of the most common production failures for WebSocket-heavy applications like AI video platforms.

### Everything Is a File

Linux has a famous design philosophy: **everything is a file**. This does not mean everything is literally stored on disk. It means that the kernel provides a uniform interface --- open, read, write, close --- for interacting with many different things:

| Thing | File? | Example |
|-------|-------|---------|
| Regular file on disk | Yes | `/opt/video-platform/config.json` |
| Network socket (TCP connection) | Treated as a file | Connection to client at 203.0.113.42:52843 |
| Pipe between processes | Treated as a file | `cat log.txt \| grep error` |
| Terminal | Treated as a file | `/dev/tty0` |
| Random number generator | Treated as a file | `/dev/urandom` |
| Null device (data sink) | Treated as a file | `/dev/null` |

### What a File Descriptor Is

A **file descriptor** (fd) is a non-negative integer that your process uses as a handle to refer to an open file (or socket, or pipe). When your process opens a file, the kernel:

1. Creates an entry in the system-wide open file table.
2. Adds a pointer to that entry in your process's file descriptor table.
3. Returns the lowest available integer (the file descriptor) to your process.

Every process starts with three file descriptors already open:

| fd | Name | Purpose | Where it goes |
|----|------|---------|---------------|
| 0 | **stdin** | Standard input | Keyboard, or pipe from another process |
| 1 | **stdout** | Standard output | Terminal, or pipe to another process |
| 2 | **stderr** | Standard error | Terminal (error messages) |

Every subsequent open operation gets the next available integer: 3, 4, 5, and so on.

### Network Connections Are File Descriptors

This is the critical insight. When your WebSocket server accepts a new connection, the kernel creates a new socket and gives your process a file descriptor for it. If you have 1,000 concurrent WebSocket connections, your process has 1,000 file descriptors for those connections, plus fd 0, 1, 2, plus file descriptors for your database connection, Redis connection, log files, and the listening socket itself.

<svg viewBox="0 0 860 380" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <text x="430" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">File Descriptor Table for a Video Platform Process</text>

  <!-- Process box -->
  <rect x="30" y="50" width="350" height="310" rx="8" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
  <text x="205" y="75" text-anchor="middle" font-size="13" font-weight="bold" fill="#333">Process: node server.js (PID 48271)</text>

  <!-- FD table header -->
  <rect x="50" y="90" width="50" height="25" fill="#e3f2fd" stroke="#1976d2" stroke-width="1"/>
  <text x="75" y="107" text-anchor="middle" font-size="10" font-weight="bold" fill="#1976d2">fd</text>
  <rect x="100" y="90" width="260" height="25" fill="#e3f2fd" stroke="#1976d2" stroke-width="1"/>
  <text x="230" y="107" text-anchor="middle" font-size="10" font-weight="bold" fill="#1976d2">Description</text>

  <!-- FD entries -->
  <rect x="50" y="115" width="50" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="75" y="130" text-anchor="middle" font-size="10" fill="#333">0</text>
  <rect x="100" y="115" width="260" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="130" text-anchor="middle" font-size="10" fill="#555">stdin (/dev/null)</text>

  <rect x="50" y="137" width="50" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="75" y="152" text-anchor="middle" font-size="10" fill="#333">1</text>
  <rect x="100" y="137" width="260" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="152" text-anchor="middle" font-size="10" fill="#555">stdout (→ journald)</text>

  <rect x="50" y="159" width="50" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="75" y="174" text-anchor="middle" font-size="10" fill="#333">2</text>
  <rect x="100" y="159" width="260" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="174" text-anchor="middle" font-size="10" fill="#555">stderr (→ journald)</text>

  <rect x="50" y="181" width="50" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="75" y="196" text-anchor="middle" font-size="10" fill="#333">3</text>
  <rect x="100" y="181" width="260" height="22" fill="#e8f5e9" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="196" text-anchor="middle" font-size="10" fill="#2e7d32">TCP listener :3000</text>

  <rect x="50" y="203" width="50" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="75" y="218" text-anchor="middle" font-size="10" fill="#333">4</text>
  <rect x="100" y="203" width="260" height="22" fill="#fff3e0" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="218" text-anchor="middle" font-size="10" fill="#e65100">PostgreSQL connection</text>

  <rect x="50" y="225" width="50" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="75" y="240" text-anchor="middle" font-size="10" fill="#333">5</text>
  <rect x="100" y="225" width="260" height="22" fill="#ffcdd2" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="240" text-anchor="middle" font-size="10" fill="#c62828">Redis connection</text>

  <rect x="50" y="247" width="50" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="75" y="262" text-anchor="middle" font-size="10" fill="#333">6</text>
  <rect x="100" y="247" width="260" height="22" fill="#e3f2fd" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="262" text-anchor="middle" font-size="10" fill="#1565c0">WebSocket: client 203.0.113.42</text>

  <rect x="50" y="269" width="50" height="22" fill="#fff" stroke="#ddd" stroke-width="1"/>
  <text x="75" y="284" text-anchor="middle" font-size="10" fill="#333">7</text>
  <rect x="100" y="269" width="260" height="22" fill="#e3f2fd" stroke="#ddd" stroke-width="1"/>
  <text x="230" y="284" text-anchor="middle" font-size="10" fill="#1565c0">WebSocket: client 198.51.100.7</text>

  <text x="205" y="310" text-anchor="middle" font-size="10" fill="#888">... fd 8 through fd 63,847 ...</text>

  <rect x="50" y="320" width="50" height="22" fill="#fff" stroke="#ef5350" stroke-width="2"/>
  <text x="75" y="335" text-anchor="middle" font-size="9" fill="#ef5350">63847</text>
  <rect x="100" y="320" width="260" height="22" fill="#ffcdd2" stroke="#ef5350" stroke-width="2"/>
  <text x="230" y="335" text-anchor="middle" font-size="10" fill="#c62828">WebSocket: leaked connection</text>

  <!-- Limit annotation -->
  <rect x="430" y="140" width="400" height="80" rx="8" fill="#ffcdd2" stroke="#ef5350" stroke-width="2"/>
  <text x="630" y="165" text-anchor="middle" font-size="12" font-weight="bold" fill="#c62828">⚠ ulimit -n = 65536</text>
  <text x="630" y="185" text-anchor="middle" font-size="10" fill="#555">Process has 63,847 open fds</text>
  <text x="630" y="202" text-anchor="middle" font-size="10" fill="#ef5350">Only 1,689 remaining before failure!</text>

  <!-- Arrow from FD table to limit -->
  <line x1="360" y1="180" x2="428" y2="180" stroke="#ef5350" stroke-width="2" stroke-dasharray="6,3"/>
</svg>

### The Default Limit: 1024

Here is the problem. The default per-process file descriptor limit on most Linux distributions is **1024**. That means your process can have at most 1024 open file descriptors. Subtract stdin (0), stdout (1), stderr (2), your database connection (3), your Redis connection (4), your listening socket (5), and you have **1018 file descriptors left for client connections**.

For a WebSocket server, that means 1018 concurrent connections. The 1019th connection attempt fails silently --- the kernel returns an `EMFILE` (too many open files) error, and depending on your framework, the connection either drops or the server crashes.

Check your current limits:

```bash
# Soft limit (what your process actually gets)
ulimit -Sn
# 1024

# Hard limit (maximum the soft limit can be raised to)
ulimit -Hn
# 65536
```

### Raising the Limit

**Temporarily** (for the current shell session):

```bash
ulimit -n 65535
```

**Permanently** (system-wide, in `/etc/security/limits.conf`):

```
# /etc/security/limits.conf
deploy    soft    nofile    65535
deploy    hard    nofile    65535
```

**For a systemd service** (the right way for production):

```ini
[Service]
LimitNOFILE=65535
```

### Diagnosing File Descriptor Leaks

The opening story was a file descriptor leak: WebSocket connections were being opened but never properly closed when clients disconnected. Each leaked connection kept its file descriptor open. Over three days, 63,847 accumulated.

Here is how to diagnose this:

```bash
# Count open file descriptors for a process
ls /proc/$(pgrep -f "node server")/fd | wc -l

# See what those file descriptors actually are
ls -la /proc/$(pgrep -f "node server")/fd | head -20

# Count by type
lsof -p $(pgrep -f "node server") | awk '{print $5}' | sort | uniq -c | sort -rn
```

If the count is growing over time without bound, you have a leak. The `lsof` output will tell you what type of file descriptor is leaking (usually `IPv4` or `IPv6` for socket leaks).

The fix for WebSocket leaks in Node.js:

```javascript
wss.on('connection', (ws, req) => {
  // Handle disconnection
  ws.on('close', () => {
    // Clean up any references to this connection
    activeConnections.delete(ws);
  });

  // Handle errors (which also means the connection is dead)
  ws.on('error', (err) => {
    console.error('WebSocket error:', err.message);
    activeConnections.delete(ws);
    ws.terminate(); // Force close to release the fd
  });

  // Detect broken connections with ping/pong
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });
});

// Periodic cleanup: terminate connections that don't respond to ping
setInterval(() => {
  wss.clients.forEach((ws) => {
    if (!ws.isAlive) {
      ws.terminate();
      return;
    }
    ws.isAlive = false;
    ws.ping();
  });
}, 30000); // Every 30 seconds
```

---

## 5. The /proc Filesystem: Your Window Into the Kernel

The `/proc` filesystem is not a real filesystem. There are no files stored on disk. Instead, `/proc` is a virtual filesystem where the kernel exposes its internal state as readable files. It is the single most powerful diagnostic tool on a Linux system, and most developers never look at it.

### Per-Process Information: /proc/\<PID\>/

Every running process has a directory under `/proc/` named by its PID:

```bash
# Find your process
PID=$(pgrep -f "node server")

# Process command line (how it was started)
cat /proc/$PID/cmdline | tr '\0' ' '
# node /opt/video-platform/dist/server.js

# Process status (memory, state, threads)
cat /proc/$PID/status

# Key fields in status:
# VmRSS:    resident set size (physical memory used)
# VmSize:   virtual memory size (allocated address space)
# Threads:  number of threads
# State:    R (running), S (sleeping), etc.
```

### Monitoring Memory Over Time

One of the most practical uses of `/proc` is tracking memory usage over time to detect leaks:

```bash
# Watch RSS (physical memory) every 5 seconds
while true; do
  RSS=$(grep VmRSS /proc/$PID/status | awk '{print $2}')
  FD_COUNT=$(ls /proc/$PID/fd 2>/dev/null | wc -l)
  echo "$(date '+%H:%M:%S') RSS: ${RSS} kB, FDs: ${FD_COUNT}"
  sleep 5
done
```

If RSS or FD count climbs steadily without ever decreasing, you have a leak.

### System-Wide Information

```bash
# CPU information
cat /proc/cpuinfo | grep "model name" | head -1
# model name : Intel(R) Xeon(R) CPU @ 2.20GHz

# Memory overview
cat /proc/meminfo | head -5
# MemTotal:        8052444 kB
# MemFree:          524288 kB
# MemAvailable:    4215808 kB
# Buffers:          312456 kB
# Cached:          3248912 kB

# Load average (1min, 5min, 15min)
cat /proc/loadavg
# 2.41 1.89 1.54 3/487 52819

# System-wide file descriptor usage
cat /proc/sys/fs/file-nr
# 12416  0  9223372036854775807
# (allocated, free, max)
```

### Tunable Kernel Parameters: /proc/sys/

The `/proc/sys/` directory contains kernel parameters you can read and modify at runtime. These are the knobs you turn when tuning a production server:

```bash
# Maximum system-wide file descriptors
cat /proc/sys/fs/file-max
# 9223372036854775807

# TCP listen backlog (max pending connections)
cat /proc/sys/net/core/somaxconn
# 4096

# Increase it for high-traffic servers:
echo 65535 > /proc/sys/net/core/somaxconn

# Make it permanent (survives reboot):
echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
sysctl -p
```

The parameters that matter most for a web server:

| Parameter | Default | Production Value | Purpose |
|-----------|---------|-----------------|---------|
| `net.core.somaxconn` | 4096 | 65535 | TCP listen backlog size |
| `net.ipv4.tcp_tw_reuse` | 0 | 1 | Reuse TIME_WAIT sockets |
| `net.ipv4.tcp_fin_timeout` | 60 | 30 | Seconds before closing FIN_WAIT_2 |
| `fs.file-max` | varies | 2097152 | System-wide file descriptor limit |
| `vm.swappiness` | 60 | 10 | How aggressively to swap (lower = less swapping) |

---

## 6. Systemd: Managing Long-Running Services

You should not run production services by SSHing into a server and typing `node server.js` in a `tmux` session. This is what many vibe-coded projects do, and it fails in every predictable way: the process crashes and nobody restarts it, the server reboots and the service does not start, logs go to a tmux buffer that fills up and gets lost.

**Systemd** is the init system and service manager on virtually all modern Linux distributions (Ubuntu, Debian, RHEL, Fedora, Arch). It starts services on boot, restarts them when they crash, manages their logs, and controls their resource limits. If you deploy to a Linux VPS, systemd is how you should run your application.

### Anatomy of a Unit File

A systemd **unit file** describes a service: what to run, how to run it, when to start it, and what to do when it fails. Here is a complete, production-ready unit file for an AI video platform API:

```ini
[Unit]
Description=AI Video Platform API Server
Documentation=https://github.com/your-org/video-platform
After=network.target postgresql.service redis.service
Requires=postgresql.service
Wants=redis.service

[Service]
Type=simple
User=deploy
Group=deploy
WorkingDirectory=/opt/video-platform
ExecStart=/usr/bin/node dist/server.js
Restart=on-failure
RestartSec=5
StartLimitBurst=5
StartLimitIntervalSec=60

# Resource limits
LimitNOFILE=65535
MemoryMax=2G
CPUQuota=200%

# Environment
Environment=NODE_ENV=production
Environment=PORT=3000
EnvironmentFile=/opt/video-platform/.env

# Security hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/video-platform/uploads
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Let me explain every section:

**`[Unit]` section --- metadata and dependencies:**

- `Description`: Human-readable name shown in `systemctl status`.
- `After`: Start this service after these other services are running. This controls ordering only --- it does not create a hard dependency.
- `Requires`: If `postgresql.service` fails to start, this service will not start either. Hard dependency.
- `Wants`: Soft dependency. Try to start `redis.service`, but if it fails, start our service anyway.

**`[Service]` section --- how to run:**

- `Type=simple`: The process started by `ExecStart` is the main service process. This is correct for Node.js, which stays in the foreground.
- `User=deploy`: Run as the `deploy` user, not root. Never run application code as root.
- `WorkingDirectory`: Set the working directory before starting the process.
- `ExecStart`: The command to run. Must be an absolute path.
- `Restart=on-failure`: If the process exits with a non-zero exit code (crash), restart it. Does not restart on clean exit (exit code 0).
- `RestartSec=5`: Wait 5 seconds between restarts. Prevents crash loops from consuming resources.
- `StartLimitBurst=5` and `StartLimitIntervalSec=60`: If the service restarts 5 times within 60 seconds, stop trying. This prevents infinite crash-restart loops.
- `LimitNOFILE=65535`: Set the file descriptor limit for this service.
- `MemoryMax=2G`: Kill the process if it uses more than 2 GB of RAM. Prevents memory leaks from consuming the entire server.
- `CPUQuota=200%`: Limit to 2 CPU cores (200% = 2 × 100%).
- `EnvironmentFile`: Load environment variables from a file. Keep secrets out of the unit file.

**Security hardening:**

- `NoNewPrivileges`: The process cannot escalate privileges (e.g., via setuid binaries).
- `ProtectSystem=strict`: The filesystem is read-only except for explicitly allowed paths.
- `ReadWritePaths`: Explicitly allow writing to the uploads directory.
- `PrivateTmp`: The service gets its own `/tmp` directory, isolated from other services.

**`[Install]` section --- when to start:**

- `WantedBy=multi-user.target`: Start this service when the system reaches multi-user mode (normal boot).

### Managing the Service

Save the unit file to `/etc/systemd/system/video-platform.service`, then:

```bash
# Reload systemd to pick up the new unit file
sudo systemctl daemon-reload

# Start the service
sudo systemctl start video-platform

# Check status
sudo systemctl status video-platform

# Enable auto-start on boot
sudo systemctl enable video-platform

# Stop the service (sends SIGTERM, then SIGKILL after timeout)
sudo systemctl stop video-platform

# Restart (stop + start)
sudo systemctl restart video-platform

# View recent logs
sudo journalctl -u video-platform --since "5 minutes ago"

# Follow logs in real time
sudo journalctl -u video-platform -f
```

### The Status Output

`systemctl status` is the first thing you should run when diagnosing a problem:

```
● video-platform.service - AI Video Platform API Server
     Loaded: loaded (/etc/systemd/system/video-platform.service; enabled)
     Active: active (running) since Fri 2026-03-07 14:23:01 UTC; 2h 14min ago
   Main PID: 48271 (node)
      Tasks: 11 (limit: 4915)
     Memory: 247.3M (max: 2.0G)
        CPU: 4min 23.451s
     CGroup: /system.slice/video-platform.service
             └─48271 /usr/bin/node dist/server.js

Mar 07 16:35:12 vps-01 node[48271]: {"level":"info","msg":"Request completed","status":200,...}
Mar 07 16:35:14 vps-01 node[48271]: {"level":"info","msg":"Generation started","id":"gen_abc",...}
```

This tells you: the service is running, it has been up for 2 hours 14 minutes, it is using 247 MB of its 2 GB limit, and here are the most recent log lines. If it had crashed, you would see the exit code and the timestamp of the last restart.

---

## 7. Journald: Where Your Logs Actually Go

When your systemd service writes to stdout or stderr, those outputs are captured by **journald** --- the logging daemon that is part of systemd. Journald stores logs in a structured binary format that supports efficient querying, automatic rotation, and metadata tagging.

### Reading Logs

```bash
# All logs for your service
journalctl -u video-platform

# Follow mode (like tail -f)
journalctl -u video-platform -f

# Logs since a specific time
journalctl -u video-platform --since "2026-03-07 02:00:00"
journalctl -u video-platform --since "1 hour ago"

# Only errors
journalctl -u video-platform -p err

# Specific time range
journalctl -u video-platform --since "02:00" --until "02:30"

# Show logs without pager (for scripting)
journalctl -u video-platform --no-pager

# Show in JSON format (for parsing)
journalctl -u video-platform -o json-pretty | head -50

# Show only the last 100 lines
journalctl -u video-platform -n 100

# Logs from previous boot (if the server restarted)
journalctl -u video-platform -b -1
```

### Why Journald Is Better Than Manual Log Files

Many vibe-coded projects write logs to a file: `node server.js > /var/log/app.log 2>&1`. This works until it does not:

| Problem | Manual log files | Journald |
|---------|-----------------|----------|
| Log rotation | You must configure `logrotate` yourself or the file grows forever | Automatic. Configurable size limit. |
| Disk full | Logs fill the disk, server crashes | Journald has configurable max size, auto-deletes old entries |
| Querying by time | `grep` through entire file | `--since` and `--until` flags |
| Querying by severity | Hope you included severity in the log format | `-p err`, `-p warning` |
| Multiple services | Multiple files, multiple formats | `journalctl -u service1 -u service2` |
| Structured data | Text parsing with regex | JSON output with `journalctl -o json` |

### Journald Configuration

Journald's configuration is in `/etc/systemd/journald.conf`:

```ini
[Journal]
# Maximum disk space for logs
SystemMaxUse=500M

# Maximum size of individual journal files
SystemMaxFileSize=50M

# How long to keep logs
MaxRetentionSec=30day

# Compress stored logs
Compress=yes

# Forward to syslog if needed
ForwardToSyslog=no
```

After changing the configuration:

```bash
sudo systemctl restart systemd-journald
```

### Structured Logging With Journald

If your application outputs structured JSON logs (which it should --- see Part 4 of this series), journald preserves the structure. You can query by custom fields:

```bash
# If your app logs: {"level":"error","service":"generation","error":"timeout"}
journalctl -u video-platform -o json | jq 'select(.MESSAGE | fromjson? | .level == "error")'
```

---

## 8. Putting It All Together: The 2 AM Triage Runbook

When something is wrong in production, you need a systematic approach. Panic-driven Googling at 2 AM is not a strategy. Here is a step-by-step triage runbook that covers the most common production failures. Run each step in order. The first abnormal result is usually your answer.

### Step 1: Is the service running?

```bash
systemctl status video-platform
```

**Look for:**
- `Active: active (running)` --- good
- `Active: failed` --- the process crashed. Check the exit code and recent logs.
- `Active: activating (auto-restart)` --- crash loop. Check `RestartSec` and `StartLimitBurst`.

If failed, check why:

```bash
journalctl -u video-platform --since "10 minutes ago" --no-pager
```

### Step 2: System overview

```bash
htop
```

**Look for:**
- CPU maxed out (all bars full): check which process is consuming CPU
- Memory maxed out (RAM bar full, swap in use): likely a memory leak or under-provisioned server
- Load average > number of CPU cores: the system is overloaded

### Step 3: Recent logs

```bash
journalctl -u video-platform --since "30 minutes ago" -p err --no-pager
```

**Look for:** Error messages, stack traces, connection refused errors, timeout errors.

### Step 4: File descriptors

```bash
PID=$(pgrep -f "node server" | head -1)
echo "Open FDs: $(ls /proc/$PID/fd 2>/dev/null | wc -l)"
echo "FD limit: $(cat /proc/$PID/limits | grep 'Max open files' | awk '{print $4}')"
```

**Look for:** FD count approaching the limit. If it is above 80% of the limit and growing, you have a file descriptor leak.

### Step 5: Memory details

```bash
cat /proc/$PID/status | grep -E "VmRSS|VmSize|Threads"
```

**Look for:**
- `VmRSS` (physical memory) growing without bound: memory leak
- `Threads` count much higher than expected: thread leak

### Step 6: Network connections

```bash
# What is listening on which ports
ss -tlnp

# Count established connections
ss -tn state established | wc -l

# Count connections by state
ss -tan | awk '{print $1}' | sort | uniq -c | sort -rn
```

**Look for:**
- Your service not listening on the expected port: it crashed or the port is wrong
- Thousands of `TIME_WAIT` connections: connections not being closed properly
- Thousands of `CLOSE_WAIT` connections: your application is not closing connections

### Step 7: Kernel messages

```bash
dmesg | tail -30
```

**Look for:**
- `Out of memory: Killed process 48271 (node)`: the OOM killer terminated your process because the system ran out of memory. You need more RAM or your app has a memory leak.
- `nf_conntrack: table full`: the connection tracking table is full. Increase `net.netfilter.nf_conntrack_max`.

### Step 8: Disk space

```bash
df -h
```

**Look for:**
- Any filesystem at 100%: logs filling the disk, uploads filling the disk, Docker images filling the disk.
- The root filesystem (`/`) is the most critical. If it fills up, the system can behave in bizarre ways.

### Step 9: Memory overview

```bash
free -h
```

```
              total        used        free      shared  buff/cache   available
Mem:           7.7G        3.2G        512M        124M        4.0G        4.1G
Swap:          2.0G        128M        1.9G
```

**Look for:**
- `available` (not `free`) is the real measure of available memory. Linux uses "free" memory for disk caching, so `free` is always low. `available` accounts for cache that can be reclaimed.
- Swap usage: if swap is heavily used, your server is under memory pressure and performance is degraded.

### Step 10: Connection to the application

```bash
# Health check from localhost
curl -w "\n%{http_code} %{time_total}s\n" http://localhost:3000/health

# If the health check hangs, the event loop might be blocked
# Check from outside the server too
```

**Look for:**
- Connection refused: the app is not listening (crashed or wrong port)
- Timeout: the app is alive but not responding (event loop blocked, thread pool exhausted)
- 503/500: the app is responding but unhealthy

---

## 9. The 2 AM Incident Checklist

Use this checklist to prepare for incidents before they happen. Every item should be in place before you consider your service production-ready.

1. **Set `LimitNOFILE=65535` in your systemd unit file.** The default of 1024 is too low for any WebSocket server.

2. **Add SIGTERM graceful shutdown handlers.** Close the HTTP server, close WebSocket connections with proper close frames, drain in-progress requests, close database and Redis connections, then exit cleanly.

3. **Use `tini` or `--init` in Docker containers.** Your application should not be PID 1. If it is, orphaned child processes become zombies and signals are not forwarded.

4. **Monitor file descriptor count.** Export it as a Prometheus metric or log it periodically. Set an alert at 80% of the limit.

5. **Configure systemd `Restart=on-failure` with `RestartSec=5`.** Your process will crash. Automatic restart with a 5-second delay is the correct response. Set `StartLimitBurst` and `StartLimitIntervalSec` to prevent infinite crash loops.

6. **Use journald for logs, not manual log files.** Journald handles rotation, size limits, time-based querying, and structured data out of the box.

7. **Know your process tree.** Run `pstree -p` and understand what is running. Every unexpected process is a potential problem.

8. **Test your graceful shutdown.** Send `kill -TERM <PID>` to your running application and verify it shuts down within a few seconds, closes all connections, and exits with code 0.

9. **Write a triage runbook before the incident.** The 10-step runbook in Section 8 is a starting point. Customize it for your specific stack and failure modes. Print it out if you have to.

10. **Set up kernel parameter tuning.** `somaxconn`, `tcp_tw_reuse`, `file-max`, `vm.swappiness` --- tune these for your workload and make them permanent in `/etc/sysctl.conf`.

---

## 10. Series Navigation

This article is Part 6 of the series on taking vibe-coded AI projects to production.

| Part | Title | Focus |
|------|-------|-------|
| 1 | [Performance Engineering](/2026/03/02/vibe-code-to-production-performance-engineering.html) | Profiling, N+1 queries, caching, async I/O |
| 2 | [Containerizing & Deploying](/2026/03/03/containerizing-deploying-ai-video-platform.html) | Docker, Nginx, TLS, CI/CD |
| 3 | [Load Testing](/2026/03/04/load-testing-breaking-video-pipeline.html) | k6, stress/soak/spike testing, SLOs |
| 4 | [Observability](/2026/03/05/observability-failure-modes-production-ai.html) | Logging, Prometheus, Grafana, OpenTelemetry |
| 5 | [How Your Computer Runs Your Code](/2026/03/06/how-computers-run-your-code.html) | CPU caches, memory layout, branch prediction |
| **6** | **Linux for the 2 AM Incident** (this post) | **Processes, file descriptors, signals, systemd** |
| 7 | [Networking from Packet to Page Load](/2026/03/08/networking-from-packet-to-page-load.html) | DNS, TCP, TLS, reverse proxies, firewalls |

Parts 1--4 tell you **what to do** in production. Parts 5--7 explain **why it works** --- the foundational systems knowledge that makes the practices in Parts 1--4 make sense.

---

The 2 AM incident from the opening was a file descriptor leak. The WebSocket server was opening a file descriptor for every new connection and never closing them when clients disconnected. Over three days, 63,847 accumulated. The fix was six lines of code: a `close` event handler and a periodic ping/pong health check to terminate dead connections.

The diagnosis took 90 minutes because the developer had never heard of a file descriptor, did not know about `/proc`, and had no systematic triage approach. With the knowledge in this post, the same diagnosis takes 5 minutes: `systemctl status` (running), `htop` (CPU and memory fine), file descriptor count (63,847 out of 65,536), `lsof` (all sockets, most in CLOSE_WAIT state), grep the code for missing `close` handlers, deploy the fix.

Linux is not magic. It is a system, and systems can be understood. The processes, signals, file descriptors, and service managers in this post are the building blocks of every production deployment. You do not need to memorize them. You need to know they exist, what they do, and where to look when something breaks at 2 AM.
