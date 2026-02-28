---
layout: post
title: "Networking from Packet to Page Load: DNS, TCP, Ports, and Why Your Deploy Broke"
date: 2026-03-08
category: infra
---

# Networking from Packet to Page Load: DNS, TCP, Ports, and Why Your Deploy Broke

*This is Part 7 of the series on taking vibe-coded AI projects to production. Parts 1--4 covered [performance engineering](/2026/03/02/vibe-code-to-production-performance-engineering.html), [containerization](/2026/03/03/containerizing-deploying-ai-video-platform.html), [load testing](/2026/03/04/load-testing-breaking-video-pipeline.html), and [observability](/2026/03/05/observability-failure-modes-production-ai.html). Parts 5--6 covered [CPU caches and memory layout](/2026/03/06/how-computers-run-your-code.html) and [Linux fundamentals](/2026/03/07/linux-for-the-2am-incident.html). This concludes the Foundations sub-series.*

You deploy your AI video platform to a VPS. The Docker containers are running. You check `docker ps` --- all healthy, green checkmarks. You run `curl localhost:3000/health` from the server terminal and get `{"status":"ok"}`. Everything works.

Then you open your browser, type `https://video-platform.com`, and get `ERR_CONNECTION_REFUSED`. Nothing. The server is running. The application is healthy. But the browser cannot connect.

You spend three hours trying different things. You restart Docker. You redeploy. You check DNS. You read Stack Overflow answers about Nginx configuration. Nothing helps. Finally, at 1 AM, someone in a Discord channel asks: "What address is your app listening on?"

You check. It is `127.0.0.1:3000`. Not `0.0.0.0:3000`. The difference is one number --- `127.0.0.1` means "only accept connections from this machine." Your browser, on your laptop, is not this machine. The application was listening, but only for local connections.

The fix is changing one line. But finding the problem took three hours because you did not understand what an IP address means, what a port does, or how a request gets from your browser to your server. This post fills that gap. It traces the complete journey of a network request, from the moment you press Enter in the browser to the moment pixels appear on screen, defining every concept along the way.

---

## Table of Contents

1. [The Request Lifecycle: What Happens When You Type a URL](#1-the-request-lifecycle-what-happens-when-you-type-a-url)
2. [DNS: Translating Names to Numbers](#2-dns-translating-names-to-numbers)
3. [TCP: The Reliable Transport](#3-tcp-the-reliable-transport)
4. [Ports: What They Are and Why They Matter](#4-ports-what-they-are-and-why-they-matter)
5. [IP Addresses and Interfaces: 0.0.0.0 vs 127.0.0.1](#5-ip-addresses-and-interfaces-0000-vs-127001)
6. [TLS: How HTTPS Actually Works](#6-tls-how-https-actually-works)
7. [Reverse Proxies: Why Nginx Sits in Front of Your App](#7-reverse-proxies-why-nginx-sits-in-front-of-your-app)
8. [Firewalls: Controlling What Gets In](#8-firewalls-controlling-what-gets-in)
9. [Common Networking Failures and How to Debug Them](#9-common-networking-failures-and-how-to-debug-them)
10. [The Networking Debug Checklist](#10-the-networking-debug-checklist)
11. [Series Navigation](#11-series-navigation)

---

## 1. The Request Lifecycle: What Happens When You Type a URL

When you type `https://video-platform.com` into your browser and press Enter, a precise sequence of events occurs. Each step is a distinct operation using a distinct protocol, and each step can fail independently. Understanding this sequence is the foundation of networking knowledge.

Here is what happens, in order:

1. **URL parsing:** The browser breaks the URL into components: protocol (`https`), host (`video-platform.com`), port (443, implied by `https`), path (`/`).

2. **DNS resolution:** The browser translates the hostname `video-platform.com` into an IP address, say `203.0.113.42`. This involves querying a chain of DNS servers.

3. **TCP connection:** The browser opens a TCP connection to `203.0.113.42:443` using a three-way handshake (SYN, SYN-ACK, ACK). This establishes a reliable, ordered byte stream between your browser and the server.

4. **TLS handshake:** Since the protocol is HTTPS, the browser and server negotiate encryption. The server presents its certificate, the browser verifies it, and both sides agree on encryption keys.

5. **HTTP request:** The browser sends an HTTP request over the encrypted connection: `GET / HTTP/2`.

6. **Server processing:** The request arrives at Nginx (the reverse proxy), which forwards it to the Node.js application on `127.0.0.1:3000`. The application processes the request and returns a response.

7. **HTTP response:** The response travels back: Node.js → Nginx → TLS encryption → TCP → internet → your browser.

8. **Rendering:** The browser parses the HTML, requests additional resources (CSS, JS, images), and renders the page.

Every one of these steps has its own failure modes, its own debugging tools, and its own latency cost. The rest of this article walks through each one.

<svg viewBox="0 0 880 480" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <defs>
    <marker id="arr-net" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
  </defs>

  <text x="440" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">The Complete Request Lifecycle</text>

  <!-- Browser -->
  <rect x="30" y="60" width="120" height="50" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="90" y="82" text-anchor="middle" font-size="11" font-weight="bold" fill="#1976d2">Browser</text>
  <text x="90" y="98" text-anchor="middle" font-size="9" fill="#555">Your laptop</text>

  <!-- Steps -->
  <rect x="30" y="140" width="160" height="40" rx="6" fill="#fff3e0" stroke="#f57c00" stroke-width="1.5"/>
  <text x="110" y="157" text-anchor="middle" font-size="10" font-weight="bold" fill="#e65100">1. Parse URL</text>
  <text x="110" y="171" text-anchor="middle" font-size="8" fill="#888">https://video-platform.com</text>

  <rect x="30" y="195" width="160" height="40" rx="6" fill="#e8f5e9" stroke="#388e3c" stroke-width="1.5"/>
  <text x="110" y="212" text-anchor="middle" font-size="10" font-weight="bold" fill="#2e7d32">2. DNS Lookup</text>
  <text x="110" y="226" text-anchor="middle" font-size="8" fill="#888">→ 203.0.113.42 (~50ms)</text>

  <rect x="220" y="195" width="160" height="40" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="1.5"/>
  <text x="300" y="212" text-anchor="middle" font-size="10" font-weight="bold" fill="#1565c0">3. TCP Handshake</text>
  <text x="300" y="226" text-anchor="middle" font-size="8" fill="#888">SYN → SYN-ACK → ACK (~30ms)</text>

  <rect x="410" y="195" width="160" height="40" rx="6" fill="#fce4ec" stroke="#e91e63" stroke-width="1.5"/>
  <text x="490" y="212" text-anchor="middle" font-size="10" font-weight="bold" fill="#c2185b">4. TLS Handshake</text>
  <text x="490" y="226" text-anchor="middle" font-size="8" fill="#888">Cert + key exchange (~40ms)</text>

  <rect x="600" y="195" width="160" height="40" rx="6" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="1.5"/>
  <text x="680" y="212" text-anchor="middle" font-size="10" font-weight="bold" fill="#6a1b9a">5. HTTP Request</text>
  <text x="680" y="226" text-anchor="middle" font-size="8" fill="#888">GET / HTTP/2</text>

  <!-- Arrows between steps -->
  <line x1="190" y1="215" x2="218" y2="215" stroke="#333" stroke-width="1.5" marker-end="url(#arr-net)"/>
  <line x1="380" y1="215" x2="408" y2="215" stroke="#333" stroke-width="1.5" marker-end="url(#arr-net)"/>
  <line x1="570" y1="215" x2="598" y2="215" stroke="#333" stroke-width="1.5" marker-end="url(#arr-net)"/>

  <!-- Server side -->
  <rect x="250" y="290" width="400" height="120" rx="10" fill="#f5f5f5" stroke="#333" stroke-width="2"/>
  <text x="450" y="310" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Server (203.0.113.42)</text>

  <rect x="270" y="325" width="110" height="40" rx="6" fill="#e3f2fd" stroke="#1976d2" stroke-width="1.5"/>
  <text x="325" y="342" text-anchor="middle" font-size="10" font-weight="bold" fill="#1565c0">Nginx</text>
  <text x="325" y="356" text-anchor="middle" font-size="8" fill="#888">:443 → :3000</text>

  <rect x="400" y="325" width="110" height="40" rx="6" fill="#fff3e0" stroke="#f57c00" stroke-width="1.5"/>
  <text x="455" y="342" text-anchor="middle" font-size="10" font-weight="bold" fill="#e65100">Node.js</text>
  <text x="455" y="356" text-anchor="middle" font-size="8" fill="#888">:3000</text>

  <rect x="530" y="325" width="100" height="40" rx="6" fill="#e8f5e9" stroke="#388e3c" stroke-width="1.5"/>
  <text x="580" y="342" text-anchor="middle" font-size="10" font-weight="bold" fill="#2e7d32">Postgres</text>
  <text x="580" y="356" text-anchor="middle" font-size="8" fill="#888">:5432</text>

  <line x1="380" y1="345" x2="398" y2="345" stroke="#333" stroke-width="1.5" marker-end="url(#arr-net)"/>
  <line x1="510" y1="345" x2="528" y2="345" stroke="#333" stroke-width="1.5" marker-end="url(#arr-net)"/>

  <!-- Connection arrow from request to server -->
  <line x1="680" y1="235" x2="680" y2="270" stroke="#333" stroke-width="1.5"/>
  <line x1="680" y1="270" x2="450" y2="270" stroke="#333" stroke-width="1.5"/>
  <line x1="450" y1="270" x2="450" y2="288" stroke="#333" stroke-width="1.5" marker-end="url(#arr-net)"/>

  <!-- Response -->
  <rect x="280" y="440" width="340" height="30" rx="6" fill="#e8f5e9" stroke="#388e3c" stroke-width="1.5"/>
  <text x="450" y="460" text-anchor="middle" font-size="10" font-weight="bold" fill="#2e7d32">7. HTTP Response → 8. Browser Renders Page</text>

  <line x1="450" y1="410" x2="450" y2="438" stroke="#333" stroke-width="1.5" marker-end="url(#arr-net)"/>

  <!-- Timing -->
  <text x="440" y="490" text-anchor="middle" font-size="10" fill="#888">Total time to first byte: ~150-300ms (DNS + TCP + TLS + server processing)</text>
</svg>

---

## 2. DNS: Translating Names to Numbers

Computers communicate using IP addresses --- numbers like `203.0.113.42`. Humans communicate using domain names like `video-platform.com`. **DNS** (Domain Name System) is the system that translates between the two.

### The Resolution Chain

When your browser needs to resolve `video-platform.com`, it does not ask one DNS server. It follows a chain of increasingly authoritative sources:

**Step 1: Browser cache.** Has this domain been resolved recently? If so, use the cached result. No network request needed.

**Step 2: Operating system cache.** The OS maintains its own DNS cache. On Linux, `systemd-resolved` or `nscd` handles this. On macOS, it is `mDNSResponder`.

**Step 3: Resolver (recursive DNS server).** If neither cache has the answer, the OS sends a query to its configured DNS resolver. This is typically your ISP's DNS server, or a public resolver like `1.1.1.1` (Cloudflare) or `8.8.8.8` (Google). The resolver does the heavy lifting.

**Step 4: Root nameservers.** The resolver asks a root nameserver: "Who handles `.com`?" There are 13 root nameserver clusters (labeled A through M), distributed globally. The root responds: "The `.com` TLD nameservers are at `a.gtld-servers.net`, `b.gtld-servers.net`, etc."

**Step 5: TLD (Top-Level Domain) nameservers.** The resolver asks the `.com` nameserver: "Who handles `video-platform.com`?" The TLD server responds: "The authoritative nameservers for `video-platform.com` are `ns1.cloudflare.com` and `ns2.cloudflare.com`."

**Step 6: Authoritative nameserver.** The resolver asks `ns1.cloudflare.com`: "What is the IP address of `video-platform.com`?" The authoritative nameserver responds: "The A record for `video-platform.com` is `203.0.113.42`."

The resolver caches this result and returns it to your browser. The entire chain typically completes in 20--100ms, but if everything is cached, it can be under 1ms.

<svg viewBox="0 0 880 400" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <defs>
    <marker id="arr-dns" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#1976d2"/>
    </marker>
    <marker id="arr-dns-back" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#81c784"/>
    </marker>
  </defs>

  <text x="440" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">DNS Resolution Chain</text>

  <!-- Browser -->
  <rect x="30" y="160" width="100" height="50" rx="8" fill="#e3f2fd" stroke="#1976d2" stroke-width="2"/>
  <text x="80" y="185" text-anchor="middle" font-size="10" font-weight="bold" fill="#1976d2">Browser</text>
  <text x="80" y="200" text-anchor="middle" font-size="8" fill="#555">cache miss</text>

  <!-- Resolver -->
  <rect x="200" y="160" width="120" height="50" rx="8" fill="#e8f5e9" stroke="#388e3c" stroke-width="2"/>
  <text x="260" y="180" text-anchor="middle" font-size="10" font-weight="bold" fill="#2e7d32">Resolver</text>
  <text x="260" y="195" text-anchor="middle" font-size="8" fill="#555">1.1.1.1</text>
  <text x="260" y="205" text-anchor="middle" font-size="7" fill="#888">(does the work)</text>

  <!-- Root -->
  <rect x="400" y="60" width="130" height="45" rx="8" fill="#fff3e0" stroke="#f57c00" stroke-width="2"/>
  <text x="465" y="80" text-anchor="middle" font-size="10" font-weight="bold" fill="#e65100">Root NS</text>
  <text x="465" y="95" text-anchor="middle" font-size="8" fill="#555">a.root-servers.net</text>

  <!-- TLD -->
  <rect x="400" y="160" width="130" height="45" rx="8" fill="#fce4ec" stroke="#e91e63" stroke-width="2"/>
  <text x="465" y="180" text-anchor="middle" font-size="10" font-weight="bold" fill="#c2185b">.com TLD NS</text>
  <text x="465" y="195" text-anchor="middle" font-size="8" fill="#555">a.gtld-servers.net</text>

  <!-- Authoritative -->
  <rect x="400" y="260" width="130" height="45" rx="8" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="2"/>
  <text x="465" y="280" text-anchor="middle" font-size="10" font-weight="bold" fill="#6a1b9a">Authoritative NS</text>
  <text x="465" y="295" text-anchor="middle" font-size="8" fill="#555">ns1.cloudflare.com</text>

  <!-- Arrows: Browser to Resolver -->
  <line x1="130" y1="178" x2="198" y2="178" stroke="#1976d2" stroke-width="1.5" marker-end="url(#arr-dns)"/>
  <text x="165" y="172" font-size="8" fill="#1976d2">query</text>

  <!-- Resolver to Root -->
  <line x1="320" y1="170" x2="398" y2="90" stroke="#1976d2" stroke-width="1.5" marker-end="url(#arr-dns)"/>
  <text x="340" y="120" font-size="7" fill="#1976d2">1. Who has .com?</text>

  <!-- Root to Resolver -->
  <line x1="398" y1="95" x2="320" y2="175" stroke="#81c784" stroke-width="1.5" marker-end="url(#arr-dns-back)"/>
  <text x="375" y="145" font-size="7" fill="#388e3c">→ gtld-servers</text>

  <!-- Resolver to TLD -->
  <line x1="320" y1="185" x2="398" y2="182" stroke="#1976d2" stroke-width="1.5" marker-end="url(#arr-dns)"/>
  <text x="350" y="178" font-size="7" fill="#1976d2">2. Who has video-platform.com?</text>

  <!-- TLD to Resolver -->
  <line x1="398" y1="192" x2="320" y2="195" stroke="#81c784" stroke-width="1.5" marker-end="url(#arr-dns-back)"/>
  <text x="350" y="206" font-size="7" fill="#388e3c">→ ns1.cloudflare.com</text>

  <!-- Resolver to Auth -->
  <line x1="320" y1="200" x2="398" y2="272" stroke="#1976d2" stroke-width="1.5" marker-end="url(#arr-dns)"/>
  <text x="330" y="246" font-size="7" fill="#1976d2">3. IP of video-platform.com?</text>

  <!-- Auth to Resolver -->
  <line x1="398" y1="280" x2="320" y2="205" stroke="#81c784" stroke-width="1.5" marker-end="url(#arr-dns-back)"/>
  <text x="375" y="260" font-size="7" fill="#388e3c">→ 203.0.113.42</text>

  <!-- Resolver back to Browser -->
  <line x1="200" y1="195" x2="132" y2="195" stroke="#81c784" stroke-width="1.5" marker-end="url(#arr-dns-back)"/>
  <text x="165" y="208" font-size="8" fill="#388e3c">203.0.113.42</text>

  <!-- Result box -->
  <rect x="580" y="160" width="270" height="90" rx="8" fill="#f5f5f5" stroke="#ddd" stroke-width="1.5"/>
  <text x="715" y="182" text-anchor="middle" font-size="10" font-weight="bold" fill="#333">Result (cached with TTL)</text>
  <text x="715" y="200" text-anchor="middle" font-size="9" fill="#555">video-platform.com → 203.0.113.42</text>
  <text x="715" y="216" text-anchor="middle" font-size="9" fill="#555">TTL: 300 seconds (5 minutes)</text>
  <text x="715" y="236" text-anchor="middle" font-size="8" fill="#888">Total resolution: ~50-100ms (uncached)</text>
</svg>

### DNS Record Types

DNS does not only map domains to IP addresses. Different record types serve different purposes:

| Record Type | Purpose | Example |
|-------------|---------|---------|
| **A** | Domain → IPv4 address | `video-platform.com → 203.0.113.42` |
| **AAAA** | Domain → IPv6 address | `video-platform.com → 2001:db8::1` |
| **CNAME** | Domain → another domain (alias) | `www.video-platform.com → video-platform.com` |
| **MX** | Mail routing | `video-platform.com → mail.provider.com` (priority 10) |
| **TXT** | Arbitrary text (verification, SPF, DKIM) | `"v=spf1 include:_spf.google.com ~all"` |
| **NS** | Nameserver delegation | `video-platform.com → ns1.cloudflare.com` |

### TTL and the "Propagation" Myth

Every DNS record has a **TTL** (Time To Live) --- the number of seconds a resolver should cache the result before asking again. If your A record has a TTL of 300, resolvers will cache your IP address for 5 minutes.

When people say "DNS propagation takes 24--48 hours," they are describing a misunderstanding. There is no global propagation wave. What happens is:

1. You change your A record from `203.0.113.42` to `198.51.100.7`.
2. Resolvers that have the old record cached will continue serving `203.0.113.42` until their cache expires (up to the old TTL).
3. After the old TTL expires, they will query the authoritative nameserver and get the new IP.

If your TTL was 86400 (24 hours), then yes, it can take up to 24 hours. If your TTL was 300 (5 minutes), the "propagation" completes in 5 minutes.

**Pro tip:** Before a migration, lower your TTL to 60 seconds a few days in advance. After the migration, raise it back to 300--3600 seconds.

### Debugging DNS

```bash
# Simple lookup
dig video-platform.com A +short
# 203.0.113.42

# Full detail (shows TTL, authoritative server)
dig video-platform.com A

# Trace the full resolution chain
dig video-platform.com A +trace

# Query a specific DNS server
dig @1.1.1.1 video-platform.com A

# Check all record types
dig video-platform.com ANY +short

# Reverse lookup (IP → domain)
dig -x 203.0.113.42
```

### Common DNS Failures

- **Wrong A record:** Your domain points to the old server IP after a migration.
- **Missing A record:** You set up a CNAME for `www` but forgot the bare domain.
- **CNAME at apex:** You cannot have a CNAME on the root domain (`video-platform.com`) per the DNS spec. Use an A record or your provider's ALIAS/ANAME feature.
- **TTL too high:** You changed the IP but nobody sees the change for 24 hours.
- **Nameserver misconfiguration:** You registered your domain with one registrar but configured DNS at another, and forgot to update the NS records.

---

## 3. TCP: The Reliable Transport

IP (Internet Protocol) delivers packets from one machine to another, but it provides no guarantees. Packets can arrive out of order, be duplicated, or be lost entirely. This is fine for some use cases (video streaming, where a dropped frame is acceptable) but catastrophic for others (your database query, where one missing byte corrupts the entire result).

**TCP** (Transmission Control Protocol) builds reliability on top of IP. It guarantees that data arrives in order, without duplicates, and without loss. If a packet is lost, TCP retransmits it. If packets arrive out of order, TCP reorders them. The application sees a reliable, ordered stream of bytes.

### The Three-Way Handshake

Before any data can flow, the client and server must establish a TCP connection using a **three-way handshake**:

**Step 1 --- SYN:** The client sends a SYN (synchronize) packet to the server, saying: "I want to open a connection. My initial sequence number is X."

**Step 2 --- SYN-ACK:** The server responds with a SYN-ACK (synchronize-acknowledge): "I acknowledge your sequence number X. My initial sequence number is Y."

**Step 3 --- ACK:** The client responds with an ACK (acknowledge): "I acknowledge your sequence number Y. We are now connected."

After these three packets, the connection is established and data can flow in both directions.

<svg viewBox="0 0 700 350" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif">
  <defs>
    <marker id="arr-tcp" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#1976d2"/>
    </marker>
    <marker id="arr-tcp-r" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#388e3c"/>
    </marker>
  </defs>

  <text x="350" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333">TCP Three-Way Handshake</text>

  <!-- Client line -->
  <line x1="130" y1="60" x2="130" y2="330" stroke="#1976d2" stroke-width="2"/>
  <rect x="70" y="40" width="120" height="25" rx="5" fill="#e3f2fd" stroke="#1976d2" stroke-width="1.5"/>
  <text x="130" y="57" text-anchor="middle" font-size="11" font-weight="bold" fill="#1976d2">Client</text>

  <!-- Server line -->
  <line x1="530" y1="60" x2="530" y2="330" stroke="#388e3c" stroke-width="2"/>
  <rect x="470" y="40" width="120" height="25" rx="5" fill="#e8f5e9" stroke="#388e3c" stroke-width="1.5"/>
  <text x="530" y="57" text-anchor="middle" font-size="11" font-weight="bold" fill="#388e3c">Server</text>

  <!-- SYN -->
  <line x1="135" y1="100" x2="525" y2="140" stroke="#1976d2" stroke-width="2" marker-end="url(#arr-tcp)"/>
  <rect x="240" y="95" width="180" height="25" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="1"/>
  <text x="330" y="112" text-anchor="middle" font-size="10" font-weight="bold" fill="#1565c0">SYN (seq=1000)</text>
  <text x="130" y="95" text-anchor="middle" font-size="9" fill="#888">t=0ms</text>

  <!-- SYN-ACK -->
  <line x1="525" y1="170" x2="135" y2="210" stroke="#388e3c" stroke-width="2" marker-end="url(#arr-tcp-r)"/>
  <rect x="220" y="170" width="220" height="25" rx="4" fill="#e8f5e9" stroke="#388e3c" stroke-width="1"/>
  <text x="330" y="187" text-anchor="middle" font-size="10" font-weight="bold" fill="#2e7d32">SYN-ACK (seq=5000, ack=1001)</text>
  <text x="530" y="165" text-anchor="middle" font-size="9" fill="#888">t=15ms</text>

  <!-- ACK -->
  <line x1="135" y1="240" x2="525" y2="280" stroke="#1976d2" stroke-width="2" marker-end="url(#arr-tcp)"/>
  <rect x="260" y="240" width="140" height="25" rx="4" fill="#e3f2fd" stroke="#1976d2" stroke-width="1"/>
  <text x="330" y="257" text-anchor="middle" font-size="10" font-weight="bold" fill="#1565c0">ACK (ack=5001)</text>
  <text x="130" y="235" text-anchor="middle" font-size="9" fill="#888">t=30ms</text>

  <!-- Data -->
  <line x1="135" y1="300" x2="525" y2="320" stroke="#f57c00" stroke-width="2" stroke-dasharray="6,3" marker-end="url(#arr-tcp)"/>
  <text x="330" y="305" text-anchor="middle" font-size="10" fill="#f57c00">Data can now flow...</text>
  <text x="130" y="295" text-anchor="middle" font-size="9" fill="#888">t=30ms</text>

  <!-- RTT annotation -->
  <line x1="600" y1="100" x2="600" y2="210" stroke="#ef5350" stroke-width="1.5"/>
  <line x1="595" y1="100" x2="605" y2="100" stroke="#ef5350" stroke-width="1.5"/>
  <line x1="595" y1="210" x2="605" y2="210" stroke="#ef5350" stroke-width="1.5"/>
  <text x="625" y="160" font-size="10" fill="#ef5350" font-weight="bold">1 RTT</text>
  <text x="625" y="175" font-size="8" fill="#888">~30ms</text>
</svg>

### Why the Handshake Matters for Performance

The handshake costs one **round-trip time (RTT)** --- the time for a packet to travel from client to server and back. If the client is in New York and the server is in Frankfurt, the RTT might be 80ms. That means every new TCP connection starts with an 80ms delay before any data flows.

This is why:
- **HTTP/2 multiplexing** is important: it sends multiple requests over one TCP connection, paying the handshake cost only once.
- **Connection pooling** in your application matters: reusing TCP connections to the database avoids repeated handshakes.
- **CDNs** like Cloudflare place servers close to users, reducing RTT.

### Connection Teardown and TIME_WAIT

When a TCP connection closes, it goes through a four-step teardown (FIN, ACK, FIN, ACK). After the initiator sends its final ACK, it enters the **TIME_WAIT** state for 60 seconds (by default on Linux). During TIME_WAIT, the port is reserved and cannot be reused.

This is a deliberate design: it ensures that any delayed packets from the old connection do not get confused with a new connection on the same port. But it means that if your server handles thousands of short-lived connections, you can accumulate thousands of sockets in TIME_WAIT.

Check TIME_WAIT connections:

```bash
ss -tan state time-wait | wc -l
```

If you see thousands, consider enabling `tcp_tw_reuse`:

```bash
echo 1 > /proc/sys/net/ipv4/tcp_tw_reuse
```

### TCP Window Size and Throughput

TCP uses a **sliding window** to control how much data can be in-flight (sent but not yet acknowledged). The window size determines the maximum throughput on a connection.

The fundamental relationship is:

$$\text{Max Throughput} = \frac{\text{Window Size}}{\text{RTT}}$$

If the window size is 64 KB and the RTT is 100ms:

$$\text{Max Throughput} = \frac{64 \times 1024 \text{ bytes}}{0.1 \text{ s}} = 655{,}360 \text{ bytes/s} \approx 640 \text{ KB/s}$$

That is 640 KB/s regardless of how much bandwidth your network actually has. This is why high-latency connections (satellite internet, cross-continent links) are slow even with high bandwidth --- the TCP window limits how much data can be in-flight.

Modern TCP implementations use **window scaling** to allow windows up to 1 GB, which is sufficient for most connections. But on high-latency, high-bandwidth paths (the "long fat network" problem), you may need to tune kernel parameters:

```bash
# Increase TCP buffer sizes for high-bandwidth links
echo "net.core.rmem_max = 16777216" >> /etc/sysctl.conf
echo "net.core.wmem_max = 16777216" >> /etc/sysctl.conf
echo "net.ipv4.tcp_rmem = 4096 87380 16777216" >> /etc/sysctl.conf
echo "net.ipv4.tcp_wmem = 4096 65536 16777216" >> /etc/sysctl.conf
sysctl -p
```

---

## 4. Ports: What They Are and Why They Matter

An IP address identifies a machine on the network. But a machine runs many services simultaneously: a web server, an SSH daemon, a database, Redis. **Ports** distinguish between these services. A port is a 16-bit number (0--65535) that, combined with an IP address, identifies a specific service on a specific machine.

The combination of IP address + port is called a **socket address**. When your browser connects to `203.0.113.42:443`, it is connecting to port 443 on machine `203.0.113.42`.

### Port Ranges

| Range | Name | Purpose |
|-------|------|---------|
| 0--1023 | **Well-known ports** | Reserved for standard services. Require root/admin to bind. |
| 1024--49151 | **Registered ports** | Assigned by IANA for specific applications. |
| 49152--65535 | **Ephemeral ports** | Used by the OS for outgoing client connections. |

Common well-known ports:

| Port | Service | Protocol |
|------|---------|----------|
| 22 | SSH | Secure shell access |
| 80 | HTTP | Unencrypted web traffic |
| 443 | HTTPS | Encrypted web traffic |
| 3000 | (convention) | Node.js development server |
| 5432 | PostgreSQL | Database |
| 6379 | Redis | Cache/message broker |
| 8080 | (convention) | Alternative HTTP port |

### "Port Already in Use"

One of the most common deployment errors:

```
Error: listen EADDRINUSE: address already in use :::3000
```

This means another process is already listening on port 3000. Two processes cannot bind to the same port on the same address.

Diagnosing it:

```bash
# What is listening on port 3000?
ss -tlnp | grep :3000
# LISTEN  0  511  0.0.0.0:3000  0.0.0.0:*  users:(("node",pid=48271,fd=3))

# More detail with lsof
lsof -i :3000
# COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
# node    48271 deploy  3u  IPv4 123456      0t0  TCP *:3000 (LISTEN)
```

Now you know: PID 48271 (a Node.js process owned by `deploy`) is using port 3000. You can either stop that process or configure your new application to use a different port.

### Docker Port Mapping

When you run `docker run -p 8080:3000 video-platform`, you are creating a **port mapping**: traffic arriving at the host's port 8080 is forwarded to the container's port 3000. Inside the container, the application thinks it is listening on port 3000. Outside the container, clients connect to port 8080.

This is implemented using network namespace bridging --- Docker creates virtual network interfaces and uses `iptables` rules to route traffic between the host network and the container network.

```bash
# See the port mappings
docker ps
# CONTAINER ID  IMAGE           PORTS
# abc123def456  video-platform  0.0.0.0:8080->3000/tcp

# This means: host 0.0.0.0:8080 → container 3000
```

---

## 5. IP Addresses and Interfaces: 0.0.0.0 vs 127.0.0.1

This is the section that explains the opening story. The difference between `0.0.0.0` and `127.0.0.1` is the difference between "my deploy works" and "my deploy is broken."

### What an IP Address Is

An **IP address** is a numerical identifier assigned to a **network interface**. A network interface is a connection point to a network. Your server typically has several:

```bash
ip addr

# 1: lo: <LOOPBACK,UP>
#     inet 127.0.0.1/8
#     inet6 ::1/128
#
# 2: eth0: <BROADCAST,MULTICAST,UP>
#     inet 203.0.113.42/24
#     inet6 2001:db8::1/64
```

| Interface | IP Address | Purpose |
|-----------|-----------|---------|
| `lo` (loopback) | `127.0.0.1` | Traffic that stays on the machine. Never leaves the network card. |
| `eth0` (ethernet) | `203.0.113.42` | The public IP address. Traffic from the internet arrives here. |

### The Critical Difference: Binding Addresses

When your application "listens on a port," it binds to a specific address and port combination. The address determines **which network interface accepts connections**:

| Binding | Meaning | Who Can Connect |
|---------|---------|----------------|
| `127.0.0.1:3000` | Listen on loopback only | Only processes on this machine |
| `203.0.113.42:3000` | Listen on the public interface only | Anyone who can reach this IP |
| `0.0.0.0:3000` | Listen on **all interfaces** | Anyone --- local or remote |
| `[::]:3000` | Listen on all interfaces (IPv4 and IPv6) | Anyone --- local or remote |

**This is why the opening story happened.** The application was listening on `127.0.0.1:3000`. When you ran `curl localhost:3000` from the server, it worked because `localhost` resolves to `127.0.0.1` --- a local connection. When you tried from your browser (a remote connection), the traffic arrived on the `eth0` interface (`203.0.113.42`), but nobody was listening on that interface.

The fix in Express.js:

```javascript
// Wrong: only accessible from localhost
app.listen(3000, '127.0.0.1');

// Right: accessible from all interfaces
app.listen(3000, '0.0.0.0');

// Also right: Express defaults to 0.0.0.0 if you omit the host
app.listen(3000);
```

### Private vs Public IP Addresses

Not all IP addresses are routable on the public internet. Three ranges are reserved for **private networks**:

| Range | Number of Addresses | Common Use |
|-------|-------------------|------------|
| `10.0.0.0` -- `10.255.255.255` | 16.7 million | Cloud VPCs, corporate networks |
| `172.16.0.0` -- `172.31.255.255` | 1 million | Docker default bridge network |
| `192.168.0.0` -- `192.168.255.255` | 65,536 | Home routers |

If your server's `ip addr` shows a private IP (like `10.0.2.15`) as its only address, it is behind a **NAT** (Network Address Translation) device --- typically a cloud provider's networking layer or a home router. The NAT translates your private IP to a public IP for outgoing traffic. Incoming traffic must be explicitly routed through the NAT (using port forwarding, security groups, or load balancers).

### Docker Networking

Docker creates its own virtual networks and assigns private IP addresses to containers:

```bash
# Default bridge network: containers get 172.17.0.x addresses
docker inspect <container> | grep IPAddress
# "IPAddress": "172.17.0.2"
```

Containers can communicate with each other using these internal IPs, but they are not reachable from outside unless you publish ports with `-p`.

Docker's networking modes:

| Mode | How It Works | Use Case |
|------|-------------|----------|
| **bridge** (default) | Container gets its own network namespace. Port mapping required for external access. | Most applications |
| **host** | Container shares the host's network namespace. No port mapping needed. | Performance-sensitive (avoids NAT overhead) |
| **none** | No network access at all | Security-sensitive workloads |

---

## 6. TLS: How HTTPS Actually Works

When you connect to `https://video-platform.com`, the traffic between your browser and the server is encrypted. This is done by **TLS** (Transport Layer Security), which wraps the TCP connection with encryption. Without TLS, anyone on the network path --- your ISP, the coffee shop WiFi, a compromised router --- can read your passwords, API keys, and user data in plaintext.

### The TLS Handshake

After the TCP three-way handshake establishes a connection, TLS adds its own handshake to negotiate encryption. In TLS 1.3 (the current standard), this adds just **one round-trip**:

**Step 1 --- Client Hello:** The browser sends its supported cipher suites (encryption algorithms), its supported TLS versions, and a random number.

**Step 2 --- Server Hello + Certificate:** The server chooses a cipher suite, sends its **certificate** (which contains the server's public key and is signed by a trusted Certificate Authority), and sends its key exchange parameters.

**Step 3 --- Client Verification + Finished:** The browser verifies the certificate (is it signed by a trusted CA? Is the domain name correct? Is it expired?), completes the key exchange, and both sides derive the shared encryption key.

After this handshake, all data flows over the encrypted connection. The total overhead is one RTT for TLS 1.3 (two RTT for TLS 1.2).

### Certificates

A **certificate** is a digital document that binds a domain name to a public key. It is signed by a **Certificate Authority** (CA) that your browser trusts. The chain of trust works like this:

1. Your server has a certificate for `video-platform.com`, signed by Let's Encrypt.
2. Your browser has Let's Encrypt's root certificate pre-installed (shipped with the browser/OS).
3. The browser can verify the chain: the server's certificate was signed by Let's Encrypt, and Let's Encrypt's root certificate is trusted.

If any link in this chain fails --- the certificate is expired, the domain name does not match, the CA is not trusted --- the browser shows a security warning.

### Let's Encrypt: Free Certificates

[Let's Encrypt](https://letsencrypt.org/) provides free, automated TLS certificates. The standard tool is **Certbot**:

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get a certificate (Nginx will be automatically configured)
sudo certbot --nginx -d video-platform.com -d www.video-platform.com

# Certificates are valid for 90 days. Auto-renew:
sudo certbot renew --dry-run

# The renewal timer is set up automatically:
systemctl status certbot.timer
```

### Why Your Reverse Proxy Should Terminate TLS

TLS termination means: the reverse proxy (Nginx) handles the encryption/decryption, and the connection between Nginx and your application is unencrypted HTTP over the local network.

Why this is the correct architecture:

- **Performance:** TLS encryption/decryption is CPU-intensive. Nginx is optimized for it. Your application should spend its CPU on business logic.
- **Simplicity:** Your application code does not need to manage certificates, handle TLS configuration, or deal with cipher suites.
- **Certificate management:** Certbot integrates with Nginx. Renewal is automatic.
- **Security:** The unencrypted traffic between Nginx and your app never leaves the machine (it is on `127.0.0.1`).

### Common TLS Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `NET::ERR_CERT_DATE_INVALID` | Certificate expired | Renew with `certbot renew` |
| `NET::ERR_CERT_COMMON_NAME_INVALID` | Certificate is for wrong domain | Re-issue for correct domain |
| `SSL_ERROR_RX_RECORD_TOO_LONG` | Connecting to HTTP port with HTTPS | Check port configuration |
| `ERR_SSL_PROTOCOL_ERROR` | TLS misconfiguration | Check `ssl_protocols` in Nginx |
| Mixed content warnings | Page loads HTTP resources on HTTPS page | Fix all resource URLs to use HTTPS |

---

## 7. Reverse Proxies: Why Nginx Sits in Front of Your App

A **reverse proxy** is a server that sits between clients and your application. It receives requests from the internet and forwards them to your application on a private port. Every production web deployment should use one.

### Why Not Expose Your Application Directly?

You could bind your Node.js application to port 443 and let it handle TLS, serve static files, and accept connections directly from the internet. Here is why you should not:

| Concern | Direct Exposure | Behind Nginx |
|---------|----------------|-------------|
| TLS termination | Your app manages certificates and TLS config | Nginx handles it (optimized, automatic renewal) |
| Static files | Node.js serves every CSS/JS/image file (slow) | Nginx serves them directly from disk (fast) |
| Connection buffering | Slow clients tie up your app's event loop | Nginx buffers the full request, forwards it instantly |
| Rate limiting | You implement it yourself | Nginx `limit_req` module |
| Load balancing | Not possible with one process | Nginx distributes across multiple app instances |
| Security | Your app is directly on the internet | Nginx shields your app; only port 443 is exposed |

### The Configuration

Here is a production Nginx configuration for an AI video platform:

```nginx
# /etc/nginx/sites-available/video-platform
server {
    listen 80;
    server_name video-platform.com www.video-platform.com;

    # Redirect all HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name video-platform.com www.video-platform.com;

    # TLS certificates (managed by certbot)
    ssl_certificate /etc/letsencrypt/live/video-platform.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/video-platform.com/privkey.pem;

    # TLS configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # API requests → Node.js
    location /api/ {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 10s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;  # Long for video generation
    }

    # WebSocket connections → Node.js
    location /ws {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # WebSocket timeout (keep connection alive)
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # Static files → serve directly from disk
    location /static/ {
        alias /opt/video-platform/public/static/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Health check (bypass the app for load balancer checks)
    location /health {
        proxy_pass http://127.0.0.1:3000;
        access_log off;  # Don't spam logs with health checks
    }
}
```

### The Key Headers Explained

**`proxy_set_header Host $host`:** The `Host` header tells your application which domain was requested. Without this, your app sees `127.0.0.1` as the host instead of `video-platform.com`.

**`proxy_set_header X-Real-IP $remote_addr`:** Since your app receives connections from Nginx (127.0.0.1), it cannot see the real client IP. This header passes the client's actual IP address.

**`proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for`:** A chain of all proxy IPs the request passed through. Your app should use this for logging and rate limiting.

**`proxy_set_header X-Forwarded-Proto $scheme`:** Tells your app whether the original request was HTTP or HTTPS. Important for generating correct redirect URLs.

### WebSocket Proxying

WebSocket connections require special configuration because they "upgrade" the HTTP connection to a persistent bidirectional channel. The `Upgrade` and `Connection` headers tell Nginx to switch protocols:

```nginx
proxy_http_version 1.1;                    # WebSocket requires HTTP/1.1
proxy_set_header Upgrade $http_upgrade;     # Pass the Upgrade header
proxy_set_header Connection "upgrade";      # Tell Nginx to upgrade
```

Without these lines, WebSocket connections fail with a 400 or 502 error because Nginx treats them as regular HTTP requests.

### Testing the Configuration

```bash
# Check for syntax errors
sudo nginx -t
# nginx: the configuration file /etc/nginx/nginx.conf syntax is ok
# nginx: configuration file /etc/nginx/nginx.conf test is successful

# Reload without downtime
sudo nginx -s reload
```

---

## 8. Firewalls: Controlling What Gets In

A **firewall** filters network traffic based on rules you define. Without a firewall, every service on your server is potentially accessible from the internet. Your PostgreSQL database on port 5432, your Redis on port 6379, your debug endpoints --- all exposed.

### UFW: The Uncomplicated Firewall

UFW is the standard firewall tool on Ubuntu. It wraps `iptables` (the kernel's packet filtering system) with a human-friendly interface:

```bash
# Start with deny-all
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow only the services you need
sudo ufw allow 22/tcp    # SSH (so you can still connect!)
sudo ufw allow 80/tcp    # HTTP (for certbot and redirects)
sudo ufw allow 443/tcp   # HTTPS (the only port clients need)

# Enable the firewall
sudo ufw enable

# Check status
sudo ufw status verbose
# Status: active
#
# To                         Action      From
# --                         ------      ----
# 22/tcp                     ALLOW       Anywhere
# 80/tcp                     ALLOW       Anywhere
# 443/tcp                    ALLOW       Anywhere
```

### The Common Mistake: Exposing Databases

This is a mistake made by approximately 100% of first-time deployers: exposing the database port to the internet.

```bash
# NEVER DO THIS
sudo ufw allow 5432/tcp   # PostgreSQL open to the world

# Attackers actively scan for open database ports.
# If your PostgreSQL has a weak password, you will be compromised.
```

Your database should only accept connections from `localhost` (your application server) or from specific private IPs (your other servers in a VPC). Configure this in `postgresql.conf`:

```
# PostgreSQL: only listen on localhost
listen_addresses = 'localhost'
```

And in `pg_hba.conf`:

```
# Only allow local connections
local   all   all                 peer
host    all   all   127.0.0.1/32  scram-sha-256
```

### Cloud Provider Security Groups

If you are on AWS, GCP, or Azure, the cloud provider has its own firewall layer called **security groups** (AWS/Azure) or **firewall rules** (GCP). These operate at the cloud network level, before traffic even reaches your VPS.

Best practice is to configure both: cloud security groups as the outer layer, and UFW as the inner layer. Defense in depth.

```bash
# AWS CLI example: allow only HTTP/HTTPS and SSH
aws ec2 authorize-security-group-ingress \
  --group-id sg-12345 \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
  --group-id sg-12345 \
  --protocol tcp --port 22 --cidr YOUR_IP/32  # Restrict SSH to your IP
```

---

## 9. Common Networking Failures and How to Debug Them

When a network connection fails, the error message usually tells you the symptom, not the cause. Here is a field guide to the most common failures and how to diagnose them.

### ERR_CONNECTION_REFUSED

**What it means:** The TCP SYN packet reached the server, but nothing is listening on that port. The server responded with a TCP RST (reset).

**Causes:**
- Your application is not running
- Your application is listening on a different port
- Your application is listening on `127.0.0.1` (not reachable from outside)
- A firewall is rejecting (not dropping) the connection

**Debug:**

```bash
# Is anything listening on the expected port?
ss -tlnp | grep :443

# Is the application running?
systemctl status video-platform

# What address is it binding to?
ss -tlnp | grep :3000
# Look for 0.0.0.0:3000 (good) vs 127.0.0.1:3000 (local only)
```

### ERR_CONNECTION_TIMED_OUT

**What it means:** The TCP SYN packet was sent, but no response came back. The browser waited (typically 30--60 seconds) and gave up.

**Causes:**
- A firewall is silently **dropping** packets (not rejecting them)
- Wrong IP address (the server is not at that IP)
- Network routing issue between client and server

**Debug:**

```bash
# Can you reach the server at all?
ping 203.0.113.42

# Is the firewall dropping packets?
sudo ufw status

# Trace the network path
traceroute 203.0.113.42
```

### ERR_NAME_NOT_RESOLVED

**What it means:** DNS lookup failed. The browser cannot translate the hostname to an IP address.

**Causes:**
- The domain does not exist
- DNS record is not configured
- Your DNS resolver is down

**Debug:**

```bash
# Does the domain resolve?
dig video-platform.com A +short

# Try a different resolver
dig @1.1.1.1 video-platform.com A +short

# Check nameserver configuration
dig video-platform.com NS +short
```

### 502 Bad Gateway

**What it means:** Nginx received the request but could not connect to the upstream application.

**Causes:**
- Your application crashed
- Your application is listening on a different port than Nginx expects
- The socket file does not exist (if using Unix sockets)

**Debug:**

```bash
# Is the app running?
systemctl status video-platform

# Is it listening where Nginx expects?
ss -tlnp | grep :3000

# Check Nginx error log
tail -20 /var/log/nginx/error.log
# Look for: "connect() failed (111: Connection refused)"
```

### 504 Gateway Timeout

**What it means:** Nginx connected to your application, but the application took too long to respond (exceeded `proxy_read_timeout`).

**Causes:**
- Your application is overloaded
- A database query is running too long
- An external API call (like a video generation API) is slow
- The event loop is blocked

**Debug:**

```bash
# Check application logs for slow operations
journalctl -u video-platform --since "5 minutes ago" | grep -i "timeout\|slow"

# Check system load
htop

# Check database connection
psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"
```

### Connection Reset by Peer

**What it means:** The server abruptly closed the connection mid-transfer.

**Causes:**
- The application crashed during request processing
- A proxy or load balancer terminated the connection
- Network equipment (firewall, NAT) timed out an idle connection

**Debug:**

```bash
# Check for application crashes
journalctl -u video-platform -p err --since "10 minutes ago"

# Check if the process restarted
systemctl status video-platform  # Look at uptime
```

### The Debugging Toolkit Summary

```bash
# DNS
dig video-platform.com A +short               # Does the domain resolve?

# Connectivity
curl -vvv https://video-platform.com           # Verbose connection info
curl -w "%{http_code} %{time_total}s\n" -o /dev/null -s URL  # Quick status check

# Listening services
ss -tlnp                                       # What is listening on which ports?

# Firewall
sudo ufw status                                # Is the firewall allowing traffic?

# TLS
openssl s_client -connect video-platform.com:443  # TLS certificate details

# Network path
traceroute video-platform.com                  # Route from here to there

# Packet capture (advanced)
sudo tcpdump -i eth0 port 443 -n -c 20        # See raw packets

# Nginx
sudo nginx -t                                  # Config syntax check
tail -50 /var/log/nginx/error.log              # Recent errors
```

---

## 10. The Networking Debug Checklist

When your deploy is not working, run through this checklist in order. The first failing step is usually your problem.

1. **Verify DNS resolves correctly.** `dig your-domain.com A +short` should return your server's IP address.

2. **Verify the application is running.** `systemctl status your-service` should show `active (running)`.

3. **Verify the application is listening on `0.0.0.0`, not `127.0.0.1`.** `ss -tlnp | grep :3000` should show `0.0.0.0:3000`, not `127.0.0.1:3000`.

4. **Verify the correct port.** `ss -tlnp` should show your application on the expected port and Nginx on ports 80 and 443.

5. **Verify the firewall allows traffic.** `sudo ufw status` should show ports 22, 80, and 443 allowed.

6. **Verify TLS certificates are valid.** `openssl s_client -connect your-domain.com:443` should show a valid certificate chain and no errors.

7. **Verify Nginx configuration.** `sudo nginx -t` should report syntax ok. Check `proxy_pass` points to the correct port.

8. **Verify Docker port mapping.** If using Docker, `docker ps` should show the correct port mapping (e.g., `0.0.0.0:3000->3000/tcp`).

9. **Test from outside the server.** `curl https://your-domain.com` from your laptop, not from the server. This catches 127.0.0.1 binding issues.

10. **Check for port conflicts.** `lsof -i :3000` should show exactly one process. If two processes are competing for the same port, one of them will fail silently.

11. **Check TIME_WAIT accumulation.** `ss -tan state time-wait | wc -l` should not be in the tens of thousands.

12. **Check Nginx error logs.** `tail -50 /var/log/nginx/error.log` often contains the exact error message that explains a 502 or 504.

---

## 11. Series Navigation

This article is Part 7 of the series on taking vibe-coded AI projects to production.

| Part | Title | Focus |
|------|-------|-------|
| 1 | [Performance Engineering](/2026/03/02/vibe-code-to-production-performance-engineering.html) | Profiling, N+1 queries, caching, async I/O |
| 2 | [Containerizing & Deploying](/2026/03/03/containerizing-deploying-ai-video-platform.html) | Docker, Nginx, TLS, CI/CD |
| 3 | [Load Testing](/2026/03/04/load-testing-breaking-video-pipeline.html) | k6, stress/soak/spike testing, SLOs |
| 4 | [Observability](/2026/03/05/observability-failure-modes-production-ai.html) | Logging, Prometheus, Grafana, OpenTelemetry |
| 5 | [How Your Computer Runs Your Code](/2026/03/06/how-computers-run-your-code.html) | CPU caches, memory layout, branch prediction |
| 6 | [Linux for the 2 AM Incident](/2026/03/07/linux-for-the-2am-incident.html) | Processes, file descriptors, signals, systemd |
| **7** | **Networking from Packet to Page Load** (this post) | **DNS, TCP, TLS, reverse proxies, firewalls** |

Parts 1--4 tell you **what to do** in production. Parts 5--7 explain **why it works** --- the foundational systems knowledge that makes the practices in Parts 1--4 make sense.

---

The deploy from the opening story broke because of one number: `127.0.0.1` instead of `0.0.0.0`. That one number meant the application was listening only for connections from itself, while every connection from the outside world arrived on a different network interface and was refused.

Three hours of debugging reduced to a one-character fix. But those three hours were not wasted on a hard problem --- they were wasted on a knowledge gap. The developer did not know what a network interface was, what a binding address means, or how a request travels from browser to server.

Networking is not optional knowledge for anyone deploying software to production. Every request your users make traverses DNS, TCP, TLS, a reverse proxy, and a firewall. Each of those is a link in a chain, and each can break independently. Understanding the chain --- the full journey from packet to page load --- is the difference between a three-hour debugging session and a five-minute fix.

You now understand the chain. Use the debug checklist. Check your binding addresses. Verify your DNS. Test from outside the server. And never ship a deploy you have not tested from a network that is not your server.
