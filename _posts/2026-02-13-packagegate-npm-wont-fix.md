---
layout: post
title: "PackageGate: npm Won't Fix Its Zero-Days"
date: 2026-02-13
category: security
urgency: critical
score: 92
type: cluster
---

In January 2026, security researchers at Koi disclosed six zero-day vulnerabilities across every major JavaScript package manager — npm, pnpm, vlt, and Bun. The collective name: PackageGate. The core finding is devastating: every recommended defense adopted after the 2025 Shai-Hulud supply chain attack can be bypassed. Disabling lifecycle scripts doesn't protect you. Lockfiles don't protect you. The trust model is broken at the protocol level.

pnpm, vlt, and Bun all shipped patches within weeks. npm — now under Microsoft — closed the report as "Informative" and left the vulnerability unpatched.

## The Vulnerabilities

Each package manager has its own variant, but the root cause is the same: Git-based dependencies bypass the security controls that were supposed to prevent arbitrary code execution during `npm install`.

**npm (UNPATCHED):** A malicious package can include a `.npmrc` file that overrides the `git` binary path. When npm resolves a Git dependency, it executes whatever binary `.npmrc` points to — even with `--ignore-scripts` enabled. This means a dependency you install from a Git URL can run arbitrary code on your machine without triggering any of npm's script-blocking protections.

**pnpm (CVE-2025-69263, CVE-2025-69264, PATCHED):** Git dependencies bypass the security allowlist and unconditionally run `prepare` and `prepublish` scripts. The allowlist — pnpm's defense against malicious lifecycle scripts — simply doesn't apply to Git-sourced packages.

**vlt (PATCHED):** Path traversal in tarball extraction allows writing files to arbitrary locations on the filesystem. A malicious package can escape its extraction directory and overwrite any file the current user has write access to.

**Bun (PATCHED):** The TrustedDependencies allowlist validates package names but not sources. An attacker can publish a package with the same name as a trusted dependency from a different registry and bypass the allowlist entirely.

## Why This Matters More Than a Typical CVE

Supply chain attacks aren't theoretical. The 2025 Shai-Hulud attack proved that. After Shai-Hulud, the JavaScript community adopted three key defenses:

1. Disable lifecycle scripts (`--ignore-scripts`)
2. Use lockfiles to pin dependency versions
3. Audit dependencies before installation

PackageGate demonstrates that all three defenses have holes. `--ignore-scripts` doesn't block `.npmrc` binary path overrides in npm. Lockfiles don't prevent the script execution — they just pin versions. And auditing doesn't help when the attack vector is in the package manager's resolution logic, not in the package's code.

The attack surface is specifically Git-based dependencies — packages installed from URLs like `git+https://github.com/org/repo.git` rather than from the npm registry. This pattern is common for:

- Forked packages with custom patches
- Private packages in monorepos
- Pre-release versions before they're published to a registry
- Internal tooling shared across projects

If your `package.json` or any transitive dependency uses a Git URL, you're potentially exposed.

## npm's Response (Or Lack Thereof)

This is the part that should alarm you. When Koi disclosed the npm vulnerability through responsible disclosure channels, npm's security team classified it as "Informative" — meaning they acknowledged the report but don't consider it a vulnerability that requires a fix.

Their reasoning, as best as can be inferred: `.npmrc` file handling is "working as designed," and Git dependencies have always been a trust boundary that users opt into explicitly.

The counterargument is straightforward: when npm added `--ignore-scripts` as a security feature and recommended it as a defense against supply chain attacks, they created an implicit promise that it would actually block script execution. A bypass via `.npmrc` binary path override breaks that promise. Users who followed npm's own security guidance are vulnerable.

pnpm, vlt, and Bun all treated their equivalent bugs as genuine security issues and shipped fixes. npm's refusal to patch puts it in a uniquely bad position among JavaScript package managers.

## What You Should Do

**Audit your dependencies now:**

```bash
# Find all Git-based dependencies in your project
grep -r "git+" package.json
grep -r "github:" package.json

# Check your lockfile for Git-sourced packages
grep "git+" package-lock.json | head -20
```

**In your GitHub Actions and CI pipelines:**

- Replace Git-based dependencies with registry-published versions wherever possible
- If you must use Git dependencies, pin them to specific commit SHAs, not branches or tags
- Run `npm install` in a containerized environment with limited filesystem access
- Consider switching CI pipelines from npm to pnpm, which has patched this class of bug

**For your local development:**

- Don't run `npm install` on untrusted projects without reviewing `package.json` first
- Check for `.npmrc` files in dependency directories: `find node_modules -name ".npmrc" -not -path "*/node_modules/npm/*"`
- Consider migrating to pnpm for security-sensitive projects

**For your team:**

- Pin all GitHub Actions to commit SHAs, not tags (the `tj-actions/changed-files` compromise in January showed that popular actions are also supply chain attack vectors)
- If you maintain internal packages, publish them to a private registry (GitHub Packages, Verdaccio) instead of using Git URLs

## The Bigger Picture

JavaScript's package ecosystem was built on trust. npm's founding ethos was radical openness — anyone can publish anything, and the community self-regulates. That model worked when the ecosystem was small. At 2+ million packages with billions of weekly downloads, it's a supply chain attack waiting to happen — and it keeps happening.

The divergence in responses is telling. pnpm, vlt, and Bun — smaller, more agile projects — treated the disclosure seriously and shipped fixes. npm, now a Microsoft subsidiary managing the world's largest package registry, chose not to act. Whether this is bureaucratic inertia, a genuine philosophical disagreement about threat models, or something else, the practical result is the same: npm's security posture is weaker than its competitors for this specific class of attack.

The security community is increasingly recommending pnpm for teams that care about supply chain security. Between its stricter dependency resolution, content-addressable storage, and faster patching of security issues, it's becoming the responsible default. The performance benefits are a bonus.

The trust model for package management needs to evolve. Until it does, verify everything, trust nothing, and maybe stop using `git+https://` in your `package.json`.
