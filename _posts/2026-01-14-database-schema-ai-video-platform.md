---
layout: post
title: "Database Schema Design for AI Video Platforms: Firestore Document Models, Relational Alternatives, and Query Patterns"
date: 2026-01-14
category: architecture
---

Every AI video platform lives or dies by its data layer. You can have the most beautiful UI and the fastest generation pipeline in the world, but if your database schema cannot efficiently answer the question "show me all of this user's projects with their latest generation status and total cost," you are going to have a bad time. I have spent months iterating on schemas for AI video products, and the decisions you make at the data layer ripple through every feature you build afterward.

This is the post I wish I had when I started. We will design a complete database schema for an AI video platform, first in Firestore (because that is what most of us ship v1 on), then in PostgreSQL (because that is where many of us migrate parts of our data as we scale). We will cover every entity, every query pattern, every index, and every tradeoff.

---

## The Data Entities

Before we write a single line of schema, let us enumerate the core entities that any AI video platform needs to model. If you have built one of these products, this list will feel familiar. If you have not, study it carefully because missing an entity at the beginning means painful migrations later.

<svg viewBox="0 0 900 520" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif;">
  <defs>
    <marker id="arrowhead-er" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#333"/>
    </marker>
    <filter id="shadow-er" x="-5%" y="-5%" width="110%" height="110%">
      <feDropShadow dx="1" dy="1" stdDeviation="2" flood-color="#00000020"/>
    </filter>
  </defs>
  <!-- Title -->
  <text x="450" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#333">Core Entities — AI Video Platform</text>
  <!-- Users -->
  <rect x="30" y="60" width="160" height="100" rx="8" fill="#4fc3f7" filter="url(#shadow-er)"/>
  <text x="110" y="90" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Users</text>
  <text x="110" y="110" text-anchor="middle" font-size="10" fill="#fff">uid, email, displayName</text>
  <text x="110" y="125" text-anchor="middle" font-size="10" fill="#fff">plan, credits, createdAt</text>
  <text x="110" y="140" text-anchor="middle" font-size="10" fill="#fff">stripeCustomerId</text>
  <!-- Projects -->
  <rect x="250" y="60" width="160" height="100" rx="8" fill="#ef5350" filter="url(#shadow-er)"/>
  <text x="330" y="90" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Projects</text>
  <text x="330" y="110" text-anchor="middle" font-size="10" fill="#fff">projectId, userId, title</text>
  <text x="330" y="125" text-anchor="middle" font-size="10" fill="#fff">status, sceneCount</text>
  <text x="330" y="140" text-anchor="middle" font-size="10" fill="#fff">createdAt, updatedAt</text>
  <!-- Scenes -->
  <rect x="470" y="60" width="160" height="100" rx="8" fill="#8bc34a" filter="url(#shadow-er)"/>
  <text x="550" y="90" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Scenes</text>
  <text x="550" y="110" text-anchor="middle" font-size="10" fill="#fff">sceneId, projectId, order</text>
  <text x="550" y="125" text-anchor="middle" font-size="10" fill="#fff">prompt, duration, style</text>
  <text x="550" y="140" text-anchor="middle" font-size="10" fill="#fff">voiceoverText</text>
  <!-- Generations -->
  <rect x="690" y="60" width="160" height="100" rx="8" fill="#ffa726" filter="url(#shadow-er)"/>
  <text x="770" y="90" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Generations</text>
  <text x="770" y="110" text-anchor="middle" font-size="10" fill="#fff">genId, sceneId, model</text>
  <text x="770" y="125" text-anchor="middle" font-size="10" fill="#fff">status, cost, duration</text>
  <text x="770" y="140" text-anchor="middle" font-size="10" fill="#fff">startedAt, completedAt</text>
  <!-- Assets -->
  <rect x="30" y="240" width="160" height="100" rx="8" fill="#ffa726" filter="url(#shadow-er)"/>
  <text x="110" y="270" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Assets</text>
  <text x="110" y="290" text-anchor="middle" font-size="10" fill="#fff">assetId, genId, type</text>
  <text x="110" y="305" text-anchor="middle" font-size="10" fill="#fff">r2Key, size, mimeType</text>
  <text x="110" y="320" text-anchor="middle" font-size="10" fill="#fff">thumbnailKey, metadata</text>
  <!-- Billing -->
  <rect x="250" y="240" width="160" height="100" rx="8" fill="#4fc3f7" filter="url(#shadow-er)"/>
  <text x="330" y="270" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Billing</text>
  <text x="330" y="290" text-anchor="middle" font-size="10" fill="#fff">invoiceId, userId</text>
  <text x="330" y="305" text-anchor="middle" font-size="10" fill="#fff">amount, credits, type</text>
  <text x="330" y="320" text-anchor="middle" font-size="10" fill="#fff">stripePaymentId, date</text>
  <!-- Analytics -->
  <rect x="470" y="240" width="160" height="100" rx="8" fill="#ef5350" filter="url(#shadow-er)"/>
  <text x="550" y="270" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">Analytics</text>
  <text x="550" y="290" text-anchor="middle" font-size="10" fill="#fff">eventId, userId, type</text>
  <text x="550" y="305" text-anchor="middle" font-size="10" fill="#fff">metadata, timestamp</text>
  <text x="550" y="320" text-anchor="middle" font-size="10" fill="#fff">sessionId</text>
  <!-- ApiKeys -->
  <rect x="690" y="240" width="160" height="100" rx="8" fill="#8bc34a" filter="url(#shadow-er)"/>
  <text x="770" y="270" text-anchor="middle" font-size="14" font-weight="bold" fill="#fff">API Keys</text>
  <text x="770" y="290" text-anchor="middle" font-size="10" fill="#fff">keyId, userId, name</text>
  <text x="770" y="305" text-anchor="middle" font-size="10" fill="#fff">hashedKey, permissions</text>
  <text x="770" y="320" text-anchor="middle" font-size="10" fill="#fff">rateLimit, lastUsed</text>
  <!-- Relationship lines -->
  <line x1="190" y1="110" x2="250" y2="110" stroke="#333" stroke-width="1.5" marker-end="url(#arrowhead-er)"/>
  <line x1="410" y1="110" x2="470" y2="110" stroke="#333" stroke-width="1.5" marker-end="url(#arrowhead-er)"/>
  <line x1="630" y1="110" x2="690" y2="110" stroke="#333" stroke-width="1.5" marker-end="url(#arrowhead-er)"/>
  <line x1="770" y1="160" x2="770" y2="185" stroke="#333" stroke-width="1.5"/>
  <line x1="770" y1="185" x2="110" y2="185" stroke="#333" stroke-width="1.5"/>
  <line x1="110" y1="185" x2="110" y2="240" stroke="#333" stroke-width="1.5" marker-end="url(#arrowhead-er)"/>
  <line x1="110" y1="160" x2="110" y2="185" stroke="#333" stroke-width="1" stroke-dasharray="4"/>
  <line x1="110" y1="185" x2="330" y2="185" stroke="#333" stroke-width="1" stroke-dasharray="4"/>
  <line x1="330" y1="185" x2="330" y2="240" stroke="#333" stroke-width="1.5" marker-end="url(#arrowhead-er)"/>
  <!-- Cardinality labels -->
  <text x="210" y="103" font-size="9" fill="#666">1:N</text>
  <text x="430" y="103" font-size="9" fill="#666">1:N</text>
  <text x="650" y="103" font-size="9" fill="#666">1:N</text>
  <text x="100" y="200" font-size="9" fill="#666">1:N</text>
  <text x="315" y="200" font-size="9" fill="#666">1:N</text>
  <!-- Legend -->
  <rect x="30" y="400" width="840" height="100" rx="6" fill="#f8f9fa" stroke="#ddd"/>
  <text x="50" y="425" font-size="13" font-weight="bold" fill="#333">Cardinality Summary</text>
  <text x="50" y="450" font-size="11" fill="#555">User → Projects (1:N) | Project → Scenes (1:N) | Scene → Generations (1:N)</text>
  <text x="50" y="470" font-size="11" fill="#555">Generation → Assets (1:N) | User → Billing (1:N) | User → Analytics (1:N) | User → API Keys (1:N)</text>
  <text x="50" y="490" font-size="11" fill="#555">A single generation can produce multiple assets: the video file, a thumbnail, an audio track, and metadata JSON.</text>
</svg>

Here is the complete list with their purposes:

| Entity | Purpose | Write Frequency | Read Frequency | Relationships |
|--------|---------|----------------|----------------|---------------|
| **Users** | Authentication, profile, subscription state | Low (updates on plan change) | High (every authenticated request) | Root entity |
| **Projects** | Container for a video creation workflow | Medium (creation, updates) | High (dashboard, listing) | Belongs to User |
| **Scenes** | Individual segments within a project | Medium (editing workflow) | High (editor view) | Belongs to Project |
| **Generations** | Each attempt to generate video/image/audio | High (every generation request) | Medium (status polling, history) | Belongs to Scene |
| **Assets** | Binary file references (video, image, audio) | High (created with each generation) | High (playback, gallery, download) | Belongs to Generation |
| **Billing** | Invoices, credit transactions, payments | Low (on purchase/subscription events) | Low-Medium (billing page, admin) | Belongs to User |
| **Analytics** | Usage events, page views, feature usage | Very High (every user action) | Low (dashboard queries, batch) | Belongs to User |
| **API Keys** | Developer access tokens | Very Low | Medium (validated on each API call) | Belongs to User |

The read-to-write ratio is the most important thing to understand here. AI video platforms are **overwhelmingly read-heavy**. The average user spends far more time browsing their gallery, reviewing generations, and previewing scenes than they do triggering new generations. This read-heavy pattern drives almost every schema decision we will make.

---

## Firestore Document Model: Complete Structure

Firestore is a document database. There are no tables, no rows, no columns. There are **collections** (analogous to tables), **documents** (analogous to rows), and **fields** (analogous to columns). The critical difference is that documents can contain **subcollections**, creating a tree structure that relational databases cannot natively express.

### Users Collection

```javascript
// Collection: users
// Document ID: Firebase Auth UID (e.g., "abc123def456")
{
  // Identity
  uid: "abc123def456",
  email: "user@example.com",
  displayName: "Jane Creator",
  photoURL: "https://lh3.googleusercontent.com/...",
  authProvider: "google", // "google" | "email" | "github"

  // Subscription state (denormalized from Stripe)
  plan: "pro",           // "free" | "pro" | "enterprise"
  credits: 847,          // Current credit balance
  creditsMonthly: 1000,  // Monthly credit allocation
  creditsResetAt: Timestamp("2026-02-01T00:00:00Z"),

  // Stripe integration
  stripeCustomerId: "cus_R1a2b3c4d5",
  stripeSubscriptionId: "sub_X9y8z7w6v5",
  stripePriceId: "price_pro_monthly_v2",

  // Usage counters (denormalized for fast reads)
  totalGenerations: 342,
  totalProjects: 12,
  totalStorageBytes: 5368709120, // 5 GB

  // Preferences
  defaultModel: "kling-v2",
  defaultResolution: "1080p",
  defaultAspectRatio: "16:9",
  emailNotifications: true,

  // Timestamps
  createdAt: Timestamp("2025-08-15T10:30:00Z"),
  updatedAt: Timestamp("2026-01-14T08:22:00Z"),
  lastLoginAt: Timestamp("2026-01-14T08:22:00Z"),

  // Feature flags / A/B test assignments
  features: {
    newEditor: true,
    pipelineV2: false,
    betaPricing: "variant_b"
  }
}
```

**Key design decisions:**

1. **Document ID is the Firebase Auth UID.** This eliminates a lookup. When a user authenticates, you have their UID immediately. `db.collection('users').doc(uid)` is O(1).

2. **Stripe data is denormalized onto the user document.** The canonical source of truth for subscription state is Stripe, but we cache the current plan, credits, and subscription IDs on the user document. This means every authenticated request can check the user's plan without calling the Stripe API. We keep this in sync via Stripe webhooks (`customer.subscription.updated`, `invoice.paid`, etc.).

3. **Usage counters are denormalized.** `totalGenerations`, `totalProjects`, and `totalStorageBytes` are incremented atomically using `FieldValue.increment()`. This avoids expensive aggregation queries. The downside is the counters can drift if increment operations fail, but for UI display this level of accuracy is fine. For billing-critical data, we always recompute from source.

4. **Feature flags live on the user document.** This lets you check feature assignments without a separate lookup. When you assign a user to an A/B test, you write directly to their document.

### Projects Collection

```javascript
// Collection: projects
// Document ID: auto-generated (e.g., "proj_Kx8mN2pQ4r")
{
  projectId: "proj_Kx8mN2pQ4r",
  userId: "abc123def456",       // Owner

  // Content
  title: "Product Demo Video",
  description: "30-second demo for the new feature launch",

  // Structure
  sceneCount: 5,
  totalDuration: 30.0,           // Seconds (sum of scene durations)

  // Status
  status: "editing",             // "editing" | "generating" | "complete" | "archived"

  // Rendering
  resolution: "1080p",
  aspectRatio: "16:9",
  fps: 24,

  // Denormalized for gallery view
  thumbnailUrl: "https://r2.example.com/thumbs/proj_Kx8mN2pQ4r.jpg",
  previewVideoUrl: "https://r2.example.com/previews/proj_Kx8mN2pQ4r.mp4",

  // Sharing
  isPublic: false,
  shareToken: null,              // Generated when shared

  // Cost tracking
  totalCreditsUsed: 45,
  totalApiCost: 2.37,           // USD, for internal tracking

  // Timestamps
  createdAt: Timestamp("2026-01-10T14:00:00Z"),
  updatedAt: Timestamp("2026-01-14T09:15:00Z"),

  // Tags for organization
  tags: ["product", "demo", "q1-launch"]
}
```

**Why projects are a flat collection and not a subcollection of users:**

This is the single most important schema decision for Firestore in this domain. You have two options:

- **Option A:** `users/{userId}/projects/{projectId}` (subcollection)
- **Option B:** `projects/{projectId}` with a `userId` field (flat collection)

I strongly recommend **Option B**. Here is why:

| Consideration | Subcollection (A) | Flat Collection (B) |
|---|---|---|
| Query "all projects for user X" | `users/X/projects` (natural) | `projects.where('userId', '==', X)` (requires index) |
| Query "all projects across all users" (admin) | **Impossible** without collection group query | `projects.where(...)` (natural) |
| Query "project by ID when you know the ID" | Need userId AND projectId | Just projectId |
| Share a project via URL | URL must encode userId (leaks info) | URL only needs projectId |
| Move project between users (team features) | Delete and recreate in new subcollection | Update `userId` field |
| Security rules | Path-based (`request.auth.uid == userId`) | Field-based (`resource.data.userId == request.auth.uid`) |

The flat collection wins on flexibility. The subcollection wins on security rule elegance, but that advantage is minor. The inability to easily query across users (for admin dashboards, public galleries, featured content) is a dealbreaker for subcollections.

### Scenes Collection

```javascript
// Collection: scenes
// Document ID: auto-generated (e.g., "scene_7Wn3xP9q2m")
{
  sceneId: "scene_7Wn3xP9q2m",
  projectId: "proj_Kx8mN2pQ4r",
  userId: "abc123def456",        // Denormalized for security rules

  // Ordering
  order: 2,                      // 0-indexed position in project

  // Content specification
  prompt: "A sleek laptop on a minimalist desk, camera slowly zooms in, soft natural lighting",
  negativePrompt: "blurry, low quality, watermark",
  style: "cinematic",           // "cinematic" | "anime" | "realistic" | "3d-render"

  // Timing
  duration: 5.0,                // Seconds
  transitionIn: "fade",         // "cut" | "fade" | "dissolve" | "wipe"
  transitionDuration: 0.5,

  // Reference media
  referenceImageUrl: "https://r2.example.com/refs/scene_7Wn3xP9q2m_ref.jpg",
  referenceImageR2Key: "refs/scene_7Wn3xP9q2m_ref.jpg",

  // Voiceover
  voiceoverText: "Introducing the next generation of creative tools.",
  voiceId: "eleven_rachel",
  voiceoverAssetId: "asset_Vz5tN8w2",  // Reference to generated audio

  // Music / Audio
  backgroundMusicTrack: "ambient-tech-01",
  backgroundMusicVolume: 0.3,

  // Generation settings
  model: "kling-v2",
  seed: 42,
  guidanceScale: 7.5,

  // Current best generation (denormalized)
  selectedGenerationId: "gen_Pq4rW8n3",
  selectedVideoUrl: "https://r2.example.com/videos/gen_Pq4rW8n3.mp4",
  selectedThumbnailUrl: "https://r2.example.com/thumbs/gen_Pq4rW8n3.jpg",

  // Timestamps
  createdAt: Timestamp("2026-01-10T14:05:00Z"),
  updatedAt: Timestamp("2026-01-14T09:10:00Z")
}
```

Scenes are also a flat collection. The `projectId` field lets us query all scenes for a project, while the `userId` field enables security rules without needing the project document. Denormalizing `userId` onto scenes saves a read: without it, every security rule check on a scene would require reading the parent project to verify ownership.

### Generations Collection

```javascript
// Collection: generations
// Document ID: auto-generated (e.g., "gen_Pq4rW8n3")
{
  generationId: "gen_Pq4rW8n3",
  sceneId: "scene_7Wn3xP9q2m",
  projectId: "proj_Kx8mN2pQ4r",  // Denormalized
  userId: "abc123def456",          // Denormalized

  // Generation type
  type: "video",                   // "video" | "image" | "audio" | "upscale"

  // Model and parameters
  model: "kling-v2",
  modelVersion: "2.1.3",
  prompt: "A sleek laptop on a minimalist desk...",  // Snapshot at generation time
  negativePrompt: "blurry, low quality, watermark",
  seed: 42,
  guidanceScale: 7.5,
  duration: 5.0,
  resolution: "1080p",
  aspectRatio: "16:9",
  fps: 24,

  // Status tracking
  status: "completed",  // "queued" | "processing" | "completed" | "failed" | "cancelled"
  progress: 100,        // 0-100 percentage
  error: null,          // Error message if failed
  retryCount: 0,

  // External API tracking
  externalJobId: "kling_job_abc123",  // ID from the generation API
  externalStatus: "success",
  callbackReceived: true,

  // Quality assessment (from Gemini Flash QA)
  qualityScore: 0.87,               // 0-1 score
  qualityFlags: ["good_motion", "stable_subject"],
  qualityNotes: "Smooth camera movement, consistent lighting",

  // Cost tracking
  creditsCost: 10,
  apiCostUsd: 0.52,                 // Actual cost from provider

  // Asset references
  outputAssetId: "asset_M3nR7w9p",
  outputVideoUrl: "https://r2.example.com/videos/gen_Pq4rW8n3.mp4",
  outputThumbnailUrl: "https://r2.example.com/thumbs/gen_Pq4rW8n3.jpg",

  // Timing
  queuedAt: Timestamp("2026-01-14T09:00:00Z"),
  startedAt: Timestamp("2026-01-14T09:00:05Z"),
  completedAt: Timestamp("2026-01-14T09:02:30Z"),
  processingDurationMs: 145000,      // 2 min 25 sec

  // Metadata
  createdAt: Timestamp("2026-01-14T09:00:00Z")
}
```

**Why we denormalize `projectId` and `userId` onto every generation:**

Generations are the entity you query most aggressively. Consider these queries:

- "Show me all generations for this scene" (editor view)
- "Show me all generations for this project" (project overview)
- "Show me all generations for this user" (usage dashboard)
- "Show me all failed generations in the last hour" (admin monitoring)
- "Show me this user's total API spend this month" (billing)

Without denormalization, queries 2 through 5 would each require reading the parent scene to get the projectId, then the parent project to get the userId. That is 2 extra reads per generation per query. If a user has 500 generations, that is 1,000 extra reads. At Firestore's pricing of $0.06 per 100K reads, this adds up fast and also adds latency.

### Assets Collection

```javascript
// Collection: assets
// Document ID: auto-generated (e.g., "asset_M3nR7w9p")
{
  assetId: "asset_M3nR7w9p",
  generationId: "gen_Pq4rW8n3",
  sceneId: "scene_7Wn3xP9q2m",    // Denormalized
  projectId: "proj_Kx8mN2pQ4r",   // Denormalized
  userId: "abc123def456",           // Denormalized

  // Asset type
  type: "video",                    // "video" | "image" | "audio" | "thumbnail" | "metadata"

  // Storage
  storageProvider: "r2",            // "r2" | "gcs" | "s3"
  r2Key: "videos/gen_Pq4rW8n3/output.mp4",
  r2Bucket: "ai-video-assets",
  publicUrl: "https://r2.example.com/videos/gen_Pq4rW8n3/output.mp4",
  cdnUrl: "https://cdn.example.com/videos/gen_Pq4rW8n3/output.mp4",

  // File metadata
  mimeType: "video/mp4",
  sizeBytes: 15728640,              // 15 MB
  durationMs: 5000,                 // For video/audio
  width: 1920,                      // For video/image
  height: 1080,
  codec: "h264",
  bitrate: 25165824,                // bps

  // Thumbnail (for video assets)
  thumbnailR2Key: "thumbs/gen_Pq4rW8n3.jpg",
  thumbnailUrl: "https://cdn.example.com/thumbs/gen_Pq4rW8n3.jpg",

  // Status
  status: "available",              // "uploading" | "processing" | "available" | "deleted"

  // Lifecycle
  expiresAt: null,                  // Null = permanent, or Timestamp for temp assets
  deletedAt: null,

  createdAt: Timestamp("2026-01-14T09:02:30Z")
}
```

### Billing Collection

```javascript
// Collection: billing_events
// Document ID: auto-generated
{
  eventId: "bill_Qw3rT7y9",
  userId: "abc123def456",

  // Event type
  type: "credit_purchase",  // "subscription_payment" | "credit_purchase" | "credit_usage"
                            // | "refund" | "credit_expiry" | "credit_grant"

  // Financial
  amount: 999,              // In cents (USD)
  currency: "usd",
  credits: 100,             // Credits involved
  creditsBefore: 747,       // Balance before event
  creditsAfter: 847,        // Balance after event

  // Stripe reference
  stripePaymentIntentId: "pi_3Ox1234567890",
  stripeInvoiceId: "in_1Ox9876543210",

  // Context
  description: "100 credit pack purchase",
  generationId: null,       // Set for credit_usage events
  projectId: null,

  // Timestamps
  createdAt: Timestamp("2026-01-14T08:00:00Z")
}
```

### Analytics Events Collection

```javascript
// Collection: analytics_events
// Document ID: auto-generated
{
  eventId: "evt_Nn8mK3p7",
  userId: "abc123def456",       // Null for anonymous
  sessionId: "sess_Ww2xY5z1",

  // Event classification
  type: "generation_completed",  // See event taxonomy below
  category: "generation",       // "navigation" | "generation" | "editing" | "billing" | "auth"

  // Event data (flexible schema)
  data: {
    model: "kling-v2",
    duration: 5.0,
    resolution: "1080p",
    processingTimeMs: 145000,
    creditsCost: 10,
    qualityScore: 0.87
  },

  // Context
  source: "web",                // "web" | "api" | "mobile"
  userAgent: "Mozilla/5.0...",
  ipCountry: "US",

  createdAt: Timestamp("2026-01-14T09:02:30Z")
}
```

---

## Firestore Document Hierarchy

Let me visualize the complete document hierarchy, showing how collections and documents relate:

<svg viewBox="0 0 860 680" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif;">
  <defs>
    <marker id="arrow-h" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#555"/>
    </marker>
  </defs>
  <text x="430" y="28" text-anchor="middle" font-size="17" font-weight="bold" fill="#333">Firestore Document Hierarchy — Flat Collection Strategy</text>
  <!-- Root -->
  <rect x="340" y="50" width="180" height="36" rx="6" fill="#333"/>
  <text x="430" y="73" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold">Firestore Root</text>
  <!-- Level 1: Collections -->
  <!-- users -->
  <rect x="20" y="130" width="130" height="32" rx="4" fill="#4fc3f7"/>
  <text x="85" y="151" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">/users</text>
  <line x1="370" y1="86" x2="85" y2="130" stroke="#555" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- projects -->
  <rect x="170" y="130" width="130" height="32" rx="4" fill="#ef5350"/>
  <text x="235" y="151" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">/projects</text>
  <line x1="400" y1="86" x2="235" y2="130" stroke="#555" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- scenes -->
  <rect x="320" y="130" width="130" height="32" rx="4" fill="#8bc34a"/>
  <text x="385" y="151" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">/scenes</text>
  <line x1="430" y1="86" x2="385" y2="130" stroke="#555" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- generations -->
  <rect x="470" y="130" width="130" height="32" rx="4" fill="#ffa726"/>
  <text x="535" y="151" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">/generations</text>
  <line x1="460" y1="86" x2="535" y2="130" stroke="#555" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- assets -->
  <rect x="620" y="130" width="110" height="32" rx="4" fill="#4fc3f7"/>
  <text x="675" y="151" text-anchor="middle" font-size="12" fill="#fff" font-weight="bold">/assets</text>
  <line x1="490" y1="86" x2="675" y2="130" stroke="#555" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- billing_events -->
  <rect x="750" y="130" width="95" height="32" rx="4" fill="#ef5350"/>
  <text x="797" y="151" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">/billing</text>
  <line x1="510" y1="86" x2="797" y2="130" stroke="#555" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- Document examples -->
  <!-- User doc -->
  <rect x="10" y="195" width="150" height="110" rx="4" fill="#e1f5fe" stroke="#4fc3f7" stroke-width="1.5"/>
  <text x="85" y="215" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">user doc: abc123</text>
  <text x="20" y="235" font-size="9" fill="#555">email: "jane@example.com"</text>
  <text x="20" y="250" font-size="9" fill="#555">plan: "pro"</text>
  <text x="20" y="265" font-size="9" fill="#555">credits: 847</text>
  <text x="20" y="280" font-size="9" fill="#555">totalGenerations: 342</text>
  <text x="20" y="295" font-size="9" fill="#555">stripeCustomerId: "cus_..."</text>
  <line x1="85" y1="162" x2="85" y2="195" stroke="#4fc3f7" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- Project doc -->
  <rect x="175" y="195" width="150" height="110" rx="4" fill="#ffebee" stroke="#ef5350" stroke-width="1.5"/>
  <text x="250" y="215" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">project doc: proj_Kx8m</text>
  <text x="185" y="235" font-size="9" fill="#555">userId: "abc123"</text>
  <text x="185" y="250" font-size="9" fill="#555">title: "Product Demo"</text>
  <text x="185" y="265" font-size="9" fill="#555">sceneCount: 5</text>
  <text x="185" y="280" font-size="9" fill="#555">status: "editing"</text>
  <text x="185" y="295" font-size="9" fill="#555">thumbnailUrl: "https://..."</text>
  <line x1="235" y1="162" x2="250" y2="195" stroke="#ef5350" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- Scene doc -->
  <rect x="340" y="195" width="150" height="110" rx="4" fill="#f1f8e9" stroke="#8bc34a" stroke-width="1.5"/>
  <text x="415" y="215" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">scene doc: scene_7Wn3</text>
  <text x="350" y="235" font-size="9" fill="#555">projectId: "proj_Kx8m"</text>
  <text x="350" y="250" font-size="9" fill="#555">userId: "abc123"</text>
  <text x="350" y="265" font-size="9" fill="#555">order: 2</text>
  <text x="350" y="280" font-size="9" fill="#555">prompt: "A sleek laptop..."</text>
  <text x="350" y="295" font-size="9" fill="#555">selectedVideoUrl: "..."</text>
  <line x1="385" y1="162" x2="415" y2="195" stroke="#8bc34a" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- Generation doc -->
  <rect x="505" y="195" width="150" height="110" rx="4" fill="#fff3e0" stroke="#ffa726" stroke-width="1.5"/>
  <text x="580" y="215" text-anchor="middle" font-size="11" font-weight="bold" fill="#333">gen doc: gen_Pq4r</text>
  <text x="515" y="235" font-size="9" fill="#555">sceneId: "scene_7Wn3"</text>
  <text x="515" y="250" font-size="9" fill="#555">projectId: "proj_Kx8m"</text>
  <text x="515" y="265" font-size="9" fill="#555">userId: "abc123"</text>
  <text x="515" y="280" font-size="9" fill="#555">model: "kling-v2"</text>
  <text x="515" y="295" font-size="9" fill="#555">status: "completed"</text>
  <line x1="535" y1="162" x2="580" y2="195" stroke="#ffa726" stroke-width="1.2" marker-end="url(#arrow-h)"/>
  <!-- FK arrows between docs -->
  <path d="M 250 305 L 250 340 L 415 340 L 415 305" fill="none" stroke="#888" stroke-width="1" stroke-dasharray="5,3"/>
  <text x="333" y="355" text-anchor="middle" font-size="9" fill="#888">projectId reference</text>
  <path d="M 415 305 L 415 370 L 580 370 L 580 305" fill="none" stroke="#888" stroke-width="1" stroke-dasharray="5,3"/>
  <text x="497" y="385" text-anchor="middle" font-size="9" fill="#888">sceneId reference</text>
  <path d="M 85 305 L 85 395 L 250 395 L 250 305" fill="none" stroke="#888" stroke-width="1" stroke-dasharray="5,3"/>
  <text x="167" y="410" text-anchor="middle" font-size="9" fill="#888">userId reference</text>
  <!-- Note box -->
  <rect x="30" y="440" width="800" height="220" rx="8" fill="#f8f9fa" stroke="#ddd"/>
  <text x="50" y="468" font-size="13" font-weight="bold" fill="#333">Flat Collection Strategy: Why Not Subcollections?</text>
  <text x="50" y="495" font-size="11" fill="#555">All entities are top-level collections with foreign-key-style references (userId, projectId, sceneId).</text>
  <text x="50" y="520" font-size="11" fill="#555">Advantages over nested subcollections:</text>
  <text x="70" y="543" font-size="11" fill="#555">1. Any entity is addressable by a single ID (no compound paths like /users/X/projects/Y/scenes/Z)</text>
  <text x="70" y="563" font-size="11" fill="#555">2. Cross-user queries work natively (admin dashboards, public galleries, analytics)</text>
  <text x="70" y="583" font-size="11" fill="#555">3. Entity transfer between owners requires updating one field, not moving subcollections</text>
  <text x="70" y="603" font-size="11" fill="#555">4. Simplified security rules: single field check vs. path parsing</text>
  <text x="50" y="633" font-size="11" fill="#555">Tradeoff: Requires composite indexes for queries like (userId + createdAt), whereas subcollections inherit</text>
  <text x="50" y="650" font-size="11" fill="#555">parent-level scoping automatically. At scale, the flat approach pays for itself.</text>
</svg>

---

## Denormalization Strategy

In a relational database, normalization is the default. You store each fact once and join to reconstruct it. In Firestore, there are no joins. Every query hits exactly one collection. This means data that would be "one join away" in SQL must instead be **duplicated** across documents.

This is not a bug. It is the fundamental design philosophy of document databases. But you need a disciplined strategy for what to denormalize, or you end up with an unmaintainable mess.

### The Denormalization Matrix

| Field | Canonical Location | Denormalized To | Reason | Update Strategy |
|---|---|---|---|---|
| `userId` | `users` | `projects`, `scenes`, `generations`, `assets`, `billing_events` | Security rules, query filtering | Never changes (immutable) |
| `projectId` | `projects` | `scenes`, `generations`, `assets` | Query without parent lookups | Never changes (immutable) |
| `sceneId` | `scenes` | `generations`, `assets` | Query without parent lookups | Never changes (immutable) |
| `user.plan` | `users` | None | Not frequently needed on other entities | N/A |
| `user.credits` | `users` | None | Changes too frequently to denormalize | N/A |
| `project.thumbnailUrl` | Latest generation's asset | `projects` | Gallery view needs thumbnail without reading generations | Updated when user selects a generation |
| `scene.selectedVideoUrl` | Selected generation's asset | `scenes` | Editor needs video URL without reading generations | Updated when user selects a generation |
| `generation.outputVideoUrl` | `assets` | `generations` | Generation list needs video URL without reading assets | Set once on generation completion |
| `project.sceneCount` | Count of scenes | `projects` | Dashboard shows scene count | Incremented/decremented atomically |
| `user.totalGenerations` | Count of generations | `users` | Profile shows total generations | Incremented atomically |

### Read Amplification Analysis

Let us calculate the read cost for common UI views under different schema designs.

**Scenario: User opens their project dashboard (shows 10 projects with thumbnails and scene counts)**

**Design A: Fully Normalized (no denormalization)**

```
1 read  - user document
10 reads - project documents
50 reads - scene documents (avg 5 per project, to count them)
50 reads - latest generation per scene (to get thumbnails)
---------
111 reads total
```

**Design B: Denormalized (our design)**

```
1 read  - user document
10 reads - project documents (contain thumbnailUrl, sceneCount)
---------
11 reads total
```

**Read amplification factor:** $\frac{111}{11} = 10.1\times$

At Firestore's pricing of $0.06 per 100,000 reads:

| Metric | Normalized | Denormalized | Savings |
|--------|-----------|--------------|---------|
| Reads per dashboard load | 111 | 11 | 90.1% |
| Cost per 1M dashboard loads | $66.60 | $6.60 | $60.00 |
| Latency (est. at 10ms/read) | ~200ms (batched) | ~20ms (batched) | 90% |

The cost difference is meaningful at scale, but the **latency** difference is what matters most. Users perceive the difference between 20ms and 200ms. That is the difference between "instant" and "noticeable."

### What NOT to Denormalize

Not everything should be duplicated. Here are the criteria for what to leave normalized:

1. **Data that changes frequently.** User credit balance changes on every generation. Denormalizing it onto every project would mean updating dozens of documents per generation.

2. **Data that is only needed in detail views.** The full generation parameters (prompt, seed, guidance scale) are only needed when viewing a specific generation. They do not need to be on the scene document.

3. **Data where consistency is critical.** Billing amounts and credit transactions should have a single source of truth. Denormalizing financial data creates reconciliation nightmares.

4. **Data that would exceed document size limits.** Firestore documents have a 1 MB limit. Do not store arrays that could grow unboundedly (like a list of all generation IDs on a project document).

---

## Index Design

Firestore automatically creates single-field indexes for every field in every document. But composite queries (filtering or sorting on multiple fields) require **composite indexes** that you must create manually.

### Required Composite Indexes

```
// projects collection
(userId ASC, createdAt DESC)          // "My projects, newest first"
(userId ASC, status ASC, updatedAt DESC)  // "My active projects, recently updated"
(userId ASC, tags ARRAY_CONTAINS, createdAt DESC)  // "My projects tagged 'demo'"
(isPublic ASC, createdAt DESC)        // "Public gallery, newest first"

// scenes collection
(projectId ASC, order ASC)            // "Scenes in project, in order"
(userId ASC, createdAt DESC)          // "All user's scenes" (rarely used but needed for admin)

// generations collection
(sceneId ASC, createdAt DESC)         // "Generations for scene, newest first"
(projectId ASC, createdAt DESC)       // "All generations in project"
(userId ASC, createdAt DESC)          // "User's generation history"
(userId ASC, status ASC, createdAt DESC)  // "User's pending generations"
(status ASC, createdAt ASC)           // "All pending generations" (worker queue)
(userId ASC, model ASC, createdAt DESC)   // "User's generations by model"

// assets collection
(generationId ASC, type ASC)          // "Assets for generation, by type"
(userId ASC, type ASC, createdAt DESC)    // "User's video assets"

// billing_events collection
(userId ASC, createdAt DESC)          // "User's billing history"
(userId ASC, type ASC, createdAt DESC)    // "User's credit purchases"

// analytics_events collection
(userId ASC, type ASC, createdAt DESC)    // "User's events by type"
(type ASC, createdAt DESC)            // "All events of type" (admin analytics)
```

That is **16 composite indexes** for the core schema. Firestore limits you to 200 composite indexes per database, so we are well within limits. But each index consumes storage and adds write latency (every write must update every relevant index).

### The Firestore Index Limitation

The fundamental limitation of Firestore indexing is that **inequality filters can only be applied to one field per query** (unless you use the newer Firestore Pipeline operations or the `where()` chain with `and/or` composites introduced in later SDK versions). This shapes query design profoundly.

Consider this query: "Show me all generations for user X that used model Y, completed successfully, in the last 7 days, sorted by quality score."

In SQL:
```sql
SELECT * FROM generations
WHERE user_id = 'abc123'
  AND model = 'kling-v2'
  AND status = 'completed'
  AND created_at > NOW() - INTERVAL '7 days'
ORDER BY quality_score DESC;
```

In Firestore (classic approach):
```javascript
// This requires a composite index on
// (userId ASC, model ASC, status ASC, createdAt DESC)
// But we CANNOT also sort by qualityScore —
// that would require an inequality on two different fields
// (createdAt > 7daysAgo AND orderBy qualityScore)

const snapshot = await db.collection('generations')
  .where('userId', '==', 'abc123')
  .where('model', '==', 'kling-v2')
  .where('status', '==', 'completed')
  .where('createdAt', '>', sevenDaysAgo)
  .orderBy('createdAt', 'desc')
  .get();

// Then sort by qualityScore client-side
const sorted = snapshot.docs
  .map(doc => doc.data())
  .sort((a, b) => b.qualityScore - a.qualityScore);
```

This is ugly but works for small result sets. For large result sets, you need a different approach.

### Pipeline Operations (New Firestore Feature)

Firestore Pipeline operations (available in preview) allow more complex queries:

```javascript
import { pipeline, where, sort, limit } from 'firebase/firestore/pipeline';

const result = await pipeline(db.collection('generations'))
  .where(
    and(
      eq('userId', 'abc123'),
      eq('model', 'kling-v2'),
      eq('status', 'completed'),
      gt('createdAt', sevenDaysAgo)
    )
  )
  .sort(field('qualityScore').descending())
  .limit(20)
  .execute();
```

Pipeline operations are a game-changer because they remove the single-inequality-field limitation, enable server-side sorting on fields not in the where clause, and support aggregations (sum, avg, count) across complex filters. They do not eliminate the need for indexes, but they make complex queries possible without client-side post-processing.

---

## The Relational Alternative: PostgreSQL Schema

There comes a point in every AI video platform's life where Firestore's limitations start to pinch. That point usually arrives when you need:

1. Complex analytical queries (monthly revenue by model by region)
2. Transactional integrity across multiple entities
3. Full-text search on prompts
4. Multi-table joins for admin dashboards

Here is the same schema in PostgreSQL:

```sql
-- Users table
CREATE TABLE users (
    uid VARCHAR(128) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    display_name VARCHAR(255),
    photo_url TEXT,
    auth_provider VARCHAR(20) NOT NULL DEFAULT 'email',

    -- Subscription
    plan VARCHAR(20) NOT NULL DEFAULT 'free',
    credits INTEGER NOT NULL DEFAULT 0,
    credits_monthly INTEGER NOT NULL DEFAULT 0,
    credits_reset_at TIMESTAMPTZ,

    -- Stripe
    stripe_customer_id VARCHAR(255) UNIQUE,
    stripe_subscription_id VARCHAR(255),
    stripe_price_id VARCHAR(255),

    -- Counters (materialized, updated via triggers)
    total_generations INTEGER NOT NULL DEFAULT 0,
    total_projects INTEGER NOT NULL DEFAULT 0,
    total_storage_bytes BIGINT NOT NULL DEFAULT 0,

    -- Preferences (JSONB for flexibility)
    preferences JSONB NOT NULL DEFAULT '{}',
    features JSONB NOT NULL DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login_at TIMESTAMPTZ
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_stripe_customer ON users(stripe_customer_id);
CREATE INDEX idx_users_plan ON users(plan);

-- Projects table
CREATE TABLE projects (
    project_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL REFERENCES users(uid),

    title VARCHAR(500) NOT NULL,
    description TEXT,

    scene_count INTEGER NOT NULL DEFAULT 0,
    total_duration DECIMAL(10,2) NOT NULL DEFAULT 0,

    status VARCHAR(20) NOT NULL DEFAULT 'editing',
    resolution VARCHAR(10) NOT NULL DEFAULT '1080p',
    aspect_ratio VARCHAR(10) NOT NULL DEFAULT '16:9',
    fps INTEGER NOT NULL DEFAULT 24,

    thumbnail_url TEXT,
    preview_video_url TEXT,

    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    share_token VARCHAR(64) UNIQUE,

    total_credits_used INTEGER NOT NULL DEFAULT 0,
    total_api_cost DECIMAL(10,4) NOT NULL DEFAULT 0,

    tags TEXT[] DEFAULT '{}',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_projects_user_id ON projects(user_id);
CREATE INDEX idx_projects_user_created ON projects(user_id, created_at DESC);
CREATE INDEX idx_projects_public ON projects(is_public, created_at DESC) WHERE is_public = TRUE;
CREATE INDEX idx_projects_tags ON projects USING GIN(tags);

-- Scenes table
CREATE TABLE scenes (
    scene_id VARCHAR(64) PRIMARY KEY,
    project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
    user_id VARCHAR(128) NOT NULL REFERENCES users(uid),

    scene_order INTEGER NOT NULL,

    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    style VARCHAR(50),

    duration DECIMAL(6,2) NOT NULL DEFAULT 5.0,
    transition_in VARCHAR(20) DEFAULT 'cut',
    transition_duration DECIMAL(4,2) DEFAULT 0,

    reference_image_url TEXT,
    reference_image_r2_key TEXT,

    voiceover_text TEXT,
    voice_id VARCHAR(50),
    voiceover_asset_id VARCHAR(64),

    background_music_track VARCHAR(100),
    background_music_volume DECIMAL(3,2) DEFAULT 0.3,

    model VARCHAR(50),
    seed INTEGER,
    guidance_scale DECIMAL(4,2),

    selected_generation_id VARCHAR(64),
    selected_video_url TEXT,
    selected_thumbnail_url TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(project_id, scene_order)
);

CREATE INDEX idx_scenes_project_order ON scenes(project_id, scene_order);
CREATE INDEX idx_scenes_user_id ON scenes(user_id);

-- Generations table
CREATE TABLE generations (
    generation_id VARCHAR(64) PRIMARY KEY,
    scene_id VARCHAR(64) NOT NULL REFERENCES scenes(scene_id) ON DELETE CASCADE,
    project_id VARCHAR(64) NOT NULL REFERENCES projects(project_id),
    user_id VARCHAR(128) NOT NULL REFERENCES users(uid),

    type VARCHAR(20) NOT NULL,  -- video, image, audio, upscale

    model VARCHAR(50) NOT NULL,
    model_version VARCHAR(20),
    prompt TEXT NOT NULL,
    negative_prompt TEXT,
    seed INTEGER,
    guidance_scale DECIMAL(4,2),
    duration DECIMAL(6,2),
    resolution VARCHAR(10),
    aspect_ratio VARCHAR(10),
    fps INTEGER,

    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,

    external_job_id VARCHAR(255),
    external_status VARCHAR(50),
    callback_received BOOLEAN DEFAULT FALSE,

    quality_score DECIMAL(4,3),
    quality_flags TEXT[],
    quality_notes TEXT,

    credits_cost INTEGER NOT NULL DEFAULT 0,
    api_cost_usd DECIMAL(10,6) NOT NULL DEFAULT 0,

    output_asset_id VARCHAR(64),
    output_video_url TEXT,
    output_thumbnail_url TEXT,

    queued_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    processing_duration_ms INTEGER,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gen_scene ON generations(scene_id, created_at DESC);
CREATE INDEX idx_gen_project ON generations(project_id, created_at DESC);
CREATE INDEX idx_gen_user ON generations(user_id, created_at DESC);
CREATE INDEX idx_gen_status ON generations(status, created_at ASC);
CREATE INDEX idx_gen_user_model ON generations(user_id, model, created_at DESC);
CREATE INDEX idx_gen_user_status ON generations(user_id, status, created_at DESC);
CREATE INDEX idx_gen_quality ON generations(quality_score DESC) WHERE quality_score IS NOT NULL;

-- Assets table
CREATE TABLE assets (
    asset_id VARCHAR(64) PRIMARY KEY,
    generation_id VARCHAR(64) NOT NULL REFERENCES generations(generation_id) ON DELETE CASCADE,
    scene_id VARCHAR(64) NOT NULL,
    project_id VARCHAR(64) NOT NULL,
    user_id VARCHAR(128) NOT NULL,

    type VARCHAR(20) NOT NULL,

    storage_provider VARCHAR(10) NOT NULL DEFAULT 'r2',
    r2_key TEXT NOT NULL,
    r2_bucket VARCHAR(100) NOT NULL,
    public_url TEXT,
    cdn_url TEXT,

    mime_type VARCHAR(100) NOT NULL,
    size_bytes BIGINT NOT NULL,
    duration_ms INTEGER,
    width INTEGER,
    height INTEGER,
    codec VARCHAR(20),
    bitrate INTEGER,

    thumbnail_r2_key TEXT,
    thumbnail_url TEXT,

    status VARCHAR(20) NOT NULL DEFAULT 'available',
    expires_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_assets_generation ON assets(generation_id, type);
CREATE INDEX idx_assets_user_type ON assets(user_id, type, created_at DESC);

-- Billing events table
CREATE TABLE billing_events (
    event_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL REFERENCES users(uid),

    type VARCHAR(30) NOT NULL,

    amount INTEGER NOT NULL DEFAULT 0,  -- cents
    currency VARCHAR(3) NOT NULL DEFAULT 'usd',
    credits INTEGER NOT NULL DEFAULT 0,
    credits_before INTEGER NOT NULL DEFAULT 0,
    credits_after INTEGER NOT NULL DEFAULT 0,

    stripe_payment_intent_id VARCHAR(255),
    stripe_invoice_id VARCHAR(255),

    description TEXT,
    generation_id VARCHAR(64),
    project_id VARCHAR(64),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_billing_user ON billing_events(user_id, created_at DESC);
CREATE INDEX idx_billing_user_type ON billing_events(user_id, type, created_at DESC);
CREATE INDEX idx_billing_stripe ON billing_events(stripe_payment_intent_id);

-- Analytics events (consider using TimescaleDB extension for this)
CREATE TABLE analytics_events (
    event_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(128),
    session_id VARCHAR(64),

    type VARCHAR(50) NOT NULL,
    category VARCHAR(30) NOT NULL,

    data JSONB NOT NULL DEFAULT '{}',

    source VARCHAR(20),
    user_agent TEXT,
    ip_country VARCHAR(2),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_analytics_user ON analytics_events(user_id, type, created_at DESC);
CREATE INDEX idx_analytics_type ON analytics_events(type, created_at DESC);
CREATE INDEX idx_analytics_created ON analytics_events(created_at DESC);
CREATE INDEX idx_analytics_data ON analytics_events USING GIN(data);
```

### Entity Relationship Diagram (PostgreSQL)

<svg viewBox="0 0 920 750" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:'Courier New',monospace;">
  <defs>
    <marker id="fk-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
    <marker id="fk-diamond" markerWidth="12" markerHeight="8" refX="0" refY="4" orient="auto">
      <polygon points="6 0, 12 4, 6 8, 0 4" fill="#333"/>
    </marker>
  </defs>
  <text x="460" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#333" font-family="Arial">PostgreSQL ERD — AI Video Platform</text>
  <!-- USERS TABLE -->
  <rect x="20" y="50" width="250" height="230" rx="3" fill="#e1f5fe" stroke="#4fc3f7" stroke-width="2"/>
  <rect x="20" y="50" width="250" height="28" rx="3" fill="#4fc3f7"/>
  <text x="145" y="69" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold" font-family="Arial">users</text>
  <text x="30" y="95" font-size="10" fill="#333">PK uid VARCHAR(128)</text>
  <line x1="20" y1="102" x2="270" y2="102" stroke="#4fc3f7" stroke-width="0.5"/>
  <text x="30" y="118" font-size="10" fill="#555">   email VARCHAR(255) UNIQUE</text>
  <text x="30" y="133" font-size="10" fill="#555">   display_name VARCHAR(255)</text>
  <text x="30" y="148" font-size="10" fill="#555">   plan VARCHAR(20)</text>
  <text x="30" y="163" font-size="10" fill="#555">   credits INTEGER</text>
  <text x="30" y="178" font-size="10" fill="#555">   stripe_customer_id VARCHAR</text>
  <text x="30" y="193" font-size="10" fill="#555">   total_generations INTEGER</text>
  <text x="30" y="208" font-size="10" fill="#555">   total_projects INTEGER</text>
  <text x="30" y="223" font-size="10" fill="#555">   preferences JSONB</text>
  <text x="30" y="238" font-size="10" fill="#555">   features JSONB</text>
  <text x="30" y="253" font-size="10" fill="#555">   created_at TIMESTAMPTZ</text>
  <text x="30" y="268" font-size="10" fill="#555">   updated_at TIMESTAMPTZ</text>
  <!-- PROJECTS TABLE -->
  <rect x="340" y="50" width="250" height="230" rx="3" fill="#ffebee" stroke="#ef5350" stroke-width="2"/>
  <rect x="340" y="50" width="250" height="28" rx="3" fill="#ef5350"/>
  <text x="465" y="69" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold" font-family="Arial">projects</text>
  <text x="350" y="95" font-size="10" fill="#333">PK project_id VARCHAR(64)</text>
  <line x1="340" y1="102" x2="590" y2="102" stroke="#ef5350" stroke-width="0.5"/>
  <text x="350" y="118" font-size="10" fill="#ef5350">FK user_id → users.uid</text>
  <text x="350" y="133" font-size="10" fill="#555">   title VARCHAR(500)</text>
  <text x="350" y="148" font-size="10" fill="#555">   status VARCHAR(20)</text>
  <text x="350" y="163" font-size="10" fill="#555">   scene_count INTEGER</text>
  <text x="350" y="178" font-size="10" fill="#555">   total_duration DECIMAL</text>
  <text x="350" y="193" font-size="10" fill="#555">   resolution VARCHAR(10)</text>
  <text x="350" y="208" font-size="10" fill="#555">   thumbnail_url TEXT</text>
  <text x="350" y="223" font-size="10" fill="#555">   is_public BOOLEAN</text>
  <text x="350" y="238" font-size="10" fill="#555">   tags TEXT[]</text>
  <text x="350" y="253" font-size="10" fill="#555">   total_credits_used INTEGER</text>
  <text x="350" y="268" font-size="10" fill="#555">   created_at TIMESTAMPTZ</text>
  <!-- SCENES TABLE -->
  <rect x="660" y="50" width="240" height="230" rx="3" fill="#f1f8e9" stroke="#8bc34a" stroke-width="2"/>
  <rect x="660" y="50" width="240" height="28" rx="3" fill="#8bc34a"/>
  <text x="780" y="69" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold" font-family="Arial">scenes</text>
  <text x="670" y="95" font-size="10" fill="#333">PK scene_id VARCHAR(64)</text>
  <line x1="660" y1="102" x2="900" y2="102" stroke="#8bc34a" stroke-width="0.5"/>
  <text x="670" y="118" font-size="10" fill="#ef5350">FK project_id → projects</text>
  <text x="670" y="133" font-size="10" fill="#4fc3f7">FK user_id → users</text>
  <text x="670" y="148" font-size="10" fill="#555">   scene_order INTEGER</text>
  <text x="670" y="163" font-size="10" fill="#555">   prompt TEXT</text>
  <text x="670" y="178" font-size="10" fill="#555">   duration DECIMAL</text>
  <text x="670" y="193" font-size="10" fill="#555">   style VARCHAR(50)</text>
  <text x="670" y="208" font-size="10" fill="#555">   model VARCHAR(50)</text>
  <text x="670" y="223" font-size="10" fill="#555">   voiceover_text TEXT</text>
  <text x="670" y="238" font-size="10" fill="#555">   selected_generation_id</text>
  <text x="670" y="253" font-size="10" fill="#555">   selected_video_url TEXT</text>
  <text x="670" y="268" font-size="10" fill="#555">   created_at TIMESTAMPTZ</text>
  <!-- GENERATIONS TABLE -->
  <rect x="340" y="340" width="250" height="230" rx="3" fill="#fff3e0" stroke="#ffa726" stroke-width="2"/>
  <rect x="340" y="340" width="250" height="28" rx="3" fill="#ffa726"/>
  <text x="465" y="359" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold" font-family="Arial">generations</text>
  <text x="350" y="385" font-size="10" fill="#333">PK generation_id VARCHAR(64)</text>
  <line x1="340" y1="392" x2="590" y2="392" stroke="#ffa726" stroke-width="0.5"/>
  <text x="350" y="408" font-size="10" fill="#8bc34a">FK scene_id → scenes</text>
  <text x="350" y="423" font-size="10" fill="#ef5350">FK project_id → projects</text>
  <text x="350" y="438" font-size="10" fill="#4fc3f7">FK user_id → users</text>
  <text x="350" y="453" font-size="10" fill="#555">   type VARCHAR(20)</text>
  <text x="350" y="468" font-size="10" fill="#555">   model VARCHAR(50)</text>
  <text x="350" y="483" font-size="10" fill="#555">   status VARCHAR(20)</text>
  <text x="350" y="498" font-size="10" fill="#555">   quality_score DECIMAL</text>
  <text x="350" y="513" font-size="10" fill="#555">   credits_cost INTEGER</text>
  <text x="350" y="528" font-size="10" fill="#555">   api_cost_usd DECIMAL</text>
  <text x="350" y="543" font-size="10" fill="#555">   processing_duration_ms INT</text>
  <text x="350" y="558" font-size="10" fill="#555">   created_at TIMESTAMPTZ</text>
  <!-- ASSETS TABLE -->
  <rect x="660" y="340" width="240" height="190" rx="3" fill="#e1f5fe" stroke="#4fc3f7" stroke-width="2"/>
  <rect x="660" y="340" width="240" height="28" rx="3" fill="#4fc3f7"/>
  <text x="780" y="359" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold" font-family="Arial">assets</text>
  <text x="670" y="385" font-size="10" fill="#333">PK asset_id VARCHAR(64)</text>
  <line x1="660" y1="392" x2="900" y2="392" stroke="#4fc3f7" stroke-width="0.5"/>
  <text x="670" y="408" font-size="10" fill="#ffa726">FK generation_id → generations</text>
  <text x="670" y="423" font-size="10" fill="#555">   type VARCHAR(20)</text>
  <text x="670" y="438" font-size="10" fill="#555">   r2_key TEXT</text>
  <text x="670" y="453" font-size="10" fill="#555">   mime_type VARCHAR(100)</text>
  <text x="670" y="468" font-size="10" fill="#555">   size_bytes BIGINT</text>
  <text x="670" y="483" font-size="10" fill="#555">   width INTEGER</text>
  <text x="670" y="498" font-size="10" fill="#555">   height INTEGER</text>
  <text x="670" y="513" font-size="10" fill="#555">   status VARCHAR(20)</text>
  <text x="670" y="523" font-size="10" fill="#555">   created_at TIMESTAMPTZ</text>
  <!-- BILLING TABLE -->
  <rect x="20" y="340" width="250" height="180" rx="3" fill="#f3e5f5" stroke="#ab47bc" stroke-width="2"/>
  <rect x="20" y="340" width="250" height="28" rx="3" fill="#ab47bc"/>
  <text x="145" y="359" text-anchor="middle" font-size="13" fill="#fff" font-weight="bold" font-family="Arial">billing_events</text>
  <text x="30" y="385" font-size="10" fill="#333">PK event_id VARCHAR(64)</text>
  <line x1="20" y1="392" x2="270" y2="392" stroke="#ab47bc" stroke-width="0.5"/>
  <text x="30" y="408" font-size="10" fill="#4fc3f7">FK user_id → users</text>
  <text x="30" y="423" font-size="10" fill="#555">   type VARCHAR(30)</text>
  <text x="30" y="438" font-size="10" fill="#555">   amount INTEGER (cents)</text>
  <text x="30" y="453" font-size="10" fill="#555">   credits INTEGER</text>
  <text x="30" y="468" font-size="10" fill="#555">   credits_before INTEGER</text>
  <text x="30" y="483" font-size="10" fill="#555">   credits_after INTEGER</text>
  <text x="30" y="498" font-size="10" fill="#555">   stripe_payment_intent_id</text>
  <text x="30" y="513" font-size="10" fill="#555">   created_at TIMESTAMPTZ</text>
  <!-- Relationship lines -->
  <!-- users -> projects -->
  <line x1="270" y1="118" x2="340" y2="118" stroke="#333" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <text x="300" y="112" font-size="9" fill="#666" font-family="Arial">1:N</text>
  <!-- projects -> scenes -->
  <line x1="590" y1="118" x2="660" y2="118" stroke="#333" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <text x="620" y="112" font-size="9" fill="#666" font-family="Arial">1:N</text>
  <!-- scenes -> generations -->
  <line x1="780" y1="280" x2="780" y2="310" stroke="#333" stroke-width="1"/>
  <line x1="780" y1="310" x2="560" y2="310" stroke="#333" stroke-width="1"/>
  <line x1="560" y1="310" x2="560" y2="340" stroke="#333" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <text x="670" y="305" font-size="9" fill="#666" font-family="Arial">1:N</text>
  <!-- generations -> assets -->
  <line x1="590" y1="450" x2="660" y2="450" stroke="#333" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <text x="620" y="444" font-size="9" fill="#666" font-family="Arial">1:N</text>
  <!-- users -> billing -->
  <line x1="145" y1="280" x2="145" y2="340" stroke="#333" stroke-width="1.5" marker-end="url(#fk-arrow)"/>
  <text x="155" y="315" font-size="9" fill="#666" font-family="Arial">1:N</text>
  <!-- Legend -->
  <rect x="20" y="590" width="880" height="140" rx="6" fill="#f8f9fa" stroke="#ddd"/>
  <text x="40" y="615" font-size="13" font-weight="bold" fill="#333" font-family="Arial">PostgreSQL vs Firestore: When to Choose Which</text>
  <text x="40" y="640" font-size="11" fill="#555" font-family="Arial">Firestore: Realtime listeners, rapid prototyping, auto-scaling, simple CRUD, mobile/web SDKs</text>
  <text x="40" y="660" font-size="11" fill="#555" font-family="Arial">PostgreSQL: Complex joins, analytical queries, ACID transactions, full-text search, mature tooling</text>
  <text x="40" y="685" font-size="11" fill="#555" font-family="Arial">Hybrid: Use Firestore for user-facing reads (projects, scenes, generations) and PostgreSQL for</text>
  <text x="40" y="705" font-size="11" fill="#555" font-family="Arial">analytics, billing, admin dashboards. Sync via Cloud Functions or Change Data Capture (CDC).</text>
</svg>

### When Relational Makes More Sense

The PostgreSQL schema gives you capabilities that Firestore simply cannot match:

**1. Complex Analytical Queries**

```sql
-- Revenue by model by month (impossible in single Firestore query)
SELECT
    model,
    DATE_TRUNC('month', created_at) AS month,
    COUNT(*) AS generation_count,
    SUM(credits_cost) AS total_credits,
    SUM(api_cost_usd) AS total_cost,
    AVG(quality_score) AS avg_quality,
    AVG(processing_duration_ms) AS avg_processing_ms
FROM generations
WHERE created_at >= '2025-01-01'
GROUP BY model, DATE_TRUNC('month', created_at)
ORDER BY month DESC, total_credits DESC;
```

**2. Cross-Entity Joins**

```sql
-- Top users by spend with their most-used model (requires 3-way join)
SELECT
    u.display_name,
    u.plan,
    COUNT(g.generation_id) AS total_gens,
    SUM(g.credits_cost) AS total_credits,
    MODE() WITHIN GROUP (ORDER BY g.model) AS favorite_model
FROM users u
JOIN generations g ON u.uid = g.user_id
WHERE g.created_at >= NOW() - INTERVAL '30 days'
GROUP BY u.uid, u.display_name, u.plan
ORDER BY total_credits DESC
LIMIT 20;
```

**3. Transactional Integrity**

```sql
-- Deduct credits and record generation atomically
BEGIN;
    UPDATE users SET credits = credits - 10 WHERE uid = 'abc123' AND credits >= 10;
    -- Check rowcount; if 0, user didn't have enough credits
    INSERT INTO generations (generation_id, scene_id, ..., credits_cost)
    VALUES ('gen_new', 'scene_7Wn3', ..., 10);
    INSERT INTO billing_events (event_id, user_id, type, credits, ...)
    VALUES ('bill_new', 'abc123', 'credit_usage', -10, ...);
COMMIT;
```

In Firestore, you would use a transaction for this, but Firestore transactions have a limit of 500 writes and can fail if any of the documents change during the transaction window. PostgreSQL transactions are ACID-compliant with isolation levels that give you much stronger guarantees.

---

## The 10 Most Common Queries

Here are the queries that will account for 90%+ of your database traffic, implemented in both Firestore and SQL.

### Query 1: Get User Profile

**Firestore:**
```javascript
const userDoc = await db.collection('users').doc(uid).get();
const user = userDoc.data();
// Cost: 1 read. Latency: ~10ms
```

**SQL:**
```sql
SELECT * FROM users WHERE uid = $1;
-- Cost: 1 index lookup. Latency: ~2ms
```

### Query 2: List User's Projects (Paginated, Newest First)

**Firestore:**
```javascript
const snapshot = await db.collection('projects')
  .where('userId', '==', uid)
  .orderBy('createdAt', 'desc')
  .limit(20)
  .startAfter(lastDoc) // Cursor-based pagination
  .get();
// Cost: 20 reads + 1 (for the query itself). Latency: ~30ms
// Requires index: (userId ASC, createdAt DESC)
```

**SQL:**
```sql
SELECT * FROM projects
WHERE user_id = $1
ORDER BY created_at DESC
LIMIT 20 OFFSET $2;
-- Cost: 1 index scan. Latency: ~5ms
-- Or cursor-based:
SELECT * FROM projects
WHERE user_id = $1 AND created_at < $2
ORDER BY created_at DESC
LIMIT 20;
```

### Query 3: Get All Scenes for a Project (Ordered)

**Firestore:**
```javascript
const snapshot = await db.collection('scenes')
  .where('projectId', '==', projectId)
  .orderBy('order', 'asc')
  .get();
// Cost: N reads (one per scene). Latency: ~20ms for 5 scenes
// Requires index: (projectId ASC, order ASC)
```

**SQL:**
```sql
SELECT * FROM scenes
WHERE project_id = $1
ORDER BY scene_order ASC;
-- Cost: 1 index scan. Latency: ~3ms
```

### Query 4: Get Generation History for a Scene

**Firestore:**
```javascript
const snapshot = await db.collection('generations')
  .where('sceneId', '==', sceneId)
  .orderBy('createdAt', 'desc')
  .limit(10)
  .get();
// Cost: 10 reads. Latency: ~25ms
```

**SQL:**
```sql
SELECT * FROM generations
WHERE scene_id = $1
ORDER BY created_at DESC
LIMIT 10;
```

### Query 5: Count User's Generations This Month

**Firestore:**
```javascript
// Option A: Read denormalized counter (fast but approximate)
const userDoc = await db.collection('users').doc(uid).get();
const count = userDoc.data().totalGenerations; // Lifetime, not monthly

// Option B: Aggregation query (accurate but slower)
const snapshot = await db.collection('generations')
  .where('userId', '==', uid)
  .where('createdAt', '>=', monthStart)
  .count()
  .get();
// Cost: 1 aggregation query (billed as N reads / 1000). Latency: ~100ms
```

**SQL:**
```sql
SELECT COUNT(*) FROM generations
WHERE user_id = $1 AND created_at >= DATE_TRUNC('month', CURRENT_DATE);
-- Cost: 1 index scan with count. Latency: ~5ms
```

### Query 6: Get Pending Generations (Worker Queue)

**Firestore:**
```javascript
const snapshot = await db.collection('generations')
  .where('status', '==', 'queued')
  .orderBy('createdAt', 'asc')
  .limit(50)
  .get();
// Cost: 50 reads. Latency: ~40ms
// Requires index: (status ASC, createdAt ASC)
```

**SQL:**
```sql
SELECT * FROM generations
WHERE status = 'queued'
ORDER BY created_at ASC
LIMIT 50
FOR UPDATE SKIP LOCKED; -- Prevents multiple workers grabbing the same job
```

The `FOR UPDATE SKIP LOCKED` pattern in PostgreSQL is extremely powerful for job queues. Firestore has no equivalent; you need to use distributed locks or Cloud Tasks.

### Query 7: User's Billing History

**Firestore:**
```javascript
const snapshot = await db.collection('billing_events')
  .where('userId', '==', uid)
  .orderBy('createdAt', 'desc')
  .limit(50)
  .get();
```

**SQL:**
```sql
SELECT * FROM billing_events
WHERE user_id = $1
ORDER BY created_at DESC
LIMIT 50;
```

### Query 8: Monthly Revenue Report

**Firestore:**
```javascript
// This is painful in Firestore. You need to either:
// A) Pre-aggregate into a separate collection (revenue_monthly)
// B) Read all billing events and aggregate client-side
// C) Use Pipeline operations for server-side aggregation

// Option A (pre-aggregated):
const doc = await db.collection('revenue_monthly')
  .doc('2026-01')
  .get();
```

**SQL:**
```sql
SELECT
    DATE_TRUNC('month', created_at) AS month,
    type,
    SUM(amount) / 100.0 AS revenue_usd,
    COUNT(*) AS event_count
FROM billing_events
WHERE type IN ('subscription_payment', 'credit_purchase')
  AND created_at >= '2025-01-01'
GROUP BY DATE_TRUNC('month', created_at), type
ORDER BY month DESC;
```

### Query 9: Search Projects by Title (Full-Text)

**Firestore:**
```javascript
// Firestore has NO full-text search. Options:
// A) Use Algolia or Typesense as a search index
// B) Implement prefix matching with range queries
const snapshot = await db.collection('projects')
  .where('userId', '==', uid)
  .where('title', '>=', searchTerm)
  .where('title', '<=', searchTerm + '\uf8ff')
  .get();
// This only matches prefixes, not substrings
```

**SQL:**
```sql
-- With pg_trgm extension for fuzzy search
SELECT * FROM projects
WHERE user_id = $1
  AND title ILIKE '%' || $2 || '%'
ORDER BY similarity(title, $2) DESC
LIMIT 20;

-- Or with full-text search
SELECT * FROM projects
WHERE user_id = $1
  AND to_tsvector('english', title || ' ' || COALESCE(description, ''))
      @@ plainto_tsquery('english', $2)
LIMIT 20;
```

### Query 10: Admin Dashboard — Platform Statistics

**Firestore:**
```javascript
// Requires pre-aggregated documents
const stats = await db.collection('platform_stats').doc('current').get();
// {
//   totalUsers: 15234,
//   totalGenerations: 892341,
//   activeUsersToday: 3421,
//   revenueThisMonth: 45678.90,
//   ...
// }
// Updated periodically by a Cloud Function
```

**SQL:**
```sql
SELECT
    (SELECT COUNT(*) FROM users) AS total_users,
    (SELECT COUNT(*) FROM users WHERE last_login_at >= CURRENT_DATE) AS active_today,
    (SELECT COUNT(*) FROM generations WHERE created_at >= CURRENT_DATE) AS gens_today,
    (SELECT SUM(amount)/100.0 FROM billing_events
     WHERE type IN ('subscription_payment','credit_purchase')
     AND created_at >= DATE_TRUNC('month', CURRENT_DATE)) AS revenue_mtd;
```

### Performance Comparison Summary

| Query | Firestore Reads | Firestore Latency | SQL Latency | Winner |
|---|---|---|---|---|
| User profile | 1 | ~10ms | ~2ms | SQL (marginal) |
| Project list (20) | 21 | ~30ms | ~5ms | SQL |
| Scene list (5) | 6 | ~20ms | ~3ms | SQL |
| Generation history (10) | 11 | ~25ms | ~4ms | SQL |
| Monthly count | ~1000/1000 | ~100ms | ~5ms | SQL |
| Pending queue (50) | 51 | ~40ms | ~3ms | SQL |
| Billing history (50) | 51 | ~35ms | ~5ms | SQL |
| Revenue report | Pre-computed | ~10ms | ~50ms | Firestore* |
| Text search | Not supported | N/A | ~20ms | SQL |
| Platform stats | Pre-computed | ~10ms | ~100ms | Firestore* |

*Firestore wins on pre-computed analytics only because the computation happened offline. The actual query is fast because it reads a single document that was pre-built.

<svg viewBox="0 0 780 400" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif;">
  <text x="390" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#333">Query Latency Comparison: Firestore vs PostgreSQL</text>
  <!-- Axes -->
  <line x1="80" y1="50" x2="80" y2="350" stroke="#333" stroke-width="1.5"/>
  <line x1="80" y1="350" x2="760" y2="350" stroke="#333" stroke-width="1.5"/>
  <text x="40" y="200" text-anchor="middle" font-size="11" fill="#333" transform="rotate(-90 40 200)">Latency (ms)</text>
  <!-- Y-axis labels -->
  <text x="70" y="354" text-anchor="end" font-size="9" fill="#666">0</text>
  <text x="70" y="290" text-anchor="end" font-size="9" fill="#666">20</text>
  <line x1="78" y1="288" x2="760" y2="288" stroke="#eee" stroke-width="0.5"/>
  <text x="70" y="230" text-anchor="end" font-size="9" fill="#666">40</text>
  <line x1="78" y1="228" x2="760" y2="228" stroke="#eee" stroke-width="0.5"/>
  <text x="70" y="170" text-anchor="end" font-size="9" fill="#666">60</text>
  <line x1="78" y1="168" x2="760" y2="168" stroke="#eee" stroke-width="0.5"/>
  <text x="70" y="110" text-anchor="end" font-size="9" fill="#666">80</text>
  <line x1="78" y1="108" x2="760" y2="108" stroke="#eee" stroke-width="0.5"/>
  <text x="70" y="55" text-anchor="end" font-size="9" fill="#666">100</text>
  <line x1="78" y1="53" x2="760" y2="53" stroke="#eee" stroke-width="0.5"/>
  <!-- Bar groups -->
  <!-- Q1: User Profile -->
  <rect x="100" y="320" width="25" height="30" fill="#4fc3f7"/>
  <rect x="128" y="344" width="25" height="6" fill="#ef5350"/>
  <text x="126" y="370" text-anchor="middle" font-size="8" fill="#333">Profile</text>
  <!-- Q2: Project List -->
  <rect x="175" y="260" width="25" height="90" fill="#4fc3f7"/>
  <rect x="203" y="335" width="25" height="15" fill="#ef5350"/>
  <text x="201" y="370" text-anchor="middle" font-size="8" fill="#333">Projects</text>
  <!-- Q3: Scene List -->
  <rect x="250" y="290" width="25" height="60" fill="#4fc3f7"/>
  <rect x="278" y="341" width="25" height="9" fill="#ef5350"/>
  <text x="276" y="370" text-anchor="middle" font-size="8" fill="#333">Scenes</text>
  <!-- Q4: Gen History -->
  <rect x="325" y="275" width="25" height="75" fill="#4fc3f7"/>
  <rect x="353" y="338" width="25" height="12" fill="#ef5350"/>
  <text x="351" y="370" text-anchor="middle" font-size="8" fill="#333">Gens</text>
  <!-- Q5: Monthly Count -->
  <rect x="400" y="53" width="25" height="297" fill="#4fc3f7"/>
  <rect x="428" y="335" width="25" height="15" fill="#ef5350"/>
  <text x="426" y="370" text-anchor="middle" font-size="8" fill="#333">Count</text>
  <!-- Q6: Queue -->
  <rect x="475" y="230" width="25" height="120" fill="#4fc3f7"/>
  <rect x="503" y="341" width="25" height="9" fill="#ef5350"/>
  <text x="501" y="370" text-anchor="middle" font-size="8" fill="#333">Queue</text>
  <!-- Q7: Billing -->
  <rect x="550" y="245" width="25" height="105" fill="#4fc3f7"/>
  <rect x="578" y="335" width="25" height="15" fill="#ef5350"/>
  <text x="576" y="370" text-anchor="middle" font-size="8" fill="#333">Billing</text>
  <!-- Q8: Revenue -->
  <rect x="625" y="320" width="25" height="30" fill="#4fc3f7"/>
  <rect x="653" y="200" width="25" height="150" fill="#ef5350"/>
  <text x="651" y="370" text-anchor="middle" font-size="8" fill="#333">Revenue</text>
  <!-- Q9: Stats -->
  <rect x="700" y="320" width="25" height="30" fill="#4fc3f7"/>
  <rect x="728" y="53" width="25" height="297" fill="#ef5350"/>
  <text x="726" y="370" text-anchor="middle" font-size="8" fill="#333">Stats</text>
  <!-- Legend -->
  <rect x="300" y="385" width="14" height="10" fill="#4fc3f7"/>
  <text x="320" y="394" font-size="10" fill="#333">Firestore</text>
  <rect x="400" y="385" width="14" height="10" fill="#ef5350"/>
  <text x="420" y="394" font-size="10" fill="#333">PostgreSQL</text>
</svg>

---

## Data Migration Patterns

Your schema will evolve. Fields will be added, renamed, restructured. In PostgreSQL, you run `ALTER TABLE` migrations. In Firestore, there are no migrations because there is no schema enforcement. This is both a blessing and a curse.

### Firestore Schema Evolution

**Rule 1: Always add, never rename or remove fields.**

When you add a new field, old documents simply will not have it. Your code must handle this:

```javascript
// Good: Default value for missing field
const user = userDoc.data();
const plan = user.plan ?? 'free'; // Old users won't have this field
const features = user.features ?? {};
const betaPricing = features.betaPricing ?? 'control';
```

**Rule 2: Use a schema version field for major changes.**

```javascript
// Document with version tracking
{
  _schemaVersion: 2,
  // v1 fields
  name: "Jane Creator",
  // v2 fields (added later)
  displayName: "Jane Creator",  // New field, same data
  preferences: { ... }          // Restructured from flat fields
}
```

**Rule 3: Backfill lazily or in batches.**

When you add a new field that needs to be populated on existing documents, you have two approaches:

```javascript
// Lazy backfill: Update documents as users access them
async function getUser(uid) {
  const doc = await db.collection('users').doc(uid).get();
  const data = doc.data();

  if (data._schemaVersion < 2) {
    // Migrate on read
    const updates = migrateUserV1toV2(data);
    await doc.ref.update({
      ...updates,
      _schemaVersion: 2
    });
    return { ...data, ...updates };
  }
  return data;
}

// Batch backfill: Cloud Function that processes all documents
async function batchMigrateUsers() {
  const batchSize = 500;
  let lastDoc = null;

  while (true) {
    let query = db.collection('users')
      .where('_schemaVersion', '<', 2)
      .limit(batchSize);

    if (lastDoc) {
      query = query.startAfter(lastDoc);
    }

    const snapshot = await query.get();
    if (snapshot.empty) break;

    const batch = db.batch();
    snapshot.docs.forEach(doc => {
      const updates = migrateUserV1toV2(doc.data());
      batch.update(doc.ref, { ...updates, _schemaVersion: 2 });
    });

    await batch.commit();
    lastDoc = snapshot.docs[snapshot.docs.length - 1];

    // Rate limit to avoid hotspotting
    await sleep(1000);
  }
}
```

### PostgreSQL Schema Evolution

PostgreSQL migrations are more structured. Use a migration tool like `golang-migrate`, `Flyway`, or `Prisma Migrate`.

```sql
-- Migration 001: Add quality_flags to generations
ALTER TABLE generations ADD COLUMN quality_flags TEXT[] DEFAULT '{}';
ALTER TABLE generations ADD COLUMN quality_notes TEXT;

-- Migration 002: Add composite index for new query pattern
CREATE INDEX CONCURRENTLY idx_gen_user_quality
    ON generations(user_id, quality_score DESC)
    WHERE quality_score IS NOT NULL;

-- Migration 003: Restructure preferences (backward compatible)
ALTER TABLE users ADD COLUMN default_model VARCHAR(50) DEFAULT 'kling-v2';
ALTER TABLE users ADD COLUMN default_resolution VARCHAR(10) DEFAULT '1080p';
-- Backfill from JSONB preferences
UPDATE users SET
    default_model = COALESCE(preferences->>'defaultModel', 'kling-v2'),
    default_resolution = COALESCE(preferences->>'defaultResolution', '1080p');
```

The `CONCURRENTLY` keyword on index creation is critical in production. Without it, PostgreSQL takes a write lock on the entire table while building the index, which can lock your application for minutes on large tables.

---

## Storage Strategy: Database vs Object Storage

Not all data belongs in the database. The general rule is:

- **Metadata** goes in the database (Firestore or PostgreSQL)
- **Binary blobs** go in object storage (Cloudflare R2, S3, GCS)
- **The database stores a reference (URL or key) to the blob**

| Data Type | Storage | Reference in DB | Example |
|---|---|---|---|
| Video files | R2 | `r2Key`, `publicUrl` | `videos/gen_Pq4r/output.mp4` |
| Image files | R2 | `r2Key`, `publicUrl` | `images/gen_Pq4r/frame_001.jpg` |
| Audio files | R2 | `r2Key`, `publicUrl` | `audio/gen_Pq4r/voiceover.mp3` |
| Thumbnails | R2 + CDN | `thumbnailUrl` | `thumbs/gen_Pq4r.jpg` |
| User avatars | R2 + CDN | `photoURL` (on user doc) | `avatars/abc123.jpg` |
| Generation parameters | Database | Direct fields | `prompt`, `seed`, `model` |
| Billing data | Database | Direct fields | `amount`, `credits` |
| Analytics events | Database | Direct fields | `type`, `data` (JSONB) |

### R2 Key Naming Convention

Use a predictable, hierarchical key structure:

```
{entity_type}/{entity_id}/{file_type}.{extension}

Examples:
videos/gen_Pq4rW8n3/output.mp4
videos/gen_Pq4rW8n3/preview.mp4
thumbs/gen_Pq4rW8n3/thumb_256.jpg
thumbs/gen_Pq4rW8n3/thumb_512.jpg
audio/gen_Pq4rW8n3/voiceover.mp3
refs/scene_7Wn3xP9q2m/reference.jpg
avatars/abc123def456/profile.jpg
exports/proj_Kx8mN2pQ4r/final_render.mp4
```

This structure makes it easy to:
1. Delete all assets for a generation: `DELETE videos/gen_Pq4rW8n3/*`
2. Calculate storage per user: list keys matching `*/gen_*` and cross-reference with generation's userId
3. Implement lifecycle policies: delete preview files after 30 days

### URL Reference Patterns

Store both the R2 key and the public URL:

```javascript
// In the asset document
{
  r2Key: "videos/gen_Pq4rW8n3/output.mp4",         // For server-side operations
  r2Bucket: "ai-video-assets",                       // Bucket name
  publicUrl: "https://pub-xxx.r2.dev/videos/gen_Pq4rW8n3/output.mp4",  // Direct R2 URL
  cdnUrl: "https://cdn.example.com/videos/gen_Pq4rW8n3/output.mp4",    // CDN URL
}
```

Why both? The `r2Key` is what you use server-side to generate signed URLs, delete files, or move files. The `publicUrl` is what you serve to the client. If you change CDN providers, you update the CDN URL without touching the R2 key. If you need to generate a time-limited signed URL for premium content, you use the R2 key:

```javascript
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

const signedUrl = await getSignedUrl(
  r2Client,
  new GetObjectCommand({
    Bucket: asset.r2Bucket,
    Key: asset.r2Key,
  }),
  { expiresIn: 3600 } // 1 hour
);
```

---

## Analytics Data Model

Analytics data is different from operational data. It is append-only, time-series, and queried in aggregate rather than individually. This means different storage strategies.

### Pre-Computed Aggregations

For dashboards that show "generations per day" or "revenue per month," pre-compute the aggregations and store them in dedicated documents:

```javascript
// Collection: daily_stats
// Document ID: "2026-01-14"
{
  date: "2026-01-14",

  // Generation metrics
  totalGenerations: 12453,
  generationsByModel: {
    "kling-v2": 5234,
    "runway-gen4": 3421,
    "minimax-video-01": 2189,
    "pika-2.0": 1609
  },
  generationsByStatus: {
    "completed": 11208,
    "failed": 892,
    "cancelled": 353
  },
  avgProcessingTimeMs: 134500,
  avgQualityScore: 0.82,

  // User metrics
  activeUsers: 3421,
  newUsers: 234,
  churningUsers: 45,  // Users who haven't logged in for 30 days

  // Revenue metrics
  subscriptionRevenue: 15234.00,
  creditRevenue: 3456.00,
  totalRevenue: 18690.00,

  // Cost metrics
  totalApiCost: 6789.00,
  costByModel: {
    "kling-v2": 3200.00,
    "runway-gen4": 2100.00,
    "minimax-video-01": 987.00,
    "pika-2.0": 502.00
  },
  totalStorageCost: 45.00,
  totalBandwidthCost: 123.00,

  // Derived
  grossMargin: 0.627,  // (revenue - costs) / revenue

  computedAt: Timestamp("2026-01-15T00:05:00Z")
}
```

This document is computed by a scheduled Cloud Function that runs at midnight:

```javascript
// Cloud Function: computeDailyStats
exports.computeDailyStats = onSchedule('0 0 * * *', async () => {
  const yesterday = getYesterdayDateString();
  const dayStart = new Date(yesterday + 'T00:00:00Z');
  const dayEnd = new Date(yesterday + 'T23:59:59.999Z');

  // Query all generations for the day
  const gensSnapshot = await db.collection('generations')
    .where('createdAt', '>=', dayStart)
    .where('createdAt', '<=', dayEnd)
    .get();

  // Aggregate...
  const stats = aggregateGenerations(gensSnapshot.docs);

  // Query billing events
  const billingSnapshot = await db.collection('billing_events')
    .where('createdAt', '>=', dayStart)
    .where('createdAt', '<=', dayEnd)
    .get();

  stats.revenue = aggregateBilling(billingSnapshot.docs);

  // Write the aggregated document
  await db.collection('daily_stats').doc(yesterday).set(stats);
});
```

### User-Level Analytics

For per-user dashboards ("your usage this month"), store user-level monthly aggregations:

```javascript
// Collection: user_monthly_stats
// Document ID: "{userId}_{YYYY-MM}"
{
  userId: "abc123def456",
  month: "2026-01",

  generationCount: 47,
  generationsByModel: { "kling-v2": 30, "runway-gen4": 17 },
  creditsUsed: 470,
  apiCostUsd: 24.50,

  storageUsedBytes: 2147483648, // 2 GB

  topPromptKeywords: ["product", "demo", "cinematic", "tech"],
  avgQualityScore: 0.85,

  computedAt: Timestamp("2026-01-14T00:05:00Z")
}
```

### Query-Time vs Pre-Compute: Decision Matrix

| Metric | Pre-Compute | Query-Time | Recommendation |
|---|---|---|---|
| Daily active users | Yes | Too expensive at scale | Pre-compute |
| User's generation count this month | Yes (or denormalize on user doc) | Feasible for single user | Either |
| Revenue by model (last 30 days) | Yes | Feasible in SQL, expensive in Firestore | Pre-compute for Firestore, query-time for SQL |
| Single generation's status | No | Single document read | Query-time |
| Platform-wide quality trend | Yes | Too many documents to scan | Pre-compute |
| User's billing history | No | Straightforward paginated query | Query-time |

The rule of thumb: if the query touches more than ~100 documents or if it is displayed on a frequently-viewed dashboard, pre-compute it. If it is a single-entity lookup or a rare admin query, compute it at query time.

---

## Putting It All Together: A Complete Query Flow

Let us trace a complete user interaction through the data layer. The user opens the video editor and clicks "Generate" on a scene.

<svg viewBox="0 0 860 700" xmlns="http://www.w3.org/2000/svg" style="max-width:100%;height:auto;background:#fff;font-family:Arial,Helvetica,sans-serif;">
  <defs>
    <marker id="flow-arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#333"/>
    </marker>
  </defs>
  <text x="430" y="25" text-anchor="middle" font-size="15" font-weight="bold" fill="#333">Query Flow: User Clicks "Generate"</text>
  <!-- Step boxes -->
  <!-- Step 1 -->
  <rect x="20" y="50" width="200" height="55" rx="6" fill="#4fc3f7"/>
  <text x="120" y="72" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">1. Auth Check</text>
  <text x="120" y="90" text-anchor="middle" font-size="9" fill="#fff">READ users/{uid}</text>
  <text x="120" y="100" text-anchor="middle" font-size="8" fill="#e1f5fe">Verify plan + credits >= cost</text>
  <!-- Step 2 -->
  <rect x="240" y="50" width="200" height="55" rx="6" fill="#ef5350"/>
  <text x="340" y="72" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">2. Scene Lookup</text>
  <text x="340" y="90" text-anchor="middle" font-size="9" fill="#fff">READ scenes/{sceneId}</text>
  <text x="340" y="100" text-anchor="middle" font-size="8" fill="#ffebee">Get prompt, model, settings</text>
  <!-- Step 3 -->
  <rect x="460" y="50" width="200" height="55" rx="6" fill="#8bc34a"/>
  <text x="560" y="72" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">3. Deduct Credits</text>
  <text x="560" y="90" text-anchor="middle" font-size="9" fill="#fff">UPDATE users/{uid}</text>
  <text x="560" y="100" text-anchor="middle" font-size="8" fill="#f1f8e9">credits -= cost (atomic)</text>
  <!-- Arrow 1->2 -->
  <line x1="220" y1="77" x2="240" y2="77" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Arrow 2->3 -->
  <line x1="440" y1="77" x2="460" y2="77" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 4 -->
  <rect x="680" y="50" width="160" height="55" rx="6" fill="#ffa726"/>
  <text x="760" y="72" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">4. Create Gen Doc</text>
  <text x="760" y="90" text-anchor="middle" font-size="9" fill="#fff">WRITE generations/</text>
  <text x="760" y="100" text-anchor="middle" font-size="8" fill="#fff3e0">status: "queued"</text>
  <line x1="660" y1="77" x2="680" y2="77" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 5 -->
  <rect x="680" y="140" width="160" height="55" rx="6" fill="#ffa726"/>
  <text x="760" y="162" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">5. Create Billing Evt</text>
  <text x="760" y="180" text-anchor="middle" font-size="9" fill="#fff">WRITE billing_events/</text>
  <text x="760" y="190" text-anchor="middle" font-size="8" fill="#fff3e0">type: "credit_usage"</text>
  <line x1="760" y1="105" x2="760" y2="140" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 6 -->
  <rect x="680" y="230" width="160" height="55" rx="6" fill="#4fc3f7"/>
  <text x="760" y="252" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">6. Queue Job</text>
  <text x="760" y="270" text-anchor="middle" font-size="9" fill="#fff">Cloud Tasks / PubSub</text>
  <text x="760" y="280" text-anchor="middle" font-size="8" fill="#e1f5fe">Send to worker</text>
  <line x1="760" y1="195" x2="760" y2="230" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Worker section -->
  <rect x="30" y="170" width="620" height="30" rx="4" fill="#f5f5f5" stroke="#ddd"/>
  <text x="340" y="190" text-anchor="middle" font-size="12" font-weight="bold" fill="#888">--- Worker Process (Async) ---</text>
  <!-- Step 7 -->
  <rect x="30" y="230" width="180" height="55" rx="6" fill="#ef5350"/>
  <text x="120" y="252" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">7. Call Gen API</text>
  <text x="120" y="270" text-anchor="middle" font-size="9" fill="#fff">Kling / Runway / etc.</text>
  <text x="120" y="280" text-anchor="middle" font-size="8" fill="#ffebee">External HTTP call</text>
  <line x1="680" y1="257" x2="210" y2="257" stroke="#333" stroke-width="1" stroke-dasharray="5,3" marker-end="url(#flow-arrow)"/>
  <!-- Step 8 -->
  <rect x="30" y="320" width="180" height="55" rx="6" fill="#8bc34a"/>
  <text x="120" y="342" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">8. Update Status</text>
  <text x="120" y="360" text-anchor="middle" font-size="9" fill="#fff">UPDATE generations/</text>
  <text x="120" y="370" text-anchor="middle" font-size="8" fill="#f1f8e9">status: "processing"</text>
  <line x1="120" y1="285" x2="120" y2="320" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 9 -->
  <rect x="250" y="320" width="180" height="55" rx="6" fill="#ffa726"/>
  <text x="340" y="342" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">9. Receive Callback</text>
  <text x="340" y="360" text-anchor="middle" font-size="9" fill="#fff">Download video from API</text>
  <text x="340" y="370" text-anchor="middle" font-size="8" fill="#fff3e0">Webhook or polling</text>
  <line x1="210" y1="347" x2="250" y2="347" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 10 -->
  <rect x="470" y="320" width="180" height="55" rx="6" fill="#4fc3f7"/>
  <text x="560" y="342" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">10. Upload to R2</text>
  <text x="560" y="360" text-anchor="middle" font-size="9" fill="#fff">Store video + thumbnail</text>
  <text x="560" y="370" text-anchor="middle" font-size="8" fill="#e1f5fe">S3-compatible API</text>
  <line x1="430" y1="347" x2="470" y2="347" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 11 -->
  <rect x="250" y="410" width="180" height="55" rx="6" fill="#ef5350"/>
  <text x="340" y="432" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">11. Create Asset Doc</text>
  <text x="340" y="450" text-anchor="middle" font-size="9" fill="#fff">WRITE assets/</text>
  <text x="340" y="460" text-anchor="middle" font-size="8" fill="#ffebee">r2Key, mimeType, size</text>
  <line x1="560" y1="375" x2="560" y2="395" stroke="#333" stroke-width="1"/>
  <line x1="560" y1="395" x2="340" y2="395" stroke="#333" stroke-width="1"/>
  <line x1="340" y1="395" x2="340" y2="410" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 12 -->
  <rect x="250" y="500" width="180" height="55" rx="6" fill="#8bc34a"/>
  <text x="340" y="522" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">12. QA Check</text>
  <text x="340" y="540" text-anchor="middle" font-size="9" fill="#fff">Gemini Flash analysis</text>
  <text x="340" y="550" text-anchor="middle" font-size="8" fill="#f1f8e9">Score: 0.87</text>
  <line x1="340" y1="465" x2="340" y2="500" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Step 13 -->
  <rect x="250" y="590" width="180" height="55" rx="6" fill="#ffa726"/>
  <text x="340" y="612" text-anchor="middle" font-size="11" fill="#fff" font-weight="bold">13. Finalize</text>
  <text x="340" y="630" text-anchor="middle" font-size="9" fill="#fff">UPDATE generation, scene</text>
  <text x="340" y="640" text-anchor="middle" font-size="8" fill="#fff3e0">status: "completed"</text>
  <line x1="340" y1="555" x2="340" y2="590" stroke="#333" stroke-width="1.5" marker-end="url(#flow-arrow)"/>
  <!-- Writes count box -->
  <rect x="500" y="450" width="340" height="200" rx="6" fill="#f8f9fa" stroke="#ddd"/>
  <text x="670" y="475" text-anchor="middle" font-size="12" font-weight="bold" fill="#333">Total DB Operations</text>
  <text x="520" y="500" font-size="10" fill="#555">Reads:  2 (user + scene)</text>
  <text x="520" y="520" font-size="10" fill="#555">Writes: 7 (user update, generation x2,</text>
  <text x="520" y="540" font-size="10" fill="#555">         billing, asset, scene update,</text>
  <text x="520" y="560" font-size="10" fill="#555">         analytics event)</text>
  <text x="520" y="590" font-size="10" fill="#555">External: 2 (Gen API, Gemini Flash QA)</text>
  <text x="520" y="610" font-size="10" fill="#555">R2 Ops:   2 (video upload, thumb upload)</text>
  <text x="520" y="640" font-size="10" fill="#888">Total Firestore cost: ~$0.000014</text>
</svg>

This flow involves:
- **2 Firestore reads** ($0.06 per 100K = $0.0000012)
- **7 Firestore writes** ($0.18 per 100K = $0.0000126)
- **2 external API calls** (the expensive part: $0.05 - $0.50+ depending on model)
- **2 R2 operations** (essentially free within normal usage)

The database operations cost a fraction of a cent. The external API calls dominate the cost by 3-4 orders of magnitude. This is why AI video platforms obsess over API costs, not database costs.

---

## Conclusion: Schema Principles for AI Video

After building and iterating on these schemas, here are the principles I have landed on:

1. **Flat collections over subcollections.** The flexibility of cross-collection queries far outweighs the convenience of nested paths.

2. **Denormalize aggressively for read paths.** Your users spend 90% of their time reading, not writing. Optimize for reads even if it means more complex write logic.

3. **Store IDs up the chain.** Every document should contain the IDs of all its ancestors (userId on generations, projectId on assets). The storage cost is negligible; the query flexibility is invaluable.

4. **Pre-compute analytics.** Dashboard queries should hit pre-computed aggregation documents, not scan raw event collections.

5. **Database for metadata, R2 for blobs.** Never store binary data in Firestore. Store a reference and keep the blob in object storage.

6. **Plan for the relational migration.** Even if you start on Firestore, design your schema as if it might need to become relational. Avoid Firestore-specific patterns (like deeply nested maps) that would be painful to migrate.

7. **Version your schema.** Add a `_schemaVersion` field from day one. Your future self will thank you.

The data layer is the foundation. Get it right, and every feature you build on top will be easier. Get it wrong, and you will spend months paying down technical debt instead of shipping features. Take the time to think through your entities, your query patterns, and your denormalization strategy before you write your first document.
