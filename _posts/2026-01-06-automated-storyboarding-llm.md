---
layout: post
title: "Automated Storyboarding: From Script to Shot List to Generated Video with LLM-Powered Scene Decomposition"
date: 2026-01-06
category: architecture
---

You have a user prompt: "A product demo showing our app in action." You need to turn this into a 45-second video with establishing shots, close-ups, transitions, and a coherent narrative arc. A human storyboard artist would spend hours on this. Your platform needs to do it in under ten seconds.

This post presents a complete automated storyboarding pipeline -- from unstructured user input through LLM-powered scene decomposition, structured scene graph representation, parallel generation scheduling, and final video assembly. Every component has working code. Every optimization has the math behind it.

The core insight: storyboarding is a *structured generation* problem. The LLM doesn't just write better prompts -- it decomposes an intention into a formally structured plan that downstream systems can execute deterministically.

---

## Table of Contents

1. [The Storyboarding Problem](#the-storyboarding-problem)
2. [LLM Decomposition: System Prompt and Few-Shot Design](#llm-decomposition)
3. [Scene Graph Representation](#scene-graph-representation)
4. [Character and Asset Tracking](#character-and-asset-tracking)
5. [Camera Language Generation](#camera-language-generation)
6. [Duration Estimation](#duration-estimation)
7. [Parallel Generation Planning](#parallel-generation-planning)
8. [Quality-Cost Optimization: Model Routing per Scene](#quality-cost-optimization)
9. [Full Implementation](#full-implementation)
10. [Pipeline Architecture Diagram](#pipeline-architecture-diagram)

---

## The Storyboarding Problem {#the-storyboarding-problem}

Professional filmmaking never starts with "let's just point the camera and see what happens." Every shot is planned: framing, camera movement, lighting, duration, the relationship between one shot and the next. The language of cinema has rules -- the 180-degree rule, shot/reverse-shot, the rule of thirds, the rhythm of cuts that creates pacing.

When a user writes "A product demo showing our app in action," they're implicitly asking for all of this. They want:

- An **establishing shot** that sets context (someone at a desk, a workspace, a device)
- **Close-ups** showing the product interface
- **Reaction shots** showing the user's response
- **Transitions** that feel natural rather than jarring
- **Pacing** that matches professional demo videos (not too fast, not too slow)
- **A narrative arc**: setup, demonstration, payoff

The gap between "a product demo" and a seven-scene storyboard with specific camera angles, lens lengths, lighting setups, and timing is enormous. Bridging this gap is the job of the storyboard pipeline.

### Why Not Just Use Better Prompts?

You could try to cram everything into a single video generation prompt: "Start with a wide shot of a person at a desk, then zoom into the laptop screen showing the app, hold for 3 seconds on the interface..." But this fails for three reasons:

1. **Current models can't handle compositional prompts.** Even the best models (Veo 3.1, Kling 3.0) generate 5-15 second clips. You can't describe a multi-scene narrative in a single prompt and expect coherent output.

2. **No temporal control.** You can't specify "spend the first 2 seconds on the establishing shot, then 4 seconds on the close-up." Models don't parse temporal structure from text.

3. **No consistency guarantees.** Even if a model could generate a 60-second video from a long prompt, characters and settings would drift. The person in frame 1 wouldn't look like the person in frame 300.

The solution is decomposition: break the user's intention into discrete scenes, generate each scene with a focused prompt, and stitch the results together. The storyboard is the intermediary representation that makes this possible.

---

## LLM Decomposition: System Prompt and Few-Shot Design {#llm-decomposition}

The core of the pipeline is an LLM call (I use Gemini 2.0 Flash for speed, but any capable model works) that transforms an unstructured user prompt into a structured storyboard. The quality of this decomposition determines the quality of the final video.

### The Output Format

Every storyboard is a JSON array of scenes, each with precise parameters:

```typescript
interface StoryboardScene {
  sceneNumber: number;
  description: string;         // detailed visual description for the video model
  camera: {
    shotType: 'extreme-wide' | 'wide' | 'medium-wide' | 'medium' | 'medium-close-up' | 'close-up' | 'extreme-close-up';
    lensLength: string;        // e.g., "35mm", "50mm", "85mm"
    movement: 'static' | 'pan-left' | 'pan-right' | 'tilt-up' | 'tilt-down' | 'dolly-in' | 'dolly-out' | 'tracking' | 'crane-up' | 'crane-down' | 'orbit' | 'handheld';
    movementSpeed: 'slow' | 'medium' | 'fast';
    angle: 'eye-level' | 'low-angle' | 'high-angle' | 'bird-eye' | 'dutch' | 'overhead';
  };
  lighting: {
    type: 'natural' | 'studio' | 'dramatic' | 'soft' | 'hard' | 'backlit' | 'silhouette' | 'golden-hour' | 'neon';
    mood: string;              // e.g., "warm and inviting", "cool and professional"
  };
  duration: number;            // seconds
  transition: {
    type: 'cut' | 'dissolve' | 'fade-in' | 'fade-out' | 'wipe' | 'match-cut' | 'j-cut' | 'l-cut';
    duration: number;          // transition duration in seconds
  };
  characters: string[];        // character IDs present in this scene
  props: string[];             // significant objects
  location: string;            // location identifier for consistency
  audio: {
    dialogue: string | null;   // spoken text, if any
    ambientSound: string;      // description of ambient audio
    music: string;             // music direction
  };
  priority: 'hero' | 'supporting' | 'filler';  // for quality routing
}
```

### The System Prompt

This is the full system prompt I use in production. It includes the role definition, output format specification, cinematography rules, and few-shot examples. The few-shot examples are the most important part -- they teach the LLM the level of specificity and the filmmaking grammar we expect.

```typescript
const STORYBOARD_SYSTEM_PROMPT = `You are an expert cinematographer and storyboard artist working for an AI video generation platform. Your job is to decompose a user's video idea into a structured shot list that can be executed by AI video generation models.

## Your Task

Given a user's description of a video they want, produce a detailed storyboard as a JSON array. Each element represents one shot/scene.

## Rules

1. SHOT VARIETY: Alternate between shot types. Never use the same shot type for consecutive scenes. A typical sequence: wide → medium → close-up → medium-wide → close-up → wide.

2. DURATION: Total video duration should be 30-60 seconds unless the user specifies otherwise. Individual shots should be 3-8 seconds. Action shots are shorter (2-4s), establishing shots are longer (4-6s), dialogue shots match the speech duration.

3. CAMERA MOVEMENT: Use movement purposefully. Establishing shots benefit from slow dollies or cranes. Close-ups are usually static or with minimal movement. Action scenes use tracking shots. Never specify fast camera movement on a close-up.

4. 180-DEGREE RULE: When two characters interact, keep the camera on one side of the action line. This means: if Character A is on the left in one shot, they should be on the left in subsequent shots of the same interaction.

5. TRANSITIONS: Use cuts for most transitions. Dissolves for time passage. Fade-in for the first shot, fade-out for the last shot. Match cuts when two scenes share visual similarity. Don't overuse dissolves.

6. LIGHTING CONSISTENCY: Scenes in the same location should have consistent lighting descriptions.

7. NARRATIVE ARC: Even a simple product demo has structure: setup (context), development (demonstration), conclusion (impact/CTA). Your shot list should follow this arc.

8. DESCRIPTIONS: Write descriptions as if instructing a video model. Be specific about what's visible, the composition, and the mood. Include details about the subject's appearance, the environment, colors, and textures.

9. AUDIO: Include dialogue only when the user's concept implies speaking. Ambient sound should match the environment. Music direction should specify genre and energy level, not specific tracks.

10. PRIORITY: Mark the most important shots (the "hero" shots that sell the video) as 'hero'. These will be routed to higher-quality generation models. Supporting shots get standard quality. Filler shots (transitions, B-roll) can use the fastest/cheapest model.

## Output Format

Return ONLY a valid JSON array. No markdown, no explanation, no preamble. Each element must conform to the StoryboardScene schema.

## Example 1

User: "A morning coffee routine in a cozy apartment"

[
  {
    "sceneNumber": 1,
    "description": "Warm morning sunlight streaming through large apartment windows with sheer curtains. A cozy living space with plants, books, and warm wood tones visible in soft focus. Dust motes float in the golden light beams.",
    "camera": {
      "shotType": "wide",
      "lensLength": "35mm",
      "movement": "slow dolly-in",
      "movementSpeed": "slow",
      "angle": "eye-level"
    },
    "lighting": {
      "type": "golden-hour",
      "mood": "warm, peaceful, inviting morning light"
    },
    "duration": 5,
    "transition": {
      "type": "fade-in",
      "duration": 1.5
    },
    "characters": ["person-1"],
    "props": ["apartment-windows", "plants", "books"],
    "location": "cozy-apartment",
    "audio": {
      "dialogue": null,
      "ambientSound": "Quiet morning ambience, distant birds, soft city sounds",
      "music": "Gentle acoustic guitar, lo-fi warmth, minimal and contemplative"
    },
    "priority": "hero"
  },
  {
    "sceneNumber": 2,
    "description": "Close-up of hands reaching for a ceramic coffee mug on a wooden bedside table. The hands are relaxed, just waking up. A small succulent plant and a phone sit beside the mug. Steam is not yet rising — the mug is from last night.",
    "camera": {
      "shotType": "close-up",
      "lensLength": "85mm",
      "movement": "static",
      "movementSpeed": "slow",
      "angle": "overhead"
    },
    "lighting": {
      "type": "natural",
      "mood": "soft directional morning light from window, warm tones"
    },
    "duration": 3,
    "transition": {
      "type": "cut",
      "duration": 0
    },
    "characters": ["person-1"],
    "props": ["coffee-mug", "bedside-table", "succulent", "phone"],
    "location": "cozy-apartment",
    "audio": {
      "dialogue": null,
      "ambientSound": "Soft rustling of sheets, the ceramic mug against wood",
      "music": "Continuation of acoustic guitar, slightly building"
    },
    "priority": "hero"
  },
  {
    "sceneNumber": 3,
    "description": "Medium shot of a person walking into a small kitchen, seen from behind. They wear an oversized sweater and soft pants. The kitchen has open shelving with jars and plants. Morning light fills the space. They reach for a bag of coffee beans on the counter.",
    "camera": {
      "shotType": "medium",
      "lensLength": "50mm",
      "movement": "tracking",
      "movementSpeed": "slow",
      "angle": "eye-level"
    },
    "lighting": {
      "type": "natural",
      "mood": "bright morning kitchen light, clean and warm"
    },
    "duration": 4,
    "transition": {
      "type": "cut",
      "duration": 0
    },
    "characters": ["person-1"],
    "props": ["kitchen-counter", "coffee-beans", "shelving"],
    "location": "cozy-apartment-kitchen",
    "audio": {
      "dialogue": null,
      "ambientSound": "Bare feet on wooden floor, the rustle of a coffee bag",
      "music": "Acoustic guitar continues, adding light percussion"
    },
    "priority": "supporting"
  },
  {
    "sceneNumber": 4,
    "description": "Extreme close-up of dark coffee beans being poured from a bag into a hand grinder. The beans are glossy and richly colored. Shallow depth of field — only the beans in the center of the grinder are in sharp focus.",
    "camera": {
      "shotType": "extreme-close-up",
      "lensLength": "100mm macro",
      "movement": "static",
      "movementSpeed": "slow",
      "angle": "high-angle"
    },
    "lighting": {
      "type": "natural",
      "mood": "warm side-lighting emphasizing bean texture and gloss"
    },
    "duration": 3,
    "transition": {
      "type": "cut",
      "duration": 0
    },
    "characters": [],
    "props": ["coffee-beans", "hand-grinder"],
    "location": "cozy-apartment-kitchen",
    "audio": {
      "dialogue": null,
      "ambientSound": "Beans rattling into the grinder, satisfying clatter",
      "music": "Music continues, now with a light shaker percussion"
    },
    "priority": "hero"
  },
  {
    "sceneNumber": 5,
    "description": "Medium close-up of a pour-over coffee setup. Hot water pours from a gooseneck kettle in a thin, controlled stream onto coffee grounds in a paper filter. Steam rises visibly. The coffee blooms and darkens. The pour is slow and deliberate.",
    "camera": {
      "shotType": "medium-close-up",
      "lensLength": "50mm",
      "movement": "static",
      "movementSpeed": "slow",
      "angle": "eye-level"
    },
    "lighting": {
      "type": "backlit",
      "mood": "steam illuminated by backlight, creating a halo effect"
    },
    "duration": 5,
    "transition": {
      "type": "dissolve",
      "duration": 0.8
    },
    "characters": ["person-1"],
    "props": ["gooseneck-kettle", "pour-over", "paper-filter"],
    "location": "cozy-apartment-kitchen",
    "audio": {
      "dialogue": null,
      "ambientSound": "Water pouring, gentle bubbling, steam hissing",
      "music": "Music builds to a warm, full arrangement"
    },
    "priority": "hero"
  },
  {
    "sceneNumber": 6,
    "description": "Wide shot of the person sitting in a comfortable armchair by the window, holding the steaming coffee mug with both hands. They look out the window with a content expression. Plants frame the shot on both sides. The entire scene is bathed in golden morning light.",
    "camera": {
      "shotType": "wide",
      "lensLength": "35mm",
      "movement": "slow dolly-out",
      "movementSpeed": "slow",
      "angle": "eye-level"
    },
    "lighting": {
      "type": "golden-hour",
      "mood": "warm, peaceful, satisfying golden light"
    },
    "duration": 5,
    "transition": {
      "type": "cut",
      "duration": 0
    },
    "characters": ["person-1"],
    "props": ["coffee-mug", "armchair", "window", "plants"],
    "location": "cozy-apartment",
    "audio": {
      "dialogue": null,
      "ambientSound": "Distant city sounds, birds, a clock ticking softly",
      "music": "Music resolves to a gentle, satisfied conclusion"
    },
    "priority": "hero"
  },
  {
    "sceneNumber": 7,
    "description": "Close-up of the coffee mug resting on the armchair's arm, steam rising in slow curls. The person's hand rests beside it. Focus shifts slowly from the mug to the blurred window view of the city beyond. A moment of stillness.",
    "camera": {
      "shotType": "close-up",
      "lensLength": "85mm",
      "movement": "static",
      "movementSpeed": "slow",
      "angle": "eye-level"
    },
    "lighting": {
      "type": "golden-hour",
      "mood": "final golden warmth, soft and reflective"
    },
    "duration": 4,
    "transition": {
      "type": "fade-out",
      "duration": 2
    },
    "characters": ["person-1"],
    "props": ["coffee-mug", "armchair"],
    "location": "cozy-apartment",
    "audio": {
      "dialogue": null,
      "ambientSound": "Almost silence, just the faintest ambient hum",
      "music": "Final chord fades out with the image"
    },
    "priority": "supporting"
  }
]

## Example 2

User: "Show our project management tool helping a team collaborate"

[
  {
    "sceneNumber": 1,
    "description": "Modern open-plan office with large monitors, whiteboards covered in sticky notes, and a team of four people visible at different workstations. Natural light from floor-to-ceiling windows. The space feels productive and energetic but not chaotic.",
    "camera": {
      "shotType": "wide",
      "lensLength": "24mm",
      "movement": "crane-down",
      "movementSpeed": "slow",
      "angle": "high-angle"
    },
    "lighting": {
      "type": "natural",
      "mood": "bright, professional, energetic daylight"
    },
    "duration": 4,
    "transition": {
      "type": "fade-in",
      "duration": 1
    },
    "characters": ["team-member-1", "team-member-2", "team-member-3", "team-member-4"],
    "props": ["monitors", "whiteboards", "sticky-notes"],
    "location": "modern-office",
    "audio": {
      "dialogue": null,
      "ambientSound": "Light office buzz, keyboards, distant conversation",
      "music": "Upbeat corporate, building energy, modern and clean"
    },
    "priority": "supporting"
  },
  {
    "sceneNumber": 2,
    "description": "Medium close-up of a person's hands on a laptop keyboard and trackpad. The laptop screen shows a Kanban board interface with colorful task cards. The person drags a card from 'In Progress' to 'Review'. The interface is clean and modern with a dark sidebar and bright accent colors.",
    "camera": {
      "shotType": "medium-close-up",
      "lensLength": "50mm",
      "movement": "static",
      "movementSpeed": "slow",
      "angle": "high-angle"
    },
    "lighting": {
      "type": "soft",
      "mood": "screen-lit face with soft ambient fill"
    },
    "duration": 5,
    "transition": {
      "type": "cut",
      "duration": 0
    },
    "characters": ["team-member-1"],
    "props": ["laptop", "kanban-board-interface"],
    "location": "modern-office",
    "audio": {
      "dialogue": null,
      "ambientSound": "Mouse click, subtle UI sound effect",
      "music": "Continues building, adds light synth melody"
    },
    "priority": "hero"
  }
]

(Continue for 5-7 total scenes following the same structure...)

Now generate the storyboard for the user's prompt.`;
```

### Calling the LLM

```typescript
import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

async function decomposeToStoryboard(
  userPrompt: string,
  options: { targetDuration?: number; shotCount?: number; style?: string } = {}
): Promise<StoryboardScene[]> {
  const { targetDuration = 45, shotCount, style } = options;

  let userMessage = userPrompt;
  if (targetDuration) {
    userMessage += `\n\nTarget total duration: ${targetDuration} seconds.`;
  }
  if (shotCount) {
    userMessage += `\nTarget shot count: ${shotCount} scenes.`;
  }
  if (style) {
    userMessage += `\nVisual style direction: ${style}.`;
  }

  const response = await ai.models.generateContent({
    model: 'gemini-2.0-flash',
    contents: [{ role: 'user', parts: [{ text: userMessage }] }],
    config: {
      systemInstruction: STORYBOARD_SYSTEM_PROMPT,
      responseMimeType: 'application/json',
      temperature: 0.7,   // some creative latitude
      maxOutputTokens: 8192,
    },
  });

  const raw = response.text;
  if (!raw) throw new Error('Empty response from storyboard LLM');

  const scenes: StoryboardScene[] = JSON.parse(raw);

  // Validate and fix
  return validateStoryboard(scenes, targetDuration);
}
```

### Validation and Correction

The LLM's output is good but not perfect. Validation catches structural errors and enforces constraints:

```typescript
function validateStoryboard(
  scenes: StoryboardScene[],
  targetDuration: number
): StoryboardScene[] {
  // 1. Ensure scene numbers are sequential
  scenes.forEach((s, i) => { s.sceneNumber = i + 1; });

  // 2. Enforce shot type variety
  for (let i = 1; i < scenes.length; i++) {
    if (scenes[i].camera.shotType === scenes[i - 1].camera.shotType) {
      // Swap with the nearest scene that has a different shot type
      const swap = scenes.slice(i + 1).findIndex(
        s => s.camera.shotType !== scenes[i - 1].camera.shotType
      );
      if (swap >= 0) {
        [scenes[i], scenes[i + 1 + swap]] = [scenes[i + 1 + swap], scenes[i]];
        scenes.forEach((s, idx) => { s.sceneNumber = idx + 1; });
      }
    }
  }

  // 3. Adjust durations to hit target
  const totalDuration = scenes.reduce((sum, s) => sum + s.duration, 0);
  if (Math.abs(totalDuration - targetDuration) > 3) {
    const scale = targetDuration / totalDuration;
    scenes.forEach(s => {
      s.duration = Math.max(2, Math.min(8, Math.round(s.duration * scale)));
    });
  }

  // 4. Ensure first scene has fade-in, last has fade-out
  scenes[0].transition = { type: 'fade-in', duration: 1 };
  scenes[scenes.length - 1].transition = { type: 'fade-out', duration: 1.5 };

  // 5. Validate required fields
  scenes.forEach(s => {
    if (!s.description) throw new Error(`Scene ${s.sceneNumber} missing description`);
    if (!s.camera?.shotType) throw new Error(`Scene ${s.sceneNumber} missing camera.shotType`);
    if (!s.duration || s.duration <= 0) s.duration = 4;
    if (!s.priority) s.priority = 'supporting';
    if (!s.characters) s.characters = [];
    if (!s.props) s.props = [];
    if (!s.location) s.location = 'default';
  });

  return scenes;
}
```

---

## Scene Graph Representation {#scene-graph-representation}

A storyboard isn't just a list of scenes -- it's a **directed graph** where nodes are scenes and edges encode the relationships between them. Modeling it as a graph lets us reason about narrative flow, detect consistency requirements, and optimize generation order.

### Formal Definition

A scene graph $G = (V, E)$ where:

- $V = \{s_1, s_2, \ldots, s_n\}$ is the set of scenes
- $E \subseteq V \times V$ is the set of directed edges representing transitions

Each edge $e_{ij} = (s_i, s_j)$ has attributes:
- **Transition type** $\tau(e_{ij}) \in \{\text{cut}, \text{dissolve}, \text{fade}, \text{wipe}, \text{match-cut}\}$
- **Transition duration** $\delta(e_{ij}) \in \mathbb{R}^+$

The total video duration is:

$$T = \sum_{i=1}^{n} d(s_i) + \sum_{(i,j) \in E} \delta(e_{ij}) \cdot \mathbb{1}[\tau(e_{ij}) \neq \text{cut}]$$

where $d(s_i)$ is the duration of scene $i$. Cuts have zero transition time; dissolves and fades overlap adjacent scenes.

### Shared Entity Edges

Beyond sequential transitions, we add **consistency edges** that connect scenes sharing characters, props, or locations:

$$E_{\text{consistency}} = \{(s_i, s_j) \mid i \neq j \land (\text{chars}(s_i) \cap \text{chars}(s_j) \neq \emptyset \lor \text{loc}(s_i) = \text{loc}(s_j))\}$$

These edges impose constraints: connected scenes must maintain visual consistency for the shared entities. The consistency subgraph tells us which scenes can be generated independently and which must share conditioning information.

### Graph Properties

Several graph properties are useful for pipeline optimization:

**Connected components by character**: If character "person-1" appears in scenes 1, 3, 5, and 7, those scenes form a consistency cluster. All scenes in a cluster should use the same reference image and generation parameters for that character.

**Topological ordering**: The sequential edge structure defines a total order, but consistency edges create additional constraints. Scenes that share characters with earlier scenes should be generated after the reference frames are established.

**Longest path**: In a DAG with durations as weights, the longest path gives the minimum wall-clock generation time (assuming unlimited parallelism for independent scenes).

```typescript
interface SceneGraph {
  scenes: StoryboardScene[];
  sequentialEdges: Array<{
    from: number;
    to: number;
    transition: StoryboardScene['transition'];
  }>;
  consistencyEdges: Array<{
    from: number;
    to: number;
    sharedEntities: string[];  // character/location IDs
  }>;
}

function buildSceneGraph(scenes: StoryboardScene[]): SceneGraph {
  const sequentialEdges = scenes.slice(0, -1).map((s, i) => ({
    from: s.sceneNumber,
    to: scenes[i + 1].sceneNumber,
    transition: scenes[i + 1].transition,
  }));

  const consistencyEdges: SceneGraph['consistencyEdges'] = [];
  for (let i = 0; i < scenes.length; i++) {
    for (let j = i + 1; j < scenes.length; j++) {
      const sharedChars = scenes[i].characters.filter(
        c => scenes[j].characters.includes(c)
      );
      const sharedLocation = scenes[i].location === scenes[j].location
        ? [scenes[i].location]
        : [];
      const sharedEntities = [...sharedChars, ...sharedLocation];
      if (sharedEntities.length > 0) {
        consistencyEdges.push({
          from: scenes[i].sceneNumber,
          to: scenes[j].sceneNumber,
          sharedEntities,
        });
      }
    }
  }

  return { scenes, sequentialEdges, consistencyEdges };
}
```

### Scene Graph Visualization

<svg viewBox="0 0 900 420" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;background:#ffffff;font-family:Inter,system-ui,sans-serif;">
  <!-- Title -->
  <text x="450" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a1a1a">Scene Graph: Sequential + Consistency Edges</text>

  <!-- Scene nodes -->
  <rect x="30" y="160" width="90" height="60" rx="8" fill="#4fc3f7" stroke="#0288d1" stroke-width="2"/>
  <text x="75" y="185" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Scene 1</text>
  <text x="75" y="200" text-anchor="middle" font-size="9" fill="#1a1a1a">Wide, 5s</text>
  <text x="75" y="212" text-anchor="middle" font-size="8" fill="#333">person-1</text>

  <rect x="160" y="160" width="90" height="60" rx="8" fill="#4fc3f7" stroke="#0288d1" stroke-width="2"/>
  <text x="205" y="185" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Scene 2</text>
  <text x="205" y="200" text-anchor="middle" font-size="9" fill="#1a1a1a">CU, 3s</text>
  <text x="205" y="212" text-anchor="middle" font-size="8" fill="#333">person-1</text>

  <rect x="290" y="160" width="90" height="60" rx="8" fill="#4fc3f7" stroke="#0288d1" stroke-width="2"/>
  <text x="335" y="185" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Scene 3</text>
  <text x="335" y="200" text-anchor="middle" font-size="9" fill="#1a1a1a">Med, 4s</text>
  <text x="335" y="212" text-anchor="middle" font-size="8" fill="#333">person-1</text>

  <rect x="420" y="160" width="90" height="60" rx="8" fill="#ffa726" stroke="#e65100" stroke-width="2"/>
  <text x="465" y="185" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Scene 4</text>
  <text x="465" y="200" text-anchor="middle" font-size="9" fill="#1a1a1a">ECU, 3s</text>
  <text x="465" y="212" text-anchor="middle" font-size="8" fill="#333">no chars</text>

  <rect x="550" y="160" width="90" height="60" rx="8" fill="#4fc3f7" stroke="#0288d1" stroke-width="2"/>
  <text x="595" y="185" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Scene 5</text>
  <text x="595" y="200" text-anchor="middle" font-size="9" fill="#1a1a1a">MCU, 5s</text>
  <text x="595" y="212" text-anchor="middle" font-size="8" fill="#333">person-1</text>

  <rect x="680" y="160" width="90" height="60" rx="8" fill="#8bc34a" stroke="#558b2f" stroke-width="2"/>
  <text x="725" y="185" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Scene 6</text>
  <text x="725" y="200" text-anchor="middle" font-size="9" fill="#1a1a1a">Wide, 5s</text>
  <text x="725" y="212" text-anchor="middle" font-size="8" fill="#333">person-1</text>

  <!-- Sequential edges (straight arrows) -->
  <defs>
    <marker id="arrowS" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
    </marker>
    <marker id="arrowC" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ef5350"/>
    </marker>
  </defs>

  <line x1="120" y1="190" x2="158" y2="190" stroke="#333" stroke-width="2" marker-end="url(#arrowS)"/>
  <line x1="250" y1="190" x2="288" y2="190" stroke="#333" stroke-width="2" marker-end="url(#arrowS)"/>
  <line x1="380" y1="190" x2="418" y2="190" stroke="#333" stroke-width="2" marker-end="url(#arrowS)"/>
  <line x1="510" y1="190" x2="548" y2="190" stroke="#333" stroke-width="2" marker-end="url(#arrowS)"/>
  <line x1="640" y1="190" x2="678" y2="190" stroke="#333" stroke-width="2" marker-end="url(#arrowS)"/>

  <!-- Consistency edges (curved, red, dashed) -->
  <!-- Scene 1 to Scene 2 (person-1) -->
  <path d="M 75 160 Q 140 100 205 160" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3" fill="none" marker-end="url(#arrowC)"/>

  <!-- Scene 1 to Scene 3 (person-1) -->
  <path d="M 75 160 Q 200 70 335 160" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3" fill="none" marker-end="url(#arrowC)"/>

  <!-- Scene 2 to Scene 5 (person-1, same location) -->
  <path d="M 205 160 Q 400 55 595 160" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3" fill="none" marker-end="url(#arrowC)"/>

  <!-- Scene 3 to Scene 6 (person-1) -->
  <path d="M 335 160 Q 530 55 725 160" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3" fill="none" marker-end="url(#arrowC)"/>

  <!-- Scene 5 to Scene 6 (person-1, location) -->
  <path d="M 595 160 Q 660 110 725 160" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3" fill="none" marker-end="url(#arrowC)"/>

  <!-- Legend -->
  <rect x="250" y="280" width="400" height="120" rx="8" fill="#f5f5f5" stroke="#ccc" stroke-width="1"/>
  <text x="450" y="305" text-anchor="middle" font-size="13" font-weight="bold" fill="#1a1a1a">Legend</text>

  <line x1="280" y1="330" x2="340" y2="330" stroke="#333" stroke-width="2" marker-end="url(#arrowS)"/>
  <text x="355" y="335" font-size="11" fill="#1a1a1a">Sequential transition (cut/dissolve)</text>

  <line x1="280" y1="360" x2="340" y2="360" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arrowC)"/>
  <text x="355" y="365" font-size="11" fill="#1a1a1a">Consistency edge (shared entity)</text>

  <rect x="280" y="375" width="16" height="12" rx="3" fill="#4fc3f7"/>
  <text x="305" y="385" font-size="11" fill="#1a1a1a">Has characters</text>
  <rect x="430" y="375" width="16" height="12" rx="3" fill="#ffa726"/>
  <text x="455" y="385" font-size="11" fill="#1a1a1a">No characters (insert)</text>
</svg>

---

## Character and Asset Tracking {#character-and-asset-tracking}

The storyboard assigns character IDs like "person-1" and "team-member-3" to scenes, but these IDs are abstract. Before generation, we need to resolve them into concrete visual descriptions and track them for consistency.

### Entity Extraction

A second LLM call extracts all unique entities from the storyboard and generates detailed descriptions:

```typescript
const ENTITY_EXTRACTION_PROMPT = `Given the following storyboard JSON, extract all unique characters, significant props, and locations. For each entity, provide a detailed visual description that can be used to maintain consistency across video generation.

For characters: describe appearance (age range, build, hair color/style, skin tone), clothing (specific items, colors, textures), and any distinguishing features.

For props: describe size, color, material, brand style (generic).

For locations: describe architecture, lighting conditions, color palette, furnishings, atmosphere.

Return as JSON:
{
  "characters": {
    "character-id": {
      "description": "detailed visual description",
      "scenes": [1, 3, 5],
      "referencePrompt": "prompt fragment for consistent generation"
    }
  },
  "props": { ... },
  "locations": { ... }
}`;

interface EntityRegistry {
  characters: Record<string, {
    description: string;
    scenes: number[];
    referencePrompt: string;
    referenceImageUrl?: string;  // generated or uploaded reference
    faceEmbedding?: number[];    // for face consistency checking
  }>;
  props: Record<string, {
    description: string;
    scenes: number[];
  }>;
  locations: Record<string, {
    description: string;
    scenes: number[];
    referenceImageUrl?: string;
  }>;
}

async function extractEntities(
  storyboard: StoryboardScene[]
): Promise<EntityRegistry> {
  const response = await ai.models.generateContent({
    model: 'gemini-2.0-flash',
    contents: [{
      role: 'user',
      parts: [{ text: JSON.stringify(storyboard, null, 2) }],
    }],
    config: {
      systemInstruction: ENTITY_EXTRACTION_PROMPT,
      responseMimeType: 'application/json',
      temperature: 0.3,
    },
  });

  return JSON.parse(response.text!);
}
```

### Generating Reference Images

For each character, we generate a reference image that will be used as conditioning for all scenes featuring that character:

```typescript
async function generateReferenceImages(
  entities: EntityRegistry
): Promise<EntityRegistry> {
  const characterEntries = Object.entries(entities.characters);

  const referencePromises = characterEntries.map(
    async ([id, character]) => {
      // Generate a neutral reference image: front-facing, well-lit, neutral expression
      const refPrompt = `Portrait photograph, front-facing, neutral expression, studio lighting, plain gray background. ${character.description}. Photorealistic, high detail, sharp focus.`;

      const imageResult = await ai.models.generateImages({
        model: 'imagen-3.0-generate-002',
        prompt: refPrompt,
        config: {
          numberOfImages: 1,
          aspectRatio: '1:1',
        },
      });

      // Store the reference image
      const imageUrl = await uploadToStorage(imageResult.generatedImages[0].image);
      character.referenceImageUrl = imageUrl;

      return [id, character] as const;
    }
  );

  const results = await Promise.all(referencePromises);
  results.forEach(([id, char]) => {
    entities.characters[id] = char;
  });

  return entities;
}
```

### Consistency Injection

When generating each scene, the character reference images are injected into the prompt and (where supported) used as IP-Adapter conditioning:

```typescript
function buildScenePrompt(
  scene: StoryboardScene,
  entities: EntityRegistry
): string {
  let prompt = scene.description;

  // Append character consistency descriptions
  for (const charId of scene.characters) {
    const char = entities.characters[charId];
    if (char) {
      prompt += ` ${char.referencePrompt}`;
    }
  }

  // Append location consistency
  const loc = entities.locations[scene.location];
  if (loc) {
    prompt += ` Setting: ${loc.description}`;
  }

  // Append camera direction
  prompt += ` Camera: ${scene.camera.shotType} shot, ${scene.camera.lensLength} lens, ${scene.camera.movement}, ${scene.camera.angle}.`;

  // Append lighting
  prompt += ` Lighting: ${scene.lighting.type}, ${scene.lighting.mood}.`;

  return prompt;
}
```

---

## Camera Language Generation {#camera-language-generation}

One of the most common failures in AI-generated storyboards is the gap between how users describe shots and how video models interpret camera directions. Users say "show the product up close." Video models need "medium close-up, 50mm lens, slow push-in, product centered in frame, shallow depth of field."

### The Translation Table

This table maps common user intentions to specific cinematographic language. The storyboard LLM uses these mappings implicitly (trained by the few-shot examples), but they're also useful as a reference for validation:

| User Intent | Shot Type | Lens | Movement | Angle | Notes |
|---|---|---|---|---|---|
| "Show the whole scene" | Wide / Establishing | 24-35mm | Slow dolly or static | Eye-level or high | Use for scene-setting, first shots |
| "Show the product up close" | Medium close-up or Close-up | 50-85mm | Slow dolly-in or static | Eye-level or slightly high | Shallow DOF emphasizes subject |
| "Show someone's reaction" | Close-up | 85mm | Static | Eye-level | Tight framing, minimal movement |
| "Show detail/texture" | Extreme close-up | 100mm macro | Static | Varies | Very shallow DOF, fill lighting |
| "Follow the action" | Medium-wide | 35-50mm | Tracking or Steadicam | Eye-level | Smooth, continuous movement |
| "Dramatic reveal" | Starts tight, pulls wide | 35mm | Dolly-out or crane-up | Low or eye-level | Reveal context progressively |
| "Overhead view" | Top-down | 24mm | Static or slow rotation | Bird's-eye | Good for flat-lay, workspace |
| "Epic/powerful" | Low-angle wide | 24mm | Slow tilt-up or crane | Low-angle | Subject appears powerful, imposing |
| "Intimate/personal" | Medium close-up | 50mm | Handheld, slight | Eye-level | Subtle shake adds realism |
| "Professional/clean" | Medium | 50mm | Static or slow dolly | Eye-level | Stable, symmetric framing |
| "Time passing" | Wide to medium | 35mm | Static | Eye-level | Use dissolve transitions |
| "Energetic/fast" | Various, quick cuts | 24-35mm | Tracking or handheld | Varies | Short duration (2-3s per shot) |

### Shot Composition Rules

Beyond individual shot parameters, the sequence of shots matters. Here are the composition rules encoded in the validation step:

```typescript
const COMPOSITION_RULES = {
  // Don't jump from extreme close-up to extreme wide (or vice versa)
  maxJumpDistance: 3, // in shot type ordinal distance

  // Shot type ordinal values for measuring jumps
  shotOrdinal: {
    'extreme-wide': 0,
    'wide': 1,
    'medium-wide': 2,
    'medium': 3,
    'medium-close-up': 4,
    'close-up': 5,
    'extreme-close-up': 6,
  } as Record<string, number>,

  // Preferred transitions for different jump types
  transitionForJump: (distance: number): string => {
    if (distance <= 1) return 'cut';
    if (distance <= 2) return 'cut';
    if (distance <= 3) return 'dissolve';
    return 'dissolve'; // large jumps need dissolves to smooth the shift
  },
};

function validateComposition(scenes: StoryboardScene[]): string[] {
  const warnings: string[] = [];

  for (let i = 1; i < scenes.length; i++) {
    const prevOrdinal = COMPOSITION_RULES.shotOrdinal[scenes[i - 1].camera.shotType] ?? 3;
    const currOrdinal = COMPOSITION_RULES.shotOrdinal[scenes[i].camera.shotType] ?? 3;
    const jump = Math.abs(currOrdinal - prevOrdinal);

    if (jump > COMPOSITION_RULES.maxJumpDistance) {
      warnings.push(
        `Scenes ${i} → ${i + 1}: shot jump distance ${jump} exceeds max ${COMPOSITION_RULES.maxJumpDistance}. ` +
        `Consider inserting a transition shot between ${scenes[i - 1].camera.shotType} and ${scenes[i].camera.shotType}.`
      );
    }
  }

  return warnings;
}
```

---

## Duration Estimation {#duration-estimation}

How long should each scene be? This question has a surprisingly well-studied answer in film theory. The **average shot length** (ASL) varies by genre, era, and individual filmmaker, but the patterns are consistent enough to build heuristics.

### Average Shot Length by Genre

Research by Barry Salt and David Bordwell on thousands of films gives us these baselines:

| Genre / Content Type | Average Shot Length | Range |
|---|---|---|
| Action / Fast-paced | 2.0 - 3.0 seconds | 1 - 5s |
| Drama / Dialogue | 4.0 - 6.0 seconds | 2 - 12s |
| Documentary | 5.0 - 8.0 seconds | 3 - 15s |
| Music video | 1.5 - 3.0 seconds | 0.5 - 5s |
| Product demo | 3.0 - 5.0 seconds | 2 - 8s |
| Establishing / B-roll | 4.0 - 6.0 seconds | 3 - 10s |
| Social media (short-form) | 1.5 - 3.0 seconds | 0.5 - 5s |
| Commercial / Ad | 2.0 - 4.0 seconds | 1 - 6s |
| Tutorial / Educational | 5.0 - 10.0 seconds | 3 - 20s |

### The Pacing Model

Good pacing isn't uniform -- it follows a rhythm. Action scenes use shorter shots to build tension; establishing shots use longer durations to let the viewer orient. We can model this with a pacing function:

$$d(s_i) = d_{\text{base}}(\text{type}(s_i)) \cdot \rho(i, n)$$

where $d_{\text{base}}$ is the base duration for the content type, and $\rho(i, n)$ is a pacing modifier based on the scene's position $i$ in a storyboard of $n$ total scenes:

$$\rho(i, n) = 1 + \alpha \cdot \sin\left(\frac{\pi \cdot i}{n}\right)$$

This creates an arc: slightly longer shots at the beginning (setup), building through the middle, and tapering at the end. The parameter $\alpha$ controls how much the pacing varies (typically $\alpha \in [0.1, 0.3]$).

For a 7-scene storyboard with $\alpha = 0.2$ and all base durations of 4 seconds:

| Scene | Position $i/n$ | $\sin(\pi i / n)$ | $\rho$ | Duration |
|---|---|---|---|---|
| 1 | 0.14 | 0.43 | 1.09 | 4.3s |
| 2 | 0.29 | 0.78 | 1.16 | 4.6s |
| 3 | 0.43 | 0.97 | 1.19 | 4.8s |
| 4 | 0.57 | 0.97 | 1.19 | 4.8s |
| 5 | 0.71 | 0.78 | 1.16 | 4.6s |
| 6 | 0.86 | 0.43 | 1.09 | 4.3s |
| 7 | 1.00 | 0.00 | 1.00 | 4.0s |

Total: 31.4 seconds. The middle scenes are slightly longer, creating a natural rhythm that mirrors the narrative arc.

```typescript
function estimateDurations(
  scenes: StoryboardScene[],
  targetDuration: number,
  contentStyle: string = 'product-demo'
): StoryboardScene[] {
  const baseDurations: Record<string, Record<string, number>> = {
    'product-demo': {
      'extreme-wide': 4, 'wide': 5, 'medium-wide': 4,
      'medium': 4, 'medium-close-up': 4, 'close-up': 3, 'extreme-close-up': 3,
    },
    'action': {
      'extreme-wide': 3, 'wide': 3, 'medium-wide': 2.5,
      'medium': 2.5, 'medium-close-up': 2, 'close-up': 2, 'extreme-close-up': 1.5,
    },
    'narrative': {
      'extreme-wide': 5, 'wide': 6, 'medium-wide': 5,
      'medium': 5, 'medium-close-up': 4, 'close-up': 4, 'extreme-close-up': 3,
    },
  };

  const bases = baseDurations[contentStyle] || baseDurations['product-demo'];
  const alpha = 0.2;
  const n = scenes.length;

  // Calculate raw durations with pacing curve
  scenes.forEach((scene, i) => {
    const base = bases[scene.camera.shotType] || 4;
    const rho = 1 + alpha * Math.sin((Math.PI * (i + 1)) / n);
    scene.duration = Math.round(base * rho * 10) / 10;
  });

  // Scale to target duration
  const rawTotal = scenes.reduce((sum, s) => sum + s.duration, 0);
  const scale = targetDuration / rawTotal;
  scenes.forEach(s => {
    s.duration = Math.max(2, Math.round(s.duration * scale * 10) / 10);
  });

  return scenes;
}
```

---

## Parallel Generation Planning {#parallel-generation-planning}

Given a storyboard with $N$ scenes and $M$ available model instances, how do we schedule generation to minimize total wall-clock time? This is a variant of the **makespan minimization** problem on parallel machines, which is NP-hard in general but admits good heuristics for our specific constraints.

### Problem Formulation

We have:
- $N$ scenes (jobs) $J = \{j_1, \ldots, j_N\}$
- $M$ model instances (machines) $P = \{p_1, \ldots, p_M\}$
- Processing time $t_{ij}$ for scene $j_i$ on machine $p_j$ (varies by model)
- Precedence constraints from the scene graph's consistency edges

The objective is to minimize the **makespan** $C_{\max}$:

$$\min C_{\max} = \min \max_{k \in P} \sum_{j \in S_k} t_{kj}$$

where $S_k$ is the set of scenes assigned to machine $k$.

Subject to precedence constraints: if scene $j_a$ must generate before scene $j_b$ (because $j_b$ uses the last frame of $j_a$ as its start frame for consistency), then $\text{finish}(j_a) \leq \text{start}(j_b)$.

### Which Scenes Can Run in Parallel?

Two scenes can generate simultaneously if and only if:
1. Neither depends on the other's output (no frame-chaining requirement)
2. They don't both need the same model instance (resource constraint)

In practice, most scenes *can* run in parallel. The exceptions are:
- **Frame-chained scenes**: Scene $i+1$ starts from the last frame of scene $i$
- **Same-character first appearances**: The first scene containing a character should generate first to establish the reference

### The LPT Heuristic

The **Longest Processing Time first** (LPT) heuristic is simple and effective: sort scenes by expected generation time (descending), then assign each scene to the machine with the earliest available time.

For identical machines, LPT guarantees a makespan within $\frac{4}{3}$ of optimal:

$$C_{\max}^{\text{LPT}} \leq \frac{4}{3} \cdot C_{\max}^{\text{OPT}}$$

For our case with heterogeneous machines (different models with different speeds), we adapt LPT by considering the best machine for each scene:

```typescript
interface GenerationJob {
  sceneNumber: number;
  scene: StoryboardScene;
  estimatedDuration: number;  // generation time in seconds
  modelId: string;
  dependencies: number[];     // scene numbers that must complete first
}

interface ScheduleEntry {
  job: GenerationJob;
  machineId: string;
  startTime: number;
  endTime: number;
}

function scheduleGeneration(
  scenes: StoryboardScene[],
  graph: SceneGraph,
  availableModels: Array<{ id: string; estimatedTimePerSecond: number; maxConcurrent: number }>
): ScheduleEntry[] {
  // Step 1: Estimate generation time for each scene on each model
  const jobs: GenerationJob[] = scenes.map(scene => {
    // Find the best model for this scene
    const bestModel = availableModels.reduce((best, model) => {
      const time = scene.duration * model.estimatedTimePerSecond;
      return time < best.time ? { model, time } : best;
    }, { model: availableModels[0], time: Infinity });

    // Find dependencies from consistency edges
    const deps = graph.consistencyEdges
      .filter(e => e.to === scene.sceneNumber && e.sharedEntities.some(
        ent => ent.startsWith('person') || ent.startsWith('team')
      ))
      .map(e => e.from);

    return {
      sceneNumber: scene.sceneNumber,
      scene,
      estimatedDuration: bestModel.time,
      modelId: bestModel.model.id,
      dependencies: deps,
    };
  });

  // Step 2: Topological sort respecting dependencies
  const sorted = topologicalSort(jobs);

  // Step 3: LPT scheduling with dependencies
  // Machine availability: machineId → earliest available time
  const machineAvailability: Record<string, number> = {};
  availableModels.forEach(m => {
    for (let i = 0; i < m.maxConcurrent; i++) {
      machineAvailability[`${m.id}-${i}`] = 0;
    }
  });

  const schedule: ScheduleEntry[] = [];
  const completionTimes: Record<number, number> = {};

  for (const job of sorted) {
    // Earliest start time based on dependencies
    const depFinish = job.dependencies.length > 0
      ? Math.max(...job.dependencies.map(d => completionTimes[d] || 0))
      : 0;

    // Find the machine with earliest availability (after dependency constraint)
    const candidateMachines = Object.entries(machineAvailability)
      .filter(([id]) => id.startsWith(job.modelId))
      .sort(([, a], [, b]) => a - b);

    if (candidateMachines.length === 0) {
      // Fallback: use any available machine
      const [machineId, availTime] = Object.entries(machineAvailability)
        .sort(([, a], [, b]) => a - b)[0];

      const startTime = Math.max(availTime, depFinish);
      const endTime = startTime + job.estimatedDuration;

      schedule.push({ job, machineId, startTime, endTime });
      machineAvailability[machineId] = endTime;
      completionTimes[job.sceneNumber] = endTime;
    } else {
      const [machineId, availTime] = candidateMachines[0];
      const startTime = Math.max(availTime, depFinish);
      const endTime = startTime + job.estimatedDuration;

      schedule.push({ job, machineId, startTime, endTime });
      machineAvailability[machineId] = endTime;
      completionTimes[job.sceneNumber] = endTime;
    }
  }

  return schedule;
}

function topologicalSort(jobs: GenerationJob[]): GenerationJob[] {
  const inDegree = new Map<number, number>();
  const adjacency = new Map<number, number[]>();

  jobs.forEach(j => {
    inDegree.set(j.sceneNumber, j.dependencies.length);
    j.dependencies.forEach(dep => {
      if (!adjacency.has(dep)) adjacency.set(dep, []);
      adjacency.get(dep)!.push(j.sceneNumber);
    });
  });

  const queue = jobs.filter(j => j.dependencies.length === 0);
  const result: GenerationJob[] = [];
  const jobMap = new Map(jobs.map(j => [j.sceneNumber, j]));

  // Sort queue by estimated duration (descending) for LPT
  queue.sort((a, b) => b.estimatedDuration - a.estimatedDuration);

  while (queue.length > 0) {
    const job = queue.shift()!;
    result.push(job);

    const successors = adjacency.get(job.sceneNumber) || [];
    for (const succ of successors) {
      const newDeg = (inDegree.get(succ) || 1) - 1;
      inDegree.set(succ, newDeg);
      if (newDeg === 0) {
        queue.push(jobMap.get(succ)!);
        queue.sort((a, b) => b.estimatedDuration - a.estimatedDuration);
      }
    }
  }

  return result;
}
```

### Scheduling Gantt Chart

<svg viewBox="0 0 900 380" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;background:#ffffff;font-family:Inter,system-ui,sans-serif;">
  <!-- Title -->
  <text x="450" y="25" text-anchor="middle" font-size="16" font-weight="bold" fill="#1a1a1a">Generation Schedule: 7 Scenes across 3 Model Instances</text>

  <!-- Y-axis labels (machines) -->
  <text x="15" y="85" font-size="11" fill="#1a1a1a" font-weight="bold">Veo 3.1</text>
  <text x="15" y="145" font-size="11" fill="#1a1a1a" font-weight="bold">Runway</text>
  <text x="15" y="148" font-size="9" fill="#666">(instance 1)</text>
  <text x="15" y="205" font-size="11" fill="#1a1a1a" font-weight="bold">Runway</text>
  <text x="15" y="208" font-size="9" fill="#666">(instance 2)</text>

  <!-- Timeline axis -->
  <line x1="100" y1="240" x2="860" y2="240" stroke="#333" stroke-width="1"/>
  <text x="100" y="260" font-size="10" fill="#666" text-anchor="middle">0s</text>
  <text x="253" y="260" font-size="10" fill="#666" text-anchor="middle">30s</text>
  <text x="405" y="260" font-size="10" fill="#666" text-anchor="middle">60s</text>
  <text x="557" y="260" font-size="10" fill="#666" text-anchor="middle">90s</text>
  <text x="710" y="260" font-size="10" fill="#666" text-anchor="middle">120s</text>
  <text x="860" y="260" font-size="10" fill="#666" text-anchor="middle">150s</text>

  <!-- Tick marks -->
  <line x1="100" y1="237" x2="100" y2="243" stroke="#333" stroke-width="1"/>
  <line x1="253" y1="237" x2="253" y2="243" stroke="#333" stroke-width="1"/>
  <line x1="405" y1="237" x2="405" y2="243" stroke="#333" stroke-width="1"/>
  <line x1="557" y1="237" x2="557" y2="243" stroke="#333" stroke-width="1"/>
  <line x1="710" y1="237" x2="710" y2="243" stroke="#333" stroke-width="1"/>
  <line x1="860" y1="237" x2="860" y2="243" stroke="#333" stroke-width="1"/>

  <!-- Grid lines -->
  <line x1="100" y1="60" x2="100" y2="240" stroke="#e0e0e0" stroke-width="0.5"/>
  <line x1="253" y1="60" x2="253" y2="240" stroke="#e0e0e0" stroke-width="0.5"/>
  <line x1="405" y1="60" x2="405" y2="240" stroke="#e0e0e0" stroke-width="0.5"/>
  <line x1="557" y1="60" x2="557" y2="240" stroke="#e0e0e0" stroke-width="0.5"/>
  <line x1="710" y1="60" x2="710" y2="240" stroke="#e0e0e0" stroke-width="0.5"/>

  <!-- Machine rows (horizontal bands) -->
  <rect x="100" y="60" width="760" height="50" fill="#fafafa" stroke="#eee"/>
  <rect x="100" y="120" width="760" height="50" fill="#f5f5f5" stroke="#eee"/>
  <rect x="100" y="180" width="760" height="50" fill="#fafafa" stroke="#eee"/>

  <!-- Veo 3.1: Scene 1 (hero, dialogue) 0-90s -->
  <rect x="100" y="65" width="456" height="40" rx="4" fill="#4fc3f7" stroke="#0288d1" stroke-width="1.5"/>
  <text x="328" y="88" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a1a1a">Scene 1 (hero, 5s clip)</text>
  <text x="328" y="100" text-anchor="middle" font-size="9" fill="#333">~90s gen time</text>

  <!-- Veo 3.1: Scene 6 (hero) 90-150s -->
  <rect x="560" y="65" width="253" height="40" rx="4" fill="#8bc34a" stroke="#558b2f" stroke-width="1.5"/>
  <text x="686" y="88" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a1a1a">Scene 6 (hero, 5s)</text>
  <text x="686" y="100" text-anchor="middle" font-size="9" fill="#333">~50s gen</text>

  <!-- Runway 1: Scene 2 (hero) 0-25s -->
  <rect x="100" y="125" width="127" height="40" rx="4" fill="#4fc3f7" stroke="#0288d1" stroke-width="1.5"/>
  <text x="164" y="148" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a1a1a">Scene 2</text>
  <text x="164" y="160" text-anchor="middle" font-size="9" fill="#333">~25s</text>

  <!-- Runway 1: Scene 4 (filler) 25-45s -->
  <rect x="230" y="125" width="101" height="40" rx="4" fill="#ffa726" stroke="#e65100" stroke-width="1.5"/>
  <text x="280" y="148" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a1a1a">Scene 4</text>
  <text x="280" y="160" text-anchor="middle" font-size="9" fill="#333">~20s</text>

  <!-- Runway 1: Scene 7 (supporting) 45-70s -->
  <rect x="334" y="125" width="127" height="40" rx="4" fill="#8bc34a" stroke="#558b2f" stroke-width="1.5"/>
  <text x="398" y="148" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a1a1a">Scene 7</text>
  <text x="398" y="160" text-anchor="middle" font-size="9" fill="#333">~25s</text>

  <!-- Runway 2: Scene 3 (supporting) 0-30s -->
  <rect x="100" y="185" width="152" height="40" rx="4" fill="#4fc3f7" stroke="#0288d1" stroke-width="1.5"/>
  <text x="176" y="208" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a1a1a">Scene 3</text>
  <text x="176" y="220" text-anchor="middle" font-size="9" fill="#333">~30s</text>

  <!-- Runway 2: Scene 5 (hero) 30-60s -->
  <rect x="255" y="185" width="152" height="40" rx="4" fill="#ef5350" stroke="#c62828" stroke-width="1.5"/>
  <text x="331" y="208" text-anchor="middle" font-size="11" font-weight="bold" fill="#1a1a1a">Scene 5</text>
  <text x="331" y="220" text-anchor="middle" font-size="9" fill="#333">~30s</text>

  <!-- Makespan line -->
  <line x1="813" y1="55" x2="813" y2="245" stroke="#ef5350" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="813" y="280" text-anchor="middle" font-size="11" font-weight="bold" fill="#ef5350">Makespan: ~140s</text>
  <text x="450" y="300" text-anchor="middle" font-size="10" fill="#666">Sequential would take ~270s. Parallel saves 48%.</text>

  <!-- Legend -->
  <rect x="250" y="315" width="400" height="55" rx="6" fill="#f5f5f5" stroke="#ccc"/>
  <rect x="270" y="330" width="14" height="10" rx="2" fill="#4fc3f7"/>
  <text x="290" y="339" font-size="10" fill="#1a1a1a">Character present</text>
  <rect x="400" y="330" width="14" height="10" rx="2" fill="#ffa726"/>
  <text x="420" y="339" font-size="10" fill="#1a1a1a">No characters</text>
  <rect x="520" y="330" width="14" height="10" rx="2" fill="#ef5350"/>
  <text x="540" y="339" font-size="10" fill="#1a1a1a">Hero scene</text>
  <line x1="270" y1="355" x2="290" y2="355" stroke="#ef5350" stroke-width="2" stroke-dasharray="5,3"/>
  <text x="300" y="359" font-size="10" fill="#1a1a1a">Makespan boundary</text>
</svg>

---

## Quality-Cost Optimization: Model Routing per Scene {#quality-cost-optimization}

Not every scene needs the most expensive model. An extreme close-up of coffee beans doesn't need dialogue audio. An establishing wide shot doesn't need the highest visual fidelity. By routing each scene to the optimal model based on its requirements, we can reduce total cost by 40-60% without visible quality loss.

### The Routing Decision

For each scene, we evaluate a **value score** that balances quality and cost:

$$V(s, m) = \frac{Q(s, m)}{C(s, m)}$$

where $Q(s, m)$ is the expected quality of scene $s$ on model $m$, and $C(s, m)$ is the cost.

Quality is estimated from the scene's requirements and the model's capability profile:

$$Q(s, m) = w_{\text{visual}} \cdot q_{\text{visual}}(m) + w_{\text{audio}} \cdot q_{\text{audio}}(s, m) + w_{\text{consistency}} \cdot q_{\text{consistency}}(m) + w_{\text{priority}} \cdot p(s)$$

where:
- $q_{\text{visual}}(m)$ is the model's visual quality score (from benchmarks or internal metrics)
- $q_{\text{audio}}(s, m)$ is 0 if the scene has no dialogue requirement, or the model's audio quality if it does
- $q_{\text{consistency}}(m)$ measures how well the model handles reference image conditioning
- $p(s) \in \{1.0, 0.6, 0.3\}$ for hero, supporting, and filler priority levels

### Routing Table

Given current model capabilities (early 2026), here's the optimal routing:

| Scene Type | Best Model | Why | Cost/sec |
|---|---|---|---|
| Dialogue with audio | Veo 3.1 | Best native audio quality | $0.30 |
| Hero visual (no audio) | Runway Gen-4.5 Aleph | Highest visual quality benchmark | $0.15 |
| Supporting visual | Runway Gen-4.5 Turbo | Good quality, fastest, cheapest | $0.05 |
| Close-up product shot | Runway Gen-4.5 Aleph | Detail fidelity matters | $0.15 |
| Establishing/B-roll | Wan 2.2 (self-hosted) | Cheapest, good enough for wide shots | $0.02 |
| Character consistency critical | Kling 3.0 (via PiAPI) | Multi-image reference, native multi-shot | $0.08 |

### Implementation

```typescript
interface ModelRoute {
  modelId: string;
  reason: string;
  estimatedCost: number;
  estimatedQuality: number;
}

function routeScene(
  scene: StoryboardScene,
  modelRegistry: Record<string, ModelRegistryEntry>
): ModelRoute {
  const needsAudio = scene.audio.dialogue !== null;
  const isHero = scene.priority === 'hero';
  const isFiller = scene.priority === 'filler';
  const hasCharacters = scene.characters.length > 0;

  // Priority 1: Dialogue scenes need audio-capable models
  if (needsAudio) {
    return {
      modelId: 'veo-31-standard',
      reason: 'Scene requires dialogue audio; Veo 3.1 has best audio quality',
      estimatedCost: scene.duration * 0.30,
      estimatedQuality: 0.92,
    };
  }

  // Priority 2: Hero scenes get premium visual quality
  if (isHero && !needsAudio) {
    return {
      modelId: 'runway-gen45-aleph',
      reason: 'Hero scene; Aleph provides highest visual fidelity',
      estimatedCost: scene.duration * 0.15,
      estimatedQuality: 0.95,
    };
  }

  // Priority 3: Filler/B-roll scenes use cheapest option
  if (isFiller) {
    return {
      modelId: 'wan-22-selfhosted',
      reason: 'Filler scene; open-source model minimizes cost',
      estimatedCost: scene.duration * 0.02,
      estimatedQuality: 0.70,
    };
  }

  // Priority 4: Character-heavy supporting scenes
  if (hasCharacters) {
    return {
      modelId: 'runway-gen45-turbo',
      reason: 'Supporting scene with characters; Turbo balances quality and speed',
      estimatedCost: scene.duration * 0.05,
      estimatedQuality: 0.85,
    };
  }

  // Default: Turbo for everything else
  return {
    modelId: 'runway-gen45-turbo',
    reason: 'Default supporting scene routing',
    estimatedCost: scene.duration * 0.05,
    estimatedQuality: 0.85,
  };
}

function estimateTotalCost(
  scenes: StoryboardScene[],
  modelRegistry: Record<string, ModelRegistryEntry>
): { totalCost: number; breakdown: ModelRoute[] } {
  const breakdown = scenes.map(s => routeScene(s, modelRegistry));
  const totalCost = breakdown.reduce((sum, r) => sum + r.estimatedCost, 0);
  return { totalCost, breakdown };
}
```

### Cost Comparison

For a 7-scene, 30-second storyboard:

| Strategy | Total Cost | Notes |
|---|---|---|
| All Veo 3.1 | $9.00 | Highest quality, highest cost |
| All Runway Aleph | $4.50 | High quality, no audio |
| All Runway Turbo | $1.50 | Fast, cheap, good enough |
| Intelligent routing | $2.80 | Hero on Aleph, dialogue on Veo, filler on Wan |
| Intelligent + Wan for B-roll | $1.90 | Maximum optimization |

Intelligent routing saves 50-70% versus using the best model for everything, with minimal perceptible quality loss in the final stitched video.

---

## Full Implementation {#full-implementation}

Here's the complete end-to-end pipeline: user prompt in, finished video out.

```typescript
import { GoogleGenAI } from '@google/genai';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as path from 'path';

const execAsync = promisify(exec);

// ──────────────────────────────────────────
// Pipeline Configuration
// ──────────────────────────────────────────

interface PipelineConfig {
  targetDuration: number;
  outputResolution: '720p' | '1080p';
  style?: string;
  maxConcurrentGenerations: number;
  outputDir: string;
  geminiApiKey: string;
}

// ──────────────────────────────────────────
// Main Pipeline
// ──────────────────────────────────────────

async function generateVideoFromPrompt(
  userPrompt: string,
  config: PipelineConfig
): Promise<string> {
  const ai = new GoogleGenAI({ apiKey: config.geminiApiKey });
  const startTime = Date.now();

  console.log('[pipeline] Starting storyboard generation...');

  // Step 1: Decompose prompt into storyboard
  const storyboard = await decomposeToStoryboard(userPrompt, {
    targetDuration: config.targetDuration,
    style: config.style,
  });
  console.log(`[pipeline] Storyboard: ${storyboard.length} scenes, ${
    storyboard.reduce((s, sc) => s + sc.duration, 0)
  }s total`);

  // Step 2: Extract entities and generate references
  const entities = await extractEntities(storyboard);
  console.log(`[pipeline] Entities: ${
    Object.keys(entities.characters).length
  } characters, ${
    Object.keys(entities.locations).length
  } locations`);

  // Step 3: Generate reference images for characters
  const entitiesWithRefs = await generateReferenceImages(entities);
  console.log('[pipeline] Reference images generated');

  // Step 4: Build scene graph
  const graph = buildSceneGraph(storyboard);
  console.log(`[pipeline] Scene graph: ${
    graph.consistencyEdges.length
  } consistency edges`);

  // Step 5: Route each scene to optimal model
  const routes = storyboard.map(s => routeScene(s, MODEL_REGISTRY));
  const totalCost = routes.reduce((sum, r) => sum + r.estimatedCost, 0);
  console.log(`[pipeline] Estimated cost: $${totalCost.toFixed(2)}`);

  // Step 6: Schedule parallel generation
  const schedule = scheduleGeneration(storyboard, graph, AVAILABLE_MODELS);
  const makespan = Math.max(...schedule.map(s => s.endTime));
  console.log(`[pipeline] Estimated generation time: ${makespan}s`);

  // Step 7: Execute generation
  const clips = await executeGeneration(
    storyboard,
    entitiesWithRefs,
    routes,
    schedule,
    config
  );
  console.log(`[pipeline] ${clips.length} clips generated`);

  // Step 8: Stitch clips into final video
  const outputPath = await stitchVideo(clips, storyboard, config);
  console.log(`[pipeline] Output: ${outputPath}`);

  const elapsed = (Date.now() - startTime) / 1000;
  console.log(`[pipeline] Total pipeline time: ${elapsed.toFixed(1)}s`);

  return outputPath;
}

// ──────────────────────────────────────────
// Generation Execution
// ──────────────────────────────────────────

interface GeneratedClip {
  sceneNumber: number;
  filePath: string;
  duration: number;
  modelUsed: string;
}

async function executeGeneration(
  storyboard: StoryboardScene[],
  entities: EntityRegistry,
  routes: ModelRoute[],
  schedule: ScheduleEntry[],
  config: PipelineConfig
): Promise<GeneratedClip[]> {
  const clips: GeneratedClip[] = [];
  const completedScenes = new Set<number>();

  // Group schedule entries by wave (scenes that can run in parallel)
  const waves = groupIntoWaves(schedule);

  for (const wave of waves) {
    // Wait for any dependencies to complete
    for (const entry of wave) {
      const deps = entry.job.dependencies;
      if (deps.some(d => !completedScenes.has(d))) {
        // This shouldn't happen with correct wave grouping
        throw new Error(
          `Scene ${entry.job.sceneNumber} has unmet dependencies`
        );
      }
    }

    // Generate all scenes in this wave concurrently
    const waveResults = await Promise.allSettled(
      wave.map(async (entry) => {
        const scene = entry.job.scene;
        const prompt = buildScenePrompt(scene, entities);
        const route = routes[scene.sceneNumber - 1];

        // Get character reference images for IP-Adapter conditioning
        const referenceImages = scene.characters
          .map(cId => entities.characters[cId]?.referenceImageUrl)
          .filter(Boolean) as string[];

        // Call the appropriate model adapter
        const clip = await generateWithModel(
          route.modelId,
          prompt,
          {
            duration: scene.duration,
            resolution: config.outputResolution,
            referenceImages,
            camera: scene.camera,
          }
        );

        const filePath = path.join(
          config.outputDir,
          `scene-${scene.sceneNumber}.mp4`
        );
        await fs.writeFile(filePath, clip);

        return {
          sceneNumber: scene.sceneNumber,
          filePath,
          duration: scene.duration,
          modelUsed: route.modelId,
        };
      })
    );

    // Collect results, handle failures
    for (const result of waveResults) {
      if (result.status === 'fulfilled') {
        clips.push(result.value);
        completedScenes.add(result.value.sceneNumber);
      } else {
        console.error(`[pipeline] Scene generation failed:`, result.reason);
        // TODO: retry with fallback model
      }
    }
  }

  // Sort clips by scene number
  clips.sort((a, b) => a.sceneNumber - b.sceneNumber);
  return clips;
}

function groupIntoWaves(
  schedule: ScheduleEntry[]
): ScheduleEntry[][] {
  // Group entries that start at the same time (or close to it)
  const sorted = [...schedule].sort((a, b) => a.startTime - b.startTime);
  const waves: ScheduleEntry[][] = [];
  let currentWave: ScheduleEntry[] = [];
  let currentStart = -1;

  for (const entry of sorted) {
    if (currentStart === -1 || entry.startTime <= currentStart + 1) {
      currentWave.push(entry);
      if (currentStart === -1) currentStart = entry.startTime;
    } else {
      waves.push(currentWave);
      currentWave = [entry];
      currentStart = entry.startTime;
    }
  }
  if (currentWave.length > 0) waves.push(currentWave);

  return waves;
}

// ──────────────────────────────────────────
// Video Stitching with FFmpeg
// ──────────────────────────────────────────

async function stitchVideo(
  clips: GeneratedClip[],
  storyboard: StoryboardScene[],
  config: PipelineConfig
): Promise<string> {
  // Build FFmpeg filter graph for transitions
  const filterParts: string[] = [];
  const inputs = clips.map((c, i) => `-i "${c.filePath}"`).join(' ');

  // Step 1: Normalize all clips to same resolution and framerate
  clips.forEach((_, i) => {
    filterParts.push(
      `[${i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,` +
      `pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v${i}]`
    );
  });

  // Step 2: Apply transitions between clips
  let currentStream = 'v0';
  for (let i = 1; i < clips.length; i++) {
    const scene = storyboard[i];
    const transition = scene.transition;
    const outputLabel = `vt${i}`;

    if (transition.type === 'dissolve' && transition.duration > 0) {
      // Cross-dissolve: overlap the end of prev clip with start of next
      const offsetSec = clips[i - 1].duration - transition.duration;
      filterParts.push(
        `[${currentStream}][v${i}]xfade=transition=dissolve:` +
        `duration=${transition.duration}:offset=${offsetSec}[${outputLabel}]`
      );
    } else if (transition.type === 'fade-in' && i === 0) {
      filterParts.push(
        `[v${i}]fade=t=in:st=0:d=${transition.duration}[${outputLabel}]`
      );
    } else {
      // Default: hard cut via concat
      filterParts.push(
        `[${currentStream}][v${i}]concat=n=2:v=1:a=0[${outputLabel}]`
      );
    }

    currentStream = outputLabel;
  }

  // Step 3: Fade out on last clip
  const lastScene = storyboard[storyboard.length - 1];
  if (lastScene.transition.type === 'fade-out') {
    const totalDur = clips.reduce((s, c) => s + c.duration, 0);
    const fadeStart = totalDur - lastScene.transition.duration;
    filterParts.push(
      `[${currentStream}]fade=t=out:st=${fadeStart}:` +
      `d=${lastScene.transition.duration}[vout]`
    );
    currentStream = 'vout';
  }

  const filterGraph = filterParts.join(';\n');
  const outputPath = path.join(config.outputDir, 'final-output.mp4');

  const ffmpegCmd = `ffmpeg -y ${inputs} ` +
    `-filter_complex "${filterGraph}" ` +
    `-map "[${currentStream}]" ` +
    `-c:v libx264 -preset medium -crf 18 ` +
    `"${outputPath}"`;

  await execAsync(ffmpegCmd, { timeout: 120000 });
  return outputPath;
}
```

### Usage

```typescript
const outputPath = await generateVideoFromPrompt(
  'A product demo showing our project management app helping a team collaborate',
  {
    targetDuration: 45,
    outputResolution: '1080p',
    style: 'professional, modern, clean, bright office setting',
    maxConcurrentGenerations: 4,
    outputDir: '/tmp/video-output',
    geminiApiKey: process.env.GEMINI_API_KEY!,
  }
);

console.log(`Video generated: ${outputPath}`);
```

---

## Pipeline Architecture Diagram {#pipeline-architecture-diagram}

<svg viewBox="0 0 900 700" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:900px;background:#ffffff;font-family:Inter,system-ui,sans-serif;">
  <!-- Title -->
  <text x="450" y="30" text-anchor="middle" font-size="18" font-weight="bold" fill="#1a1a1a">Automated Storyboard Pipeline Architecture</text>

  <!-- Arrow marker -->
  <defs>
    <marker id="pipeArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>
    </marker>
  </defs>

  <!-- Stage 1: User Input -->
  <rect x="350" y="50" width="200" height="45" rx="8" fill="#4fc3f7" stroke="#0288d1" stroke-width="2"/>
  <text x="450" y="77" text-anchor="middle" font-size="13" font-weight="bold" fill="#1a1a1a">User Prompt</text>

  <line x1="450" y1="95" x2="450" y2="118" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>

  <!-- Stage 2: LLM Decomposition -->
  <rect x="300" y="120" width="300" height="50" rx="8" fill="#ffa726" stroke="#e65100" stroke-width="2"/>
  <text x="450" y="142" text-anchor="middle" font-size="13" font-weight="bold" fill="#1a1a1a">LLM Decomposition</text>
  <text x="450" y="158" text-anchor="middle" font-size="10" fill="#333">Gemini 2.0 Flash + System Prompt</text>

  <!-- Two outputs from LLM -->
  <line x1="380" y1="170" x2="250" y2="200" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>
  <line x1="520" y1="170" x2="650" y2="200" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>

  <!-- Stage 3a: Storyboard JSON -->
  <rect x="130" y="200" width="240" height="50" rx="8" fill="#8bc34a" stroke="#558b2f" stroke-width="2"/>
  <text x="250" y="222" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Storyboard JSON</text>
  <text x="250" y="238" text-anchor="middle" font-size="10" fill="#333">N scenes with full parameters</text>

  <!-- Stage 3b: Entity Registry -->
  <rect x="530" y="200" width="240" height="50" rx="8" fill="#8bc34a" stroke="#558b2f" stroke-width="2"/>
  <text x="650" y="222" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Entity Registry</text>
  <text x="650" y="238" text-anchor="middle" font-size="10" fill="#333">Characters, props, locations</text>

  <!-- Arrows down from both -->
  <line x1="250" y1="250" x2="250" y2="278" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>
  <line x1="650" y1="250" x2="650" y2="278" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>

  <!-- Stage 4a: Scene Graph -->
  <rect x="130" y="280" width="240" height="50" rx="8" fill="#ffa726" stroke="#e65100" stroke-width="2"/>
  <text x="250" y="302" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Scene Graph Builder</text>
  <text x="250" y="318" text-anchor="middle" font-size="10" fill="#333">Sequential + consistency edges</text>

  <!-- Stage 4b: Reference Images -->
  <rect x="530" y="280" width="240" height="50" rx="8" fill="#ffa726" stroke="#e65100" stroke-width="2"/>
  <text x="650" y="302" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Reference Image Gen</text>
  <text x="650" y="318" text-anchor="middle" font-size="10" fill="#333">Imagen 3.0 per character</text>

  <!-- Both converge -->
  <line x1="250" y1="330" x2="380" y2="368" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>
  <line x1="650" y1="330" x2="520" y2="368" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>

  <!-- Stage 5: Scheduler + Router -->
  <rect x="280" y="370" width="340" height="55" rx="8" fill="#ef5350" stroke="#c62828" stroke-width="2"/>
  <text x="450" y="392" text-anchor="middle" font-size="13" font-weight="bold" fill="#ffffff">Scheduler + Model Router</text>
  <text x="450" y="410" text-anchor="middle" font-size="10" fill="#ffcdd2">LPT scheduling, quality-cost routing</text>

  <line x1="450" y1="425" x2="450" y2="448" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>

  <!-- Stage 6: Parallel Generation -->
  <rect x="100" y="450" width="700" height="70" rx="8" fill="#f5f5f5" stroke="#999" stroke-width="1.5"/>
  <text x="450" y="472" text-anchor="middle" font-size="13" font-weight="bold" fill="#1a1a1a">Parallel Generation</text>

  <!-- Model boxes inside -->
  <rect x="120" y="480" width="130" height="30" rx="5" fill="#4fc3f7" stroke="#0288d1"/>
  <text x="185" y="500" text-anchor="middle" font-size="10" font-weight="bold" fill="#1a1a1a">Veo 3.1 (audio)</text>

  <rect x="270" y="480" width="130" height="30" rx="5" fill="#4fc3f7" stroke="#0288d1"/>
  <text x="335" y="500" text-anchor="middle" font-size="10" font-weight="bold" fill="#1a1a1a">Runway Aleph</text>

  <rect x="420" y="480" width="130" height="30" rx="5" fill="#4fc3f7" stroke="#0288d1"/>
  <text x="485" y="500" text-anchor="middle" font-size="10" font-weight="bold" fill="#1a1a1a">Runway Turbo</text>

  <rect x="570" y="480" width="130" height="30" rx="5" fill="#4fc3f7" stroke="#0288d1"/>
  <text x="635" y="500" text-anchor="middle" font-size="10" font-weight="bold" fill="#1a1a1a">Wan 2.2 (self)</text>

  <!-- Down to quality gate -->
  <line x1="450" y1="520" x2="450" y2="548" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>

  <!-- Stage 7: Quality Gate -->
  <rect x="300" y="550" width="300" height="45" rx="8" fill="#ffa726" stroke="#e65100" stroke-width="2"/>
  <text x="450" y="572" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">Quality Gate + Consistency Check</text>
  <text x="450" y="585" text-anchor="middle" font-size="10" fill="#333">Gemini Flash visual QA</text>

  <line x1="450" y1="595" x2="450" y2="618" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>

  <!-- Stage 8: FFmpeg Stitch -->
  <rect x="300" y="620" width="300" height="45" rx="8" fill="#8bc34a" stroke="#558b2f" stroke-width="2"/>
  <text x="450" y="642" text-anchor="middle" font-size="12" font-weight="bold" fill="#1a1a1a">FFmpeg Stitching</text>
  <text x="450" y="655" text-anchor="middle" font-size="10" fill="#333">Transitions, fades, audio mixing</text>

  <line x1="450" y1="665" x2="450" y2="683" stroke="#333" stroke-width="2" marker-end="url(#pipeArrow)"/>

  <!-- Output -->
  <rect x="350" y="685" width="200" height="40" rx="20" fill="#4fc3f7" stroke="#0288d1" stroke-width="2"/>
  <text x="450" y="710" text-anchor="middle" font-size="13" font-weight="bold" fill="#1a1a1a">Final Video</text>

  <!-- Feedback loop arrow (quality gate failure → regenerate) -->
  <path d="M 600 572 L 750 572 L 750 485 L 705 485" stroke="#ef5350" stroke-width="1.5" stroke-dasharray="5,3" fill="none" marker-end="url(#pipeArrow)"/>
  <text x="760" y="530" font-size="9" fill="#ef5350" transform="rotate(-90, 760, 530)">Regenerate on failure</text>
</svg>

---

## Putting It Together: What This Unlocks

The automated storyboard pipeline transforms a platform's capabilities in three ways:

**1. Users stop thinking in clips.** Instead of crafting individual video prompts, users describe what they want at the concept level. "A 30-second product demo for our fitness app" becomes a seven-scene storyboard with establishing shots, close-ups, and a conclusion -- automatically.

**2. Quality goes up because the system applies filmmaking grammar.** Most users don't know about the 180-degree rule, shot type variety, or pacing curves. The storyboard LLM applies these constraints, producing video that *feels* professionally structured even if the user has no filmmaking knowledge.

**3. Cost goes down because of intelligent routing.** Not every scene needs the most expensive model. Routing filler shots to Wan 2.2 and hero shots to Runway Aleph reduces cost by 50-60% versus using a single premium model for everything.

The pipeline I've described here is what we're building at 10x News Digest. It's not perfect -- character consistency across independently generated scenes remains the hardest problem (more on this in the next post). But it's the foundation: get the structure right first, then solve consistency on top of it.

### What's Next

The next two posts in this series go deeper on the two hardest sub-problems:

- **IP-Adapter and Reference Image Conditioning** -- how cross-attention injection gives you visual control over what models generate
- **Character Consistency Across Shots** -- the mathematical framework and practical techniques for keeping characters looking the same across a multi-shot storyboard

The storyboard pipeline generates the plan. These techniques execute it faithfully.
