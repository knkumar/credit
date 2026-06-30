# calmmm Demo Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an editable PowerPoint walkthrough and synthetic sample data for the `calmmm` package.

**Architecture:** Generate deterministic synthetic CSV inputs, then build a seven-slide editable PowerPoint with `@oai/artifact-tool`. Render slide previews and inspect the final deck before delivery.

**Tech Stack:** Node.js ES modules, `@oai/artifact-tool`, local repository docs, synthetic CSV files.

---

### Task 1: Generate Data And Deck

**Files:**
- Create: `outputs/calmmm_sample_weekly_panel.csv`
- Create: `outputs/calmmm_sample_lift_tests.csv`
- Create: `outputs/calmmm_demo_walkthrough.pptx`
- Create: external scratch builder under the presentation workspace
- Modify: `.agents/MEMORY.md`

- [ ] **Step 1: Generate deterministic sample data**

Create weekly panel rows for 78 weeks, 4 geos, 4 media channels, 4 KPIs, and realistic controls. Create lift-test rows for search and direct-mail experiments.

- [ ] **Step 2: Build the PowerPoint**

Use `@oai/artifact-tool` from a presentation scratch workspace. Create seven editable slides that match the approved design.

- [ ] **Step 3: Render and inspect**

Render all slides to PNG and a montage. Inspect layout JSON and previews for visible defects.

- [ ] **Step 4: Update project memory**

Append an `[OUTCOMES]` entry to `.agents/MEMORY.md` with generated artifact paths and verification status.

- [ ] **Step 5: Final response**

Return the final PPTX path and sample data paths.
