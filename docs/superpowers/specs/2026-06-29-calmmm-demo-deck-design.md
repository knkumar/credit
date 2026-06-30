# calmmm Demo Deck Design

## Goal

Create an editable PowerPoint walkthrough that explains the `calmmm` package end to end using synthetic credit-marketing data.

## Deliverables

- `outputs/calmmm_demo_walkthrough.pptx`: editable PowerPoint deck.
- `outputs/calmmm_sample_weekly_panel.csv`: synthetic weekly panel data used by the deck.
- `outputs/calmmm_sample_lift_tests.csv`: synthetic incrementality experiment table used by the deck.

## Audience

Analytics, data science, or stakeholder reviewers who need to understand what the package does, what inputs it expects, and what outputs it produces.

## Content

The deck will use seven slides:

1. `calmmm` package overview.
2. Synthetic weekly panel data: weeks, geos, media channels, KPIs, controls.
3. End-to-end workflow: dataframe to `MMMData`, experiments, fit, outputs.
4. Calibration example: search geo-holdout lift test.
5. Model fit and validation: holdout metrics and posterior diagnostics.
6. Attribution outputs: contribution, marginal contribution, ROI, saturation.
7. How to run the package on real data.

## Visual Direction

Use a clean, editable PowerPoint style based on the bundled Codex Grid layout system: white canvas, black typography, light-gray structure, and one orange highlight. Use native text boxes, tables, simple process shapes, and charts rather than rasterized screenshots.

## Data Scope

The sample data will be synthetic and non-sensitive. It will represent a weekly credit-acquisition funnel with geographies, search/social/direct-mail/affiliate spend, impressions or clicks where relevant, price index, approval-rate proxy, visits, applications, approvals, and funded revenue.

## Verification

Render every final slide to image previews and inspect for clipping, overlap, unreadable text, broken charts, or unresolved placeholders. Export only after the deck renders cleanly.
