# Changelog

## 0.1.0

- Integrated external red-libero selection and isolated simulator resource configuration.
- Added YAML-selectable physical placement strategies and full-budget model evaluation examples.

- Added constrained CPU evaluation setup, a one-command reproduction wrapper, environment records and a scene/rule/config helper with optional XY object relocation and state export.
- Expanded English/Chinese setup, model integration and custom risk scene documentation.

- Extracted the framework into an installable `src/redvla` package with a single CLI.
- Added native OpenPI WebSocket inference with bounded timeouts.
- Added a versioned HTTP policy service for separate OpenVLA, VLA-Adapter and other model environments.
- Removed personal path defaults and simulation imports from client/serving code.
- Added portable configuration, independent episode/seed selection, suite-based model routing and strict failure behavior.
- Included the 135-task manifest, checksummed scene data and BDDL rules; the simulator is installed from the external red-libero project.
- Fixed stale observations after simulator reset, missed evaluation settings, cross-rollout policy reset and ambiguous output directories.
- Added configuration, preprocessing and wire-protocol regression tests, release tooling and bilingual documentation.
