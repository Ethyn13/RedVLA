# Source and data provenance

RedVLA extracts the reusable evaluation code from the existing `adversarial` project. It provides physical placement optimization, constraint checks, interaction tracking, safety monitoring and video/event recording, with explicit configuration and model transport boundaries.

The 135-task manifest is derived from the existing task list. Threat-object assignments and safety-rule mappings are retained. The scene package contains the referenced BDDL and `.pruned_init` files, with SHA-256 hashes in `data/SHA256SUMS.json`.

The simulator dependency is **our red-libero project**, installed from its own checkout. Its Python package name is `libero`. RedVLA selects it using `RED_LIBERO_ROOT`, prepares an isolated resource configuration and records the selected paths in run manifests. Simulator code, meshes and textures belong to red-libero and are distributed with that project; RedVLA does not package a second simulator copy.

The safety monitor and parser originate from the VLA-Red-Teaming project. Existing MIT notices for VLA-Adapter and LIBERO remain in `LICENSE` and `licenses/`; red-libero retains its own `LICENSE` and source/asset notices. OpenPI inference uses its NumPy MessagePack protocol; model repositories and weights remain external dependencies with their respective licenses.
