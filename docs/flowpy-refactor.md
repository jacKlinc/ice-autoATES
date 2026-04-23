# FlowPy Vectorisation — Parked Refactor Idea

## Background

FlowPy (archived: `avaframe/FlowPy`) is the avalanche routing engine used in AutoATES v2.
Benchmarking shows it takes ~13 min per area for ~15k release pixels at 30m resolution.
At 177 areas this is ~38 hours total — a meaningful bottleneck for dataset generation.

The core bottleneck is `calculation_effect` in `flow_core.py`, which uses a dynamic
BFS-style graph traversal implemented entirely in Python loops.

## Why the Loops Can't be Trivially Vectorised

The algorithm grows `cell_list` dynamically — each cell spawns up to 8 neighbours
whose inclusion depends on DEM values, flux threshold, and alpha at runtime.
The topology (which cells are visited, in what order, with what flux) is unknown
until the algorithm runs.

The "shift-by-1" trick that vectorises linear recurrences (e.g. AR(1), cumulative sum)
doesn't transfer because:
- Dependencies are over an irregular spatial graph, not a 1D sequence
- Paths branch (one cell → up to 8 children) and converge (multiple paths → same cell)
- Path length varies per release pixel
- The inner duplicate-check loop is O(n²) over a dynamically growing list

## The Vectorisable Approach: Level-Synchronous BFS

The analogy that *does* apply is **level-synchronous BFS**, used in vectorised flow
routing tools (TauDEM, D-infinity, richdem):

1. **Sort all release cells by elevation** (already done in `get_start_idx`)
2. For each wave front (cells at the same BFS depth), compute flux and z_delta
   updates as array operations over all cells in that wave simultaneously
3. Propagate to the next wave front
4. Terminate when flux drops below threshold or z_delta exceeds max_z

All cells within a wave are independent of each other — this is the vectorisable unit.

## Key Implementation Challenges

- **Convergence detection**: the current inner loop checks whether a downstream cell
  was already reached by a different path (O(n²) dict lookup). A vectorised version
  needs a sparse adjacency representation (COO or CSR matrix) to handle merges.
- **Flux splitting**: FlowPy distributes flux across multiple downhill neighbours using
  the steepest-descent weighting. This is a local neighbourhood operation and maps
  well to `scipy.ndimage` or sliding-window convolutions.
- **z_delta tracking**: `np.maximum.at` or scatter-reduce operations replace the
  per-cell max updates.

## Likely Performance Gains

- Eliminating Python-level loops: 10–100× (typical numpy vs pure Python)
- Replacing O(n²) duplicate check with O(1) dict/array indexing: significant for
  high-density release areas
- Potential for GPU acceleration (CuPy drop-in) once fully vectorised

## Scope

This is a full algorithmic rewrite, not a refactor. The existing `flow_core.py`
interface (`calculation_effect`, `split_release`) could be preserved as the public API
so `build_dataset.py` doesn't need to change.

Reference implementations to study:
- `richdem` — C++/Cython flow accumulation, Python bindings
- TauDEM D-infinity — parallel flow direction/accumulation
- `pysheds` — pure Python but vectorised flow routing with numpy
