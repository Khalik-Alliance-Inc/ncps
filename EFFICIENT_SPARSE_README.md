# Efficient Sparse CfC Implementation

This implementation introduces an efficient sparse variant of the Closed-form Continuous-time (CfC) neural network that significantly reduces memory usage and computational overhead when using sparse wiring configurations. The implementation maintains full backward compatibility while providing substantial performance improvements for sparse networks.

## Problem

The original NCPs implementation uses a mask-based approach for sparsity:
- Allocates memory for ALL possible connections
- Multiplies by a binary mask to zero out non-existent connections
- Wastes memory and computation on connections that will always be zero

## Solution

This implementation:
- Only creates parameters for connections that exist in the wiring
- Uses edge-list representation for sparse computation
- Provides the same functionality with significant memory and speed improvements

## Usage

### Using the Efficient Implementation

Simply add `use_efficient=True` to your existing CfC usage:

```python
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# Create sparse wiring
wiring = AutoNCP(units=128, output_size=32, sparsity_level=0.7)

# Before (original implementation)
model = CfC(input_size=64, units=wiring)

# After (efficient sparse implementation)
model = CfC(
    input_size=64,
    units=wiring,
    use_efficient=True  # Enable efficient sparse implementation
)
```

### 🚀  Performance and Efficiency Enhancements
- **Substantial Parameter Reduction:** Delivers significant reductions in parameter count, closely mirroring the sparsity level of the wiring (e.g., 30-70% reduction observed for typical AutoNCP sparsity levels).
- **Accelerated Training:** Training can be accelerated due to fewer gradient calculations and updates, with potential speedups in epoch times, particularly noticeable in highly sparse configurations.
- **Quicker Inference:** Inference latency can be reduced by performing computations only on active pathways, leading to more efficient forward passes.
- **Efficient Memory Usage:** Employs sparse data structures for connectivity, leading to a more compact model representation in memory.
- 
### 🔧 Technical Highlights
- **Drop-in replacement**: Simply add `use_efficient=True` to existing CfC instantiations
- **Full backward compatibility**: Existing code continues to work unchanged
- **Supports all CfC modes**: Compatible with "default", "pure", and "no_gate" modes
- **Maintains gradient flow**: Proper backpropagation through sparse connections
- **Automatic sparsity detection**: Leverages wiring adjacency matrices

## Testing
 

The implementation includes a comprehensive test suite (`efficient_wiring_test.py`) that verifies:
- Output shape compatibility
- Parameter reduction achievements
- Gradient flow correctness
- Mode compatibility (default, pure, no_gate)
- Performance improvements

### Formulas and Scaling (AutoNCP and Efficient Wired)

- **AutoNCP splits**: `I_c = U − O`; `C ≈ 0.4 · I_c`; `I = I_c − C`; `D = 1 − sparsity`.
- **Fanouts**: `F_sens ≈ I · D`; `F_inter ≈ C · D`; `F_mot ≈ C · D`.
- **Sensory/prev nonzeros per layer**: `N0 ≈ input_dim · F_sens`; `N1 ≈ I · C · D`; `N2 ≈ O · C · D`.
- **Recurrent nonzeros per layer**:
  - Dense: `R_ℓ = h_ℓ^2`
  - Directed: `R0 = 0; R1 ≈ ⌊C · D · 2⌋; R2 = 0`
  - Symmetric: `R1 ∈ [⌊C · D · 2⌋, 2 · ⌊C · D · 2⌋]`
- **Total edges**: `E = Σ_ℓ (N_ℓ + R_ℓ)`.
- **Params per layer**: `default = 4 · E_ℓ + 4 · h_ℓ`; `pure = E_ℓ + 3 · h_ℓ`.

These reflect that parameter and compute scale with the number of active edges (non‑zeros), not dense matrix sizes.

## Implementation Details

### New Files:
- `ncps/torch/efficient_cfc_cell.py`: Efficient CfC cell implementation
- `ncps/torch/efficient_wired_cfc_cell.py`: Efficient wired CfC cell

### Key Changes:

Instead of allocating full weight matrices, the implementation:
1. Identifies non-zero connections from the wiring adjacency matrix
2. Allocates parameters only for existing connections
3. Uses efficient sparse operations (scatter_add) for computation
4. Recurrent connectivity is configurable; efficient wired now defaults to symmetric sparse within‑layer recurrence (A ∪ Aᵀ)
5. Uses edge list representation of connectivity
6. Sparse linear operations using gather/scatter instead of matrix multiply
7. Backward compatible - use_efficient=False gives original behavior


## Compatibility

- Fully backward compatible when `use_efficient=False` (default)
- Same API and functionality as original implementation
- Can switch between implementations for testing/comparison

### Recurrent Connectivity Options (Advanced)

- **Directed sparse (AutoNCP, most faithful)**:
  - Recurrence only where wiring defines it (primarily command layer).
  - Lowest parameter count; strictly directed graph.

- **Symmetric sparse (current default in efficient wired)**:
  - Recurrence where wiring had either direction: `A ∪ Aᵀ`.
  - Still sparse; slightly more edges than directed; can modestly smooth dynamics.

- **Dense (legacy wired behavior)**:
  - Full `h × h` within‑layer recurrence; highest parameter/compute cost; not aligned with NCP wiring.

Why symmetry can help

`A ∪ Aᵀ` keeps recurrence local to the intended wiring while allowing reciprocal pairs (biologically plausible). It often produces a more balanced recurrent operator, which may reduce directional oscillations. Primary stability still comes from CfC gating (time interpolation).

Performance implications

- Default (dense recurrence): recurrent edges per layer ≈ `h^2` (large).
- Directed sparse recurrence (AutoNCP): recurrent edges ≈ `⌊C · D · 2⌋` in command layer, ~0 elsewhere.
- Symmetric sparse recurrence (this change): recurrent edges ≈ `nnz(A ∪ Aᵀ)` ∈ `[⌊C · D · 2⌋, 2 · ⌊C · D · 2⌋]`.
