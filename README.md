# Lyman-alpha forest analysis with wavelet scattering transform
**A novel approach to extract non-Gaussian information from the Lyman-α forest for precision cosmology**


## Overview and Key Results


This repository implements the **Wavelet Scattering Transform (WST)** for analyzing Lyman-α forest data to constrain cosmological parameters. Our method demonstrates **order-of-magnitude improvements** over traditional flux power spectrum analysis, achieving constraint improvements of 30-60× across key cosmological parameters.

### Key Results

- **5σ detection capability** for minimal neutrino mass with DESI
- **Robust extraction** of non-Gaussian cosmological information
- **Novel summary statistic** that outperforms traditional methods

## Scientific Impact

Our Fisher matrix forecasts show that the WST captures substantial cosmological information missed by the flux power spectrum:

| Parameter | Power Spectrum | WST | Improvement Factor |
|-----------|----------------|-----|-------------------|
| nₚ | 0.178 | 0.0059 | **30×** |
| Aₚ | 0.503 | 0.0079 | **64×** |
| Hₛ | 0.207 | 0.0049 | **42×** |
| Hₐ | 0.258 | 0.0062 | **42×** |


## Installation

### Prerequisites
```bash
pip install numpy astropy matplotlib scipy h5py
pip install kymatio  # For wavelet scattering transforms
pip install colossus  # For cosmological calculations
Quick Start
bashgit clone https://github.com/yourusername/lyman-alpha-wst
cd lyman-alpha-wst
python fisher_matrix_with_wst.py


##  Repository Structure
├── fisher_matrix_with_wst.py    # Main WST analysis pipeline
├── fisher_matrix_with_ps.py     # Power spectrum comparison
├── plotting.py                  # Visualization and figure generation
├── notebooks/                        # Simulation data processing
├── results/                     # Output files and Fisher matrices



## Methodology

### Wavelet Scattering Transform

The WST provides a mathematically robust, interpretable framework for extracting higher-order statistical information:

- **Zeroth-order**: `S₀ = ⟨F(x) ⋆ φ[0]⟩` (mean flux)
- **First-order**: `S₁(j₁) = ⟨|F(x) ⋆ ψⱼ₁| ⋆ φ[1]⟩` (power-like)
- **Second-order**: `S₂(j₁,j₂) = ⟨||F(x) ⋆ ψⱼ₁| ⋆ ψⱼ₂| ⋆ φ[2]⟩` (non-Gaussian)

### Fisher Information Framework

We implement a comprehensive Fisher matrix analysis:
Fᵢⱼ = ∂S/∂pᵢ · Σ⁻¹ · ∂S/∂pⱼ

Where `S` represents WST coefficients and `Σ` is the covariance matrix accounting for realistic observational noise.

## Usage Examples

### Basic WST Analysis
```python
from kymatio.numpy import Scattering1D
import numpy as np

# Initialize scattering transform
J, Q = 5, 1  # 5 scales, 1 wavelet per octave
scattering = Scattering1D(J=J, shape=(378,), Q=Q, max_order=2)

# Process Lyman-α spectra
flux_data = load_lya_spectra()  # Your data loading function
wst_coefficients = scattering(flux_data)

# Calculate Fisher information
fisher_matrix = compute_fisher_matrix(wst_coefficients)
Parameter Constraint Forecasting
python# Compare WST vs Power Spectrum constraints
wst_constraints = forecast_constraints(fisher_wst)
ps_constraints = forecast_constraints(fisher_ps)

print(f"WST improvement: {ps_constraints/wst_constraints}")


##  Applications

### Current Surveys
- **eBOSS**: Validation with existing data
- **DESI**: Primary target for implementation
- **WEAVE-QSO**: Future survey applications

### Scientific Goals
- **Dark matter** properties via small-scale structure
- **Neutrino mass** constraints (minimal mass detection)
- **Inflationary physics** (running spectral index αₛ ~ -6×10⁻⁴)





### Computational Efficiency

- **Memory**: Scales linearly with number of spectra
- **Robustness**: Stable to deformations (Lipschitz continuous)

### Noise Resilience
The WST demonstrates superior performance across noise levels, maintaining constraint power even in systematics-dominated regimes anticipated for DESI.

## Future Development

### Planned Enhancements
-  Maximum likelihood estimator for WST coefficients
-  Full systematic error analysis
-  Emulator-based inference pipeline
-  3D extension for cross-forest correlations





## 📖 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{Tohfa2024,
    title={Forecast Cosmological Constraints with the 1D Wavelet Scattering Transform and the Lyman-alpha forest},
    author={Tohfa, Hurum Maksora and Bird, Simeon and Ho, Ming-Feng and Qezlou, Mahdi and Fernandez, Martin},
    journal={arXiv preprint arXiv:2310.06010},
    year={2024}
}


Contact

Hurum Tohfa: htohfa@uw.edu (Lead Author)
Simeon Bird: sbird@ucr.edu (Principal Investigator)
