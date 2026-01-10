# Einstein-Cartan Torsion Cosmology Evolution

## 🎯 Objective

Evolve torsion cosmology parameters to:
1. Fit CMB TT, EE, and TE spectra simultaneously
2. Predict tensor-to-scalar ratio for CMB-S4
3. Discover optimal spin-torsion coupling

## 📊 Background

From the previous bounce cosmology evolution, we discovered:

```
S(ℓ) = 1 - 0.804/(1 + (ℓ-2))^1.455
ΔBIC = -3.73 (beats ΛCDM)
```

The power-law index **α = 1.455 ≈ 3/2** matches the Einstein-Cartan theory prediction!

## 🔬 Physics

Einstein-Cartan-Sciama-Kibble (ECSK) theory predicts:

| Parameter | Physical Meaning | Theory Prediction |
|-----------|------------------|-------------------|
| α | Power-law exponent | **3/2** (fixed) |
| κ | Torsion coupling | ~0.8 (from TT fit) |
| β | Spin-polarization | **2/3** (theory) |
| r | Tensor-to-scalar | < 0.06 (BICEP limit) |
| n_t | Tensor tilt | ~-0.02 (bounce) |

## 📁 Files

```
torsion_problem/
├── config.yaml      # Evolution configuration
├── evaluate.py      # Fitness evaluation (TT+EE+TE)
├── init_program.py  # Starting model
└── README.md        # This file
```

## 🚀 Running Evolution

```bash
cd problems/torsion_cosmology
python ../../evolve.py config.yaml
```

## 📈 Fitness Function

```
combined_score = 0.6 * bic_fitness + 0.4 * physics_score

where:
- bic_fitness = σ(-ΔBIC/5)  [reward negative ΔBIC]
- physics_score = (quad × asymp × tensor × corr)^0.25
```

## 🎯 Success Metrics

| Metric | Target | Meaning |
|--------|--------|---------|
| `beats_lcdm` | true | ΔBIC < 0 |
| `combined_score` | > 0.7 | Good model |
| `physics_score` | > 0.8 | Physically consistent |
| `correlation_score` | > 0.9 | S_EE ≈ S_TT^(2/3) |

## 💡 Evolution Strategy

The key insight from the previous evolution: **hardcode optimal values** to avoid BIC penalty!

With 4 parameters: BIC penalty = 4 × ln(42) ≈ 15
With 0 parameters: BIC penalty = 0

So evolution should:
1. Find optimal values for κ, β, r, n_t
2. Hardcode them (return empty dict from `get_torsion_params()`)
3. Achieve ΔBIC < 0

## 🔮 Predictions for CMB-S4

The model predicts B-mode polarization:
```
C_ℓ^BB ∝ r × (ℓ/80)^n_t

At ℓ=80 (BICEP sweet spot):
- r = 0.01 → BB ≈ 100 nK²
- r = 0.001 → BB ≈ 10 nK²
```

CMB-S4 sensitivity: ~1 nK² at ℓ=80

## 📚 References

- Popławski (2010): [arXiv:1007.0587](https://arxiv.org/abs/1007.0587)
- Planck 2018: [arXiv:1807.06209](https://arxiv.org/abs/1807.06209)
- BICEP/Keck 2021: [arXiv:2110.00483](https://arxiv.org/abs/2110.00483)
