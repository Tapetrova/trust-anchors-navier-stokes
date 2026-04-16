# Trust Anchors for Agentic Scientific Computing

**Analytical Navier–Stokes Solutions as Verification Primitives in Multi-Agent AI Ecosystems**

Tatiana Petrova — SEDAN/SnT, University of Luxembourg

## Abstract

We propose that exact analytical solutions of the Navier–Stokes equations serve as *trust anchors* for agentic scientific computing — computationally verifiable ground truths that enable AI agents to self-validate, diagnose failure modes, and establish mutual trust without human intervention. We experimentally validate a diagnostic protocol by training three architecturally distinct PINN solvers on four trust anchors and show that error patterns match predicted failure mode signatures.

## Repository Structure

```
paper/
  trust-anchors-article.tex   — LaTeX source
  trust-anchors-article.pdf   — Compiled paper
  IEEEtran.cls                — Document class

code/
  trust_anchors.py             — 4 analytical N-S solutions (Lamb-Oseen, Stokes plate, Burgers vortex, Kovasznay)
  run_experiments.py           — 72 PINN training runs via DeepXDE
  generate_figures.py          — Figure and LaTeX generation

results/
  exp1_table2.json             — Table 2: 3 solvers × 4 anchors L² errors
  exp2_spectral.json           — Spectral bias diagnosis (Figure 2)
  exp3_temporal.json           — Temporal drift diagnosis (Figure 3)
  exp4_routing.json            — Trust-informed routing (Figure 4)
  exp5_trust_decay.json        — Runtime trust decay

figures/
  figure2_spectral_bias.pdf    — ε(ω) spectral bias plot
  figure3_temporal_drift.pdf   — ε(t) temporal drift plot
  figure4_routing.pdf          — Routing comparison bar chart
```

## Reproducing Experiments

```bash
pip install deepxde torch numpy matplotlib scipy
cd code
DDE_BACKEND=pytorch python run_experiments.py
python generate_figures.py
```

## Key Results

- **Spectral bias diagnosis**: Fourier-Feature PINNs reduce error by 81% at 8ω₁ vs Vanilla
- **Temporal drift diagnosis**: Causal PINNs reduce error by 82% at 50t₀ vs Vanilla  
- **Trust-informed routing**: 70% lower mean error than the best fixed solver

## Citation

```bibtex
@article{petrova2025trustanchors,
  title={Trust Anchors for Agentic Scientific Computing: Analytical Navier--Stokes Solutions as Verification Primitives in Multi-Agent AI Ecosystems},
  author={Petrova, Tatiana},
  year={2025}
}
```

## License

MIT
