# HeteroSWE: Hardware-Aware Speculative Task-to-Model Router

This directory contains the LaTeX source for the paper:

**"HeteroSWE: Hardware-Aware Speculative Task-to-Model Routing for Cost-Efficient Agentic Software Engineering on Heterogeneous Consumer Clusters"**

## Files

- `hetero_swe_router.tex` — Main paper (IEEE conference format, ~10 pages)
- `references.bib` — Bibliography with 40+ citations

## Building the PDF

### Option 1: Overleaf (Recommended)
1. Upload both files to a new Overleaf project
2. Compile with `pdfLaTeX` → `bibtex` → `pdfLaTeX` × 2

### Option 2: Local LaTeX
```bash
# Requires: texlive-full (or miktex on Windows)
pdflatex hetero_swe_router.tex
bibtex hetero_swe_router
pdflatex hetero_swe_router.tex
pdflatex hetero_swe_router.tex
```

### Option 3: Docker
```bash
docker run --rm -v $(pwd):/workspace -w /workspace texlive/texlive:latest \
  sh -c "pdflatex hetero_swe_router.tex && bibtex hetero_swe_router && pdflatex hetero_swe_router.tex && pdflatex hetero_swe_router.tex"
```

## Paper Structure

| Section | Pages | Key Content |
|---------|-------|-------------|
| Abstract | 0.3 | Problem, insight, results summary |
| 1. Introduction | 1.0 | Motivation, hardware asymmetry, contributions |
| 2. Background | 1.5 | Agentic SWE workload, RTX vs Spark specs, SD landscape |
| 3. HeteroSWE Framework | 2.5 | System model, routing optimization, value function, cost model, disaggregated SD, algorithm |
| 4. Theoretical Analysis | 1.0 | **Theorem 1**: Hardware-Aware Bayes Optimality; SD regret bound |
| 5. Implementation | 1.0 | Software stack, model zoo, trajectory data collection |
| 6. Evaluation | 2.0 | Setup, baselines, metrics, main results, latency breakdown, ablations, sensitivity |
| 7. Discussion | 0.5 | When it wins, limitations, future work |
| 8. Related Work | 0.5 | Routing, heterogeneous serving, SD, agentic SWE |
| 9. Conclusion | 0.2 | Summary |

## Key Claims (Simulated/Expected Data)

| Metric | HeteroSWE | Spark-Only | RTX-Only | Sonnet 4 |
|--------|-----------|------------|----------|----------|
| Resolve Rate | **46.2%** | 42.1% | 35.4% | 48.2% |
| Cost/Task | **$0.28** | $0.85 | $0.32 | $3.00 |
| Latency | **78s** | 182s | 95s | 45s |
| Route-AUC | **0.821** | 0.627 | 0.581 | — |

*Ablation attribution: Hardware awareness 54%, Trajectory 36%, Speculative Decoding 18%.*

## Theoretical Contribution

**Theorem 1 (Hardware-Aware Bayes Optimality):** Routing conditioned on both partial trajectory **and** hardware cost model strictly dominates model-only routing when device cost/bandwidth ratios differ (proven via decision space expansion).

## Citation

If you use this work, please cite:
```bibtex
@article{heteroswe2026,
  title={HeteroSWE: Hardware-Aware Speculative Task-to-Model Routing for Cost-Efficient Agentic Software Engineering on Heterogeneous Consumer Clusters},
  author={Anonymous Authors},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License — see LICENSE file.