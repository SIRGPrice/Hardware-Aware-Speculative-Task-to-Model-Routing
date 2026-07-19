# Hardware Efficient SWE: Hardware-Aware Speculative Task-to-Model Routing

## Summary

Agentic software engineering (SWE) workloads exhibit a bimodal structure: ~80% of turns are cheap exploration (file reads, grep, test runs) while ~20% require expensive synthesis (multi-file patches, reasoning). Existing LLM routers (SWE-Router, RouteLLM, FrugalGPT) optimize model selection alone, ignoring the hardware substrate.

This paper introduces **Hardware Efficient SWE**, the first routing framework that jointly optimizes three orthogonal axes per subtask: **model capability**, **hardware placement**, and **speculative decoding configuration** — all conditioned on the agent's partial trajectory.

### Key Insight

Consumer GPUs (RTX 5090: 32 GB, 1,792 GB/s) offer 3–4× higher decode throughput but limited VRAM; unified-memory systems (DGX Spark: 128 GB, 273 GB/s) provide massive capacity at lower bandwidth. Neither alone is Pareto-optimal. By deliberately pairing **RTX for high-throughput exploration** with **Spark for high-capacity synthesis**, and bridging them with **speculative decoding**, heterogeneous hardware becomes a design decision that maximizes token/sec/$, not a constraint to work around.

### Contributions

1. **Hardware Efficient SWE Framework** — Joint optimization of model selection, device placement, and SD config via constrained optimization over a heterogeneous device pool.

2. **Theorem 1 (Hardware-Aware Bayes Optimality)** — Proves that conditioning routing on both trajectory *and* hardware cost model strictly dominates model-only routing under any hardware asymmetry (cost/bandwidth ratios differ across devices). Gain scales with asymmetry ratio (~20× for RTX 5090 vs DGX Spark).

3. **Disaggregated Speculative Execution** — Runtime scheduler placing prefill (compute-bound) on Spark and decode (memory-bound) on RTX with speculative drafting, using Mooncake/DistServe for KV-cache transfer over 10GbE.

4. **Harness-Driven Deterministic Context Strategy** — Eliminates KV-cache transfer entirely by shifting context management to the harness layer: structured directives and investigation summaries live as versioned artifacts in the repository, enabling zero-sync, reproducible, auditable handoffs between small/large models (~1000× bandwidth reduction).

5. **Evaluation on SWE-Bench Verified** — Hardware Efficient SWE achieves **46.2% resolve rate at $0.28/task** (3.0× cheaper than Spark-only, 10.7× cheaper than Claude Sonnet 4) with **78s latency** (2.3× faster than Spark-only). Ablation: hardware awareness contributes 54% of cost savings, trajectory conditioning 36%, speculative decoding 18%.
