# DDECS---Dynamic-DNN-early-exit-exploration

Deep neural networks achieve state-of-the-art performance in many tasks by extracting increasingly high-level features at deeper layers. However, the added depth often comes with higher latency and energy usage during inference—prohibitive for real-time or energy-constrained applications. To address these challenges, our framework integrates **Reinforcement Learning (RL)** with **BranchyNet-inspired** side branches, allowing high-confidence samples to exit early while only complex cases propagate to deeper layers.

This repository contains code to reproduce results from our paper *“RL-Agent-based Early-Exit DNN Architecture Search Framework,”* which demonstrates how dynamically optimized early exits can deliver significant inference speedups (up to 69.7×) and reduce power consumption, all while maintaining accuracy within 1–2% of a static network baseline.

---

## Citation

If you use this codebase or build upon our RL-based early-exit approach, please cite:

