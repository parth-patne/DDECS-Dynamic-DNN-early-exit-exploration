# DDECS - Dynamic DNN Early Exit Exploration

Deep neural networks achieve state-of-the-art performance in many tasks by extracting increasingly high-level features at deeper layers. However, the added depth often comes with higher latency and energy usage during inference—prohibitive for real-time or energy-constrained applications. To address these challenges, our framework integrates **Reinforcement Learning (RL)** with **BranchyNet-inspired** side branches, allowing high-confidence samples to exit early while only complex cases propagate to deeper layers.

This repository contains code to reproduce results demonstrating how dynamically optimized early exits can deliver significant inference speedups and reduce power consumption, all while maintaining competitive accuracy compared to a static network baseline[cite: 1].

---

## Citation

If you use this codebase or build upon our RL-based early-exit approach, please cite:

    @article{YourCitationKey,
      title={DDECS - Dynamic DNN Early Exit Exploration},
      author={}
    }
*(Please update the citation details as needed)*.

## Requirements

- A machine with an NVIDIA GPU is recommended for training acceleration and required for power monitoring features.
- Python ≥3.7 (tested with Python 3.8/3.9)
- Dependencies listed in `requirements.txt`

### Python Dependencies

- PyTorch (>=1.7.0 recommended)
- Torchvision (>=0.8.0 recommended)
- NumPy (>=1.19.0)
- Pandas (>=1.0.0)
- Matplotlib (>=3.2.0)
- Scikit-learn (>=0.24.0)
- Seaborn (>=0.11.0)
- pynvml (>=8.0.4) (for GPU power monitoring)

Install everything with:

```bash
pip install -r requirements.txt
