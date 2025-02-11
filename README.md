# DDECS - Dynamic DNN Early Exit Exploration

Deep neural networks achieve state-of-the-art performance in many tasks by extracting increasingly high-level features at deeper layers. However, the added depth often comes with higher latency and energy usage during inference—prohibitive for real-time or energy-constrained applications. To address these challenges, our framework integrates **Reinforcement Learning (RL)** with **BranchyNet-inspired** side branches, allowing high-confidence samples to exit early while only complex cases propagate to deeper layers.

This repository contains code to reproduce results from our paper *“RL-Agent-based Early-Exit DNN Architecture Search Framework,”* which demonstrates how dynamically optimized early exits can deliver significant inference speedups (up to 69.7×) and reduce power consumption, all while maintaining accuracy within 1–2% of a static network baseline.

---

## Citation

If you use this codebase or build upon our RL-based early-exit approach, please cite:

@inproceedings{YourRLAgentEarlyExitPaper, title = {RL-Agent-based Early-Exit DNN Architecture Search Framework}, author = {Your Name and Coauthors}, booktitle = {Some Conference or Journal}, year = {2023}, ... }

markdown
Copy
Edit

---

## Requirements

- A machine with a decent GPU (optional but recommended)
- Python ≥3.7 (tested with Python 3.8/3.9)
- Dependencies listed in `requirements.txt` (e.g., PyTorch, pynvml, etc.)

### Python Dependencies

- [PyTorch](https://pytorch.org/) (≥1.7)
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `seaborn`, etc.
- `pynvml` for GPU power monitoring

Install everything with:

```bash
pip install -r requirements.txt
```

## Quickstart
1. Clone and Set Up
```bash
git clone https://github.com/YourUsername/rl-agent-early-exit-dnn.git
cd rl-agent-early-exit-dnn
pip install -r requirements.txt
```
2. Run Training & Evaluation
```bash
python src/dynamic-alexnet-2-early-exits.py --dataset cifar10 --epochs 30 --learning_rate 0.001
```
This script trains a 2-exit AlexNet model with RL-based thresholding on the CIFAR-10 dataset.
To evaluate or measure GPU power usage, add flags such as --evaluate or --power_measurement.


## Run with Docker
Install Docker and ensure it’s running.
Build the Docker image:
```bash
./docker/build.sh
```
Start a shell inside the container:
```bash
./docker/shell.sh
```
Once inside the container, run:
```bash
python src/dynamic-alexnet-2-early-exits.py --dataset cifar10 --epochs 30
```
For GPU support, you may need to customize the Dockerfile with CUDA libraries, and run it via nvidia-docker.
