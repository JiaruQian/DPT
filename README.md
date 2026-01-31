# DPT

Parallel thinking has emerged as a promising paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs) by exploring multiple reasoning paths simultaneously. However, we identify that parallel thinking via Reinforcement Learning (RL) typically suffers from the mode collapse of reasoning paths, where parallel thinking trajectories degenerate into similar ones. This homogeneity undermines the fundamental motivation of parallel thinking, that maintaining multiple hypotheses concurrently to prevent premature loss into a suboptimal solution. In this paper, we propose Diverse Parallel Thinking (DPT), a simple yet effective RL approach for parallel thinking designed to encourage path diversity. DPT rewards the model to develop distinct reasoning strategies by penalizing the prefix similarity of parallel thinking paths, thereby enhancing a more comprehensive exploration of the reasoning space.
We conduct various experiments and analyses across multiple reasoning benchmarks to demonstrate the effectiveness of DPT, and the results show that DPT successfully outperforms previous parallel thinking methods, achieving state-of-the-art performance over baselines. We validate path diversity as a key element to improve parallel thinking. 


## 🚀 Usage

### **1️⃣ Environment Setup**

We recommend using **Python 3.10+** and creating a fresh conda environment:

```bash
cd verl
conda create -n DPT python=3.10 -y
conda activate DPT
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .
```

### **2️⃣ Perform SFT**

```bash
cd verl
sh training_scripts/cold_start.sh
```

### **3️⃣ Perform RL**

To train DPT-Unseen from scratch 
```bash
cd verl
sh training_scripts/DPT_Unseen.sh
```


To train DPT-Seen from scratch 
```bash
cd verl
sh training_scripts/DPT_Seen.sh
```