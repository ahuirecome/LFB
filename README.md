# README\.md

# LFB

This repository contains the official Python implementation of **LFB algorithm**, proposed in the paper **\[Learn From the Best: Population-Based Elite Behavior Cloning for Black-Box Optimization with Sparse Terminal Rewards\]**\. The project focuses on **black\-box optimization / curve fitting** tasks and provides reproducible experimental results\.

> ⚠ Note: This code is for academic non‑commercial research only. Commercial usage requires written authorization.

**✅ Reproducible \| Easy to Use**

---

## 📌 Introduction

Traditional gradient\-based or evolutionary optimization methods suffer from problems such as sparse reward failure, high dimensional explosion, slow convergence, and local optimum trapping\. This work proposes **\[LFB-ES\]** to address the above limitations, achieving better convergence speed, lower fitting error, and stronger high\-dimensional optimization capability\.

This codebase provides:

- Full implementation of the proposed algorithm

- Baseline algorithms \(OpenAI\-ES, CMA\-ES, etc\.\)

- Black\-box optimization \& curve\-fitting test environments

- Reproducible training and evaluation pipelines

- Result visualization tools

---


## ⚙️ Installation

Clone this repository:

```bash

git clone https://github.com/ahuirecome/LFB.git
cd LFB
    
```

Install dependencies:

```bash

pip install -r requirements.txt
   
```

Main dependencies:

- python \>= 3\.10

- numpy

- torch

- matplotlib

- cma

---

## 🚀 Quick Start

Run the LFB algorithm for curve fitting with default parameters:

```bash

python lfb_fit.py --curve_id=0
    
```

To fit a set of curves to obtain statistical results:

```bash

python lfb_fit_batch.py 
    
```

Users may add their own curves for testing in the file below.:

```bash

curves.py
    
```
---

## 📊 Experimental Results

The proposed method outperforms baseline methods in terms of:

- Lower fitting error in high\-frequency curve regression

- Faster convergence in high\-dimensional optimization

---

## 📄 Citation

If you find this project useful, please cite our work:

```plain

@article{lfb2026,
  title={Learn From the Best: Population-Based Elite Behavior Cloning for Black-Box Optimization with Sparse Terminal Rewards},
  author={Zhijian Mo, Qihua Xiao, Zhihui Shan, Xueli Ban},
  journal={Under review},
  year={2026}
}
    
```

---

## 🪪 License

This repository is for **academic research purposes only**.
Any form of commercial use is strictly prohibited without explicit written permission from the author.

See the [LICENSE](LICENSE) file for complete terms.

> For commercial licensing requests, please contact the author via email.

---

## 🙋 Contact

If you have any questions or suggestions, feel free to open an issue or contact me via email.
