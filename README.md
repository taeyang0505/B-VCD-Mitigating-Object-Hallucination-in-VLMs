# B-VCD: Enforcing Visual Evidentialism in VLMs via Blurred-Visual Contrastive Decoding

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Model](https://img.shields.io/badge/Model-LLaVA--1.5-green)](https://llava-vl.github.io/)
[![Evaluator](https://img.shields.io/badge/Evaluator-Gemini%202.5%20Flash-orange)](https://deepmind.google/technologies/gemini/)

This repository contains the official implementation of the term project: **"B-VCD: Mitigating Object Hallucination in VLMs for Visually Impaired Navigation using Visual Contrastive Decoding"**.

This project introduces **B-VCD (Blurred-Visual Contrastive Decoding)**, a training-free inference-stage defense mechanism designed to suppress "Object Hallucination" in Vision-Language Models (VLMs). By utilizing a sophisticated combination of **Motion Blur** and **Poisson-Gaussian Noise**, B-VCD forces the VLM to rely strictly on verifiable visual evidence, maximizing operational safety in high-risk domains such as Blind and Low-Vision (BLV) assistive technologies.

---

## 🎯 Experimental Design & Rationale

### 1. Baseline Model: Why LLaVA-1.5?
We selected **LLaVA-1.5** as our baseline model. While it boasts excellent visual context understanding and sophisticated text generation capabilities, its powerful LLM backbone paradoxically makes it highly vulnerable to the **"Language Prior"** phenomenon. When visual information is ambiguous, LLaVA-1.5 tends to rely heavily on statistical linguistic templates learned during training, confidently inventing non-existent objects. This makes it the perfect benchmark to rigorously test the effectiveness of our hallucination defense mechanism.

### 2. Dataset: Why VizWiz-VQA?
Instead of using clean, high-quality benchmark datasets (like MSCOCO), we utilized the **VizWiz-VQA** validation set. This dataset consists of images taken directly by visually impaired (BLV) individuals using smartphones. It inherently includes severe **real-world sensor distortions** such as low light, motion blur, and poor framing. These extreme conditions strongly trigger the VLM's reliance on language priors, providing an optimal and realistic environment to prove the practical social value of B-VCD.

### 3. Proposed Methodology: How B-VCD Works
The original Visual Contrastive Decoding (VCD) relies on simple Gaussian noise, which only disrupts high-frequency pixel details while leaving the macro-silhouette intact, failing to fully disrupt the VLM's linguistic guesswork. 

To overcome this, **B-VCD** implements a dual-layered degradation pipeline that mirrors actual physical CMOS sensor artifacts:
1. **Motion Blur Convolution:** Completely wipes out macro-structures (Low-frequency information).
2. **Poisson-Gaussian Noise Modeling:** Simulates photon shot noise and electronic read noise under low-light conditions.

By subtracting the "biased control logits" (generated from this severely degraded image) from the original logits, B-VCD mathematically penalizes linguistic guesses and enforces strict **Visual Evidentialism**—ensuring the model says "unclear" rather than generating a fatal hallucination.

---

## 🌟 Key Features
* **Training-Free Hallucination Mitigation:** Drastically reduces overconfident hallucinations in LLaVA-1.5 without any parameter fine-tuning.
* **Physical Sensor Degradation Modeling:** Implements realistic motion blur and Poisson-Gaussian noise.
* **Automated LLM-as-a-Judge Pipeline:** Utilizes `Gemini 2.5 Flash` to strictly evaluate visual factualness and conservativeness (Safe Refusal).
* **System-Engineered for Scale:** Features Asynchronous Parallel Processing (`ThreadPoolExecutor`), Logit Caching, and Fault-tolerant Regex Parsing to optimize compute time and API costs by over 60%.

---

## 📊 Main Results

B-VCD achieved a **Global Optimum** at **$M_{blur} = 30$** and **$\sigma_{read} = 2.5$** on the stratified VizWiz validation set (Hold-out 100 samples).

| Method | Blur Kernel | Noise Type | Avg Score (out of 5) | Win Rate vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| Baseline (Vanilla LLaVA) | None | None | 2.35 | - |
| Comparison (Original VCD) | None | Gaussian | 3.05 | 65.4% |
| **Ours (B-VCD Optima)** | **30** | **Poisson-Gaussian** | **3.33** | **69.5%** |

*Note: B-VCD successfully eliminated critical failure cases (0-point responses) and increased the frequency of safety avoidance keywords (e.g., "unclear", "cannot") by 1.5 times compared to the baseline, proving its reliability in high-risk scenarios.*

---

## 📂 Repository Structure

The experimental pipeline is strictly modularized into sequential Jupyter Notebooks and Python source modules:

```text
├── Data/
│   ├── annotations/                  # Dataset annotations (e.g., val.json)
│   └── VizWiz_val/
│       └── val/                      # VizWiz-VQA Validation Dataset (Image files)
├── notebooks/                        # Sequential Jupyter Notebooks for the experiment
│   ├── 01_EDA_VizWiz.ipynb           # Exploratory Data Analysis & Stratified Sampling logic
│   ├── 02_Baseline_Inference.ipynb   # Vanilla LLaVA-1.5 baseline response generation
│   ├── 03_BVCD_Experiment.ipynb      # Implementation of B-VCD degradation & Candidate generation
│   ├── 04_ReRanking_Pipeline.ipynb   # Gemini-based LLM-as-a-Judge API calling & scoring
│   ├── 05_Result_Analysis.ipynb      # Quantitative evaluation (Phase 1 results & visualizations)
│   ├── 06_Hyperparamter_Tuning.ipynb # 3x3 Grid Search loops over M_blur and Sigma_read
│   ├── 07_GridSearch_Analysis.ipynb  # Global Optimum extraction and Heatmap generation
│   └── 08_Qualitative_Analysis.ipynb # Extraction of Top 3 dramatic hallucination defense cases
├── Results/                          # JSON output logs, cached logits, and evaluation results
├── Src/                              # Custom Python modules for core operations
│   ├── dataset.py                    # Dataset loading, preprocessing, and Base64 encoding
│   ├── pipeline.py                   # Main pipeline orchestration
│   ├── utils.py                      # Helper utilities and visualization tools
│   └── vcd_decoding.py               # B-VCD degradation and decoding logic
├── B-VCD_Term_Project_Report.pdf     # Term Project Final Report and Analysis
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## 📚 References & Acknowledgements
- This project was conducted as a Term Project for the Machine Learning course at **Inha University** (Department of Artificial Intelligence).
- **VCD:** Leng, S., et al. "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding." *CVPR 2024*. [GitHub](https://github.com/DAMO-NLP-SG/VCD)
- **LLaVA:** Liu, H., et al. "Improved Baselines with Visual Instruction Tuning." [GitHub](https://llava-vl.github.io/)
- **VizWiz:** Gurari, D., et al. "VizWiz Grand Challenge: Answering Visual Questions from Blind People." *CVPR 2018*.
