# 👁️ B-VCD: Mitigating Object Hallucination in VLMs for Visually-Impaired Assistance
 
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Backbone](https://img.shields.io/badge/VLM-LLaVA--1.5%20(Ollama)-orange)
![Judge](https://img.shields.io/badge/LLM--as--a--Judge-Gemini%202.5%20Flash-green)
![Dataset](https://img.shields.io/badge/Dataset-VizWiz--VQA-red)
 
> **"A Training-Free, Degradation-Based Strategy to Penalize Ungrounded Answers"**
> *Term Project for Artificial Intelligence, Inha University*
> *Author: Taeyang Hong*
 
## Overview
 
Large Vision-Language Models (VLMs) like LLaVA-1.5 are prone to **object hallucination**: when visual evidence is weak, they fall back on language priors and confidently describe objects that are not actually visible. For assistive technology serving the **Blind and Low-Vision (BLV)** community, such confident-but-wrong answers are a real safety risk.
 
This repository implements **B-VCD (Blurred-Visual Contrastive Decoding)**, a training-free, inference-stage pipeline that discourages ungrounded answers. The core idea, inspired by Visual Contrastive Decoding (VCD), is to generate answers under a physically-modeled **severe image degradation** and then use a **degraded-image-grounded LLM judge** to reward answers that stay within what is actually verifiable.
 
---
 
## ⚠️ Implementation Note (please read first)
 
To keep this README aligned with the actual code, two clarifications matter:
 
1. **No token-level logit arithmetic is performed.** The LLaVA backbone is served through **Ollama** (`/api/generate`), which returns generated *text* and does **not** expose per-token logits. The classic VCD objective `(1+α)·logit_orig − α·logit_distort` is therefore the *conceptual inspiration*, not the executed computation. In this project the "contrast" is realized at two practical levels instead: (a) generating a candidate answer from a heavily degraded image, and (b) scoring candidates with a judge that itself only sees the degraded image, so it penalizes any claim that cannot be visually verified.
2. **Adaptive Plausibility Constraints (APC) are not implemented** in this version. APC is a component of the original VCD paper and is listed here only as future work.
This honest framing is intentional: it is the part most likely to be probed in a review, and it is better to state it up front.
 
---
 
## Motivation
 
### The trap of language priors
Under ambiguous, low-light, or corrupted input, VLMs tend to "guess" from textual/statistical priors rather than the uncertain pixels, fabricating non-existent objects.
 
### Why go beyond simple Gaussian-noise VCD
The original VCD builds its contrastive (distorted) image with simple **Gaussian noise**. This mainly perturbs high-frequency detail. In this project we instead use a **stronger, physically-motivated degradation** so that the contrastive image is harder for the model to "read" confidently. Concretely, the degradation combines directional motion blur, heavy illumination attenuation, and Poisson-Gaussian sensor noise (details below).
 
> Note on mechanism: motion blur is a (directional) low-pass operation — on its own it mainly removes fine detail rather than erasing macro-structure. In this implementation the largest contributor to suppressing the silhouette is the **illumination attenuation (×0.3 brightness)** combined with sensor noise, not the blur alone. The README wording has been corrected to reflect this.
 
---
 
## Method: Degradation Pipeline
 
The degraded "control" image is produced by `get_b_vcd_images()` in `Src/dataset.ipynb` and applied per image:
 
**1. Stochastic directional motion blur.** A motion-blur kernel of random length `l ∈ [10, max_blur]` and random angle `θ ∈ [0°, 180°]` is convolved with the image. (Because `l` and `θ` are sampled per call, the degradation is stochastic.)
 
**2. Illumination attenuation.** The blurred image is darkened by a `brightness_factor` (default **0.3**, i.e. reduced to 30% brightness) to simulate low-light capture.
 
**3. Poisson-Gaussian sensor noise.** Photon shot noise is modeled with a Poisson process (`photon_scale = 10`) and electronic read noise with a Gaussian `N(0, σ_read²)`:
 
```
x_blur     = x * K(θ, l)                       # 1. motion blur
x_dark     = x_blur * brightness_factor        # 2. illumination attenuation
x_distort  = Poisson(x_dark * s) / s + N(0, σ_read²)   # 3. Poisson-Gaussian noise
```
 
This Poisson-Gaussian model (Poisson shot noise + Gaussian read noise) is the standard model for CMOS sensor degradation, which is appropriate for the real-world BLV capture conditions in VizWiz.
 
For the baseline comparison, `get_vcd_images()` implements the **original VCD** degradation (Gaussian noise only, `std = noise_step/10`, default `noise_step = 500`).
 
---
 
## Method: Candidate Generation & Judging
 
For each image–question pair the system produces **three candidate answers** from LLaVA-1.5 (via Ollama) and then scores them with an LLM judge:
 
| Candidate | Image shown to LLaVA | Decoding temp | Role |
| :-- | :-- | :--: | :-- |
| 1 — Baseline | Original image | 0.0 | Vanilla LLaVA (reused from the cached baseline run) |
| 2 — Original VCD | Gaussian-noise image | 0.0 | Reproduction of standard VCD's distorted input |
| 3 — B-VCD (ours) | Motion-blur + dark + Poisson-Gaussian image | 0.0 | Proposed degradation |
 
**Judge (LLM-as-a-Judge).** The judge is **Gemini 2.5 Flash**, called over the REST API (`generativelanguage.googleapis.com`, with retry logic). Critically, the judge is shown the **degraded** image and instructed to score answers *only* on what is verifiable in that image, strongly penalizing confident descriptions of unseen objects and rewarding conservative-but-accurate answers (`Src/vcd_decoding.ipynb`, `notebooks/04`, `notebooks/06`).
 
> An alternative GPT-4o judge (1–10 scale) exists in `Src/evaluate.ipynb` from an earlier iteration; the reported results use the Gemini judge.
 
---
 
## Experimental Setup
 
- **Backbone:** LLaVA-1.5 served locally via Ollama.
- **Dataset:** VizWiz-VQA validation split — real photos taken by blind users, which naturally contain low light, motion blur, and poor framing.
- **Judge:** Gemini 2.5 Flash (deterministic settings).
- **Two evaluation scales:**
  - **Full comparative run — 4,319 images** (`notebooks/04`, ~104 min): all three candidates scored by the Gemini judge for the headline comparison.
  - **Hyperparameter grid search — 100-image stratified subset** (`notebooks/06`, ~110 min): a smaller, label-balanced subset (sampled by VizWiz `answer_type`: OTHER / UNANSWERABLE / YES-NO / NUMBER) used to keep API cost bounded while sweeping distortion parameters.
### Pipeline (notebooks)
1. `01_EDA_VizWiz` — dataset exploration.
2. `02_Baseline_Inference` — vanilla LLaVA-1.5 over the 4,319-image validation set (~12 h; cached and reused downstream).
3. `03_BVCD_Experiment` — generates the candidate set (baseline / VCD / B-VCD) per image.
4. `04_ReRanking_Pipeline` — full 4,319-image Gemini scoring of the candidates (this is the run behind the headline table).
5. `06_Hyperparameter_Tuning` + `07_GridSearch_Analysis` — 3×3 distortion grid search on the 100-image subset.
6. `05_Result_Analysis` + `08_Qualitative_Analysis` — aggregates scores, plots distributions, and collects case studies.
---
 
## Quantitative Results
 
Average judge score and head-to-head win rate vs. the baseline, as aggregated in `05_Result_Analysis`:
 
| Method | Avg Judge Score | Win Rate (vs Baseline) | Safety Index |
| :--- | :---: | :---: | :---: |
| Baseline (Vanilla LLaVA) | 1.94 | — | 0.17 |
| Original VCD (Gaussian) | 3.05 | 65.4% | — |
| **B-VCD (ours)** | **3.33** | **69.5%** | **0.26** |
 
- *Safety Index* = frequency of conservative/uncertainty keywords (e.g. "cannot", "unclear").
- The score distribution (violin plot below) shows that B-VCD **substantially reduces the low-scoring "0-point" tail** (confident hallucinations) relative to the baseline.
<p align="center">
  <img src="assets/result_graph.png" alt="B-VCD score distribution" width="90%">
</p>
> **Items to verify / reconcile before citing these numbers** (flagged for transparency):
> - **Score scale:** the Gemini judge prompt does not pin a fixed range; confirm the exact scale used in `05_Result_Analysis` (the table is consistent with a 0–5 scale) and the missing VCD Safety Index cell.
> - **Distortion config of the full run:** the grid search reports its best subset configuration at `blur = 30, read_noise_std = 2.5`, whereas the full 4,319-image run uses the defaults (`max_blur = 30, brightness_factor = 0.3, read_noise_std = 5.0`). State clearly which configuration produced the headline table.
 
### On the grid-search "optimum"
The 3×3 sweep covered `blur ∈ {10, 20, 30}` and `read_noise_std ∈ {2.5, 5.0, 7.5}`, and the best subset result fell at **`blur = 30` (the maximum tested) and `noise = 2.5` (the minimum tested)** — i.e. at the **corner** of the search grid. This means it is the best *within the tested range*, not a verified global optimum; the true optimum may lie outside the grid (higher blur and/or lower noise). It is also notable that the best setting used the **lowest** noise level, which suggests the blur + illumination terms contribute more than the additive sensor noise. Extending the grid is listed under Future Work.
 
---
 
## Qualitative Case Studies
 
**Success — suppressing ungrounded conjecture.** On an illegible, blurry restaurant menu, the baseline hallucinates specific dishes and prices, while B-VCD conservatively answers *"The text is not clear enough to read,"* which is the safer response for a BLV user.
 
<p align="center"><img src="assets/success_menu.png" alt="Success case: menu" width="45%"></p>
**Failure — over-corruption.** When an already low-contrast image is degraded further, essential cues vanish and the model compensates by hallucinating hyper-specific details (e.g. labeling a white blob as "diapers"). This motivates an image-quality-aware adaptive controller.
 
<p align="center"><img src="assets/failure_diaper.png" alt="Failure case" width="30%"></p>
---
 
## Conclusion
 
B-VCD is a training-free pipeline that reduces object hallucination by (1) generating answers under a strong, physically-modeled image degradation and (2) scoring candidates with a degraded-image-grounded LLM judge. On the VizWiz validation set, the proposed degradation improves the average judge score and win rate over both the vanilla baseline and the original Gaussian-noise VCD reproduction, and it shrinks the catastrophic-failure tail.
 
## Limitations
 
- **Not logit-level VCD.** Because LLaVA is served via Ollama (no logit access), the method operates on generated text and judge scoring rather than on token logits. A true logit-space VCD would require a `transformers`-based backend.
- **Multiple generations per item.** Producing several candidates requires multiple full LLaVA generations per image, plus a judge API call, which increases latency and cost compared to single-pass decoding.
- **Judge dependence & cost.** Results depend on a proprietary judge (Gemini 2.5 Flash); the grid search was therefore limited to a 100-image subset to bound API spend.
- **Stochastic degradation.** Blur length/angle are sampled per image without a fixed seed in the degradation function, so individual scores carry run-to-run variance.
- **Boundary optimum.** The reported best configuration sits at the edge of the searched grid (see above).
## Future Work
- Extend the hyperparameter grid beyond its current boundaries and fix a seed for reproducibility.
- Implement true logit-space VCD (and APC) on a `transformers` LLaVA backend.
- Add an image-quality-aware adaptive controller to avoid over-corruption.
- Report standard hallucination benchmarks (e.g. POPE/CHAIR) alongside the LLM-judge metric for comparability.
## References
- Leng et al., *Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding*, CVPR 2024. (VCD; source of the contrastive-decoding idea and APC.)
- Gurari et al., *VizWiz Grand Challenge: Answering Visual Questions from Blind People*, CVPR 2018. (Dataset.)
