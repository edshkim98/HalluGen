# HalluGen: Synthesizing Realistic and Controllable Hallucinations for Evaluating Image Restoration

**An official repository for HalluGen (Accepted to CVPR'26)**

> **Status:** Code under active development

---

## Overview

![Framework](imgs/total_framework.png)
*Schematic diagram of the overall framework of HalluGen — a controllable and systematic diffusion-based hallucination generator.*

---

## Results

### Generated Hallucinations
![Main Results](imgs/main_results.png)
*HalluGen-generated samples for both intrinsic and extrinsic hallucinations on brain MRI and inspection images.*

### SHAFE Evaluation
![Main Table](imgs/main_table.png)
*SHAFE outperforms all other image quality metrics on the HalluGen-generated hallucination dataset.*

### Hallucination Localization
![SHAFE Heatmap](imgs/shafe_heatmap.png)
*SHAFE can clearly localize hallucinations from real restoration outputs.*

---

## Usage

### Generating a Hallucination Dataset with HalluGen

To run HalluGen and create your own hallucination dataset:

```bash
python image_sample_hallugen.py
```

You will first need a pre-trained diffusion model. If you do not have one, you can train one from scratch:

```bash
python image_train.py
```

### Using the SHAFE Hallucination Metric

SHAFE can be used as a drop-in API. Import and instantiate the class directly from `SHAFE.py`:

```python
from SHAFE import SHAFE

metric = SHAFE(model_name='resnetaa50d.d_in12k', device='cuda')
score = metric(pred, gt)  # pred, gt: torch.Tensor of shape (B, 1, H, W), values in [0, 1]
```
