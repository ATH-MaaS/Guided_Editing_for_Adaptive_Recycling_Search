
<div align="center">

# Test-Time Scaling for Video Diffusion Models via Diagnosis-Guided Candidate Recycling


<p align="center">  
    <a href="https://hangzhouhe.com/">Hangzhou He</a>,
    <a href="https://scholar.google.com/citations?user=VZbOk2MAAAAJ&hl=en">Lunhao Duan</a>,
    <a href="https://scholar.google.com.hk/citations?user=TFp72vEAAAAJ&hl=zh-CN">Shanshan Zhao♯</a>,
    Kaiwen Li,
    <a href="https://scholar.google.com/citations?user=GlqRHLcAAAAJ">Qing-Guo Chen</a>,
    <a href="https://scholar.google.com/citations?user=tsKl9GUAAAAJ">Weihua Luo</a>,
    <a href="https://scholar.google.com/citations?user=WSFToOMAAAAJ">Yanye Lu♯</a>
</p>

<p>Peking University, Alibaba Group</p>
<p><strong>SIGGRAPH ASIA 2026 & TOG | </strong> <a href="https://arxiv.org/abs/2608.29322"><strong>Paper</strong></a></p>

</div>

## Introduction
This is the implementation of GEARS(**G**uided **E**diting for **A**daptive **R**ecycling **S**earch) from paper "Test-Time Scaling for Video Diffusion Models via Diagnosis-Guided Candidate Recycling". GEARS is a test-time scaling framework that improves text-to-video generation by turning low-scoring yet structurally plausible candidates into editable priors instead of discarding them after evaluation. GEARS consists of two components:
1.  **Stage-Aware Scheduler** determines what to repair, when to repair it, and which candidates should be preserved, recycled, or discarded.
2. **Candidate Recycler** diagnoses recoverable failures from keyframes and multi-dimensional reward feedback, derives candidatespecific repair prompts, and repairs the corresponding candidates through manifold-aware latent SDEdit.

![vis](vis.png)


## Quick Start

The code was tested with Python 3.10.20, PyTorch 2.7.0 with
CUDA 12.8, Transformers 5.3.0, Diffusers 0.37.0 and Flash-Attn 2.7.3. We have also modified the
VideoAlign codebase to to match with the transformers version.

Download the following resources:

1. A [Wan2.1](https://github.com/Wan-Video/Wan2.1) checkpoint, such as `Wan2.1-T2V-1.3B` or
   `Wan2.1-T2V-14B`.
2. The [VideoReward](https://github.com/KwaiVGI/VideoAlign) checkpoint.
3. The standard [Vbench](https://github.com/Vchitect/VBench) prompts arranged as one text file per category. Each
   non-empty line must contain one prompt.

Enter API key, endpoint, and generation configs in
`run_wan_gears_vbench.sh` then run the script.

## Acknowledgements

We thank the authors of [Wan2.1](https://github.com/Wan-Video/Wan2.1), [VideoReward](https://github.com/KwaiVGI/VideoAlign), [EvoSearch](https://github.com/tinnerhrhe/EvoSearch-codes), and [Vbench](https://github.com/Vchitect/VBench) for their open-source contributions.

## Citation

If you find GEARS useful for your research, please consider citing:

```bibtex
@article{he2026gears,
  title     = {Test-Time Scaling for Video Diffusion Models via Diagnosis-Guided Candidate Recycling},
  author    = {He, Hangzhou and Duan, Lunhao and Zhao, Shanshan and Li, Kaiwen and Chen, Qing-Guo and Luo, Weihua and Lu, Yanye},
  journal   = {ACM Transactions on Graphics},
  year      = {2025},
  note      = {SIGGRAPH Asia 2026},
}
```
