---
pipeline_tag: image-to-3d
tags:
- computer-vision
- 3d-reconstruction
- video-processing
---

# LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory

This repository contains the reimplemented model checkpoints for **LoGeR**, as presented in the paper [LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory](https://huggingface.co/papers/2603.03269). 

LoGeR processes long video streams in chunks with a hybrid memory design to improve large-scale geometric reconstruction quality and consistency.

[![Project Page](https://img.shields.io/badge/Project-Webpage-blue)](https://LoGeR-project.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-green)](https://github.com/Junyi42/LoGeR)
[![arXiv](https://img.shields.io/badge/arXiv-2603.03269-b31b1b.svg)](https://arxiv.org/abs/2603.03269)

<p align="center">
  <img src="https://loger-project.github.io/figs/fig1_teaser.png" alt="LoGeR Teaser" width="100%">
</p>

## Checkpoints

We provide two main pre-trained models. To use them, please clone the [corresponding GitHub repository](https://github.com/Junyi42/LoGeR) and place the downloaded `.pt` files in your local `ckpts/` directory as follows:

- `ckpts/LoGeR/latest.pt`
- `ckpts/LoGeR_star/latest.pt`

### Download Commands

You can download the weights directly via `wget`:

```bash
# Download LoGeR
wget -O ckpts/LoGeR/latest.pt https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR/latest.pt?download=true

# Download LoGeR_star
wget -O ckpts/LoGeR_star/latest.pt https://huggingface.co/Junyi42/LoGeR_star/latest.pt?download=true
```

## Usage
For detailed instructions on installation, running demos, and evaluation, please refer to the main [GitHub repository](https://github.com/Junyi42/LoGeR).

## Citation
If you find our work or these models useful, please cite our paper:

```bibtex
@article{zhang2026loger,
  title={LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory},
  author={Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Sun, Chen and Yang, Ming-Hsuan and Cole, Forrester and Darrell, Trevor and Sun, Deqing},
  journal={arXiv preprint arXiv:2603.03269},
  year={2026}
}
```