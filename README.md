# VideoMolmo: Spatio-Temporal Grounding meets Pointing
![](https://i.imgur.com/waxVImv.png)

[Ghazi Shazan Ahmad](https://github.com/khufia)* , [Ahmed Heakl](https://github.com/ahmedheakl)*, [Hanan Gani](https://github.com/hananshafi), [Abdelrahman Shaker](https://amshaker.github.io/), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en). [Salman Khan](https://salman-h-khan.github.io/),

**Mohamed bin Zayed University of Artificial Intelligence, University of Washington, Allen Institute for Artificial Intelligence, 
Linköping University, Australian National University**

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://mbzuai-oryx.github.io/VideoGLaMM/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.04923)

---

## 📢 Latest Updates

- **May-2025:** Paper and inference is released!

---

## Overview

<p align="center">
  <img src="assets/videomolmo_teaser.png" width="70%" alt="VideoMolmo Architectural Overview">
</p>

VideoGLaMM is a large video multimodal video model capable of pixel-level visual grounding. The model responds to natural language queries from the user and intertwines spatio-temporal object masks in its generated textual responses to provide a detailed understanding of video content. VideoGLaMM seamlessly connects three key components: a Large Language Model (LLM); dual vision encoders; and a spatio-temporal pixel decoder. The dual vision encoders extract spatial and temporal features separately, which are jointly passed to the LLM to output responses rich in both spatial and temporal cues. This is facilitated by end-to-end training on our proposed benchmark Grounded conversation Generation (GCG) dataset featuring 38k Video-QA triplets with 87k objects and 671k fine-grained masks.

---
## 🏆 Highlights
1. We introduce Video Grounded Large Multi-modal Model (VideoGLaMM), a video large multimodal model, capable of pixel-level visual grounding, featuring an end-to-end alignment mechanism.

2. To achieve fine-grained spatio-temporal alignment, we introduce a benchmark grounded conversation generation (GCG) dataset consisting of 38k grounded video-QA triplet pairs and 83k objects and roughly 671k fine-grained spatio-temporal masks.

3. We assess the performance of VideoGLaMM across diverse tasks spanning grounded conversation generation, visual grounding, and referring video segmentation, where it achieves state-of-the-art performance

---
<!-- Architecture -->
## Architecture

<p align="center">
  <img src="assets/videomolmo_architecture.png" alt="VideoGLaMM Architecture">
</p>

VideoGLaMM consists of following key components: (i) Spatio-Temporal Dual Encoder, (ii) Dual Alignment V-L Adapters for image and video features, (iii) Large Language Model (LLM) iv) L-V Adapter and (iv) Promptable Pixel Decoder.

---
## Benchmark and Annotation Pipeline

<p align="center">
  <img src="assets/videomolmo_annotation_pipeline.png" alt="Annotation Pipeline">
</p>

We propose a semi-automatic annotation pipeline for creating a grounded conversation generation (GCG) dataset for videos.


---

## Running VideoMolmo 🔧

### Environment setup

    conda create --name=videomolmo python=3.11

    conda activate videomolmo

    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.41.0
    DS_BUILD_FUSED_ADAM=1 pip install deepspeed==0.14.0

    pip install -r VideoMolmo/requirements_sam2_cluster.txt 

    cd VideoMolmo/model/segment_anything_2
    python setup.py build_ext --inplace
    cd ../../..

### Training and Evaluation

Please refer [here](RUN_VideoGLaMM.md) for instructions


## Citation 📜

```bibtex
@article{munasinghe2024videoglamm,
  title={VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos}, 
  author={Shehan Munasinghe and Hanan Gani and Wenqi Zhu and Jiale Cao and Eric Xing and Fahad Khan and Salman Khan},
  journal={ArXiv},
  year={2024},
  url={https://arxiv.org/abs/2411.04923}
}
```

---

[<img src="assets/MBZUAI_logo.png" width="360" height="85">](https://mbzuai.ac.ae)
[<img src="assets/allenai_logo.png" width="360" height="85">](https://mbzuai.ac.ae)
[<img src="assets/UW_logo.png" width="360" height="85">](https://mbzuai.ac.ae)
