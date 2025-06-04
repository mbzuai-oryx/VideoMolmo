# VideoMolmo: Spatio-Temporal Grounding meets Pointing
![](https://i.imgur.com/waxVImv.png)

[Ghazi Shazan Ahmad](https://github.com/khufia)* , [Ahmed Heakl](https://scholar.google.com/citations?user=JcWO9OUAAAAJ&hl=en)*, [Hanan Gani](https://github.com/hananshafi), [Abdelrahman Shaker](https://amshaker.github.io/), [Ranjay Krishna](https://ranjaykrishna.com/index.html), [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en). [Salman Khan](https://salman-h-khan.github.io/),

**MBZUAI, University of Washington, Allen Institute for Artificial Intelligence, 
Linköping University, ANU**

\* Equal Technical Contributions

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://mbzuai-oryx.github.io/VideoMolmo/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2411.04923)

---

## 📢 Latest Updates

- **May-2025:** Paper and inference is released!

---

## Overview

<p align="center">
  <img src="assets/videomolmo_teaser.png" width="70%" alt="VideoMolmo Architectural Overview">
</p>

**VideoMolmo** is a a large multimodal model tailored for fine-grained spatio-temporal pointing conditioned on textual descriptions. Building upon the Molmo architecture, VideoMolmo incorporates a temporal module utilizing an attention mechanism to condition each frame on preceding frames, ensuring temporal consistency. Additionally, our novel temporal mask fusion pipeline employs SAM2 for bidirectional point propagation, significantly enhancing coherence across video sequences. This two-step decomposition i.e., first using the LLM to generate precise pointing coordinates, then relying on a sequential mask-fusion module to produce coherent segmentation, not only simplifies the task for the language model but also enhances interpretability. Due to the lack of suitable datasets, we curate a comprehensive dataset comprising 72k video-caption pairs annotated with 100k object points. To evaluate the generalization of VideoMolmo, we introduce VPoS-Bench, a challenging out-of-distribution benchmark spanning five real-world scenarios: Cell Tracking, Egocentric Vision, Autonomous Driving, Video-GUI Interaction, and Robotics. We also evaluate our model on Referring Video Object Segmentation (Refer-VOS) and Reasoning VOS tasks. In comparison to existing models, \method substantially improves spatio-temporal pointing accuracy and reasoning capability.

---
## 🏆 Highlights
Key contributions of **VideoMolmo**:
1. We introduce VideoMolmo , an LMM that accepts natural-language queries and produces point-level predictions for target objects across entire video sequences, ensuring temporal consistency.

2. We further introduce Temporal module to leverage past temporal context and propose a novel temporal mask fusion pipeline for enhanced temporal coherence.

3. To achieve fine-grained spatio-temporal pointing, we introduce a comprehensive dataset of 72k video-caption pairs and 100k object points.

4. To evaluate the generalization of VideoMolmo, we introduce VPoS-Bench, a challenging out-of-distribution benchmark spanning five real-world scenarios: Cell Tracking, Egocentric Vision, Autonomous Driving, Video-GUI Interaction, and Robotics. We also assess our model on Referring Video Object Segmentation (Ref-VOS) and Reasoning VOS tasks.

---
<!-- Architecture -->
## Architecture

<p align="center">
  <img src="assets/videomolmo_architecture.png" alt="VideoGLaMM Architecture">
</p>

**VideoMolmo** consists of four end-to-end trainable components: (1) a visual encoder, (2) a temporal module, (3) visual projector (4) a decoder-only large language model (LLM); and a post-processing module.

---
## Benchmark and Annotation Pipeline

<p align="center">
  <img src="assets/videomolmo_annotation_pipeline.png" alt="Annotation Pipeline">
</p>

We propose a semi-automatic annotation pipeline for creating a grounded conversation generation (GCG) dataset for videos.


---

## Running VideoMolmo 🔧

### Environment setup

```bash
git clone https://github.com/mbzuai-oryx/VideoMolmo
cd VideoMolmo
conda create -n .videomolmo python=3.10 -y
conda activate .videomolmo
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
git clone https://github.com/allenai/molmo.git
cd molmo && pip install -e .[all] && cd .. && rm -rf molmo # setup molmo requirements
pip install -r requirements.txt
python setup.py build_ext --inplace # build sam2
gdown https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O VideoMolmo/sam2/sam2_checkpoints/sam2_hiera_large.pt
```

### Inference on sample video
To inference on sample video in `examples/video_sample1`
```bash
python infer.py
```

### Training and Evaluation 🚀

To be released soon! Stay tuned for updates.


## Todos

- [ ] Release training and evaluation scripts.
- [ ] Provide detailed benchmarks and comparisons.
- [ ] Add support for additional datasets.
- [ ] Release dataset creation pipeline.


## Citation 📜

```bibtex
@article{ghazi2025videomolmo,
  title={VideoMolmo: Spatio-Temporal Grounding meets Pointing}, 
  author={Ghazi Ahmad, Ahmed Heakl, Hanan Ghani, Abdelrahman Shaker, Ranjay Krishna, Fahad Shahbaz Khan, Salman Khan},
  year={2025}
}
```

---

[<img src="assets/MBZUAI_logo.png" width="360" height="90">](https://mbzuai.ac.ae)
[<img src="assets/allenai_logo.png" width="300" height="85">](https://allenai.org/)
[<img src="assets/UW_logo.png" width="300">](https://www.washington.edu/)
