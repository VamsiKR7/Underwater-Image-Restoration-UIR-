# Underwater-Image-Restoration-UIR-
## Underwater Image Restoration via SE-Blocks and Depthwise Separable Convolutions

## 📌 Project Overview
This repository contains the implementation of an unsupervised deep learning model designed to restore underwater images degraded by light absorption and scattering. By integrating Squeeze-and-Excitation (SE) Blocks and Depthwise Separable Convolutions (DSConv), we achieved a high-performance restoration model that is significantly more efficient than standard architectures.  
## 🚀 Key Highlights & Metrics
* Parameter Efficiency: Reduced model parameters by ~87.55% compared to the base USUIR model (30.3K vs 225.5K).
* Performance: Achieved the highest SSIM (0.844) and UCIQE (37.573) scores among compared state-of-the-art models like Deep WaveNet and RAUNE-Net.
* Unsupervised Learning: Implemented a Homology Constraint Loss that allows training without paired ground-truth images by enforcing consistency between raw and re-degraded outputs.
## 🛠 Methodology & Architecture
The architecture is composed of two primary sub-networks:  
* JDS-Net (Scene Radiance): Estimates the clean image content using DSConv and an SE-Block for adaptive channel re-weighting.
* TD-Net (Transmission Map): Estimates wavelength-dependent attenuation (Red, Green, Blue) to model underwater light physics.
Optimization Techniques
* Squeeze and Excitation Blocks(SEBlocks): Recalibrates channel-wise features to focus on informative color and texture while suppressing noise.
* Depthwise Separable Convolutions: Factorizes spatial and channel processing to drastically reduce computational cost while preserving local textures.
## 📊 Results
The model was trained and validated on the UIEBD (Underwater Image Enhancement Benchmark Dataset).  

|   Method  | PSNR |SSIM |Parameters|
|-----------|------|-----|----------|
|USUIR(Base)|20.449|0.839|  225.5K  |
|Our Model  |20.177|0.844|   30.3K  |

For more information, access the [paper link](https://drive.google.com/file/d/1y_urSudGHQekxacvUEx_T6UNqxoHbEHL/view?usp=sharing).
