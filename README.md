# Semi-Supervised Learning with GANs for Device-Free Fingerprinting Indoor Localization (IEEE GLOBECOM 2020)
######  Last update: 6/10/2021
## Introduction:
Implementation of semi-supervised deep convolutional generative adversarial network (DCGAN) for Device Free Wi-Fi Fingerprinting Indoor Localization. For more details and evaluation results, please check out our original [paper](https://ieeexplore.ieee.org/document/9322456).

## Concept:
<img src="https://github.com/aciculachen/CSI-SemiGAN/blob/master/sGAN.png" width="600">
Add two indoor localization scenarios: Lounge, Office (see exp2, exp3).

## Features:

- main.py: train the models under the pre-defined indoor localization scenarios.
- generate_CSI.py: Generate CSI samples with pretrained GAN model.
- plot_CSI.py: code for plotting CSI samples
- models.py: definde semisupervised GAN and supervised CNN (benchmark)
- models: (1) old models for GLOBECOM 2020 (2) new simuliation results for exp1, exp2, and exp3
- dataset: pre-collected CSI samples of the scenarios saved as pickle in the form of (X_train, y_train, X_tst, y_tst)
## Dependencies:
- tensorflow 1.13
- python 3.6
