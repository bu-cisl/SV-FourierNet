# SV-FourierNet
This repository contains the python implementations of the paper: **Wide-Field, High-Resolution Reconstruction in Computational Multi-Aperture Miniscope Using a Fourier Neural Network**. We provide model, pre-trained weights, test data and a quick demo.

<p align="center">
  <img src="/images/overview.png">
</p>

### Citation
If you find this project useful in your research, please consider citing our paper: 
[**Qianwan Yang, Ruipeng Guo, Guorong Hu, Yujia Xue, Yunzhe Li, and Lei Tian, "Wide-Field, High-Resolution Reconstruction in Computational Multi-Aperture Miniscope Using a Fourier Neural Network"**]([https://arxiv.org/abs/2403.06439])



### Abstract
Traditional fluorescence microscopy is constrained by inherent trade-offs among resolution, field-of-view, and system complexity. To navigate these challenges, we introduce a simple and low-cost computational multi-aperture miniature microscope, utilizing a microlens array for single-shot wide-field, high-resolution imaging. Addressing the challenges posed by extensive view multiplexing and non-local, shift-variant aberrations in this device, we present SV-FourierNet, a novel multi-channel Fourier neural network. SV-FourierNet facilitates high-resolution image reconstruction across the entire imaging field through its learned global receptive field. We establish a close relationship between the physical spatially-varying point-spread functions and the network's learned effective receptive field. This ensures that SV-FourierNet has effectively encapsulated the spatially-varying aberrations in our system, and learned a physically meaningful function for image reconstruction. Training of SV-FourierNet is conducted entirely on a physics-based simulator. We showcase wide-field, high-resolution video reconstructions on colonies of freely moving C. elegans and imaging of a mouse brain section. Our computational multi-aperture miniature microscope, augmented with SV-FourierNet, represents a major advancement in computational microscopy and may find broad applications in biomedical research and other fields requiring compact microscopy solutions.

## Environment
- Python 3.10.10
- pytorch 2.0.0
- TensorboardX
- numpy, scikit-image, tifffile, pandas, pytorch_msssim

### Download pre-trained weights
You can download pre-trained weights from [here](https://drive.google.com/drive/folders/1I7HQJXW6_HEUPf69cDuRfdlF9KAEYhKU?usp=sharing)

### How to use
After download the pre-trained weights file, put it under the root directory and run [test.py](test.py).

## Contact
For further information, please feel free to contact the author Qianwan Yang (yaw@bu.edu).

