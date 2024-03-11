# SV-FourierNet
This repository contains the python implementations of the paper: **Wide-Field, High-Resolution Reconstruction in Computational Multi-Aperture Miniscope Using a Fourier Neural Network**. We provide model, pre-trained weights, test data and a quick demo.


### Citation
If you find this project useful in your research, please consider citing our paper:

[**Yunzhe Li, Yujia Xue, and Lei Tian, "Deep speckle correlation: a deep learning approach toward scalable imaging through scattering media," Optica 5, 1181-1190 (2018)**](https://www.osapublishing.org/optica/abstract.cfm?uri=optica-5-10-1181)


### Abstract
Traditional fluorescence microscopy is constrained by inherent trade-offs among resolution, field-of-view, and system complexity. To navigate these challenges, we introduce a simple and low-cost computational multi-aperture miniature microscope, utilizing a microlens array for single-shot wide-field, high-resolution imaging. Addressing the challenges posed by extensive view multiplexing and non-local, shift-variant aberrations in this device, we present SV-FourierNet, a novel multi-channel Fourier neural network. SV-FourierNet facilitates high-resolution image reconstruction across the entire imaging field through its learned global receptive field. We establish a close relationship between the physical spatially-varying point-spread functions and the network's learned effective receptive field. This ensures that SV-FourierNet has effectively encapsulated the spatially-varying aberrations in our system, and learned a physically meaningful function for image reconstruction. Training of SV-FourierNet is conducted entirely on a physics-based simulator. We showcase wide-field, high-resolution video reconstructions on colonies of freely moving C. elegans and imaging of a mouse brain section. Our computational multi-aperture miniature microscope, augmented with SV-FourierNet, represents a major advancement in computational microscopy and may find broad applications in biomedical research and other fields requiring compact microscopy solutions.

<p align="center">
  <img src="/images/img1.png">
</p>


### Requirements
python

pytorch 

numpy

matplotlib

scikit-image

tifffile

pandas

pytorch_msssim


### CNN architecture
<p align="center">
  <img src="/images/img2.png">
</p>


### Download pre-trained weights
You can download pre-trained weights from [here](https://www.dropbox.com/s/e1qcrv9o3i0h8z3/pretrained_weights.hdf5?dl=0)


### How to use
After download the pre-trained weights file, put it under the root directory and run [demo.py](demo.py).


### Results
<p align="center">
  <img src="/images/img3.png">
</p>
