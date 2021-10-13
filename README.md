## Overview
This project aims to reconstruct a 3D face mesh given a single color image.
It is inspired by multiple papers including:
- https://niessnerlab.org/papers/2015/11face/thies2015realtime.pdf
- https://arxiv.org/pdf/1612.00523.pdf

It is based on top of the library Pytorch3D to take advantage of its differentiable renders

![Alt text](assets/donald_rendered.jpg?raw=true "Donald rendered")

## Getting Started

This project has been developped with python 3.8.5
It is highly recommended to make use of GPUs to avoid very
long computation time. (GTX1090 has been used for development)

To run the project locally follow these steps:
- Create a conda environment with the version 3.8.5 of python
- Install dependencies with pip3 install -r requirements.txt
  You might face difficulties to install Pytorch3D, if so please refer to
  https://github.com/facebookresearch/pytorch3d/ for help
- Download the Basel Face Model (BFM) from the website https://faces.dmi.unibas.ch/bfm/bfm2019.html
  The file you should download is model2019_fullHead.h5
  Then copy it to the folder modelling.
- Then run "python3 main.py"
  By default it will run a face fitting of Donald Trump which photo is stored in people/tests/donald.jpg and store the rendered 3D face fitted mesh in the render folder of the directory.

### Fitting you own photos
If you want to run a 3D face fitting on your own photos, please follow these steps:
- create a folder in people/tests containing a single photo you want to fit
- run "python3 main.py"

## License

This code is free to use for academic research purpose but not for commercial purposes.



## Contact

Raphael Jaiswal - rjaiswal@hotmail.fr
