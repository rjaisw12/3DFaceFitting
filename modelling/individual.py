""" This module contains the definition of the class Individual
which encapsulate all relevant informations on a image to fit a model:
- Image
- Mask
- Landmarks
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


class Individual:
    def __init__(self, image, mask, lmks, device):
        self.device = device

        self.source_image = np.asarray(image) / 255.0
        self.source_image = torch.from_numpy(self.source_image).to(self.device)
        self.image_shape = np.asarray(image).shape

        self.image_mask = torch.from_numpy(mask).to(self.device)

        self.image_lmks = torch.from_numpy(lmks).to(self.device)
