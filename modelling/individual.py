import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex
)

class Individual():
    def __init__(self, image, mask, lmks, device):
        self.device = device

        self.source_image = np.asarray(image)/255.0
        self.source_image = torch.from_numpy(self.source_image).to(self.device)
        self.image_shape = np.asarray(image).shape

        self.image_mask = torch.from_numpy(mask).to(self.device)
        
        self.image_lmks = torch.from_numpy(lmks).to(self.device)
