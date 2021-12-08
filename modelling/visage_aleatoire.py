import h5py
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from face_model import FaceModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fm = FaceModel(device)

fm.shape_parameters = torch.normal(torch.zeros(fm.n_components), fm.pcaSTD_shape)
fm.color_parameters = torch.normal(torch.zeros(fm.n_components), fm.pcaSTD_color)

image_3d = fm.compute_face()