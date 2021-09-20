import h5py
import numpy as np
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesVertex
)

class FaceModel(nn.Module):
    def __init__(self, device):
        super(FaceModel, self).__init__()
        self.f = h5py.File('modelling/model2019_fullHead.h5', 'r')
        self.device = device
        self.n_components = 80

        # PCA Basis
        self.pcaBasis_shape = torch.from_numpy(self.f['shape']['model']['pcaBasis'][()][:,:self.n_components]).to(device)
        self.pcaBasis_color = torch.from_numpy(self.f['color']['model']['pcaBasis'][()][:,:self.n_components]).to(device)

        # PCA Variances
        self.pcaSTD_shape = torch.from_numpy(self.f['shape']['model']['pcaVariance'][()][:self.n_components]).to(device)
        self.pcaSTD_shape = torch.sqrt(self.pcaSTD_shape)
        
        self.pcaSTD_color = torch.from_numpy(self.f['color']['model']['pcaVariance'][()][:self.n_components]).to(device)
        self.pcaSTD_color = torch.sqrt(self.pcaSTD_color)

        # Model Parameters
        self.shape_parameters = nn.Parameter(torch.zeros(self.n_components).to(device), requires_grad=True)
        self.color_parameters = nn.Parameter(torch.zeros(self.n_components).to(device), requires_grad=True)

        # Data for Mean Face
        self.mean_face = self.load_mean_face()

    def load_mean_face(self):
        """ Create Mesh of Mean Face from BFM """

        verts = torch.from_numpy(self.f['shape']['model']['mean'][()].reshape(1, -1, 3)).to(self.device)
        triangles = torch.from_numpy(self.f['shape']['representer']['cells'][()]).T.unsqueeze(0).to(self.device)
        colors = torch.from_numpy(self.f['color']['model']['mean'][()]).reshape(-1, 3).unsqueeze(0).to(self.device)
        textures = TexturesVertex(verts_features=colors)
        mean_face = Meshes(verts=verts,
                           faces=triangles,
                           textures=textures).to(self.device)

        return mean_face

    def compute_face(self):
        """ Create Mesh of parametrized face from BFM """

        shape_deform = torch.sum(self.pcaBasis_shape * self.shape_parameters, dim=1).view(-1,3)
        texture_deform = torch.sum(self.pcaBasis_color * self.color_parameters, dim=1).view(-1,3)
        face = self.mean_face.offset_verts(shape_deform)
        face.textures = TexturesVertex(verts_features=(face.textures.verts_features_padded() + texture_deform))

        return face

    '''def compute_deformed_mesh(self):
        """ Sample deformation form BFM model and compute deformed mesh """

        shape_deform = torch.sum(self.pcaBasis_shape * self.shape_parameters, dim=1).view(-1,3)
        texture_deform = torch.sum(self.pcaBasis_color * self.color_parameters, dim=1).view(-1,3)
        deformed_mesh = self.mean_face.clone()
        deformed_mesh = deformed_mesh.offset_verts(shape_deform)
        deformed_mesh.textures = TexturesVertex(verts_features=(deformed_mesh.textures.verts_features_padded() + texture_deform))

        return deformed_mesh '''

    def compute_loss_reg(self):
        loss_reg_shape = (self.shape_parameters/self.pcaSTD_shape) ** 2
        loss_reg_color = (self.color_parameters/self.pcaSTD_color) ** 2
        loss_reg = torch.sum(loss_reg_shape + loss_reg_color)

        return loss_reg

    
