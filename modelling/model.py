import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelling.face_model import FaceModel
from modelling.individual import Individual
from modelling.render_model import RenderModel
from modelling.bfm_lmks import BFM_lmks_v

class Model(nn.Module):
    def __init__(self, individual, device):
        super(Model, self).__init__()
        self.device = device
        self.face_model = FaceModel(device)
        self.individual = individual
        self.render_model = RenderModel(individual, device)
        self.content_weight = 2000
        self.lm_weight = 1.0
        self.reg_weight = 0.002

    def compute_projected_lmks(self, mesh, R, T):
        """ Compute the (x, y) positions of the 68 face landmarks on the screen space """

        lmks_verts = mesh.verts_padded()[:, BFM_lmks_v, :]
        camera = self.render_model.renderer.rasterizer.cameras
        proj_lmks = camera.transform_points_screen(lmks_verts,
                                                   torch.tensor([[self.individual.image_shape[1], self.individual.image_shape[0]]]).to(self.device),
                                                   R=R, 
                                                   T=T)
        return proj_lmks[0, :, :2]

    def forward(self):
        mesh_to_render = self.face_model.compute_face()
        R , T = self.render_model.compute_rigid_transformation()

        image = self.render_model.renderer(meshes_world=mesh_to_render, R=R, T=T)
        proj_lmks = self.compute_projected_lmks(mesh_to_render, R, T)

        # Calculate the Loss
        loss_content = self.content_weight * (1/torch.sum(self.individual.image_mask)) * torch.sum(((image[0,:,:,:3] - self.individual.source_image)*self.individual.image_mask.unsqueeze(-1)) ** 2)
        loss_lm = self.lm_weight*(1/68.0)*torch.sum(torch.norm(proj_lmks - self.individual.image_lmks[...,:2], dim=1) ** 2)
        loss_reg = self.reg_weight * self.face_model.compute_loss_reg()
        total_loss = loss_content + loss_lm + loss_reg
        losses = {'loss_content': loss_content, 'loss_lm': loss_lm, 'loss_reg': loss_reg, 'total_loss': total_loss}

        return losses, image

