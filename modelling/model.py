""" This module contains the definition of the class Model
which contains all the relevant methods to compute the loss for
the fitting of the 3D model """

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures.meshes import Meshes
from torch import Tensor
from torch.nn.parameter import Parameter

from modelling.bfm_lmks import BFM_lmks_v
from modelling.face_model import FaceModel
from modelling.individual import Individual
from modelling.render_model import RenderModel

Losses = Dict[str, float]


class Model(nn.Module):
    def __init__(self, individual: Individual, device: torch.device):
        """Initialization of the class

        Args:
            individual (Individual): Individual (image + mask + landmarks)
            device (torch.device): CPU or GPU
        """
        super(Model, self).__init__()
        self.device = device
        self.face_model = FaceModel(device)
        self.individual = individual
        self.render_model = RenderModel(individual, device)
        # hand-found weights that give good reconstruction results
        self.content_weight = 2000
        self.lm_weight = 1.0
        self.reg_weight = 0.002

    def compute_projected_lmks(self, mesh: Meshes, R: Tensor, T: Parameter) -> Tensor:
        """Compute the (x, y) positions of the 68 face landmarks on the screen space

        Args:
            mesh (Meshes): 3D model being fitted
            R (Tensor): Rotation of the camera in space (size 1x3x3)
            T (Parameter): Translation of the camera in space (size 1x3)

        Returns:
            proj_lmks (Tensor): 68x2 tensor that contains the 2D position of the 3D models
            vertices corresponding to the landmarks once projected on the camera plane.
        """

        lmks_verts = mesh.verts_padded()[:, BFM_lmks_v, :]
        camera = self.render_model.renderer.rasterizer.cameras
        proj_lmks = camera.transform_points_screen(
            lmks_verts,
            torch.tensor(
                [[self.individual.image_shape[1], self.individual.image_shape[0]]]
            ).to(self.device),
            R=R,
            T=T,
        )
        return proj_lmks[0, :, :2]

    def forward(self) -> Tuple[Losses, Tensor]:
        """method that render the scene given mesh, camera, light, material, ...
        and computes the losses of the 3D reconstruction

        Returns:
            losses (Losses): dictionnary that contains the losses corresponding
                             to pixels, landmarks and regulatization
            image (Tensor): rendered image
        """
        # Scene rendering
        mesh_to_render = self.face_model.compute_face()
        R, T = self.render_model.compute_rigid_transformation()
        image = self.render_model.renderer(meshes_world=mesh_to_render, R=R, T=T)

        # Losses computation
        loss_content = (
            self.content_weight
            * (1 / torch.sum(self.individual.image_mask))
            * torch.sum(
                (
                    (image[0, :, :, :3] - self.individual.source_image)
                    * self.individual.image_mask.unsqueeze(-1)
                )
                ** 2
            )
        )
        proj_lmks = self.compute_projected_lmks(mesh_to_render, R, T)
        loss_lm = (
            self.lm_weight
            * (1 / 68.0)
            * torch.sum(
                torch.norm(proj_lmks - self.individual.image_lmks[..., :2], dim=1) ** 2
            )
        )
        loss_reg = self.reg_weight * self.face_model.compute_loss_reg()
        total_loss = loss_content + loss_lm + loss_reg
        losses = {
            "loss_content": loss_content,
            "loss_lm": loss_lm,
            "loss_reg": loss_reg,
            "total_loss": total_loss,
        }

        return losses, image
