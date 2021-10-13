""" This module contains the definition of the class RenderModel which is used
to render a scene given multiple parameters 
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from pytorch3d.renderer import (DirectionalLights, FoVOrthographicCameras,
                                FoVPerspectiveCameras, HardPhongShader,
                                Materials, MeshRasterizer, MeshRenderer,
                                RasterizationSettings, look_at_rotation,
                                look_at_view_transform)
from pytorch3d.transforms import euler_angles_to_matrix
from torch.nn.parameter import Parameter
from torch.tensor import Tensor

from modelling.individual import Individual
from modelling.sh_light import SHLight


class RenderModel(nn.Module):
    def __init__(self, individual: Individual, device: torch.device):
        """Initialization of all parameters relevant for rendering:
        (camera, light, material, rasterization, shader settings)
        """
        super(RenderModel, self).__init__()

        self.device = device
        self.image_size = torch.tensor(
            [[individual.image_shape[0], individual.image_shape[1]]]
        ).to(device)
        self.init_camera_params()
        self.init_light_params()
        self.init_material_params()

        self.raster_settings = RasterizationSettings(
            image_size=(individual.image_shape[0], individual.image_shape[1]),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,
            cull_backfaces=True,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.camera, raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                device=device,
                cameras=self.camera,
                lights=self.light,
                materials=self.material,
            ),
        )

    def init_material_params(self):
        """Initialize the material parameters"""

        self.shininess = torch.tensor([64.0]).to(self.device)
        self.material = Materials(
            ambient_color=torch.zeros((1, 3)),
            diffuse_color=torch.ones((1, 3)),
            specular_color=torch.zeros((1, 3)),
            shininess=self.shininess,
            device=self.device,
        )

    def init_camera_params(self):
        """Initialize the camera parameters"""

        # PERSPECTIVE CAMERA
        # self.euler_angles = nn.Parameter(torch.tensor([[-0.2237,  3.1450, -0.0516]]).to(self.device))
        # self.T = nn.Parameter(torch.tensor([[-6.3663,  -15.7904, 1234.1265]]).to(self.device))
        # self.fov = nn.Parameter(torch.tensor([12.0755]).to(self.device))
        # self.camera = FoVPerspectiveCameras(fov=self.fov, device=self.device)

        # ORTHOGRAPHIC CAMERA
        self.euler_angles = nn.Parameter(
            torch.tensor([[-0.0568, 3.1195, 0.1509]]).to(self.device)
        )
        self.T = nn.Parameter(
            torch.tensor([[12.8240, -18.4864, 300.0000]]).to(self.device)
        )
        self.max_x = nn.Parameter(torch.tensor([97.9774]).to(self.device))
        self.scale = nn.Parameter(torch.tensor([[0.7, 1.0, 1.0000]]).to(self.device))
        self.camera = FoVOrthographicCameras(
            device=self.device,
            min_x=-self.max_x,
            max_x=self.max_x,
            min_y=-self.max_x,
            max_y=self.max_x,
            scale_xyz=self.scale,
        )

    def init_light_params(self):
        """Initialize the light parameters"""

        self.light = SHLight(device=self.device)

    def compute_rigid_transformation(self) -> Tuple[Tensor, Parameter]:
        """Method that computes Rotation and Translation matrix from camera parameters

        Returns:
            R (Tensor): Rotation matrix
            T (Parameter): Translation vector
        """

        R = euler_angles_to_matrix(self.euler_angles, convention="XYZ")
        return R, self.T
