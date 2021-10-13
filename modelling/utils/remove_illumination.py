""" This module contains function to remove illumination on an image
given an estimated illumination
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import Materials
from pytorch3d.renderer.mesh.shading import _apply_lighting

from data_handler import DataHandler
from modelling.fit_image import load_model
from modelling.model import Model


def remove_illumination(model: Model) -> np.array:
    """ Function that computes an image with illumination removed
    given an estimated illumination

    Args:
        model (Model): model containing image and illumination

    Returns:
        new_image (np.array): image with illumination removed
    """
    mesh = model.face_model.compute_face()
    fragments = model.render_model.renderer.rasterizer(mesh)

    camera = model.render_model.camera
    texels = mesh.sample_textures(fragments)
    lights = model.render_model.light
    materials = Materials(
        ambient_color=torch.zeros((1, 3)),
        diffuse_color=torch.ones((1, 3)),
        specular_color=torch.zeros((1, 3)),
        shininess=torch.tensor([64.0]).to(model.device),
        device=model.device,
    )

    verts = mesh.verts_packed()  # (V, 3)
    faces = mesh.faces_packed()  # (F, 3)
    vertex_normals = mesh.verts_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    faces_normals = vertex_normals[faces]
    pixel_coords = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts
    )
    pixel_normals = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_normals
    )
    ambient, diffuse, specular = _apply_lighting(
        pixel_coords, pixel_normals, lights, camera, materials
    )

    new_image = model.individual.source_image / diffuse.squeeze()

    return new_image
