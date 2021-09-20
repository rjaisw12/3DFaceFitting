import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch3d.renderer.utils import TensorProperties, convert_to_tensors_and_broadcast

class SHLight(nn.Module):
    """ Class defining Spherical Harmonics Light
    See https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf
    for more information
    """

    def __init__(self, ambient_color=(0.0,0.0,0.0), device: str = "cpu"):
        super(SHLight, self).__init__()
        self.device = device
        self.ambient_color = torch.tensor(ambient_color).to(self.device)
        self.init_SH_coefs()

    def init_SH_coefs(self):
        self.sh_coefs_R = nn.Parameter(torch.tensor([ 0.7062,  0.0589, -0.0064, -0.2396, -0.0467,  0.1415,  0.3112,  0.0597,
         0.2472], requires_grad=True).to(self.device))
        self.sh_coefs_G = nn.Parameter(torch.tensor([ 0.9726,  0.0158, -0.5242, -0.1780, -0.1436,  0.2836,  0.8258, -0.1344,
         0.3074], requires_grad=True).to(self.device))
        self.sh_coefs_B = nn.Parameter(torch.tensor([ 1.4181,  0.0594, -1.1839, -0.0680, -0.2719,  0.3132,  1.3769, -0.3756,
         0.1604], requires_grad=True).to(self.device))

    def compute_illumination_matrix(self, coefs):
        M = torch.stack((0.429043 * coefs[8], 0.429043 * coefs[4],  0.429043 * coefs[7],  0.511664 * coefs[3], 
                        0.429043 * coefs[4], -0.429043 * coefs[4],  0.429043 * coefs[5],  0.511664 * coefs[1],
                        0.429043 * coefs[7], 0.429043 * coefs[5],  0.743125 * coefs[6],  0.511664 * coefs[2],
                        0.511664 * coefs[3], 0.511664 * coefs[1],  0.511664 * coefs[2],  0.886227 * coefs[0] -  0.247708 * coefs[6]
        ))
        M = M.view(4,4)
        return M

    def compute_illumination_matrices(self):
        M_R = self.compute_illumination_matrix(self.sh_coefs_R)
        M_G = self.compute_illumination_matrix(self.sh_coefs_G)
        M_B = self.compute_illumination_matrix(self.sh_coefs_B)

        return M_R, M_G, M_B

    def compute_irradiance(self, normals, M):
        irradiance = torch.matmul(normals, M)
        irradiance = torch.sum(irradiance * normals, dim=1).view(-1, 1)
        return irradiance

    def diffuse(self, normals, points=None) -> torch.Tensor:
        normals = F.normalize(normals, p=2, dim=-1, eps=1e-6)
        normals_dims = normals.shape
        normals = normals.view(-1, 3)
        n_normals = normals.shape[0]
        normals = torch.cat([normals, torch.ones((n_normals, 1)).to(self.device)], dim=1)
        M_R, M_G, M_B = self.compute_illumination_matrices()
        irradiance_R = self.compute_irradiance(normals, M_R)
        irradiance_G = self.compute_irradiance(normals, M_G)
        irradiance_B = self.compute_irradiance(normals, M_B)
        irradiance = torch.cat([irradiance_R, irradiance_G, irradiance_B], dim=1)

        irradiance = irradiance.view(normals_dims)
        return irradiance

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        return torch.zeros(normals.shape).to(self.device)

        

if __name__ == '__main__':
    pass