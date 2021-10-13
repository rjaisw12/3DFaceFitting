from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from torch._C import device

from preprocessing.mask_generation.models import LinkNet34


def load_model() -> Tuple[LinkNet34, torch.device]:
    """ This function load a pretrained segmentation model.

    Returns:
        model (LinkNet34): pretrained segmentation model
        device (torch.device): the device on which the model
                               will make inference
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LinkNet34()
    model.load_state_dict(
        torch.load(
            "preprocessing/mask_generation/linknet.pth",
            map_location=lambda storage, loc: storage,
        )
    )
    model.eval()
    model.to(device)

    return model, device


def compute_mask(img: JpegImageFile, name: str, dest_folder: str):
    """ This function computes a segmentation mask
    given an image and saves it to a file.

    Args:
        img (JpegImageFile): Image to segment
        name (str): name of the image resolution (quarter, half, full)
        dest_folder (str): folder where to save the mask
    """
    model, device = load_model()

    img_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
            ),
        ]
    )

    to_original_size = transforms.Resize(size=np.shape(img)[:2])
    transformed_img = img_transform(img).to(device)
    transformed_img = transformed_img.unsqueeze(0)
    pred = model(transformed_img).detach().cpu().numpy().squeeze()
    pred = to_original_size(Image.fromarray(pred))
    pred = np.array(pred)
    pred[pred < 0.5] = 0
    pred[pred > 1.0] = 1.0

    np.save(dest_folder + name + ".npy", pred)
