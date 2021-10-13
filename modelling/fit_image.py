""" This is the main module that contains the functions to optimize the model 
and thus fit the 3D model to the image """

import os
import pickle
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import io
from torch.optim.adam import Adam
from tqdm import tqdm

from modelling.individual import Individual
from modelling.model import Model


def load_model(
    individual: Individual, folder: str, device: torch.device
) -> Tuple[Model, Adam]:
    """Function that initialize a model for 3D model fitting
    and loads model parameters and optimizer state if exist

    Args:
        individual (Individual): (image + mask + landmarks)
        folder (str): folder where are stored model parameters and optimizer state if exist
        device (torch.device): CPU or GPU

    Returns:
        model (Model): model to render and compute losses
        optimizer (Adam): Adam optimizer
    """
    model = Model(individual, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    if os.path.exists(folder + "model_params.pth"):
        checkpoint = torch.load(folder + "model_params.pth")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer


def print_loss(losses: Dict[str, float]):
    """ Function that display the current loss of the model

    Args:
        losses (Dict[str, float]): the current losses of the model
    """
    print(f"loss_content: {losses['loss_content']}")
    print(f"loss_lm: {losses['loss_lm']}")
    print(f"loss_reg: {losses['loss_reg']}")


def fit_image(individual: Individual, folder: str, image_type: str):
    """ Main loop for model optimization (on a single image).
    Model is trained with gradient descent.
    parameters and optimizer state is saved
    Render image is saved

    Args:
        individual (Individual): (image + mask + landmarks)
        folder (str): folder where model parameters and optimizer state is saved
        image_type (str): (quarter, half or full)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, optimizer = load_model(individual, folder, device)

    loop = tqdm(range(2000))
    previous_loss = np.float("inf")
    for i in loop:
        optimizer.zero_grad()
        losses, image = model()
        losses["total_loss"].backward()
        optimizer.step()
        if i % 100 == 0:
            print_loss(losses)
            curr_loss = losses["total_loss"].cpu().detach().numpy()
            pct_loss_decrease = 1 - curr_loss / previous_loss
            previous_loss = curr_loss
            if pct_loss_decrease < 0.0001:
                break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        folder + "model_params.pth",
    )

    image = Image.fromarray(np.uint8(image[0, ..., :3].detach().cpu().numpy() * 255))
    image.save(folder + "render/" + image_type + ".png")
