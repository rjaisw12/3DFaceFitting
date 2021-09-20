import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from modelling.model import Model
from PIL import Image
from skimage import io
import pickle

def load_model(individual, folder, device):
    model = Model(individual, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    if os.path.exists(folder + 'model_params.pth'):
        checkpoint = torch.load(folder + 'model_params.pth')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

def print_loss(losses):
    print(f"loss_content: {losses['loss_content']}")
    print(f"loss_lm: {losses['loss_lm']}")
    print(f"loss_reg: {losses['loss_reg']}")

def fit_image(individual, folder, image_type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, optimizer = load_model(individual, folder, device)

    loop = tqdm(range(2000))
    previous_loss = np.float('inf')
    for i in loop:
        optimizer.zero_grad()
        losses, image = model()
        losses['total_loss'].backward()
        optimizer.step()
        if i%100==0:
            print_loss(losses)
            curr_loss = losses['total_loss'].cpu().detach().numpy()
            pct_loss_decrease = 1 - curr_loss/previous_loss
            previous_loss = curr_loss
            if pct_loss_decrease < 0.0001:
                break

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, folder + 'model_params.pth')

    image = Image.fromarray(np.uint8(image[0, ..., :3].detach().cpu().numpy()* 255))
    image.save(folder + 'render/' + image_type + '.png')
