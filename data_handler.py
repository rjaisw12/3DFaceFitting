""" This module contains the definition of the class DataHandler
which encapsulate all the information regarding an image:
- images (sampled at 3 resolution)
- masks
- landmark files
"""

import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image, ImageDraw
from PIL.JpegImagePlugin import JpegImageFile

from modelling.individual import Individual
from preprocessing.gaussian_pyramid import generate_gaussian_pyramid
from preprocessing.landmarks_extraction.extract_landmarks import extract_lmks
from preprocessing.mask_generation.compute_mask import compute_mask


class DataHandler:
    def __init__(self, folder: str, device: torch.device):
        """Init function that load images, masks and landmarks files

        Args:
            folder (str): Folder containing the image to process
            device (torch.device): CPU or GPU
        """
        self.device = device
        self.folder = folder
        self.images = self.load_images()
        self.masks = self.load_masks()
        self.lmks = self.load_lmks()

        self.full_individual = Individual(
            self.images["full"], self.masks["full"], self.lmks["full"], self.device
        )
        self.half_individual = Individual(
            self.images["half"], self.masks["half"], self.lmks["half"], self.device
        )
        self.quarter_individual = Individual(
            self.images["quarter"],
            self.masks["quarter"],
            self.lmks["quarter"],
            self.device,
        )
        Path(self.folder + "render/").mkdir(parents=True, exist_ok=True)

    def load_images(self) -> Dict[str, JpegImageFile]:
        """This function generates a gaussian pyramid given a main full
        resolution image to facilitate further fitting of a 3D model

        Returns:
            Dict[str: PIL.JpegImagePlugin.JpegImageFile]: Dictionnary containing the images of the pyramid
        """
        if not os.path.exists(self.folder + "images/"):
            image_files = [
                fname for fname in os.listdir(self.folder) if fname.endswith(".jpg")
            ]
            try:
                main_image = image_files[0]
            except IOError:
                print("Couldn't find input file")

            Path(self.folder + "images/").mkdir(parents=True, exist_ok=True)
            generate_gaussian_pyramid(self.folder, main_image)

        full_image = Image.open(self.folder + "images/full.jpg")
        half_image = Image.open(self.folder + "images/half.jpg")
        quarter_image = Image.open(self.folder + "images/quarter.jpg")

        return {"quarter": quarter_image, "half": half_image, "full": full_image}

    def load_masks(self) -> Dict[str, np.array]:
        """ Function that generates masks for the images of the pyramid
        The masks are used the know which pixels of the images
        correspond to face or not

        Returns:
            Dict[str, np.array]: Dictionnary containing a mask
            for each image of the pyramid.
        """
        if not os.path.exists(self.folder + "masks/"):
            Path(self.folder + "masks/").mkdir(parents=True, exist_ok=True)
            for name, img in self.images.items():
                compute_mask(img, name, self.folder + "masks/")

        full_mask = np.load(self.folder + "masks/full.npy")
        half_mask = np.load(self.folder + "masks/half.npy")
        quarter_mask = np.load(self.folder + "masks/quarter.npy")

        return {"quarter": quarter_mask, "half": half_mask, "full": full_mask}

    def load_lmks(self) -> Dict[str, np.array]:
        """ Function that generates landmark arrays for the images of the pyramid

        Returns:
            Dict[str, np.array]: Dictionnary of 68x3 numpy arrays that contain
                                 the 3D estimated positions of the 68 landmarks
                                 of DLIB for each image of the pyramid
        """
        if not os.path.exists(self.folder + "lmks/"):
            Path(self.folder + "lmks/").mkdir(parents=True, exist_ok=True)
            for name, img in self.images.items():
                extract_lmks(img, name, self.folder + "lmks/")

        full_lmks = np.load(self.folder + "lmks/full.npy")
        half_lmks = np.load(self.folder + "lmks/half.npy")
        quarter_lmks = np.load(self.folder + "lmks/quarter.npy")

        return {"quarter": quarter_lmks, "half": half_lmks, "full": full_lmks}
