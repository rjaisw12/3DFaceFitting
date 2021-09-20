import os
import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from preprocessing.gaussian_pyramid import generate_gaussian_pyramid
from preprocessing.mask_generation.compute_mask import compute_mask
from preprocessing.landmarks_extraction.extract_landmarks import extract_lmks
from modelling.individual import Individual


class DataHandler():
    def __init__(self, folder, device):
        self.device = device
        self.folder = folder
        self.images = self.load_images()
        self.masks = self.load_masks()
        self.lmks = self.load_lmks()

        self.full_individual = Individual(self.images['full'], self.masks['full'], self.lmks['full'], self.device)
        self.half_individual = Individual(self.images['half'], self.masks['half'], self.lmks['half'], self.device)
        self.quarter_individual = Individual(self.images['quarter'], self.masks['quarter'], self.lmks['quarter'], self.device)
        Path(self.folder + "render/").mkdir(parents=True, exist_ok=True)
        
    def load_images(self):
        if not os.path.exists(self.folder + "images/"):
            image_files = [fname for fname in os.listdir(self.folder) if fname.endswith('N.jpg')]
            try:
                main_image = image_files[0]
            except IOError:
                print("Couldn't find input file")

            Path(self.folder+"images/").mkdir(parents=True, exist_ok=True)
            generate_gaussian_pyramid(self.folder, main_image)
            files = glob.glob(self.folder + '*.jpg')
            for f in files:
                os.remove(f)

        full_image = Image.open(self.folder + 'images/full.jpg')
        half_image = Image.open(self.folder + 'images/half.jpg')
        quarter_image = Image.open(self.folder + 'images/quarter.jpg')

        return {'quarter': quarter_image, 'half': half_image, 'full': full_image}

    def load_masks(self):
        if not os.path.exists(self.folder + "masks/"):
            Path(self.folder + "masks/").mkdir(parents=True, exist_ok=True)
            for name, img in self.images.items():
                compute_mask(img, name, self.folder+"masks/")

        full_mask = np.load(self.folder + 'masks/full.npy')
        half_mask = np.load(self.folder + 'masks/half.npy')
        quarter_mask = np.load(self.folder + 'masks/quarter.npy')

        return {'quarter': quarter_mask, 'half': half_mask, 'full': full_mask}

    def load_lmks(self):
        if not os.path.exists(self.folder + "lmks/"):
            Path(self.folder + "lmks/").mkdir(parents=True, exist_ok=True)
            for name, img in self.images.items():
                extract_lmks(img, name, self.folder+"lmks/")

        full_lmks = np.load(self.folder + 'lmks/full.npy')
        half_lmks = np.load(self.folder + 'lmks/half.npy')
        quarter_lmks = np.load(self.folder + 'lmks/quarter.npy')

        return {'quarter': quarter_lmks, 'half': half_lmks, 'full': full_lmks}
        


if __name__ == '__main__':
    pass
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #dh = DataHandler(folder = 'people/CFD/WF-029/', device)