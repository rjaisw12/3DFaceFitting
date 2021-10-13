""" This module is the main entry point for the project """

import os

import torch

from data_handler import DataHandler
from modelling.fit_image import fit_image


def fit_person(folder: str):
    """ This is the main function to fit a 3D model given 
    a folder containing a photo.

    Args:
        folder (str): [description]
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dh = DataHandler(folder, device)
    print('fitting quarter image')
    fit_image(dh.quarter_individual, dh.folder, 'quarter')
    print('fitting half image')
    fit_image(dh.half_individual, dh.folder, 'half')
    print('fitting full image')
    fit_image(dh.full_individual, dh.folder, 'full')

def main():
    """ Main Function to fit multiple 3D models """
    
    for subfolder in os.listdir('people/tests/'):
        folder = 'people/tests/' + subfolder + '/'
        print(folder)
        fit_person(folder)

if __name__ == '__main__':
    main()
