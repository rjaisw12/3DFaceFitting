import os

import numpy as np
from skimage import io
from skimage.transform import pyramid_gaussian


def generate_gaussian_pyramid(folder: str, image_file: str):
    """This function generates a gaussian pyramid of image
    given a main full resolution image

    Args:
        folder (str): name of the folder containing the image
        image_file (str): name of the image we want to generate
                          the gaussian pyramid on
    """
    image = io.imread(folder + image_file)
    rows, cols, dim = image.shape
    pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True))

    dest_folder = folder + "images/"
    outfilename = dest_folder + "full.jpg"
    io.imsave(outfilename, pyramid[0])
    outfilename = dest_folder + "half.jpg"
    io.imsave(outfilename, pyramid[1])
    outfilename = dest_folder + "quarter.jpg"
    io.imsave(outfilename, pyramid[2])
