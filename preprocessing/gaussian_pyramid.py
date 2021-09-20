import os
from skimage import io
from skimage.transform import pyramid_gaussian
import numpy as np

def generate_gaussian_pyramid(folder, image_file):
    image = io.imread(folder + image_file)
    rows, cols, dim = image.shape
    pyramid = tuple(pyramid_gaussian(image, downscale=2, multichannel=True))

    dest_folder = folder + 'images/'
    outfilename = dest_folder + 'full.jpg'
    io.imsave(outfilename, pyramid[0])
    outfilename = dest_folder + 'half.jpg'
    io.imsave(outfilename, pyramid[1])
    outfilename = dest_folder + 'quarter.jpg'
    io.imsave(outfilename, pyramid[2])

if __name__ == '__main__':
    generate_gaussian_pyramid('../people/CFD/WF-029/CFD-WF-029-002-N.jpg')