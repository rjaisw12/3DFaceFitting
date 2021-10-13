import face_alignment
import numpy as np
import torch
from PIL.JpegImagePlugin import JpegImageFile
from skimage import io


def extract_lmks(image: JpegImageFile, name: str, dest_folder: str):
    """ Function that computes an array of 3D DLIB landmarks for a given
    image, and save them in a file.

    Args:
        image (JpegImageFile): input image to compute landmarks on
        name (str): name of the image size
        dest_folder (str): folder where to store the landmark array
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._3D, flip_input=True, device=device
    )
    image = np.array(image)
    preds = fa.get_landmarks(image)

    np.save(dest_folder + name + ".npy", preds[0])
