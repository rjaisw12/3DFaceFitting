import numpy as np
import face_alignment
from skimage import io
import torch

def extract_lmks(image, name, dest_folder):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device=device)
    image = np.array(image)
    preds = fa.get_landmarks(image)
    
    np.save(dest_folder + name + '.npy', preds[0])