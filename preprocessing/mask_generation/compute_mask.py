import numpy as np
import torch
from PIL import Image
from preprocessing.mask_generation.models import LinkNet34
import torchvision.transforms as transforms

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LinkNet34()
    model.load_state_dict(torch.load('preprocessing/mask_generation/linknet.pth', map_location=lambda storage, loc: storage))
    model.eval()
    model.to(device)

    return model, device

def compute_mask(img, name, dest_folder):
    model, device = load_model()

    img_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
    ])

    to_original_size = transforms.Resize(size=np.shape(img)[:2])
    transformed_img = img_transform(img).to(device)
    transformed_img = transformed_img.unsqueeze(0)
    pred = model(transformed_img).detach().cpu().numpy().squeeze()
    pred = to_original_size(Image.fromarray(pred))
    pred = np.array(pred)
    pred[pred<0.5]=0
    pred[pred>1.0]=1.0

    np.save(dest_folder + name + '.npy', pred)

if __name__ == '__main__':
    pass