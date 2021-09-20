import os
import torch
from data_handler import DataHandler
from modelling.fit_image import fit_image

def fit_person(folder):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dh = DataHandler(folder, device)
    print('fitting quarter image')
    fit_image(dh.quarter_individual, dh.folder, 'quarter')
    #print('fitting half image')
    #fit_image(dh.half_individual, dh.folder, 'half')
    #print('fitting full image')
    #fit_image(dh.full_individual, dh.folder, 'full')

def main():
    for subfolder in os.listdir('people/CFD/'):
        folder = 'people/CFD/' + subfolder + '/'
        print(folder)
        fit_person(folder)

if __name__ == '__main__':
    main()
