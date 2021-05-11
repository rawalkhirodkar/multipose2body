from os.path import split
import imageio
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
import torch.optim as optim
import numpy as np
import glob
import os
ROOT_DIR = '/home/rawal/Desktop/multipose2body/pix2pixHD/datasets/person_left'

# --------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256

loader = transforms.Compose([
    transforms.CenterCrop((2*imsize,imsize)),
    transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()  # reconvert into PIL image

def load_image(image_name):
    image = Image.open(image_name)
    w, h = image.size
    if h>=2*w:
        image = image.resize((imsize, int(imsize*1.0/w*h)))
    else:
        image = image.resize((int(imsize*1.0*2/h*w),imsize*2))
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def save_image(tensor, save_name):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save(save_name)

if __name__ == '__main__':

    root_dir = ROOT_DIR

    # ---------------------------------------
    filepath = os.path.join(root_dir, 'pose')
    datapath = glob.glob('{}/*.jpg'.format(filepath))
    savepath = os.path.join(root_dir, 'train_A')

    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for data in datapath:
        split_path = os.path.split(data)
        filename = split_path[-1]
        print(filename)
        img = load_image(data)
        resize_img = save_image(img, os.path.join(savepath, filename))

    # ---------------------------------------
    filepath = os.path.join(root_dir, 'rgb')
    datapath = glob.glob('{}/*.jpg'.format(filepath))
    savepath = os.path.join(root_dir, 'train_B')

    if not os.path.exists(savepath):
        os.mkdir(savepath)
    for data in datapath:
        split_path = os.path.split(data)
        filename = split_path[-1]
        print(filename)
        img = load_image(data)
        resize_img = save_image(img, os.path.join(savepath,filename))