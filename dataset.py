import torch
from torchvision import datasets
from PIL import Image

import os
import os.path
import sys
from torchvision import transforms
from torchvision import models
from torch.utils.data import DataLoader

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class PACS_Dataset():
  def __init__(self, root, transform=None, target_transform=None):
  #classes (list): List of the class names sorted alphabetically.
  #class_to_idx (dict): Dict with items (class_name, class_index).
  #imgs (list): List of (image path, class_index) tuples

    image_dataset = datasets.ImageFolder(root, transform)
    self.classes = image_dataset.classes 
    self.items = image_dataset.imgs
    self.class_to_idx = image_dataset.class_to_idx
    self.transform = transform

  def __len__(self):
    length = len(self.items)
    return length

  def __getitem__(self, index):
    t = transforms.ToTensor()

    #By the index, access directly the img path
    image, label = self.items[index]    
    image = pil_loader(image)
    # Applies preprocessing when accessing the image
    if self.transform is not None:
        image = self.transform(image)

    #return t(image), label
    
    return image, label

  def meanAndStd(self, loader):  
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
