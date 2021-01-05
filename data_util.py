import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import Compose, Grayscale, ToTensor, Normalize, ToPILImage

classes = ['D', 'UD']

class Towers(Dataset):
    def __init__(self, data_path=None, transform=None):
        self.data = datasets.ImageFolder(root = data_path, transform = transform)
        self.data_path = data_path
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img_tensor = self.data[index][0]
        img = ToPILImage()(img_tensor).convert('RGB')
        if self.transform != None:
            img = self.transform(img)
        label = self.data[index][1]
        return (img, label)