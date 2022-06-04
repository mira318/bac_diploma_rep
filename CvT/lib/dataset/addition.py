from torchvision.datasets import VisionDataset
from PIL import Image
import os
import torch
import time

class rvl_cdip_dataset(VisionDataset):
    def __init__(self, data_path, is_train=True, is_validation=False, transform=None):
        
        # Трансформации изображения надо задавать отдельно.
        # Трансформации результата известны и делаются прямо в классе.
        super().__init__(data_path, transform=transform)
        self.data_path = data_path
        
        if is_train:
            self.lines = open(self.data_path + '/labels/train.txt').readlines()
        elif is_validation:
            self.lines = open(self.data_path + '/labels/val.txt').readlines()
        else:
            self.lines = open(self.data_path + '/labels/test.txt').readlines()
            
        self.len = len(self.lines)
        
    def __getitem__(self, index):
        line = self.lines[index]
        name, target = line.split(' ')
        input_file_name = self.data_path + '/images/' + name
        img = Image.open(input_file_name).convert('RGB') # аналог загрузки через loader
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(int(target)).long()

    def __len__(self):
        return self.len

