from torchvision.datasets import VisionDataset
from PIL import Image
import os
import torch
import time

class rvl_cdip_train_dataset(VisionDataset):
    def __init__(self, data_path, transform, target_transform=None):
        
        # Трансформации изображения надо задавать отдельно.
        # Трансформации результата известны и делаются прямо в классе.
        super().__init__(data_path, transform=transform, 
                         target_transform=target_transform)
        self.data_path = data_path
        self.lines = open(self.data_path + '/labels/train.txt').readlines()
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


class rvl_cdip_val_dataset(VisionDataset):
    def __init__(self, data_path, transform, target_transform=None):
        
        # Трансформации изображения надо задавать отдельно.
        # Трансформации результата известны и делаются прямо в классе.
        super().__init__(data_path, transform=transform, 
                         target_transform=target_transform)
        self.data_path = data_path
        self.lines = open(data_path + '/labels/val.txt').readlines()
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

