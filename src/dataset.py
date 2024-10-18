# -*- coding: utf-8 -*-
import os
import cv2
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms
from utils import *

class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel', args.label_path), sep=' ', header=None)
        
        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = df[df[name_c].str.startswith('Training')]
        else:
            df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/ferEmoLabellist.txt'), sep=' ', header=None)
            dataset = df[df[name_c].str.startswith('Test')]
            
        self.label = dataset.iloc[:, label_c].values - 1
        images_names = dataset.iloc[:, name_c].values
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'ferEmoLabellist.txt')
        
        for f in images_names:
            f = f.split(".")[0]
            f += '.jpg'
            file_name = os.path.join(self.raf_path, 'images', f)
            self.file_paths.append(file_name)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
            
        image = image[:, :, ::-1]
        
        
        if not self.clean:    
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, idx, image1