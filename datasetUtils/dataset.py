from __future__ import print_function
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PRIDdataset(Dataset):
    def __init__(self, datasetInfo, subset, transform=None):
        self.dir = datasetInfo[subset]['dir']
        self.info = pd.read_csv(datasetInfo[subset]['info'], header=0, delimiter=' ')
        self.transform = transform

        print("load image from %s\n" % self.dir)
        print("load label from %s\n" % datasetInfo[subset]['info'])

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        name = self.info.iloc[idx]['img'].rstrip()
        label = int(self.info.iloc[idx]['label'])
        camId = int(self.info.iloc[idx]['camId'])
        ts = self.info.iloc[idx]['ts'].rstrip()

        # return image as numpy.ndarray
        # image = io.imread(os.path.join(self.dir, name))

        # return image as PIL.JpegImagePlugin.JpegImageFile
        if os.path.isfile(os.path.join(self.dir, name)):
            img = Image.open(os.path.join(self.dir, name))
        else:
            img = Image.open(os.path.join(self.dir, name.replace("jpg", "png")))

        if self.transform:
            # https://pytorch.org/docs/master/torchvision/transforms.html
            # transforms.ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
            # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
            img = self.transform(img)

        return {'name': name, 'img': img, 'label': label, 'camId': camId, 'ts': ts}


# refer to https://jdhao.github.io/2017/10/23/pytorch-load-data-and-make-batch/
def dataset_collate(batch):
    name = torch.LongTensor([item['name'] for item in batch])
    img = np.array([item['img'] for item in batch])
    label = torch.LongTensor([item['label'] for item in batch])

    return {'name': name, 'img': img, 'label': label}
