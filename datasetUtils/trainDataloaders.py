from __future__ import print_function, division

from datasetUtils.dataset import *
from datasetUtils.infoGenerator import *
from pridUtils.random_erasing import RandomErasing


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        if h == self.height and w == self.width:
            return img
        return img.resize((self.width, self.height), self.interpolation)


def getTransforms(use_erasing=False, use_colorjitter=False):
    data_transform = {'train': [transforms.Resize((432, 144), interpolation=3),
                                transforms.RandomCrop((384, 128))],
                      'val': [transforms.Resize(size=(384, 128), interpolation=3)]}

    data_transform['train'] = data_transform['train'] + [transforms.RandomHorizontalFlip()]

    for k in ['train', 'val']:
        data_transform[k] = data_transform[k] + [
            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    if use_erasing:
        data_transform['train'] = data_transform['train'] + [
            RandomErasing(probability=[0.3, 0.3, 0.3], mean=[0.0, 0.0, 0.0])]

    # Randomly change the brightness, contrast and saturation of an image.
    if use_colorjitter:
        colorjitter = [0.3, 0.3, 0.3, 0]
        data_transform['train'] = [transforms.ColorJitter(
            brightness=colorjitter[0], contrast=colorjitter[1],
            saturation=colorjitter[2], hue=colorjitter[3])] + data_transform['train']

    return data_transform


def getDataloader(use_dataset, batch_size):
    data_transforms = getTransforms()

    train_datasets = {x: PRIDdataset(datasetInfo=dataset_info[use_dataset], subset=x,
                                     transform=transforms.Compose(data_transforms[x])) for x in ['train', 'val']}
    train_datasets['ctrd'] = PRIDdataset(datasetInfo=dataset_info[use_dataset], subset='train',
                                         transform=transforms.Compose(data_transforms['val']))
    train_dataloaders = {x: torch.utils.data.DataLoader(train_datasets[x], batch_size=batch_size, shuffle=True,
                                                        num_workers=8) for x in train_datasets}

    return train_dataloaders


print("\nDataloader ... OK!\n")
