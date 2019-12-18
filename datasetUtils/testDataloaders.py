from __future__ import print_function, division

from datasetUtils.dataset import *
from datasetUtils.infoGenerator import *


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


def getDataloader(use_dataset, batch_size):
    transform = [transforms.Resize(size=(384, 128), interpolation=3),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    test_datasets = {x: PRIDdataset(datasetInfo=dataset_info[use_dataset], subset=x,
                                    transform=transforms.Compose(transform)) for x in ['query', 'gallery']}

    test_dataloaders = {x: torch.utils.data.DataLoader(test_datasets[x], batch_size=batch_size,
                                                       shuffle=False, num_workers=8) for x in ['query', 'gallery']}

    return test_dataloaders


print("\nDataloader ... OK!\n")
