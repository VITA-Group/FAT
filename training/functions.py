from __future__ import print_function, division

from models.models import *
from datasetUtils.datasetStat import Nclass, NcamId


# Define model
# ---------------------------
def getModel(use_dataset, use_model, use_loss, use_gpu):
    numClass = Nclass[use_dataset]['train']

    if use_model == "midnet":
        model = ft_resnet50mid(numClass)
    elif use_model == "resnet":
        model = ft_resnet50(numClass)
    elif use_model == "resNet101":
        model = ft_resnet101(numClass)
    elif use_model == "resNet152":
        model = ft_resnet152(numClass)
    elif use_model == "densenet":
        model = ft_densenet121(numClass)
    elif use_model == "denseNet169":
        model = ft_densenet169(numClass)
    elif use_model == "denseNet201":
        model = ft_densenet201(numClass)
    elif use_model == "denseNet161":
        model = ft_densenet161(numClass)
    else:
        raise Exception('unknown model')

    return model.cuda() if use_gpu else model
