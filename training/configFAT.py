from __future__ import print_function, division
import os
from tqdm import tqdm
from pridUtils.utils import save_parallel_model
from training.functionsFAT import *


def train_epoch_fat(model, optimizer, writer, train_dataloaders, use_loss, phase, epoch,
                    negSet, margin, batch_size, lossf, lossRegW,
                    galleryFtNorm, centroidsNorm, centroids, sortedKeyDNorm, sortedKeyD, negAvgNorm, negAvg):
    running_loss = 0.0
    running_corrects = 0

    for batch, sample in enumerate(tqdm(train_dataloaders[phase])):
        images = Variable(sample['img']).cuda() if phase == 'train' else Variable(sample['img'], volatile=True).cuda()
        labels = Variable(sample['label']).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        anchor, outputs = model(images)
        anchorNorm = anchor.div(torch.norm(anchor, p=2, dim=1, keepdim=True).expand_as(anchor))

        posNorm, pos = getPosSet(centroidsNorm, centroids, labels.data)

        if negSet == "ctrdAll" and (batch == 0 or len(labels) != batch_size):
            negFtNorm, negFt = getNegSetAll(centroidsNorm, centroids, len(labels))
        elif negSet == "ctrdAvg":
            negFtNorm, negFt = getNegSetAvg(negAvgNorm, negAvg, labels.data)
        elif negSet == "ctrdHM":
            negFtNorm, negFt = getNegSetHM(centroidsNorm, centroids, sortedKeyDNorm, sortedKeyD, labels.data)
        elif negSet == "batchNeg":
            negFtNorm, negFt = getNegSetBatch(anchorNorm, anchor, labels.data)

        loss = getLoss(lossf, lossRegW, outputs, labels,
                       anchorNorm, posNorm, negFtNorm, anchor, pos, negFt, margin)
        _, preds = torch.max(torch.mm(anchorNorm.data, galleryFtNorm), 1)
        correct = torch.sum(preds == labels.data)

        # statistics
        running_loss += 1.0 * loss.data[0] * len(labels)
        running_corrects += correct

        # if in training phase backward + optimize
        if phase == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # statistics
    epoch_loss = 1. * running_loss / len(train_dataloaders[phase].dataset)
    epoch_acc = 1. * running_corrects / len(train_dataloaders[phase].dataset)

    # tensorboardX
    writer.add_scalar('epoch_loss/' + phase, epoch_loss, epoch + 1)
    writer.add_scalar('epoch_accuracy/' + phase, epoch_acc, epoch + 1)

    # show epoch log (loss, accuracy ...)
    print('\n{}, Epoch: {}, Loss: {:.6f}, Accuracy: {:.4f}%\n'.format(phase, epoch + 1, epoch_loss, 100. * epoch_acc))

    return epoch_loss, epoch_acc


# Training the model
# ------------------
def train_model_fat(model, optimizer, scheduler, writer, train_dataloaders, num_epochs,
                    negSet, margin, batch_size, lossf, lossRegW,
                    use_model, use_loss, log_dir, lr_base, lr_class):
    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if (epoch == 0 or phase == 'val'):
                centroids, centroidsNorm = getCentroids(model, train_dataloaders['ctrd'])
                galleryFtNorm, galleryFt = getGallerySet(centroidsNorm, centroids)

                sortedKeyDNorm, sortedKeyD, negAvgNorm, negAvg = None, None, None, None

                if negSet == "ctrdHM":
                    sortedKeyDNorm, sortedKeyD = getCtrdHM(centroidsNorm, centroids, galleryFtNorm, galleryFt)

                if negSet == "ctrdAvg":
                    negAvgNorm, negAvg = getCtrdAvgNeg(centroidsNorm, centroids)

            if phase == 'train':
                model.module.model.train() if lr_base else model.module.model.eval()
                model.module.classifier.train() if lr_class else model.module.classifier.eval()
            else:
                model.train(False)

            if phase == 'train':
                scheduler.step()

            train_epoch_fat(model, optimizer, writer, train_dataloaders, use_loss, phase, epoch,
                            negSet, margin, batch_size, lossf, lossRegW,
                            galleryFtNorm, centroidsNorm, centroids, sortedKeyDNorm, sortedKeyD, negAvgNorm, negAvg)

            # if in validation phase save model
            if phase == 'train':
                save_path = os.path.join(log_dir, 'saved_model_%s.pth' % (epoch + 1))
                model = save_parallel_model(model, save_path)
