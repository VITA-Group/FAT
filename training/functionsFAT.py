from __future__ import print_function, division

from tqdm import tqdm
import torch.nn.functional as F

from training.functions import *


# Get centroids
# ---------------------------
def getCentroids(model, train_dataloaders):
    # pre-compute centroid
    model.eval()
    centroidsOri = {}
    centroidsNorm = {}
    centroidsCount = {}

    for batch, sample in enumerate(tqdm(train_dataloaders)):
        images = Variable(sample['img'], volatile=True).cuda()
        labels = Variable(sample['label']).data.cuda()

        features = model(images)[0].data
        featuresNorm = features.div(torch.norm(features, p=2, dim=1, keepdim=True).expand_as(features))

        for n, k in enumerate(labels):
            if k in centroidsCount:
                centroidsOri[k] = centroidsOri[k] + features[n]
                centroidsNorm[k] = centroidsNorm[k] + featuresNorm[n]
                centroidsCount[k] = centroidsCount[k] + 1
            else:
                centroidsOri[k] = features[n]
                centroidsNorm[k] = featuresNorm[n]
                centroidsCount[k] = 1

    # mean of original feature
    centroidsOri = {k: centroidsOri[k] / centroidsCount[k] for k in centroidsOri}

    # mean of normalized feature
    centroidsNorm = {k: centroidsNorm[k] / centroidsCount[k] for k in centroidsNorm}

    # normalized mean of original feature
    # centroidsOriNorm = {k: centroidsOri[k].div(torch.norm(centroidsOri[k])) for k in centroidsOri}

    # normalized mean of normalized feature
    centroidsNormNorm = {k: centroidsNorm[k].div(torch.norm(centroidsNorm[k])) for k in centroidsNorm}

    print("\npre-compute centroid done!\n")
    return centroidsOri, centroidsNormNorm


# Get gallery set
# ---------------------------
def getGallerySet(centroidsNorm, centroids):
    assert len(centroidsNorm) == len(centroids)
    N = len(centroids)

    gVectorNorm = torch.t(torch.stack([centroidsNorm[k] for k in range(N)]))
    gVector = torch.t(torch.stack([centroids[k] for k in range(N)]))

    return gVectorNorm, gVector


# Get hard mining centroids
# ---------------------------
def getCtrdHM(centroidsNorm, centroids, gVectorNorm, gVector):
    assert len(centroidsNorm) == len(centroids)
    N = len(centroids)

    ctrdDistanceNorm = [F.pairwise_distance(centroidsNorm[k].expand(N, -1),
                                            torch.transpose(gVectorNorm, 0, 1)) for k in range(N)]
    ctrdDistance = [F.pairwise_distance(centroids[k].expand(N, -1),
                                        torch.transpose(gVector, 0, 1)) for k in range(N)]

    sortedKeyCtrdDNorm = {k: torch.sort(ctrdDistanceNorm[k].cpu(), dim=0)[1] for k in range(N)}
    sortedKeyCtrdD = {k: torch.sort(ctrdDistance[k].cpu(), dim=0)[1] for k in range(N)}

    if any([sortedKeyCtrdDNorm[k][0][0] != k for k in range(N)]):
        raise Exception("Centroid Distance Error")

    return sortedKeyCtrdDNorm, sortedKeyCtrdD


# Get average negative centroids
# ---------------------------
def getCtrdAvgNeg(centroidsNorm, centroids):
    assert len(centroidsNorm) == len(centroids)
    N = len(centroids)

    allVectorNorm = torch.cuda.FloatTensor(centroidsNorm[0].size()[0]).zero_()
    allVector = torch.cuda.FloatTensor(centroidsNorm[0].size()[0]).zero_()
    for k in range(N):
        allVectorNorm += centroidsNorm[k]
        allVector += centroids[k]

    negAvgNorm = {k: (allVectorNorm - centroidsNorm[k]) / (N - 1) for k in range(N)}
    negAvg = {k: (allVector - centroids[k]) / (N - 1) for k in range(N)}

    return negAvgNorm, negAvg


# Get positive pairs
# ---------------------------
def getPosSet(centroidsNorm, centroids, labels):
    posNorm = Variable(torch.stack([centroidsNorm[k] for k in labels]), requires_grad=False)

    pos = Variable(torch.stack([centroids[k] for k in labels]), requires_grad=False)

    return posNorm, pos


# Get negative pairs
# ---------------------------
def getNegSetAll(centroidsNorm, centroids, size):
    negFtNorm = [Variable(centroidsNorm[k].expand(size, -1), requires_grad=False) for k in centroidsNorm]

    negFt = [Variable(centroids[k].expand(size, -1), requires_grad=False) for k in centroids]

    return negFtNorm, negFt


def getNegSetAvg(negAvgNorm, negAvg, labels):
    negFtNorm = [Variable(torch.stack([negAvgNorm[k] for k in labels]), requires_grad=False)]

    negFt = [Variable(torch.stack([negAvg[k] for k in labels]), requires_grad=False)]

    return negFtNorm, negFt


def getNegSetHM(centroidsNorm, centroids, sortedKeyCtrdDNorm, sortedKeyCtrdD, labels):
    minLen = 5

    # first element in sortedKey is label it self
    negFtNorm = [Variable(torch.stack([centroidsNorm[sortedKeyCtrdDNorm[n][k + 1][0]] for n in labels]),
                          requires_grad=False) for k in range(minLen)]

    negFt = [Variable(torch.stack([centroids[sortedKeyCtrdD[n][k + 1][0]] for n in labels]),
                      requires_grad=False) for k in range(minLen)]

    return negFtNorm, negFt


def getNegSetBatch(anchorNorm, anchor, labels):
    minLen = 5

    if len(anchorNorm) == len(anchor):
        N = len(anchor)

    batchDistanceNorm = [F.pairwise_distance(anchorNorm[k].expand(N, -1), anchorNorm) for k in range(N)]
    batchDistance = [F.pairwise_distance(anchor[k].expand(N, -1), anchor) for k in range(N)]

    sortedKeyCtrdDNorm = {k: torch.sort(batchDistanceNorm[k].cpu(), dim=0)[1] for k in range(N)}
    sortedKeyCtrdD = {k: torch.sort(batchDistance[k].cpu(), dim=0)[1] for k in range(N)}

    # remove hard sample with same label in sortedKey
    sortedKeyCtrdDNorm = {k: [int(n) for n in sortedKeyCtrdDNorm[k] if labels[int(n)] != labels[k]] for k in range(N)}
    sortedKeyCtrdD = {k: [int(n) for n in sortedKeyCtrdD[k] if labels[int(n)] != labels[k]] for k in range(N)}

    minLen = min(min([len(sortedKeyCtrdDNorm[k]) for k in sortedKeyCtrdDNorm]),
                 min([len(sortedKeyCtrdD[k]) for k in sortedKeyCtrdD]), minLen)

    negFtNorm = [Variable(torch.stack([anchorNorm[sortedKeyCtrdDNorm[n][k]].data for n in range(N)]),
                          requires_grad=False) for k in range(minLen)]

    negFt = [Variable(torch.stack([anchor[sortedKeyCtrdD[n][k]].data for n in range(N)]),
                      requires_grad=False) for k in range(minLen)]

    return negFtNorm, negFt


# Get loss
# ---------------------------
def getLoss(lossf, lossRegWeight, outputs, labels, anchorNorm, posNorm, negFtNorm, anchor, pos, negFt, use_margin):
    if 'XE' in lossf:
        xe = lossRegWeight["XE"] * F.cross_entropy(outputs, labels)

    if 'triNorm' in lossf or 'doubleTri' in lossf:
        triMarginNorm = 0
        for negNorm in negFtNorm:
            triMarginNorm += F.triplet_margin_loss(anchorNorm, posNorm, negNorm, margin=use_margin['TriCtrdNorm'])
        triMarginNorm /= len(negFtNorm)
        triMarginNorm *= lossRegWeight["TriCtrdNorm"]

    if ((not 'triNorm' in lossf) and 'tri' in lossf) or 'doubleTri' in lossf:
        triMargin = 0
        for neg in negFt:
            triMargin += F.triplet_margin_loss(anchor, pos, neg, margin=use_margin['TriCtrd'])
        triMargin /= len(negFt)
        triMargin *= lossRegWeight["TriCtrd"]

    if 'mgNorm' in lossf or 'doubleMG' in lossf:
        ctrdMarginNorm = lossRegWeight["scaleTriCtrdNorm"] * torch.mean(F.pairwise_distance(posNorm, anchorNorm))

    if ((not 'mgNorm' in lossf) and 'mg' in lossf) or 'doubleMG' in lossf:
        ctrdMargin = lossRegWeight["scaleTriCtrd"] * torch.mean(F.pairwise_distance(pos, anchor))

    if lossf == "XE":
        loss = xe
    elif lossf == "XE-mgNorm":
        loss = xe + ctrdMarginNorm
    elif lossf == "XE-mg":
        loss = xe + ctrdMargin
    elif lossf == "XE-doubleMG":
        loss = xe + ctrdMarginNorm + ctrdMargin

    elif lossf == "triNorm":
        loss = triMarginNorm
    elif lossf == "triNorm-mgNorm":
        loss = triMarginNorm + ctrdMarginNorm
    elif lossf == "triNorm-doubleMG":
        loss = triMarginNorm + ctrdMarginNorm + ctrdMargin

    elif lossf == "tri":
        loss = triMargin
    elif lossf == "tri-mg":
        loss = triMargin + ctrdMargin
    elif lossf == "tri-doubleMG":
        loss = triMargin + ctrdMarginNorm + ctrdMargin

    elif lossf == "doubleTri":
        loss = triMarginNorm + triMargin
    elif lossf == "doubleTri-mgNorm":
        loss = triMarginNorm + triMargin + ctrdMarginNorm
    elif lossf == "doubleTri-mg":
        loss = triMarginNorm + triMargin + ctrdMargin
    elif lossf == "doubleTri-doubleMG":
        loss = triMarginNorm + triMargin + ctrdMarginNorm + ctrdMargin

    elif lossf == "XE-triNorm":
        loss = xe + triMarginNorm
    elif lossf == "XE-triNorm-mgNorm":
        loss = xe + triMarginNorm + ctrdMarginNorm
    elif lossf == "XE-triNorm-doubleMG":
        loss = xe + triMarginNorm + ctrdMarginNorm + ctrdMargin

    elif lossf == "XE-tri":
        loss = xe + triMargin
    elif lossf == "XE-tri-mg":
        loss = xe + triMargin + ctrdMargin
    elif lossf == "XE-tri-doubleMG":
        loss = xe + triMargin + ctrdMarginNorm + ctrdMargin

    elif lossf == "XE-doubleTri":
        loss = xe + triMarginNorm + triMargin
    elif lossf == "doubleTri-mgNorm":
        loss = xe + triMarginNorm + triMargin + ctrdMarginNorm
    elif lossf == "doubleTri-mg":
        loss = xe + triMarginNorm + triMargin + ctrdMargin
    elif lossf == "XE-doubleTri-doubleMG":
        loss = xe + triMarginNorm + triMargin + ctrdMarginNorm + ctrdMargin

    else:
        raise Exception("unknown loss function")

    return loss
