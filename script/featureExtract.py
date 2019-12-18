from __future__ import print_function, division

import scipy.io
from tqdm import tqdm

from training.functions import *
from datasetUtils.testDataloaders import *

from argument import *

args = parse_args()
use_gpu = True

args.log_dir = '/'.join(args.resume_path.split('/')[:-1])


######################################################################
# Extract feature
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract(model, dataloader):
    test_features = torch.FloatTensor()
    test_names, test_labels, test_camIds, test_timestamps = [], [], [], []

    for batch, sample in enumerate(tqdm(dataloader)):
        images = sample['img']
        names, labels, camIds, timestamps = sample['name'], sample['label'], sample['camId'], sample['ts']

        test_names = test_names + names
        test_labels = test_labels + list(labels)
        test_camIds = test_camIds + list(camIds)
        test_timestamps = test_timestamps + timestamps

        n, c, h, w = images.size()

        if args.model in ["resnet", "resNet101", "resNet152"]:
            ff = torch.FloatTensor(n, 2048).zero_()
        elif args.model in ["densenet"]:
            ff = torch.FloatTensor(n, 1024).zero_()
        elif args.model == "denseNet169":
            ff = torch.FloatTensor(n, 1664).zero_()
        elif args.model == "denseNet201":
            ff = torch.FloatTensor(n, 1920).zero_()
        elif args.model in ["denseNet161"]:
            ff = torch.FloatTensor(n, 2208).zero_()
        elif args.model in ['multibranch1']:
            ff = torch.FloatTensor(n, 3072).zero_()
        elif args.model in ['multibranch2']:
            ff = torch.FloatTensor(n, 3072).zero_()
        else:
            raise Exception('unknown model')

        ff = ff + model(Variable(images.cuda(), volatile=True))[0].data.cpu()
        ff = ff + model(Variable(fliplr(images).cuda(), volatile=True))[0].data.cpu()

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        test_features = torch.cat((test_features, ff), 0)

    return test_names, test_labels, test_camIds, test_timestamps, test_features


######################################################################
# load model
model = getModel(args.dataset, args.model, args.loss, use_gpu)

assert args.resume_checkpoint
args.resume_checkpoint = args.resume_checkpoint.split('/')[-1].split('_')
assert args.resume_checkpoint[1] == args.model

pretrained = getModel(args.resume_checkpoint[0], args.resume_checkpoint[1], args.resume_checkpoint[2], use_gpu)
pretrained = load_model(pretrained, args.resume_path)

model.model = pretrained.model
if args.resume_checkpoint[2] == args.loss:
    model.classifier = pretrained.classifier
del pretrained

model = nn.DataParallel(model).cuda()

######################################################################
# Extract feature and info and save to Matlab for check
model = model.eval()

extractSubsets = ['query', 'gallery']
test_dataloaders = getDataloader(args.dataset, args.batch_size)

for subset in extractSubsets:
    details = extract(model, test_dataloaders[subset])
    test_names, test_labels, test_camIds, test_timestamps, test_features = details

    results = {'names': test_names, 'labels': test_labels, 'camIds': test_camIds, 'timestamps': test_timestamps,
               'features': test_features.numpy()}

    scipy.io.savemat(clean_file(os.path.join(args.log_dir, 'feature_%s.mat' % (subset))), results)
    print('save file as feature_%s.mat' % (subset))
