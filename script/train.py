from __future__ import print_function, division
import os, sys
from torch.optim import lr_scheduler
import torch.optim as optim
from tensorboardX import SummaryWriter

sys.path.append(os.getcwd())

from training.config import *
from training.configFAT import *
from datasetUtils.trainDataloaders import *
from argument import *

args = parse_args()

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
name = '%s_%s_%s_%s' % (args.dataset, args.model, args.loss, timestamp)
print(name + '\n')

os.makedirs(args.log_dir, exist_ok=True)
args.log_dir = os.path.join(args.log_dir, args.dataset, name)
os.makedirs(args.log_dir, exist_ok=True)

os.makedirs(args.tfboard_dir, exist_ok=True)
args.tfboard_dir = os.path.join(args.tfboard_dir, args.dataset, name)
os.makedirs(args.tfboard_dir, exist_ok=True)

# define writer
writer = SummaryWriter(log_dir=args.tfboard_dir)
use_gpu = True

######################################################################
model = getModel(args.dataset, args.model, args.loss, use_gpu)

if args.resume_checkpoint:
    args.resume_checkpoint = args.resume_checkpoint.split('/')[-1].split('_')
    assert args.resume_checkpoint[1] == args.model

    pretrained = getModel(args.resume_checkpoint[0], args.resume_checkpoint[1], args.resume_checkpoint[2], use_gpu)
    pretrained = load_model(pretrained, args.resume_path)

    model.model = pretrained.model
    if args.resume_checkpoint[2] == args.loss:
        model.classifier = pretrained.classifier
    del pretrained

model = nn.DataParallel(model).cuda()

# define optimizer
if args.optimizer == 'sgd':
    optimizer = optim.SGD([
        {'params': model.module.model.parameters(), 'lr': args.lr_base},
        {'params': model.module.classifier.parameters(), 'lr': args.lr_class}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)
elif args.optimizer == 'adam':
    optimizer = optim.Adam([
        {'params': model.module.model.parameters(), 'lr': args.lr_base},
        {'params': model.module.classifier.parameters(), 'lr': args.lr_class}
    ], weight_decay=5e-4, betas=(0.9, 0.999))

# define scheduler
if args.scheduler == 'step':
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
elif args.scheduler == 'multiStep':
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
elif args.scheduler == 'plateau':
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True,
                                               threshold=0.02, threshold_mode='rel', cooldown=5, min_lr=1e-06,
                                               eps=1e-08)
else:
    raise Exception('unknown scheduler')

######################################################################
model.module.model.train() if args.lr_base else model.module.model.eval()
model.module.classifier.train() if args.lr_class else model.module.classifier.eval()

print("model.model.train()" if args.lr_base else "model.model.eval()")
print("model.classifier.train()" if args.lr_class else "model.classifier.eval()")

train_dataloaders = getDataloader(args.dataset, args.batch_size)

if args.loss == "crossEntropy":
    train_model(model, optimizer, scheduler, writer, train_dataloaders, args.num_epochs,
                args.model, args.loss, args.log_dir, args.lr_base, args.lr_class)

elif args.loss == "triCtrd":
    lossRegWeight = {"XE": 1, "TriCtrd": 1, "TriCtrdNorm": 1, "scaleTriCtrd": 0.1, "scaleTriCtrdNorm": 1}
    margin = {"TriCtrd": 1, "TriCtrdNorm": 0.1}
    train_model_fat(model, optimizer, scheduler, writer, train_dataloaders, args.num_epochs,
                    args.neg_set, margin, args.batch_size, args.loss_fat, lossRegWeight,
                    args.model, args.loss, args.log_dir, args.lr_base, args.lr_class)

else:
    raise Exception('Unknown Loss')

writer.close()
