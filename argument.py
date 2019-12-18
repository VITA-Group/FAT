from __future__ import print_function, division
import os, argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--dataset', default='MSMT17', type=str, metavar='DATASET',
                        help='training dataset')
    parser.add_argument('--model', default='resnet', type=str, metavar='MODEL',
                        help='training model')
    parser.add_argument('--loss', default='triCtrd', type=str, metavar='LOSS',
                        help='training loss')
    parser.add_argument('--resume-checkpoint', default='', type=str,
                        help='resume timestamp')
    parser.add_argument('--batch-size', default=64, type=int, metavar='BATCHSIZE',
                        help='training batch size')
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='training optimizer')
    parser.add_argument('--scheduler', default='step', type=str,
                        help='training scheduler')
    parser.add_argument('--lr-base', default=0.01, type=float,
                        help='training learning rate for base')
    parser.add_argument('--lr-class', default=0.1, type=float,
                        help='training learning rate for classifier')
    parser.add_argument('--num-epochs', default=60, type=int,
                        help='training epoches')
    parser.add_argument('--step-size', default=40, type=int,
                        help='training step size')
    parser.add_argument('--milestones', default='30,45,55', type=str,
                        help='training milestones')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='training gamma for optimizer')

    parser.add_argument('--log-dir', default='./prid_log', type=str)
    parser.add_argument('--tfboard-dir', default='./prid_tfboard', type=str)

    parser.add_argument('--rerank', action='store_true',
                        help='enable self-supervision training')
    parser.set_defaults(rerank=False)

    parser.add_argument('--neg-set', default='batchNeg', type=str)
    parser.add_argument('--loss-fat', default='XE-tri-mg', type=str)

    args = parser.parse_args()
    args.milestones = [int(x) for x in args.milestones.split(',')]

    if args.resume_checkpoint:
        resume_checkpoint = args.resume_checkpoint.split('/')[-1].split('_')
        resume_path = os.path.join(args.log_dir, resume_checkpoint[0], args.resume_checkpoint)
        resume_epoch = max([int(m.split('_')[-1].split('.')[0]) for m in os.listdir(resume_path) if m.endswith(".pth")])
        args.resume_path = os.path.join(resume_path, 'saved_model_%s.pth' % resume_epoch)

    return args


if __name__ == '__main__':
    print(parse_args())
