# In Defense of the Triplet Loss Again: Learning Robust Person Re-Identification with Fast Approximated Triplet Loss and Label Distillation

<a href="https://arxiv.org/pdf/1912.07863v1.pdf">In Defense of the Triplet Loss Again: Learning Robust Person Re-Identification with Fast Approximated Triplet Loss and Label Distillation</a>

Ye Yuan, Wuyang Chen, Yang Yang, Zhangyang Wang

## Overview

The comparative losses (typically, triplet loss) are appealing choices for learning person re-identification (ReID)
features. However, the triplet loss is computationally much more expensive than the (practically more popular) classification loss, limiting their wider usage in massive datasets.
Moreover, the abundance of label noise and outliers in ReID datasets may also put the margin-based loss in jeopardy. This work addresses the above two shortcomings of triplet loss, extending its effectiveness to large-scale ReID datasets with potentially noisy labels. 

We propose a fastapproximated triplet (FAT) loss, which provably converts the point-wise triplet loss into its upper bound form, consisting of a point-to-set loss term plus cluster compactness regularization. It preserves the effectiveness of triplet loss, while leading to linear complexity to the training set size.

A label distillation strategy is further designed to learn refined soft-labels in place of the potentially noisy labels, from
only an identified subset of confident examples, through teacher-student networks. We conduct extensive experiments on three most popular ReID benchmarks (Market1501, DukeMTMC-reID, and MSMT17), and demonstrate that FAT loss with distilled labels lead to ReID features with remarkable accuracy, efficiency, robustness, and direct transferability to unseen datasets.

## Training

Please sequentially finish the following steps:
1. `python script/train.py --dataset MSMT17 --loss triCtrd`

## Evaluation

Run script
1. `python script/featureExtract.py`
1. `python script/evaluate.py`

## Citation

If you use this code for your research, please cite our paper.
```
@misc{yuan2019fat,
    title={In Defense of the Triplet Loss Again: Learning Robust Person Re-Identification with Fast Approximated Triplet Loss and Label Distillation},
    author={Ye Yuan and Wuyang Chen and Yang Yang and Zhangyang Wang},
    year={2019},
    eprint={1912.07863},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
