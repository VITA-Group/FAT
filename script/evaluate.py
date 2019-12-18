import torch
import numpy as np
import scipy.io
from tqdm import tqdm

from pridUtils.re_ranking import re_ranking
from argument import *

args = parse_args()
use_gpu = True

args.log_dir = '/'.join(args.resume_path.split('/')[:-1])


######################################################################
def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size == 0:  # if empty
        cmc[0] = -1
        return ap, cmc, None

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2

    return ap, cmc[:20], index[:20]


def evaluate(score, ql, qc, gl, gc):
    # predict index sort from small to large
    index = np.argsort(score)
    if not args.rerank:
        index = index[::-1]

    # good index
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)

    if args.dataset in ["MSMT17", "DukeMTMC", "Market1501"]:
        junk_index1 = np.argwhere(gl == -1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1)  # .flatten())

    return compute_mAP(index, good_index, junk_index)


######################################################################


def get_features(matfiles):
    def get_details(matfile):
        return matfile['names'], matfile['labels'][0], matfile['camIds'][0], matfile['timestamps'], matfile['features']

    query_names, query_labels, query_camIds, query_timestamps, query_features = get_details(matfiles['query'])
    print("query_indices processing ... done!")
    print("use_query_images", len(query_labels))

    gallery_names, gallery_labels, gallery_camIds, gallery_timestamps, gallery_features = get_details(
        matfiles['gallery'])
    print("gallery_indices processing ... done!")
    print("use_gallery_images", len(gallery_labels))

    query_features = torch.FloatTensor(query_features).cuda() if use_gpu else torch.FloatTensor(query_features)
    gallery_features = torch.FloatTensor(gallery_features).cuda() if use_gpu else torch.FloatTensor(gallery_features)

    query_infos = query_features, query_labels, query_camIds, query_names
    gallery_infos = gallery_features, gallery_labels, gallery_camIds, gallery_names
    return query_infos, gallery_infos


######################################################################
def get_scores(query_infos, gallery_infos):
    query_features, query_labels, query_camIds, query_names = query_infos
    gallery_features, gallery_labels, gallery_camIds, gallery_names = gallery_infos
    if args.rerank:
        print("prepare for rerank...")
        q_g_dist = np.dot(query_features, np.transpose(gallery_features))
        q_q_dist = np.dot(query_features, np.transpose(query_features))
        g_g_dist = np.dot(gallery_features, np.transpose(gallery_features))
        re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist)

    # CMC = torch.IntTensor(len(gallery_labels)).zero_()
    CMC = torch.IntTensor(20).zero_()
    ap = 0.0

    for i in tqdm(range(len(query_labels))):
        if args.rerank:
            score = re_rank[i, :]
        else:
            if use_gpu:
                score = torch.mm(gallery_features, query_features[i].view(-1, 1))
                score = score.squeeze(1).cpu().numpy()
            else:
                score = np.dot(gallery_features, query_features[i])

        ap_tmp, CMC_tmp, index_tmp = evaluate(score, query_labels[i], query_camIds[i], gallery_labels, gallery_camIds)
        if CMC_tmp[0] == -1:
            continue

        open(res_file, 'a').write(
            "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (
                query_names[i].rstrip(), ap_tmp, CMC_tmp[0], CMC_tmp[4], CMC_tmp[9], CMC_tmp[19],
                str([gallery_names[j] for j in index_tmp]).replace(" ", "")))

        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC / len(query_labels)  # average CMC
    print('top1: %.4f top5: %.4f top10: %.4f mAP: %.4f' % (CMC[0], CMC[4], CMC[9], ap / len(query_labels)))

    return CMC[0], CMC[4], CMC[9], ap / len(query_labels)


res_file = os.path.join(args.log_dir, 'res.txt')

matfiles = {subset: scipy.io.loadmat(os.path.join(args.log_dir, 'feature_%s.mat' % subset))
            for subset in ['query', 'gallery']}

query_infos, gallery_infos = get_features(matfiles)
get_scores(query_infos, gallery_infos)
