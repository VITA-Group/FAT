import pandas as pd

from pridUtils.utils import *

prid_data_root_dir = '/home/yeyuan/prid_data'
dataset_info = {
    "MSMT17": {
        "train": {
            "dir": os.path.join(prid_data_root_dir, "MSMT17_V1", "train"),
            "csv": os.path.join(prid_data_root_dir, "MSMT17_V1", "list_train.txt"),
            "info": os.path.join(prid_data_root_dir, "MSMT17_V1", "info", "train_info.txt")},

        "val": {
            "dir": os.path.join(prid_data_root_dir, "MSMT17_V1", "train"),
            "csv": os.path.join(prid_data_root_dir, "MSMT17_V1", "list_val.txt"),
            "info": os.path.join(prid_data_root_dir, "MSMT17_V1", "info", "val_info.txt")},

        "query": {
            "dir": os.path.join(prid_data_root_dir, "MSMT17_V1", "test"),
            "csv": os.path.join(prid_data_root_dir, "MSMT17_V1", "list_query.txt"),
            "info": os.path.join(prid_data_root_dir, "MSMT17_V1", "info", "query_info.txt")},

        "gallery": {
            "dir": os.path.join(prid_data_root_dir, "MSMT17_V1", "test"),
            "csv": os.path.join(prid_data_root_dir, "MSMT17_V1", "list_gallery.txt"),
            "info": os.path.join(prid_data_root_dir, "MSMT17_V1", "info", "gallery_info.txt")},
    },

    "DukeMTMC": {
        "train": {
            "dir": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "bounding_box_train"),
            "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "train_info.txt")},
        # "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "train_Distractors_All_info.txt")},

        "val": {
            "dir": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "bounding_box_train"),
            "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "val_info.txt")},

        "query": {
            "dir": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "query"),
            "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "query_info.txt")},

        "gallery": {
            "dir": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "bounding_box_test"),
            "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "gallery_info.txt")},
        # "info": os.path.join(prid_data_root_dir, "DukeMTMC-reID", "info", "gallery_Distractors_info.txt")},
    },

    "Market1501": {
        "train": {
            "dir": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "bounding_box_train"),
            "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "train_info.txt")},
        # "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "train_Distractors_All_info.txt")},

        "val": {
            "dir": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "bounding_box_train"),
            "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "val_info.txt")},

        "query": {
            "dir": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "query"),
            "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "query_info.txt")},

        "gallery": {
            "dir": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "bounding_box_test"),
            "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "gallery_info.txt")},
        # "info": os.path.join(prid_data_root_dir, "Market-1501-v15.09.15", "info", "gallery_Distractors_info.txt")},
    },

}


def savecsv(info, filepath):
    df = pd.DataFrame(info)
    df.to_csv(clean_file(filepath), sep=' ', index=None, header=['img', 'label', 'camId', 'ts'])
    print("save file", filepath)


def MSMT17_info(MSMT17_datasetInfo):
    os.makedirs("/home/yeyuan/prid_data/MSMT17_V1/info/", exist_ok=True)

    for subset in ['val', 'train', 'query', 'gallery']:
        df = pd.read_csv(MSMT17_datasetInfo[subset]['csv'], header=None, delimiter=' ', names=['img', 'label'])

        info = []
        for index, row in df.iterrows():
            if int(row['img'].split("/")[0]) != int(row['label']):
                raise Exception("invalid label", str(row))
            if not row['img'].split("/")[1].split("_")[3][4:] in ['morning', 'noon', 'afternoon', 'None']:
                raise Exception("invalid timestamp", str(row))

            info.append([row['img'].rstrip(), int(row['label']), int(row['img'].split("/")[1].split("_")[2]),
                         row['img'].split("/")[1].split("_")[3][4:].rstrip()])

        labels = [s[1] for s in info]
        if min(labels) != 0 or max(labels) != len(set(labels)) - 1:
            raise Exception("invalid label", min(labels), max(labels), len(labels))

        savecsv(info, MSMT17_datasetInfo[subset]['info'])


def relabel_dict(identities):
    identities = list(set(identities))
    identities.sort()

    if -1 in identities:
        print("-1 label existed")
        identities.remove(-1)

    label_map = {identities[n]: n for n in range(len(identities))}
    label_map[-1] = -1
    return label_map


def info4train(datasetInfo):
    all_train_identities = set()
    train_imgs, val_imgs, identities = [], [], []

    for img in os.listdir(datasetInfo['train']['dir']):
        if img.endswith("jpg") or img.endswith("png"):
            identities.append(int(img.split("_")[0]))

            if img.split("_")[0] in all_train_identities:
                train_imgs.append(img)
            else:
                all_train_identities.add(img.split("_")[0])
                val_imgs.append(img)

    label_map = relabel_dict(identities)

    train_info = [[img, label_map[int(img.split("_")[0])], int(img.split("_")[1][1]), 'None'] for img in train_imgs]
    savecsv(train_info, datasetInfo['train']['info'])

    val_info = [[img, label_map[int(img.split("_")[0])], int(img.split("_")[1][1]), 'None'] for img in val_imgs]
    savecsv(val_info, datasetInfo['val']['info'])


def info4train_noVal(datasetInfo):
    train_imgs, identities = [], []

    for img in os.listdir(datasetInfo['train']['dir']):
        if img.endswith("jpg") or img.endswith("png"):
            train_imgs.append(img)
            identities.append(int(img.split("_")[0]))

    label_map = relabel_dict(identities)

    train_info = [[img, label_map[int(img.split("_")[0])], int(img.split("_")[1][1]), 'None'] for img in train_imgs]
    savecsv(train_info, datasetInfo['train']['info'].replace('train', 'trainAll'))


def info4test(datasetInfo):
    query_imgs = [img for img in os.listdir(datasetInfo['query']['dir']) if
                  img.endswith("jpg") or img.endswith("png")]
    gallery_imgs = [img for img in os.listdir(datasetInfo['gallery']['dir']) if
                    img.endswith("jpg") or img.endswith("png")]

    identities = [int(img.split("_")[0]) for img in query_imgs + gallery_imgs]
    label_map = relabel_dict(identities)

    query_info = [[img, label_map[int(img.split("_")[0])], int(img.split("_")[1][1]), 'None'] for img in query_imgs]
    savecsv(query_info, datasetInfo['query']['info'])

    gallery_info = [[img, label_map[int(img.split("_")[0])], int(img.split("_")[1][1]), 'None'] for img in gallery_imgs]
    savecsv(gallery_info, datasetInfo['gallery']['info'])


def DukeMTMC_info(DukeMTMC_datasetInfo):
    os.makedirs("/home/yeyuan/prid_data/DukeMTMC-reID/info/", exist_ok=True)
    info4train(DukeMTMC_datasetInfo)
    info4test(DukeMTMC_datasetInfo)


def Market1501_info(Market1501_datasetInfo):
    os.makedirs("/home/yeyuan/prid_data/Market-1501-v15.09.15/info/", exist_ok=True)
    info4train(Market1501_datasetInfo)
    info4test(Market1501_datasetInfo)


if __name__ == '__main__':
    Market1501_info(dataset_info["Market1501"])
    DukeMTMC_info(dataset_info["DukeMTMC"])
    MSMT17_info(dataset_info["MSMT17"])

    # info4train_noVal(dataset_info["Market1501"])
    # info4train_noVal(dataset_info["DukeMTMC"])
