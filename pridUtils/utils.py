import os
import time
import torch
import shutil
import GPUtil as GPU


def clean_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    return dir


def clean_file(fname):
    if os.path.isfile(fname):
        os.remove(fname)
    return fname


# Load model
# ---------------------------
def load_model(model, resumepath):
    print("load model from %s\n" % resumepath)
    model.load_state_dict(torch.load(resumepath))
    return model


# Save model
# ---------------------------
def save_model(model, savepath):
    print("model saved in %s\n" % savepath)
    torch.save(model.cpu().state_dict(), savepath)
    model = model.cuda()
    return model


# Save model trained with multiple GPU
# ---------------------------
def save_parallel_model(model, savepath):
    print("model saved in %s\n" % savepath)
    torch.save(model.module.cpu().state_dict(), savepath)
    model = model.cuda()
    return model
