import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import random
import time
import cv2
from tqdm import tqdm
from PIL import Image
import torchvision
from VKD_model import VKD
from prettytable import PrettyTable
from torchvision import transforms


from Utils.dataset_loading import get_data_loaders
from Utils.load_data import (
    get_multimodal_data,
    prepare_embeddings,
    getTokenEmbed,
    getTargetWeights,
)
from Utils.trainer import pytorch_model_run, predict_classification


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(model)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def get_data_paths(c):

    if c == 0:
        # full ehr
        TXT = "Path_text"
        IMG = "Path_compr"
        NF = "No Finding"
        LA = "Lung Opacity"
        TRAIN = "../train_mimic_cxr.csv"
        TEST = "../test_mimic_cxr.csv"
        VAL = "../val_mimic_cxr.csv"
    elif c == 1:
        # full ehr
        TXT = "Report"
        IMG = "Img"
        NF = "No findings"
        LA = "Airspace Opacity"
        TRAIN = "../train_openi.csv"
        TEST = "../test_openi.csv"
        VAL = "../val_openi.csv"

    elif c == 2:

        TXT = "Img"
        IMG = "Img"
        NF = "No Finding"
        LA = "Lung Opacity"
        TRAIN = "../train_xray14.csv"
        TEST = "../test_xray14.csv"
        VAL = "../val_xray14.csv"

    return TXT, IMG, NF, LA, TRAIN, TEST, VAL


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 512
    bert_embed_size = 768
    n_feature_maps = 1024
    feature_map_size = 49
    maxlen = 256
    batch_size = 30
    n_epochs = 50
    SEED = 10
    debug = 0
    embed_size = 200
    c = 0
    TXT, IMG, NF, LA, TRAIN, TEST, VAL = get_data_paths(c)
    print("Loading data...")
    # x1_train, x2_train, y1_train, x1_val, x2_val, y1_val, x1_test, x2_test, y1_test = get_multimodal_data(TRAIN, VAL, TEST, IMG, TXT, maxlen)
    # class_weights = getTargetWeights(y1_train)
    if c == 0:
        pre_load_model_name = ""
        model_name = ""

    elif c == 1:
        pre_load_model_name = ""
        model_name = ""
    if c == 2:
        pre_load_model_name = ""
        model_name = ""

    if pre_load_model_name == "":
        pre_load_model_name = model_name
    # train_loader, val_loader, test_loader = get_data_loaders(x1_train, x2_train, y1_train, x1_val, x2_val, y1_val, x1_test, x2_test, y1_test, batch_size)
    vkd_model = VKD(
        latent_dim,
        bert_embed_size,
        n_feature_maps,
        feature_map_size,
        maxlen,
        class_weights,
        turn_off_recognition_grad=False,
    )
    count_parameters(vkd_model)

    # vkd_model.load_state_dict(torch.load(pre_load_model_name), strict=False)
    # vkd_model.load_state_dict(torch.load(model_name), strict=False)

    # vkd_model = pytorch_model_run(train_loader, val_loader, vkd_model, model_name, n_epochs = n_epochs, batch_size = batch_size)
    # predict_classification(vkd_model, test_loader, cap, device, batch_size = batch_size , embed_size = 1024)
