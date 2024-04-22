





import math#3.14添加
import argparse

import torch.nn as nn
from network import *
from utils.tools import *
import torch.nn.functional as F#3.14添加
import xlrd#3.14添加
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
# from models.vit import ViT
import os
import torch
import torch.optim as optim
import time
import numpy as np
from modeling1 import VisionTransformer, VIT_CONFIGS
import random
torch.multiprocessing.set_sharing_strategy('file_system')
import network
from networkpitlca import *
from torchvision import models

from networkdit import DistilledVisionTransformer

# DPN(IJCAI2020)
# paper [Deep Polarized Network for Supervised Learning of Accurate Binary Hashing Codes](https://www.ijcai.org/Proceedings/2020/115)
# code [DPN](https://github.com/kamwoh/DPN)
# [DPN] epoch:150, bit:48, dataset:imagenet, MAP:0.675, Best MAP: 0.688
# [DPN] epoch:70, bit:48, dataset:cifar10-1, MAP:0.778, Best MAP: 0.787
# [DPN] epoch:10, bit:48, dataset:nuswide_21, MAP:0.818, Best MAP: 0.818
# [DPN-T] epoch:10, bit:48, dataset:cifar10-1, MAP:0.134, Best MAP: 0.134

def get_config():
    config = {
        "m": 1,
        "p": 0.5,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 5e-6, "weight_decay": 1e-5}},
        "info": "[DPN]",
        # "info": "[DPN-A]",
        # "info": "[DPN-T]",
        # "info": "[DPN-A-T]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        # "net": AlexNet,
        # "net": ResNet,
        "net": pit_s, "net_print": "PIT",
        "dataset": "cifar10",
        #"dataset": "imagenet",
        #"dataset": "coco",
        # "dataset": "nuswide_21",
        "epoch": 150,
        "test_map": 10,
        "save_path": "save/DPNLCALO",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [32],
        "loss_rate":0.01
    }
    config = config_dataset(config)
    return config


###
class LO(torch.nn.Module):
    def __init__(self, num_classes, config, scale=1.0):
        super(LO, self).__init__()
        self.device = config["device"]
        self.num_classes = num_classes
        # self.scale = scale

    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = torch.nn.functional.one_hot(labels.to(torch.int64), self.num_classes).float().to(self.device)
        # print(label_one_hot.shape)
        # print(pred.shape)
        # exit()
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return  nce.mean()
###



class DPNLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(DPNLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.target_vectors = self.get_target_vectors(config["n_class"], bit, config["p"]).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.m = config["m"]
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])
        ###
        self.LO = LO(config["n_class"], config)
        self.cla = nn.Linear(bit, config["n_class"]).to(config["device"])
        ###
    def forward(self, u, y, ind, config):
       
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        if "-T" in config["info"]:
            # Ternary Assignment
            u = (u.abs() > self.m).float() * u.sign()

        t = self.label2center(y)
        polarization_loss = (self.m - u * t).clamp(0).mean()
        ###
        u = self.cla(u)
        y=y.max(dim=1)[0] 
        # print(y.shape)
        # exit()
        loss = self.LO(u, y)
        ###
        return polarization_loss + config["loss_rate"] * loss
        
    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.target_vectors[y.argmax(axis=1)]
        else:
            # for multi label, use the same strategy as CSQ
            center_sum = y @ self.target_vectors
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # Random Assignments of Target Vectors
    def get_target_vectors(self, n_class, bit, p=0.5):
        target_vectors = torch.zeros(n_class, bit)
        for k in range(20):
            for index in range(n_class):
                ones = torch.ones(bit)
                sa = random.sample(list(range(bit)), int(bit * p))
                ones[sa] = -1
                target_vectors[index] = ones
        return target_vectors

    # Adaptive Updating
    def update_target_vectors(self):
        self.U = (self.U.abs() > self.m).float() * self.U.sign()
        self.target_vectors = (self.Y.t() @ self.U).sign()


def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = config["net"](pretrained=True,bit=bit).to(device)#3.20注释，修改transform

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = DPNLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        if "-A" in config["info"]:
            criterion.update_target_vectors()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)



if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        config["pr_curve_path"] = f"log/PIT/DPN_{config['dataset']}_{bit}.json"
        train_val(config, bit)
