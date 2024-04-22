





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
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 1e-5}},
        # "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 50,
                       # "optim_params": {"lr": 3e-2, "weight_decay": 1e-4}},
        "info": "[0.1DPNLCALT]",
        # "info": "[DPN-A]",
        # "info": "[DPN-T]",
        # "info": "[DPN-A-T]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        # "net": AlexNet,
        # "net": ResNet,
        "net": pit_s, "net_print": "PIT",
        #"dataset": "cifar10",
        #"dataset": "imagenet",
        "dataset": "coco",
        # "dataset": "nuswide_21",
        "epoch": 100,
        "test_map": 10,
        "save_path": "savepr/DPNLCALT",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:2"),
        "bit_list": [16],
        "loss_rate":0.1,
        "noadd":0
    }
    config = config_dataset(config)
    return config


###
class LT(nn.Module):
    def __init__(self,):
        super(dual_softmax_loss, self).__init__()
        
    def forward(self, sim_matrix, temp=1000):
        print(sim_matrix.shape)
        t = F.softmax(sim_matrix/temp, dim=0)
        q = len(sim_matrix)
        print(t.shape)
        print(q)
        exit()
        sim_matrix = sim_matrix * F.softmax(sim_matrix/temp, dim=0)*len(sim_matrix) #With an appropriate temperature parameter, the model achieves higher performance
        
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        # print(logpt.shape)
        # exit()
        loss = -logpt
        # print(loss.shape)
        # exit()
        return loss.mean()





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
        self.LT = LT()
        self.cla = nn.Linear(bit, config["n_class"]).to(config["device"])
        ###
    def forward(self, u, y, ind, config):
        
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        if "-T" in config["info"]:
            # Ternary Assignment
            u = (u.abs() > self.m).float() * u.sign()
            
        t = self.label2center(y)
        # print(u.shape)
        # print(t.shape)
        # exit()
        polarization_loss = (self.m - u * t).clamp(0).mean()
        ###
        # u = self.cla(u)
        # y=y.max(dim=1)[0] 
        # print(u.shape)
        # print(y.shape)
        # exit()
        loss = self.dual_softmax_loss(u)
        # print(loss)
        # exit()
        return polarization_loss + config["loss_rate"] * loss
        ###
        
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
            # Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
             # Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            # trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            # np.savetxt('imagenetdateset1_binary.txt',trn_binary,fmt='%d')
            # np.savetxt('imagenetdateset1_label.txt',trn_label,fmt='%d')
            # Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            # Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            # trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            # np.savetxt('resnetdateset_binary.txt',trn_binary,fmt='%d')
            # np.savetxt('resnetdateset_label.txt',trn_label,fmt='%d')
            tst_binary, tst_label = compute_result(test_loader, net, device=device)
            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
            # np.savetxt('SNEcifar1064dateset_binary.txt',trn_binary,fmt='%d')
            # np.savetxt('SNEcifar1064dateset_label.txt',trn_label,fmt='%d')
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            P, R = pr_curve(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )
            P_top = p_top(
                    tst_binary,
                    trn_binary,
                    tst_label,
                    trn_label,
                )         
            print(
                "%s epoch:%d, bit:%d, dataset:%s, MAP:%.4f" % (config["info"], epoch + 1, bit, config["dataset"], mAP))
            if mAP > Best_mAP:
                Best_mAP = mAP
                checkpoint = {                  
                    'P': P,
                    'R': R,
                    'P_top':P_top,
                }
            if "save_path" in config:
                 if not os.path.exists(config["save_path"]):
                     os.makedirs(config["save_path"])
                 print("save in ", config["save_path"])
                 np.save(os.path.join(config["save_path"], config["dataset"] + "-" + str(config["bit_list"]) + "-" + str(mAP) + "-" + "tst_binary.npy"),
                         tst_binary.numpy())
                 np.save(os.path.join(config["save_path"], config["dataset"] + "-" + str(config["bit_list"]) + "-" + str(mAP) + "-" + "trn_binary.npy"),
                         trn_binary.numpy())
                 np.save(os.path.join(config["save_path"], config["dataset"] + "-" + str(config["bit_list"]) + "-" + str(mAP) + "-" + "tst_label.npy"),
                         tst_label.numpy())
                 np.save(os.path.join(config["save_path"], config["dataset"] + "-" + str(config["bit_list"]) + "-" + str(mAP) + "-" + "trn_label.npy"),
                         trn_label.numpy())
                 torch.save(net.state_dict(), os.path.join(config["save_path"], config["dataset"] + "-" + str(config["bit_list"]) + "-" + str(mAP) + "-model.pth"))
                  
            print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
            print(config)
    print("bit:%d,Best MAP:%.4f" % (bit, Best_mAP))
    # logger.info(str(bit)+'_map: {:.4f}'.format(mAP))
    # np.savetxt ('loss2.txt',ln,fmt='%3.5f') 
    if not os.path.isdir('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)):
                os.makedirs('result//'+config["dataset"]+'//'+config["info"]+'//'+str(bit))
    P = checkpoint['P']
    R = checkpoint['R']
    P_top = checkpoint['P_top']
    np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'p.txt',P,fmt='%3.5f')  
    np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'r.txt',R,fmt='%3.5f')
    np.savetxt ('result/'+config["dataset"]+'/'+config["info"]+'/'+str(bit)+'/'+'P_top.txt',P_top,fmt='%3.5f') 




if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        # config["pr_curve_path"] = f"DELETEHASH/PITCOTSOFTMAX/COCO_{config['dataset']}_{config['loss_rate']}_{bit}.json"
        train_val(config, bit)
