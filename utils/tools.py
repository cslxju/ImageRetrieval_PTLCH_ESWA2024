import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json
import ssl
from torch.utils.data.dataset import Dataset
from torch.utils import *
ssl._create_default_https_context = ssl._create_unverified_context
def config_dataset(config):
    if "cifar10" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    config["data_path"] = "/home/rh/data/" + config["dataset"] + "/"
    if config["dataset"] == "cifar":
        config["data_path"] = "/home/rh/cifar10"
    if config["dataset"] == "nuswide":
        config["data_path"] = "/home/rh/data/nus_wide/"
    if config["dataset"] == "imagenet":
        config["data_path"] = "/home/rh/data/Imagenet/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "/dataset/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "/home/rh/data/coco/"
        config["data_path"] = "/home/rh/data/coco/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "/dataset/"
    if config["dataset"] == "mirflickr":
        config["data_path"] = "/home/rh/data/mirflickr/"
    if config["dataset"] == "VOC2012":
        config["data_path"] = "/home/rh/data/voc2012/"
    
    config["data"] = {
        "train_set": {"list_path": "/home/rh/HyP2-Loss-main/HyP2-Loss-main/data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},#{"list_path": "./data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},12.11 15.54修改
        "database": {"list_path": "/home/rh/HyP2-Loss-main/HyP2-Loss-main/data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},#{"list_path": "./data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},12.11 15.55修改
        "test": {"list_path": "/home/rh/HyP2-Loss-main/HyP2-Loss-main/data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}#{"list_path": "./data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}12.11 15.56修改    
       

    return config









class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def cifar_dataset(config):
    batch_size = config["batch_size"]
    
    train_size = 500
    test_size = 100

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),#将图片短边缩放至x，长宽比保持不变
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = '/home/rh/DeepHash-pytorch-master/cifar'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=8)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=8)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=8)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


class Flickr25k(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
   
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        # self.diff = None
        if mode == 'train':
            self.data = [Image.open(os.path.join(root, 'mirflickr', i)).convert('RGB') for i in Flickr25k.TRAIN_DATA]
            self.targets = Flickr25k.TRAIN_TARGETS
            # self.targets.dot(self.targets.T) == 0
        elif mode == 'query':
            self.data = [Image.open(os.path.join(root, 'mirflickr', i)).convert('RGB') for i in Flickr25k.QUERY_DATA]
            self.targets = Flickr25k.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = [Image.open(os.path.join(root, 'mirflickr', i)).convert('RGB') for i in Flickr25k.RETRIEVAL_DATA]
            self.targets = Flickr25k.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index], index
    def __len__(self):
        return len(self.data)

    def get_targets(self):
        return torch.FloatTensor(self.targets)

    @staticmethod
    def init(root, num_query, num_train):
        # Load dataset
        img_txt_path = os.path.join('/home/rh/HyP2-Loss-main/HyP2-Loss-main/data/flickr25k', 'img.txt')
        #targets_txt_path = os.path.join(root, 'test_label_onehot.txt')
        targets_txt_path = os.path.join('/home/rh/HyP2-Loss-main/HyP2-Loss-main/data/flickr25k', 'targets.txt')

        # Read files
        with open(img_txt_path, 'r') as f:
            data = np.array([i.strip() for i in f])#strip(),去除字符串两边的空格
        targets = np.loadtxt(targets_txt_path, dtype=np.int64)

        # Split dataset拆分数据集
        perm_file = 'flickr.txt'
        if os.path.exists(perm_file):
            perm_index = np.array(json.loads(open(perm_file, 'r').read()))
            print('------------- flickr loaded -------------')
        else:
            perm_index = np.random.permutation(data.shape[0]).tolist()
            flickr_txt = open(perm_file, 'w')
            flickr_txt.write(json.dumps(perm_index))
            flickr_txt.close()
            print('------------- flickr initialized -------------')
        
        query_index = perm_index[:num_query]
        train_index = perm_index[num_query: num_query + num_train]
        retrieval_index = perm_index[num_query + num_train:]

        Flickr25k.QUERY_DATA = data[query_index]
        Flickr25k.QUERY_TARGETS = targets[query_index, :]

        Flickr25k.TRAIN_DATA = data[train_index]
        Flickr25k.TRAIN_TARGETS = targets[train_index, :]

        Flickr25k.RETRIEVAL_DATA = data[retrieval_index]
        Flickr25k.RETRIEVAL_TARGETS = targets[retrieval_index, :]






def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)
    elif "mirflickr" in config["dataset"]:
        data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
                                #transforms被改过
        batch_size = config["batch_size"]
        Flickr25k.init('/home/rh/data/', 1000, 4000)
        trainset = Flickr25k('/home/rh/data/', 'train', transform = data_transform['train'])
        testset = Flickr25k('/home/rh/data/', 'query', transform = data_transform['val'])
        database = Flickr25k('/home/rh/data/', 'retrieval', transform = data_transform['val'])
        train_num = len(trainset)
        test_num = len(testset)
        database_num = len(database)
        print('train_dataset:',train_num)
        print('test_dataset:',test_num)
        print('database_dataset:',database_num)
        train_loader = data.DataLoader(trainset,
                                     batch_size =  batch_size,
                                     shuffle = True,#打乱数据集
                                     num_workers = 8)#dataloader一次性创建num_worker个worker

        test_loader = data.DataLoader(testset,
                             batch_size =  batch_size,
                             shuffle = False,
                             num_workers = 8)

        database_loader = data.DataLoader(database,
                             batch_size =  batch_size,
                             shuffle = False,
                             num_workers = 8)#是告诉DataLoader实例要使用多少个子进程进行数据加载(和CPU有关，和GPU无关)

        return train_loader, test_loader, database_loader, \
           train_num, test_num, database_num
    

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle= (data_set == "train_set") , num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def pr_curve(tst_binary, trn_binary, tst_label, trn_label):
  
    trn_binary = np.asarray(trn_binary, np.int32)
    
    trn_label = trn_label.numpy()
 
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.numpy()
    query_times = tst_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    
    AP = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)

    sum_p = np.zeros(trainset_len)
    sum_r = np.zeros(trainset_len)
    #f = open("./queryimg.txt","a+")
    for i in range(query_times):
  
        query_label = tst_label[i]
        
        query_binary = tst_binary[i,:]
        
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        
        sort_indices = np.argsort(query_result)
        
       
        buffer_yes = ((query_label @ trn_label[sort_indices].transpose())>0).astype(float)
        # print(buffer_yes)
        # exit()
        
        P = np.cumsum(buffer_yes) / Ns       
        
        R = np.cumsum(buffer_yes)/np.sum(buffer_yes)#(trainset_len)*10
        sum_p = sum_p+P
        sum_r = sum_r+R
        #f.writelines(str(sort_indices[:10]))    检索可视化
        #f.write('\r\n')
    return sum_p/query_times,sum_r/query_times
 
def p_top(tst_binary, trn_binary, tst_label, trn_label):
  
    trn_binary = np.asarray(trn_binary, np.int32)
    trn_label = trn_label.numpy()
 
    tst_binary = np.asarray(tst_binary, np.int32)
    tst_label = tst_label.numpy()
    query_times = tst_binary.shape[0]
    trainset_len = trn_binary.shape[0]
    # print(trainset_len)
    AP = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)

    sum_p = np.zeros(trainset_len)
    sum_r = np.zeros(trainset_len)
    
    for i in range(query_times):
  
        query_label = tst_label[i]
        query_binary = tst_binary[i,:]
        query_result = np.count_nonzero(query_binary != trn_binary, axis=1)    #don't need to divide binary length
        sort_indices = np.argsort(query_result)
        
       
        buffer_yes = ((query_label @ trn_label[sort_indices].transpose())>0).astype(float)
        
        P = np.cumsum(buffer_yes) / Ns       
        R = np.cumsum(buffer_yes)/np.sum(buffer_yes)#(trainset_len)*10
        sum_p = sum_p+P
        sum_r = sum_r+R
 
    return sum_p/query_times
# faster but more memory
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec = np.zeros((num_query, num_gallery))
    recall = np.zeros((num_query, num_gallery))
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        return_images = np.arange(1, num_gallery + 1)

        prec[iter, :] = prec_sum / return_images
        recall[iter, :] = prec_sum / all_sim_num

        assert recall[iter, -1] == 1.0
        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    index = np.argwhere(recall[:, -1] == 1.0)
    index = index.squeeze()
    prec = prec[index]
    recall = recall[index]
    cum_prec = np.mean(prec, 0)
    cum_recall = np.mean(recall, 0)

    return topkmap, cum_prec, cum_recall

# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset):
    device = config["device"]
    # print("calculating test binary code......")
    tst_binary, tst_label = compute_result(test_loader, net, device=device)

    # print("calculating dataset binary code.......")
    trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
    np.savetxt('cifar10dateset1_binary.txt',trn_binary,fmt='%d')
    np.savetxt('cifar10dateset1_label.txt',trn_label,fmt='%d')
    if "pr_curve_path" not in  config:
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(), config["topK"])
    else:
        # need more memory
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(tst_binary.numpy(), tst_label.numpy(),
                                                     trn_binary.numpy(), trn_label.numpy(),
                                                     config["topK"])
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]

        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }
        os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
        with open(config["pr_curve_path"], 'w') as f:
            f.write(json.dumps(pr_data))
        print("pr curve save to ", config["pr_curve_path"])
        
    if mAP > Best_mAP:
        Best_mAP = mAP
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
            np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
            np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
            np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
            torch.save(net.state_dict(), os.path.join(save_path, "model.pt"))
    print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print(config)
    return Best_mAP
