import copy
import glob
import os
import random
from collections import defaultdict
import torch

import numpy as np
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from torch.backends import cudnn
from torch.utils.data import Sampler

from softmax_loss import CrossEntropyLabelSmooth

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# torch.manual_seed(345)
# torch.cuda.manual_seed(345)
# torch.cuda.manual_seed_all(345)
# np.random.seed(345)
# random.seed(345)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
import utils
##################
# 参数列表
###################
from Dataset import LM
from torchvision import transforms
import matplotlib.pyplot as plt
# 超参数
save_path = 'model'
image_size = 224
batch_size = 25#40
#weight3:(5e-3)/2,(1e-4)/2;weight4:0.01,1e-9;weight5:2.5e-3,5e-5,
lr = 0.01#5e-2  初始学习率
max_epoch = 5

#提前停止参数
eval_T = 8  # 每训练几次，判断一次
P = 2  # 下次精度比上次精度小的次数
count_acc = 0  # 统计小于的次数f

Loss_list = []
Accuracy_list = []
epoh = []


import os.path as osp


###################

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    # 采样器  防止小车因数量过多  规定batch   p 类  k 样本数
    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


# 数据预处理   图片对应label    x_tran 与 y（类别）
def get_data():
    label = []
    label_dict = {}
    txt = np.loadtxt('label.txt', 'str', delimiter=',')
    for t in txt:
        l = int(t[1])

        label_dict[t[0]] = l - 1
    img_paths = glob.glob(osp.join('shuju-bi', '*/*.jpg'))
    for path in img_paths:
        # print(path)
        label.append(label_dict[path.split('\\')[1]])
    X_train = img_paths
    y_train = label
    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=0)
    # 数据增强 Resize 一种细胞
    trans = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        # torch_transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 标准化
        transforms.RandomErasing()
        # torch_transforms.RandomErasing(p=1,scale=(0.33,0.77),ratio=(0.33,0.77)),
        # torch_transforms.RandomErasing(p=1,scale=(0.33,0.77),ratio=(0.33,0.77)),
        # torch_transforms.RandomRotation(30)
    ])
    trans2 = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # torch_transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # torch_transforms.RandomErasing(p=1,scale=(0.33,0.77),ratio=(0.33,0.77)),
        # torch_transforms.RandomRotation(30)
    ])

    # pytorch
    valid_dataset = LM(X_test, y_test, trans2)
    train_dataset = LM(X_train, y_train, trans)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=100, shuffle=False, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               sampler=RandomIdentitySampler(train_dataset, batch_size, 6),
                                               shuffle=False, pin_memory=True)
    return train_loader, valid_dataloader

# 训练
def train(train_loader, valid_loader):
    temp_val_loss = 99999  # 上次精度
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=6)  # b1 小分类
    loss_fn = CrossEntropyLabelSmooth(num_classes=6)  # 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # Adam优化器,weight_decay权重衰减防止过拟合
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=8, last_epoch=-1,gamma=0.8)  # 学习率  weigth5:gamma=0.5,20

    model.cuda()

    acc_meter = utils.AverageMeter()
    loss_meter = utils.AverageMeter()
    loader = train_loader
    for epoch in range(max_epoch):
        loss_meter.reset()
        acc_meter.reset()
        model.train()
        for i, (x, y) in enumerate(loader):  # x图 y label
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            optimizer.zero_grad()
            loss.backward()  # 方向传播
            optimizer.step()
            acc = (outputs.max(1)[1] == y).float().mean()
            loss_meter.update(loss.item(), x.shape[0])
            acc_meter.update(acc, 1)

            print("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                  .format(epoch, (i + 1), len(train_loader),
                          loss_meter.avg, acc_meter.avg, optimizer.state_dict()['param_groups'][0]['lr']))


        acc_list = []
        with torch.no_grad():
            for i, (x, y) in enumerate(valid_loader):
                x, y = x.cuda(), y.cuda()
                outputs = model(x)
                acc = (outputs.max(1)[1] == y).float().mean()
                acc_list.append((acc.item()))
        acc = np.mean(acc_list)
        print(acc)
        # if epoch % 2 == 0:
        torch.save(model, 'weight3/' + str(acc)[:5] + '-' + str(epoch) + '.pt')
        scheduler.step()


        Loss_list.append(loss.item())
        Accuracy_list.append(100 * acc.item())
        epoh.append(epoch+1)



        # early stopping
        if (epoch % eval_T) == 0:

            if (temp_val_loss > acc):
                temp_val_loss = acc
                count_acc = 0  # reset count_acc
            else:
                count_acc = count_acc + 1
        if count_acc > P:
            print("Early Stopping! Epoch : ", epoch, )
            break



t, v = get_data()
train(t, v)
plt.xlim((1, len(epoh)))
plt.ylim((0, 100))
x1 = epoh
y1 = Accuracy_list
plt.plot(x1, y1, 'b')
plt.title('Train accuracy vs. epoches')
plt.xlabel('epoches')
plt.ylabel('Train accuracy')
plt.savefig("train_accuracy.jpg")
plt.show()
x2 = epoh
y2 = Loss_list
plt.plot(x2, y2, 'r')
plt.title('Train loss vs. epoches')
plt.xlabel('epoches')
plt.ylabel('Train loss')
plt.savefig("train_loss.jpg")
plt.show()