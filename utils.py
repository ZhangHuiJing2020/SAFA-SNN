import os
import sys
import io
import time
import random
import logging
import datetime
import pprint
import pickle
import csv
from collections import OrderedDict, defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import tool

def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
            return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def save_list_to_txt(name, input_list):
    f = open(name, mode='a')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()

def get_optimizer_scheduler(args, optimize_parameters=None):
    if args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(optimize_parameters, args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=args.decay)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(optimize_parameters, args.lr_base, weight_decay=args.decay)
    if args.schedule == 'Step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)
    elif args.schedule == 'Milestone':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                         gamma=args.gamma)
    elif args.schedule == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_base, eta_min=1e-6)
    return optimizer, scheduler

def set_save_path(self):
    time_str = datetime.datetime.now().strftime('%m%d-%H-%M-%S-%f')[:-3]
    self.args.time_str = time_str
    mode = self.args.base_mode + '_' + self.args.new_mode
    self.args.save_path = '%s/%s/%s_%s/epo_b%d-epo_n%d-bs_b%d-bs_n%d-bs_t%d_step%s_seed_%d_lr_b_%.4f_way_%d_shot_%d' % (self.args.dataset, self.args.project, self.args.network, mode,
                                                        self.args.epochs_base, self.args.epochs_new, self.args.batch_size_base, self.args.batch_size_new, self.args.test_batch_size, self.args.time_step, self.args.seed,  self.args.lr_base,
                                                                                                                         self.args.way, self.args.shot)

    if self.args.project == 'sparsity':
        self.args.save_path = self.args.save_path + '-beta_%.4f-theta_%.4f' % (self.args.beta, self.args.theta)

    self.args.save_path = os.path.join('checkpoint', self.args.save_path)
    if os.path.exists(self.args.save_path):
        pass
    else:
        print('create folder:', self.args.save_path)
        os.makedirs(self.args.save_path)
    return None


def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1)
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)

def log_to_file(log_name):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_name, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("hello")
    return logger


def confmatrix(logits, label, filename):
    font = {'family': 'DejaVu Sans', 'size': 18}
    matplotlib.rc('font', **font)
    matplotlib.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18})
    plt.rcParams["font.family"] = "DejaVu Sans"

    pred = torch.argmax(logits, dim=1)
    cm = confusion_matrix(label, pred, normalize='true')
    # print(cm)
    clss = len(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    if clss <= 100:
        plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
        plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    elif clss <= 200:
        plt.yticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
        plt.xticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
    else:
        plt.yticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
        plt.xticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)

    # plt.xlabel('Predicted Label', fontsize=20)
    # plt.ylabel('True Label', fontsize=20)
    # plt.tight_layout()
    # plt.savefig(filename + '.pdf', bbox_inches='tight')
    # plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    cbar = plt.colorbar(cax)  # This line includes the color bar
    cbar.ax.tick_params(labelsize=16)
    if clss <= 100:
        plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
        plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    elif clss <= 200:
        plt.yticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
        plt.xticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
    else:
        plt.yticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
        plt.xticks([0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16)
    # plt.xlabel('Predicted Label', fontsize=20)
    # plt.ylabel('True Label', fontsize=20)
    # plt.tight_layout()
    # plt.savefig(filename + '_cbar.pdf', bbox_inches='tight')
    # plt.close()

    return cm

def harm_mean(seen, unseen):
    # compute from session1
    assert len(seen) == len(unseen)
    harm_means = []
    for _seen, _unseen in zip(seen, unseen):
        _hmean = (2 * _seen * _unseen) / (_seen + _unseen + 1e-12)
        _hmean = float('%.3f' % (_hmean))
        harm_means.append(_hmean)
    return harm_means


def print_config(args, logger=None, is_end=False):
    if is_end:
        logger.info(f'method: {args.project}, dataset: {args.dataset}, backbone: {args.network}, epochs_base: {args.epochs_base}, epochs_new: {args.epochs_new},'
                     f' bs_base: {args.batch_size_base}, bs_new: {args.batch_size_new}, test_bs: {args.test_batch_size},'
                     f' base_mode: {args.base_mode}, new_mode: {args.new_mode}, seed: {args.seed}, n_way: {args.way}, k_shot: {args.shot}, lr_new: {args.lr_new}, time_step: {args.time_step}, sg:{args.sg}'
                    f' device: {args.device}')
    else:
        print(
            f'method: {args.project}, dataset: {args.dataset}, backbone: {args.network}, epochs_base: {args.epochs_base}, epochs_new: {args.epochs_new},'
            f' bs_base: {args.batch_size_base}, bs_new: {args.batch_size_new}, test_bs: {args.test_batch_size},'
            f' base_mode: {args.base_mode}, new_mode: {args.new_mode}, seed: {args.seed}, n_way: {args.way}, k_shot: {args.shot}, time_step: {args.time_step}, lr_base: {args.lr_base}, sg: {args.sg}'
            f' device: {args.device}')

def debug_pr(x, name):
    print(f'{name}.shape = {x.shape}')



def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def print_param_size(model, self):
    dtype_size = 4
    if len(self.args.gpu) > 1:
        num_params = count_parameters(model.module)
        num_params_train = count_parameters(model, True)
    else:
        num_params = count_parameters(model)
        num_params_train = count_parameters(model, True)
    num_params_size = num_params * dtype_size / (1024 ** 2)
    num_params_train_size = num_params_train * dtype_size / (1024 ** 2)
    # if isinstance(logger, list):
    if hasattr(self, 'logs'):
        log_and_print(self, "All params: {}, Size: {}".format(num_params, num_params_size))
        log_and_print(self, "Trainable params: {}, Size: {}".format(num_params_train, num_params_train_size))
    else:
        print_log("All params: {}, Size: {}".format(num_params, num_params_size))
        print_log("Trainable params: {}, Size: {}".format(num_params_train, num_params_train_size))



def plot_sparsity_accuracy_tradeoff(epoch_sparsity, epoch_acc, save_path, title="Sparsity–Accuracy Trade-off"):
    """
    绘制稀疏性-准确率 Trade-off 曲线
    Args:
        epoch_sparsity: list，每个 epoch 的平均脉冲稀疏性
        epoch_acc: list，每个 epoch 的准确率
        title: 图表标题
    """
    plt.figure(figsize=(7, 5))

    # 主线
    plt.plot(epoch_sparsity, epoch_acc,
             marker='o', markersize=6, linestyle='-', linewidth=1.8,
             color='tab:blue', label='Epoch steps')

    # 轴标签 & 标题
    plt.xlabel("Sparsity", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    # plt.title(fontsize=14, weight='bold')

    # 网格 & 坐标轴美化
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 范围限制，留一点空间给标注
    plt.xlim(min(epoch_sparsity)*0.95, max(epoch_sparsity)*1.05)
    plt.ylim(min(epoch_acc)*0.95, max(epoch_acc)*1.05)

    # 图例
    plt.legend(fontsize=10, loc='best')

    plt.tight_layout()

    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 图像已保存到: {save_path}")
    else:
        plt.show()

# 保存每组实验的数据
def save_experiment_data(save_dir, exp_name, epoch_sparsity, epoch_acc):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{exp_name}.pkl")
    data = {
        "sparsity": epoch_sparsity,
        "accuracy": epoch_acc
    }
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"✅ 保存实验数据到 {file_path}")


def save_experiment_csv(save_dir, exp_name, epoch_sparsity, epoch_acc):
    """
    保存实验数据到文本文件，每行：epoch,sparsity,accuracy
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{exp_name}.csv")

    with open(file_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "sparsity", "accuracy"])  # 表头
        for i, (s, a) in enumerate(zip(epoch_sparsity, epoch_acc)):
            writer.writerow([i+1, s, a])

    print(f"✅ 实验数据已保存到 {file_path}")


def save_experiment_simple_txt(save_dir, exp_name, epoch_sparsity, epoch_acc):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{exp_name}.txt")

    with open(file_path, "w") as f:
        f.write("Epoch\tSparsity\tAccuracy\n")
        for i, (s, a) in enumerate(zip(epoch_sparsity, epoch_acc)):
            f.write(f"{i+1}\t{s:.4f}\t{a:.4f}\n")

    print(f"✅ 实验数据已保存到 {file_path}")


def log_and_print(self, msg_template, *args, **kwargs):
    """
    将日志信息格式化后追加到列表并打印。
    支持：
      - 关键字参数 {key}
      - 位置参数 {}
      - 或直接传入已经格式化好的字符串

    参数:
        log_list (list): 存储日志的列表
        msg_template (str): 日志模板，支持 {} 或 {key} 占位符
        *args: 位置参数
        **kwargs: 关键字参数
    """
    try:
        if args or kwargs:
            # 使用 format 格式化
            formatted_msg = msg_template.format(*args, **kwargs)
        else:
            # 没有参数，直接使用原字符串
            formatted_msg = msg_template
    except KeyError as e:
        # 捕获 KeyError，防止模板占位符缺失
        formatted_msg = msg_template + f" [Format KeyError: {e}]"

    self.logs.append(formatted_msg)
    print(formatted_msg)

def multi_gpu(args):
    if len(args.gpu) > 1:
        return True
    else:
        return False

def print_log(msg_template, *args, **kwargs):
    try:
        if args or kwargs:
            # 使用 format 格式化
            formatted_msg = msg_template.format(*args, **kwargs)
        else:
            # 没有参数，直接使用原字符串
            formatted_msg = msg_template
    except KeyError as e:
        # 捕获 KeyError，防止模板占位符缺失
        formatted_msg = msg_template + f" [Format KeyError: {e}]"
    tool.logs.append(formatted_msg)
    print(formatted_msg)
