import torch
import random
import numpy as np
import math
import torchvision.transforms as transforms
# from dataloader.dvscifar10.dvscifar10 import *
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed} for reproducibility.")


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)

def split_caltech_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, session: int = 0, random_split: bool = False, args=None):
    """
    :param train_ratio: 训练集占比（这里只用于 session=0 时）
    :param origin_dataset: 原始数据集
    :param num_classes: 总类别数
    :param session: 当前训练 session：
        - session=0：训练集和测试集都取前 6 个类的所有样本
        - session≥1：训练集取 session+6 号类别的 5 个样本，测试集包含前 session+6 号类别的所有样本
    :param random_split: 是否对每个类别的样本随机打乱
    :return: 训练集 (train_set), 测试集 (test_set)
    """

    # 1. 初始化类别索引
    label_idx = [[] for _ in range(num_classes+1)]
    labels = []  # 存储标签

    # 2. 遍历数据集，构建类别索引
    for i, item in enumerate(origin_dataset):
        y = item[1]  # 获取标签
        if isinstance(y, (np.ndarray, torch.Tensor)):
            y = y.item()  # 转换为整数
        label_idx[y].append(i)  # 存储索引
        labels.append(y)  # 存储对应标签


    # 3. 随机打乱（如果需要）
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    train_idx = []
    test_idx = []
    train_labels = []  # 存储训练集的标签
    test_labels = []   # 存储测试集的标签

    # 4. session=0 时，选前 6 个类别的所有样本
    if session == 0:
        selected_classes = list(range(min(args.base_class, num_classes)))  # 选前 6 类
        print("selected_classes:", selected_classes)
        for i in selected_classes:
            train_idx.extend(label_idx[i])  # 训练集
            test_idx.extend(label_idx[i])   # 测试集
            train_labels.extend([labels[idx] for idx in label_idx[i]])  # 记录对应标签
            test_labels.extend([labels[idx] for idx in label_idx[i]])  # 记录对应标签

    # 5. session ≥ 1 时
    else:
        # 训练数据：只取当前 session 的 args.way 个类，每类取 args.shot 个样本
        train_classes = list(range(args.base_class + (session - 1) * args.way, args.base_class + session * args.way))
        train_classes = [cls for cls in train_classes if cls < num_classes]  # 确保不超过类别总数

        print(f"session:{session},train_classes:{train_classes}")

        for cls in train_classes:
            train_idx.extend(label_idx[cls][:args.shot])
            train_labels.extend([origin_dataset[idx][1] for idx in label_idx[cls][:args.shot]])

        # 1. 初始化测试类别列表，包含 session=0 的 base_class 个类
        test_classes = list(range(args.base_class))

        # 2. 遍历每个 session（从 1 到当前 session）
        for s in range(1, session + 1):
            session_classes = list(range(args.base_class + (s - 1) * args.way, args.base_class + s * args.way))
            print(f"session:{session},session_class:{session_classes}")
            test_classes.extend([cls for cls in session_classes if cls < num_classes])

        # 3. 获取测试集数据
        for cls in test_classes:
            if cls < args.base_class:
                # session=0 的 base_class 类，获取该类别的所有样本
                test_idx.extend(label_idx[cls])
                test_labels.extend([origin_dataset[idx][1] for idx in label_idx[cls]])
            else:
                # session≥1 时，每个 session 取 args.way 个类，每个类取 args.shot 个样本
                test_idx.extend(label_idx[cls][:args.shot])
                test_labels.extend([origin_dataset[idx][1] for idx in label_idx[cls][:args.shot]])

    for item in np.unique(train_labels):
        print(f"session:{session} train label item:{item}")
    for item in np.unique(test_labels):
        print(f"session:{session} test label item:{item}")

    # 6. 返回划分后的数据集
    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx), train_labels, test_labels

def split_dvs128gesture_train_set(origin_dataset: torch.utils.data.Dataset, num_classes: int, session: int = 0, random_split: bool = False, args=None):
    # 1. 初始化类别索引
    label_idx = [[] for _ in range(num_classes+1)]
    labels = []  # 存储标签
    # 2. 遍历数据集，构建类别索引
    for i, item in enumerate(origin_dataset):

        y = item[1]  # 获取标签
        if isinstance(y, (np.ndarray, torch.Tensor)):
            y = y.item()  # 转换为整数
        label_idx[y].append(i)  # 存储索引
        labels.append(y)  # 存储对应标签
    # 3. 随机打乱（如果需要）
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])
    train_idx = []

    train_labels = []  # 存储训练集的标签


    # 4. session=0 时，选前 6 个类别的所有样本
    if session == 0:
        selected_classes = list(range(min(args.base_class, num_classes)))  # 选前 6 类
        print("selected_classes:", selected_classes)
        for i in selected_classes:
            train_idx.extend(label_idx[i])  # 训练集
            train_labels.extend([labels[idx] for idx in label_idx[i]])  # 记录对应标签


    # 5. session ≥ 1 时
    else:
        # 训练数据：只取当前 session 的 args.way 个类，每类取 args.shot 个样本
        train_classes = list(range(args.base_class + (session - 1) * args.way, args.base_class + session * args.way))
        train_classes = [cls for cls in train_classes if cls < num_classes]  # 确保不超过类别总数

        print(f"session:{session},train_classes:{train_classes}")

        for cls in train_classes:
            train_idx.extend(label_idx[cls][:args.shot])
            train_labels.extend([origin_dataset[idx][1] for idx in label_idx[cls][:args.shot]])

    for item in np.unique(train_labels):
        print(f"session:{session} train label item:{item}")

    # 6. 返回划分后的数据集
    return torch.utils.data.Subset(origin_dataset, train_idx), train_labels

def split_dvs128gesture_test_set(origin_dataset: torch.utils.data.Dataset, num_classes: int, session: int = 0, random_split: bool = False, args=None):

    # 1. 初始化类别索引
    label_idx = [[] for _ in range(num_classes+1)]
    labels = []  # 存储标签

    # 2. 遍历数据集，构建类别索引
    for i, item in enumerate(origin_dataset):
        y = item[1]  # 获取标签
        if isinstance(y, (np.ndarray, torch.Tensor)):
            y = y.item()  # 转换为整数

        label_idx[y].append(i)  # 存储索引
        labels.append(y)  # 存储对应标签


    # 3. 随机打乱
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    # train_idx = []
    test_idx = []
    # train_labels = []  # 存储训练集的标签
    test_labels = []   # 存储测试集的标签

    selected_classes = list(range(args.base_class + session * args.way))
    print("selected_classes:", selected_classes)
    for i in selected_classes:
        # train_idx.extend(label_idx[i])  # 训练集
        test_idx.extend(label_idx[i])   # 测试集
        test_labels.extend([labels[idx] for idx in label_idx[i]])  # 记录对应标签

    for item in np.unique(test_labels):
        print(f"session:{session} test label item:{item}")

    # 6. 返回划分后的数据集
    return torch.utils.data.Subset(origin_dataset, test_idx), test_labels