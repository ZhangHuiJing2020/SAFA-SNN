import numpy as np
import torch
# from ansible.modules.cloud.rackspace.rax_files_objects import download
from dataloader.sampler import CategoriesSampler
import torchvision.transforms as transforms
from .neur_data_utils import *
from torch.utils.data import DataLoader
def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes =100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    elif args.dataset == 'n_caltech101':
        from spikingjelly.datasets.n_caltech101 import NCaltech101 as Dataset
        args.base_class = 61
        args.num_classes = 101
        args.way = 5
        args.shot = 500
        args.sessions = 9
    elif args.dataset == 'cifar10dvs' or args.dataset == 'dvscifar10':
        from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as Dataset
        # from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS as Dataset
        # import dataloader.dvscifar10.dvscifar10 as Dataset
        args.base_class = 6
        args.num_classes = 10
        args.way = 1
        args.shot = 50
        args.sessions = 5
    elif args.dataset == 'dvs128gesture':
        from spikingjelly.datasets.dvs128_gesture import DVS128Gesture as Dataset
        args.base_class = 6
        args.num_classes = 11
        args.way = 1
        args.shot = 1
        args.sessions = 6
    elif args.dataset == 'mnist':
        from spikingjelly.datasets.n_mnist import NMNIST as Dataset
        args.base_class = 6
        args.num_classes = 10
        args.way = 1
        args.shot = 5
        args.sessions = 5
    args.Dataset = Dataset
    return args


def get_dataloader(args,session, is_meta=None, transform=None):
    if args.dataset in ['cifar10dvs', 'n_caltech101']:
        c_dataset = args.Dataset(root=args.dataroot, data_type='frame', frames_number=args.time_step, split_by='number')
        trainset, testset = split_to_train_test_set(0.9, c_dataset, args.num_classes)
        trainset, train_idx = split_dvs128gesture_train_set(trainset, args.num_classes, session=session, args=args)
        testset, test_idx = split_dvs128gesture_test_set(testset, args.num_classes, session=session, args=args)
        trainloader = DataLoader(trainset, batch_size=args.batch_size_base, shuffle=True, num_workers=8)
        testloader = DataLoader(testset, batch_size=args.batch_size_base, shuffle=False, num_workers=8)

        return trainset, trainloader, testloader, train_idx, test_idx
    elif args.dataset in ['dvs128gesture']:
        train_set = args.Dataset(args.dataroot, train=True, data_type='frame', frames_number=20, split_by='number')
        test_set = args.Dataset(args.dataroot, train=False, data_type='frame', frames_number=20, split_by='number')
        train_set, train_idx = split_dvs128gesture_train_set( train_set, args.num_classes, session=session, args=args)
        test_set, test_idx = split_dvs128gesture_test_set( test_set, args.num_classes, session=session, args=args)
        trainloader = DataLoader(train_set, batch_size=args.batch_size_base, shuffle=True, num_workers=8)
        testloader = DataLoader(test_set, batch_size=args.batch_size_base, shuffle=False, num_workers=8)
        return train_set, trainloader, testloader, train_idx, test_idx
    if session == 0:
        if is_meta == "meta":
            trainset, trainloader, testloader = get_base_dataloader_meta(args)
        else:
            trainset, trainloader, testloader = get_base_dataloader(args, transform)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader

def get_base_dataloader(args, transform=None):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True, transform=transform,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    elif args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    elif args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)
    elif args.dataset == 'n_caltech101':
        trainset = args.Dataset.NCaltech101(root=args.dataroot, train=True, index=class_index, base_sess=True)
        testset = args.Dataset.NCaltech101(root=args.dataroot, train=False, index=class_index)

    elif args.dataset == 'dvscifar10' or 'cifar10dvs':
        # trainset = args.Dataset.DVSCifar10(root=args.dataroot, train=True, index=class_index, base_sess=True)
        trainset = args.Dataset.DVSCifar10(root=args.dataroot, train=True)
        testset = args.Dataset.DVSCifar10(root=args.dataroot, train=False)
    elif args.dataset == 'dvs128gesture':
        trainset = args.Dataset.DVS128Gesture(root=args.dataroot, train=True)
        testset = args.Dataset.DVS128Gesture(root=args.dataroot, train=False)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True, index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True, index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

def get_novel_test_dataloader(args, session):
    txt_path = "./data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_new = get_all_novel_task_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                             index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=8, pin_memory=True)

    return testloader

def get_task_specific_test_dataloader(self, session):
    if session == 0:
        _, _, testloader = get_base_dataloader(self.args)
    else:
        testloader = get_task_specific_new_dataloader(self.args, session)
    return testloader

def get_task_specific_new_dataloader(args, session):
    txt_path = "./data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    class_new = get_task_session_classes(args, session)
    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                             index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=8, pin_memory=True)
    return testloader

def get_all_novel_task_session_classes(args, session):
    class_list = np.arange(args.base_class, args.base_class + session * args.way)
    return class_list
def get_task_session_classes(args, session):
    class_list = np.arange(args.base_class + (session-1) * args.way, args.base_class + session * args.way)
    return class_list

def get_dataloader_pk(self, session):
    if session == 0:
        trainset, trainloader, testloader, trainloader_pk = get_base_dataloader_pk(self.args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        trainloader_pk = None
    return trainset, trainloader, testloader, trainloader_pk

def get_base_dataloader_pk(args):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True, size=args.image_size)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index, size=args.image_size)

    sampler = PKsampler(trainset, p=args.p, k=args.k)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=args.num_workers, pin_memory=True)
    #
    trainloader_pk = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, sampler=sampler,
                                              num_workers=args.num_workers, pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader, trainloader_pk
