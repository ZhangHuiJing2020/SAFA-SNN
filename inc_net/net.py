from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tool


class NET(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        if args.dataset == 'cifar100':
            C, H, W = 3, 32, 32
        elif args.dataset == 'mini_imagenet':
            C, H, W = 3, 84, 84
        elif args.dataset == 'cub200':
            C, H, W = 3, 224, 224
        elif args.dataset == 'n_caltech101':
            C, H, W = 3, 300, 200

        if args.network == 'svgg9':
            from backbone.spiking_vgg import VGG9SNN
            self.encoder = VGG9SNN(time_step=args.time_step)
            self.num_features = 1024
        elif args.network == 'svgg5':
            from backbone.spiking_vgg import VGG5SNN
            self.encoder = VGG5SNN(time_step=args.time_step, args=args)
            self.num_features = 1024
        elif args.network == 'svgg16':
            from backbone.spiking_vgg import VGG16SNN
            self.encoder = VGG16SNN(time_step=args.time_step, args=args)
            self.num_features = 4096
        elif args.network == 'svgg11':
            from backbone.spiking_vgg import VGG11SNN
            self.encoder = VGG11SNN(time_step=args.time_step, args=args)
            self.num_features = 4096
        elif args.network == 'spikingformer':
            from backbone.spikingformer import Spikingformer
            self.encoder = Spikingformer(H=H, W=W, C=C)
            self.num_features = 128
        elif args.network == 'sresnet19':
            from backbone.spiking_resnet import resnet19
            self.encoder = resnet19(num_classes=100, time_step=args.time_step)
            self.num_features = 256
        elif args.network == 'sresnet18':
            from backbone.spiking_resnet import spiking_resnet18
            self.encoder = spiking_resnet18(args)
            self.num_features = 512
        elif args.network == 'sresnet20':
            from backbone.spiking_resnet import spiking_resnet20
            self.encoder = spiking_resnet20(num_classes=100, time_step=args.time_step)
            self.num_features = 256
        elif args.network == 'sresnet34':
            from backbone.spiking_resnet import spiking_resnet34
            self.encoder = spiking_resnet34(num_classes=100, time_step=args.time_step)
            self.num_features = 256

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_features, args.num_classes, bias=False)

    def encode(self, x, session=0):
        return self.encoder(x, args=self.args)

    def forward_metric(self, x, session=0):
        x = self.encode(x, session=session)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
        elif 'dot' in self.mode:
            x = self.fc(x)
        return self.args.temperature * x

    def forward(self, x, session=0):
        if self.mode == 'encoder':
            return self.encode(x, session=session)
        elif self.mode is not None:
            return self.forward_metric(x, session=session)
        else:
            raise ValueError("Unknown mode")

    def update_fc(self, dataloader, class_list, session):
        features_all = []
        labels_all = []
        for data, label in dataloader:
            if len(self.args.gpu) > 1:
                data = data.cuda()
                label = label.cuda()
            else:
                data = data.to(self.args.device)
                label = label.to(self.args.device)
            features = self.encode(data, session=session).detach()
            if self.args.network == 'spikingformer':
                features = features.mean(0)
            else:
                features = features.mean(1)
            features_all.append(features)
            labels_all.append(label)
        data = torch.cat(features_all, dim=0)
        label = torch.cat(labels_all, dim=0)
        if self.args.not_data_init:
            new_fc = nn.Parameter(torch.rand(len(class_list), self.num_features, device=data.device), requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)
        return new_fc

    def update_fc_avg(self, data, label, class_list):
        new_fc = []
        for class_index in class_list:
            index = (label == class_index).nonzero().squeeze(-1)
            embedding = data[index]
            proto = embedding.mean(0)
            self.fc.weight.data[class_index] = proto
            new_fc.append(proto)
        return torch.stack(new_fc, dim=0)

    def get_logits(self, x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def subspace_projection(self, args, session):
        base_start = 0
        base_end = args.base_class
        cur_start = args.base_class + (session - 1) * args.way
        cur_end = args.base_class + session * args.way
        base_protos = F.normalize(self.fc.weight.data[base_start:base_end].detach(), p=2, dim=-1)
        cur_protos = F.normalize(self.fc.weight.data[cur_start:cur_end].detach(), p=2, dim=-1)
        BBt_inv = torch.linalg.pinv(base_protos @ base_protos.T)
        proj = (cur_protos @ base_protos.T) @ BBt_inv @ base_protos
        updated = F.normalize((1 - args.shift_weight) * cur_protos + args.shift_weight * proj, p=2, dim=-1)
        self.fc.weight.data[cur_start:cur_end] = updated.to(self.fc.weight.device)