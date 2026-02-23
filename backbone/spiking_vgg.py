import random
from backbone.sparsity_neuron import *
import tool


class VGG9SNN(nn.Module):
    def __init__(self, time_step=4):
        super(VGG9SNN, self).__init__()
        args = tool.args
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        if tool.args.dataset in ['cifar100', 'mini_imagenet']:
            in_channel = 3
        else:
            in_channel = 2
        self.features = nn.Sequential(
            Layer(in_channel, 64, 3, 1, 1),
            Layer(64, 64, 3, 1, 1),
            pool,
            Layer(64, 128, 3, 1, 1),
            Layer(128, 128, 3, 1, 1),
            pool,
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
            Layer(256, 256, 3, 1, 1),
            pool
        )
        if args.dataset == 'cifar100':
            H, W = 32, 32
        elif args.dataset == 'mini_imagenet':
            Hsize, Wsize = 84, 84
        elif args.dataset in ['cifar10dvs', 'dvs128gesture']:
            Hsize, Wsize = 128, 128
        elif args.dataset == 'n_caltech101':
            Hsize, Wsize = 180, 240
        H = int(H/ 2 / 2 / 2)
        W = int(W/ 2 / 2 / 2)
        self.T = time_step
        self.adapt_ratio = args.adapt_ratio
        self.classifier = SeqToANNContainer(nn.Linear(256 * H * W, 1024))
        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input, session=0, args=None):
        if tool.args.dataset not in ['cifar10dvs', 'dvs128gesture', 'n_caltech101']:
            input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGSNNwoAP(nn.Module):
    def __init__(self):
        super(VGGSNNwoAP, self).__init__()
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            Layer(64, 128, 3, 2, 1),
            Layer(128, 256, 3, 1, 1),
            Layer(256, 256, 3, 2, 1),
            Layer(256, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
            Layer(512, 512, 3, 1, 1),
            Layer(512, 512, 3, 2, 1),
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 4
        self.classifier = SeqToANNContainer(nn.Linear(512 * W * W, 10))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        # x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = VGGSNNwoAP()
