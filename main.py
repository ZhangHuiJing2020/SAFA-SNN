import json
import argparse
import importlib
from utils import *

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./jsons/safa.json',
                        help='Json file of settings.')

    return parser

class Args:
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)
if __name__ == '__main__':
    args = setup_parser().parse_args()
    with open(args.config) as data_file:
        param = json.load(data_file)
    args = vars(args)
    args.update(param)
    args.update(args["config"][args["dataset"]])
    args = Args(args)
    print(args)

    from dataloader.data_utils import set_up_datasets
    args = set_up_datasets(args)
    set_seed(args.seed)
    args.class_name = args.project.upper()
    trainer = getattr(importlib.import_module('methods.%s' % args.project), args.class_name)(args)
    trainer.train()