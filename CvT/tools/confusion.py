from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import time

import torch
import torch.nn.parallel
import torch.optim
from torch.utils.collect_env import get_pretty_env_info
import numpy as np

import _init_paths
from config import config
from config import update_config
from core.count_confusion import count_confusion 
from dataset import build_dataloader
from models import build_model
from utils.comm import comm
from utils.utils import create_logger
from utils.utils import init_distributed
from utils.utils import setup_cudnn
from utils.utils import summary_model_on_master
from utils.utils import strip_prefix_if_present

def parse_args():
    parser = argparse.ArgumentParser(
        description='Count confusion matrix for network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'confusion')

    logging.info("=> collecting env info (might take some time)")
    logging.info("\n" + get_pretty_env_info())
    logging.info(pprint.pformat(args))
    logging.info(config)
    
    output_matrix_path = os.path.join(final_output_dir, 'confusion_matrix.npy')

    model = build_model(config)
    model.to(torch.device('cuda'))

    model_file = config.TEST.MODEL_FILE if config.TEST.MODEL_FILE \
        else os.path.join(final_output_dir, 'model_best.pth')
    logging.info('=> load model file: {}'.format(model_file))
    ext = model_file.split('.')[-1]
    if ext == 'pth':
        state_dict = torch.load(model_file, map_location="cpu")
    else:
        raise ValueError("Unknown model file")

    model.load_state_dict(state_dict, strict=False)
    model.to(torch.device('cuda'))

    summary_model_on_master(model, config, final_output_dir, False)

    loader = build_dataloader(config, False, False)

    logging.info('=> start counting')
    start = time.time()
    confusion_matrix = count_confusion(loader, model)
    logging.info('=> counting time: {:.2f}s'.format(time.time()-start))
    np.save(output_matrix_path, confusion_matrix)

if __name__ == '__main__':
    main()
