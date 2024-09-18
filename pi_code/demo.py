import os
import sys
import time
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
import argparse
from tqdm import tqdm
import numpy as np
from importlib import import_module
#
import torch
from torch.utils.data import DataLoader
#
from loader import Loader
from utils.logger import Logger
from utils.utils import AverageMeterForDict

def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="val", type=str, help="Mode, train/val/test")
    parser.add_argument("--features_dir", required=True, default="", type=str, help="Path to the dataset")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")
    parser.add_argument("--adv_cfg_path", required=True, default="", type=str)
    parser.add_argument("--model_path", required=False, type=str, help="path to the saved model")
    #
    parser.add_argument("--seq_id", default=-1, type=int, help="Selected sequence ID")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle order")
    parser.add_argument("--visualizer", default="", type=str, help="Type of visualizer")
    parser.add_argument("--show_conditioned", action="store_true", help="Show missed sample only")
    return parser.parse_args()

def step(net, data, evaluator, vis, args):
    data_in = net.pre_process(data)
    out = net(data_in)
    post_out = net.post_process(out)

    torch.cuda.synchronize()

    eval_out = evaluator.evaluate(post_out, data)


    print(f'\n\nSequence ID: {data["SEQ_ID"][0]}')
    print(f'Evaluation Metrics:')
    for name, val in eval_out.items():
        print('-- {}: {:.4}'.format(name, val))

    vis.draw_once(post_out, data, eval_out, show_map=True, split=args.mode)


def main():
    args = parse_arguments()
    print('Args: {}\n'.format(args))
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')
        raise Exception("CUDA is not available")
    
    vis_file, vis_name = args.visualizer.split(':')
    print('[Loader] load visualizer {} from {}'.format(vis_name, vis_file))
    vis = getattr(import_module(vis_file), vis_name)()
    
    loader = Loader(args, device, is_ddp=False)
    print('[Resume] Loading state_dict from {}'.format(args.model_path))
    loader.set_resmue(args.model_path)
    (train_set, val_set), net, _, _, evaluator = loader.load()
    net.eval()
    dataloader = DataLoader(val_set,
                                    batch_size=1,
                                    shuffle=args.shuffle,
                                    num_workers=0,
                                    collate_fn=val_set.collate_fn,
                                    drop_last=False)
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            if args.seq_id == -1:
                step(net, data, evaluator, vis, args)
            else:
                if args.seq_id == data['SEQ_ID'][0]:
                    step(net, data, evaluator, vis, args)
                    
                    
            
    print('\nExit...')
    
    

if __name__ == "__main__":
    main()
