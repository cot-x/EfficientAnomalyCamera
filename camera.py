#!/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm
from pickle import load, dump

import os
import cv2
import random
import datetime
import argparse


from solver import *


def main(args):
    solver = Solver(args)
    solver.load_state()
    
    if not args.noresume:
        solver = solver.load_resume()
        solver.args = args
    
    if args.generate > 0:
        solver.generate(args.generate)
        return
    
    solver.capture()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--result_dir', type=str, default='log')
    parser.add_argument('--weight_dir', type=str, default='log')
    parser.add_argument('--image_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mul_lr_dis', type=float, default=4)
    parser.add_argument('--num_scorelist', type=int, default=1000)
    parser.add_argument('--aug_threshold', type=float, default=0.6)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate', type=int, default=0)
    parser.add_argument('--noresume', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    if not os.path.exists(args.weight_dir):
        os.mkdir(args.weight_dir)
    
    main(args)