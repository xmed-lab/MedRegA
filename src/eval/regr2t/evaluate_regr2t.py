import argparse
import itertools
import json
import os
import random
import re
import time
from functools import partial

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
import ast
from evaluator import R2TEvaluator

import pdb

PATTERN = re.compile(r'\[([^\[\]]*)\]')
PATTERN_TARGET = re.compile(r'<box>(.*?)</box>', re.DOTALL)
BBOX_SCALE = 999


def evaluate_chat_model(ans_files):
    summaries = []
    
    ans_files = ans_files.split(',')
    
    merged_outputs = []
    for ans_file in ans_files:
        with open(ans_file, 'r') as file:
            merged_outputs.extend(json.load(file))
    evaluator = R2TEvaluator()
    metrics = evaluator.eval_pred_list(merged_outputs)
    
    print(ans_file, metrics)
    summaries.append([args.checkpoint, ans_file, metrics])
    
    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()
        
    for summary in summaries:
        print(summary)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='')
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--ans_file', default='')
    args = parser.parse_args()

    
    evaluate_chat_model(args.ans_file)