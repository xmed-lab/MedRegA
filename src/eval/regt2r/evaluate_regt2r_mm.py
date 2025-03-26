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
from evaluator_reg import MultiRegionMultiObjectEvaluator
import pdb

PATTERN = re.compile(r'\[([^\[\]]*)\]')
PATTERN_TARGET = re.compile(r'<box>(.*?)</box>', re.DOTALL)
BBOX_SCALE = 999


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    hws = [_['hw'] for _ in batches]
    return pixel_values, questions, question_ids, annotations, hws


def evaluate_chat_model(ans_file_list):
    random.seed(args.seed)
    summaries = []
    
    merged_outputs = []
    ans_files = ans_file_list.split(',')
    for ans_file in ans_files:
        with open(ans_file, 'r') as file:
            merged_outputs.extend(json.load(file))
            
    ds_name = '_'.join(os.path.split(ans_file)[-1].split('_')[0:2])
    ds_task = ds_name.split('_')[-1]
    evaluator = MultiRegionMultiObjectEvaluator()

    metrics = evaluator.eval_pred_list(merged_outputs)
    
    print(ans_file, metrics)
    summaries.append([args.checkpoint, ans_file, metrics])
    
    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    # print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        # print(summary)
        writer.write(f'{summary}\n')
    writer.close()
    
    # for summary in summaries:
    #     print(summary)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="ours")
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

    # ans_files = []
    # all_jsons = os.listdir('/scratch/uusoundlfm/lhwang/internvlv1-2_test_2')
    # for j in all_jsons:
    #     ds_name = '_'.join(j.split('_')[0:2])
    #     ds_task = ds_name.split('_')[-1]
    #     if ds_task == 'regt2r':
    #         ans_files.append(os.path.join('/scratch/uusoundlfm/lhwang/internvlv1-2_test_2', j))
    # args.ans_file = ','.join(ans_files)

    evaluate_chat_model(args.ans_file)
    