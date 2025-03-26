import argparse
import itertools
import json
import os
import random
import subprocess
import time
from functools import partial
from typing import Optional

import torch
import numpy as np
from PIL import Image
from evaluator_rg import ReportEvaluator, ReportEvaluatorChinese
from tqdm import tqdm
from transformers import AutoTokenizer
from evaluator_rg import CompositeMetric
from postprocess_utils import cleanStr
import re

import nltk
nltk.download('punkt_tab')

# os.environ["CUDA_VISIBLE_DEVICES"] = '4'


# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None

    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float -
                              target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()


def evaluate_relaxed_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            relaxed_correctness(elem['answer'].strip(), ann)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def evaluate_exact_match_accuracy(entries):
    scores = []
    for elem in entries:
        if isinstance(elem['annotation'], str):
            elem['annotation'] = [elem['annotation']]
        score = max([
            (1.0 if
             (elem['answer'].strip().lower() == ann.strip().lower()) else 0.0)
            for ann in elem['annotation']
        ])
        scores.append(score)
    return sum(scores) / len(scores)


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    questions = [_['question'] for _ in batches]
    question_ids = [_['question_id'] for _ in batches]
    annotations = [_['annotation'] for _ in batches]
    exists = [_['exists'] for _ in batches]

    return pixel_values, questions, question_ids, annotations, exists



def post_process(response):
    response = response.replace('\n', ' ')
    response = cleanStr(response)
    return response


def evaluate_chat_model(ans_file=None):
    random.seed(args.seed)
    summaries = []
    
    with open(ans_file, 'r') as file:
        merged_outputs = json.load(file)
    evaluator = ReportEvaluator()
    
    with open(ans_file, 'r') as file:
        merged_outputs = json.load(file)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--out-dir', type=str, default='')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--ans_file', default='')
    args = parser.parse_args()

    evaluate_chat_model(args.ans_file)
