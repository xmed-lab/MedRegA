import re
import json
import string
import os

from tqdm import tqdm
import torch
from torchmetrics import F1Score, Accuracy, AUROC, BLEUScore
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np

import pdb

class ClassificationEvaluator:
    def __init__(self, score_type, average_type='macro'):
        self.score_type = score_type
        self.average_type = average_type
    
    def eval_pred_list(self, pred_list, class_list):
        predictions = []
        ground_truth = []
        
        all_outputs = []
        
        num_classes = len(class_list)
        
        self._set_scorers(num_classes)

        for entry in pred_list:

            gt = entry['gt_answer'].rstrip(string.punctuation).replace(', ', ',')
            res = entry['pred_answer'].rstrip(string.punctuation).replace(', ', ',').lower()
            
            if self.score_type == 'multiclass':
                gt_idx = self._get_label_id(gt, class_list)
                res_idx = self._get_answer_id(res, class_list)[0]
            elif self.score_type == 'multilabel':
                gt_idx = [self._get_label_id(g, class_list) for g in gt.split(',')]
                res_idx = self._get_answer_id(res, class_list)
            predictions.append(res_idx)
            ground_truth.append(gt_idx)
        if self.score_type == 'multilabel':
            predictions = self._convert_to_onehot_torch(predictions, num_classes)
            predictions = predictions.to(torch.float32)
            ground_truth = self._convert_to_onehot_torch(ground_truth, num_classes)
            ground_truth = ground_truth.to(torch.long)
        

        if self.score_type == 'multiclass':
            predictions = np.array(predictions)
            ground_truth = np.array(ground_truth)
        elif self.score_type == 'multilabel':
            predictions = predictions.numpy()
            ground_truth = ground_truth.numpy()
        f1 = f1_score(ground_truth, predictions, average='macro', zero_division=0)
        acc = accuracy_score(predictions, ground_truth)

        metrics = {"F1-macro": f1, "Accuracy": acc}
        return metrics
        
    def _set_scorers(self, num_classes):
        self.scorerArgs = {"task": self.score_type, 'average': self.average_type}
        self.scorerArgs.update(
            {"num_classes": num_classes} if self.score_type == "multiclass" else {"num_labels": num_classes}
        )
        self.f1Scorer = F1Score(**self.scorerArgs)
        self.aucScorer = AUROC(**self.scorerArgs)
        self.bleu = BLEUScore(n_gram=1)
        if self.score_type == "multiclass":
            self.accScorer = Accuracy(**self.scorerArgs)
        else:
            self.accscorerArgs = {"task": 'multiclass', 'average': self.average_type, "num_classes": 2}
            self.accScorer = Accuracy(**self.accscorerArgs)
    
    
    def _get_label_id(self, target, class_list):
        return class_list.index(target)

    
    def _get_answer_id(self, answer, options):
    
        answer = answer.lower()
        answer_list = answer.split(',')
        ansid_list = []
        for ans in answer_list:
            scores = [self.bleu([ans], [[option.lower()]]) for option in options]
            ansid = scores.index(max(scores))
            ansid_list.append(ansid)
        return ansid_list
    
    def _convert_to_onehot_torch(self, data, num_classes):
        onehot_encoded = torch.zeros(len(data), num_classes)
        
        for i, indices in enumerate(data):
            if indices:
                indices_tensor = torch.tensor(indices, dtype=torch.long)
                onehot_encoded[i].scatter_(0, indices_tensor, 1)
                    
        return onehot_encoded
    
