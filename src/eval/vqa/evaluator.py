import re
import json
import string
import os

from tqdm import tqdm
import torch
from torchmetrics.text import BLEUScore
import jieba
from postprocess_utils import cleanStr
from bert_score import BERTScorer
import random
import numpy as np
import pandas as pd
import re

import pdb

class VQAEvaluator:
    def __init__(self):
        self.bleu = BLEUScore(n_gram=1)
    
    def eval_pred_list(self, pred_list, gt_list=None):
        f1 = []
        recall = []
        bleuScores = []
        closedQuestions = []
        closedQuestionsCorrect = 0
        ress, gts = [], []
        for i, entry in enumerate(pred_list):
            gt = cleanStr(entry['gt_answer'])
            res = cleanStr(entry['pred_answer'])
            
            ress.append(res)
            gts.append(gt)
            
            predictedTokens = self._preprocess(res)
            correctTokens = self._preprocess(gt)
            currentPrecision = len(predictedTokens.intersection(correctTokens)) / len(predictedTokens) if len(predictedTokens) > 0 else 0
            currentRecall = len(predictedTokens.intersection(correctTokens)) / len(correctTokens) if len(correctTokens) > 0 else 0
            currentF1 = 2 * (currentPrecision * currentRecall) / (currentPrecision + currentRecall + 1e-8)
            currentBleu = self.bleu([res], [[gt]]).item()
            bleuScores.append(currentBleu)
            f1.append(currentF1)
            recall.append(currentRecall)
            
            if gt in ["yes", "no"]:
                
                closedQuestions.append(True)
                closedQRecall = len(predictedTokens.intersection(correctTokens)) / len(correctTokens) if len(correctTokens) > 0 else 0
                if closedQRecall >= 0.8:
                    closedQuestionsCorrect += 1
            else:
                closedQuestions.append(False)
                openQuestionsRecall = []
        
        openQuestionsAccuracy = 0
        for idx, isClosed in enumerate(closedQuestions):
            if not isClosed:
                openQuestionsRecall.append(recall[idx])
                if recall[idx] >= 0.75:
                    openQuestionsAccuracy += 1
        
        bertscorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=False,
            idf=True,
            idf_sents=ress,
        )
        
        P, R, f1_bertscore = bertscorer.score(ress, gts)
        f1_bertscore = f1_bertscore.tolist()
        f1_bertscore = sum(f1_bertscore) / len(f1_bertscore)
        

        metrics = {
            "bleu1": sum(bleuScores) / len(bleuScores) if len(bleuScores) > 0 else 0,
            # "bleu1": bleu1,
            "F1": sum(f1) / len(f1) if len(f1) > 0 else 0,
            "recall": sum(recall) / len(recall) if len(recall) > 0 else 0,
            "closedQuestionsAccuracy": closedQuestionsCorrect / sum(closedQuestions) if sum(closedQuestions) > 0 else 0,
            "openQuestionsRecall": (
                sum(openQuestionsRecall) / len(openQuestionsRecall) if len(openQuestionsRecall) > 0 else 0
            ),
            "openQuestionsAccuracy": (
                openQuestionsAccuracy / len(openQuestionsRecall) if len(openQuestionsRecall) > 0 else 0
            ),
            'bertScore': f1_bertscore
        }
        
        return metrics
    
    
    def _preprocess(self, text):
        tokenizedText = set(text.split())
        tokenizedText.discard("")
        return tokenizedText

class VQAEvaluatorChinese:
    def __init__(self):
        self.bleu = BLEUScore(n_gram=1)
    
    def eval_pred_list(self, pred_list):
        f1 = []
        recall = []
        bleuScores = []
        closedQuestions = []
        closedQuestionsCorrect = 0
        ress, gts = [], []
        res_s, gt_s = [], []
        for entry in pred_list:
            gt = entry['gt_answer']
            res = entry['pred_answer']
            

            res_s.append(res)
            gt_s.append(gt)
            ress.append(' '.join(jieba.cut(res)))
            gts.append(' '.join(jieba.cut(gt)))
            predictedTokens = self._preprocess(res)
            correctTokens = self._preprocess(gt)
            currentPrecision = len(predictedTokens.intersection(correctTokens)) / len(predictedTokens) if len(predictedTokens) > 0 else 0
            currentRecall = len(predictedTokens.intersection(correctTokens)) / len(correctTokens) if len(correctTokens) > 0 else 0
            currentF1 = 2 * (currentPrecision * currentRecall) / (currentPrecision + currentRecall + 1e-8)
            currentBleu = self.bleu([' '.join(jieba.cut(res))], [[' '.join(jieba.cut(gt))]]).item()
            bleuScores.append(currentBleu)
            f1.append(currentF1)
            recall.append(currentRecall)
            
            if res in ["yes", "no"]:
                closedQuestions.append(True)

                if correctTokens == {"no"}:
                    correctTokens.add("not")

                closedQRecall = len(predictedTokens.intersection(correctTokens)) / len(correctTokens) if len(correctTokens) > 0 else 0
                if closedQRecall >= 0.4:
                    closedQuestionsCorrect += 1
            else:
                closedQuestions.append(False)
                openQuestionsRecall = []

        openQuestionsAccuracy = 0
        for idx, isClosed in enumerate(closedQuestions):
            if not isClosed:
                openQuestionsRecall.append(recall[idx])
                if recall[idx] >= 0.75:
                    openQuestionsAccuracy += 1
                    
        f1_bertscore= self.compute_bertscore(res_s, gt_s, rescale=False)
        
        f1_bertscore = f1_bertscore.tolist()
        f1_bertscore = sum(f1_bertscore) / len(f1_bertscore)
        
        metrics = {
            "bleu1": sum(bleuScores) / len(bleuScores) if len(bleuScores) > 0 else 0,
            "F1": sum(f1) / len(f1) if len(f1) > 0 else 0,
            "recall": sum(recall) / len(recall) if len(recall) > 0 else 0,
            "closedQuestionsAccuracy": closedQuestionsCorrect / sum(closedQuestions) if sum(closedQuestions) > 0 else 0,
            "openQuestionsRecall": (
                sum(openQuestionsRecall) / len(openQuestionsRecall) if len(openQuestionsRecall) > 0 else 0
            ),
            "openQuestionsAccuracy": (
                openQuestionsAccuracy / len(openQuestionsRecall) if len(openQuestionsRecall) > 0 else 0
            ),
            "BertScore": f1_bertscore
        }
        
        return metrics
    
    
    def _preprocess(self, text):
        tokenizedText = set(jieba.cut(text))
        tokenizedText.discard("")
        return tokenizedText
    
    def compute_bertscore(self, ress, gts, rescale=True):
        scorer = BERTScorer(
            model_type="bert-base-multilingual-cased",
            lang="zh",
            rescale_with_baseline=rescale,
            idf=True,
            idf_sents=ress,
            device='cuda:1'
        )

        P, R, f1_bertscore = scorer.score(ress, gts)
        return f1_bertscore

    