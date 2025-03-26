import re
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from torchvision.ops.boxes import box_area
from torchmetrics.text import BLEUScore
from bert_score import BERTScorer
import ast
import pdb
    
class R2TEvaluator:
    '''
    The same as VQA evaluator.
    '''
    def __init__(self):
        self.bleu = BLEUScore(n_gram=1)
    
    def eval_pred_list(self, pred_list):
        f1 = []
        recall = []
        bleuScores = []
        gts, ress = [], []
        accuracy = 0
        for entry in pred_list:
            gt = entry['gt_answer'].lower()
            res = entry['pred_answer'].lower()
                    
            res = re.sub(r'\[\[.*?\]\]', '', res)
            gts.append(gt)
            ress.append(res)
            predictedTokens = self._preprocess(res)
            correctTokens = self._preprocess(gt)
            currentPrecision = len(predictedTokens.intersection(correctTokens)) / len(predictedTokens) if len(predictedTokens) > 0 else 0
            currentRecall = len(predictedTokens.intersection(correctTokens)) / len(correctTokens) if len(correctTokens) > 0 else 0
            currentF1 = 2 * (currentPrecision * currentRecall) / (currentPrecision + currentRecall + 1e-8)
            currentBleu = self.bleu([res], [[gt]]).item()
            if currentRecall >= 0.75:
                accuracy += 1
            
            bleuScores.append(currentBleu)
            f1.append(currentF1)
            recall.append(currentRecall)
            
        bertscorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=False,
            idf=True,
            idf_sents=ress,
            device="cuda:4"
        )
        
        P, R, f1_bertscore = bertscorer.score(ress, gts)
        f1_bertscore = f1_bertscore.tolist()
        f1_bertscore = sum(f1_bertscore) / len(f1_bertscore)
        metrics = {
            "bleu1": sum(bleuScores) / len(bleuScores) if len(bleuScores) > 0 else 0,
            "F1": sum(f1) / len(f1) if len(f1) > 0 else 0,
            "recall": sum(recall) / len(recall) if len(recall) > 0 else 0,
            "accuracy": (
                accuracy / len(recall) if len(recall) > 0 else 0
            ),
            "bertscore": f1_bertscore
        }
        return metrics
        
    def _preprocess(self, text):
        tokenizedText = set(text.split()) # set([self.wnl.lemmatize(token) for token in text.split()])
        tokenizedText.discard("")
        return tokenizedText
    