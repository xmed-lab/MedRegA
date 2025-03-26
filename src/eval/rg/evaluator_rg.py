import os
from tqdm import tqdm
import gdown
import numpy as np
import pandas as pd
import torch
from torchmetrics.text import BLEUScore, ROUGEScore
from bert_score import BERTScorer
from chexbert.label import encode, label
from radgraph import F1RadGraph
import pickle
import evaluate
from postprocess_utils import cleanStr, cleanStr_zh
import json
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score

from rouge_chinese import Rouge
import jieba # you can use any other word cutting library

import pdb


class CompositeMetric:
    """The RadCliQ-v1 composite metric.

    Attributes:
        scaler: Input normalizer.
        coefs: Coefficients including the intercept.
    """

    def __init__(self, scaler, coefs):
        """Initializes the composite metric with a normalizer and coefficients.

        Args:
            scaler: Input normalizer.
            coefs: Coefficients including the intercept.
        """
        self.scaler = scaler
        self.coefs = coefs

    def predict(self, x):
        """Generates composite metric score for input.

        Args:
            x: Input data.

        Returns:
            Composite metric score.
        """
        norm_x = self.scaler.transform(x)
        norm_x = np.concatenate((norm_x, np.ones((norm_x.shape[0], 1))), axis=1)
        pred = norm_x @ self.coefs
        return pred


class ReportEvaluatorChinese:
    def __init__(self):
        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_4 = BLEUScore(n_gram=4)
        self.bleu_metric = evaluate.load('bleu')
        self.rouge_metric = evaluate.load('rouge')

    def eval_pred_list(self, pred_list):
        bleu1Scores = []
        # bleu2Scores = []
        bleu4Scores = []
        rougeLScores = []
        rouge1Scores = []
        rouge = Rouge()
        gts, ress = [], []
        gt_s, res_s = [], []
        
        for i, entry in enumerate(tqdm(pred_list, total=len(pred_list))):
            gt = entry['gt_answer']
            res = cleanStr_zh(entry['pred_answer']).replace(' ', '')
            
            gts.append(gt)
            ress.append(res)
            gt_s.append(' '.join(jieba.cut(gt)))
            res_s.append(' '.join(jieba.cut(res)))
            
            try:
                bleu1 = self.bleu_metric.compute(predictions=[' '.join(jieba.cut(res))], references=[[' '.join(jieba.cut(gt))]], max_order=1)['bleu']
                bleu1Scores.append(bleu1)
                if bleu1 > 0.5:
                    print(entry)
                bleu4Scores.append(self.bleu_metric.compute(predictions=[' '.join(jieba.cut(res))], references=[[' '.join(jieba.cut(gt))]], max_order=4)['bleu'])
            except:
                bleu1Scores.append(0)
                bleu4Scores.append(0)
            
            try:
                rouge_ = self.rouge_metric.compute(predictions=[' '.join(jieba.cut(res))], references=[[' '.join(jieba.cut(gt))]])
                rougeLScores.append(rouge_['rouge1'])
                rouge1Scores.append(rouge_['rougeL'])
            except:
                rougeLScores.append(0)
                rouge1Scores.append(0)

        f1_bertscore = self.compute_bertscore(ress, gts, rescale=False)
        meteor_scores = self.compute_meteor(ress, gts)

        f1_bertscore = f1_bertscore.tolist()

        metrics = {
            "bleu1": sum(bleu1Scores) / len(bleu1Scores),
            "bleu4": sum(bleu4Scores) / len(bleu4Scores),
            "F1_bertscore": sum(f1_bertscore) / len(f1_bertscore),
            "meteor": sum(meteor_scores) / len(meteor_scores),
            "rouge1": sum(rouge1Scores) / len(rouge1Scores),
            "rougeL": sum(rougeLScores) / len(rougeLScores),
        }
        
        return metrics

    def compute_bertscore(self, ress, gts, rescale=True):
        scorer = BERTScorer(
            model_type="bert-base-multilingual-cased",
            lang="zh",
            rescale_with_baseline=rescale,
            idf=True,
            idf_sents=ress,
            device='cuda:5'
        )

        P, R, f1_bertscore = scorer.score(ress, gts)
        return f1_bertscore
    
    def compute_meteor(self, ress, gts):
        meteor_scores = []
        for gt, res in zip(gts, ress):
            # Tokenize the reference and hypothesis
            ref_tokens = list(jieba.cut(gt))
            hyp_tokens = list(jieba.cut(res))
            # Compute the meteor score
            meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))

        return meteor_scores



class ReportEvaluator:
    def __init__(self, chexbert_path='rg/chexbert', radcliq_path='rg/radcliq-v1.pkl'):
        self.bleu_1 = BLEUScore(n_gram=1)
        self.bleu_2 = BLEUScore(n_gram=2, weights=[1 / 2, 1 / 2])
        self.bleu_4 = BLEUScore(n_gram=4)
        self.rougeL = ROUGEScore(rouge_keys=("rougeL", "rouge1"))
        self.chexbert_path = chexbert_path
        self.radcliq_path = radcliq_path
        
        self.bleu_metric = evaluate.load('bleu')
        self.rouge_metric = evaluate.load('rouge')

    def eval_pred_list(self, pred_list):
        bleu2Scores = []
        gts, ress = [], []
        
        for entry in tqdm(pred_list, total=len(pred_list)):
            gt = cleanStr(entry['gt_answer'])
            res = cleanStr(entry['pred_answer'])

            gts.append(gt)
            ress.append(res)

            try:
                bleu2Scores.append(self.bleu_metric.compute(predictions=[res], references=[[gt]], max_order=2)['bleu'])
            except:
                bleu2Scores.append(0)

        
        bleu1_ = self.bleu_metric.compute(predictions=ress, references=gts, max_order=1)  # default: max_order=4 (n_gram)
        bleu1 = bleu1_['bleu']
        bleu4_ = self.bleu_metric.compute(predictions=ress, references=gts, max_order=4)  # default: max_order=4 (n_gram)
        bleu4 = bleu4_['bleu']

        rouge_ = self.rouge_metric.compute(predictions=ress, references=gts)
        rouge1 = rouge_['rouge1']
        rougel = rouge_['rougeL']
        meteor_scores = self.compute_meteor(ress, gts)
        
        f1_bertscore = self.compute_bertscore(ress, gts, rescale=False)
        self._preprare_chexbert()
        chexbert_similarity = self.compute_chexbert(ress,gts)
        self._prepare_radgraph()
        f1_radgraph = self.compute_radgraph(ress, gts)
        radcliq_scores = self.compute_radcliq(torch.tensor(bleu2Scores), f1_bertscore, chexbert_similarity, f1_radgraph)
        
        f1_bertscore = f1_bertscore.tolist()
        chexbert_similarity = chexbert_similarity.tolist()
        f1_radgraph = f1_radgraph.tolist()
        radcliq_scores = radcliq_scores.tolist()
        
        metrics = {
            "bleu1": bleu1,
            "bleu4": bleu4,
            "meteor": sum(meteor_scores) / len(meteor_scores),
            "rouge1": rouge1,
            "rougeL": rougel,
        }
        
        return metrics
        
    
    def _preprare_chexbert(self):
        chexbert_checkpoint = os.path.join(self.chexbert_path, 'chexbert.pth')
        if not os.path.exists(chexbert_checkpoint):
            os.makedirs(self.chexbert_path, exist_ok=True)
            gdown.download(
                "https://stanfordmedicine.app.box.com/shared/static/c3stck6w6dol3h36grdc97xoydzxd7w9",
                chexbert_checkpoint,
                quiet=False,
            )
        # Check if deepspeed is installed and initialized
        # try:
        #     from deepspeed.comm.comm import is_initialized

        #     # Test if deepspeed is initialized
        #     if not is_initialized():
        #         raise Exception("Deepspeed is not initialized.")

        #     deepspeedEnabled = True
        # except:
        #     deepspeedEnabled = False

        self.encoder = encode(chexbert_checkpoint, verbose=False)
        self.labeler = label(chexbert_checkpoint, verbose=False)
    
    def _prepare_radgraph(self):
        # Check if deepspeed is installed and initialized
        # try:
        #     from deepspeed.comm.comm import is_initialized

        #     # Test if deepspeed is initialized
        #     if not is_initialized():
        #         raise Exception("Deepspeed is not initialized.")
        # except:
        #     pass
        # else:
        #     raise Exception("Deepspeed is initialized.")

        self.radgraph = F1RadGraph(reward_level="partial", cuda=5, model_type="radgraph")

    
    def compute_bertscore(self, ress, gts, rescale=True):
        scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            lang="en",
            rescale_with_baseline=rescale,
            idf=True,
            idf_sents=ress,
            device='cuda:5'
        )

        P, R, f1_bertscore = scorer.score(ress, gts)
        return f1_bertscore
    
    def compute_chexbert(self, ress, gts):
        df = pd.DataFrame(columns=["Report Impression"], data=gts)
        labelsReference = self.encoder(df)

        df = pd.DataFrame(columns=["Report Impression"], data=ress)
        labelsHypothesis = self.encoder(df)

        # Compute the vector similarity between the reference and the geenrated reports
        return torch.cosine_similarity(labelsReference, labelsHypothesis)

    def compute_radgraph(self, ress, gts):
        f1_radgraph = []
        for res, gt in zip(ress, gts):
            (_, _, hyp_annotation_lists, ref_annotation_lists) = self.radgraph(refs=[gt], hyps=[res])
            if len(hyp_annotation_lists) > 0 and len(ref_annotation_lists) > 0:
                f1_radgraph.append(
                    self.exact_entity_token_if_rel_exists_reward(hyp_annotation_lists[0], ref_annotation_lists[0])
                )
            else:
                f1_radgraph.append(0.0)
                
        return torch.tensor(f1_radgraph)
    
    def compute_meteor(self, ress, gts):
        meteor_scores = []
        for gt, res in zip(gts, ress):
            # Tokenize the reference and hypothesis
            ref_tokens = word_tokenize(gt)
            hyp_tokens = word_tokenize(res)

            # Compute the meteor score
            meteor_scores.append(meteor_score([ref_tokens], hyp_tokens))

        return meteor_scores

    def compute_radcliq(self, bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph):
        # Get the current path to the module
        # with open(self.radcliq_path, "rb") as f:
        #     composite_metric_v0_model: CompositeMetric = dill.load(f)
        with open(self.radcliq_path, "rb") as f:
            composite_metric_v0_model = pickle.load(f)

        # The column need to be in the order [bleu, bertscore, chexbert, radgraph]
        input_data = torch.stack(
            [bleu_scores, f1_bertscore, chexbert_similarity, f1_radgraph],
            dim=1,
        )
        return composite_metric_v0_model.predict(input_data)
    
    def exact_entity_token_if_rel_exists_reward(self, hypothesis_annotation_list, reference_annotation_list):
        candidates = []
        for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
            candidate = []
            for entity in annotation_list["entities"].values():
                if not entity["relations"]:
                    candidate.append((entity["tokens"], entity["label"]))
                if entity["relations"]:
                    candidate.append((entity["tokens"], entity["label"], True))

            candidate = set(candidate)
            candidates.append(candidate)

        hypothesis_relation_token_list, reference_relation_token_list = candidates

        precision = (
            sum([1 for x in hypothesis_relation_token_list if (x in reference_relation_token_list)])
            / len(hypothesis_relation_token_list)
            if len(hypothesis_relation_token_list) > 0
            else 0.0
        )
        recall = (
            sum([1 for x in reference_relation_token_list if (x in hypothesis_relation_token_list)])
            / len(reference_relation_token_list)
            if len(reference_relation_token_list) > 0
            else 0.0
        )
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return f1_score
