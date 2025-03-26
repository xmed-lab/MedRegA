import argparse
import itertools
import json
import os
import string
import random
import subprocess
import time
from functools import partial
from typing import Optional

import torch
from PIL import Image
from evaluator import ClassificationEvaluator
from tqdm import tqdm
from transformers import AutoTokenizer
from postprocess_utils import unify_comma
import numpy as np

import pdb

class_list_dict = {
    'bhx_test': "Epidural\nSubdural\nChronic\nIntraventricular\nIntraparenchymal\nSubarachnoid\nNone",
    'brset_test': "normal\nhypertensive retinopathy\ndiabetic macular edema\ndrusens\nincreased cup disc ratio\nnondiabetic retinal hemorrhage\nvascular occlusion\nage-related macular degeneration\nother\nmyopic fundus\ndiabetic retinopathy\nretinal detachment\nscar\nnevus",
    'chexpert_test': "Enlarged Cardiomediastinum\nCardiomegaly\nLung Opacity\nLung Lesion\nEdema\nConsolidation\nPneumonia\nAtelectasis\nPneumothorax\nPleural Effusion\nPleural Other\nFracture\nSupport Devices\nNo Finding",
    'cxr14_test': "Atelectasis\nCardiomegaly\nConsolidation\nEdema\nEffusion\nEmphysema\nFibrosis\nHernia\nInfiltration\nMass\nNodule\nPleural Thickening\nPneumonia\nPneumothorax\nPneumoperitoneum\nPneumomediastinum\nSubcutaneous Emphysema\nTortuous Aorta\nCalcification of the Aorta\nNo Finding",
    'isic2016_test': "benign\nmelanoma",
    'isic2017_test': "melanoma\nseborrheic keratosis\nnevous",
    'isic2018_test': "Melanoma\nMelanocytic nevus\nBasal cell carcinoma\nBowen's disease\nBenign keratosis\nDermatofibroma\nVascular lesion",
    'isic2019_test': "Melanoma\nMelanocytic nevus\nBasal cell carcinoma\nActinic keratosis\nBenign keratosis\nDermatofibroma\nVascular lesion\nSquamous cell carcinoma",
    'mura_test': "normal\nabnormal",
    'oct2017_test': "normal\ndrusen\ndiabetic macular edema\nchoroidal neovascularization",
    'organamnist_test': "bladder\nleft femur\nright femur\nheart\nleft kidney\nright kidney\nliver\nleft lung\nright lung\npancreas\nspleen",
    'organcmnist_test': "bladder\nleft femur\nright femur\nheart\nleft kidney\nright kidney\nliver\nleft lung\nright lung\npancreas\nspleen",
    'organsmnist_test': "bladder\nleft femur\nright femur\nheart\nleft kidney\nright kidney\nliver\nleft lung\nright lung\npancreas\nspleen",
    'padufes_test': "Basal Cell Carcinoma\nSquamous Cell Carcinoma\nActinic Keratosis\nSeborrheic Keratosis\nMelanoma\nNevus",
    'pannuke_test': "Breast\nColon\nBile duct\nEsophagus\nUterus\nLung\nCervix\nHead or Neck\nSkin\nAdrenal gland\nKidney\nStomach\nProstate\nTestis\nLiver\nThyroid\nPancreatic\nOvarian\nBladder",
    'kathercolon_test': "adipose\nbackground\ndebris\nlymphocytes\nmucus\nsmooth muscle\nnormal colon mucosa\ncancer-associated stroma\ncolorectal adenocarcinoma epithelium",
    'rfmid_test': "Optic disc pallor\nChorioretinitis\nMacular hole\nLaser scars\nAge-related macular degeneration\nRetinal pigment epithelium changes\nRetinitis pigmentosa\nOptic disc edema\nAnterior ischemic optic neuropathy\nDrusens\nMyopia\nParafoveal telangiectasia\nRetinitis\nExudation\nTortuous vessels\nAsteroid hyalosis\nEpiretinal membrane\nDiabetic retinopathy\nOptic disc cupping\nMedia Haze\nTessellation\nOptociliary shunt\nCentral serous retinopathy\nBranch retinal vein occlusion\nCentral retinal vein occlusion\nMacular scar\nRetinal traction\nnormal\nOther",
    'vindrcxr_test': "Aortic enlargement\nAtelectasis\nCalcification\nCardiomegaly\nClavicle fracture\nConsolidation\nEdema\nEmphysema\nEnlarged PA\nILD\nInfiltration\nLung Opacity\nLung cavity\nLung cyst\nMediastinal shift\nNodule or Mass\nPleural effusion\nPleural thickening\nPneumothorax\nPulmonary fibrosis\nRib fracture\nOther lesion\nCOPD\nLung tumor\nPneumonia\nTuberculosis\nNo finding\nOther disease",
    'vindrmammo_test': "Suspicious Calcification\nNipple Retraction\nSkin Retraction\nArchitectural Distortion\nGlobal Asymmetry\nFocal Asymmetry\nSuspicious Lymph Node\nAsymmetry\nMass\nSkin Thickening\nNo Finding",
    'vindrpcxr_test': "Bronchitis\nBrocho-pneumonia\nBronchiolitis\nSitus inversus\nPneumonia\nPleuro-pneumonia\nDiagphramatic hernia\nTuberculosis\nCongenital emphysema\nCPAM\nHyaline membrane disease\nMediastinal tumor\nLung tumor\nOther disease\nNo finding",
    'vindrspinexr_test': "Surgical implant\nForaminal stenosis\nVertebral collapse\nDisc space narrowing\nOsteophytes\nSpondylolysthesis\nOther lesions\nNo finding",
    'busi_test': "benign\nmalignant\nnormal",
    'butterfly_test': "Morisons pouch\nBladder\nPLAX view of the heart\n4 chambers of the heart\n2 chambers of the heart\nIVC\nCarotid artery\nLungs\nThyroid",
    'ultracovid_test': "pneumonia\nregular\ncovid19",
    'messidor2_test': "None\nDR",
    'breastmnist_test': 'malignant\nnormal or benign',
    'chestmnist_test': 'Atelectasis\nCardiomegaly\nConsolidation\nEdema\nEffusion\nEmphysema\nFibrosis\nHernia\nInfiltration\nMass\nNodule\nPleural Thickening\nPneumonia\nPneumothorax\nNo Finding',
    'octmnist_test': 'choroidal neovascularization\ndiabetic macular edema\ndrusen\nnormal',
    'pneumoniamnist_test': 'normal\npneumonia',
}

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
    response = unify_comma(response)
    response = response.rstrip(string.punctuation)
    return response


def evaluate_chat_model(ans_file=None):    
    random.seed(args.seed)
    summaries = []
    
    with open(ans_file, 'r') as file:
        merged_outputs = json.load(file)
    ds_name = '_'.join(os.path.split(ans_file)[-1].split('_')[0:2])  # get the dataset name
    
    if ds_name in ['chestmnist_test', 'brset_test', 'chexpert_test', 'cxr14_test', 'rfmid_test', 'vindrcxr_test','vindrmammo_test','vindrpcxr_test','vindrspinexr_test']:
        score_type = 'multilabel'
    else:
        score_type = 'multiclass'
        
    vqa_evaluator = ClassificationEvaluator(score_type)
    
    metrics = (vqa_evaluator.eval_pred_list(merged_outputs, class_list_dict[ds_name].split('\n')))
    
    print(ds_name, metrics)
    summaries.append([args.checkpoint, ds_name, metrics])

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out-dir', type=str, default='')
    parser.add_argument('--few-shot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--ans_file', default='')
    args = parser.parse_args()
 
    # vindrcxr, vindrpcxr, rfmid
    
    for ans_file in args.ans_file.split(','):
        evaluate_chat_model(ans_file)
