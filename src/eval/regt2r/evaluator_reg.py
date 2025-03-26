import re
import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
from torchvision.ops.boxes import box_area
from torchmetrics.text import BLEUScore
from bert_score import BERTScorer
import ast
import pdb
from sklearn.metrics import auc
# from postprocess_utils import cleanStr
import sys
sys.path.append("test_mllm/rg") 
from evaluator_rg import ReportEvaluatorChinese, ReportEvaluator

PATTERN = re.compile(r'(\[\[\d+,\s*\d+,\s*\d+,\s*\d+\](?:,\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\])*\])')
PATTERN_TARGET = re.compile(r'<box>(.*?)</box>')


PATTERN_REGCAP = re.compile(r'([^\[\]]*)(\[\[\d+,\s*\d+,\s*\d+,\s*\d+\](?:,\s*\[\d+,\s*\d+,\s*\d+,\s*\d+\])*\])')
PATTERN_TARGET_RECAP = re.compile(r'<ref>(.*?)</ref><box>(.*?)</box>')
BBOX_SCALE = 999



class SingleRegionSingleObjectEvaluator:
    def __init__(self):
        pass
    
    def eval_pred_list(self, pred_list):
        correct_all_25, correct_all_50, correct_all_75 = 0, 0, 0
        ious = []
        for i, output in enumerate(pred_list):   
            hw = output['hw']
            all_boxes_pred, all_boxes_target = [], []

            try:
                predict_bboxes_str = re.findall(PATTERN_REGCAP, output['answer'])
                for predict_bbox_str in predict_bboxes_str:
                    predict_object = predict_bbox_str[0].lower()
                    predict_regions = predict_bbox_str[1]
                    predict_bbox = [self.process_box(bbox, hw[1], hw[0]) for bbox in ast.literal_eval(predict_regions)]
                    
                    all_boxes_pred.extend(predict_bbox)
            except:
                predict_bbox = [[0, 0, 0, 0]]
                all_boxes_pred.extend(predict_bbox)
            target_bboxes_str = re.findall(PATTERN_TARGET_RECAP, output['gt'])

            for target_bbox_str in target_bboxes_str:
                target_object = target_bbox_str[0].lower()
                target_regions = target_bbox_str[1]
                target_bbox = [self.process_box(bbox, hw[1], hw[0]) for bbox in ast.literal_eval(target_regions)]
                
                all_boxes_target.extend(target_bbox)
                
                
            ## all boxes
            all_boxes_target = torch.tensor(all_boxes_target, dtype=torch.float32).view(-1, 4)
            all_boxes_pred = torch.tensor(all_boxes_pred, dtype=torch.float32).view(-1, 4)
            total_cnt_all = all_boxes_target.shape[0]
            pred_cnt_all = all_boxes_pred.shape[0]
            
            max_iou = 0
            if pred_cnt_all > 0:
                if pred_cnt_all > total_cnt_all:
                    for p in all_boxes_pred:
                        cur_iou = self.box_iou(torch.tensor([p.detach().numpy().tolist()]), torch.tensor([all_boxes_target[0].detach().numpy().tolist()]))
                        cur_iou = cur_iou[0].detach().numpy().tolist()[0][0]
                        if cur_iou > max_iou:
                            max_iou = cur_iou
                else:
                    max_iou = self.box_iou(torch.tensor([all_boxes_pred[0].detach().numpy().tolist()]), torch.tensor([all_boxes_target[0].detach().numpy().tolist()]))
                    max_iou = max_iou[0].detach().numpy().tolist()[0][0]
            
            ious.append(max_iou)
            if max_iou > 0.25:
                correct_all_25 += 1
            if max_iou  > 0.5:
                correct_all_50 += 1
            if max_iou  > 0.75:
                correct_all_75 += 1

        metrics = {
                'acc25': '{:.4f}'.format(correct_all_25 / len(pred_list)),
                'acc50': '{:.4f}'.format(correct_all_50 / len(pred_list)),
                'acc75': '{:.4f}'.format(correct_all_75 / len(pred_list)),
                'iou': '{:.4f}'.format(sum(ious) / len(ious))
            }

        return metrics
    
    def process_box(self, bbox, width, height, square_pad=False):
        bbox = torch.tensor(bbox, dtype=torch.float32).view(-1, 4) / BBOX_SCALE
        bbox[:, 0::2] *= max(width, height) if square_pad else width
        bbox[:, 1::2] *= max(width, height) if square_pad else height
        return bbox[0].numpy().tolist()

    def box_iou(self, boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union


    def hungarian_matching(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou_matrix[i, j], _ = self.box_iou(torch.tensor([pred.detach().numpy().tolist()]), torch.tensor([gt.detach().numpy().tolist()]))
        
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        match_ious = []
        all_ious = []
        for r, c in zip(row_ind, col_ind):
            all_ious.append(iou_matrix[r, c])
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r, c))
                match_ious.append(iou_matrix[r, c])
        
        return matches, row_ind, sum(all_ious), sum(match_ious)
    
    def process_text_wo_box(self, text):
        text = re.sub(r'</?ref>', '', text)
        text = re.sub(r'</?box>', '', text)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'-\s*', '', text)
        return text

class SingleRegionMultiObjectEvaluator:
    def __init__(self):
        pass
    
    def eval_pred_list(self, pred_list):
        ## find object
        all_captions = []
        precision_all, recall_all, f1_all, averaged_iou_all, pair_all = [], [], [], 0, 0
        for i, output in enumerate(pred_list):   
            hw = output['hw']
            cur_caption_dict = {}
            all_boxes_pred, all_boxes_target = [], []
            try:
                predict_bboxes_str = re.findall(PATTERN_REGCAP, output['answer'])
                for predict_bbox_str in predict_bboxes_str:
                    predict_object = predict_bbox_str[0].lower()
                    predict_regions = predict_bbox_str[1]
                    predict_bbox = [self.process_box(bbox, hw[1], hw[0]) for bbox in ast.literal_eval(predict_regions)]
                    
                    all_boxes_pred.extend(predict_bbox)
            except:
                predict_bbox = [[0, 0, 0, 0]]
                all_boxes_pred.extend(predict_bbox)
            target_bboxes_str = re.findall(PATTERN_TARGET_RECAP, output['gt'])

            for target_bbox_str in target_bboxes_str:
                target_regions = target_bbox_str[1]
                target_bbox = [self.process_box(bbox, hw[1], hw[0]) for bbox in ast.literal_eval(target_regions)]
                
                all_boxes_target.extend(target_bbox)
                
            ## all boxes
            all_boxes_target = torch.tensor(all_boxes_target, dtype=torch.float32).view(-1, 4)
            all_boxes_pred = torch.tensor(all_boxes_pred, dtype=torch.float32).view(-1, 4)
            total_cnt_all = all_boxes_target.shape[0]
            pred_cnt_all = all_boxes_pred.shape[0]
            matches_all, pairs_all, ious_all, match_ious_all = self.hungarian_matching(all_boxes_pred, all_boxes_target)
            correct_all = len(matches_all)
            
            pre = correct_all / total_cnt_all if total_cnt_all != 0 else 0
            rec = correct_all / pred_cnt_all if pred_cnt_all != 0 else 0
            
            precision_all.append(pre)
            recall_all.append(rec)
            f1_all.append(2 * pre * rec / (pre + rec) if (pre + rec) != 0 else 0)
            averaged_iou_all += ious_all 
            pair_all += len(pairs_all)

        metrics= {
                'precision': sum(precision_all) / len(precision_all) if len(precision_all) > 0 else 0,
                'recall': sum(recall_all) / len(recall_all) if len(recall_all) > 0 else 0,
                'f1': sum(f1_all) / len(f1_all) if len(f1_all) > 0 else 0,
                'iou' : averaged_iou_all / pair_all
            }
        return metrics

    def process_box(self, bbox, width, height, square_pad=False):
        bbox = torch.tensor(bbox, dtype=torch.float32).view(-1, 4) / BBOX_SCALE
        bbox[:, 0::2] *= max(width, height) if square_pad else width
        bbox[:, 1::2] *= max(width, height) if square_pad else height
        return bbox[0].numpy().tolist()

    def box_iou(self, boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union


    def hungarian_matching(self, pred_boxes, gt_boxes, iou_threshold=0.75):
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou_matrix[i, j], _ = self.box_iou(torch.tensor([pred.detach().numpy().tolist()]), torch.tensor([gt.detach().numpy().tolist()]))
        
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        match_ious = []
        all_ious = []
        for r, c in zip(row_ind, col_ind):
            all_ious.append(iou_matrix[r, c])
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r, c))
                match_ious.append(iou_matrix[r, c])
        
        return matches, row_ind, sum(all_ious), sum(match_ious)
    
    def process_text_wo_box(self, text):
        text = re.sub(r'</?ref>', '', text)
        text = re.sub(r'</?box>', '', text)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'-\s*', '', text)
        return text


class MultiRegionSingleObjectEvaluator:
    def __init__(self):
        pass
    
    def eval_pred_list(self, pred_list):
        ## find object
        ious = []
        averaged_iou_box, averaged_iou_ob, box_pres, box_recs, pair_all, align_all_all = 0, 0, [], [], 0, 0
        align_in_all_gts, align_in_all_preds, align_in_all_matches, align_obs, align_in_all_detects = [], [], [], [], []
        
        for i, output in enumerate(pred_list):   
            hw = output['hw']
            boxid2object_pred, boxid2object_tar = {}, {}
            all_boxes_pred, all_boxes_target = [], []
            correct_align = 0
            try:
                predict_bboxes_str = re.findall(PATTERN_REGCAP, output['answer'])
                for boxid, predict_bbox_str in enumerate(predict_bboxes_str):
                    predict_object = predict_bbox_str[0].lower()
                    predict_regions = predict_bbox_str[1]
                    predict_bbox = [self.process_box(bbox, hw[1], hw[0]) for bbox in ast.literal_eval(predict_regions)]
                    
                    all_boxes_pred.extend(predict_bbox)
                    boxid2object_pred[boxid] = predict_object
            except:
                predict_bbox = [[0, 0, 0, 0]]
                all_boxes_pred.extend(predict_bbox)
            target_bboxes_str = re.findall(PATTERN_TARGET_RECAP, output['gt'])

            for boxid, target_bbox_str in enumerate(target_bboxes_str):
                target_object = target_bbox_str[0].lower()
                target_regions = target_bbox_str[1]
                target_bbox = [self.process_box(bbox, hw[1], hw[0]) for bbox in ast.literal_eval(target_regions)]
                
                all_boxes_target.extend(target_bbox)
                boxid2object_tar[boxid] = target_object
                
            ## all boxes
            all_boxes_target = torch.tensor(all_boxes_target, dtype=torch.float32).view(-1, 4)
            all_boxes_pred = torch.tensor(all_boxes_pred, dtype=torch.float32).view(-1, 4)
            total_cnt_all = all_boxes_target.shape[0]
            pred_cnt_all = all_boxes_pred.shape[0]                                                       
            matches_all, pairs_all, ious_all, match_ious_all = self.hungarian_matching(all_boxes_pred, all_boxes_target)

            for (r, c) in matches_all:
                if len(boxid2object_pred) > r and len(boxid2object_tar) > c:
                    if boxid2object_pred[r] == '' or boxid2object_pred[r] in boxid2object_tar[c] or boxid2object_tar[c] in boxid2object_pred[r]:
                        correct_align += 1
            
            correct_all = len(matches_all)
            box_pre = correct_all / total_cnt_all if total_cnt_all != 0 else 0
            
            
            if len(ious_all) > 0 and sum(ious_all) / len(ious_all) > 0.75:
                print(output)
            
            averaged_iou_box += sum(ious_all)
            pair_all += total_cnt_all
            box_pres.append(box_pre)
            
            align_in_all_gt = correct_align / total_cnt_all
            align_in_all_match = correct_align / correct_all if correct_all != 0 else 0
            
            
            align_in_all_gts.append(align_in_all_gt)
            align_in_all_matches.append(align_in_all_match)
            
            
            align_all = 0
            for ii, (r, c) in enumerate(pairs_all):
                if len(boxid2object_pred) > r and len(boxid2object_tar) > c:
                    if boxid2object_pred[r] == '' or boxid2object_pred[r] in boxid2object_tar[c] or boxid2object_tar[c] in boxid2object_pred[r]:
                        align_all += 1
                    averaged_iou_ob += ious_all[ii]
            align_all_all += align_all
            align_ob_in_all = align_all / total_cnt_all
            align_obs.append(align_ob_in_all)
            
            align_in_all_detect = correct_align / align_all if align_all != 0 else 0
            align_in_all_detects.append(align_in_all_detect)
            
        metrics = {
                    'box': {
                    'box_acc': sum(box_pres) / len(box_pres), 
                    'box_iou': averaged_iou_box / pair_all  
                },
                    'object': {
                        'object_acc': sum(align_obs) / len(align_obs), 
                        'object_iou': averaged_iou_ob / align_all_all 
                    },
                    'object-box': {
                        'object_box_acc': sum(align_in_all_gts) / len(align_in_all_gts),  
                        'object_box_alignment': sum(align_in_all_matches) / len(align_in_all_matches), 
                        'object_box_detection': sum(align_in_all_detects) / len(align_in_all_detects) 
                    }
            }
        
        return metrics
    
    def process_box(self, bbox, width, height, square_pad=False):
        bbox = torch.tensor(bbox, dtype=torch.float32).view(-1, 4) / BBOX_SCALE
        bbox[:, 0::2] *= max(width, height) if square_pad else width
        bbox[:, 1::2] *= max(width, height) if square_pad else height
        return bbox[0].numpy().tolist()

    def box_iou(self, boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union


    def hungarian_matching(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou_matrix[i, j], _ = self.box_iou(torch.tensor([pred.detach().numpy().tolist()]), torch.tensor([gt.detach().numpy().tolist()]))
        
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        match_ious = []
        all_ious = []
        match_rows = []
        for r, c in zip(row_ind, col_ind):
            all_ious.append(iou_matrix[r, c])
            match_rows.append((r, c))
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r, c))
                match_ious.append(iou_matrix[r, c])
        
        return matches, match_rows,  all_ious, sum(match_ious)
    
    def process_text_wo_box(self, text):
        text = re.sub(r'</?ref>', '', text)
        text = re.sub(r'</?box>', '', text)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'-\s*', '', text)
        return text
    

class MultiRegionMultiObjectEvaluator:
    def __init__(self):
        pass
    
    def eval_pred_list(self, pred_list):
        ## find object
        ious = []
        averaged_iou_box, averaged_iou_ob, box_pres, box_recs, pair_all, align_all_all = 0, 0, [], [], 0, 0
        align_in_all_gts, align_in_all_preds, align_in_all_matches, align_ob_pres, align_ob_recs, align_in_all_detects = [], [], [], [], [], []
        box_f1s, ob_f1s, align_f1s = [], [], []
        
        for i, output in enumerate(pred_list):   
            hw = output['hw']
            boxid2object_pred, boxid2object_tar = {}, {}
            all_boxes_pred, all_boxes_target = [], []
            correct_align = 0
            try:
                predict_bboxes_str = re.findall(PATTERN_REGCAP, output['answer'])
                boxid = 0
                for predict_bbox_str in predict_bboxes_str:
                    predict_object = predict_bbox_str[0].lower()
                    predict_regions = predict_bbox_str[1]
                    predict_bbox = [self.process_box(bbox, hw[1], hw[0]) for bbox in ast.literal_eval(predict_regions)]
                    
                    all_boxes_pred.extend(predict_bbox)
                    for pp in predict_bbox:
                        boxid2object_pred[boxid] = predict_object
                        boxid += 1
            except:
                predict_bbox = [[0, 0, 0, 0]]
                all_boxes_pred.extend(predict_bbox)
            target_bboxes_str = re.findall(PATTERN_TARGET_RECAP, output['gt'])

            boxid = 0
            for target_bbox_str in target_bboxes_str:
                target_object = target_bbox_str[0].lower()
                target_regions = target_bbox_str[1]
                target_bbox = [self.process_box(bbox, hw[1], hw[0]) for bbox in ast.literal_eval(target_regions)]
                
                all_boxes_target.extend(target_bbox)
                for tt in target_bbox:
                    boxid2object_tar[boxid] = target_object
                    boxid += 1
                
            ## all boxes
            all_boxes_target = torch.tensor(all_boxes_target, dtype=torch.float32).view(-1, 4)
            all_boxes_pred = torch.tensor(all_boxes_pred, dtype=torch.float32).view(-1, 4)
            total_cnt_all = all_boxes_target.shape[0]
            pred_cnt_all = all_boxes_pred.shape[0]                                                       
            matches_all, pairs_all, ious_all, match_ious_all = self.hungarian_matching(all_boxes_pred, all_boxes_target)

            for (r, c) in matches_all:
                if boxid2object_pred[r] == '' or boxid2object_pred[r] in boxid2object_tar[c] or boxid2object_tar[c] in boxid2object_pred[r]:
                    correct_align += 1
            
            correct_all = len(matches_all)
            box_pre = correct_all / total_cnt_all if total_cnt_all != 0 else 0
            box_rec = correct_all / pred_cnt_all if pred_cnt_all != 0 else 0
            box_f1 = 2 * box_pre * box_rec / (box_pre + box_rec) if (box_pre + box_rec) != 0 else 0
            averaged_iou_box += sum(ious_all)
            pair_all += total_cnt_all
            box_pres.append(box_pre)
            box_recs.append(box_rec)
            box_f1s.append(box_f1)
            
            align_in_all_gt = correct_align / total_cnt_all
            align_in_all_pred = correct_align / pred_cnt_all if pred_cnt_all != 0 else 0
            align_in_all_match = correct_align / correct_all if correct_all != 0 else 0
            align_f1 = 2 * align_in_all_pred * align_in_all_gt / (align_in_all_gt + align_in_all_pred) if (align_in_all_gt + align_in_all_pred) != 0 else 0
            
            if '], [' in output['answer'] and align_f1 > 0.4:
                print(output)
            
            align_f1s.append(align_f1)
            
            align_in_all_gts.append(align_in_all_gt)
            align_in_all_preds.append(align_in_all_pred)
            align_in_all_matches.append(align_in_all_match)
            
            
            align_all = 0
            for ii, (r, c) in enumerate(pairs_all):
                if boxid2object_pred[r] == '' or boxid2object_pred[r] in boxid2object_tar[c] or boxid2object_tar[c] in boxid2object_pred[r]:
                    align_all += 1
                    averaged_iou_ob += ious_all[ii]
            align_all_all += align_all
            align_ob_in_all_pre = align_all / total_cnt_all
            align_ob_in_all_rec = align_all / pred_cnt_all if pred_cnt_all != 0 else 0
            align_ob_pres.append(align_ob_in_all_pre)
            align_ob_recs.append(align_ob_in_all_rec)
            ob_f1 = 2 * align_ob_in_all_pre * align_ob_in_all_rec / (align_ob_in_all_pre + align_ob_in_all_rec) if (align_ob_in_all_pre + align_ob_in_all_rec) != 0 else 0
            ob_f1s.append(ob_f1)
            
            align_in_all_detect = correct_align / align_all if align_all != 0 else 0
            align_in_all_detects.append(align_in_all_detect)
            
        metrics = {
                    'box': {
                    'box_pre': sum(box_pres) / len(box_pres),  
                    'box_recall': sum(box_recs) / len(box_recs),
                    'box_f1': sum(box_f1s) / len(box_f1s),
                    'box_iou': averaged_iou_box / pair_all 
                },
                    'object': {
                        'object_pre': sum(align_ob_pres) / len(align_ob_pres), 
                        'object_rec': sum(align_ob_recs) / len(align_ob_recs), 
                        'object_f1': sum(ob_f1s) / len(ob_f1s),
                        'object_iou': averaged_iou_ob / align_all_all if align_all_all != 0 else 0 
                    },
                    'object-box': {
                        'object_box_pre': sum(align_in_all_gts) / len(align_in_all_gts),  
                        'object_box_recall': sum(align_in_all_preds) / len(align_in_all_preds),
                        'object_box_f1': sum(align_f1s) / len(align_f1s),
                        'object_box_alignment': sum(align_in_all_matches) / len(align_in_all_matches), 
                        'object_box_detection': sum(align_in_all_detects) / len(align_in_all_detects) 
                    }
            }
        
        return metrics
    
    def process_box(self, bbox, width, height, square_pad=False):
        bbox = torch.tensor(bbox, dtype=torch.float32).view(-1, 4) / BBOX_SCALE
        bbox[:, 0::2] *= max(width, height) if square_pad else width
        bbox[:, 1::2] *= max(width, height) if square_pad else height
        return bbox[0].numpy().tolist()

    def box_iou(self, boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union


    def hungarian_matching(self, pred_boxes, gt_boxes, iou_threshold=0.5):
        iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, pred in enumerate(pred_boxes):
            for j, gt in enumerate(gt_boxes):
                iou_matrix[i, j], _ = self.box_iou(torch.tensor([pred.detach().numpy().tolist()]), torch.tensor([gt.detach().numpy().tolist()]))
        
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        match_ious = []
        all_ious = []
        match_rows = []
        for r, c in zip(row_ind, col_ind):
            all_ious.append(iou_matrix[r, c])
            match_rows.append((r, c))
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r, c))
                match_ious.append(iou_matrix[r, c])
        
        return matches, match_rows,  all_ious, sum(match_ious)
    
    def process_text_wo_box(self, text):
        text = re.sub(r'</?ref>', '', text)
        text = re.sub(r'</?box>', '', text)
        text = re.sub(r'\[\[.*?\]\]', '', text)
        text = re.sub(r'-\s*', '', text)
        return text

