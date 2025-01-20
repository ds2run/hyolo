# Ultralytics YOLO 🚀, GPL-3.0 license
from copy import copy
import math
import sys
import os
import re
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# from torchvision.utils import save_image

from ultralytics.nn.tasks import DetectionModel
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, RANK, colorstr
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def get_dataloader(self, dataset_path, batch_size, mode='train', rank=0):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(
            int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 augment=mode == 'train',
                                 cache=self.args.cache,
                                 pad=0 if mode == 'train' else 0.5,
                                 rect=self.args.rect or mode == 'val',
                                 rank=rank,
                                 workers=self.args.workers,
                                 close_mosaic=self.args.close_mosaic != 0,
                                 prefix=colorstr(f'{mode}: '),
                                 shuffle=mode == 'train',
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, rank=rank, mode=mode,
                             rect=mode == 'val', names=self.data['names'], hierarchy=self.data['hierarchy_names'])[0]
        # rect=mode == 'val', names=self.data['names'], hierarchy=self.data['hierarchy'])[0]

    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device,
                                       non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.hierarchy_names = self.data['hierarchy_names']
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
        self.hier_arch_version = self.args.hier_arch_version

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DetectionModel(cfg,
                               nc=self.data['nc'],
                               hier=self.data['hierarchy_names'],
                               verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        l_names = []
        for l in range(len(self.data['hierarchy_names'])):
            l_names.append(f"b_lv{l}")
            l_names.append(f"cl_lv{l}")
            l_names.append(f"dfl_lv{l}")
        self.loss_names = tuple(l_names)
        return v8.detect.DetectionValidator(self.test_loader,
                                            save_dir=self.save_dir,
                                            args=copy(self.args),
                                            hier=self.data['hierarchy_names'])

    def criterion(self, preds, batch):
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(de_parallel(self.model))
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items
                          ]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%11s' * (4 + len(self.loss_names))) % (
            'Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        for l in range(len(self.data['hierarchy_names'])):
            plot_images(
                images=batch['img'],
                batch_idx=batch['batch_idx'],
                cls=torch.index_select(batch['hierarchy_names'], 1,
                                       torch.tensor([l])).squeeze(-1),
                bboxes=batch['bboxes'],
                paths=batch['im_file'],
                fname=self.save_dir / f'train_batch{ni}_level{l}.jpg',
                names={
                    int(k): v
                    for k, v in self.data['hierarchy_names'][str(l)].items()
                })

    def plot_metrics(self):
        plot_results(file=self.csv)  # save results.png

    def plot_training_labels(self):
        boxes = np.concatenate(
            [lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate(
            [lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes,
                    cls.squeeze(),
                    names=self.data['names'],
                    save_dir=self.save_dir)


# Criterion class for computing training losses
class Loss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1
        self.bbox_loss = BboxLoss(m.reg_max - 1,
                                  use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        self.dependency_loss = self.hyp.dependency_loss
        self.dataset_hier_version = self.hyp.dataset_hier_version
        self.regul_alpha = self.hyp.regul_alpha

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:

            b, a, c = pred_dist.shape  # batch, anchors, channels #batch, 8400, 64
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        assigner = {}
        feats = {}
        pred_scores = {}
        hier = {}
        targets = {}
        gt_labels = {}
        target_bboxes = {}
        target_scores = {}
        fg_mask = {}
        target_scores_sum = {}
        if isinstance(preds, list):
            feats_last = preds[0]

            num_hier_levels = len(preds[1])
            for l in range(num_hier_levels):
                if len(batch['hierarchy_names']) == 0:
                    hier[f"hier_level{l}"] = batch['cls']
                    # sys.exit(0)
                else:
                    hier[f"hier_level{l}"] = torch.index_select(
                        batch['hierarchy_names'], 1, torch.tensor([l]))
                feats[f"feats_level{l}"] = preds[1][f"cv3_level{l}_out"]
                assigner[f"assigner_level{l}"] = TaskAlignedAssigner(
                    topk=10,
                    num_classes=feats[f"feats_level{l}"][1].shape[1],
                    alpha=0.5,
                    beta=6.0)
        else:
            batch['hierarchy_names'] = batch['hierarchy_names'].to(
                self.device, non_blocking=True)
            feats_last = preds[0]
            num_hier_levels = len(preds[1])
            for l in range(num_hier_levels):
                if len(batch['hierarchy_names']) == 0:
                    hier[f"hier_level{l}"] = batch['hierarchy_names']
                else:
                    hier[f"hier_level{l}"] = torch.index_select(
                        batch['hierarchy_names'], 1,
                        torch.tensor([l]).to(self.device))
                feats[f"feats_level{l}"] = preds[1][f"cv3_level{l}_out"]
                assigner[f"assigner_level{l}"] = TaskAlignedAssigner(
                    topk=10,
                    num_classes=feats[f"feats_level{l}"][1].shape[1],
                    alpha=0.5,
                    beta=6.0)
        loss = torch.zeros(3 * num_hier_levels,
                           device=self.device)  # box, cls, dfl

        pred_distri, pred_scores[
            f"pred_scores_level{num_hier_levels-1}"] = torch.cat([
                xi.view(feats_last[0].shape[0], self.no, -1)
                for xi in feats_last
            ], 2).split((self.reg_max * 4, self.nc), 1)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_scores[f"pred_scores_level{num_hier_levels - 1}"] = pred_scores[
            f"pred_scores_level{num_hier_levels - 1}"].permute(0, 2,
                                                               1).contiguous()

        for l in range(0, num_hier_levels - 1):
            pred_scores[f"pred_scores_level{l}"] = torch.cat([
                xi.view(feats[f"feats_level{l}"][0].shape[0],
                        feats[f"feats_level{l}"][0].shape[1], -1)
                for xi in feats[f"feats_level{l}"]
            ], 2)
            pred_scores[f"pred_scores_level{l}"] = pred_scores[
                f"pred_scores_level{l}"].permute(0, 2, 1).contiguous()

        dtype = pred_scores[f"pred_scores_level0"].dtype
        batch_size = pred_scores[f"pred_scores_level0"].shape[0]
        imgsz = torch.tensor(
            feats[f"feats_level{num_hier_levels-1}"][0].shape[2:],
            device=self.device,
            dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(
            feats[f"feats_level{num_hier_levels-1}"], self.stride, 0.5)
        # targets
        for l in range(num_hier_levels):
            targets[f"targets_level{l}"] = torch.cat((batch['batch_idx'].view(
                -1, 1), hier[f"hier_level{l}"].view(-1, 1), batch['bboxes']),
                                                     1)
            targets[f"targets_level{l}"] = self.preprocess(
                targets[f"targets_level{l}"].to(self.device),
                batch_size,
                scale_tensor=imgsz[[1, 0, 1, 0]])

        # gt_bboxes are the same across hier levels, # cls, xyxy
        gt_labels[f"gt_labels_level{num_hier_levels-1}"], gt_bboxes = targets[
            f"targets_level{num_hier_levels-1}"].split((1, 4), 2)
        for l in range(num_hier_levels - 1):
            gt_labels[f"gt_labels_level{l}"], _ = targets[
                f"targets_level{l}"].split((1, 4), 2)

        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points,
                                       pred_distri)  # xyxy, (b, h*w, 4)

        for l in range(num_hier_levels):
            _, target_bboxes[f"target_bboxes_level{l}"], target_scores[
                f"target_scores_level{l}"], fg_mask[
                    f"fg_mask_level{l}"], _ = assigner[f"assigner_level{l}"](
                        pred_scores[f"pred_scores_level{l}"].detach().sigmoid(
                        ), (pred_bboxes.detach() * stride_tensor).type(
                            gt_bboxes.dtype), anchor_points * stride_tensor,
                        gt_labels[f"gt_labels_level{l}"], gt_bboxes, mask_gt)

            target_bboxes[f"target_bboxes_level{l}"] /= stride_tensor

        for l in range(num_hier_levels):
            target_scores_sum[f"target_scores_sum_level{l}"] = max(
                target_scores[f"target_scores_level{l}"].sum(), 1)

        if self.dependency_loss:

            # Apply sigmoid to all elements in pred_scores
            fp_scores_sigmoid = {}
            for key, value in pred_scores.items():
                fp_scores_sigmoid[key] = torch.sigmoid(value)

            # Match the indices on all levels
            # Store indices where at least one position in target_scores is non-zero
            non_zero_sum_indices = {}
            for key, tensor in target_scores.items():
                summed = tensor.sum(dim=-1)  # Sum along the last axis
                nonzero_sum = (summed
                               != 0).unsqueeze(-1)  # returns true or false
                non_zero_sum_indices[key] = nonzero_sum.type(
                    torch.float32)  # change to 1 or 0

            # Multiply between levels
            common_lev_idx = {}

            commonind = {}
            for l in range(num_hier_levels):
                num_classes = target_scores[f'target_scores_level{l}'].shape[2]
                if l == 0:
                    commonind[
                        f'target_scores_level{l}'] = non_zero_sum_indices[
                            f'target_scores_level{l}'] * non_zero_sum_indices[
                                f'target_scores_level{l+1}']
                    common_lev_idx[f'target_scores_level{l}'] = commonind[
                        f'target_scores_level{l}'].repeat(1, 1, num_classes)
                else:
                    commonind[
                        f'target_scores_level{l}'] = non_zero_sum_indices[
                            f'target_scores_level{l}'] * non_zero_sum_indices[
                                f'target_scores_level{l-1}']
                    common_lev_idx[f'target_scores_level{l}'] = commonind[
                        f'target_scores_level{l}'].repeat(1, 1, num_classes)

            # Zero out all anchors with level mismatch
            fp_scores_sigmoid_zeroed_level = {}
            for l in range(num_hier_levels):
                fp_scores_sigmoid_zeroed_level[
                    f"pred_scores_level{l}"] = fp_scores_sigmoid[
                        f"pred_scores_level{l}"] * common_lev_idx[
                            f'target_scores_level{l}']

            # Zero out values below threshold
            thresh = 0.001
            fp_scores_sigmoid_zeroed_level_thresh = {}
            for key, value in fp_scores_sigmoid_zeroed_level.items():
                fp_scores_sigmoid_zeroed_level_thresh[
                    key] = fp_scores_sigmoid_zeroed_level[key].clone()
                fp_scores_sigmoid_zeroed_level_thresh[key][
                    fp_scores_sigmoid_zeroed_level[key] < thresh] = 0

            # Zero out all tp confidences
            # Initialize an empty dictionary for output
            zero_out_tp = {}

            for key, value in target_scores.items():
                # reverse 0s to 1s and non-zero elements to 0s
                zero_out_tp[key] = torch.where(value == 0, torch.tensor(1),
                                               torch.tensor(0))

            # Apply the mask
            fp_scores_zeroed = {}
            for l in range(num_hier_levels):
                fp_scores_zeroed[
                    f"pred_scores_level{l}"] = fp_scores_sigmoid_zeroed_level_thresh[
                        f"pred_scores_level{l}"] * zero_out_tp[
                            f'target_scores_level{l}']

         
            # Get the child_parent relationships depending on the dataset hierarchical version           
            if self.dataset_hier_version==1:
                with open("hierarchical_configs/child_parent_map_V1.json", "r") as json_file:
                    child_parent_map=json.load(json_file)
                                # Convert the parent keys to integers    
                child_parent_map = {int(key): value for key, value in child_parent_map.items()}
                # Convert the nested keys to integers                
                for parent_key in child_parent_map:
                    if isinstance(child_parent_map[parent_key], dict):
                        child_parent_map[parent_key] = {int(k): v for k, v in child_parent_map[parent_key].items()}
            elif self.dataset_hier_version==2:
                with open("hierarchical_configs/child_parent_map_V2.json", "r") as json_file:
                    child_parent_map=json.load(json_file)
                # Convert the parent keys to integers    
                child_parent_map = {int(key): value for key, value in child_parent_map.items()}
                # Convert the nested keys to integers                
                for parent_key in child_parent_map:
                    if isinstance(child_parent_map[parent_key], dict):
                        child_parent_map[parent_key] = {int(k): v for k, v in child_parent_map[parent_key].items()}

            else: 
                raise ValueError(f"Invalid dataset_hier_version {self.dataset_hier_version}")
            
            # Check if CUDA is available; otherwise, use the CPU
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')

            # Get all candidates for punishment
            fp_dict = {}
            for key, tensor in fp_scores_zeroed.items():
                conf_anchor_class = []
                tensor = tensor.to(device)

                for i in range(len(tensor)):
                    non_zero_indices = torch.nonzero(
                        tensor[i], as_tuple=False).to(
                            device)  # non-zero [row_num, column_num]

                    # Get corresponding non-zero values
                    values = tensor[i][non_zero_indices[:, 0],
                                       non_zero_indices[:, 1]].to(device)

                    # Convert indices to a list and move to the appropriate device
                    indices = non_zero_indices.tolist()
                    conf_anchor_class.append(
                        torch.cat((values.unsqueeze(1).to(device),
                                   torch.tensor(indices, device=device)),
                                  dim=1))
                fp_dict[key] = conf_anchor_class

            # Create a new dictionary with modified keys
            fp_dict_new_keys = {}
            for idx, (key, value) in enumerate(fp_dict.items()):
                new_key = int(re.search(r'\d+', key).group())
                fp_dict_new_keys[new_key] = value

            # Replace the last element of each column with the corresponding value from child_parent_map
            # i.e. replace the child class with the parent class for easier comparison

            for key, tensor in fp_dict_new_keys.items():
                for i in range(len(tensor)):
                    if key != 0:
                        last_indices = tensor[i].shape[1] - 1
                        for row_idx, parent_key in enumerate(
                                tensor[i][:, last_indices]):
                            tensor[i][row_idx,
                                      last_indices] = child_parent_map[key][
                                          int(parent_key)]

            # Get the non-zero indices in each level
            nonzero_indices = {}
            for key, tensor in target_scores.items():
                # Extract the integer part from the key
                level = int(''.join(filter(str.isdigit, key)))
                indices = []
                for i in range(len(tensor)):
                    # Get non-zero indices
                    indices.append(tensor[i].nonzero())
                nonzero_indices[level] = indices

            # Zero out the confidence if the parent matches
            fp_dict_final = {}
            for l in range(num_hier_levels):
                fp_dict_final[l] = []
                for i, row in enumerate(fp_dict_new_keys[l]):
                    fp_dict_final[l].append(row.clone())
                    for j, line in enumerate(row):
                        if [int(line[1]), int(line[2])
                            ] in nonzero_indices[l][i].tolist():
                            fp_dict_final[l][i][j] = torch.tensor(
                                [0, line[1], line[2]], dtype=row.dtype)

            # If there are empty tensors, replace them by [0,0,0]
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            # If there are empty tensors, replace them with [0, 0, 0]
            for key, value in fp_dict_final.items():
                for i in range(len(value)):
                    if value[i].numel() == 0:  # Check if the tensor is empty
                        fp_dict_final[key][i] = torch.tensor([[0, 0, 0]],
                                                             device=device)

            # Sum all the confidences per level which are in column 0
            # and normalize by the number of non-zero elements
            punishment_sum = {}
            non_zero_counts = {}
            for key, tensor in fp_dict_final.items():
                total_sum = 0.0
                non_zero = 0.0
                for i in range(len(tensor)):
                    for row_tensor in fp_dict_final[key][i]:
                        total_sum += row_tensor[0]
                        non_zero += torch.nonzero(row_tensor[0]).size(0)

                punishment_sum[key] = total_sum
                non_zero_counts[key] = non_zero

            self.punishment_normalized = {}
            for l in range(num_hier_levels):
                if (non_zero_counts[l] == 0):
                    self.punishment_normalized[
                        f"pred_scores_level{l}"] = torch.tensor(0)
                else:
                    self.punishment_normalized[
                        f"pred_scores_level{l}"] = punishment_sum[
                            l] / non_zero_counts[l]

            # Punishment on level 0 is always 0
            self.punishment_normalized[f"pred_scores_level{0}"] = torch.zeros(
                1)

            # Transfer punishment to the appropriate device - cpu or gpu
            for key, value in self.punishment_normalized.items():
                if torch.cuda.is_available():
                    self.punishment_normalized[key] = value.cuda()
                else:
                    self.punishment_normalized[key] = value.cpu()

            self.bce_no_dependency = {}
            # cls loss
            for l in range(len(loss)):
                if l % 3 == 1:  #cls
                    if math.floor(l / 3) == 0:
                        self.bce_no_dependency[l] = self.bce(
                            pred_scores[f"pred_scores_level{math.floor(l/3)}"],
                            target_scores[
                                f"target_scores_level{math.floor(l/3)}"].
                            to(dtype)).sum() / target_scores_sum[
                                f"target_scores_sum_level{math.floor(l/3)}"]
                        loss[l] = self.bce(
                            pred_scores[f"pred_scores_level{math.floor(l/3)}"],
                            target_scores[
                                f"target_scores_level{math.floor(l/3)}"].
                            to(dtype)).sum() / target_scores_sum[
                                f"target_scores_sum_level{math.floor(l/3)}"]
                    else:
                        self.bce_no_dependency[l] = self.bce(
                            pred_scores[f"pred_scores_level{math.floor(l/3)}"],
                            target_scores[
                                f"target_scores_level{math.floor(l/3)}"].
                            to(dtype)).sum() / target_scores_sum[
                                f"target_scores_sum_level{math.floor(l/3)}"]
                        loss[l] = self.bce(
                            pred_scores[f"pred_scores_level{math.floor(l/3)}"],
                            target_scores[
                                f"target_scores_level{math.floor(l/3)}"].to(
                                    dtype)
                        ).sum() / target_scores_sum[
                            f"target_scores_sum_level{math.floor(l/3)}"] + self.punishment_normalized[
                                f"pred_scores_level{math.floor(l / 3)}"] * self.regul_alpha
        else:
            for l in range(len(loss)):
                if l % 3 == 1:  #cls
                    loss[l] = self.bce(
                        pred_scores[f"pred_scores_level{math.floor(l/3)}"],
                        target_scores[f"target_scores_level{math.floor(l/3)}"].
                        to(dtype)).sum() / target_scores_sum[
                            f"target_scores_sum_level{math.floor(l/3)}"]

        # bbox loss
        if fg_mask[f"fg_mask_level{num_hier_levels-1}"].sum():

            for l in range(len(loss)):
                if l % 3 == 0:
                    loss[l], loss[l + 2] = self.bbox_loss(
                        pred_distri, pred_bboxes, anchor_points,
                        target_bboxes[f"target_bboxes_level{math.floor(l/3)}"],
                        target_scores[f"target_scores_level{math.floor(l/3)}"],
                        target_scores_sum[
                            f"target_scores_sum_level{math.floor(l/3)}"],
                        fg_mask[f"fg_mask_level{math.floor(l/3)}"])

        for l in range(0, len(loss)):
            if l % 3 == 0:
                loss[l] = loss[l] * self.hyp.box / num_hier_levels  # box gain
            elif l % 3 == 1:
                loss[l] = loss[l] * self.hyp.cls / num_hier_levels  # cls gain
                if self.dependency_loss:
                    self.bce_no_dependency[l] = self.bce_no_dependency[
                        l] * self.hyp.cls / num_hier_levels
                    self.bce_no_dependency[l] = self.bce_no_dependency[
                        l].detach()
                    self.punishment_normalized[
                        f"pred_scores_level{math.floor(l / 3)}"] = self.punishment_normalized[
                            f"pred_scores_level{math.floor(l / 3)}"].detach()
            else:
                loss[l] = loss[l] * self.hyp.dfl / num_hier_levels  # dfl gain
        if self.dependency_loss:
            return loss.sum() * batch_size, loss.detach(
            ), self.bce_no_dependency, self.punishment_normalized
        else:
            return loss.sum() * batch_size, loss.detach(
            ),  # loss(box, cls, dfl)


def train(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
