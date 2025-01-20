# Ultralytics YOLO 🚀, GPL-3.0 license

import os
from pathlib import Path
import copy

import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
import torch
import time

from ultralytics.yolo.data import build_dataloader
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, colorstr, ops
from ultralytics.yolo.utils.checks import check_requirements
from ultralytics.yolo.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.yolo.utils.plotting import output_to_target, plot_images
from ultralytics.yolo.utils.torch_utils import de_parallel


class DetectionValidator(BaseValidator):

    def __init__(self,
                 dataloader=None,
                 save_dir=None,
                 pbar=None,
                 args=None,
                 hier=None):
        super().__init__(dataloader, save_dir, pbar, args, hier)
        self.args.task = 'detect'
        self.is_coco = False
        self.class_map = None
        self.metrics = {}
        for l in range(self.args.hier_depth):
            self.metrics[f"metrics_level{l}"] = DetMetrics(
                save_dir=self.save_dir)
        self.iouv = torch.linspace(0.5, 0.95,
                                   10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()

    def preprocess(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half()
                        if self.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes', 'hierarchy_names']:
            batch[k] = batch[k].to(self.device)
        nb = len(batch['img'])
        self.lb = [
            torch.cat([batch['cls'], batch['bboxes']],
                      dim=-1)[batch['batch_idx'] == i] for i in range(nb)
        ] if self.args.save_hybrid else []  # for autolabelling
        return batch

    def init_metrics(self, model):
        val = self.data.get(self.args.split, '')  # validation path
        self.is_coco = isinstance(val, str) and val.endswith(
            f'coco{os.sep}val2017.txt')  # is COCO dataset
        self.class_map = ops.coco80_to_coco91_class(
        ) if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.hierarchy_names = model.hierarchy_names
        self.metrics[f"metrics_level0"].plot = self.args.plots
        self.confusion_matrix = {}
        self.stats = {}
        for l in range(len(self.hierarchy_names)):
            self.metrics[f"metrics_level{l}"].names = self.hierarchy_names[str(
                l)]
            self.confusion_matrix[
                f"confusion_matrix_level{l}"] = ConfusionMatrix(
                    nc=len(self.hierarchy_names[str(l)]),
                    file_name=f"confusion_matrix_level{l}.png")
            self.stats[f"stats_level{l}"] = []
        self.seen = 0
        self.jdict = []

        self.gt_cls = []
        self.pred_cls = []
        self.fps_cls = []

    def get_desc(self):
        return ('%11s' + '%60s' + '%11s' * 6) % ('H-Level', 'Class', 'Images',
                                                 'Instances', 'Box(P', 'R',
                                                 'mAP50', 'mAP50-95)')

    def postprocess(self, preds):
        preds_out = {}
        for l in range(len(preds[2])):
            preds_out[f"preds_out_level{l}"] = ops.non_max_suppression(
                preds[2][f"y_hier_level{l}"],
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls,
                max_det=self.args.max_det)
        return preds_out

    def update_metrics(self, preds, batch):
        # Metrics
        hier = {}
        nl = {}
        npr = {}
        labelsn = {}
        correct_bboxes = {}
        # for new metric

        if self.args.get_hier_paths:
            bbox_class_gt = {}
            gt_classes_list = []
            pred_classes_list = []
            fp_bboxes = []
            fp_classes = []

        for si in range(len(torch.unique(batch['batch_idx']))):
            idx = batch['batch_idx'] == si
            # cls = batch['cls'][idx]
            bbox = batch['bboxes'][idx]
            shape = batch['ori_shape'][si]
            for l in range(len(preds)):
                hier[f"hier_level{l}"] = torch.index_select(
                    batch['hierarchy_names'], 1,
                    torch.tensor([l]).to(self.device))[idx]
                nl[f"nl_level{l}"], npr[f"npr_level{l}"] = hier[
                    f"hier_level{l}"].shape[0], preds[f"preds_out_level{l}"][
                        si].shape[0]  # number of labels, predictions
                correct_bboxes[f"correct_bboxes_level{l}"] = torch.zeros(
                    npr[f"npr_level{l}"],
                    self.niou,
                    dtype=torch.bool,
                    device=self.device)  # init

            self.seen += 1

            for l in range(len(self.hierarchy_names)):
                if npr[f"npr_level{l}"] == 0:
                    if nl[f"nl_level{l}"]:
                        self.stats[f"stats_level{l}"].append(
                            (correct_bboxes[f"correct_bboxes_level{l}"],
                             *torch.zeros((2, 0), device=self.device),
                             hier[f"hier_level{l}"].squeeze(-1)))

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = copy.deepcopy(preds)
            for l in range(len(self.hierarchy_names)):
                ops.scale_boxes(
                    batch['img'][si].shape[1:],
                    predn[f"preds_out_level{l}"][si][:, :4],
                    shape,
                    ratio_pad=batch['ratio_pad'][si])  # native-space pred

            # getting paths for tp and fps predictions across all hier levels
            if self.args.get_hier_paths:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height),
                    device=self.device)  # target boxes
                ops.scale_boxes(
                    batch['img'][si].shape[1:],
                    tbox,
                    shape,
                    ratio_pad=batch['ratio_pad'][si])  # native-space labels
                for key, value in zip(tbox, batch['hierarchy_names']):
                    bbox_class_gt[key] = value
                    gt_classes_list.append(value.tolist())
                    pred_classes_list.append([[]])

                fps_first_flag = True
                for l in range(len(self.hierarchy_names)):
                    iou_matrix = []
                    predn_filtered = predn[f"preds_out_level{l}"][0][
                        predn[f"preds_out_level{l}"][0][:, 4] > 0.25]
                    for box1 in predn_filtered[:, :4]:
                        row_iou = [
                            ops.bb_intersection_over_union(box1, box2)
                            for box2 in tbox
                        ]
                        iou_matrix.append(row_iou)
                    iou_matrix_list = np.array([[
                        float(elem) if not isinstance(elem, torch.Tensor) else
                        float(elem.item()) for elem in row
                    ] for row in iou_matrix])
                    predn_fps_curr_level = []

                    for en, iou in enumerate(iou_matrix_list):
                        #max iou between class with index en and all predictions
                        max_iou = np.max(iou)
                        # Find the index of the maximum value in the array
                        index_of_pred = np.argmax(iou)
                        gt_num_preds = len(pred_classes_list[index_of_pred])

                        if max_iou > 0.45:
                            if len(pred_classes_list[index_of_pred][0]) == 0:
                                pred_classes_list[index_of_pred][0].append(
                                    int(predn_filtered[en][5]))
                            elif gt_num_preds == 1 and len(
                                    pred_classes_list[index_of_pred][0]) == l:
                                pred_classes_list[index_of_pred][0].append(
                                    int(predn_filtered[en][5]))
                            elif gt_num_preds == 1 and len(
                                    pred_classes_list[index_of_pred][0]) > l:
                                pred_classes_list[index_of_pred].append(
                                    pred_classes_list[index_of_pred][0][:-1] +
                                    [int(predn_filtered[en][5])])
                            elif gt_num_preds > 1:
                                pred_flag = True
                                for ek, pred in enumerate(
                                        pred_classes_list[index_of_pred]):
                                    if len(pred) == l and pred_flag:
                                        pred_classes_list[index_of_pred][
                                            ek].append(
                                                int(predn_filtered[en][5]))
                                        pred_flag = False
                                if pred_flag:
                                    pred_classes_list[index_of_pred].append(
                                        pred_classes_list[index_of_pred][-1]
                                        [:-1] + [int(predn_filtered[en][5])])
                        else:
                            predn_fps_curr_level.append(predn_filtered[en])

                    if predn_fps_curr_level:
                        if fps_first_flag:
                            for pred in predn_fps_curr_level:
                                fp_bboxes.append(pred[:4])
                                fp_classes.append([None] * l + [int(pred[5])])
                                fps_first_flag = False
                        else:
                            iou_fps_matrix = []

                            for box1 in predn_fps_curr_level[:4]:
                                row_iou = [
                                    ops.bb_intersection_over_union(box1, box2)
                                    for box2 in fp_bboxes
                                ]
                                iou_fps_matrix.append(row_iou)
                            iou_fps_matrix = np.array([[
                                float(elem)
                                if not isinstance(elem, torch.Tensor) else
                                float(elem.item()) for elem in row
                            ] for row in iou_fps_matrix])
                            for en, iou in enumerate(iou_fps_matrix):
                                max_iou = np.max(iou)
                                index_of_pred = np.argmax(iou)

                                if max_iou > 0.80:
                                    if len(fp_classes[index_of_pred]) == l:
                                        fp_classes[index_of_pred].append(
                                            int(predn_fps_curr_level[en][5]))
                                    else:
                                        fp_classes.append(
                                            fp_classes[index_of_pred][:-1] +
                                            [int(predn_fps_curr_level[en][5])])
                                        fp_bboxes.append(
                                            predn_fps_curr_level[en][:4])
                                else:
                                    fp_bboxes.append(
                                        predn_fps_curr_level[en][:4])
                                    fp_classes.append(
                                        [None] * l +
                                        [int(predn_fps_curr_level[en][5])])

                    for enu, det in enumerate(
                            pred_classes_list):  # [[[0]], [[]]]
                        if len(det) > 1:  # more then 1 detection per GT bbox
                            for el in range(len(det)):
                                if len(det[el]) < l + 1:
                                    pred_classes_list[enu][el].append(None)
                        else:  # 1 detection per GT bbox
                            if len(det[0]) < l + 1:
                                pred_classes_list[enu][0].append(None)
                    # add None to current hier if missing prediction for FPS
                    for enu, det in enumerate(fp_classes):  # [[0], []]
                        if len(det) < l + 1:
                            fp_classes[enu].append(None)

                self.gt_cls += gt_classes_list
                self.pred_cls += pred_classes_list
                if len(fp_classes):
                    self.fps_cls += fp_classes

            # Evaluate
            for l in range(len(self.hierarchy_names)):
                if nl[f"nl_level{l}"]:
                    height, width = batch['img'].shape[2:]
                    tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                        (width, height, width, height),
                        device=self.device)  # target boxes
                    ops.scale_boxes(batch['img'][si].shape[1:],
                                    tbox,
                                    shape,
                                    ratio_pad=batch['ratio_pad']
                                    [si])  # native-space labels
                    labelsn[f"labelsn_level{l}"] = torch.cat(
                        (hier[f"hier_level{l}"], tbox), 1)
                    correct_bboxes[
                        f"correct_bboxes_level{l}"] = self._process_batch(
                            predn[f"preds_out_level{l}"][si],
                            labelsn[f"labelsn_level{l}"])

                    # TODO: maybe remove these `self.` arguments as they already are member variable

                    if self.args.plots:
                        self.confusion_matrix[
                            f"confusion_matrix_level{l}"].process_batch(
                                predn[f"preds_out_level{l}"][si],
                                labelsn[f"labelsn_level{l}"])
            for l in range(len(self.hierarchy_names)):
                self.stats[f"stats_level{l}"].append(
                    (correct_bboxes[f"correct_bboxes_level{l}"],
                     preds[f"preds_out_level{l}"][si][:, 4],
                     preds[f"preds_out_level{l}"][si][:, 5],
                     hier[f"hier_level{l}"].squeeze(-1)))  # (conf, pcls, tcls)
            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            if self.args.save_txt:
                file = self.save_dir / 'labels' / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, shape, file)

    def finalize_metrics(self, *args, **kwargs):

        for l in range(len(self.hierarchy_names)):
            self.metrics[f"metrics_level{l}"].confusion_matrix = {}
            self.metrics[f"metrics_level{l}"].speed = self.speed
            self.metrics[f"metrics_level{l}"].confusion_matrix[
                f"confusion_matrix_level{l}"] = self.confusion_matrix[
                    f"confusion_matrix_level{l}"]

    def get_stats(self):
        self.nt_per_class = {}
        results_dict = {}
        for l in range(len(self.hierarchy_names)):
            stats_l = [
                torch.cat(x, 0).cpu().numpy()
                for x in zip(*self.stats[f"stats_level{l}"])
            ]

            self.metrics[f"metrics_level{l}"].process(*stats_l)
            self.nt_per_class[f"nt_per_class_level{l}"] = np.bincount(
                stats_l[-1].astype(int),
                minlength=len(self.hierarchy_names[str(
                    l)]))  # number of targets per class
        for l in range(len(self.hierarchy_names)):
            results_dict[f"results_dict_level{l}"] = self.metrics[
                f'metrics_level{l}'].results_dict
        return results_dict

    def calculate_set_metrics(self):

        # Calculate number of ground true classes and hierarchy depth
        len_gt_classes = len(self.gt_cls)
        hier_depth = len(self.gt_cls[0])

        # For each GT class get the number of its ancestors (assuming all paths are correct)
        # .fill should be faster than .full below
        num_ancest_GT = np.empty(len_gt_classes, dtype=int)
        num_ancest_GT.fill(hier_depth)

        # Calculate how many predictions per object
        num_pred_per_class = np.array([len(cls) for cls in self.pred_cls])

        # For each prediction get the number of its ancestors,
        # assuming the missings are already filled with Nones
        # the .fill method should be faster
        total_num_pred = np.sum(num_pred_per_class)
        num_ancest_pred = np.empty(total_num_pred, dtype=int)
        num_ancest_pred.fill(hier_depth)

        # Find the common ancestors
        count_comm_ancest_list = []

        for j in range(len_gt_classes):  # loop1
            count_comm_ancest = 0

            for i in range(num_pred_per_class[j]):  # loop2
                count_comm_ancest = 0
                count_comm_ancest += sum(  # loop3
                    int(self.gt_cls[j][k]) == self.pred_cls[j][i][k]
                    for k in range(hier_depth))
                count_comm_ancest_list.append(count_comm_ancest)

        # If multiple predictions for the same ground truth, take the min F1
        F1_list = []
        F1_updated = []
        cnt_iter = 0
        F1_min = 1
        cnt_num_pred = 0
        num_pred = num_pred_per_class[cnt_num_pred]
        for i in range(len(count_comm_ancest_list)):  # loop4
            cnt_iter = cnt_iter + 1
            Prec = count_comm_ancest_list[
                i] / hier_depth  # all predictions have hier_depth number of ancestors
            Recall = count_comm_ancest_list[
                i] / hier_depth  # all ground truths have the same number of ancestors

            if Prec == 0 and Recall == 0:
                F1 = 0
            else:
                F1 = (2 * Prec * Recall) / (Prec + Recall)
            F1_list.append(F1)
            F1_min = min(F1_min, F1)
            if cnt_iter == num_pred:
                F1_updated.append(F1)
                cnt_iter = 0
                cnt_num_pred = cnt_num_pred + 1
                if cnt_num_pred < len(num_pred_per_class):
                    num_pred = num_pred_per_class[cnt_num_pred]

        # Convert the list to dictionary
        F1_dict = defaultdict(list)

        for i in range(len_gt_classes):
            key = int(self.gt_cls[i][hier_depth - 1])
            value = F1_updated[i]
            F1_dict[key].append(value)

        # Convert the defaultdict to a regular dictionary
        F1_dict = dict(F1_dict)

        # For all classes present in fp_classes_list, append F1=0 to that class to penalize
        # non-existent ground truths predicted to be this class
        # For FP with None in the terminal node, create a background class and calculate its F1
        # for i in range(len(self.fps_cls)):

        # Convert the list to dictionary
        F1_dict = defaultdict(list, F1_dict)

        for i in range(len(self.fps_cls)):
            key = self.fps_cls[i][hier_depth - 1]
            value = 0
            if key is None:
                cnt_nones = 1
                key = "background"

                cnt_nones += sum(self.fps_cls[i][j] is None
                                 for j in range(hier_depth - 1))
                value = cnt_nones / hier_depth
            F1_dict[key].append(value)

        # Convert the defaultdict to a regular dictionary
        F1_dict = dict(F1_dict)

        F1_mean_dict = {k: np.mean(v) for k, v in F1_dict.items()}

        return F1_mean_dict

    def count_numbers(self):
        # number of objects in image
        num_gt_objects = len(self.gt_cls)

        # number of predictions per object
        num_pred_per_object = [len(pred) for pred in self.pred_cls]

        # How many objects with 1 pred, how many with 2 pred, etc
        cnt_dict = Counter(num_pred_per_object)

        # Turn dictionary into a dataframe for readability
        df = pd.DataFrame(cnt_dict.items(),
                          columns=['num_predictions', 'num_objects'])
        # Shift column 'num_objects' to the first position
        df = df[['num_objects', 'num_predictions']]
        print(df)

        # Count the number of Nones
        cnt_nones_per_pred = [
            pred.count(None) for sublist in self.pred_cls for pred in sublist
        ]

        cnt_nones = Counter(cnt_nones_per_pred)
        df_nones = pd.DataFrame(cnt_nones.items(),
                                columns=['num_nones', 'num_objects'])
        # Shift column 'num_objects' to the first position
        df_nones = df_nones[['num_objects', 'num_nones']]
        print(df_nones)
        return df, df_nones

    def print_results(self):
        DF_eval_list = list()
        eval_metrics_all = pd.DataFrame(columns=[
            'hierarchy_names', 'class_number', 'Class', 'Images', 'Instances',
            'Precision', 'Recall', 'F1', 'mAP50', 'mAP50-95', 'TP_count',
            'FP_count', 'Same_sub_count', 'Diff_sub_count', 'Same_sub_frac',
            'Diff_sub_frac'
        ])
        round_to = 3  # decimals to round to
        for l in range(len(self.hierarchy_names)):
            # LOGGER.info(f"Metrics for Hierarcical Level {l}")
            pf = '%11s' + '%60s' + '%11i' * 2 + '%11.3g' * len(
                self.metrics[f"metrics_level{l}"].keys[:4])  # print format
            # pf_full = '%11s' + '%60s' + '%11i' * 2 + '%11.3g' * len(
            #     self.metrics[f"metrics_level{l}"].keys)
            LOGGER.info(pf %
                        (f"H-Level{l}", 'all', self.seen,
                         self.nt_per_class[f"nt_per_class_level{l}"].sum(),
                         *self.metrics[f"metrics_level{l}"].mean_results()))

            if self.nt_per_class[f"nt_per_class_level{l}"].sum() == 0:
                LOGGER.warning(
                    f'WARNING тЪая╕П no labels found in {self.args.task} set, can not compute metrics without labels'
                )

            # Get metrics for all classes combined
            if not self.training:
                eval_metrics_all.loc[l, 'hierarchy_names'] = f"H-Level{l}"
                eval_metrics_all.loc[l, 'class_number'] = None
                eval_metrics_all.loc[l, 'Class'] = 'all'
                eval_metrics_all.loc[l, 'Images'] = self.seen
                eval_metrics_all.loc[l, 'Instances'] = self.nt_per_class[
                    f"nt_per_class_level{l}"].sum()
                eval_metrics_all.loc[l, 'Precision'] = np.round(
                    self.metrics[f"metrics_level{l}"].mean_results()[0],
                    round_to)
                eval_metrics_all.loc[l, 'Recall'] = np.round(
                    self.metrics[f"metrics_level{l}"].mean_results()[1],
                    round_to)
                if eval_metrics_all.loc[
                        l, 'Precision'] == 0 and eval_metrics_all.loc[
                            l, 'Recall'] == 0:
                    eval_metrics_all.loc[l, 'F1'] = 0
                else:
                    eval_metrics_all.loc[l, 'F1'] = np.round(
                        (2 * eval_metrics_all.loc[l, 'Precision'] *
                         eval_metrics_all.loc[l, 'Recall']) /
                        (eval_metrics_all.loc[l, 'Precision'] +
                         eval_metrics_all.loc[l, 'Recall']), round_to)
                eval_metrics_all.loc[l, 'mAP50'] = np.round(
                    self.metrics[f"metrics_level{l}"].mean_results()[2],
                    round_to)
                eval_metrics_all.loc[l, 'mAP50-95'] = np.round(
                    self.metrics[f"metrics_level{l}"].mean_results()[3],
                    round_to)

        # Get metrics for individual classes
        for l in range(len(self.hierarchy_names)):
            if not self.training:
                eval_metrics_df = pd.DataFrame(columns=[
                    'hierarchy_names',
                    'class_number',
                    'Class',
                    'Images',
                    'Instances',
                    'Precision',
                    'Recall',
                    'F1',
                    'mAP50',
                    'mAP50-95',
                    'tp_conf',
                    'fp_fn_conf',
                    'min_tp_conf',
                    'max_tp_conf',
                    'std_tp_conf',
                    'min_fp_fn_conf',
                    'max_fp_fn_conf',
                    'std_fp_fn_conf',
                ])
            if self.args.verbose and not self.training and self.nc > 1 and len(
                    self.stats[f"stats_level{l}"]):
                #LOGGER.info(f"Metrics Per Class for Hierarcical Level {l}")
                for i, c in enumerate(
                        self.metrics[f"metrics_level{l}"].ap_class_index):
                    LOGGER.info(
                        pf %
                        (f"H-Level{l}", self.hierarchy_names[str(l)][str(c)],
                         self.seen,
                         self.nt_per_class[f"nt_per_class_level{l}"][c],
                         *self.metrics[f"metrics_level{l}"].class_result(i)[:4]
                         ))

                    eval_metrics_df.loc[c, 'hierarchy_names'] = f"H-Level{l}"
                    eval_metrics_df.loc[c, 'class_number'] = self.metrics[
                        f"metrics_level{l}"].ap_class_index[i]
                    eval_metrics_df.loc[c, 'Class'] = self.hierarchy_names[str(
                        l)][str(c)]
                    eval_metrics_df.loc[c, 'Images'] = self.seen
                    eval_metrics_df.loc[c, 'Instances'] = self.nt_per_class[
                        f"nt_per_class_level{l}"][c]
                    eval_metrics_df.loc[c, 'Precision'] = np.round(
                        self.metrics[f"metrics_level{l}"].class_result(i)[0],
                        round_to)
                    eval_metrics_df.loc[c, 'Recall'] = np.round(
                        self.metrics[f"metrics_level{l}"].class_result(i)[1],
                        round_to)
                    if (eval_metrics_df.loc[c, 'Precision']
                            == 0) and (eval_metrics_df.loc[c, 'Recall'] == 0):
                        eval_metrics_df.loc[c, 'F1'] = 0
                    else:
                        eval_metrics_df.loc[c, 'F1'] = np.round(
                            (2 * eval_metrics_df.loc[c, 'Precision'] *
                             eval_metrics_df.loc[c, 'Recall']) /
                            (eval_metrics_df.loc[c, 'Precision'] +
                             eval_metrics_df.loc[c, 'Recall']), round_to)

                    for column in range(2, 12):
                        if np.isnan(
                                self.metrics[f"metrics_level{l}"].class_result(
                                    i)[column]):
                            eval_metrics_df.loc[
                                c, eval_metrics_df.columns[column + 6]] = "NaN"
                        else:
                            eval_metrics_df.loc[c, eval_metrics_df.columns[
                                column + 6]] = np.round(
                                    self.metrics[f"metrics_level{l}"].
                                    class_result(i)[column], round_to)

                    # If set_metrics is on, append to the df at the last level
                    if l == len(self.hierarchy_names
                                ) - 1 and self.args.calc_set_metric:
                        set_metric_df = pd.DataFrame.from_dict(
                            self.set_metric, orient='index').reset_index()
                        set_metric_df.rename(columns={
                            'index': 'class_number',
                            0: 'set_metric'
                        },
                                             inplace=True)
                        eval_metrics_df_with_set = eval_metrics_df.merge(
                            set_metric_df, on=['class_number'], how='left')
                        path = self.save_dir
                        eval_metrics_df_with_set.to_csv(
                            os.path.join(
                                path, r'eval_metrics_last_level_with_set.csv'))
                DF_eval_list.append(eval_metrics_df)

        if not self.training:
            eval_metrics_df_concat = pd.concat(DF_eval_list,
                                               axis=0,
                                               ignore_index=False)

            eval_metrics_all = eval_metrics_all.drop([
                'TP_count', 'FP_count', 'Same_sub_count', 'Diff_sub_count',
                'Same_sub_frac', 'Diff_sub_frac'
            ],
                                                     axis=1)
            eval_metrics_df_concat = pd.concat(
                [eval_metrics_all, eval_metrics_df_concat],
                axis=0,
                ignore_index=False)
            columns_to_round = ['Precision', 'Recall', 'mAP50', 'mAP50-95']
            eval_metrics_df_concat[columns_to_round] = np.round(
                eval_metrics_df_concat[columns_to_round], 3)

            path = self.save_dir
            eval_metrics_df_concat.to_csv(
                os.path.join(path, r'eval_metrics_prec_recall_mAP.csv'))

        if self.args.plots:
            for l in range(len(self.hierarchy_names)):
                self.confusion_matrix[f"confusion_matrix_level{l}"].plot(
                    save_dir=self.save_dir,
                    names=list(self.hierarchy_names[str(l)].values()))


        # If calc_TP_FN_FP is on, append the metric to the evaluation results dataframe
        if self.args.calc_TP_FN_FP and not self.training:
            eval_results_with_TP_FP_lst = []
            perc_same_lst = [0]
            TP_FN_FPsame_FPdiff_dict_all_levels = []
            for l in range(len(self.hierarchy_names)):
                TP_FN_FPsame_FPdiff_dict = self.confusion_matrix[
                    f"confusion_matrix_level{l}"].extract_tp_fp_fn()
                if l == 0:
                    TP_FN_FPsame_FPdiff_df = pd.DataFrame.from_dict(
                        TP_FN_FPsame_FPdiff_dict,
                        orient='index',
                        columns=['TP_count', 'FP_count'])
                    TP_FN_FPsame_FPdiff_df['Same_sub_count'] = None
                    TP_FN_FPsame_FPdiff_df['Diff_sub_count'] = None
                    TP_FN_FPsame_FPdiff_df['Same_sub_frac'] = None
                    TP_FN_FPsame_FPdiff_df['Diff_sub_frac'] = None
                    TP_FN_FPsame_FPdiff_df.index.name = 'class_number'
                    TP_FN_FPsame_FPdiff_df.reset_index(inplace=True)
                else:
                    TP_FN_FPsame_FPdiff_df = pd.DataFrame.from_dict(
                        TP_FN_FPsame_FPdiff_dict,
                        orient='index',
                        columns=[
                            'TP_count', 'FP_count', 'Same_sub_count',
                            'Diff_sub_count', 'Same_sub_frac', 'Diff_sub_frac'
                        ])
                    TP_FN_FPsame_FPdiff_df.index.name = 'class_number'
                    TP_FN_FPsame_FPdiff_df.reset_index(inplace=True)

                if not self.training:
                    eval_results_with_TP_FP_df = DF_eval_list[l].merge(
                        TP_FN_FPsame_FPdiff_df,
                        on=['class_number'],
                        how='left')
                    eval_results_with_TP_FP_lst.append(
                        eval_results_with_TP_FP_df)

                    eval_results_with_TP_FP_concat = pd.concat(
                        eval_results_with_TP_FP_lst,
                        axis=0,
                        ignore_index=False)

                TP_FN_FPsame_FPdiff_dict_all_levels.append(
                    TP_FN_FPsame_FPdiff_dict)

                # Calculate the metrics per class
                cnts_tp = 0
                cnts_fp = 0
                for value_list in TP_FN_FPsame_FPdiff_dict.values():
                    cnts_tp += value_list[0]
                    cnts_fp += value_list[1]
                eval_metrics_all.loc[l, 'TP_count'] = cnts_tp
                eval_metrics_all.loc[l, 'FP_count'] = cnts_fp

                # For all classes calculate the percentage preds in the same and in different graphs
                if l > 0:
                    cnts_same_subg = 0
                    cnts_diff_subg = 0
                    for value_list in TP_FN_FPsame_FPdiff_dict.values():
                        cnts_same_subg += value_list[2]
                        cnts_diff_subg += value_list[3]
                    sum_cnts = cnts_same_subg + cnts_diff_subg
                    eval_metrics_all.loc[l, 'Same_sub_count'] = cnts_same_subg
                    eval_metrics_all.loc[l, 'Diff_sub_count'] = cnts_diff_subg
                    if sum_cnts > 0:
                        perc_same = np.round(cnts_same_subg / sum_cnts,
                                             round_to)
                        perc_diff = np.round(cnts_diff_subg / sum_cnts,
                                             round_to)
                    else:
                        perc_same = 0
                        perc_diff = 0
                    eval_metrics_all.loc[l, 'Same_sub_frac'] = perc_same
                    eval_metrics_all.loc[l, 'Diff_sub_frac'] = perc_diff

                    perc_same_lst.append(perc_same)
            print(perc_same_lst)

            if not self.training:
                eval_results_with_TP_FP_concat = pd.concat(
                    [eval_metrics_all, eval_results_with_TP_FP_concat],
                    axis=0,
                    ignore_index=False)
                eval_results_with_TP_FP_concat = np.round(
                    eval_results_with_TP_FP_concat, 3)
                eval_results_with_TP_FP_concat.to_csv(
                    os.path.join(path, r'eval_results_with_TP_FP_concat.csv'))
            print(
                f"TP_FN_FPsame_FPdiff_dict_all_levels {TP_FN_FPsame_FPdiff_dict_all_levels}"
            )
            print(f"perc_same_lst {perc_same_lst}")

    def _process_batch(self, detections, labels):
        """
        Return correct prediction matrix
        Arguments:
            detections (array[N, 6]), x1, y1, x2, y2, conf, class
            labels (array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (array[N, 10]), for 10 IoU levels
        """
        iou = box_iou(labels[:, 1:], detections[:, :4])
        correct = np.zeros(
            (detections.shape[0], self.iouv.shape[0])).astype(bool)
        correct_class = labels[:, 0:1] == detections[:, 5]

        for i in range(len(self.iouv)):
            x = torch.where(
                (iou >= self.iouv[i])
                & correct_class)  # IoU > threshold and classes match
            if x[0].shape[0]:
                matches = torch.cat(
                    (torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                    1).cpu().numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 1],
                                                return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(matches[:, 0],
                                                return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct,
                            dtype=torch.bool,
                            device=detections.device)

    def get_dataloader(self, dataset_path, batch_size):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 cache=False,
                                 pad=0.5,
                                 rect=self.args.rect,
                                 workers=self.args.workers,
                                 prefix=colorstr(f'{self.args.mode}: '),
                                 shuffle=False,
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, names=self.data['names'],
                             mode='val')[0] #, hierarchy=self.data['hierarchy_names']

    def plot_val_samples(self, batch, ni):
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'],
                    batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'val_batch{ni}_labels.jpg',
                    names=self.names)

    def plot_predictions(self, batch, preds, ni):
        for l in range(len(self.hierarchy_names)):
            plot_images(batch['img'],
                        *output_to_target(preds[f"preds_out_level{l}"],
                                          max_det=30),
                        paths=batch['im_file'],
                        fname=self.save_dir /
                        f'val_batch{ni}_pred_level{l}.jpg',
                        names={
                            int(k): v
                            for k, v in self.hierarchy_names[str(l)].items()
                        })  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                    gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls,
                                                         *xywh)  # label format
            with open(file, 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

    def pred_to_json(self, predn, filename):
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append({
                'image_id': image_id,
                'category_id': self.class_map[int(p[5])],
                'bbox': [round(x, 3) for x in b],
                'score': round(p[4], 5)
            })

    def eval_json(self, stats):
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data[
                'path'] / 'annotations/instances_val2017.json'  # annotations
            pred_json = self.save_dir / 'predictions.json'  # predictions
            LOGGER.info(
                f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...'
            )
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                check_requirements('pycocotools>=2.0.6')
                from pycocotools.coco import COCO  # noqa
                from pycocotools.cocoeval import COCOeval  # noqa

                for x in anno_json, pred_json:
                    assert x.is_file(), f'{x} file not found'
                anno = COCO(str(anno_json))  # init annotations api
                pred = anno.loadRes(
                    str(pred_json
                        ))  # init predictions api (must pass string, not Path)
                eval = COCOeval(anno, pred, 'bbox')
                if self.is_coco:
                    eval.params.imgIds = [
                        int(Path(x).stem)
                        for x in self.dataloader.dataset.im_files
                    ]  # images to eval
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[
                    -2]] = eval.stats[:2]  # update mAP50-95 and mAP50
            except Exception as e:
                LOGGER.warning(f'pycocotools unable to run: {e}')
        return stats


def val(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'

    args = dict(model=model, data=data)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).val(**args)
    else:
        validator = DetectionValidator(args=args)
        validator(model=args['model'])


if __name__ == '__main__':
    val()
