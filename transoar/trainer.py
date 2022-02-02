"""Module containing the trainer of the transoar project."""

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transoar.evaluator import DetectionEvaluator
from transoar.inference import inference

class Trainer:

    def __init__(
        self, train_loader, val_loader, model, criterion, optimizer, scheduler,
        device, config, path_to_run, epoch, metric_start_val
    ):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._path_to_run = path_to_run
        self._epoch_to_start = epoch
        self._config = config

        self._writer = SummaryWriter(log_dir=path_to_run)

        self._evaluator = DetectionEvaluator(
            classes=list(config['labels'].values())
        )

        # Init main metric for checkpoint
        self._main_metric_key = 'mAP_IoU_0.10_0.50_0.05_MaxDet_1'
        self._main_metric_max_val = metric_start_val

    def _train_one_epoch(self, num_epoch):
        self._model.train()
        # self._criterion.train()

        loss_agg = 0
        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        for data, mask, bboxes, _ in tqdm(self._train_loader):
            # Put data to gpu
            data, mask = data.to(device=self._device), mask.to(device=self._device)
        
            targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device) - 1
                }
                targets.append(target)

            # Make prediction 
            out = self._model(data, mask)
            loss_dict = self._criterion(out, targets)

            # Create absolute loss and mult with loss coefficient
            loss_abs = 0
            for loss_key, loss_val in loss_dict.items():
                loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            self._optimizer.zero_grad()
            loss_abs.backward()

            # Clip grads to counter exploding grads
            max_norm = self._config['clip_max_norm']
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)

            self._optimizer.step()

            loss_agg += loss_abs.item()
            loss_bbox_agg += loss_dict['bbox'].item()
            loss_giou_agg += loss_dict['giou'].item()
            loss_cls_agg += loss_dict['cls'].item()

        loss = loss_agg / len(self._train_loader)
        loss_bbox = loss_bbox_agg / len(self._train_loader)
        loss_giou = loss_giou_agg / len(self._train_loader)
        loss_cls = loss_cls_agg / len(self._train_loader)

        self._write_to_logger(
            num_epoch, 'train', 
            total_loss=loss,
            bbox_loss=loss_bbox,
            giou_loss=loss_giou,
            cls_loss=loss_cls
        )

    @torch.no_grad()
    def _validate(self, num_epoch):
        self._model.eval()
        # self._criterion.eval()

        loss_agg = 0
        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        for data, mask, bboxes, _ in tqdm(self._val_loader):
            # Put data to gpu
            data, mask = data.to(device=self._device), mask.to(device=self._device)
        
            targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device) - 1
                }
                targets.append(target)

            # Make prediction 
            out = self._model(data, mask)
            loss_dict = self._criterion(out, targets)

            # Create absolute loss and mult with loss coefficient
            loss_abs = 0
            for loss_key, loss_val in loss_dict.items():
                loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            # Evaluate validation predictions based on metric
            pred_boxes, pred_classes, pred_scores = inference(out)
            self._evaluator.add(
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                gt_boxes=[target['boxes'].detach().cpu().numpy() for target in targets],
                gt_classes=[target['labels'].detach().cpu().numpy() for target in targets]
            )

            loss_agg += loss_abs.item()
            loss_bbox_agg += loss_dict['bbox'].item()
            loss_giou_agg += loss_dict['giou'].item()
            loss_cls_agg += loss_dict['cls'].item()

        loss = loss_agg / len(self._val_loader)
        loss_bbox = loss_bbox_agg / len(self._val_loader)
        loss_giou = loss_giou_agg / len(self._val_loader)
        loss_cls = loss_cls_agg / len(self._val_loader)
        metric_scores = self._evaluator.eval()
        self._evaluator.reset()

        # Check if new best checkpoint
        if metric_scores[self._main_metric_key] >= self._main_metric_max_val \
            and not self._config['debug_mode']:
            self._main_metric_max_val = metric_scores[self._main_metric_key]
            self._save_checkpoint(
                num_epoch,
                f'model_best_{metric_scores[self._main_metric_key]:.3f}.pt'
            )

        # Write to logger
        self._write_to_logger(
            num_epoch, 'val', 
            total_loss=loss,
            bbox_loss=loss_bbox,
            giou_loss=loss_giou,
            cls_loss=loss_cls
        )

        self._write_to_logger(
            num_epoch, 'val_metric',
            mAP=metric_scores['mAP_IoU_0.10_0.50_0.05_MaxDet_1'],
            AP10=metric_scores['AP_IoU_0.10_MaxDet_1'],
            AP20=metric_scores['AP_IoU_0.20_MaxDet_1'],
            AP30=metric_scores['AP_IoU_0.30_MaxDet_1'],
            AP40=metric_scores['AP_IoU_0.40_MaxDet_1'],
            AP50=metric_scores['AP_IoU_0.50_MaxDet_1'],
            AP60=metric_scores['AP_IoU_0.60_MaxDet_1'],
            AP70=metric_scores['AP_IoU_0.70_MaxDet_1'],
            AP80=metric_scores['AP_IoU_0.80_MaxDet_1'],
            AP90=metric_scores['AP_IoU_0.90_MaxDet_1'],
        )

    def run(self):
        if self._epoch_to_start == 0:   # For initial performance estimation
            self._validate(0)

        for epoch in range(self._epoch_to_start + 1, self._config['epochs'] + 1):
            self._train_one_epoch(epoch)

            # Log learning rates
            self._write_to_logger(
                epoch, 'lr',
                neck=self._optimizer.param_groups[0]['lr'],
                backbone=self._optimizer.param_groups[1]['lr']
            )

            if epoch % self._config['val_interval'] == 0:
                self._validate(epoch)

            self._scheduler.step()

            if not self._config['debug_mode']:
                self._save_checkpoint(epoch, 'model_last.pt')

    def _write_to_logger(self, num_epoch, category, **kwargs):
        for key, value in kwargs.items():
            name = category + '/' + key
            self._writer.add_scalar(name, value, num_epoch)

    def _save_checkpoint(self, num_epoch, name):
        # Delete prior best checkpoint
        if 'best' in name:
            [path.unlink() for path in self._path_to_run.iterdir() if 'best' in str(path)]

        torch.save({
            'epoch': num_epoch,
            'metric_max_val': self._main_metric_max_val,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
        }, self._path_to_run / name)
