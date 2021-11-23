"""Module containing the trainer of the transoar project."""

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transoar.evaluator import DetectionEvaluator

class Trainer:

    def __init__(
        self, train_loader, val_loader, model, criterion, optimizer, scheduler,
        device, config, path_to_run
    ):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device

        self._train_config = config['training']
        self._writer = SummaryWriter(log_dir=path_to_run)

        self._evaluator = DetectionEvaluator(
            classes=list(config['data']['labels'].keys())
        )

    def _train_one_epoch(self, num_epoch):
        self._model.train()
        for idx, (data, mask, bboxes, _) in enumerate(tqdm(self._train_loader)):
            # Put data to gpu
            data, mask = data.to(device=self._device), mask.to(device=self._device)
        
            targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                targets.append(target)

            # Make prediction 
            out = self._model(data, mask)
            loss_bbox, loss_giou, loss_cls = self._criterion(out, targets)
            loss = sum(
                [
                    loss_bbox * self._train_config['loss_coefs']['bbox_loss_coef'],
                    loss_giou * self._train_config['loss_coefs']['giou_loss_coef'],
                    loss_cls * self._train_config['loss_coefs']['cls_loss_coef'],
                ]
            )

            self._optimizer.zero_grad()
            loss.backward()

            # Clip grads to counter exploding grads
            max_norm = self._train_config['clip_max_norm']
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)

            self._optimizer.step()

            if idx % self._train_config['log_iter'] == 0:
                # Write to logger
                num_idx = (num_epoch - 1) * len(self._train_loader) + idx
                self._write_to_logger(
                    num_idx, 'train', 
                    total_loss=loss.item(),
                    bbox_loss=loss_bbox.item(),
                    giou_loss=loss_giou.item(),
                    cls_loss=loss_cls.item()
                )

    @torch.no_grad()
    def _validate(self, num_epoch):
        self._model.eval()
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
                    'labels': item[1].to(device=self._device)
                }
                targets.append(target)

            # Make prediction 
            out = self._model(data, mask)
            loss_bbox, loss_giou, loss_cls = self._criterion(out, targets)
            loss = sum(
                [
                    loss_bbox * self._train_config['loss_coefs']['bbox_loss_coef'],
                    loss_giou * self._train_config['loss_coefs']['giou_loss_coef'],
                    loss_cls * self._train_config['loss_coefs']['cls_loss_coef'],
                ]
            )

            loss_agg += loss.item()
            loss_bbox_agg += loss_bbox.item()
            loss_giou_agg += loss_giou.item()
            loss_cls_agg += loss_cls.item()

        loss = loss_agg / len(self._val_loader)
        loss_bbox = loss_bbox_agg / len(self._val_loader)
        loss_giou = loss_giou_agg / len(self._val_loader)
        loss_cls = loss_cls_agg / len(self._val_loader)

        # Write to logger
        self._write_to_logger(
            num_epoch, 'val', 
            total_loss=loss,
            bbox_loss=loss_bbox,
            giou_loss=loss_giou,
            cls_loss=loss_cls
        )

    def run(self):
        for epoch in range(1, self._train_config['epochs'] + 1):
            self._train_one_epoch(epoch)

            # Log learning rates
            self._write_to_logger(
                epoch, 'lr',
                neck=self._optimizer.param_groups[0]['lr'],
                backbone=self._optimizer.param_groups[1]['lr']
            )

            if epoch % self._train_config['val_interval'] == 0:
                self._validate(epoch)

            self._scheduler.step()

    def _write_to_logger(self, num_epoch, category, **kwargs):
        for key, value in kwargs.items():
            name = category + '/' + key
            self._writer.add_scalar(name, value, num_epoch)
