"""Module containing the trainer of the transoar project."""

import torch
from torch.optim import optimizer
from tqdm import tqdm

class Trainer:

    def __init__(
        self, train_loader, val_loader, model, criterion, optimizer, scheduler,
        device, config
    ):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._train_config = config['training']

    def _train_one_epoch(self):
        self._model.train()
        for data, mask, bboxes, _ in tqdm(self._train_loader):
            # Put data to gpu
            data, mask = data.to(device=self._device), mask.to(device=self._device)
        
            targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': torch.tensor(item[1]).to(device=self._device) # TODO
                }
                targets.append(target)

            # Make prediction 
            out = self._model(data, mask)
            loss = self._criterion(out, targets)

            self._optimizer.zero_grad()
            loss.backward()

            # Clip grads to counter exploding grads
            max_norm = self._train_config['clip_max_norm']
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)

            self._optimizer.step()
            print(loss)

    @torch.no_grad()
    def _validate(self):
        self._model.eval()
        for data, mask, bboxes, _ in tqdm(self._val_loader):
            # Put data to gpu
            data, mask = data.to(device=self._device), mask.to(device=self._device)
        
            targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': torch.tensor(item[1]).to(device=self._device) # TODO
                }
                targets.append(target)

            # Make prediction 
            out = self._model(data, mask)
            loss = self._criterion(out, targets)

    def run(self):
        for epoch in range(1, self._train_config['epochs'] + 1):
            self._train_one_epoch()
            self._scheduler.step()

            if epoch % self._train_config['val_interval'] == 0:
                self._validate()
