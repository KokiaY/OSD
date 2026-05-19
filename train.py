import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger,WandbLogger
import random




def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


def get_trainer_kwargs(config):
    trainer_kwargs = dict(
        max_epochs=config.max_epoch,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        callbacks=[],
        logger=None,
        enable_progress_bar=getattr(config, "enable_progress_bar", True),
        inference_mode=getattr(config, "inference_mode", False),
        num_sanity_val_steps=getattr(config, "num_sanity_val_steps", 2),
    )
    accelerator = getattr(config, "accelerator", None)
    devices = getattr(config, "devices", None)
    gpus = getattr(config, "gpus", None)

    if accelerator is None:
        if torch.cuda.is_available() and gpus:
            accelerator = "gpu"
            devices = gpus
        else:
            accelerator = "cpu"
            devices = 1
    elif devices is None:
        devices = gpus if accelerator == "gpu" and gpus else 1

    trainer_kwargs["accelerator"] = accelerator
    trainer_kwargs["devices"] = devices

    strategy = getattr(config, "strategy", None)
    if strategy:
        trainer_kwargs["strategy"] = strategy

    precision = getattr(config, "precision", None)
    if precision:
        trainer_kwargs["precision"] = precision

    accumulate_grad_batches = getattr(config, "accumulate_grad_batches", None)
    if accumulate_grad_batches:
        trainer_kwargs["accumulate_grad_batches"] = accumulate_grad_batches

    return trainer_kwargs


def set_runtime_precision():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def should_fallback_to_cpu(exc):
    message = str(exc).lower()
    fallback_messages = (
        "no kernel image is available for execution on the device",
        "cuda error",
        "no cuda gpus are available",
        "invalid device function",
    )
    return any(item in message for item in fallback_messages)


def probe_trainer_kwargs(trainer_kwargs):
    if trainer_kwargs.get("accelerator") != "gpu":
        return trainer_kwargs
    try:
        device = torch.device("cuda:0")
        tensor = torch.zeros(1, device=device)
        tensor = tensor + 1
        del tensor
        torch.cuda.synchronize()
        return trainer_kwargs
    except RuntimeError as exc:
        if not should_fallback_to_cpu(exc):
            raise
        print(f"CUDA runtime is not compatible with the current PyTorch build, fallback to CPU. Details: {exc}")
        fallback_kwargs = dict(trainer_kwargs)
        fallback_kwargs.pop("strategy", None)
        fallback_kwargs["accelerator"] = "cpu"
        fallback_kwargs["devices"] = 1
        return fallback_kwargs


class EMACallback(Callback):
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.num_updates = 0

    def _initialize(self, pl_module):
        if self.shadow:
            return
        for name, parameter in pl_module.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.detach().clone()

    def on_train_start(self, trainer, pl_module):
        self._initialize(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._initialize(pl_module)
        self.num_updates += 1
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for name, parameter in pl_module.named_parameters():
                if name not in self.shadow:
                    continue
                self.shadow[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    def _apply_shadow(self, pl_module):
        if not self.shadow or self.backup:
            return
        with torch.no_grad():
            for name, parameter in pl_module.named_parameters():
                if name not in self.shadow:
                    continue
                self.backup[name] = parameter.detach().clone()
                parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))

    def _restore(self, pl_module):
        if not self.backup:
            return
        with torch.no_grad():
            for name, parameter in pl_module.named_parameters():
                if name in self.backup:
                    parameter.copy_(self.backup[name].to(device=parameter.device, dtype=parameter.dtype))
        self.backup = {}

    def on_validation_start(self, trainer, pl_module):
        self._apply_shadow(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._restore(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not self.shadow:
            return
        state_dict = checkpoint.get("state_dict", {})
        for name, shadow_parameter in self.shadow.items():
            if name in state_dict:
                state_dict[name] = shadow_parameter.to(
                    device=state_dict[name].device,
                    dtype=state_dict[name].dtype,
                ).clone()


class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net

        self.loss = config.loss

        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x):
        seg_pre = self.net(x)
        return seg_pre

    def align_prediction_to_mask(self, prediction, mask):
        target_size = mask.shape[-2:]
        if isinstance(prediction, torch.Tensor):
            if prediction.ndim < 4:
                return prediction
            if prediction.shape[-2:] == target_size:
                return prediction
            return F.interpolate(prediction, size=target_size, mode="bilinear", align_corners=False)
        if isinstance(prediction, (tuple, list)):
            return type(prediction)(self.align_prediction_to_mask(item, mask) for item in prediction)
        return prediction

    def metric_indices(self):
        if hasattr(self.config, "metric_include_indices"):
            return list(self.config.metric_include_indices)
        if hasattr(self.config, "metric_exclude_indices"):
            excluded = set(self.config.metric_exclude_indices)
            return [index for index in range(self.config.num_classes) if index not in excluded]
        if any(name in self.config.log_name for name in ("vaihingen", "potsdam", "whubuilding", "massbuilding", "cropland")):
            return list(range(self.config.num_classes - 1))
        return list(range(self.config.num_classes))

    def summarize_metrics(self, evaluator):
        indices = self.metric_indices()
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        mIoU = np.nanmean(iou_per_class[indices])
        F1 = np.nanmean(f1_per_class[indices])

        confusion = evaluator.confusion_matrix
        support = confusion.sum(axis=1)
        tp = np.diag(confusion)
        denominator = support[indices].sum() + evaluator.eps
        OA = tp[indices].sum() / denominator
        return mIoU, F1, OA, iou_per_class

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        loss = 0
        prediction = self.net(img)
        if self.config.get("has_contrastive_loss",False):
            contrastive_loss_weight = self.config.get("contrastive_loss_weight", 1.0)
            loss += contrastive_loss_weight * prediction[-1]
            prediction = prediction[:-1]
        prediction = self.align_prediction_to_mask(prediction, mask)
        loss += self.loss(prediction, mask)

        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        return {"loss": loss}

    def on_train_epoch_end(self):
        epoch_index = self.current_epoch + 1
        max_epochs = getattr(self.config, "max_epoch", "?")
        mIoU, F1, OA, iou_per_class = self.summarize_metrics(self.metrics_train)
        eval_value = {'train_mIoU': mIoU,
                      'train_F1': F1,
                      'train_OA': OA}
        print(f'train epoch {epoch_index}/{max_epochs}:', eval_value)

        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(f'train_iou epoch {epoch_index}/{max_epochs}:', iou_value)
        self.metrics_train.reset()
        log_dict = {'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        prediction = self.forward(img)
        prediction = self.align_prediction_to_mask(prediction, mask)
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].cpu().numpy(), pre_mask[i].cpu().numpy())

        loss_val = self.loss(prediction, mask)
        return {"loss_val": loss_val}

    def on_validation_epoch_end(self):
        epoch_index = self.current_epoch + 1
        max_epochs = getattr(self.config, "max_epoch", "?")
        mIoU, F1, OA, iou_per_class = self.summarize_metrics(self.metrics_val)

        eval_value = {'val_mIoU': mIoU,
                      'val_F1': F1,
                      'val_OA': OA}
        print(f'val epoch {epoch_index}/{max_epochs}:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(f'val_iou epoch {epoch_index}/{max_epochs}:', iou_value)

        self.metrics_val.reset()
        log_dict = {'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)
    set_runtime_precision()

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    # logger = [CSVLogger('logs', name=config.log_name),WandbLogger(name=config.wandb_name,save_dir=config.wand_dir,project=config.wandb_project)]
    
    logger = CSVLogger('logs', name=config.log_name)
    
    model = Supervision_Train(config)
    if config.pretrained_ckpt_path:
        model = Supervision_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer_kwargs = get_trainer_kwargs(config)
    trainer_kwargs = probe_trainer_kwargs(trainer_kwargs)
    callbacks = [checkpoint_callback]
    if getattr(config, "use_ema", False):
        callbacks.append(EMACallback(decay=getattr(config, "ema_decay", 0.999)))
    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["logger"] = logger
    trainer = pl.Trainer(**trainer_kwargs)
    try:
        trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)
    except RuntimeError as exc:
        if not should_fallback_to_cpu(exc) or trainer_kwargs.get("accelerator") == "cpu":
            raise
        print(f"Training failed on CUDA, retry on CPU. Details: {exc}")
        fallback_kwargs = dict(trainer_kwargs)
        fallback_kwargs.pop("strategy", None)
        fallback_kwargs["accelerator"] = "cpu"
        fallback_kwargs["devices"] = 1
        trainer = pl.Trainer(**fallback_kwargs)
        trainer.fit(model=model, ckpt_path=config.resume_ckpt_path)


if __name__ == "__main__":
   main()
