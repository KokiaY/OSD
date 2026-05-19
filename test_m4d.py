import argparse
import csv
import json
import multiprocessing as mp
import multiprocessing.pool as mpp
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach as tta
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from train import Supervision_Train
from tools.metric import Evaluator


PALETTE = {
    0: [0, 0, 0],
    1: [0, 255, 255],
    2: [255, 0, 0],
    3: [153, 76, 0],
    4: [0, 153, 0],
}


def label2rgb(mask):
    h, w = mask.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in PALETTE.items():
        mask_rgb[mask == cls_id] = color
    return cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)


def img_writer(inp):
    mask, mask_id, rgb = inp
    save_path = mask_id + ".png"
    if rgb:
        cv2.imwrite(save_path, label2rgb(mask))
    else:
        cv2.imwrite(save_path, mask.astype(np.uint8))


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, required=True, help="Path where to save resulting masks")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation")
    arg("--rgb", action="store_true", help="Whether output RGB masks")
    arg("--ckpt_path", type=Path, default=None, help="Checkpoint path, default uses config weights")
    arg("-b", "--batch_size", type=int, default=2, help="Batch size for testing")
    arg("--workers", type=int, default=1, help="Number of image writer processes")
    return parser.parse_args()


def get_device(config):
    if torch.cuda.is_available() and getattr(config, "gpus", None):
        return torch.device(f"cuda:{config.gpus[0]}")
    return torch.device("cpu")


def get_checkpoint_path(config, args):
    if args.ckpt_path is not None:
        return str(args.ckpt_path)
    return os.path.join(config.weights_path, config.test_weights_name + ".ckpt")


class ResizePredictionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        prediction = self.model(x)
        if prediction.shape[-2:] == x.shape[-2:]:
            return prediction
        return F.interpolate(prediction, size=x.shape[-2:], mode="bilinear", align_corners=False)


class SafeSegmentationTTAWrapper(nn.Module):
    def __init__(self, model, transforms):
        super().__init__()
        self.model = model
        self.transforms = transforms

    def forward(self, x):
        target_size = x.shape[-2:]
        merged_output = None
        transform_count = 0

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(x)
            augmented_output = self.model(augmented_image)
            deaugmented_output = transformer.deaugment_mask(augmented_output)
            if deaugmented_output.shape[-2:] != target_size:
                deaugmented_output = F.interpolate(
                    deaugmented_output,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )
            merged_output = deaugmented_output if merged_output is None else merged_output + deaugmented_output
            transform_count += 1

        return merged_output / transform_count


def build_tta_model(model, tta_mode):
    model = ResizePredictionWrapper(model)
    if tta_mode == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        return SafeSegmentationTTAWrapper(model, transforms)
    if tta_mode == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation="bicubic", align_corners=False),
            ]
        )
        return SafeSegmentationTTAWrapper(model, transforms)
    return model


def safe_float(value):
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def format_metric(value):
    if value is None:
        return "nan"
    return f"{value:.4f}"


def build_metrics_report(evaluator, class_names):
    iou = evaluator.Intersection_over_Union()
    precision = evaluator.Precision()
    recall = evaluator.Recall()
    f1 = evaluator.F1()
    oa = evaluator.OA()

    per_class = []
    for class_id, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "iou": safe_float(iou[class_id]),
                "precision": safe_float(precision[class_id]),
                "recall": safe_float(recall[class_id]),
                "f1": safe_float(f1[class_id]),
            }
        )

    summary = {
        "overall_accuracy": safe_float(oa),
        "mean_iou": safe_float(np.nanmean(iou)),
        "mean_precision": safe_float(np.nanmean(precision)),
        "mean_recall": safe_float(np.nanmean(recall)),
        "mean_f1": safe_float(np.nanmean(f1)),
    }
    return summary, per_class


def print_metrics(summary, per_class):
    print("=" * 100)
    print(
        "OA: {oa} | mIoU: {miou} | mPrecision: {mp} | mRecall: {mr} | mF1: {mf1}".format(
            oa=format_metric(summary["overall_accuracy"]),
            miou=format_metric(summary["mean_iou"]),
            mp=format_metric(summary["mean_precision"]),
            mr=format_metric(summary["mean_recall"]),
            mf1=format_metric(summary["mean_f1"]),
        )
    )
    print("-" * 100)
    print("{:<4} {:<24} {:>10} {:>10} {:>10} {:>10}".format("ID", "Class", "IoU", "Precision", "Recall", "F1"))
    print("-" * 100)
    for item in per_class:
        print(
            "{:<4} {:<24} {:>10} {:>10} {:>10} {:>10}".format(
                item["class_id"],
                item["class_name"][:24],
                format_metric(item["iou"]),
                format_metric(item["precision"]),
                format_metric(item["recall"]),
                format_metric(item["f1"]),
            )
        )
    print("=" * 100)


def save_metrics(output_path, summary, per_class):
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)

    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    with open(metrics_dir / "per_class_metrics.json", "w", encoding="utf-8") as per_class_file:
        json.dump(per_class, per_class_file, indent=2, ensure_ascii=False)

    with open(metrics_dir / "per_class_metrics.csv", "w", encoding="utf-8", newline="") as csv_file:
        fieldnames = ["class_id", "class_name", "iou", "precision", "recall", "f1"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_class:
            writer.writerow(row)

    with open(metrics_dir / "summary.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(
            "OA: {oa}\n"
            "mIoU: {miou}\n"
            "mPrecision: {mp}\n"
            "mRecall: {mr}\n"
            "mF1: {mf1}\n".format(
                oa=format_metric(summary["overall_accuracy"]),
                miou=format_metric(summary["mean_iou"]),
                mp=format_metric(summary["mean_precision"]),
                mr=format_metric(summary["mean_recall"]),
                mf1=format_metric(summary["mean_f1"]),
            )
        )


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    device = get_device(config)
    checkpoint_path = get_checkpoint_path(config, args)
    model = Supervision_Train.load_from_checkpoint(checkpoint_path, config=config, map_location="cpu")
    model = model.to(device)
    model.eval()
    model = build_tta_model(model, args.tta)

    test_dataset = getattr(config, "test_dataset", config.val_dataset)
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=getattr(config, "num_workers", 0),
            pin_memory=device.type == "cuda",
            drop_last=False,
            shuffle=False,
        )
        results = []
        for batch in tqdm(test_loader):
            raw_predictions = model(batch["img"].to(device))
            if raw_predictions.shape[-2:] != batch["gt_semantic_seg"].shape[-2:]:
                raw_predictions = F.interpolate(
                    raw_predictions,
                    size=batch["gt_semantic_seg"].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)
            masks_true = batch["gt_semantic_seg"]
            image_ids = batch["img_id"]

            for i in range(predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                gt_mask = masks_true[i].cpu().numpy()
                evaluator.add_batch(pre_image=mask, gt_image=gt_mask)
                results.append((mask, str(args.output_path / image_ids[i]), args.rgb))

    summary, per_class = build_metrics_report(evaluator, config.classes)
    print_metrics(summary, per_class)
    save_metrics(args.output_path, summary, per_class)

    t0 = time.time()
    if args.workers <= 1:
        for item in results:
            img_writer(item)
    else:
        writer_workers = min(args.workers, mp.cpu_count())
        with mpp.Pool(processes=writer_workers) as pool:
            pool.map(img_writer, results)
    t1 = time.time()
    print("images writing spends: {} s".format(t1 - t0))


if __name__ == "__main__":
    main()
