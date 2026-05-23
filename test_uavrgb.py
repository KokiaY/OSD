import argparse
import csv
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ttach as tta
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tools.cfg import py2cfg
from tools.metric import Evaluator
from train import Supervision_Train


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, required=True, help="Path where to save predictions and reports")
    arg("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation")
    arg("-rgb", "--rgb", action="store_true", help="Save prediction masks using the dataset RGB palette")
    arg("--ckpt_path", type=Path, default=None, help="Checkpoint path, default uses config weights")
    arg("-b", "--batch_size", type=int, default=2, help="Batch size for testing")
    arg("--split", choices=["val", "test"], default="test", help="Dataset split to evaluate")
    arg("--num_workers", type=int, default=None, help="Override dataloader workers")
    arg("--max_visualizations", type=int, default=0, help="Maximum number of panel images to save")
    arg("--save_pred_masks", action="store_true", help="Save raw single-channel prediction masks")
    return parser.parse_args()


def get_device(config):
    if torch.cuda.is_available() and getattr(config, "gpus", None):
        return torch.device(f"cuda:{config.gpus[0]}")
    return torch.device("cpu")


def get_checkpoint_path(config, args):
    if args.ckpt_path is not None:
        return str(args.ckpt_path)
    return os.path.join(config.weights_path, config.test_weights_name + ".ckpt")


def build_tta_model(model, tta_mode):
    if tta_mode == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        return tta.SegmentationTTAWrapper(model, transforms)
    if tta_mode == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation="bicubic", align_corners=False),
            ]
        )
        return tta.SegmentationTTAWrapper(model, transforms)
    return model


def get_test_dataset(config, split):
    if split == "val":
        return config.val_dataset
    return getattr(config, "test_dataset", config.val_dataset)


def get_palette(config):
    palette = getattr(config, "PALETTE", None)
    if palette is None:
        raise ValueError("Config must provide PALETTE for UAVRGB visualization.")
    return np.asarray(palette, dtype=np.uint8)


def label2rgb(mask, palette, ignore_index=255):
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    valid_mask = (mask != ignore_index) & (mask >= 0) & (mask < len(palette))
    if np.any(valid_mask):
        mask_rgb[valid_mask] = palette[mask[valid_mask]]
    return mask_rgb


def tensor_to_uint8_image(img_tensor):
    image = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    if image.shape[-1] == 3:
        image = (image * IMAGENET_STD + IMAGENET_MEAN).clip(0.0, 1.0)
        return (image * 255.0).round().astype(np.uint8)

    if image.shape[-1] == 4:
        image = image[..., [2, 1, 0]]
    elif image.shape[-1] >= 11:
        image = image[..., [3, 2, 1]]
    else:
        image = image[..., :3]

    image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = np.percentile(image, 2, axis=(0, 1), keepdims=True)
    high = np.percentile(image, 98, axis=(0, 1), keepdims=True)
    denom = np.where(high - low < 1e-6, 1.0, high - low)
    image = ((image - low) / denom).clip(0.0, 1.0)
    return (image * 255.0).round().astype(np.uint8)


def blend_mask(image_rgb, mask_rgb, alpha=0.45):
    return cv2.addWeighted(image_rgb, 1.0 - alpha, mask_rgb, alpha, 0.0)


def resize_mask(mask, target_shape, ignore_index=255):
    target_height, target_width = target_shape
    if mask.shape == (target_height, target_width):
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    if ignore_index != 255:
        resized = resized.astype(np.int32)
        resized[resized == 255] = ignore_index
        return resized.astype(np.uint8)
    return resized


def add_title(image, title):
    canvas = np.full((image.shape[0] + 36, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[36:] = image
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return canvas


def build_panel(image_rgb, pred_mask, gt_mask, palette, ignore_index):
    pred_mask = resize_mask(pred_mask, image_rgb.shape[:2], ignore_index=ignore_index)

    if gt_mask is None:
        pred_rgb = label2rgb(pred_mask, palette, ignore_index=ignore_index)
        pred_blend = blend_mask(image_rgb, pred_rgb)
        panel_images = [
            add_title(image_rgb, "Image"),
            add_title(pred_rgb, "Prediction"),
            add_title(pred_blend, "Prediction Overlay"),
        ]
    else:
        gt_mask = resize_mask(gt_mask, image_rgb.shape[:2], ignore_index=ignore_index)
        pred_rgb = label2rgb(pred_mask, palette, ignore_index=ignore_index)
        pred_blend = blend_mask(image_rgb, pred_rgb)
        gt_rgb = label2rgb(gt_mask, palette, ignore_index=ignore_index)
        gt_blend = blend_mask(image_rgb, gt_rgb)
        diff = np.zeros_like(pred_rgb)
        diff[(gt_mask != ignore_index) & (pred_mask == gt_mask)] = np.array([0, 200, 0], dtype=np.uint8)
        diff[(gt_mask != ignore_index) & (pred_mask != gt_mask)] = np.array([220, 0, 0], dtype=np.uint8)
        panel_images = [
            add_title(image_rgb, "Image"),
            add_title(pred_rgb, "Prediction"),
            add_title(gt_rgb, "Ground Truth"),
            add_title(pred_blend, "Prediction Overlay"),
            add_title(gt_blend, "Ground Truth Overlay"),
            add_title(diff, "Match Map"),
        ]
    return np.concatenate(panel_images, axis=1)


def safe_float(value):
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def get_metric_indices(config, class_names):
    if hasattr(config, "metric_include_indices"):
        return list(config.metric_include_indices)
    if hasattr(config, "metric_exclude_indices"):
        excluded = set(config.metric_exclude_indices)
        return [index for index in range(len(class_names)) if index not in excluded]
    return list(range(len(class_names)))


def build_metrics_report(evaluator, class_names, metric_indices=None):
    if metric_indices is None:
        metric_indices = list(range(len(class_names)))
    confusion = evaluator.confusion_matrix.copy()
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)

    precision = evaluator.Precision()
    recall = evaluator.Recall()
    f1 = evaluator.F1()
    iou = evaluator.Intersection_over_Union()
    dice = evaluator.Dice()
    metric_support = support[metric_indices]
    metric_support_sum = metric_support.sum()
    oa = tp[metric_indices].sum() / (metric_support_sum + evaluator.eps)
    if metric_support_sum > 0:
        fwiou = ((metric_support / (metric_support_sum + evaluator.eps)) * iou[metric_indices]).sum()
    else:
        fwiou = np.nan

    per_class = []
    for idx, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_id": idx,
                "class_name": class_name,
                "support_pixels": int(support[idx]),
                "predicted_pixels": int(predicted[idx]),
                "tp_pixels": int(tp[idx]),
                "fp_pixels": int(fp[idx]),
                "fn_pixels": int(fn[idx]),
                "precision": safe_float(precision[idx]),
                "recall": safe_float(recall[idx]),
                "f1": safe_float(f1[idx]),
                "iou": safe_float(iou[idx]),
                "dice": safe_float(dice[idx]),
            }
        )

    summary = {
        "num_classes": len(class_names),
        "metric_class_ids": [int(index) for index in metric_indices],
        "metric_class_names": [str(class_names[index]) for index in metric_indices],
        "valid_pixels": int(confusion.sum()),
        "overall_accuracy": safe_float(oa),
        "mean_precision": safe_float(np.nanmean(precision[metric_indices])),
        "mean_recall": safe_float(np.nanmean(recall[metric_indices])),
        "mean_f1": safe_float(np.nanmean(f1[metric_indices])),
        "mean_iou": safe_float(np.nanmean(iou[metric_indices])),
        "mean_dice": safe_float(np.nanmean(dice[metric_indices])),
        "frequency_weighted_iou": safe_float(fwiou),
    }
    return summary, per_class, confusion.astype(np.int64)


def format_metric(value):
    if value is None:
        return "nan"
    return f"{value:.4f}"


def print_metrics(summary, per_class):
    print("=" * 120)
    print(
        "OA: {oa} | mIoU: {miou} | mF1: {mf1} | mPrecision: {mp} | mRecall: {mr} | mDice: {md} | FWIoU: {fwiou}".format(
            oa=format_metric(summary["overall_accuracy"]),
            miou=format_metric(summary["mean_iou"]),
            mf1=format_metric(summary["mean_f1"]),
            mp=format_metric(summary["mean_precision"]),
            mr=format_metric(summary["mean_recall"]),
            md=format_metric(summary["mean_dice"]),
            fwiou=format_metric(summary["frequency_weighted_iou"]),
        )
    )
    print("Valid Pixels: {}".format(summary["valid_pixels"]))
    print("-" * 120)
    print(
        "{:<4} {:<24} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10}".format(
            "ID", "Class", "Support", "Predicted", "Precision", "Recall", "F1", "IoU"
        )
    )
    print("-" * 120)
    for item in per_class:
        print(
            "{:<4} {:<24} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10}".format(
                item["class_id"],
                item["class_name"][:24],
                item["support_pixels"],
                item["predicted_pixels"],
                format_metric(item["precision"]),
                format_metric(item["recall"]),
                format_metric(item["f1"]),
                format_metric(item["iou"]),
            )
        )
    print("=" * 120)


def save_metrics(output_path, summary, per_class, confusion, class_names):
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)

    with open(metrics_dir / "summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2, ensure_ascii=False)

    with open(metrics_dir / "per_class_metrics.json", "w", encoding="utf-8") as per_class_file:
        json.dump(per_class, per_class_file, indent=2, ensure_ascii=False)

    with open(metrics_dir / "per_class_metrics.csv", "w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "class_id",
            "class_name",
            "support_pixels",
            "predicted_pixels",
            "tp_pixels",
            "fp_pixels",
            "fn_pixels",
            "precision",
            "recall",
            "f1",
            "iou",
            "dice",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_class:
            writer.writerow(row)

    with open(metrics_dir / "confusion_matrix.csv", "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["gt/pred"] + list(class_names))
        for class_name, row in zip(class_names, confusion.tolist()):
            writer.writerow([class_name] + row)

    with open(metrics_dir / "summary.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write(
            "OA: {oa}\n"
            "mIoU: {miou}\n"
            "mF1: {mf1}\n"
            "mPrecision: {mp}\n"
            "mRecall: {mr}\n"
            "mDice: {md}\n"
            "FWIoU: {fwiou}\n"
            "Valid Pixels: {valid_pixels}\n".format(
                oa=format_metric(summary["overall_accuracy"]),
                miou=format_metric(summary["mean_iou"]),
                mf1=format_metric(summary["mean_f1"]),
                mp=format_metric(summary["mean_precision"]),
                mr=format_metric(summary["mean_recall"]),
                md=format_metric(summary["mean_dice"]),
                fwiou=format_metric(summary["frequency_weighted_iou"]),
                valid_pixels=summary["valid_pixels"],
            )
        )
        txt_file.write("\n")
        txt_file.write(
            "{:<4} {:<24} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10}\n".format(
                "ID", "Class", "Support", "Predicted", "Precision", "Recall", "F1", "IoU"
            )
        )
        for item in per_class:
            txt_file.write(
                "{:<4} {:<24} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10}\n".format(
                    item["class_id"],
                    item["class_name"][:24],
                    item["support_pixels"],
                    item["predicted_pixels"],
                    format_metric(item["precision"]),
                    format_metric(item["recall"]),
                    format_metric(item["f1"]),
                    format_metric(item["iou"]),
                )
            )


def save_prediction_outputs(
    output_path,
    image_id,
    image_tensor,
    pred_mask,
    gt_mask,
    palette,
    ignore_index,
    save_pred_mask,
    save_rgb,
    save_panel,
):
    panel_dir = output_path / "panels"

    if save_pred_mask:
        masks_dir = output_path / "pred_masks"
        masks_dir.mkdir(exist_ok=True, parents=True)
    if save_rgb:
        rgb_dir = output_path / "pred_masks_rgb"
        rgb_dir.mkdir(exist_ok=True, parents=True)
    if save_panel:
        panel_dir.mkdir(exist_ok=True, parents=True)

    if save_pred_mask:
        cv2.imwrite(str(masks_dir / f"{image_id}.png"), pred_mask.astype(np.uint8))

    if save_rgb:
        pred_rgb = label2rgb(pred_mask, palette, ignore_index=ignore_index)
        cv2.imwrite(str(rgb_dir / f"{image_id}.png"), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

    if save_panel:
        image_rgb = tensor_to_uint8_image(image_tensor)
        panel = build_panel(image_rgb, pred_mask, gt_mask, palette, ignore_index)
        cv2.imwrite(str(panel_dir / f"{image_id}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


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

    dataset = get_test_dataset(config, args.split)
    palette = get_palette(config)
    ignore_index = getattr(config, "ignore_index", 255)
    class_names = getattr(config, "classes", tuple(str(index) for index in range(config.num_classes)))
    metric_indices = get_metric_indices(config, class_names)
    num_workers = args.num_workers if args.num_workers is not None else getattr(config, "num_workers", 0)
    batch_size = args.batch_size if args.batch_size is not None else getattr(config, "val_batch_size", 1)
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    inference_start = time.time()
    total_images = 0
    saved_panels = 0

    with torch.no_grad():
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            shuffle=False,
        )

        for batch in tqdm(test_loader):
            logits = model(batch["img"].to(device))
            probabilities = nn.Softmax(dim=1)(logits)
            predictions = probabilities.argmax(dim=1)
            gt_batch = batch.get("gt_semantic_seg")
            image_ids = batch["img_id"]

            for index in range(predictions.shape[0]):
                pred_mask = predictions[index].cpu().numpy()
                gt_mask = None if gt_batch is None else gt_batch[index].cpu().numpy()
                image_id = str(image_ids[index])
                if gt_mask is not None:
                    evaluator.add_batch(gt_mask, pred_mask)

                save_panel = args.max_visualizations is None or saved_panels < args.max_visualizations
                save_prediction_outputs(
                    output_path=args.output_path,
                    image_id=image_id,
                    image_tensor=batch["img"][index],
                    pred_mask=pred_mask,
                    gt_mask=gt_mask,
                    palette=palette,
                    ignore_index=ignore_index,
                    save_pred_mask=args.save_pred_masks,
                    save_rgb=args.rgb,
                    save_panel=save_panel,
                )
                if save_panel:
                    saved_panels += 1
                total_images += 1

    inference_time = time.time() - inference_start
    print(
        "Processed {} images in {:.2f} s ({:.4f} s / image)".format(
            total_images, inference_time, inference_time / max(total_images, 1)
        )
    )

    summary, per_class, confusion = build_metrics_report(evaluator, class_names, metric_indices)
    summary["num_images"] = total_images
    summary["saved_panels"] = saved_panels
    summary["split"] = args.split
    summary["tta"] = args.tta
    summary["checkpoint_path"] = checkpoint_path
    summary["seconds_per_image"] = safe_float(inference_time / max(total_images, 1))

    print_metrics(summary, per_class)
    save_metrics(args.output_path, summary, per_class, confusion, class_names)


if __name__ == "__main__":
    main()
