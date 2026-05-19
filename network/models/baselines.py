import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.segmentation import deeplabv3_resnet50


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ResNetUNet(nn.Module):
    """U-Net decoder with a ResNet-50 encoder."""

    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        encoder = resnet50(weights=weights)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.dec4 = ConvBNReLU(2048 + 1024, 512)
        self.dec3 = ConvBNReLU(512 + 512, 256)
        self.dec2 = ConvBNReLU(256 + 256, 128)
        self.dec1 = ConvBNReLU(128 + 64, 64)
        self.dec0 = ConvBNReLU(64, 32)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    @staticmethod
    def _upsample_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        input_size = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.dec4(torch.cat([self._upsample_to(x4, x3), x3], dim=1))
        d3 = self.dec3(torch.cat([self._upsample_to(d4, x2), x2], dim=1))
        d2 = self.dec2(torch.cat([self._upsample_to(d3, x1), x1], dim=1))
        d1 = self.dec1(torch.cat([self._upsample_to(d2, x0), x0], dim=1))
        d0 = self.dec0(F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False))
        return self.head(d0)


class TorchvisionDeepLabV3(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=True):
        super().__init__()
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
            aux_loss=False,
        )

    def forward(self, x):
        return self.model(x)["out"]


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=(6, 12, 18)):
        super().__init__()
        branches = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        features = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        features.append(pooled)
        return self.project(torch.cat(features, dim=1))


class ResNetDeepLabV3Plus(nn.Module):
    """DeepLabV3+ decoder with an ImageNet-pretrained ResNet-50 encoder."""

    def __init__(self, num_classes, pretrained_backbone=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        encoder = resnet50(weights=weights)

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.aspp = ASPP(2048, 256)
        self.low_project = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer2(low)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.aspp(x)
        low = self.low_project(low)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, low], dim=1))
        return F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)


class HuggingFaceSegFormer(nn.Module):
    def __init__(
        self,
        num_classes,
        model_name="nvidia/mit-b3",
        pretrained=True,
        decoder_hidden_size=768,
        local_files_only=False,
    ):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        if pretrained:
            try:
                self.model = SegformerForSemanticSegmentation.from_pretrained(
                    model_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    local_files_only=local_files_only,
                )
                return
            except Exception as exc:
                print(
                    f"Failed to load pretrained SegFormer weights from {model_name}; "
                    f"falling back to random mit-b3 initialization. Details: {exc}"
                )

        config = SegformerConfig(
            num_labels=num_classes,
            depths=[3, 4, 18, 3],
            hidden_sizes=[64, 128, 320, 512],
            num_attention_heads=[1, 2, 5, 8],
            decoder_hidden_size=decoder_hidden_size,
        )
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
